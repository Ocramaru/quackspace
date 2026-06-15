"""Semantic search via DuckDB's vss extension.

`quack embed` runs the configured embedding command over each file and stores
the vectors in the catalog; `search(..., semantic=True)` ranks by cosine
similarity. Like everything else this is derived and rebuildable, and entirely
optional: with no embed command configured, semantic search is unavailable and
the structural/FTS tiers still work.

The embedding command (config `embed.command`) must print a JSON array of
floats. {text} is substituted, or the text is piped on stdin.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections import defaultdict
from typing import Callable

import duckdb

from .catalog import DB_NAME, db_path, invalidate
from .config import Config
from .core import Space, find_root
from .subprocess_utils import failure_message


class EmbedNotConfigured(Exception):
    """No embedding command set in config."""


def _embed_text(cfg, text: str) -> list[float]:
    argv = (
        shlex.split(cfg.command)
        if cfg.uses_stdin
        else shlex.split(cfg.command.replace("{text}", text))
    )
    stdin = text if cfg.uses_stdin else None
    try:
        proc = subprocess.run(
            argv, input=stdin, capture_output=True, text=True, timeout=cfg.timeout
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Embedding command not found: '{argv[0]}'. Fix `embed.command` in "
            ".quack/config.yaml."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Embedding command timed out after {cfg.timeout}s running {argv[0]!r}."
        )
    if proc.returncode != 0:
        raise RuntimeError(
            failure_message("Embedding", argv, proc.returncode, proc.stdout, proc.stderr)
        )
    vec = json.loads(proc.stdout)
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("Embedding command did not return a non-empty JSON array.")
    return [float(x) for x in vec]


def build_embeddings(
    explicit_root: str | None = None,
    progress: "Callable[[int, int, str], None] | None" = None,
) -> dict:
    """Embed every file **and** every folder, storing vectors + HNSW indexes in
    the catalog. Files go in ``embeddings``; folders go in a **separate**
    ``folder_embeddings`` table (never reusing the file table, to avoid
    name-keyspace collisions and mixed-entity ranking). A folder is embedded
    from its path, resolved description, and a compact rollup of its children's
    names and descriptions."""
    config = Config.load(explicit_root)
    if not config.embed.configured:
        raise EmbedNotConfigured()
    space = Space.load(explicit_root)
    path = db_path(space)
    if not path.exists():
        raise RuntimeError(f"No catalog at {path}. Run `quack reindex` first.")

    from . import folders as _folders

    folder_infos = _folders.resolve_folders(space)
    by_folder: dict[str, list] = defaultdict(list)
    for e in space.entries:
        by_folder[e.folder].append(e)
    kids_by_parent = _folders.children_index(folder_infos)
    folder_items = [
        (i.rel, i.parent, _folder_text(i, by_folder, kids_by_parent))
        for i in folder_infos.values()
        if not i.is_root
    ]

    invalidate(path)  # free any cached read-only connection before writing
    con = duckdb.connect(str(path))
    try:
        con.execute("INSTALL vss; LOAD vss;")
        con.execute("SET hnsw_enable_experimental_persistence = true;")

        first = _embed_text(
            config.embed, _entry_text(space.entries[0])
        ) if space.entries else []
        dim = config.embed.dim or len(first) or (
            len(_embed_text(config.embed, folder_items[0][2])) if folder_items else 0
        )
        if not dim:
            raise RuntimeError("Could not determine embedding dimension.")

        total = len(space.entries) + len(folder_items)

        con.execute("DROP TABLE IF EXISTS embeddings;")
        con.execute(
            f"CREATE TABLE embeddings (name VARCHAR, rel VARCHAR, vec FLOAT[{dim}]);"
        )
        for i, entry in enumerate(space.entries):
            if progress is not None:
                progress(i, total, f"Embedding {entry.rel}")
            vec = first if i == 0 else _embed_text(config.embed, _entry_text(entry))
            con.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?)", [entry.name, entry.rel, vec]
            )
        if space.entries:
            # HNSW index for fast cosine search (vss persists it in-file).
            con.execute(
                "CREATE INDEX emb_hnsw ON embeddings USING HNSW (vec) "
                "WITH (metric = 'cosine');"
            )
        n = con.execute("SELECT count(*) FROM embeddings").fetchone()[0]

        con.execute("DROP TABLE IF EXISTS folder_embeddings;")
        con.execute(
            "CREATE TABLE folder_embeddings "
            f"(folder VARCHAR, parent VARCHAR, vec FLOAT[{dim}]);"
        )
        for j, (rel, parent, text) in enumerate(folder_items):
            if progress is not None:
                progress(len(space.entries) + j, total, f"Embedding {rel}/")
            vec = _embed_text(config.embed, text)
            con.execute(
                "INSERT INTO folder_embeddings VALUES (?, ?, ?)", [rel, parent, vec]
            )
        if folder_items:
            con.execute(
                "CREATE INDEX folder_emb_hnsw ON folder_embeddings USING HNSW (vec) "
                "WITH (metric = 'cosine');"
            )
        n_folders = con.execute("SELECT count(*) FROM folder_embeddings").fetchone()[0]
    finally:
        con.close()
    return {"embedded": n, "folders": n_folders, "dim": dim}


def semantic_search(
    query: str, explicit_root: str | None = None, limit: int = 10
) -> list[tuple[str, str, float]]:
    """Cosine-similarity search. Returns [(rel, name, distance), ...]."""
    config = Config.load(explicit_root)
    if not config.embed.configured:
        raise EmbedNotConfigured()
    qvec = _embed_text(config.embed, query)
    db = find_root(explicit_root) / ".quack" / DB_NAME
    con = duckdb.connect(str(db), read_only=True)
    try:
        con.execute("LOAD vss;")
        dim = len(qvec)  # cast to the fixed-size array type vss requires
        return con.execute(
            f"""
            SELECT rel, name, array_cosine_distance(vec, ?::FLOAT[{dim}]) AS dist
            FROM embeddings ORDER BY dist LIMIT ?
            """,
            [qvec, limit],
        ).fetchall()
    finally:
        con.close()


def semantic_search_folders(
    query: str, explicit_root: str | None = None, limit: int = 10
) -> list[tuple[str, str, float]]:
    """Cosine-similarity search over the folder vector space. Returns
    [(folder, parent, distance), ...]. Raises if folder embeddings were never
    built (caller degrades gracefully)."""
    config = Config.load(explicit_root)
    if not config.embed.configured:
        raise EmbedNotConfigured()
    qvec = _embed_text(config.embed, query)
    db = find_root(explicit_root) / ".quack" / DB_NAME
    con = duckdb.connect(str(db), read_only=True)
    try:
        con.execute("LOAD vss;")
        dim = len(qvec)
        return con.execute(
            f"""
            SELECT folder, parent,
                   array_cosine_distance(vec, ?::FLOAT[{dim}]) AS dist
            FROM folder_embeddings ORDER BY dist LIMIT ?
            """,
            [qvec, limit],
        ).fetchall()
    finally:
        con.close()


def _entry_text(entry) -> str:
    """What we embed: name + description + body, the searchable surface."""
    return f"{entry.name}\n{entry.description}\n{entry.body}".strip()


def _folder_text(info, by_folder: dict, kids_by_parent: dict) -> str:
    """What we embed for a folder: its path + resolved description + a compact
    rollup of its direct children's names and descriptions."""
    parts: list[str] = [info.rel]
    if info.description:
        parts.append(info.description)
    files = sorted(by_folder.get(info.rel, []), key=lambda e: e.name.lower())
    for e in files[:50]:
        parts.append(f"{e.name}: {e.description}" if e.description else e.name)
    for c in kids_by_parent.get(info.rel, []):
        parts.append(f"{c.name}/: {c.description}" if c.description else f"{c.name}/")
    return "\n".join(parts).strip()
