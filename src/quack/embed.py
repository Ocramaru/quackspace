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

import duckdb

from .catalog import DB_NAME, db_path
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


def build_embeddings(explicit_root: str | None = None) -> dict:
    """Embed every file and store vectors + an HNSW index in the catalog."""
    config = Config.load(explicit_root)
    if not config.embed.configured:
        raise EmbedNotConfigured()
    space = Space.load(explicit_root)
    path = db_path(space)
    if not path.exists():
        raise RuntimeError(f"No catalog at {path}. Run `quack reindex` first.")

    con = duckdb.connect(str(path))
    try:
        con.execute("INSTALL vss; LOAD vss;")
        first = _embed_text(
            config.embed, _entry_text(space.entries[0])
        ) if space.entries else []
        dim = config.embed.dim or len(first)
        if not dim:
            raise RuntimeError("Could not determine embedding dimension.")

        con.execute("DROP TABLE IF EXISTS embeddings;")
        con.execute(
            f"CREATE TABLE embeddings (name VARCHAR, rel VARCHAR, vec FLOAT[{dim}]);"
        )
        for i, entry in enumerate(space.entries):
            vec = first if i == 0 else _embed_text(config.embed, _entry_text(entry))
            con.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?)", [entry.name, entry.rel, vec]
            )
        # HNSW index for fast cosine search (vss persists it in-file).
        con.execute("SET hnsw_enable_experimental_persistence = true;")
        con.execute(
            "CREATE INDEX emb_hnsw ON embeddings USING HNSW (vec) "
            "WITH (metric = 'cosine');"
        )
        n = con.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    finally:
        con.close()
    return {"embedded": n, "dim": dim}


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


def _entry_text(entry) -> str:
    """What we embed: name + description + body, the searchable surface."""
    return f"{entry.name}\n{entry.description}\n{entry.body}".strip()
