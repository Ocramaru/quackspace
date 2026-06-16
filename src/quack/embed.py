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
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import duckdb

from .catalog import (
    DB_NAME,
    db_path,
    embed_cache_hash,
    file_embed_text,
    folder_embed_text,
    invalidate,
    text_hash,
)
from .config import DEFAULT_AI_TIMEOUT, Config, write_embed_config
from .core import Space, find_root
from .prompts import Choice, choice, is_interactive, text, yes_no
from .subprocess_utils import failure_message

DEFAULT_EMBED_COMMAND = f"{shlex.quote(sys.executable)} -m quack.embed_provider"
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_EMBED_COMMAND = (
    f"{shlex.quote(sys.executable)} -m quack.embed_ollama {OLLAMA_MODEL}"
)


@dataclass(frozen=True)
class EmbedProvider:
    key: str
    label: str
    command: str
    aliases: tuple[str, ...] = ()
    can_pull: bool = False


EMBED_PROVIDERS = {
    "ollama": EmbedProvider(
        key="ollama",
        label=f"Ollama {OLLAMA_MODEL} (recommended, local, better)",
        command=OLLAMA_EMBED_COMMAND,
        aliases=("nomic", OLLAMA_MODEL),
        can_pull=True,
    ),
    "builtin": EmbedProvider(
        key="builtin",
        label="Built-in local (free, no setup)",
        command=DEFAULT_EMBED_COMMAND,
        aliases=("built-in",),
    ),
}
EMBED_PROVIDER_CHOICES = tuple(EMBED_PROVIDERS) + ("custom",)


class EmbedNotConfigured(Exception):
    """No embedding command set in config."""


EMBED_EXPLAINER = (
    "Semantic search uses embeddings to find related files and folders by meaning.\n"
    "Ollama is recommended for local embeddings; built-in is the no-setup fallback.\n"
    "The embedding command receives text and must print one JSON array of floats.\n"
    "Files embed labeled path, type, tags, links, description, and body.\n"
    "Folders embed labeled path, description, rollups, and direct children."
)


@dataclass
class EmbedSetupResult:
    configured: bool
    command: str = ""
    provider: str = ""
    dim: int = 0


def _embed_text(cfg, text: str) -> list[float]:
    argv = shlex.split(cfg.command)
    if not cfg.uses_stdin:
        argv = [part.replace("{text}", text) for part in argv]
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


def run_embed_setup(
    explicit_root: str | None = None,
    *,
    command: str | None = None,
    provider: str | None = None,
    pull: bool = False,
    timeout: int = DEFAULT_AI_TIMEOUT,
) -> EmbedSetupResult:
    """Configure ``embed.command`` after validating it returns a vector."""
    selected_provider = provider or ("custom" if command else None)
    if command is None and provider in EMBED_PROVIDERS:
        command = EMBED_PROVIDERS[provider].command

    if command is None and provider == "custom":
        if not is_interactive():
            raise RuntimeError("Custom embeddings need `--command`.")
        command = _ask_custom_command()

    if command is None:
        print(EMBED_EXPLAINER)
        if not is_interactive():
            command = DEFAULT_EMBED_COMMAND
            selected_provider = "builtin"
            print("\nUsing the built-in free local embedding command.")
        else:
            selected_provider, command, pull = _choose_provider_interactive(
                pull, timeout=timeout
            )
    if selected_provider is None:
        selected_provider = "custom"
    if not command:
        print("No command entered. No changes made.")
        return EmbedSetupResult(configured=False)
    if selected_provider == "ollama" and pull:
        _ensure_ollama_server(timeout=timeout)
        _pull_ollama_model(OLLAMA_MODEL, timeout=timeout)
    elif selected_provider == "ollama":
        _ensure_ollama_server(timeout=timeout)

    cfg = type(
        "_EmbedProbe",
        (),
        {
            "command": command,
            "timeout": timeout,
            "uses_stdin": "{text}" not in command,
        },
    )()
    try:
        vec = _embed_text(cfg, "quack embedding setup test")
    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Embedding setup failed: {e}") from e
    dim = len(vec)
    write_embed_config(
        command,
        explicit_root=explicit_root,
        dim=dim,
        timeout=timeout,
        provider=selected_provider,
    )
    print(f"✓ configured {selected_provider} embeddings (dim {dim})")
    print("  Run `quack embed` to build semantic search vectors.")
    return EmbedSetupResult(
        configured=True, command=command, provider=selected_provider, dim=dim
    )


def _choose_provider_interactive(pull: bool, timeout: int) -> tuple[str, str, bool]:
    print("\nChoose an embedding provider:")
    provider = choice(
        "provider",
        [
            Choice(p.key, p.label, p.aliases)
            for p in EMBED_PROVIDERS.values()
        ] + [Choice("custom", "Custom command")],
        default="ollama",
    )
    if provider in EMBED_PROVIDERS:
        preset = EMBED_PROVIDERS[provider]
        if preset.can_pull:
            if not _ollama_binary_exists():
                print("  Ollama is not installed.")
                if yes_no("Install Ollama now?", default=True):
                    if not _install_ollama(timeout=timeout):
                        print("  Ollama install was not completed.")
                if not _ollama_binary_exists():
                    print("  You can install Ollama later and rerun `quack embed init`.")
                    if yes_no("Use the built-in local embedder instead?", default=True):
                        builtin = EMBED_PROVIDERS["builtin"]
                        return builtin.key, builtin.command, False
                    return provider, preset.command, False
            try:
                _ensure_ollama_server(timeout=timeout)
            except RuntimeError as e:
                print(f"  {e}")
                if yes_no("Use the built-in local embedder instead?", default=True):
                    builtin = EMBED_PROVIDERS["builtin"]
                    return builtin.key, builtin.command, False
                return provider, preset.command, False
            if not _ollama_model_exists(OLLAMA_MODEL):
                if pull:
                    pass
                elif yes_no(f"Pull Ollama model `{OLLAMA_MODEL}` now?", default=True):
                    pull = True
                else:
                    builtin = EMBED_PROVIDERS["builtin"]
                    return builtin.key, builtin.command, False
        return provider, preset.command, pull
    if provider == "custom":
        return "custom", _ask_custom_command(), False
    # Backwards-compatible shortcut: a raw command at the provider prompt.
    return "custom", provider, False


def _ask_custom_command() -> str:
    print("\nEnter the embedding command. Use {text} where the text goes")
    print("(leave it out to pipe the text on stdin):")
    return text("  command")


def _pull_ollama_model(model: str, timeout: int) -> None:
    if _ollama_model_exists(model):
        print(f"Ollama model `{model}` is already installed.")
        return
    print(f"Pulling Ollama model `{model}`...")
    try:
        proc = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=max(timeout, 120),
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Ollama is not installed or not in PATH. Install Ollama, or choose "
            "the built-in local embedder."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"`ollama pull {model}` timed out.") from e
    if proc.returncode != 0:
        raise RuntimeError(
            failure_message(
                "Ollama pull",
                ["ollama", "pull", model],
                proc.returncode,
                proc.stdout,
                proc.stderr,
            )
        )


def _ollama_server_ready() -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/", method="HEAD")
        with urllib.request.urlopen(req, timeout=1):
            return True
    except (OSError, urllib.error.URLError):
        return False


def _ensure_ollama_server(timeout: int) -> None:
    if _ollama_server_ready():
        return
    if not _ollama_binary_exists():
        raise RuntimeError(
            "Ollama is not installed or not in PATH. Install Ollama, or choose "
            "the built-in local embedder."
        )
    print("Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + min(max(timeout, 10), 30)
    while time.monotonic() < deadline:
        if _ollama_server_ready():
            print("  Ollama server is running.")
            return
        time.sleep(0.25)
    raise RuntimeError("Ollama server did not start. Run `ollama serve` and retry.")


def _ollama_binary_exists() -> bool:
    return shutil.which("ollama") is not None


def _install_ollama(timeout: int) -> bool:
    """Install Ollama when there is a reasonably standard local installer."""
    if sys.platform == "darwin" and shutil.which("brew"):
        cmd = ["brew", "install", "ollama"]
    elif sys.platform.startswith("linux") and shutil.which("curl"):
        cmd = ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    else:
        print("  Automatic Ollama install is not available on this system.")
        print("  Install it from https://ollama.com/download")
        return False
    print("Installing Ollama...")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(timeout, 300),
        )
    except subprocess.TimeoutExpired:
        print("  Ollama install timed out.")
        return False
    if proc.returncode != 0:
        print(failure_message("Ollama install", cmd, proc.returncode, proc.stdout, proc.stderr))
        return False
    print("  Ollama installed.")
    return True


def _ollama_model_exists(model: str) -> bool:
    if not _ollama_binary_exists():
        return False
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    wanted = model.split(":", 1)[0]
    for line in proc.stdout.splitlines()[1:]:
        name = line.split(maxsplit=1)[0] if line.split() else ""
        if name == model or name.split(":", 1)[0] == wanted:
            return True
    return False


def build_embeddings(
    explicit_root: str | None = None,
    progress: "Callable[[int, int, str], None] | None" = None,
    *,
    rebuild: bool = False,
    timeout: int | None = None,
) -> dict:
    """Refresh file and folder embeddings in the catalog.

    Embeddings are a derived cache keyed by file ``rel`` or folder path plus a
    hash of the exact source text that produced the vector. By default this
    updates missing/stale rows and prunes deleted rows; ``rebuild=True`` drops
    the cache and starts over."""
    config = Config.load(explicit_root)
    if not config.embed.configured:
        raise EmbedNotConfigured()
    if timeout is not None:
        config.embed.timeout = timeout
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
    file_items = []
    for e in space.entries:
        text = file_embed_text(e)
        source_hash = text_hash(text)
        file_items.append(
            (e.rel, e.name, embed_cache_hash(source_hash, config.embed.command), text)
        )
    folder_items = []
    for i in folder_infos.values():
        if i.is_root:
            continue
        text = folder_embed_text(i, by_folder, kids_by_parent)
        source_hash = text_hash(text)
        folder_items.append(
            (i.rel, i.parent, embed_cache_hash(source_hash, config.embed.command), text)
        )

    invalidate(path)  # free any cached read-only connection before writing
    con = duckdb.connect(str(path))
    try:
        con.execute("INSTALL vss; LOAD vss;")
        con.execute("SET hnsw_enable_experimental_persistence = true;")

        existing_dim = _existing_vector_dim(con)
        dim = config.embed.dim or existing_dim
        first_key = None
        first_vec: list[float] | None = None
        if not dim:
            first_text = file_items[0][3] if file_items else (
                folder_items[0][3] if folder_items else ""
            )
            first_vec = _embed_text(config.embed, first_text) if first_text else []
            first_key = ("file", file_items[0][0]) if file_items else (
                ("folder", folder_items[0][0]) if folder_items else None
            )
            dim = len(first_vec)
        if not dim:
            raise RuntimeError("Could not determine embedding dimension.")

        if rebuild or not _embedding_schema_matches(con, dim):
            con.execute("DROP TABLE IF EXISTS embeddings;")
            con.execute("DROP TABLE IF EXISTS folder_embeddings;")
            con.execute("DROP TABLE IF EXISTS embedding_runs;")

        _ensure_embedding_schema(con, dim)
        total_items = len(file_items) + len(folder_items)
        total = total_items + 3
        old_files = _existing_hashes(con, "embeddings", "rel")
        old_folders = _existing_hashes(con, "folder_embeddings", "folder")
        con.execute("BEGIN TRANSACTION")
        deleted_files = _prune_missing(
            con, "embeddings", "rel", [r for r, _, _, _ in file_items]
        )
        deleted_folders = _prune_missing(
            con, "folder_embeddings", "folder", [r for r, _, _, _ in folder_items]
        )

        updated_files = skipped_files = updated_folders = skipped_folders = 0
        for i, (rel, name, source_hash, text) in enumerate(file_items):
            if progress is not None:
                progress(i, total, f"Embedding {rel}")
            if old_files.get(rel) == source_hash:
                skipped_files += 1
                continue
            vec = (
                first_vec
                if first_key == ("file", rel)
                else _embed_text(config.embed, text)
            )
            con.execute("DELETE FROM embeddings WHERE rel = ?", [rel])
            con.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
                [name, rel, source_hash, vec],
            )
            updated_files += 1

        for j, (rel, parent, source_hash, text) in enumerate(folder_items):
            if progress is not None:
                progress(len(file_items) + j, total, f"Embedding {rel}/")
            if old_folders.get(rel) == source_hash:
                skipped_folders += 1
                continue
            vec = (
                first_vec
                if first_key == ("folder", rel)
                else _embed_text(config.embed, text)
            )
            con.execute("DELETE FROM folder_embeddings WHERE folder = ?", [rel])
            con.execute(
                "INSERT INTO folder_embeddings VALUES (?, ?, ?, ?)",
                [rel, parent, source_hash, vec],
            )
            updated_folders += 1

        if progress is not None:
            progress(total_items, total, "Indexing file vectors")
        _ensure_hnsw_index(
            con,
            "embeddings",
            "emb_hnsw",
            rebuild=bool(updated_files or deleted_files or rebuild),
        )
        if progress is not None:
            progress(total_items + 1, total, "Indexing folder vectors")
        _ensure_hnsw_index(
            con,
            "folder_embeddings",
            "folder_emb_hnsw",
            rebuild=bool(updated_folders or deleted_folders or rebuild),
        )
        n_folders = con.execute("SELECT count(*) FROM folder_embeddings").fetchone()[0]
        n = con.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        if progress is not None:
            progress(total_items + 2, total, "Recording embedding run")
        con.execute(
            "INSERT INTO embedding_runs VALUES (now(), ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                config.embed.command,
                dim,
                updated_files,
                skipped_files,
                deleted_files,
                updated_folders,
                skipped_folders,
                deleted_folders,
            ],
        )
        con.execute("COMMIT")
        if progress is not None:
            progress(total, total, "Embeddings ready")
    finally:
        try:
            con.execute("ROLLBACK")
        except duckdb.Error:
            pass
        con.close()
    return {
        "embedded": n,
        "folders": n_folders,
        "dim": dim,
        "updated": updated_files,
        "skipped": skipped_files,
        "deleted": deleted_files,
        "folders_updated": updated_folders,
        "folders_skipped": skipped_folders,
        "folders_deleted": deleted_folders,
    }


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
            SELECT e.rel, e.name, array_cosine_distance(e.vec, ?::FLOAT[{dim}]) AS dist
            FROM embeddings e
            JOIN files f ON f.rel = e.rel
            WHERE e.source_hash = sha256(? || chr(0) || f.embed_source_hash)
            ORDER BY dist LIMIT ?
            """,
            [qvec, config.embed.command, limit],
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
            SELECT e.folder, e.parent,
                   array_cosine_distance(e.vec, ?::FLOAT[{dim}]) AS dist
            FROM folder_embeddings e
            JOIN folders f ON f.folder = e.folder
            WHERE e.source_hash = sha256(? || chr(0) || f.embed_source_hash)
            ORDER BY dist LIMIT ?
            """,
            [qvec, config.embed.command, limit],
        ).fetchall()
    finally:
        con.close()


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except duckdb.Error:
        return {}
    return {r[1]: str(r[2]) for r in rows}


def _vector_dim(type_name: str) -> int:
    prefix = "FLOAT["
    if type_name.startswith(prefix) and type_name.endswith("]"):
        try:
            return int(type_name[len(prefix):-1])
        except ValueError:
            return 0
    return 0


def _existing_vector_dim(con: duckdb.DuckDBPyConnection) -> int:
    for table in ("embeddings", "folder_embeddings"):
        dim = _vector_dim(_table_columns(con, table).get("vec", ""))
        if dim:
            return dim
    return 0


def _embedding_schema_matches(con: duckdb.DuckDBPyConnection, dim: int) -> bool:
    file_cols = _table_columns(con, "embeddings")
    folder_cols = _table_columns(con, "folder_embeddings")
    if not file_cols and not folder_cols:
        return True
    return (
        file_cols.get("source_hash") == "VARCHAR"
        and folder_cols.get("source_hash") == "VARCHAR"
        and _vector_dim(file_cols.get("vec", "")) == dim
        and _vector_dim(folder_cols.get("vec", "")) == dim
        and _embedding_runs_schema_matches(con)
    )


def _ensure_embedding_schema(con: duckdb.DuckDBPyConnection, dim: int) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS embeddings "
        f"(name VARCHAR, rel VARCHAR, source_hash VARCHAR, vec FLOAT[{dim}]);"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS folder_embeddings "
        f"(folder VARCHAR, parent VARCHAR, source_hash VARCHAR, vec FLOAT[{dim}]);"
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_runs (
            run_at TIMESTAMP,
            command VARCHAR,
            dim INTEGER,
            files_updated INTEGER,
            files_skipped INTEGER,
            files_deleted INTEGER,
            folders_updated INTEGER,
            folders_skipped INTEGER,
            folders_deleted INTEGER
        );
        """
    )


def _embedding_runs_schema_matches(con: duckdb.DuckDBPyConnection) -> bool:
    cols = _table_columns(con, "embedding_runs")
    if not cols:
        return True
    return list(cols) == [
        "run_at",
        "command",
        "dim",
        "files_updated",
        "files_skipped",
        "files_deleted",
        "folders_updated",
        "folders_skipped",
        "folders_deleted",
    ]


def _existing_hashes(
    con: duckdb.DuckDBPyConnection, table: str, key_col: str
) -> dict[str, str]:
    return {
        str(key): str(source_hash)
        for key, source_hash in con.execute(
            f"SELECT {key_col}, source_hash FROM {table}"
        ).fetchall()
    }


def _prune_missing(
    con: duckdb.DuckDBPyConnection, table: str, key_col: str, keys: list[str]
) -> int:
    before = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    con.execute("CREATE OR REPLACE TEMP TABLE _current_embedding_keys(key VARCHAR);")
    if keys:
        con.executemany("INSERT INTO _current_embedding_keys VALUES (?)", [(k,) for k in keys])
    con.execute(
        f"DELETE FROM {table} WHERE {key_col} NOT IN "
        "(SELECT key FROM _current_embedding_keys)"
    )
    after = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return int(before - after)


def _index_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(
        con.execute(
            "SELECT count(*) FROM duckdb_indexes() WHERE index_name = ?", [name]
        ).fetchone()[0]
    )


def _ensure_hnsw_index(
    con: duckdb.DuckDBPyConnection, table: str, name: str, *, rebuild: bool = False
) -> None:
    if rebuild:
        con.execute(f"DROP INDEX IF EXISTS {name};")
    elif _index_exists(con, name):
        return
    if not con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:
        return
    con.execute(
        f"CREATE INDEX {name} ON {table} USING HNSW (vec) "
        "WITH (metric = 'cosine');"
    )
