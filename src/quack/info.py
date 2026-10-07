"""Read-only space statistics for ``quack info``."""

from __future__ import annotations

import shlex

import duckdb

from . import catalog
from .config import Config
from .core import find_root
from .sync import fresh_embedded, pending_work

TOP_TYPES = 5


def _model_from_command(command: str) -> str | None:
    try:
        parts = shlex.split(command) if command else []
        return parts[parts.index("--model") + 1]
    except (ValueError, IndexError):
        return None


def space_info(explicit_root: str | None = None) -> dict | None:
    """Stats for the current space, or ``None`` when no catalog exists.

    Opens the catalog read-only and never indexes, embeds or writes.
    """
    root = find_root(explicit_root)
    db = root / ".quack" / catalog.DB_NAME
    if not db.exists():
        return None
    try:
        con = catalog.connect_path(db)
    except RuntimeError:
        return None
    try:
        files = con.execute("SELECT count(*) FROM files").fetchone()[0]
        folders = con.execute("SELECT count(*) FROM folders").fetchone()[0]
        size = con.execute("SELECT coalesce(sum(size), 0) FROM files").fetchone()[0]
        types = con.execute(
            "SELECT coalesce(nullif(ext, ''), '(none)') AS t, count(*) AS n "
            "FROM files GROUP BY t ORDER BY n DESC, t LIMIT ?",
            [TOP_TYPES],
        ).fetchall()
        row = con.execute(
            "SELECT value FROM metadata WHERE key = 'built_at'"
        ).fetchone()
        last_indexed = row[0] if row else None
        config = Config.load(str(root))
        embed = config.embed
        enabled = embed.configured and not embed.skip
        try:
            embedded = len(fresh_embedded(con, embed.command)) if enabled else 0
        except duckdb.CatalogException:
            embedded = 0
        try:
            run = con.execute("SELECT max(run_at) FROM embedding_runs").fetchone()[0]
            last_embedded = run.isoformat(timespec="seconds") if run else None
        except duckdb.CatalogException:
            last_embedded = None
    finally:
        con.close()

    missing = len(pending_work(str(root)).missing_embeddings) if enabled else 0
    return {
        "root": str(root),
        "files": files,
        "embedded": embedded,
        "missing_embeddings": missing,
        "folders": folders,
        "total_size": int(size),
        "top_types": [{"type": t, "files": n} for t, n in types],
        "last_indexed": last_indexed,
        "last_embedded": last_embedded,
        "embedding": {
            "enabled": enabled,
            "provider": embed.provider or None,
            "model": _model_from_command(embed.command),
        },
    }
