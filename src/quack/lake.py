"""DuckLake integration: Parquet-backed catalog snapshots and auto-tiering.

When lake.enabled is true in config.yaml, every quack reindex snapshots the
files and folders catalog tables to DuckLake Parquet files stored under
.quack/lake_data/. DuckLake's built-in versioning keeps a complete history of
every reindex so you can time-travel: query what files existed at any past
snapshot.

When the quack.duckdb file grows beyond lake.size_threshold_mb (or the files
table exceeds lake.row_threshold rows), body text is tiered out of DuckDB into
DuckLake, keeping the hot catalog small. FTS is rebuilt on name+description
only after tiering (body is no longer in FTS but is still retrievable from the
lake via get_file_body).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from . import catalog as _catalog

if TYPE_CHECKING:
    from .config import LakeConfig
    from .core import Space

LAKE_CATALOG_NAME = "catalog.ducklake"
LAKE_DATA_DIRNAME = "lake_data"
LAKE_ALIAS = "duck_lake"


def lake_catalog_path(space: "Space") -> Path:
    return space.root.resolve() / ".quack" / LAKE_CATALOG_NAME


def lake_data_path(space: "Space") -> Path:
    return space.root.resolve() / ".quack" / LAKE_DATA_DIRNAME


def ensure_ducklake(
    con: duckdb.DuckDBPyConnection, space: "Space", *, read_only: bool = False
) -> None:
    """Install+load the ducklake extension and ATTACH the lake (idempotent).

    With *read_only* the lake is attached read-only and no tables are created —
    use this for reads on a read-only catalog connection (the lake's tables
    already exist once anything has been tiered). A read-write attach (the
    default) cannot ride on a read-only main connection, and table creation
    would fail there anyway.
    """
    con.execute("INSTALL ducklake; LOAD ducklake;")

    attached = {
        row[0]
        for row in con.execute(
            "SELECT database_name FROM duckdb_databases() WHERE internal = false"
        ).fetchall()
    }
    if LAKE_ALIAS in attached:
        return

    catalog = str(lake_catalog_path(space))
    data_dir = lake_data_path(space)

    if read_only:
        con.execute(
            f"ATTACH 'ducklake:{catalog}' AS {LAKE_ALIAS} "
            f"(DATA_PATH '{data_dir}/', READ_ONLY)"
        )
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"ATTACH 'ducklake:{catalog}' AS {LAKE_ALIAS} "
        f"(DATA_PATH '{data_dir}/', CREATE_IF_NOT_EXISTS true, OVERRIDE_DATA_PATH true)"
    )
    _ensure_lake_tables(con)


def _ensure_lake_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create DuckLake tables if they don't already exist."""
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {LAKE_ALIAS}.files (
            name              VARCHAR,
            rel               VARCHAR,
            folder            VARCHAR,
            ext               VARCHAR,
            title             VARCHAR,
            description       VARCHAR,
            tags_csv          VARCHAR,
            n_links           INTEGER,
            n_inbound         INTEGER,
            is_orphan         BOOLEAN,
            is_binary         BOOLEAN,
            file_modified     VARCHAR,
            described_at      VARCHAR,
            stale             BOOLEAN,
            body              VARCHAR,
            embed_source_hash VARCHAR,
            snapshot_at       TIMESTAMP
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {LAKE_ALIAS}.folders (
            folder            VARCHAR,
            parent            VARCHAR,
            description       VARCHAR,
            n_files           INTEGER,
            diagram           VARCHAR,
            described_at      VARCHAR,
            embed_source_hash VARCHAR,
            snapshot_at       TIMESTAMP
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {LAKE_ALIAS}.file_bodies (
            rel         VARCHAR,
            body        VARCHAR,
            tiered_at   TIMESTAMP
        )
    """)


def snapshot_catalog(space: "Space") -> None:
    """Copy current files+folders to DuckLake after a reindex.

    DuckLake's built-in versioning means old snapshots remain queryable via
    time-travel even after new rows replace them.
    """
    db_path = _catalog.db_path(space)
    if not db_path.exists():
        return

    try:
        con = duckdb.connect(str(db_path))
    except duckdb.Error:
        # A live in-process reader holds the catalog read-only (e.g. an MCP
        # client mid-reindex). The snapshot is best-effort and reader-safety
        # wins — skip it; the next reindex without contention will catch up.
        return
    try:
        try:
            ensure_ducklake(con, space)
        except duckdb.IOException as exc:
            if "ducklake" in str(exc).lower() and "download" in str(exc).lower():
                return
            raise
        con.execute(f"DELETE FROM {LAKE_ALIAS}.files")
        con.execute(
            f"INSERT INTO {LAKE_ALIAS}.files "
            "SELECT name, rel, folder, ext, title, description, tags_csv, "
            "n_links, n_inbound, is_orphan, is_binary, file_modified, "
            "described_at, stale, body, embed_source_hash, now() "
            "FROM files"
        )
        con.execute(f"DELETE FROM {LAKE_ALIAS}.folders")
        con.execute(
            f"INSERT INTO {LAKE_ALIAS}.folders "
            "SELECT folder, parent, description, n_files, diagram, "
            "described_at, embed_source_hash, now() "
            "FROM folders"
        )
    finally:
        con.close()


def get_db_size_mb(space: "Space") -> float:
    """Return the current size of quack.duckdb in megabytes."""
    db_path = _catalog.db_path(space)
    if not db_path.exists():
        return 0.0
    return db_path.stat().st_size / (1024 * 1024)


def get_files_row_count(space: "Space") -> int:
    """Return the current row count of the files table."""
    db_path = _catalog.db_path(space)
    if not db_path.exists():
        return 0
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT count(*) FROM files").fetchone()[0]
    finally:
        con.close()


def is_body_tiered(space: "Space") -> bool:
    """Return True if body has already been tiered to DuckLake for this space."""
    db_path = _catalog.db_path(space)
    if not db_path.exists():
        return False
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            row = con.execute(
                "SELECT value FROM metadata WHERE key = 'body_tiered'"
            ).fetchone()
            return row is not None and row[0] == "true"
        finally:
            con.close()
    except Exception:
        return False


def tier_body_to_lake(space: "Space") -> None:
    """Move body text from DuckDB files table to DuckLake to free space.

    After tiering:
    - files.body is NULL in DuckDB (space reclaimed after CHECKPOINT/VACUUM)
    - duck_lake.file_bodies holds the body text
    - metadata key 'body_tiered' is set to 'true'
    """
    db_path = _catalog.db_path(space)
    if not db_path.exists():
        return

    con = duckdb.connect(str(db_path))
    try:
        ensure_ducklake(con, space)
        con.execute(f"DELETE FROM {LAKE_ALIAS}.file_bodies")
        con.execute(
            f"INSERT INTO {LAKE_ALIAS}.file_bodies "
            "SELECT rel, body, now() FROM files WHERE body IS NOT NULL"
        )
        con.execute("UPDATE files SET body = NULL")
        con.execute("CHECKPOINT")
        con.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('body_tiered', 'true')"
        )
        # Rebuild FTS over name+description so search keeps working after the
        # body is tiered out. _populate_fts_shadow also creates the shadow table
        # if this catalog predates it.
        _catalog._populate_fts_shadow(con)
        _catalog._build_fts(con)
    finally:
        con.close()


def get_file_body(space: "Space", rel: str) -> str | None:
    """Retrieve body text for a file, checking DuckLake if body has been tiered."""
    db_path = _catalog.db_path(space)
    if not db_path.exists():
        return None

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            "SELECT body FROM files WHERE rel = ?", [rel]
        ).fetchone()
        if row is not None and row[0] is not None:
            return row[0]
    finally:
        con.close()

    # Body may be in DuckLake. Open read-only so this coexists with a live
    # in-process read-only catalog connection (the MCP server's normal state);
    # the lake itself is a separate file, so reading it needs no write to the
    # catalog. A read-write open here would raise a same-file config conflict.
    if not is_body_tiered(space):
        return None

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ensure_ducklake(con, space, read_only=True)
        row = con.execute(
            f"SELECT body FROM {LAKE_ALIAS}.file_bodies WHERE rel = ?", [rel]
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def check_and_maybe_tier(space: "Space", lake_cfg: "LakeConfig") -> None:
    """Auto-tier body to DuckLake when size or row thresholds are exceeded."""
    if not lake_cfg.enabled:
        return
    if is_body_tiered(space):
        return

    # Short-circuits: the row count is only queried when the size check fails.
    if (
        get_db_size_mb(space) >= lake_cfg.size_threshold_mb
        or get_files_row_count(space) >= lake_cfg.row_threshold
    ):
        tier_body_to_lake(space)
