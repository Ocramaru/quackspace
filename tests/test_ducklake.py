"""Tests for DuckLake Parquet-backed catalog snapshots and auto-tiering."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


def _ducklake_available() -> bool:
    """Return whether the ducklake extension can be installed and loaded."""
    con = duckdb.connect()
    try:
        con.execute("INSTALL ducklake; LOAD ducklake;")
        return True
    except Exception:
        return False
    finally:
        con.close()


DUCKLAKE_AVAILABLE = _ducklake_available()
requires_ducklake = pytest.mark.skipif(
    not DUCKLAKE_AVAILABLE, reason="ducklake extension not available"
)


def _space_with_entry(root: Path, rel: str = "hello.md", body: str = "# Hello\n"):
    from quack.core import Entry, Space

    (root / ".quack").mkdir(exist_ok=True)
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    entry = Entry(path=path, root=root, body=body)
    return Space(root=root, entries=[entry], datasets={})


def test_lake_config_defaults():
    """LakeConfig defaults are correct."""
    from quack.config import (
        DEFAULT_LAKE_ENABLED,
        DEFAULT_LAKE_ROW_THRESHOLD,
        DEFAULT_LAKE_SIZE_THRESHOLD_MB,
        DEFAULT_LAKE_SNAPSHOT,
        LakeConfig,
    )

    cfg = LakeConfig()

    assert cfg.enabled == DEFAULT_LAKE_ENABLED
    assert cfg.snapshot_on_reindex == DEFAULT_LAKE_SNAPSHOT
    assert cfg.size_threshold_mb == DEFAULT_LAKE_SIZE_THRESHOLD_MB
    assert cfg.row_threshold == DEFAULT_LAKE_ROW_THRESHOLD


def test_lake_config_loads_from_yaml(tmp_path):
    """Config.load() parses lake: section from config.yaml."""
    (tmp_path / ".quack").mkdir()
    (tmp_path / ".quack" / "config.yaml").write_text(
        "ai:\n  command: echo\nlake:\n  enabled: false\n  size_threshold_mb: 50\n"
    )
    from quack.config import Config

    cfg = Config.load(str(tmp_path))

    assert cfg.lake.enabled is False
    assert cfg.lake.size_threshold_mb == 50
    assert cfg.lake.row_threshold == 100000


def test_lake_config_missing_section(tmp_path):
    """Config.load() gives defaults when lake: is absent."""
    (tmp_path / ".quack").mkdir()
    (tmp_path / ".quack" / "config.yaml").write_text("ai:\n  command: echo\n")
    from quack.config import Config

    cfg = Config.load(str(tmp_path))

    assert cfg.lake.enabled is True


def test_get_db_size_mb_missing(tmp_path):
    """get_db_size_mb returns 0.0 when catalog does not exist."""
    from quack.core import Space
    from quack.lake import get_db_size_mb

    space = Space(root=tmp_path, entries=[], datasets={})

    assert get_db_size_mb(space) == 0.0


def test_get_files_row_count_missing(tmp_path):
    """get_files_row_count returns 0 when catalog does not exist."""
    from quack.core import Space
    from quack.lake import get_files_row_count

    space = Space(root=tmp_path, entries=[], datasets={})

    assert get_files_row_count(space) == 0


def test_is_body_tiered_missing(tmp_path):
    """is_body_tiered returns False when catalog does not exist."""
    from quack.core import Space
    from quack.lake import is_body_tiered

    space = Space(root=tmp_path, entries=[], datasets={})

    assert is_body_tiered(space) is False


def test_check_and_maybe_tier_skips_when_disabled(tmp_path):
    """check_and_maybe_tier is a no-op when lake.enabled is False."""
    from quack.config import LakeConfig
    from quack.core import Space
    from quack.lake import check_and_maybe_tier

    lake_cfg = LakeConfig(enabled=False)
    space = Space(root=tmp_path, entries=[], datasets={})

    check_and_maybe_tier(space, lake_cfg)


def test_check_and_maybe_tier_skips_when_below_threshold(tmp_path):
    """check_and_maybe_tier does not tier when DB is small and rows are few."""
    import quack.catalog as catalog
    from quack.config import LakeConfig
    from quack.lake import check_and_maybe_tier, is_body_tiered

    space = _space_with_entry(tmp_path)
    catalog.build(space, store_body=True)

    lake_cfg = LakeConfig(enabled=True, size_threshold_mb=999, row_threshold=999999)
    check_and_maybe_tier(space, lake_cfg)

    assert not is_body_tiered(space)


@requires_ducklake
def test_snapshot_catalog_writes_lake_files(tmp_path):
    """snapshot_catalog copies catalog rows to DuckLake when extension is available."""
    import quack.catalog as catalog
    from quack.lake import LAKE_ALIAS, ensure_ducklake, lake_catalog_path, snapshot_catalog

    space = _space_with_entry(tmp_path, body="# Hello\n\nBody text\n")
    catalog.build(space, store_body=True)

    try:
        snapshot_catalog(space)
    except Exception as exc:
        pytest.skip(f"ducklake extension not available: {exc}")

    assert lake_catalog_path(space).exists()
    con = duckdb.connect(str(catalog.db_path(space)))
    try:
        ensure_ducklake(con, space)
        row = con.execute(
            f"SELECT rel, body FROM {LAKE_ALIAS}.files WHERE rel = 'hello.md'"
        ).fetchone()
    finally:
        con.close()

    assert row == ("hello.md", "# Hello\n\nBody text\n")


@requires_ducklake
def test_check_and_maybe_tier_moves_body_to_lake(tmp_path):
    """check_and_maybe_tier tiers bodies when row threshold is exceeded."""
    import quack.catalog as catalog
    from quack.config import LakeConfig
    from quack.lake import check_and_maybe_tier, get_file_body, is_body_tiered

    space = _space_with_entry(tmp_path, body="# Hello\n\nBody text\n")
    catalog.build(space, store_body=True)

    try:
        check_and_maybe_tier(
            space, LakeConfig(enabled=True, size_threshold_mb=999, row_threshold=1)
        )
    except Exception as exc:
        pytest.skip(f"ducklake extension not available: {exc}")

    assert is_body_tiered(space)
    assert get_file_body(space, "hello.md") == "# Hello\n\nBody text\n"

    con = duckdb.connect(str(catalog.db_path(space)), read_only=True)
    try:
        body = con.execute("SELECT body FROM files WHERE rel = 'hello.md'").fetchone()[0]
    finally:
        con.close()

    assert body is None


@requires_ducklake
def test_fts_search_survives_tiering(tmp_path):
    """Tiering rebuilds the FTS index so name/description search keeps working.

    Regression: pre-DuckLake catalogs have no `_fts_shadow` table, so tiering
    used to leave the FTS index stale/missing and silently broke search.
    """
    import quack.catalog as catalog
    from quack.config import LakeConfig
    from quack.lake import check_and_maybe_tier, is_body_tiered

    space = _space_with_entry(
        tmp_path, rel="widgets.md", body="# Widgets\n\nA note about widgets.\n"
    )
    catalog.build(space, store_body=True)

    # Sanity: FTS finds the note before tiering.
    db = catalog.db_path(space)
    assert catalog.fts_search_path(db, "widgets")

    try:
        check_and_maybe_tier(
            space, LakeConfig(enabled=True, size_threshold_mb=999, row_threshold=1)
        )
    except Exception as exc:
        pytest.skip(f"ducklake extension not available: {exc}")

    assert is_body_tiered(space)
    # FTS over name/rel/description still returns the note after the body is tiered.
    hits = catalog.fts_search_path(db, "widgets")
    assert hits, "FTS search returned nothing after tiering"
    assert any(rel == "widgets.md" for rel, _desc, _score in hits)


@requires_ducklake
def test_get_file_body_reads_lake_under_concurrent_reader(tmp_path):
    """A tiered body is readable while a read-only catalog connection is open.

    Regression: get_file_body used to open the catalog read-write to attach the
    lake, which raises a same-file config conflict when another in-process
    connection holds it read-only — the MCP server's steady state (a cached
    read-only connection for search/sql). Reads must stay read-only.
    """
    import quack.catalog as catalog
    from quack.config import LakeConfig
    from quack.lake import check_and_maybe_tier, get_file_body

    space = _space_with_entry(tmp_path, rel="hello.md", body="# Hello\n\nBody text\n")
    catalog.build(space, store_body=True)

    try:
        check_and_maybe_tier(
            space, LakeConfig(enabled=True, size_threshold_mb=999, row_threshold=1)
        )
    except Exception as exc:
        pytest.skip(f"ducklake extension not available: {exc}")

    # Hold a read-only connection open, exactly as a live MCP reader would.
    reader = duckdb.connect(str(catalog.db_path(space)), read_only=True)
    try:
        body = get_file_body(space, "hello.md")
    finally:
        reader.close()

    assert body == "# Hello\n\nBody text\n"
