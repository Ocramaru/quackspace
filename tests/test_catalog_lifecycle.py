from __future__ import annotations

import duckdb
import pytest

from quack import catalog
from quack.core import Space
from quack.indexer import reindex


def test_catalog_stores_schema_version(sample_space):
    root = sample_space
    reindex(str(root))

    con = catalog.connect(str(root))
    try:
        version = con.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    finally:
        con.close()

    assert version == str(catalog.SCHEMA_VERSION)


def test_catalog_rejects_missing_schema_metadata(sample_space):
    root = sample_space
    space = Space.load(str(root))
    path = catalog.db_path(space)
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE files (name VARCHAR)")
    finally:
        con.close()

    with pytest.raises(RuntimeError, match="missing schema metadata"):
        catalog.connect(str(root))


def test_catalog_rejects_stale_schema_version(sample_space):
    root = sample_space
    reindex(str(root))
    path = catalog.db_path(Space.load(str(root)))
    con = duckdb.connect(str(path))
    try:
        con.execute("UPDATE metadata SET value = '0' WHERE key = 'schema_version'")
    finally:
        con.close()

    with pytest.raises(RuntimeError, match="schema version 0"):
        catalog.connect(str(root))


def test_catalog_reports_recovery_for_corrupt_db(sample_space):
    root = sample_space
    reindex(str(root))
    path = catalog.db_path(Space.load(str(root)))
    path.write_text("not a duckdb database")

    with pytest.raises(RuntimeError, match="run `quack reindex`"):
        catalog.connect(str(root))
