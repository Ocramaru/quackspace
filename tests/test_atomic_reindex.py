"""
Tests for atomic catalog swap (MAR-164).

The core invariant: a reader holding an open read-only connection to quack.duckdb
must not be disrupted by a concurrent reindex.  After reindex completes the reader
(or its next call via shared_connection) should see the fresh catalog.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import pytest

from quack.catalog import DB_NAME, build, invalidate, shared_connection
from quack.indexer import reindex
from quack.scaffold import scaffold_root


@pytest.fixture
def space(tmp_path: Path) -> Path:
    root = scaffold_root(str(tmp_path / "space"))
    notes = root / "projects"
    notes.mkdir(exist_ok=True)
    (notes / "alpha.md").write_text("# Alpha\n\nSearchable alpha body.\n")
    (notes / "beta.md").write_text("# Beta\n\nSearchable beta body.\n")
    reindex(str(root))
    return root


def test_no_stale_build_tmp_left_on_success(space: Path) -> None:
    """After a successful build no .build-* temp file should remain."""
    quack_dir = space / ".quack"
    reindex(str(space))
    leftover = list(quack_dir.glob(f"{DB_NAME}.build-*"))
    assert leftover == [], f"Stale build temps: {leftover}"


def test_no_stale_build_tmp_left_on_error(space: Path, monkeypatch) -> None:
    """If build raises, the .build-* temp file is cleaned up."""
    import quack.catalog as catalog_mod

    def boom(con):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(catalog_mod, "_build_fts", boom)
    quack_dir = space / ".quack"

    # Remove the catalog so reindex is forced into a full build() call.
    (quack_dir / DB_NAME).unlink(missing_ok=True)
    invalidate(quack_dir / DB_NAME)

    with pytest.raises(RuntimeError, match="injected failure"):
        reindex(str(space))

    leftover = list(quack_dir.glob(f"{DB_NAME}.build-*"))
    assert leftover == [], f"Stale build temps after failure: {leftover}"


def test_reader_survives_concurrent_reindex(space: Path) -> None:
    """A read-only connection opened before reindex still works after it."""
    db_path = space / ".quack" / DB_NAME

    reader = duckdb.connect(str(db_path), read_only=True)
    try:
        before = reader.execute("SELECT count(*) FROM files").fetchone()[0]
        assert before == 2

        # Add a file and reindex while reader is open.
        (space / "projects" / "gamma.md").write_text("# Gamma\n\nNew file.\n")
        reindex(str(space))

        # The reader's fd still points to the old inode — it sees the old count.
        still_before = reader.execute("SELECT count(*) FROM files").fetchone()[0]
        assert still_before == 2, "Reader saw unexpected change mid-hold"
    finally:
        reader.close()

    # After the reader closes and the shared cache reopens, the fresh catalog is visible.
    invalidate(db_path)
    fresh = duckdb.connect(str(db_path), read_only=True)
    try:
        after = fresh.execute("SELECT count(*) FROM files").fetchone()[0]
        assert after == 3
    finally:
        fresh.close()


def test_shared_connection_reopens_after_atomic_swap(space: Path) -> None:
    """shared_connection() detects the inode change and reopens to the new catalog."""
    db_path = space / ".quack" / DB_NAME

    con1 = shared_connection(db_path)
    before = con1.cursor().execute("SELECT count(*) FROM files").fetchone()[0]
    assert before == 2

    (space / "projects" / "delta.md").write_text("# Delta\n\nAnother new file.\n")
    reindex(str(space))  # invalidates cache, atomically swaps file

    # shared_connection() must detect the new inode and reconnect.
    con2 = shared_connection(db_path)
    after = con2.cursor().execute("SELECT count(*) FROM files").fetchone()[0]
    assert after == 3


def test_reindex_from_thread_while_reader_holds(space: Path) -> None:
    """reindex() from a background thread must not raise even with a live reader."""
    db_path = space / ".quack" / DB_NAME
    reader = duckdb.connect(str(db_path), read_only=True)
    error: list[Exception] = []

    def do_reindex():
        try:
            reindex(str(space))
        except Exception as exc:
            error.append(exc)

    t = threading.Thread(target=do_reindex)
    t.start()
    t.join(timeout=30)

    reader.close()
    assert not error, f"reindex raised with live reader open: {error}"
