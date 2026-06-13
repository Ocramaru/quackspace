"""Tests for incremental reindex — only dirty folders are rebuilt."""

from __future__ import annotations

import os
import time
from pathlib import Path

from quack import catalog, folders
from quack.generate import record
from quack.indexer import _dirty_folders, reindex
from quack.scaffold import scaffold_root
from quack.core import Space


def _dirty(space: Space):
    """Resolve folder metadata and run dirty detection, as reindex does."""
    return _dirty_folders(space, folders.resolve_folders(space))


def _touch_future(path: Path, seconds: float = 2.0) -> None:
    """Advance a file's mtime by *seconds* so dirty detection fires reliably."""
    t = time.time() + seconds
    os.utime(path, (t, t))


def _make_space(tmp_path: Path) -> Path:
    root = scaffold_root(str(tmp_path / "space"))
    proj = root / "projects"
    proj.mkdir(exist_ok=True)
    (proj / "a.md").write_text("# A\n\nbody a\n")
    (proj / "b.md").write_text("# B\n\nbody b\n")
    return root


# ---------------------------------------------------------------------------
# _dirty_folders helper
# ---------------------------------------------------------------------------

def test_dirty_folders_returns_none_before_first_reindex(tmp_path):
    root = _make_space(tmp_path)
    space = Space.load(str(root))
    assert _dirty(space) is None


def test_dirty_folders_empty_after_reindex(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    space = Space.load(str(root))
    assert _dirty(space) == set()


def test_dirty_folders_detects_modified_file(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    p = root / "projects" / "a.md"
    p.write_text("# A\n\nupdated body\n")
    _touch_future(p)  # advance mtime past catalog's stored value
    space = Space.load(str(root))
    assert root / "projects" in _dirty(space)


def test_dirty_folders_detects_new_file(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    (root / "projects" / "c.md").write_text("# C\n\nnew file\n")
    space = Space.load(str(root))
    assert root / "projects" in _dirty(space)


def test_dirty_folders_detects_deleted_file(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    (root / "projects" / "b.md").unlink()
    space = Space.load(str(root))
    assert root / "projects" in _dirty(space)


def test_dirty_folders_detects_description_change(tmp_path):
    """record() updates description in .index.yaml — dirty detection must catch it."""
    root = _make_space(tmp_path)
    reindex(str(root))
    record(str(root), "a", "authored description", ["tag1"])
    space = Space.load(str(root))
    assert root / "projects" in _dirty(space)


def test_dirty_folders_detects_tags_change(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    record(str(root), "a", "", ["new-tag"])
    space = Space.load(str(root))
    assert root / "projects" in _dirty(space)


# ---------------------------------------------------------------------------
# reindex fast-path and incremental rebuild
# ---------------------------------------------------------------------------

def test_reindex_noop_returns_zero_folder_indexes(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    result = reindex(str(root))
    assert result["folder_indexes"] == 0


def test_reindex_detects_description_change_and_rebuilds(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    record(str(root), "a", "new description", [])
    result = reindex(str(root))
    assert result["folder_indexes"] >= 1


def test_reindex_detects_file_change_and_updates_catalog(tmp_path):
    root = _make_space(tmp_path)
    reindex(str(root))
    p = root / "projects" / "a.md"
    p.write_text("# A\n\nrevised body\n")
    _touch_future(p)
    reindex(str(root))

    _, rows = catalog.query(
        "SELECT body FROM files WHERE rel = 'projects/a.md'",
        explicit_root=str(root),
    )
    assert rows and "revised" in rows[0][0]


def test_reindex_twice_is_consistent(tmp_path):
    """Two successive full reindexes produce the same catalog content."""
    root = _make_space(tmp_path)
    reindex(str(root))
    reindex(str(root))

    _, rows = catalog.query(
        "SELECT rel FROM files ORDER BY rel",
        explicit_root=str(root),
    )
    rels = [r[0] for r in rows]
    assert "projects/a.md" in rels
    assert "projects/b.md" in rels
