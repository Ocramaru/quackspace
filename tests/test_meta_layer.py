"""Tests for the full LLM-navigable meta layer (MAR-114):

recognition defaults, Entry precedence, the per-folder ``directories:`` section
and non-sticky readback, the recursive folder walk, the catalog ``folders``
table, and folder embeddings + query routing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from quack import catalog, folders, index_store, recognize
from quack.core import Space
from quack.indexer import reindex
from quack.scaffold import scaffold_root


# ---------------------------------------------------------------------------
# recognize.py — MAR-115
# ---------------------------------------------------------------------------

def test_recognize_exact_beats_glob_beats_extension():
    # Exact name wins over the extension table.
    desc, tags = recognize.recognize_file("repo/pyproject.toml")
    assert "Python project" in desc
    assert "packaging" in tags
    # Extension fallback for an unremarkable source file.
    desc, tags = recognize.recognize_file("src/app/main.py")
    assert desc == "Python source file."
    assert tags == ["python", "source"]
    # Glob match for a lockfile (no exact entry, not an extension we list).
    desc, tags = recognize.recognize_file("uv.lock")
    assert "lockfile" in tags


def test_recognize_unknown_file_returns_none():
    assert recognize.recognize_file("weird/thing.xyzzy") is None


def test_recognize_folder_known_and_unknown():
    desc, tags = recognize.recognize_folder("tests")
    assert "test" in desc.lower()
    assert recognize.recognize_folder("totally-custom") is None


def test_recognize_tags_are_fresh_copies():
    _, tags1 = recognize.recognize_file("a.py")
    tags1.append("mutated")
    _, tags2 = recognize.recognize_file("b.py")
    assert "mutated" not in tags2


# ---------------------------------------------------------------------------
# Entry precedence — MAR-116
# ---------------------------------------------------------------------------

def _space_with(tmp_path: Path) -> Path:
    root = scaffold_root(str(tmp_path / "space"))
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "mod.py").write_text("x = 1\n")
    (root / "src" / ".gitignore").write_text("*.pyc\n")
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "guide.md").write_text(
        "---\ndescription: A hand-written guide\ntags: [howto]\n---\n# Guide\n"
    )
    return root


def test_recognition_is_lowest_precedence(tmp_path):
    root = _space_with(tmp_path)
    space = Space.load(str(root))
    by_rel = {e.rel: e for e in space.entries}

    # Recognition default (no authored, no frontmatter).
    mod = by_rel["src/mod.py"]
    assert mod.description == "Python source file."
    assert mod.tags == ["python", "source"]
    assert mod.described_at == ""  # derived ⇒ blank
    assert mod.stale is False

    # Frontmatter beats recognition.
    guide = by_rel["docs/guide.md"]
    assert guide.description == "A hand-written guide"
    assert guide.tags == ["howto"]
    assert guide.described_at != ""


def test_frontmatter_description_suppresses_recognition_tags(tmp_path):
    """A .md with a frontmatter description but no tags must NOT get generic
    recognition tags; a plain .md with no frontmatter still does (C2)."""
    root = scaffold_root(str(tmp_path / "space"))
    (root / "notes").mkdir()
    (root / "notes" / "described.md").write_text(
        "---\ndescription: Hand-written note\n---\n# Note\n"
    )
    (root / "notes" / "plain.md").write_text("# Plain\n\njust text\n")
    space = Space.load(str(root))
    by_rel = {e.rel: e for e in space.entries}

    described = by_rel["notes/described.md"]
    assert described.description == "Hand-written note"
    assert described.tags == []  # recognition tags suppressed

    plain = by_rel["notes/plain.md"]
    assert plain.tags == ["docs", "markdown"]  # recognition default applies


def test_authored_beats_recognition_and_promotes(tmp_path):
    root = _space_with(tmp_path)
    reindex(str(root))
    from quack.generate import record

    record(str(root), "src/mod.py", "The core module", ["core"])
    space = Space.load(str(root))
    mod = next(e for e in space.entries if e.rel == "src/mod.py")
    assert mod.description == "The core module"
    assert mod.tags == ["core"]
    assert mod.described_at != ""  # promoted to authored


# ---------------------------------------------------------------------------
# index_store directories + non-sticky readback — MAR-117
# ---------------------------------------------------------------------------

def test_index_store_directories_roundtrip(tmp_path):
    root = _space_with(tmp_path)
    reindex(str(root))
    # The root index lists its subfolders under directories:.
    dirs = index_store.load_authored_dirs(root)
    assert "src" in dirs and "docs" in dirs


def test_non_sticky_recognition_default_not_authored(tmp_path):
    root = _space_with(tmp_path)
    reindex(str(root))
    # src/.gitignore got a recognition description written with blank
    # described_at; it must read back as NOT authored.
    authored = index_store.load_authored(root / "src")
    assert authored[".gitignore"]["description"] == ""
    assert authored[".gitignore"]["tags"] == []


def test_recognition_default_re_derives_and_second_reindex_is_noop(tmp_path):
    """Across two reindexes a recognition default must re-derive (not get
    promoted to authored), and an unchanged second reindex is a no-op."""
    root = _space_with(tmp_path)
    reindex(str(root))
    result = reindex(str(root))
    assert result["folder_indexes"] == 0  # idempotent no-op

    space = Space.load(str(root))
    mod = next(e for e in space.entries if e.rel == "src/mod.py")
    assert mod.description == "Python source file."  # still the recognition default
    assert mod.described_at == ""  # never promoted to authored


def test_set_meta_directory_section_is_preserved(tmp_path):
    root = _space_with(tmp_path)
    reindex(str(root))
    index_store.set_meta(
        root, "src", "Authored source desc", ["src-tag"],
        datetime.now().isoformat(timespec="seconds"), section="directories",
    )
    reindex(str(root))
    dirs = index_store.load_authored_dirs(root)
    assert dirs["src"]["description"] == "Authored source desc"
    assert dirs["src"]["tags"] == ["src-tag"]


def test_folder_tags_only_authoring_is_preserved(tmp_path):
    """Authoring folder tags without a description must survive reindex (B2),
    mirroring how file tags-only authoring works."""
    root = _space_with(tmp_path)
    reindex(str(root))
    index_store.set_meta(
        root, "docs", "", ["folder-tag"],
        datetime.now().isoformat(timespec="seconds"), section="directories",
    )
    reindex(str(root))
    dirs = index_store.load_authored_dirs(root)
    assert dirs["docs"]["tags"] == ["folder-tag"]


def test_describe_authors_a_folder(tmp_path):
    """generate.record (CLI/MCP describe) can author a folder description (B3)."""
    from quack.generate import record

    root = _space_with(tmp_path)
    reindex(str(root))
    rel = record(str(root), "src", "Source root", ["srcdir"])
    assert rel == "src"
    reindex(str(root))
    _, rows = catalog.query(
        "SELECT description FROM folders WHERE folder = 'src'", explicit_root=str(root)
    )
    assert rows[0][0] == "Source root"


def test_emptied_folder_index_is_cleaned(tmp_path):
    """When a folder loses all its files, its .index.yaml must stop listing the
    deleted file (B1)."""
    root = scaffold_root(str(tmp_path / "space"))
    (root / "keep").mkdir()
    (root / "keep" / "a.py").write_text("x = 1\n")
    reindex(str(root))
    assert (root / "keep" / ".index.yaml").exists()

    (root / "keep" / "a.py").unlink()
    reindex(str(root))
    # The folder is empty now; the stale index must not still list a.py.
    idx = root / "keep" / ".index.yaml"
    assert (not idx.exists()) or ("a.py" not in idx.read_text())
    _, rows = catalog.query(
        "SELECT count(*) FROM files WHERE rel = 'keep/a.py'", explicit_root=str(root)
    )
    assert rows[0][0] == 0


# ---------------------------------------------------------------------------
# Recursive walk — MAR-118
# ---------------------------------------------------------------------------

def test_index_yaml_in_every_folder_including_subdir_only(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "leaf.md").write_text("# leaf\n")
    reindex(str(root))
    # "a" has only a subfolder, no direct files, but still gets an index that
    # lists its directories.
    assert (root / "a" / ".index.yaml").exists()
    assert (root / "a" / "b" / ".index.yaml").exists()
    dirs = index_store.load_authored_dirs(root / "a")
    assert "b" in dirs


def test_map_lists_all_nested_folders(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "leaf.md").write_text("# leaf\n")
    reindex(str(root))
    data = yaml.safe_load((root / ".quack" / "map.yaml").read_text())
    assert "a" in data["folders"]
    assert "a/b" in data["folders"]
    assert data["folders"]["a/b"]["files"] == 1


# ---------------------------------------------------------------------------
# Opaque dirs: mentioned but not indexed — MAR-135
# ---------------------------------------------------------------------------

def test_opaque_dir_is_mentioned_but_not_indexed(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    sp = root / "libs" / "site-packages"
    sp.mkdir(parents=True)
    (sp / "huge_dep.py").write_text("x = 1\n")
    (sp / "nested").mkdir()
    (sp / "nested" / "more.py").write_text("y = 2\n")
    (root / "libs" / "mine.py").write_text("z = 3\n")
    reindex(str(root))

    # The opaque folder is acknowledged, with a recognition description...
    _, frows = catalog.query(
        "SELECT folder, description FROM folders WHERE folder = 'libs/site-packages'",
        explicit_root=str(root),
    )
    assert frows and frows[0][0] == "libs/site-packages"
    assert frows[0][1]  # has a recognition description

    # ...but nothing under it is indexed, and it is not descended into.
    _, rows = catalog.query(
        "SELECT count(*) FROM files WHERE rel LIKE 'libs/site-packages/%'",
        explicit_root=str(root),
    )
    assert rows[0][0] == 0
    _, deep = catalog.query(
        "SELECT count(*) FROM folders WHERE folder LIKE 'libs/site-packages/%'",
        explicit_root=str(root),
    )
    assert deep[0][0] == 0  # the 'nested' subfolder was never walked

    # A sibling real file in the same parent is still indexed.
    _, mine = catalog.query(
        "SELECT count(*) FROM files WHERE rel = 'libs/mine.py'", explicit_root=str(root)
    )
    assert mine[0][0] == 1


def test_cache_dirs_are_hidden_entirely(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    cache = root / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "m.cpython-313.pyc").write_text("bytecode")
    (root / "pkg" / "m.py").write_text("x = 1\n")
    reindex(str(root))

    # __pycache__ is not even mentioned as a folder.
    _, rows = catalog.query(
        "SELECT count(*) FROM folders WHERE folder LIKE 'pkg/__pycache__%'",
        explicit_root=str(root),
    )
    assert rows[0][0] == 0


# ---------------------------------------------------------------------------
# catalog folders table + schema version — MAR-119
# ---------------------------------------------------------------------------

def test_schema_version_is_2():
    assert catalog.SCHEMA_VERSION == 2


def test_folders_table_parent_mapping(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "m.py").write_text("x=1\n")
    reindex(str(root))

    _, rows = catalog.query(
        "SELECT folder, parent, n_files FROM folders ORDER BY folder",
        explicit_root=str(root),
    )
    by_folder = {f: (p, n) for f, p, n in rows}
    assert by_folder["src"][0] == ""        # top-level → parent is root ("")
    assert by_folder["src/pkg"][0] == "src"  # nested → parent is "src"
    assert by_folder["src/pkg"][1] == 1      # direct file count

    # directories: of X == WHERE parent = X
    _, kids = catalog.query(
        "SELECT folder FROM folders WHERE parent = 'src'", explicit_root=str(root)
    )
    assert [k[0] for k in kids] == ["src/pkg"]


# ---------------------------------------------------------------------------
# Incremental catalog tiers (skipped / light / full) — MAR-132
# ---------------------------------------------------------------------------

def _bump_mtime(path: Path, seconds: float = 2.0) -> None:
    t = time.time() + seconds
    os.utime(path, (t, t))


def test_reindex_skips_when_nothing_changed(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "src").mkdir()
    (root / "src" / "m.py").write_text("x = 1\n")
    assert reindex(str(root))["catalog"] == "full"
    assert reindex(str(root))["catalog"] == "skipped"


def test_reindex_light_path_for_tag_only_edit(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "src").mkdir()
    (root / "src" / "m.py").write_text("x = 1\n")
    reindex(str(root))
    # Tag-only authoring: description stays the recognition default, so the
    # full-text surface is unchanged → light path, FTS not rebuilt.
    index_store.set_meta(
        root / "src", "m.py", "", ["hot"],
        datetime.now().isoformat(timespec="seconds"),
    )
    res = reindex(str(root))
    assert res["catalog"] == "light"
    _, rows = catalog.query("SELECT tag FROM tags WHERE name = 'm'", explicit_root=str(root))
    assert ("hot",) in rows
    # FTS index survived the light update and still matches the file.
    _, frows = catalog.query(
        "SELECT count(*) FROM files WHERE rel = 'src/m.py'", explicit_root=str(root)
    )
    assert frows[0][0] == 1


def test_reindex_light_path_for_folder_authoring(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "src").mkdir()
    (root / "src" / "m.py").write_text("x = 1\n")
    reindex(str(root))
    from quack.generate import record

    record(str(root), "src", "Source root", ["sd"])
    res = reindex(str(root))
    assert res["catalog"] == "light"
    _, rows = catalog.query(
        "SELECT description FROM folders WHERE folder = 'src'", explicit_root=str(root)
    )
    assert rows[0][0] == "Source root"


def test_reindex_full_path_for_body_change(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "notes").mkdir()
    p = root / "notes" / "a.md"
    p.write_text("# A\n\nhello\n")
    reindex(str(root))
    p.write_text("# A\n\nhello brand new searchable words\n")
    _bump_mtime(p)
    res = reindex(str(root))
    assert res["catalog"] == "full"
    # The new body text is searchable → FTS was rebuilt.
    from quack.search import search

    hits = search("searchable", explicit_root=str(root), expand=False)
    assert any(h.entry.rel == "notes/a.md" for h in hits)


# ---------------------------------------------------------------------------
# Folder embeddings + routing — MAR-121
# ---------------------------------------------------------------------------

def test_route_heuristic():
    from quack.search import route

    assert route("which folder handles auth") == "folders"
    assert route("find the file that parses tokens") == "files"
    assert route("payments") == "both"
    # Word-boundary matching: "areas" must not trip the "area"-style hints (C3).
    assert route("research areas overview") == "both"


def test_structural_folder_fallback_without_embeddings(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "auth").mkdir()
    (root / "auth" / "x.py").write_text("login\n")
    reindex(str(root))
    from quack.search import search_folders

    hits = search_folders("auth", explicit_root=str(root), limit=5)
    assert any(h.folder == "auth" and h.via == "structural" for h in hits)


def _fake_embedder(tmp_path: Path) -> Path:
    script = tmp_path / "embedder.py"
    script.write_text(
        "import sys, hashlib, json\n"
        "t = sys.stdin.read()\n"
        "h = hashlib.sha256(t.encode()).digest()\n"
        "print(json.dumps([b / 255.0 for b in h[:8]]))\n"
    )
    return script


def test_folder_embeddings_built_and_searched(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "auth").mkdir()
    (root / "auth" / "login.py").write_text("# login\n")
    (root / "billing").mkdir()
    (root / "billing" / "invoice.py").write_text("# invoice\n")
    reindex(str(root))

    cfg = root / ".quack" / "config.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["embed"] = {
        "command": f"{sys.executable} {_fake_embedder(tmp_path)}",
        "timeout": 10,
    }
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))

    from quack.embed import build_embeddings, semantic_search_folders

    summary = build_embeddings(str(root))
    assert summary["folders"] >= 2  # auth, billing (at least)

    # folder_embeddings is a SEPARATE table from the file embeddings.
    _, rows = catalog.query(
        "SELECT count(*) FROM folder_embeddings", explicit_root=str(root)
    )
    assert rows[0][0] == summary["folders"]

    results = semantic_search_folders("login", explicit_root=str(root), limit=3)
    assert results  # returns (folder, parent, distance) tuples
    assert all(len(r) == 3 for r in results)
