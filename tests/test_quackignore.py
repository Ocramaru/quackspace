"""Tests for IgnoreRuleset — gitignore-style .quackignore semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from quack.core import IgnoreRuleset, load_ignores
from quack.indexer import reindex
from quack.scaffold import scaffold_root


def ruleset(*lines: str) -> IgnoreRuleset:
    return IgnoreRuleset.build(set(), list(lines))


# ── plain name patterns ────────────────────────────────────────────────────────

def test_plain_name_matches_basename_anywhere():
    rs = ruleset("dist")
    assert rs.is_ignored("dist", "dist")
    assert rs.is_ignored("dist", "src/dist")
    assert not rs.is_ignored("dist2", "src/dist2")


def test_glob_matches_basename_anywhere():
    rs = ruleset("*.lock")
    assert rs.is_ignored("yarn.lock", "yarn.lock")
    assert rs.is_ignored("Cargo.lock", "vendor/Cargo.lock")
    assert not rs.is_ignored("lockfile", "lockfile")


# ── root-anchored patterns (/pattern) ─────────────────────────────────────────

def test_anchored_matches_only_at_root():
    rs = ruleset("/dist")
    assert rs.is_ignored("dist", "dist")
    assert not rs.is_ignored("dist", "src/dist")


def test_anchored_glob():
    rs = ruleset("/build*")
    assert rs.is_ignored("build", "build")
    assert rs.is_ignored("build-output", "build-output")
    assert not rs.is_ignored("build", "src/build")


# ── path patterns (contain /) ─────────────────────────────────────────────────

def test_path_pattern_matches_full_rel():
    rs = ruleset("vendor/generated")
    assert rs.is_ignored("generated", "vendor/generated")
    assert not rs.is_ignored("generated", "src/generated")
    assert not rs.is_ignored("generated", "generated")


def test_path_glob():
    rs = ruleset("docs/*/auto")
    assert rs.is_ignored("auto", "docs/api/auto")
    assert rs.is_ignored("auto", "docs/cli/auto")
    assert not rs.is_ignored("auto", "docs/auto")


# ── negation / exceptions (!pattern) ──────────────────────────────────────────

def test_negation_re_includes_after_wildcard():
    rs = ruleset("*.log", "!important.log")
    assert rs.is_ignored("debug.log", "debug.log")
    assert not rs.is_ignored("important.log", "important.log")


def test_negation_re_includes_specific_path():
    rs = ruleset("build", "!build/keep-me")
    assert rs.is_ignored("build", "build")
    # A file inside build/ that matches the negation should not be ignored
    assert not rs.is_ignored("keep-me", "build/keep-me")


def test_negation_order_matters_last_wins():
    # ignore, un-ignore, re-ignore — last rule (ignore) wins
    rs = ruleset("*.tmp", "!scratch.tmp", "*.tmp")
    assert rs.is_ignored("scratch.tmp", "scratch.tmp")


def test_negation_un_ignore_then_stays():
    rs = ruleset("dist", "!dist")
    assert not rs.is_ignored("dist", "dist")


# ── comments and blank lines ───────────────────────────────────────────────────

def test_comments_and_blanks_ignored():
    rs = ruleset("# this is a comment", "", "  # indented", "dist")
    assert rs.is_ignored("dist", "dist")
    assert not rs.is_ignored("comment", "comment")


# ── trailing slash stripped ────────────────────────────────────────────────────

def test_trailing_slash_stripped():
    rs = ruleset("build/")
    assert rs.is_ignored("build", "build")


# ── builtins cannot be negated ────────────────────────────────────────────────

def test_builtins_override_negation():
    rs = IgnoreRuleset.build({"node_modules"}, ["!node_modules"])
    # builtins check happens before user rules; negation cannot override
    assert rs.is_ignored("node_modules", "node_modules")


# ── load_ignores integration ──────────────────────────────────────────────────

def test_load_ignores_returns_ruleset(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / ".quackignore").write_text("dist\n!dist/keep\n/build\n")
    rs = load_ignores(root)
    assert isinstance(rs, IgnoreRuleset)
    assert rs.is_ignored("dist", "dist")
    assert not rs.is_ignored("keep", "dist/keep")
    assert rs.is_ignored("build", "build")
    assert not rs.is_ignored("build", "src/build")


# ── end-to-end: negation actually un-ignores a file from the walk ─────────────

def test_negation_includes_file_in_index(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    logs = root / "logs"
    logs.mkdir()
    (logs / "debug.log").write_text("noise")
    (logs / "audit.log").write_text("important audit trail")

    # Ignore all .log files but re-include audit.log
    (root / ".quackignore").write_text("*.log\n!audit.log\n")
    result = reindex(str(root))

    # audit.log should be indexed; debug.log should not
    import duckdb
    db = root / ".quack" / "quack.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    rels = {r[0] for r in con.execute("SELECT rel FROM files").fetchall()}
    con.close()

    assert "logs/audit.log" in rels
    assert "logs/debug.log" not in rels


def test_anchored_pattern_excludes_only_root_match(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "build").mkdir()
    (root / "build" / "output.txt").write_text("generated")
    src = root / "src"
    src.mkdir()
    (src / "build").mkdir()
    (src / "build" / "helper.py").write_text("# helper")

    # /build should exclude the root-level build/ but NOT src/build/
    (root / ".quackignore").write_text("/build\n")
    reindex(str(root))

    import duckdb
    db = root / ".quack" / "quack.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    rels = {r[0] for r in con.execute("SELECT rel FROM files").fetchall()}
    con.close()

    assert "build/output.txt" not in rels
    assert "src/build/helper.py" in rels
