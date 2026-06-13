"""Tests for quack.gitignore — managed .gitignore block logic."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quack.gitignore import BLOCK_END, BLOCK_START, ensure_gitignore


def _make_git_repo(path: Path) -> Path:
    """Create a minimal fake git repo at *path* and return it."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir()
    return path


def _make_quack_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".quack").mkdir()
    return path


# ---------------------------------------------------------------------------
# Self-ignoring state dir
# ---------------------------------------------------------------------------

def test_self_ignore_created(tmp_path):
    root = _make_quack_root(tmp_path / "space")
    ensure_gitignore(root)
    self_ignore = root / ".quack" / ".gitignore"
    assert self_ignore.exists()
    assert self_ignore.read_text().strip() == "*"


def test_self_ignore_rewritten_if_wrong(tmp_path):
    root = _make_quack_root(tmp_path / "space")
    (root / ".quack" / ".gitignore").write_text("# old content\n")
    ensure_gitignore(root)
    assert (root / ".quack" / ".gitignore").read_text().strip() == "*"


# ---------------------------------------------------------------------------
# No git repo — no-op beyond self-ignore
# ---------------------------------------------------------------------------

def test_no_git_repo_no_gitignore_written(tmp_path):
    root = _make_quack_root(tmp_path / "space")
    ensure_gitignore(root)
    assert not (root / ".gitignore").exists()


# ---------------------------------------------------------------------------
# Fresh .gitignore written when not present
# ---------------------------------------------------------------------------

def test_creates_gitignore_when_absent(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root)
    ensure_gitignore(quack_root)

    gi = git_root / ".gitignore"
    assert gi.exists()
    content = gi.read_text()
    assert BLOCK_START in content
    assert BLOCK_END in content
    assert ".index.yaml" in content
    assert "_diagrams.md" in content
    assert "QUACK.md" in content
    assert ".quack/" in content


# ---------------------------------------------------------------------------
# Idempotency — re-running must not duplicate the block
# ---------------------------------------------------------------------------

def test_idempotent_on_repeated_calls(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root)
    ensure_gitignore(quack_root)
    first = (git_root / ".gitignore").read_text()
    ensure_gitignore(quack_root)
    second = (git_root / ".gitignore").read_text()
    assert first == second
    assert first.count(BLOCK_START) == 1


# ---------------------------------------------------------------------------
# Existing .gitignore — user lines are preserved
# ---------------------------------------------------------------------------

def test_user_lines_preserved(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root)
    gi = git_root / ".gitignore"
    gi.write_text("*.pyc\n__pycache__/\n")
    ensure_gitignore(quack_root)
    content = gi.read_text()
    assert "*.pyc" in content
    assert "__pycache__/" in content
    assert BLOCK_START in content


def test_block_refreshed_in_place(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root)
    gi = git_root / ".gitignore"
    gi.write_text(
        "*.pyc\n"
        f"{BLOCK_START}\n"
        "old-pattern\n"
        f"{BLOCK_END}\n"
        "*.log\n"
    )
    ensure_gitignore(quack_root)
    content = gi.read_text()
    assert "old-pattern" not in content
    assert ".index.yaml" in content
    assert "*.pyc" in content
    assert "*.log" in content
    assert content.count(BLOCK_START) == 1


# ---------------------------------------------------------------------------
# Quack root is a subdirectory of the git root
# ---------------------------------------------------------------------------

def test_subdirectory_quack_root(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root / "notes")
    ensure_gitignore(quack_root)
    content = (git_root / ".gitignore").read_text()
    assert "notes/QUACK.md" in content
    assert "notes/.quack/" in content
    # Tree-wide patterns have no prefix.
    assert ".index.yaml" in content
    assert "_diagrams.md" in content


# ---------------------------------------------------------------------------
# Opt-out via config.yaml
# ---------------------------------------------------------------------------

def test_opt_out_skips_gitignore(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root)
    (quack_root / ".quack" / "config.yaml").write_text(
        yaml.safe_dump({"gitignore": False})
    )
    ensure_gitignore(quack_root)
    assert not (git_root / ".gitignore").exists()


def test_explicit_opt_in_writes_block(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root)
    (quack_root / ".quack" / "config.yaml").write_text(
        yaml.safe_dump({"gitignore": True})
    )
    ensure_gitignore(quack_root)
    assert (git_root / ".gitignore").exists()


# ---------------------------------------------------------------------------
# git status clean after scaffold + reindex
# ---------------------------------------------------------------------------

def test_scaffold_leaves_no_quack_artifacts_in_git_status(tmp_path):
    """After scaffold_root, quack state dir should be self-ignoring."""
    import subprocess

    git_root = tmp_path / "repo"
    git_root.mkdir()
    subprocess.run(["git", "init", str(git_root)], check=True, capture_output=True)

    from quack.scaffold import scaffold_root

    scaffold_root(str(git_root))

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=git_root,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in result.stdout.splitlines() if ".quack/" in ln]
    assert not lines, f"Unexpected quack state in git status: {result.stdout}"
