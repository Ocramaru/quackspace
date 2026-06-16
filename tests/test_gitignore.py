"""Tests for quack.gitignore — managed .gitignore block logic."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quack.gitignore import (
    BLOCK_HEADER,
    ensure_gitignore,
    _find_descendant_git_roots,
)


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
    summary = ensure_gitignore(root)
    self_ignore = root / ".quack" / ".gitignore"
    assert self_ignore.exists()
    assert self_ignore.read_text().strip() == "*"
    assert self_ignore in summary.updated
    assert summary.updated_count >= 1


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
    summary = ensure_gitignore(quack_root)

    gi = git_root / ".gitignore"
    assert gi.exists()
    content = gi.read_text()
    assert BLOCK_HEADER in content
    assert ".index.yaml" in content
    assert "_diagrams.md" in content
    assert "QUACK.md" in content
    assert ".quack/" in content
    assert gi in summary.updated
    assert git_root in summary.protected
    assert "protected 1 git repo" in summary.format(quack_root)


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
    assert first.count(BLOCK_HEADER) == 1


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
    assert BLOCK_HEADER in content


def test_block_refreshed_in_place(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    quack_root = _make_quack_root(git_root)
    gi = git_root / ".gitignore"
    # Existing block gets replaced in-place on next write.
    gi.write_text(
        "*.pyc\n"
        f"\n{BLOCK_HEADER}\n"
        "old-pattern\n"
        "\n"
        "*.log\n"
    )
    ensure_gitignore(quack_root)
    content = gi.read_text()
    assert "old-pattern" not in content
    assert ".index.yaml" in content
    assert "*.pyc" in content
    assert "*.log" in content
    assert content.count(BLOCK_HEADER) == 1


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


def test_scaffold_reports_progress_and_gitignore_summary(tmp_path):
    git_root = _make_git_repo(tmp_path / "repo")
    calls: list[tuple[int, int, str]] = []
    summaries = []

    from quack.scaffold import scaffold_root

    root = scaffold_root(
        str(git_root),
        progress=lambda done, total, message: calls.append((done, total, message)),
        gitignore_summary=summaries,
    )

    assert root == git_root.resolve()
    assert calls[0] == (0, 5, "Preparing space")
    assert (3, 5, "Managing gitignore") in calls
    assert calls[-1] == (5, 5, "Scaffolded space")
    assert len(summaries) == 1
    assert root in summaries[0].protected
    assert "protected" in summaries[0].format(root)


# ---------------------------------------------------------------------------
# Nested git repos — managed block applied to each (GH#11)
# ---------------------------------------------------------------------------

def test_find_descendant_git_roots(tmp_path):
    quack_root = _make_quack_root(tmp_path / "space")
    alpha = quack_root / "projects" / "alpha"
    beta = quack_root / "projects" / "beta"
    _make_git_repo(alpha)
    _make_git_repo(beta)

    found = _find_descendant_git_roots(quack_root)
    assert set(found) == {alpha, beta}


def test_find_descendant_git_roots_skips_configured_ignored_dirs(tmp_path):
    quack_root = _make_quack_root(tmp_path / "space")
    visible = _make_git_repo(quack_root / "projects" / "visible")
    hidden = _make_git_repo(quack_root / "node_modules" / "hidden")

    found, scanned, skipped = _find_descendant_git_roots(
        quack_root,
        patterns={"node_modules"},
        with_stats=True,
    )

    assert found == [visible]
    assert hidden not in found
    assert scanned >= 2
    assert skipped >= 1


def test_ensure_gitignore_uses_quackignore_for_scan_pruning(tmp_path):
    quack_root = _make_quack_root(tmp_path / "space")
    visible = _make_git_repo(quack_root / "projects" / "visible")
    hidden = _make_git_repo(quack_root / "node_modules" / "hidden")
    (quack_root / ".quackignore").write_text("node_modules\n")

    summary = ensure_gitignore(quack_root)

    assert visible in summary.protected
    assert hidden not in summary.protected
    assert summary.skipped_dirs >= 1


def test_nested_git_repos_get_managed_block(tmp_path):
    quack_root = _make_quack_root(tmp_path / "space")
    alpha = _make_git_repo(quack_root / "projects" / "alpha")
    beta = _make_git_repo(quack_root / "projects" / "beta")
    summary = ensure_gitignore(quack_root)

    for repo in (alpha, beta):
        content = (repo / ".gitignore").read_text()
        assert BLOCK_HEADER in content
        assert "_diagrams.md" in content
        # Nested repos should not contain QUACK.md or .quack/ patterns.
        assert "QUACK.md" not in content
        assert ".quack/" not in content
    assert {alpha, beta}.issubset(set(summary.protected))
    assert summary.updated_count >= 3  # .quack/.gitignore + two nested repos


def test_nested_git_repo_block_idempotent(tmp_path):
    quack_root = _make_quack_root(tmp_path / "space")
    alpha = _make_git_repo(quack_root / "projects" / "alpha")
    ensure_gitignore(quack_root)
    first = (alpha / ".gitignore").read_text()
    ensure_gitignore(quack_root)
    second = (alpha / ".gitignore").read_text()
    assert first == second
    assert first.count(BLOCK_HEADER) == 1


def test_nested_git_repo_preserves_user_lines(tmp_path):
    quack_root = _make_quack_root(tmp_path / "space")
    alpha = _make_git_repo(quack_root / "projects" / "alpha")
    (alpha / ".gitignore").write_text("*.log\nbuild/\n")
    ensure_gitignore(quack_root)
    content = (alpha / ".gitignore").read_text()
    assert "*.log" in content
    assert "build/" in content
    assert BLOCK_HEADER in content


def test_quack_root_not_in_git_nested_repos_still_get_block(tmp_path):
    """Quack root outside any git repo: nested repos still get the block."""
    quack_root = _make_quack_root(tmp_path / "space")  # no parent .git
    alpha = _make_git_repo(quack_root / "projects" / "alpha")
    ensure_gitignore(quack_root)
    content = (alpha / ".gitignore").read_text()
    assert BLOCK_HEADER in content
    assert "_diagrams.md" in content


def test_opt_out_skips_nested_repos(tmp_path):
    quack_root = _make_quack_root(tmp_path / "space")
    alpha = _make_git_repo(quack_root / "projects" / "alpha")
    (quack_root / ".quack" / "config.yaml").write_text(
        yaml.safe_dump({"gitignore": False})
    )
    summary = ensure_gitignore(quack_root)
    assert not (alpha / ".gitignore").exists()
    assert summary.opted_out is True
    assert "gitignore: false" in summary.format(quack_root)
