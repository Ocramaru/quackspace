"""Manage a quack-owned block in git repo .gitignore files.

On `quack init` and `quack reindex`, this module idempotently inserts (or
refreshes) a clearly delimited block in:

- the nearest ancestor git repo's .gitignore (if the quack root lives inside
  one), and
- every nested git repo's .gitignore (repos whose .git sits beneath the quack
  root).

It never touches the user's own lines.  A `gitignore: false` entry in
`.quack/config.yaml` opts the workspace out entirely.

It also writes `.quack/.gitignore` containing `*` so the whole state
directory self-ignores, regardless of root configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

BLOCK_START = "# >>> quack (managed) >>>"
BLOCK_END = "# <<< quack (managed) <<<"

# Patterns that match anywhere in the tree — file names quack generates
# into every folder. .index.yaml is included: descriptions are cheap to
# regenerate so there's no value in tracking them in git.
_TREE_PATTERNS = [".index.yaml", "_diagrams.md"]


@dataclass
class GitignoreSummary:
    """What gitignore management touched during init/reindex."""

    self_ignore: Path | None = None
    updated: list[Path] = field(default_factory=list)
    protected: list[Path] = field(default_factory=list)
    scanned_dirs: int = 0
    skipped_dirs: int = 0
    opted_out: bool = False

    @property
    def updated_count(self) -> int:
        return len(self.updated)

    @property
    def protected_count(self) -> int:
        return len(self.protected)

    def format(self, root: Path) -> str:
        if self.opted_out:
            if self.self_ignore in self.updated:
                return "gitignore: skipped repo files (gitignore: false); wrote .quack/.gitignore"
            return "gitignore: skipped (gitignore: false)"
        if self.protected_count:
            suffix = f"; scanned {self.scanned_dirs:,} folder(s)"
            if self.skipped_dirs:
                suffix += f", skipped {self.skipped_dirs:,}"
            return (
                f"gitignore: updated {self.updated_count:,} file(s), "
                f"protected {self.protected_count:,} git repo(s){suffix}"
            )
        if self.self_ignore in self.updated:
            return "gitignore: wrote .quack/.gitignore; no git repos found"
        return "gitignore: already up to date; no git repos found"


def _rel_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _find_git_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _ignored(name: str, rel: str, patterns: set[str]) -> bool:
    for pat in patterns:
        if name == pat or rel == pat or fnmatch(name, pat) or fnmatch(rel, pat):
            return True
    return False


def _find_descendant_git_roots(
    path: Path,
    patterns: set[str] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    with_stats: bool = False,
) -> list[Path] | tuple[list[Path], int, int]:
    """Git repos whose .git sits beneath *path* (not at *path* itself)."""
    result: list[Path] = []
    scanned = 0
    skipped = 0
    ignore_patterns = patterns or set()
    for dirpath, dirnames, _filenames in os.walk(path):
        base = Path(dirpath)
        scanned += 1
        if progress is not None and (scanned == 1 or scanned % 100 == 0):
            progress(scanned, max(scanned + 1, 1), f"Scanning {_rel_display(base, path)}")
        if base != path and (base / ".git").exists():
            result.append(base)
        kept: list[str] = []
        for name in dirnames:
            rel = (base / name).relative_to(path).as_posix()
            if name == ".git" or _ignored(name, rel, ignore_patterns):
                skipped += 1
                continue
            kept.append(name)
        dirnames[:] = kept
    if progress is not None:
        progress(scanned, max(scanned, 1), "Scanned nested git repos")
    if with_stats:
        return result, scanned, skipped
    return result


def _build_block(quack_root: Path, git_root: Path) -> str:
    lines = [BLOCK_START]
    lines.extend(_TREE_PATTERNS)

    # QUACK.md and .quack/ are anchored to the quack root within the git tree.
    try:
        rel = quack_root.relative_to(git_root)
        prefix = rel.as_posix() + "/" if rel != Path(".") else ""
    except ValueError:
        prefix = ""

    lines.append(f"{prefix}QUACK.md")
    lines.append(f"{prefix}.quack/")
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def _build_nested_block() -> str:
    """Block for nested git repos: only tree-wide patterns, no anchored paths."""
    lines = [BLOCK_START]
    lines.extend(_TREE_PATTERNS)
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def _apply_block(gitignore_path: Path, block: str) -> bool:
    """Idempotently insert or refresh the managed block in a .gitignore file."""
    content = gitignore_path.read_text() if gitignore_path.exists() else ""

    if BLOCK_START in content:
        start = content.index(BLOCK_START)
        end_marker = content.find(BLOCK_END, start)
        if end_marker == -1:
            new_content = content[:start] + block
        else:
            end = end_marker + len(BLOCK_END)
            if end < len(content) and content[end] == "\n":
                end += 1
            new_content = content[:start] + block + content[end:]
        if new_content != content:
            gitignore_path.write_text(new_content)
            return True
        return False
    else:
        sep = "\n" if content and not content.endswith("\n") else ""
        gitignore_path.write_text(content + sep + block)
        return True


def _gitignore_opt_out(quack_root: Path) -> bool:
    config_path = quack_root / ".quack" / "config.yaml"
    if not config_path.exists():
        return False
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text()) or {}
        if isinstance(data, dict):
            return not data.get("gitignore", True)
    except Exception:
        pass
    return False


def ensure_gitignore(
    quack_root: Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> GitignoreSummary:
    """Idempotently manage the quack block in all relevant git .gitignore files.

    Manages the block in:
    - the nearest ancestor git repo (if the quack root lives inside one), and
    - every nested git repo beneath the quack root.

    Also ensures `.quack/.gitignore` exists with `*` so the state dir is
    self-ignoring. No-ops when opted out via config.
    """
    summary = GitignoreSummary()
    if progress is not None:
        progress(0, 1, "Preparing gitignore rules")

    # Always keep the state dir self-ignoring.
    quack_dir = quack_root / ".quack"
    quack_dir.mkdir(exist_ok=True)
    self_ignore = quack_dir / ".gitignore"
    summary.self_ignore = self_ignore
    if not self_ignore.exists() or self_ignore.read_text().strip() != "*":
        self_ignore.write_text("*\n")
        summary.updated.append(self_ignore)

    if _gitignore_opt_out(quack_root):
        summary.opted_out = True
        if progress is not None:
            progress(1, 1, "Skipped gitignore management")
        return summary

    # Ancestor repo: quack root lives inside a git repo.
    git_root = _find_git_root(quack_root)
    if git_root is not None:
        summary.protected.append(git_root)
        path = git_root / ".gitignore"
        if _apply_block(path, _build_block(quack_root, git_root)):
            summary.updated.append(path)

    # Nested repos: git repos that live beneath the quack root.
    try:
        from .core import load_ignores

        patterns = load_ignores(quack_root)
    except Exception:
        patterns = {".git", ".quack"}
    if progress is not None:
        progress(0, 1, "Scanning nested git repos")
    nested_roots, scanned, skipped = _find_descendant_git_roots(
        quack_root, patterns=patterns, progress=progress, with_stats=True
    )
    summary.scanned_dirs = scanned
    summary.skipped_dirs = skipped
    for nested_root in nested_roots:
        summary.protected.append(nested_root)
        path = nested_root / ".gitignore"
        if _apply_block(path, _build_nested_block()):
            summary.updated.append(path)
    if progress is not None:
        progress(1, 1, "Managed gitignore rules")
    return summary


def remove_gitignore(quack_root: Path) -> bool:
    """Strip the quack-managed block from the nearest git repo's .gitignore,
    leaving the user's own lines intact. Returns True if a block was removed."""
    git_root = _find_git_root(quack_root)
    if git_root is None:
        return False
    gitignore_path = git_root / ".gitignore"
    if not gitignore_path.exists():
        return False
    content = gitignore_path.read_text()
    if BLOCK_START not in content:
        return False
    start = content.index(BLOCK_START)
    end_marker = content.find(BLOCK_END, start)
    if end_marker == -1:
        new_content = content[:start]
    else:
        end = end_marker + len(BLOCK_END)
        if end < len(content) and content[end] == "\n":
            end += 1
        new_content = content[:start] + content[end:]
    # Tidy a trailing blank left where the block was.
    new_content = new_content.rstrip("\n") + "\n" if new_content.strip() else ""
    if new_content:
        gitignore_path.write_text(new_content)
    else:
        gitignore_path.unlink()
    return True
