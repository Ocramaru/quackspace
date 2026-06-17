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

BLOCK_HEADER = "# ignore quackspace files"

# File-name patterns quack generates into every folder.
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


def _find_descendant_git_roots(
    path: Path,
    patterns: "IgnoreRuleset | None" = None,
    progress: Callable[[int, int, str], None] | None = None,
    with_stats: bool = False,
) -> "list[Path] | tuple[list[Path], int, int]":
    """Git repos whose .git sits beneath *path* (not at *path* itself)."""
    from .core import DEFAULT_IGNORED_DIRS, IgnoreRuleset
    result: list[Path] = []
    scanned = 0
    skipped = 0
    ignore_rules = patterns or IgnoreRuleset.build(DEFAULT_IGNORED_DIRS, [])
    for dirpath, dirnames, _filenames in os.walk(path):
        base = Path(dirpath)
        scanned += 1
        if progress is not None and (scanned == 1 or scanned % 100 == 0):
            progress(scanned, -1, f"Waddling through files: {scanned:,}")
        if base != path and (base / ".git").exists():
            result.append(base)
        kept: list[str] = []
        for name in dirnames:
            rel = (base / name).relative_to(path).as_posix()
            if name == ".git" or ignore_rules.is_ignored(name, rel):
                skipped += 1
                continue
            kept.append(name)
        dirnames[:] = kept
    if progress is not None:
        progress(scanned, -1, f"Checked {scanned:,} folder(s) for git repos")
    if with_stats:
        return result, scanned, skipped
    return result


def _build_block(quack_root: Path, git_root: Path) -> str:
    try:
        rel = quack_root.relative_to(git_root)
        prefix = rel.as_posix() + "/" if rel != Path(".") else ""
    except ValueError:
        prefix = ""
    patterns = [*_TREE_PATTERNS, f"{prefix}QUACK.md", f"{prefix}.quack/"]
    return "\n" + "\n".join([BLOCK_HEADER, *patterns]) + "\n"


def _build_nested_block() -> str:
    return "\n" + "\n".join([BLOCK_HEADER, *_TREE_PATTERNS]) + "\n"


def _find_block(content: str) -> tuple[int, int] | None:
    """Return (start, end) of the quack block in *content*, or None if absent.
    start includes the preceding blank-line separator; end is the position just
    after the last pattern line."""
    if BLOCK_HEADER not in content:
        return None
    header_pos = content.index(BLOCK_HEADER)
    start = header_pos - 1 if header_pos > 0 and content[header_pos - 1] == "\n" else header_pos
    pos = content.find("\n", header_pos)
    if pos == -1:
        return start, len(content)
    pos += 1
    while pos < len(content):
        line_end = content.find("\n", pos)
        line = content[pos:line_end] if line_end != -1 else content[pos:]
        if not line or line.startswith("#"):
            break
        pos = line_end + 1 if line_end != -1 else len(content)
    return start, pos


def _apply_block(gitignore_path: Path, block: str) -> bool:
    """Idempotently insert or refresh the managed block in a .gitignore file."""
    content = gitignore_path.read_text() if gitignore_path.exists() else ""
    new_content = _content_with_block(content, block)
    if new_content != content:
        gitignore_path.write_text(new_content)
        return True
    return False


def _content_with_block(content: str, block: str) -> str:
    block_range = _find_block(content)
    if block_range:
        s, e = block_range
        return content[:s] + block + content[e:]
    if content and not content.endswith("\n"):
        content += "\n"
    return content + block


def _would_apply_block(gitignore_path: Path, block: str) -> bool:
    content = gitignore_path.read_text() if gitignore_path.exists() else ""
    return _content_with_block(content, block) != content


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
        from .core import IgnoreRuleset, load_ignores

        patterns = load_ignores(quack_root)
    except Exception:
        from .core import DEFAULT_IGNORED_DIRS, IgnoreRuleset
        patterns = IgnoreRuleset.build(DEFAULT_IGNORED_DIRS, [])
    if progress is not None:
        progress(0, -1, "Waddling through files: 0")
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


def preview_gitignore(quack_root: Path) -> GitignoreSummary:
    """Return the gitignore summary that would be produced, without writing."""
    summary = GitignoreSummary()
    self_ignore = quack_root / ".quack" / ".gitignore"
    summary.self_ignore = self_ignore
    if not self_ignore.exists() or self_ignore.read_text().strip() != "*":
        summary.updated.append(self_ignore)

    if _gitignore_opt_out(quack_root):
        summary.opted_out = True
        return summary

    git_root = _find_git_root(quack_root)
    if git_root is not None:
        summary.protected.append(git_root)
        path = git_root / ".gitignore"
        if _would_apply_block(path, _build_block(quack_root, git_root)):
            summary.updated.append(path)

    try:
        from .core import IgnoreRuleset, load_ignores

        patterns = load_ignores(quack_root)
    except Exception:
        from .core import DEFAULT_IGNORED_DIRS, IgnoreRuleset
        patterns = IgnoreRuleset.build(DEFAULT_IGNORED_DIRS, [])
    if quack_root.exists():
        nested_roots, scanned, skipped = _find_descendant_git_roots(
            quack_root, patterns=patterns, with_stats=True
        )
    else:
        nested_roots, scanned, skipped = [], 0, 0
    summary.scanned_dirs = scanned
    summary.skipped_dirs = skipped
    for nested_root in nested_roots:
        summary.protected.append(nested_root)
        path = nested_root / ".gitignore"
        if _would_apply_block(path, _build_nested_block()):
            summary.updated.append(path)
    return summary


def remove_gitignore(quack_root: Path, dry_run: bool = False) -> bool:
    """Strip the quack-managed block from the nearest git repo's .gitignore
    and every nested git repo beneath the quack root, leaving user lines intact.
    Returns True if at least one block was removed."""
    git_root = _find_git_root(quack_root)
    candidates = ([git_root] if git_root else []) + list(_find_descendant_git_roots(quack_root))
    removed = False

    for root in candidates:
        gitignore_path = root / ".gitignore"
        if not gitignore_path.exists():
            continue
        content = gitignore_path.read_text()
        block_range = _find_block(content)
        if block_range is None:
            continue
        start, end = block_range
        new_content = content[:start] + content[end:]
        new_content = new_content.rstrip("\n") + "\n" if new_content.strip() else ""
        if not dry_run:
            if new_content:
                gitignore_path.write_text(new_content)
            else:
                gitignore_path.unlink()
        removed = True
    return removed
