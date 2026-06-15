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

from pathlib import Path

BLOCK_START = "# >>> quack (managed) >>>"
BLOCK_END = "# <<< quack (managed) <<<"

# Patterns that match anywhere in the tree — file names quack generates
# into every folder. .index.yaml is included: descriptions are cheap to
# regenerate so there's no value in tracking them in git.
_TREE_PATTERNS = [".index.yaml", "_diagrams.md"]


def _find_git_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _find_descendant_git_roots(path: Path) -> list[Path]:
    """Git repos whose .git sits beneath *path* (not at *path* itself)."""
    result = []
    try:
        for child in path.iterdir():
            if not child.is_dir() or child.name == ".git":
                continue
            if (child / ".git").is_dir():
                result.append(child)
            result.extend(_find_descendant_git_roots(child))
    except (PermissionError, OSError):
        pass
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


def _apply_block(gitignore_path: Path, block: str) -> None:
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
    else:
        sep = "\n" if content and not content.endswith("\n") else ""
        gitignore_path.write_text(content + sep + block)


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


def ensure_gitignore(quack_root: Path) -> None:
    """Idempotently manage the quack block in all relevant git .gitignore files.

    Manages the block in:
    - the nearest ancestor git repo (if the quack root lives inside one), and
    - every nested git repo beneath the quack root.

    Also ensures `.quack/.gitignore` exists with `*` so the state dir is
    self-ignoring. No-ops when opted out via config.
    """
    # Always keep the state dir self-ignoring.
    quack_dir = quack_root / ".quack"
    quack_dir.mkdir(exist_ok=True)
    self_ignore = quack_dir / ".gitignore"
    if not self_ignore.exists() or self_ignore.read_text().strip() != "*":
        self_ignore.write_text("*\n")

    if _gitignore_opt_out(quack_root):
        return

    # Ancestor repo: quack root lives inside a git repo.
    git_root = _find_git_root(quack_root)
    if git_root is not None:
        _apply_block(git_root / ".gitignore", _build_block(quack_root, git_root))

    # Nested repos: git repos that live beneath the quack root.
    for nested_root in _find_descendant_git_roots(quack_root):
        _apply_block(nested_root / ".gitignore", _build_nested_block())


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
