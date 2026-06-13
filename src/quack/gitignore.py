"""Manage a quack-owned block in the nearest git repo's .gitignore.

On `quack init` and `quack reindex`, if the quack root is inside a git
repository, this module idempotently inserts (or refreshes) a clearly
delimited block in the repo's root `.gitignore`.  It never touches the
user's own lines.  A `gitignore: false` entry in `.quack/config.yaml`
opts the workspace out entirely.

It also writes `.quack/.gitignore` containing `*` so the whole state
directory self-ignores, regardless of root configuration.
"""

from __future__ import annotations

from pathlib import Path

BLOCK_START = "# >>> quack (managed) >>>"
BLOCK_END = "# <<< quack (managed) <<<"

# Patterns without a leading slash match anywhere in the tree; these file
# names are quack-specific so global matching is intentional.
_TREE_PATTERNS = [".index.yaml", "_diagrams.md"]


def _find_git_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


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
    """Idempotently manage the quack block in the nearest git repo's .gitignore.

    Also ensures `.quack/.gitignore` exists with `*` so the state dir is
    self-ignoring.  No-ops when not inside a git repo or when opted out.
    """
    # Always keep the state dir self-ignoring.
    quack_dir = quack_root / ".quack"
    quack_dir.mkdir(exist_ok=True)
    self_ignore = quack_dir / ".gitignore"
    if not self_ignore.exists() or self_ignore.read_text().strip() != "*":
        self_ignore.write_text("*\n")

    git_root = _find_git_root(quack_root)
    if git_root is None:
        return

    if _gitignore_opt_out(quack_root):
        return

    gitignore_path = git_root / ".gitignore"
    block = _build_block(quack_root, git_root)
    content = gitignore_path.read_text() if gitignore_path.exists() else ""

    if BLOCK_START in content:
        start = content.index(BLOCK_START)
        end_marker = content.find(BLOCK_END, start)
        if end_marker == -1:
            # Malformed: no closing marker — replace from the opening marker.
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
