"""Scaffold new notes with correct frontmatter so descriptions are never
forgotten, the one habit the whole system depends on."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import frontmatter
import yaml

from .core import MARKER_DIR, find_root

TEMPLATE_KEYS = ("description", "tags", "created", "updated")


def new_note(
    title: str,
    folder: str = "projects",
    description: str = "",
    tags: list[str] | None = None,
    today: str | None = None,
    explicit_root: str | None = None,
) -> Path:
    """Create folder/<slug>.md with a filled frontmatter template.

    `today` is injected (not read from the clock) so callers stay
    deterministic and testable.
    """
    root = find_root(explicit_root)
    day = today or _dt.date.today().isoformat()
    slug = _slugify(title)
    target_dir = root / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{slug}.md"
    if path.exists():
        raise FileExistsError(path)

    post = frontmatter.Post(
        content=f"# {title}\n\n",
        description=description,
        tags=tags or [],
        created=day,
        updated=day,
    )
    path.write_text(frontmatter.dumps(post) + "\n")
    return path


ROOT_QUACK_MD = """# QUACK — how to navigate this work

This directory is managed by `quack`: a **meta layer** over your files (notes,
docs, code, configs, assets) so an LLM can find anything without reading
everything. Use it; do not grep the files blindly. quack works on any folder;
it plays well with Obsidian but does not require it.

## Two ways to use it

**A. MCP tools (preferred).** If the `quack` MCP server is connected, call its
tools (`map`, `search`, `get_file`, `sql`, `graph_path`, `central`, `clusters`);
each returns `root` so you can join `root` + the relative path to open a file.

**B. The `quack` CLI:**
```
quack search "<terms>"        # auto-hybrid search, prints the root
quack sql "SELECT ..."         # tables: files, tags, links
quack graph path A B           # shortest link path
```

## The retrieval ladder (cheapest first — read only what you need)

1. `.quack/map.yaml` — global nested tree. Which folder is relevant?
2. `<folder>/.index.yaml` — its direct `files:` + `directories:`. Which 1–3
   children match?
3. The file itself.
4. `search` / `sql` / `graph_*` — pull only the related slice from the catalog.

## Ground rules

- Each folder's **editable** `.index.yaml` describes its direct children — a
  `files:` section (`description`, `tags`; `links` derived from `[[wikilinks]]`)
  and a `directories:` section for its subfolders. Well-known files/folders get
  a recognition default; precedence is authored → frontmatter → recognition →
  blank. Markdown may also carry frontmatter. `quack reindex` merges it all into
  `.quack/quack.duckdb` (tables: files, folders, tags, links).
- Ignore patterns live in `.quackignore` at the root.
- Full conventions and details: `quack --help`, `quack mcp print`, and the project README.
"""

ROOT_QUACKIGNORE = """# One pattern per line. Names or root-relative paths; globs allowed.
# Built-ins (.quack, .obsidian, .git, .trash, node_modules) are always ignored.
# Common build/dependency dirs for code projects (edit to taste):
.venv
dist
build
__pycache__
target
.next
*.lock
"""

STARTER_FOLDERS = ("projects", "resources")


def _inherited_defaults(root: Path) -> dict | None:
    """Defaults inherit from the nearest parent quack space, if any."""
    for parent in root.parents:
        cfg = parent / MARKER_DIR / "config.yaml"
        if not cfg.exists():
            continue
        try:
            data = yaml.safe_load(cfg.read_text()) or {}
        except yaml.YAMLError:
            return None
        defaults = data.get("defaults") if isinstance(data, dict) else None
        return dict(defaults) if isinstance(defaults, dict) else None
    return None


def _merge_defaults(config_path: Path, defaults: dict | None) -> None:
    if not defaults:
        return
    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data["defaults"] = defaults
    config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def scaffold_root(target: str | None = None) -> Path:
    """Turn a directory into a quack root, creating it if needed. Writes only
    what is missing, so it is safe to re-run on an existing space.

    Lays down the `.quack/` marker (+ a neutral config), the visible `QUACK.md`
    navigation anchor, and a `.quackignore`. Starter content folders are created
    only for a brand-new (empty) space, never inside an existing project.
    """
    root = Path(target).expanduser().resolve() if target else Path.cwd().resolve()
    is_new = not root.exists() or not any(root.iterdir())
    (root / ".quack").mkdir(parents=True, exist_ok=True)
    if is_new:
        for folder in STARTER_FOLDERS:
            (root / folder).mkdir(parents=True, exist_ok=True)

    qmd = root / "QUACK.md"
    if not qmd.exists():
        qmd.write_text(ROOT_QUACK_MD)
    ign = root / ".quackignore"
    if not ign.exists():
        ign.write_text(ROOT_QUACKIGNORE)

    cfg = root / ".quack" / "config.yaml"
    if not cfg.exists():
        from .config import write_config  # neutral default; `quack setup` picks

        inherited = _inherited_defaults(root)
        write_config(command="", explicit_root=str(root))
        _merge_defaults(cfg, inherited)

    from .gitignore import ensure_gitignore

    ensure_gitignore(root)
    return root


def _slugify(title: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "untitled"
