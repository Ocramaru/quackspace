"""Remove quack's generated layer from a space.

Two levels, because they differ wildly in reversibility:

* **derived** (default): delete only regenerable artifacts — the DuckDB
  catalog, ``map.yaml``, and the global + per-folder Mermaid diagrams. A
  ``quack reindex`` rebuilds all of it. Authored metadata (``.index.yaml``),
  config, and ``QUACK.md`` are left untouched.
* **purge** (``--all``): uninstall quack from the space entirely — also delete
  every ``.index.yaml`` (which holds *authored* descriptions, so this loses
  data), ``QUACK.md``, ``.quackignore``, strip the managed ``.gitignore``
  block, and remove the ``.quack/`` state directory.

How it finds things: the catalog already maps every folder, so we read it for a
fast, targeted list of known artifacts; then a pruned filesystem scan catches
any **stragglers** the catalog doesn't know about (a folder added since the last
reindex, a stale `.index.yaml`, etc.). The two are unioned, and the count of
search-only extras is reported back.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import catalog, core
from .catalog import DB_NAME
from .core import MARKER_DIR, find_root

# Per-folder artifacts quack generates (one of each, at most, per folder).
INDEX_NAME = ".index.yaml"
DIAGRAM_NAME = "_diagrams.md"


def _known_from_catalog(root: Path) -> set[Path]:
    """Artifact paths the catalog already knows about (fast, no walk). Empty if
    there is no readable catalog."""
    db = root / MARKER_DIR / DB_NAME
    if not db.exists():
        return set()
    try:
        rows = catalog.list_folders_path(db)
    except Exception:
        return set()
    known: set[Path] = set()
    # The root itself isn't a row in `folders`; add it explicitly.
    for folder_rel in ["", *[r for r, _p, _d in rows]]:
        base = root if folder_rel == "" else root / folder_rel
        known.add(base / INDEX_NAME)
        known.add(base / DIAGRAM_NAME)
    return known


def _found_on_disk(root: Path) -> set[Path]:
    """Every quack per-folder artifact actually present, via a pruned scan (so we
    never wander into vendored/ignored trees). The catch-all 'search'."""
    found: set[Path] = set()
    for folder in [root, *core.iter_content_folders(root)]:
        for name in (INDEX_NAME, DIAGRAM_NAME):
            p = folder / name
            if p.exists():
                found.add(p)
    return found


def clean(
    explicit_root: str | None = None,
    purge: bool = False,
    dry_run: bool = False,
    targets: set[str] | None = None,
) -> dict:
    """Remove quack's generated artifacts. Returns counts of what was removed,
    including ``extras`` (artifacts found by the disk scan that the catalog map
    didn't list). With *purge*, fully uninstalls the quack layer (destructive —
    drops authored ``.index.yaml`` metadata)."""
    root = find_root(explicit_root)
    quack_dir = root / MARKER_DIR
    removed = {"catalog": 0, "map": 0, "diagrams": 0, "indexes": 0, "other": 0}
    selected = targets or {"catalog", "map", "diagrams"}
    if purge:
        selected = {"catalog", "map", "diagrams", "indexes", "other"}

    # Fast map from the catalog + the catch-all scan; union, note the extras.
    known = _known_from_catalog(root)
    found = _found_on_disk(root)
    extras = found - known
    artifacts = known | found

    # Derived state files inside .quack/ (config is kept on a derived clean).
    for name, key in ((DB_NAME, "catalog"), ("map.yaml", "map"), ("diagram.md", "diagrams")):
        if key not in selected:
            continue
        p = quack_dir / name
        if p.exists():
            if not dry_run:
                p.unlink()
            removed[key] += 1

    for p in artifacts:
        if p.name == DIAGRAM_NAME and "diagrams" in selected and p.exists():
            if not dry_run:
                p.unlink()
            removed["diagrams"] += 1
        elif p.name == INDEX_NAME and "indexes" in selected and p.exists():
            if not dry_run:
                p.unlink()
            removed["indexes"] += 1

    if "other" in selected:
        from .gitignore import remove_gitignore
        from .kiro import hook_definitions

        for name in ("QUACK.md", ".quackignore", ".mcp.json"):
            p = root / name
            if p.exists():
                if not dry_run:
                    p.unlink()
                removed["other"] += 1
        # Remove only quack's own Kiro hooks, never the user's .kiro/ config.
        hooks_dir = root / ".kiro" / "hooks"
        for slug in hook_definitions():
            hook = hooks_dir / f"{slug}.kiro.hook"
            if hook.exists():
                if not dry_run:
                    hook.unlink()
                removed["other"] += 1
        if remove_gitignore(root, dry_run=dry_run):
            removed["other"] += 1
        if quack_dir.exists():
            if not dry_run:
                shutil.rmtree(quack_dir)
            removed["other"] += 1

    removed["extras"] = sum(
        1
        for p in extras
        if (p.name == DIAGRAM_NAME and "diagrams" in selected)
        or (p.name == INDEX_NAME and "indexes" in selected)
    )
    removed["targets"] = sorted(selected)
    return removed
