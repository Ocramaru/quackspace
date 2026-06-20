"""The shared folder resolver: one source of folder metadata for the whole
meta layer.

Per-folder ``.index.yaml`` ``directories:`` sections, the top-level
``map.yaml`` tree, and the DuckDB ``folders`` table are all generated from
``resolve_folders``, so they can never drift from one another.

For each folder it computes:

  * a resolved **description** (parent-index authored → that folder's
    ``.folder.md`` frontmatter → folder recognition default → blank), with a
    blank ``described_at`` for any derived/recognition default (non-sticky),
  * the **direct file count**, an **extension rollup**, and a **tag rollup**,
  * a **diagram** pointer when ``<folder>/_diagrams.md`` exists.

A folder is described by its *parent's* index, symmetric with how a file is —
so the root has no description of its own.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import core, index_store, recognize
from .config import DEFAULT_TAG_ROLLUP_LIMIT
from .core import Space


@dataclass
class FolderInfo:
    rel: str  # root-relative POSIX path; "" for the root itself
    parent: str  # parent's rel; "" means the root is the parent
    name: str  # base name; "" for the root
    description: str
    described_at: str  # "" ⇒ derived/recognition default (non-sticky)
    tags: list[str] = field(default_factory=list)  # authored or recognition tags
    n_files: int = 0  # direct files (not recursive)
    diagram: str = ""  # pointer to <rel>/_diagrams.md, or ""
    types: dict[str, int] = field(default_factory=dict)  # ext → direct-file count
    tag_rollup: list[str] = field(default_factory=list)  # top child tags
    is_root: bool = False


def _folder_md_description(folder: Path) -> str:
    """A folder's own ``.folder.md`` frontmatter description, or ''. Tolerates a
    legacy encoding or malformed frontmatter rather than raising."""
    marker = folder / ".folder.md"
    if not marker.exists():
        return ""
    body, is_binary = core._read_text(marker)
    if is_binary:
        return ""
    return str(core.parse_frontmatter(body).get("description", "")).strip()


def _resolve_description(
    space: Space, rel: str, name: str, parent: str, folder_path: Path
) -> tuple[str, list[str], str]:
    """(description, tags, described_at) by precedence: parent-index authored →
    ``.folder.md`` → folder recognition default → blank.

    Description and tags resolve independently, mirroring how a file's overlay
    works (``core._overlay``): authored tags are preserved even when the
    description falls back to frontmatter/recognition, and vice versa."""
    root = space.root
    parent_path = root if parent == "" else root / parent
    authored = index_store.load_authored_dirs(parent_path).get(name) or {}
    a_desc = authored.get("description", "")
    a_tags = authored.get("tags", [])
    a_at = authored.get("described_at", "")

    if a_desc:
        return a_desc, a_tags, a_at

    fm = _folder_md_description(folder_path)
    if fm:
        # Authored tags (if any) keep their stamp; the description is derived.
        return (fm, a_tags, a_at) if a_tags else (fm, [], "")

    rec = recognize.recognize_folder(name)
    if rec is not None:
        return (rec[0], a_tags, a_at) if a_tags else (rec[0], list(rec[1]), "")

    return ("", a_tags, a_at) if a_tags else ("", [], "")


def _build_info(space: Space, rel: str, files: list, tag_rollup_limit: int = DEFAULT_TAG_ROLLUP_LIMIT) -> FolderInfo:
    root = space.root
    is_root = rel == ""
    folder_path = root if is_root else root / rel
    name = "" if is_root else rel.rsplit("/", 1)[-1]
    parent = "" if "/" not in rel else rel.rsplit("/", 1)[0]

    if is_root:
        desc, tags, described_at = "", [], ""
    else:
        desc, tags, described_at = _resolve_description(
            space, rel, name, parent, folder_path
        )

    # A folder skipped as a dataset (its files aren't indexed) is still recorded
    # here, marked so an agent knows what it is: a derived description when none
    # was authored, plus a `dataset` tag for filtering. n_files stays 0 because
    # no file rows exist — the reason string carries the real magnitude.
    if not is_root and rel in space.datasets:
        if not desc:
            desc = f"Dataset: {space.datasets[rel]}, not indexed."
        if "dataset" not in tags:
            tags = ["dataset", *tags]

    types: dict[str, int] = defaultdict(int)
    tag_counts: Counter = Counter()
    for e in files:
        types[e.ext or "other"] += 1
        tag_counts.update(e.tags)

    diagram = ""
    if not is_root and (folder_path / "_diagrams.md").exists():
        diagram = f"{rel}/_diagrams.md"

    return FolderInfo(
        rel=rel,
        parent=parent,
        name=name,
        description=desc,
        described_at=described_at,
        tags=tags,
        n_files=len(files),
        diagram=diagram,
        types=dict(types),
        tag_rollup=[t for t, _ in tag_counts.most_common(tag_rollup_limit)],
        is_root=is_root,
    )


def resolve_folders(space: Space, tag_rollup_limit: int = DEFAULT_TAG_ROLLUP_LIMIT) -> dict[str, FolderInfo]:
    """Map every folder's rel → ``FolderInfo``, including the root ("") and
    folders that contain only subfolders. The single source the per-folder
    indexes, ``map.yaml``, and the ``folders`` table all read from."""
    by_folder: dict[str, list] = defaultdict(list)
    for e in space.entries:
        by_folder[e.folder].append(e)

    # space.folders comes from the same walk that produced space.entries — no
    # second traversal. (Fall back to a walk only if a Space was built without
    # it, e.g. constructed by hand.)
    walked = space.folders or core.iter_content_folders(space.root)
    rels: set[str] = {""}
    for f in walked:
        rels.add(f.relative_to(space.root).as_posix())
    rels.update(by_folder)  # safety: any folder with files must be present

    return {rel: _build_info(space, rel, by_folder.get(rel, []), tag_rollup_limit) for rel in rels}


def children_index(infos: dict[str, FolderInfo]) -> dict[str, list[FolderInfo]]:
    """Map ``parent_rel → [immediate subfolders sorted by name]`` in one pass,
    so callers don't rescan all infos per folder (O(F) instead of O(F²))."""
    idx: dict[str, list[FolderInfo]] = defaultdict(list)
    for i in infos.values():
        if not i.is_root:
            idx[i.parent].append(i)
    for kids in idx.values():
        kids.sort(key=lambda i: i.name.lower())
    return idx
