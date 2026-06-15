"""The editable per-folder metadata store: ``<folder>/.index.yaml``.

This is the one place you (or an assistant) author metadata for **any** file —
Markdown or not. Each index describes its folder's **direct children**: a
``files:`` section and a ``directories:`` section (immediate subfolders),
symmetric with how a file is described by its parent::

    directories:
      sub:                   # immediate subfolder name
        description: ...      # authored — preserved across reindex
        tags: [...]           # authored — preserved across reindex
        n_files: 12           # derived — refreshed
        diagram: sub/_diagrams.md  # derived pointer — refreshed
        types: {py: 10, md: 2}     # derived rollup — refreshed
        described_at: ...     # when description/tags were last written ('' ⇒ derived)
    files:
      app/main.py:           # key is the filename (basename, with extension)
        description: ...      # authored — preserved across reindex
        tags: [...]           # authored — preserved across reindex
        links: [...]          # derived from [[wikilinks]] — refreshed
        file_modified: ...    # the file's mtime at last index (derived)
        described_at: ...     # when description/tags were last written

``quack reindex`` MERGES rather than overwrites: authored ``description`` /
``tags`` (and their ``described_at``) are preserved, derived fields are
refreshed, new children appear, and vanished children drop out.

**Non-sticky defaults.** A stored ``description``/``tags`` with a blank
``described_at`` is a *derived* value (a recognition default or an
un-described placeholder), not hand-authored. It is read back as **not
authored**, so it never masquerades as authored text and re-derives cleanly
each reindex (and is freely overridable by ``describe``/``generate``).

A description is **stale** when ``file_modified`` is newer than
``described_at`` (the file changed after it was described).

One ``.index.yaml`` per directory, so each child's metadata lives next to it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Prefer libyaml's C dumper/loader when present — ~5x faster YAML, same output,
# no extra dependency (it ships with PyYAML where libyaml is available). Falls
# back to the pure-Python versions otherwise.
_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def fast_dump(data) -> str:
    return yaml.dump(data, Dumper=_DUMPER, sort_keys=False, allow_unicode=True)


def fast_load(text: str):
    return yaml.load(text, Loader=_LOADER)


INDEX_NAME = ".index.yaml"
FILES_KEY = "files"
DIRS_KEY = "directories"

# Legacy/readability placeholder written for files that have no description yet.
# Treated as blank on read, so it never becomes real authored text.
NO_DESC = "(no description)"

HEADER = (
    "# EDITABLE metadata store for this folder, managed by quack.\n"
    "# Author `description` and `tags` (under `files:` or `directories:`); they\n"
    "# are PRESERVED across `quack reindex`. Everything else is derived and\n"
    "# refreshed automatically. All of it mirrors into .quack/quack.duckdb.\n"
)


def index_path(folder: Path) -> Path:
    return folder / INDEX_NAME


def _norm_tags(raw) -> list[str]:
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return [str(t).strip() for t in (raw or []) if str(t).strip()]


def _read_doc(path: Path) -> dict:
    """Return the parsed top-level document, tolerating malformed YAML."""
    if not path.exists():
        return {}
    try:
        data = fast_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _read(path: Path) -> dict:
    """Return the ``files:`` mapping, tolerating the legacy flat layout."""
    data = _read_doc(path)
    if not data:
        return {}
    files = data.get(FILES_KEY)
    if files is None:
        # Legacy flat layout: the whole doc was the files map (no `files:`).
        files = {k: v for k, v in data.items() if k != DIRS_KEY}
    return files if isinstance(files, dict) else {}


def _read_dirs(path: Path) -> dict:
    """Return the ``directories:`` mapping ({} if absent)."""
    dirs = _read_doc(path).get(DIRS_KEY)
    return dirs if isinstance(dirs, dict) else {}


def _authored_pair(meta: dict) -> dict:
    """Apply the non-sticky rule to one stored entry, returning the authored
    ``{description, tags, described_at}``. A blank ``described_at`` means the
    value is derived, so description/tags come back empty."""
    described_at = str(meta.get("described_at") or "").strip()
    desc = str(meta.get("description") or "").strip()
    if desc == NO_DESC:
        desc = ""
    tags = _norm_tags(meta.get("tags"))
    if not described_at:
        desc, tags = "", []  # non-sticky: derived, not hand-authored
    return {"description": desc, "tags": tags, "described_at": described_at}


def load_authored(folder: Path) -> dict[str, dict]:
    """Return ``{filename: {"description", "tags", "described_at"}}`` for the
    authored file fields, applying the non-sticky rule."""
    out: dict[str, dict] = {}
    for name, meta in _read(index_path(folder)).items():
        if isinstance(meta, dict):
            out[str(name)] = _authored_pair(meta)
    return out


def load_authored_dirs(folder: Path) -> dict[str, dict]:
    """Return ``{subdir: {"description", "tags", "described_at"}}`` for the
    authored subfolder fields, applying the non-sticky rule."""
    out: dict[str, dict] = {}
    for name, meta in _read_dirs(index_path(folder)).items():
        if isinstance(meta, dict):
            out[str(name)] = _authored_pair(meta)
    return out


def _file_body(entries: list[dict]) -> dict:
    return {
        e["name"]: {
            "description": (e.get("description") or "").strip() or NO_DESC,
            "tags": _norm_tags(e.get("tags")),
            "links": list(e.get("links") or []),
            "file_modified": e.get("file_modified") or "",
            "described_at": e.get("described_at") or "",
        }
        for e in entries
    }


def _dir_body(dirs: list[dict]) -> dict:
    return {
        d["name"]: {
            "description": (d.get("description") or "").strip() or NO_DESC,
            "tags": _norm_tags(d.get("tags")),
            "n_files": int(d.get("n_files") or 0),
            "diagram": d.get("diagram") or "",
            "types": dict(d.get("types") or {}),
            "described_at": d.get("described_at") or "",
        }
        for d in dirs
    }


def write_index(
    folder: Path, entries: list[dict], dirs: list[dict] | None = None
) -> Path:
    """Write ``<folder>/.index.yaml`` from ordered child entries.

    ``entries`` carry ``name, description, tags, links, file_modified,
    described_at``; ``dirs`` (immediate subfolders) carry ``name, description,
    tags, n_files, diagram, types, described_at``. The ``directories:`` section
    is written first (and only when there are subfolders)."""
    doc: dict = {}
    if dirs:
        doc[DIRS_KEY] = _dir_body(dirs)
    doc[FILES_KEY] = _file_body(entries)
    path = index_path(folder)
    path.write_text(
        HEADER + fast_dump(doc)
    )
    return path


def set_meta(
    folder: Path,
    name: str,
    description: str,
    tags: list[str],
    described_at: str,
    section: str = FILES_KEY,
) -> None:
    """Update one child's authored ``description``/``tags`` (and stamp
    ``described_at``) in place, preserving every other entry and field. Used by
    ``quack generate``/``describe``; the next ``quack reindex`` re-derives the
    rest. Pass ``section="directories"`` to author a subfolder description."""
    path = index_path(folder)
    doc = dict(_read_doc(path))
    bucket = dict(doc.get(section) if isinstance(doc.get(section), dict) else {})
    entry = bucket.get(name)
    entry = dict(entry) if isinstance(entry, dict) else {}
    entry["description"] = (description or "").strip() or NO_DESC
    entry["tags"] = _norm_tags(tags) if tags else entry.get("tags", [])
    entry["described_at"] = described_at
    if section == FILES_KEY:
        entry.setdefault("links", [])
        entry.setdefault("file_modified", "")
    else:
        entry.setdefault("n_files", 0)
        entry.setdefault("diagram", "")
        entry.setdefault("types", {})
    bucket[name] = entry
    doc[section] = bucket
    path.write_text(
        HEADER + fast_dump(doc)
    )
