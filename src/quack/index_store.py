"""The editable per-folder metadata store: ``<folder>/.index.yaml``.

This is the one place you (or an assistant) author metadata for **any** file —
Markdown or not. The file has a top-level ``files:`` section mapping a filename
to its metadata::

    files:
      app/main.py:           # key is the filename (basename, with extension)
        description: ...      # authored — preserved across reindex
        tags: [...]           # authored — preserved across reindex
        links: [...]          # derived from [[wikilinks]] — refreshed
        file_modified: ...    # the file's mtime at last index (derived)
        described_at: ...     # when description/tags were last written

``quack reindex`` MERGES rather than overwrites: authored ``description`` /
``tags`` (and their ``described_at``) are preserved, ``links`` and
``file_modified`` are refreshed, new files appear with a blank description, and
vanished files drop out. A description is **stale** when ``file_modified`` is
newer than ``described_at`` (the file changed after it was described).

One ``.index.yaml`` per directory, so each file's metadata lives next to it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

INDEX_NAME = ".index.yaml"
FILES_KEY = "files"

# Legacy/readability placeholder written for files that have no description yet.
# Treated as blank on read, so it never becomes real authored text.
NO_DESC = "(no description)"

HEADER = (
    "# EDITABLE metadata store for this folder, managed by quack.\n"
    "# Author `description` and `tags` under `files:`; they are PRESERVED across\n"
    "# `quack reindex`. `links`, `file_modified` and `described_at` are derived\n"
    "# and refreshed automatically. All of it mirrors into .quack/quack.duckdb.\n"
)


def index_path(folder: Path) -> Path:
    return folder / INDEX_NAME


def _norm_tags(raw) -> list[str]:
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return [str(t).strip() for t in (raw or []) if str(t).strip()]


def _read(path: Path) -> dict:
    """Return the parsed ``files:`` mapping, tolerating malformed YAML and the
    legacy flat (no ``files:`` wrapper) format."""
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    files = data.get(FILES_KEY, data)  # fall back to flat legacy layout
    return files if isinstance(files, dict) else {}


def load_authored(folder: Path) -> dict[str, dict]:
    """Return ``{filename: {"description", "tags", "described_at"}}`` for the
    authored fields, normalising the ``(no description)`` placeholder to ''."""
    out: dict[str, dict] = {}
    for name, meta in _read(index_path(folder)).items():
        if not isinstance(meta, dict):
            continue
        desc = str(meta.get("description") or "").strip()
        if desc == NO_DESC:
            desc = ""
        out[str(name)] = {
            "description": desc,
            "tags": _norm_tags(meta.get("tags")),
            "described_at": str(meta.get("described_at") or "").strip(),
        }
    return out


def write_index(folder: Path, entries: list[dict]) -> Path:
    """Write ``<folder>/.index.yaml`` from ordered entries with keys
    ``name, description, tags, links, file_modified, described_at``."""
    body = {
        e["name"]: {
            "description": (e.get("description") or "").strip() or NO_DESC,
            "tags": _norm_tags(e.get("tags")),
            "links": list(e.get("links") or []),
            "file_modified": e.get("file_modified") or "",
            "described_at": e.get("described_at") or "",
        }
        for e in entries
    }
    path = index_path(folder)
    path.write_text(
        HEADER
        + yaml.safe_dump({FILES_KEY: body}, sort_keys=False, allow_unicode=True)
    )
    return path


def set_meta(
    folder: Path, name: str, description: str, tags: list[str], described_at: str
) -> None:
    """Update one file's authored ``description``/``tags`` (and stamp
    ``described_at``) in place, preserving every other entry and field. Used by
    ``quack generate``; the next ``quack reindex`` re-derives the rest."""
    path = index_path(folder)
    files = dict(_read(path))
    entry = files.get(name)
    entry = dict(entry) if isinstance(entry, dict) else {}
    entry["description"] = (description or "").strip() or NO_DESC
    entry["tags"] = _norm_tags(tags) if tags else entry.get("tags", [])
    entry["described_at"] = described_at
    entry.setdefault("links", [])
    entry.setdefault("file_modified", "")
    files[name] = entry
    path.write_text(
        HEADER + yaml.safe_dump({FILES_KEY: files}, sort_keys=False, allow_unicode=True)
    )
