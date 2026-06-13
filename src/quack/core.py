"""Core model: discover files, read bodies, extract links.

quack sits on top of a directory of your work — any files. Authored metadata
(description/tags) lives in each folder's editable `.index.yaml`, overlaid onto
each file here by `Space.load`; `links` come from `[[wikilinks]]` in the body.
The map and DuckDB catalog the indexer emits are *derived*, so they can never
drift from the files + the store.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

import frontmatter

# Marker directory that identifies a quack root (like .git for a repo).
MARKER_DIR = ".quack"

# Always-ignored dirs (quack state + common noise). Users extend this with a
# `.quackignore` file at the root; these built-ins are never indexable.
DEFAULT_IGNORED_DIRS = {".quack", ".obsidian", ".git", ".trash", "node_modules"}

# quack's own generated/meta files that live alongside content but are not
# content themselves, so they are never indexed as files.
GENERATED_FILES = {
    "_diagrams.md",  # per-folder Mermaid graph
    "QUACK.md",      # the navigation anchor
    ".quackignore",  # root ignore config
    ".mcp.json",     # project-local MCP config
    ".index.yaml",   # the editable metadata store
    ".folder.md",    # folder-description marker
    ".DS_Store",
}

# Files larger than this are catalogued by path/metadata but their body is not
# read into the full-text index (keeps reindex cheap on big assets/logs).
TEXT_BODY_MAX_BYTES = 1_000_000

IGNORE_FILE = ".quackignore"


def load_ignores(root: Path) -> set[str]:
    """Built-in ignores plus any names/globs in the root's `.quackignore`.

    One entry per line; blank lines and `#` comments are skipped. Entries are
    matched against each directory/file *name* (and full relative path) during
    the walk, so both `node_modules` and `drafts/scratch` work.
    """
    ignores = set(DEFAULT_IGNORED_DIRS)
    f = root / IGNORE_FILE
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignores.add(line.rstrip("/"))
    return ignores

# [[wikilink]] or [[wikilink|alias]] or [[note#heading]]. We keep only the
# note target (strip alias and heading).
WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]")


def find_root(explicit: str | None = None) -> Path:
    """Resolve the quack root. The root can be named anything; quack locates it
    the way git locates a repo, by walking up for the `.quack/` marker.

    Order: explicit arg > walk up from cwd for `.quack/` > $QUACK_ROOT >
    $OBSIDIAN_VAULT (convenience for Obsidian users). If none resolves to a
    directory with `.quack/`, the caller is outside a quack space.
    """
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / MARKER_DIR).is_dir():
            return root
        raise RuntimeError(
            f"No quack space at {root}: missing {MARKER_DIR}/. "
            "Run `quack init` there first, or pass --root to an initialized space."
        )
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / MARKER_DIR).is_dir():
            return candidate
    for env_var in ("QUACK_ROOT", "OBSIDIAN_VAULT"):
        env = os.environ.get(env_var)
        if env:
            root = Path(env).expanduser().resolve()
            if (root / MARKER_DIR).is_dir():
                return root
            raise RuntimeError(
                f"{env_var} points to {root}, but it is missing {MARKER_DIR}/."
            )
    raise RuntimeError(
        "No quack space found. Run `quack init` in this directory, "
        "or pass --root to a directory containing .quack/."
    )


def _read_text(path: Path) -> tuple[str, bool]:
    """Return (body, is_binary). Files containing NUL bytes or that fail UTF-8
    decoding are treated as binary and yield an empty body. Oversize files are
    truncated to the first TEXT_BODY_MAX_BYTES for the index."""
    try:
        data = path.read_bytes()
    except OSError:
        return "", False
    if b"\x00" in data[:1024]:
        return "", True
    try:
        return data[:TEXT_BODY_MAX_BYTES].decode("utf-8"), False
    except UnicodeDecodeError:
        return "", True


@dataclass
class Entry:
    """A single file and its metadata. ``description``/``tags`` are *effective*:
    an authored value from ``<folder>/.index.yaml`` overrides the optional
    Markdown frontmatter seed; ``links`` is derived from ``[[wikilinks]]`` in the
    body. Authored values are attached by ``Space.load`` (see ``_overlay``)."""

    path: Path
    root: Path
    body: str = ""
    is_binary: bool = False
    _fm: "frontmatter.Post | None" = None
    # Authored overlay from the per-folder .index.yaml (set by Space.load).
    _authored_desc: str | None = None
    _authored_tags: list[str] | None = None
    _authored_described_at: str | None = None

    @property
    def rel(self) -> str:
        """Root-relative POSIX path, e.g. 'projects/usb-defect.md'."""
        return self.path.relative_to(self.root).as_posix()

    @property
    def name(self) -> str:
        """Filename without extension, the wikilink target."""
        return self.path.stem

    @property
    def ext(self) -> str:
        """Lowercase extension without the dot, '' if none (e.g. 'py', 'md')."""
        return self.path.suffix.lower().lstrip(".")

    @property
    def folder(self) -> str:
        """Root-relative folder, '' for the root itself."""
        rel = self.path.parent.relative_to(self.root).as_posix()
        return "" if rel == "." else rel

    @property
    def description(self) -> str:
        if self._authored_desc:
            return self._authored_desc
        if self._fm is not None:
            return str(self._fm.get("description", "")).strip()
        return ""

    @property
    def tags(self) -> list[str]:
        if self._authored_tags is not None:
            return self._authored_tags
        if self._fm is not None:
            raw = self._fm.get("tags", [])
            if isinstance(raw, str):
                return [t.strip() for t in raw.split(",") if t.strip()]
            return list(raw or [])
        return []

    @cached_property
    def modified(self) -> str:
        """The file's mtime as an ISO-8601 second-precision string ('' if the
        file is gone). Refreshed every reindex."""
        try:
            ts = self.path.stat().st_mtime
        except OSError:
            return ""
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    @property
    def described_at(self) -> str:
        """When the current description was written. For a value authored in the
        store, the stored timestamp; for a Markdown frontmatter description it
        lives in the file, so it tracks ``modified`` (never stale on its own)."""
        if self._authored_desc:
            return self._authored_described_at or self.modified
        if self._fm is not None and str(self._fm.get("description", "")).strip():
            return self.modified
        return ""

    @property
    def stale(self) -> bool:
        """True when the file changed after its description was written, so the
        description may no longer reflect the file."""
        return bool(
            self.description and self.described_at and self.modified > self.described_at
        )

    @cached_property
    def links(self) -> list[str]:
        """Distinct wikilink targets in body order (any text file may have them)."""
        seen: dict[str, None] = {}
        for m in WIKILINK_RE.finditer(self.body):
            seen.setdefault(m.group(1).strip(), None)
        return list(seen)


def load_entry(path: Path, root: Path) -> Entry:
    """Load one file. Markdown is parsed for an optional frontmatter seed; any
    other file is read as text (empty body if binary/undecodable)."""
    if path.suffix.lower() == ".md":
        post = frontmatter.load(path)
        return Entry(path=path, root=root, body=post.content, _fm=post)
    body, is_binary = _read_text(path)
    return Entry(path=path, root=root, body=body, is_binary=is_binary)


def _ignored(name: str, rel: str, patterns: set[str]) -> bool:
    """True if a name or its root-relative path matches any ignore pattern."""
    from fnmatch import fnmatch

    for pat in patterns:
        if name == pat or rel == pat or fnmatch(name, pat) or fnmatch(rel, pat):
            return True
    return False


def iter_files(root: Path, ignores: set[str] | None = None):
    """Yield every file in the root, skipping ignored dirs/files and quack's own
    generated artifacts. Not limited to Markdown — this is a meta layer over all
    files."""
    patterns = ignores if ignores is not None else load_ignores(root)
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if not _ignored(d, (base / d).relative_to(root).as_posix(), patterns)
        ]
        for fn in filenames:
            rel = (base / fn).relative_to(root).as_posix()
            if fn in GENERATED_FILES or _ignored(fn, rel, patterns):
                continue
            yield load_entry(base / fn, root)


def content_folders(root: Path, ignores: set[str] | None = None) -> list[Path]:
    """Top-level content folders (excludes ignored + meta dirs)."""
    patterns = ignores if ignores is not None else load_ignores(root)
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not _ignored(p.name, p.name, patterns)
    )


@dataclass
class Space:
    """Loaded view of the whole space: every file, with effective metadata.
    Build once, query many times."""

    root: Path
    entries: list[Entry] = field(default_factory=list)

    @classmethod
    def load(cls, explicit: str | None = None) -> "Space":
        root = find_root(explicit)
        entries = list(iter_files(root))
        _overlay(root, entries)
        return cls(root=root, entries=entries)

    @cached_property
    def by_name(self) -> dict[str, Entry]:
        """Map filename-stem → entry. On stem collisions the last wins; the
        wikilink graph is best-effort across arbitrary file trees."""
        return {e.name: e for e in self.entries}

    def resolve_link(self, target: str) -> Entry | None:
        """Resolve a wikilink target to a file (by filename stem)."""
        return self.by_name.get(target)


def _overlay(root: Path, entries: list[Entry]) -> None:
    """Attach authored description/tags from each folder's .index.yaml onto its
    entries, so ``description``/``tags`` are effective everywhere downstream."""
    from collections import defaultdict

    from . import index_store

    by_folder: dict[Path, list[Entry]] = defaultdict(list)
    for e in entries:
        by_folder[e.path.parent].append(e)
    for folder, es in by_folder.items():
        authored = index_store.load_authored(folder)
        for e in es:
            meta = authored.get(e.path.name)
            if not meta:
                continue
            if meta.get("description"):
                e._authored_desc = meta["description"]
            if meta.get("tags"):
                e._authored_tags = list(meta["tags"])
            if meta.get("described_at"):
                e._authored_described_at = meta["described_at"]
