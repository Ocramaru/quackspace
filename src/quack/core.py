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
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Callable

import frontmatter
from fnmatch import fnmatch

# Marker directory that identifies a quack root (like .git for a repo).
MARKER_DIR = ".quack"

# Always-ignored dirs: quack state plus pure noise/derived caches. These are
# never walked and never appear in the meta layer. Users extend this with a
# `.quackignore` file at the root.
DEFAULT_IGNORED_DIRS = {
    ".quack", ".obsidian", ".git", ".trash",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
    ".idea", ".ipynb_checkpoints",
}

# Opaque dirs: heavy, unambiguous vendored/dependency/virtualenv trees. We
# acknowledge the folder in the meta layer (so an LLM knows it exists) but never
# descend into it — its files are not indexed, keeping the catalog focused on
# the user's own work. Only names that are essentially never real content live
# here; ambiguous build outputs (build/dist/target) are left to `.quackignore`
# so quack never silently skips a user's own folder. A `.quackignore` match
# still wins and hides a dir entirely.
DEFAULT_OPAQUE_DIRS = {
    "site-packages", "node_modules", "bower_components",
    ".venv", "venv", "virtualenv", ".tox", ".eggs",
}

@dataclass(frozen=True)
class DatasetPolicy:
    """Thresholds that mark a folder as a dataset (recorded but not indexed).
    A value of 0 disables that trigger; the default disables both."""

    total: int = 0
    per_ext: int = 0
    extensions: frozenset[str] = frozenset()

    @property
    def active(self) -> bool:
        return self.total > 0 or self.per_ext > 0


def _dataset_reason(
    n_files: int, ext_counts: "Counter[str] | None", policy: DatasetPolicy
) -> str:
    if policy.total and n_files > policy.total:
        return f"{n_files} files"
    if ext_counts is not None and policy.per_ext:
        # Most-common first so the reason names the dominant data type.
        for ext, c in ext_counts.most_common():
            if ext in policy.extensions and c > policy.per_ext:
                return f"{c} .{ext} files"
    return ""


def _filter_dir(
    base: Path,
    filenames: list[str],
    root: Path,
    patterns: set[str],
    policy: DatasetPolicy,
) -> tuple[list[str], str]:
    kept: list[str] = []
    ext_counts: "Counter[str] | None" = Counter() if policy.per_ext else None
    for fn in filenames:
        rel = (base / fn).relative_to(root).as_posix()
        if not _keep_file(fn, rel, patterns):
            continue
        kept.append(fn)
        if ext_counts is not None:
            ext_counts[Path(fn).suffix.lower().lstrip(".")] += 1
    return kept, _dataset_reason(len(kept), ext_counts, policy)


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


class IgnoreRuleset:
    """Ordered ignore rules with gitignore-style semantics.

    Rules are evaluated in file order; the last matching rule wins. Negation
    rules (``!pattern``) flip a prior match to re-include the file — this lets
    users make exceptions inside an otherwise-ignored subtree.

    Pattern matching follows gitignore conventions:
      - ``pattern``          — match the *basename* of any file or directory
      - ``dir/file``         — match the *full path* relative to the workspace root
      - ``/pattern``         — root-anchored: only match at the top level
      - ``!pattern``         — negation: re-include something a prior rule excluded
      - Globs (``*``, ``?``, ``[…]``) are supported in all positions via fnmatch.

    Built-in defaults (hidden dirs, caches) are held in a separate set and are
    checked before user rules so negation cannot override them.
    """

    __slots__ = ("_builtins", "_rules")

    def __init__(
        self,
        builtins: "set[str]",
        rules: "list[tuple[str, bool, bool, bool]]",
    ) -> None:
        # builtins: names checked as-is (no glob, no negation, not overridable)
        self._builtins = builtins
        # rules: (pattern, negation, anchored, path_like)
        self._rules = rules

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, builtins: "set[str]", lines: "list[str]") -> "IgnoreRuleset":
        """Parse .quackignore lines into an ordered ruleset."""
        rules: list[tuple[str, bool, bool, bool]] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negation = line.startswith("!")
            if negation:
                line = line[1:].strip()
            line = line.rstrip("/")  # trailing slash = dir-only hint; not enforced
            anchored = line.startswith("/")
            if anchored:
                line = line[1:]
            if not line:
                continue
            path_like = "/" in line
            rules.append((line, negation, anchored, path_like))
        return cls(builtins, rules)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def is_ignored(self, name: str, rel: str) -> bool:
        """True if this file/directory should be skipped by the walker."""
        if name in self._builtins:
            return True

        result = False
        for pat, negation, anchored, path_like in self._rules:
            if anchored:
                # /pattern: rel must equal pat or start with pat/
                hit = rel == pat or rel.startswith(pat + "/") or fnmatch(rel, pat)
            elif path_like:
                # contains /: match against the full relative path
                hit = rel == pat or fnmatch(rel, pat)
            else:
                # plain name or glob: match basename anywhere (and full path)
                hit = name == pat or fnmatch(name, pat) or fnmatch(rel, pat)

            if hit:
                result = not negation  # negation flips: True→False (re-include)

        return result


def load_ignores(root: Path) -> IgnoreRuleset:
    """Built-in ignores plus ordered rules from the root's ``.quackignore``.

    Supports the full gitignore pattern vocabulary: plain names, path patterns,
    root-anchored ``/pattern``, globs, and negation ``!pattern`` for exceptions.
    Built-in noise dirs (caches, hidden quack dirs) cannot be negated.
    """
    lines: list[str] = []
    f = root / IGNORE_FILE
    if f.exists():
        lines = f.read_text().splitlines()
    return IgnoreRuleset.build(DEFAULT_IGNORED_DIRS, lines)

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


def _mtime_iso(path: Path) -> str:
    """A file's mtime as an ISO-8601 second-precision string ('' if missing)."""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _read_text(path: Path, max_bytes: int = TEXT_BODY_MAX_BYTES) -> tuple[str, bool]:
    """Return (body, is_binary). NUL bytes → binary (empty body). Reads at most
    *max_bytes* so large files don't stall the scan. Non-UTF-8 text is decoded
    leniently rather than dropped."""
    try:
        with path.open("rb") as f:
            chunk = f.read(max_bytes)
    except OSError:
        return "", False
    if b"\x00" in chunk[:1024]:
        return "", True
    try:
        return chunk.decode("utf-8"), False
    except UnicodeDecodeError:
        return chunk.decode("utf-8", errors="replace"), False


@dataclass
class Entry:
    """A single file and its metadata. ``description``/``tags`` are *effective*,
    resolved per field by precedence: authored ``<folder>/.index.yaml`` →
    Markdown frontmatter → recognition default (see ``recognize``) → empty. As a
    special case, frontmatter that supplies a *description* suppresses
    recognition *tags*, so a hand-written note isn't tagged with generic
    boilerplate. ``links`` is derived from ``[[wikilinks]]`` in the body.
    Authored values are attached by ``Space.load`` (see ``_overlay``)."""

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

    @cached_property
    def _recognition(self) -> "tuple[str, list[str]] | None":
        """Zero-cost default ``(description, tags)`` for a well-known file
        (e.g. ``.gitignore``, ``*.py``), or ``None``. The lowest precedence
        layer behind authored metadata and frontmatter (see ``recognize``)."""
        from . import recognize

        return recognize.recognize_file(self.path)

    @property
    def description(self) -> str:
        if self._authored_desc:
            return self._authored_desc
        if self._fm is not None:
            fm = str(self._fm.get("description", "")).strip()
            if fm:
                return fm
        if self._recognition is not None:
            return self._recognition[0]
        return ""

    @property
    def tags(self) -> list[str]:
        if self._authored_tags is not None:
            return self._authored_tags
        if self._fm is not None:
            raw = self._fm.get("tags", [])
            if isinstance(raw, str):
                parsed = [t.strip() for t in raw.split(",") if t.strip()]
            else:
                parsed = list(raw or [])
            if parsed:
                return parsed
            # Frontmatter with a description speaks for the file; don't inject
            # generic recognition tags on top of an authored-in-file note.
            if str(self._fm.get("description", "")).strip():
                return []
        if self._recognition is not None:
            return list(self._recognition[1])
        return []

    @cached_property
    def modified(self) -> str:
        """The file's mtime as an ISO-8601 second-precision string ('' if the
        file is gone). Refreshed every reindex."""
        return _mtime_iso(self.path)

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


def parse_frontmatter(text: str) -> "frontmatter.Post":
    """Parse frontmatter from already-decoded text, never raising. Malformed
    YAML frontmatter falls back to treating the whole text as the body (no
    metadata), so one bad file can't break a reindex."""
    try:
        return frontmatter.loads(text)
    except Exception:
        return frontmatter.Post(content=text)


def load_entry(path: Path, root: Path, body_max_bytes: int = TEXT_BODY_MAX_BYTES) -> Entry:
    """Load one file. Markdown is parsed for frontmatter; any other file is read
    as text. Non-UTF-8 and binary files degrade gracefully."""
    body, is_binary = _read_text(path, body_max_bytes)
    if not is_binary and path.suffix.lower() == ".md":
        post = parse_frontmatter(body)
        return Entry(path=path, root=root, body=post.content, _fm=post)
    return Entry(path=path, root=root, body=body, is_binary=is_binary)


def _level(
    base: Path,
    dirnames: list[str],
    root: Path,
    patterns: IgnoreRuleset,
    opaque_dirs: "frozenset[str] | None" = None,
):
    """One os.walk level → (folders_to_record, dirs_to_descend). Shared by
    :func:`walk` and :func:`count_indexable` so pruning can't diverge: ignored
    dirs are hidden; opaque dirs are recorded as folders but not descended."""
    if opaque_dirs is None:
        opaque_dirs = DEFAULT_OPAQUE_DIRS
    record: list[Path] = []
    descend: list[str] = []
    for d in sorted(dirnames):
        rel = (base / d).relative_to(root).as_posix()
        if patterns.is_ignored(d, rel):
            continue
        record.append(base / d)
        if d not in opaque_dirs:
            descend.append(d)
    return record, descend


def _keep_file(fn: str, rel: str, patterns: IgnoreRuleset) -> bool:
    """True if a file should be indexed (not a generated artifact / not ignored)."""
    return not (fn in GENERATED_FILES or patterns.is_ignored(fn, rel))


# Below this many files, the thread-pool overhead isn't worth it; load inline.
_PARALLEL_MIN_FILES = 64


def _load_entries(
    file_paths: list[Path],
    root: Path,
    on_file: "Callable[[Entry], None] | None",
    body_max_bytes: int = TEXT_BODY_MAX_BYTES,
) -> list[Entry]:
    if len(file_paths) < _PARALLEL_MIN_FILES:
        entries: list[Entry] = []
        for p in file_paths:
            e = load_entry(p, root, body_max_bytes)
            entries.append(e)
            if on_file is not None:
                on_file(e)
        return entries

    from concurrent.futures import ThreadPoolExecutor

    entries = []
    with ThreadPoolExecutor() as pool:
        for e in pool.map(lambda p: load_entry(p, root, body_max_bytes), file_paths):
            entries.append(e)
            if on_file is not None:
                on_file(e)
    return entries


def walk(
    root: Path,
    ignores: "IgnoreRuleset | None" = None,
    on_file: "Callable[[Entry], None] | None" = None,
    dataset_policy: DatasetPolicy | None = None,
    datasets_out: dict[str, str] | None = None,
    body_max_bytes: int = TEXT_BODY_MAX_BYTES,
    opaque_dirs: "frozenset[str] | None" = None,
) -> tuple[list[Entry], list[Path]]:
    """One filesystem pass → (entries, folders).

    ``entries`` is every non-ignored file (loaded), skipping quack's own
    generated artifacts. ``folders`` is every non-ignored folder under *root*
    (excluding root itself), including folders that contain only subfolders, so
    the per-folder meta layer can cover the whole tree. Directory traversal is a
    single ``os.walk``; the (slower, I/O-bound) file loading is parallelized.
    *on_file*, if given, is called with each loaded entry (drives progress).

    *dataset_policy*, when active, makes a folder whose files look like a dataset
    (see :class:`DatasetPolicy`) *recorded but not indexed*: it stays in
    ``folders`` so the meta layer knows it exists, but its files are skipped (not
    loaded, not returned in ``entries``). If *datasets_out* is given, each such
    folder's rel maps to a short reason string for the meta layer to describe."""
    patterns = ignores if ignores is not None else load_ignores(root)
    policy = dataset_policy or DatasetPolicy()
    file_paths: list[Path] = []
    folders: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        record, descend = _level(base, dirnames, root, patterns, opaque_dirs)
        folders.extend(record)
        dirnames[:] = descend
        kept, reason = _filter_dir(base, filenames, root, patterns, policy)
        if reason:
            if datasets_out is not None:
                rel = base.relative_to(root).as_posix()
                datasets_out["" if rel == "." else rel] = reason
            continue
        file_paths.extend(base / fn for fn in kept)
    entries = _load_entries(file_paths, root, on_file, body_max_bytes)
    return entries, folders


def count_indexable(
    root: Path,
    ignores: "IgnoreRuleset | None" = None,
    dataset_policy: DatasetPolicy | None = None,
    progress: "Callable[[int, int | None, str], None] | None" = None,
    opaque_dirs: "frozenset[str] | None" = None,
) -> int:
    """Count the files :func:`walk` would index, without reading any of them.
    Cheap (stat/scandir only); used to get a total up front for later progress.
    Applies the same *dataset_policy* skip as :func:`walk` so the total matches
    what is actually indexed. When *progress* is supplied, reports an unbounded
    counter because the total is exactly what this pass is discovering."""
    patterns = ignores if ignores is not None else load_ignores(root)
    policy = dataset_policy or DatasetPolicy()
    n = 0
    if progress is not None:
        progress(0, None, "Waddling through files: 0")
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        _record, descend = _level(base, dirnames, root, patterns, opaque_dirs)
        dirnames[:] = descend
        kept, reason = _filter_dir(base, filenames, root, patterns, policy)
        if reason:
            continue
        n += len(kept)
        if progress is not None and n % 100 == 0:
            progress(n, None, f"Waddling through files: {n:,}")
    if progress is not None:
        progress(n, None, f"Waddled {n:,} file(s)")
    return n


# Per-folder metadata files quack writes/reads; an edit to one of these can
# change effective metadata without changing any indexed file's mtime.
_META_MARKERS = (".index.yaml", ".folder.md")


def scan_signature(
    root: Path,
    ignores: "IgnoreRuleset | None" = None,
    dataset_policy: DatasetPolicy | None = None,
    progress: "Callable[[int, int, str], None] | None" = None,
    opaque_dirs: "frozenset[str] | None" = None,
) -> tuple[dict[str, str], set[str], int]:
    """Stat-only change signature — no file reads, no parsing.

    Returns ``(files, folders, newest_marker_ns)`` where ``files`` maps each
    indexable file's rel → its mtime string (same format as ``Entry.modified``),
    ``folders`` is the set of folder rels, and ``newest_marker_ns`` is the latest
    ``st_mtime_ns`` among ``.index.yaml``/``.folder.md`` (0 if none) — nanosecond
    precision so an authored edit made in the same second as a build is still
    seen. Used for a cheap reindex no-op check before paying for a full
    :func:`walk`. Applies the same *dataset_policy* skip as :func:`walk`, so the
    signature reflects exactly the files the catalog holds (else a dataset folder
    would make every reindex look dirty)."""
    patterns = ignores if ignores is not None else load_ignores(root)
    policy = dataset_policy or DatasetPolicy()
    total = count_indexable(root, patterns, policy, progress=progress, opaque_dirs=opaque_dirs) if progress is not None else 0
    seen = 0
    if progress is not None:
        progress(0, max(total, 1), "Checking files")
    files: dict[str, str] = {}
    folders: set[str] = set()
    newest_marker_ns = 0
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        record, descend = _level(base, dirnames, root, patterns, opaque_dirs)
        for d in record:
            folders.add(d.relative_to(root).as_posix())
        dirnames[:] = descend
        kept: list[tuple[str, Path]] = []
        ext_counts: "Counter[str] | None" = Counter() if policy.per_ext else None
        for fn in filenames:
            full = base / fn
            if fn in _META_MARKERS:
                try:
                    ns = full.stat().st_mtime_ns
                except OSError:
                    ns = 0
                if ns > newest_marker_ns:
                    newest_marker_ns = ns
                continue
            rel = full.relative_to(root).as_posix()
            if _keep_file(fn, rel, patterns):
                kept.append((rel, full))
                if ext_counts is not None:
                    ext_counts[Path(fn).suffix.lower().lstrip(".")] += 1
        if _dataset_reason(len(kept), ext_counts, policy):
            continue
        for rel, full in kept:
            seen += 1
            files[rel] = _mtime_iso(full)
            if progress is not None and (seen == total or seen % 100 == 0):
                progress(seen, max(total, 1), f"Checking {rel}")
    if progress is not None and seen == 0:
        progress(1, 1, "Checked files")
    return files, folders, newest_marker_ns


def iter_files(root: Path, ignores: "IgnoreRuleset | None" = None):
    """Yield every non-ignored file in the root. Thin wrapper over :func:`walk`
    for callers that only need files."""
    yield from walk(root, ignores)[0]


def iter_content_folders(root: Path, ignores: set[str] | None = None):
    """Yield every non-ignored folder under *root* (excluding root). Thin
    wrapper over :func:`walk` for callers that only need folders."""
    yield from walk(root, ignores)[1]


@dataclass
class Space:
    """Loaded view of the whole space: every file, with effective metadata.
    Build once, query many times."""

    root: Path
    entries: list[Entry] = field(default_factory=list)
    # Same walk as ``entries``; consumers never re-traverse the tree.
    folders: list[Path] = field(default_factory=list)
    # rel → reason for folders skipped as datasets; files not in ``entries``.
    datasets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        explicit: str | None = None,
        progress: "Callable[[int, int, str], None] | None" = None,
    ) -> "Space":
        """Load every file with effective metadata. *progress*, if given, is
        called as ``progress(done, total, message)`` per file during the scan —
        it triggers a cheap up-front count so a real total is known. Callers
        that don't need progress (search, embed, doctor) pay no extra walk."""
        from .config import Config, DEFAULT_DATASET_EXTENSIONS  # local import avoids a config<->core cycle

        root = find_root(explicit)
        index_cfg = Config.load(str(root)).index
        policy = DatasetPolicy(
            total=index_cfg.dataset_threshold,
            per_ext=index_cfg.dataset_ext_threshold,
            extensions=DEFAULT_DATASET_EXTENSIONS | frozenset(index_cfg.dataset_extensions),
        )
        body_max_bytes = index_cfg.body_max_bytes
        opaque_dirs = DEFAULT_OPAQUE_DIRS | frozenset(index_cfg.opaque_dirs)
        on_file = None
        if progress is not None:
            total = count_indexable(root, dataset_policy=policy, opaque_dirs=opaque_dirs, progress=progress)
            seen = {"n": 0}

            def on_file(entry: Entry) -> None:
                seen["n"] += 1
                progress(seen["n"], total, f"Scanning {entry.rel}")

        if progress is not None:
            progress(0, 1, "Loading file contents")
        datasets: dict[str, str] = {}
        entries, folders = walk(
            root, on_file=on_file, dataset_policy=policy, datasets_out=datasets,
            body_max_bytes=body_max_bytes, opaque_dirs=opaque_dirs,
        )
        if progress is not None:
            progress(0, 1, "Applying metadata")
        _overlay(root, entries)
        if progress is not None:
            progress(1, 1, "Applied metadata")
        return cls(root=root, entries=entries, folders=folders, datasets=datasets)

    @cached_property
    def by_name(self) -> dict[str, Entry]:
        """Map filename-stem → entry. On stem collisions the last wins; the
        wikilink graph is best-effort across arbitrary file trees."""
        return {e.name: e for e in self.entries}

    @cached_property
    def by_rel(self) -> dict[str, Entry]:
        """Map root-relative path → entry, for O(1) path lookups."""
        return {e.rel: e for e in self.entries}

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
