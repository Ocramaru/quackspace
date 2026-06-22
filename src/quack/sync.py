"""Cheap, shared pending-work detection for ``quack status`` and ``sync``."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from . import catalog
from .config import (
    Config,
    DEFAULT_DATASET_EXTENSIONS,
    DEFAULT_NONEMBEDDABLE_DIRS,
    DEFAULT_NONEMBEDDABLE_EXTENSIONS,
    DEFAULT_NONEMBEDDABLE_TAGS,
)
from .core import DEFAULT_OPAQUE_DIRS, DatasetPolicy, find_root, scan_signature


@dataclass(frozen=True)
class PendingWork:
    catalog_exists: bool
    new: set[str] = field(default_factory=set)
    modified: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    missing_embeddings: set[str] = field(default_factory=set)

    @property
    def index_count(self) -> int:
        return len(self.new) + len(self.modified) + len(self.deleted)

    @property
    def nothing_to_do(self) -> bool:
        return self.catalog_exists and not (
            self.new or self.modified or self.deleted or self.missing_embeddings
        )


def pending_work(explicit_root: str | None = None) -> PendingWork:
    """Disk/catalog drift plus embeddable files lacking vectors — cheaply.

    Index drift is a stat-only ``scan_signature`` walk (no file reads) compared
    to the catalog's stored mtimes. Embedding eligibility is derived from the
    catalog's own columns (``ext``/``description``/``tags_csv``/``is_binary``)
    via the shared :func:`catalog.embeddable` predicate, so no file *contents*
    are ever read — ``status`` stays a stat-walk plus a couple of SQL queries.
    """
    root = find_root(explicit_root)
    db = root / ".quack" / catalog.DB_NAME
    if not db.exists():
        return PendingWork(catalog_exists=False)

    config = Config.load(str(root))
    embeddings_enabled = config.embed.configured and not config.embed.skip
    index = config.index
    dataset_policy = DatasetPolicy(
        total=index.dataset_threshold,
        per_ext=index.dataset_ext_threshold,
        extensions=DEFAULT_DATASET_EXTENSIONS | frozenset(index.dataset_extensions),
    )
    opaque_dirs = DEFAULT_OPAQUE_DIRS | frozenset(index.opaque_dirs)
    files, _folders, _marker = scan_signature(
        root, dataset_policy=dataset_policy, opaque_dirs=opaque_dirs
    )

    try:
        con = catalog.connect_path(db)
        try:
            stored = {
                rel: (mtime or "")
                for rel, mtime in con.execute(
                    "SELECT rel, file_modified FROM files"
                ).fetchall()
            }
            embedded: set[str] = set()
            candidate_rows: list = []
            if embeddings_enabled:
                try:
                    embedded = {
                        rel for (rel,) in con.execute("SELECT rel FROM embeddings").fetchall()
                    }
                except Exception:
                    embedded = set()
                candidate_rows = con.execute(
                    "SELECT rel, ext, description, tags_csv, is_binary FROM files"
                ).fetchall()
        finally:
            con.close()
    except Exception:
        return PendingWork(catalog_exists=False)

    current = set(files)
    indexed = set(stored)
    new = current - indexed
    deleted = indexed - current
    modified = {rel for rel in current & indexed if files[rel] != stored[rel]}

    missing_embeddings: set[str] = set()
    if embeddings_enabled:
        embed = config.embed
        ne_extensions = DEFAULT_NONEMBEDDABLE_EXTENSIONS | frozenset(embed.nonembeddable_extensions)
        ne_tags = DEFAULT_NONEMBEDDABLE_TAGS | frozenset(embed.nonembeddable_tags)
        ne_dirs = DEFAULT_NONEMBEDDABLE_DIRS | frozenset(embed.nonembeddable_dirs)
        for rel, ext, description, tags_csv, is_binary in candidate_rows:
            # Only files still on disk and not already embedded can be "missing".
            if rel not in current or rel in embedded:
                continue
            entry = SimpleNamespace(
                rel=rel,
                ext=ext or "",
                description=description or "",
                tags=tags_csv.split(",") if tags_csv else [],
                is_binary=bool(is_binary),
            )
            if catalog.embeddable(entry, ne_extensions, ne_tags, ne_dirs):
                missing_embeddings.add(rel)

    return PendingWork(
        catalog_exists=True,
        new=new,
        modified=modified,
        deleted=deleted,
        missing_embeddings=missing_embeddings,
    )
