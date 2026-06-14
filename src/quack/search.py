"""Tiered local search over all files, the advantage `ls` can't give you.

Tiers, fused with reciprocal rank fusion:

  1. Structural: score files on where query terms hit the *short* fields —
     name, tags, description — read straight from the catalog. Body matching is
     deliberately NOT done here (it would mean pulling every file's full text
     into Python on each query); that's the FTS tier's job.
  2. FTS: DuckDB BM25 over name/description/body — the indexed way to match on
     body text.
  3. Semantic: DuckDB vss cosine, when embeddings exist.
  4. Graph expansion: pull in the wikilink neighbours of the top hits, so a
     match surfaces what it is related to even if those neighbours did not
     match the query themselves.

Everything reads the catalog snapshot, so search scales with relevance, not
space size, and never walks the filesystem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import catalog
from .core import Space, find_root
from .catalog import DB_NAME

# Field weights for the structural tier (short fields only; body is the FTS
# tier's domain). A hit in the name matters more than one in the description.
WEIGHT_NAME = 10
WEIGHT_TAG = 6
WEIGHT_DESCRIPTION = 4

# Reciprocal-rank-fusion constant. Each tier contributes 1/(RRF_K + rank);
# k flattens the contribution of low ranks so no single tier dominates.
RRF_K = 60


@dataclass
class Doc:
    """The lightweight file record search returns — sourced from the catalog,
    not a full filesystem load. Exposes the fields callers read off a hit."""

    rel: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Hit:
    entry: Doc
    score: float
    reasons: list[str] = field(default_factory=list)
    via: list[str] = field(default_factory=list)  # neighbour-of which file names
    tiers: list[str] = field(default_factory=list)  # which tiers matched

    @property
    def is_neighbour(self) -> bool:
        return bool(self.via) and not self.reasons and not self.tiers


@dataclass
class FolderHit:
    folder: str
    parent: str
    description: str
    score: float
    via: str  # "semantic" or "structural" — how the folder was found


# Query routing. A question may be asking *which folder/area* something lives
# in, or *which file* implements something. We route to the matching embedding
# space (or both) and keep the two result kinds distinct in the output.
_FOLDER_HINTS = (
    "folder", "directory", "directories", "where is", "where are",
    "which folder", "which directory", "module", "package",
    "subdir", "top-level", "where do", "where does", "located",
)
_FILE_HINTS = (
    "file", "function", "class", "def", "implement", "code for",
    "definition", "method", "which file", "the file that",
)


def _any_hint(query: str, hints: tuple[str, ...]) -> bool:
    # Word-boundary match so "areas" doesn't trip "area" and "packages" reads as
    # plural intent, not the singular hint.
    return any(re.search(rf"\b{re.escape(h)}\b", query) for h in hints)


def route(query: str) -> str:
    """Decide whether a query is about files, folders, or both."""
    q = query.lower()
    folders = _any_hint(q, _FOLDER_HINTS)
    files = _any_hint(q, _FILE_HINTS)
    if folders and not files:
        return "folders"
    if files and not folders:
        return "files"
    return "both"


def _terms(query: str) -> list[str]:
    return [t for t in re.split(r"\s+", query.lower().strip()) if t]


def _score_row(
    name: str, description: str, tags_csv: str, terms: list[str]
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    # Lowercase once; terms are already lowercase from _terms().
    name_l = name.lower()
    tags_l = (tags_csv or "").lower()
    desc_l = (description or "").lower()

    for term in terms:
        if term in name_l:
            score += WEIGHT_NAME
            reasons.append(f"name~{term}")
        if term in tags_l:
            score += WEIGHT_TAG
            reasons.append(f"tag~{term}")
        if term in desc_l:
            score += WEIGHT_DESCRIPTION
            reasons.append(f"desc~{term}")
    return score, reasons


def _open_corpus(explicit_root: str | None):
    """Return (rows, con) for a search.

    Opens the catalog **once** and reads the short structural fields; the same
    connection is reused for the FTS and graph tiers (no reopening per tier).
    Falls back to a filesystem load with no connection only when there is no
    usable catalog yet, so search still works before the first reindex."""
    db = find_root(explicit_root) / ".quack" / DB_NAME
    try:
        con = catalog.connect_path(db)
        return catalog.docs_on(con), con
    except Exception:
        space = Space.load(explicit_root)
        rows = [(e.name, e.rel, e.description, ",".join(e.tags)) for e in space.entries]
        return rows, None


def search(
    query: str,
    explicit_root: str | None = None,
    limit: int = 10,
    expand: bool = True,
) -> list[Hit]:
    """Auto-hybrid search: fuse every available tier, then expand on the graph.

    Tiers run automatically and only if usable, structural (always), FTS
    (DuckDB catalog), and semantic (DuckDB vss, only if embeddings exist). The
    LLM never chooses a mode; results are merged with reciprocal rank fusion so
    exact matches (structural/FTS) and conceptual matches (semantic) blend
    sensibly. Graph expansion then pulls in neighbours of the top hits.

    All tiers read the catalog snapshot — search no longer loads the filesystem
    on every call, so it scales with relevance, not space size.
    """
    terms = _terms(query)
    if not terms:
        return []

    rows, con = _open_corpus(explicit_root)
    try:
        doc_by_name: dict[str, Doc] = {}
        name_by_rel: dict[str, str] = {}
        for name, rel, desc, tags_csv in rows:
            doc_by_name[name] = Doc(
                rel=rel,
                name=name,
                description=desc or "",
                tags=[t for t in (tags_csv or "").split(",") if t],
            )
            name_by_rel[rel] = name

        fused: dict[str, float] = {}
        reasons_by_name: dict[str, list[str]] = {}
        tiers_by_name: dict[str, list[str]] = {}

        def add_tier(tier: str, ranked_names: list[str]) -> None:
            for rank, name in enumerate(ranked_names):
                fused[name] = fused.get(name, 0.0) + 1.0 / (RRF_K + rank)
                tiers_by_name.setdefault(name, [])
                if tier not in tiers_by_name[name]:
                    tiers_by_name[name].append(tier)

        # Tier 1: structural (always available), scored over the short catalog
        # fields. Body matches come from the FTS tier below.
        scored: list[tuple[float, str, list[str]]] = []
        for name, _rel, desc, tags_csv in rows:
            s, reasons = _score_row(name, desc, tags_csv, terms)
            if s > 0:
                scored.append((s, name, reasons))
        scored.sort(key=lambda t: -t[0])
        add_tier("structural", [name for _s, name, _r in scored])
        for _s, name, reasons in scored:
            reasons_by_name[name] = reasons

        # Tier 2: FTS — on the same connection (None only before first reindex).
        if con is not None:
            try:
                fts = catalog.fts_on(con, query, limit=max(limit * 2, 20))
                add_tier("fts", [name_by_rel.get(rel, rel) for rel, _d, _s in fts])
            except Exception:
                pass

        # Tier 3: semantic via vss (skip silently if not configured/built).
        try:
            from . import embed
            sem = embed.semantic_search(query, explicit_root, limit=max(limit * 2, 20))
            add_tier("semantic", [name for _rel, name, _dist in sem])
        except Exception:
            pass

        hits: dict[str, Hit] = {}
        for name, score in fused.items():
            doc = doc_by_name.get(name)
            if doc is None:
                continue
            hits[name] = Hit(
                entry=doc,
                score=score,
                reasons=reasons_by_name.get(name, []),
                tiers=tiers_by_name.get(name, []),
            )

        # Graph expansion: neighbours of the matches, on the same connection.
        if expand and hits and con is not None:
            seeds = list(hits)
            try:
                neigh = catalog.neighbours_on(con, seeds, hops=1)
            except Exception:
                neigh = []
            for name, rel, dist, via_seed in neigh:
                if name in hits:
                    continue
                doc = doc_by_name.get(name)
                if doc is None:
                    continue
                hits[name] = Hit(entry=doc, score=0.0, reasons=[], via=[via_seed])
    finally:
        if con is not None:
            con.close()

    ranked = sorted(hits.values(), key=lambda h: (-h.score, h.entry.rel))
    return ranked[:limit]


def search_folders(
    query: str, explicit_root: str | None = None, limit: int = 10
) -> list[FolderHit]:
    """Folder-level search, kept distinct from file hits. Prefers the folder
    embedding space; falls back to a structural scan over the ``folders`` table
    when embeddings are not built. Returns ranked ``FolderHit``s."""
    terms = _terms(query)
    if not terms:
        return []
    db = find_root(explicit_root) / ".quack" / DB_NAME

    # Folder descriptions, for enriching semantic hits and for the fallback.
    try:
        folder_rows = catalog.list_folders_path(db)
    except Exception:
        folder_rows = []
    desc_by_folder = {f: (d or "") for f, _p, d in folder_rows}
    parent_by_folder = {f: (p or "") for f, p, _d in folder_rows}

    hits: dict[str, FolderHit] = {}

    # Tier 1: folder embeddings (semantic), if present.
    try:
        from . import embed

        sem = embed.semantic_search_folders(query, explicit_root, limit=max(limit * 2, 20))
        for rank, (folder, parent, _dist) in enumerate(sem):
            hits[folder] = FolderHit(
                folder=folder,
                parent=parent or parent_by_folder.get(folder, ""),
                description=desc_by_folder.get(folder, ""),
                score=1.0 / (RRF_K + rank),
                via="semantic",
            )
    except Exception:
        pass

    # Tier 2: structural fallback over folder path + description.
    if not hits:
        scored: list[tuple[float, str]] = []
        for folder, _parent, desc in folder_rows:
            hay = f"{folder} {desc or ''}".lower()
            score = sum(hay.count(t) for t in terms)
            if score > 0:
                scored.append((score, folder))
        scored.sort(key=lambda t: -t[0])
        for score, folder in scored:
            hits[folder] = FolderHit(
                folder=folder,
                parent=parent_by_folder.get(folder, ""),
                description=desc_by_folder.get(folder, ""),
                score=float(score),
                via="structural",
            )

    ranked = sorted(hits.values(), key=lambda h: (-h.score, h.folder))
    return ranked[:limit]


def format_hits(hits: list[Hit], root: str | None = None) -> str:
    if not hits:
        return "No matches."
    lines: list[str] = []
    if root:
        lines.append(f"# root: {root}  (paths below are relative to it)")
    for h in hits:
        if h.is_neighbour:
            tag = "→ related"
        else:
            tag = "+".join(h.tiers) if h.tiers else f"score {h.score:.3f}"
        lines.append(f"{h.entry.rel}  [{tag}]")
        if h.entry.description:
            lines.append(f"    {h.entry.description}")
        detail = h.reasons[:] if h.reasons else []
        if h.via:
            detail.append("via " + ", ".join(sorted(set(h.via))))
        if detail:
            lines.append(f"    ({'; '.join(detail)})")
    return "\n".join(lines)


def format_folder_hits(hits: list[FolderHit]) -> str:
    if not hits:
        return "No folder matches."
    lines: list[str] = []
    for h in hits:
        lines.append(f"{h.folder}/  [folder · {h.via}]")
        if h.description:
            lines.append(f"    {h.description}")
    return "\n".join(lines)
