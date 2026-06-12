"""Tiered local search over all files, the advantage `ls` can't give you.

Two tiers, no model call and no embeddings:

  1. Structural: score every file on where the query terms hit, name, tags,
     description, then body, with descending weight. This answers "what is
     each file about?".
  2. Graph expansion: pull in the wikilink neighbours of the top hits, so a
     match surfaces what it is related to even if those neighbours did not
     match the query themselves. This answers "what connects to what?".

A semantic tier could slot in later behind the same interface (rank by an
embedding command), but the LLM reading these results already matches on
meaning, so structure + graph is the high-value, zero-dependency core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .core import Entry, Space

# Field weights: a hit in the name matters more than one buried in the body.
WEIGHT_NAME = 10
WEIGHT_TAG = 6
WEIGHT_DESCRIPTION = 4
WEIGHT_BODY = 1


# Reciprocal-rank-fusion constant. Each tier contributes 1/(RRF_K + rank);
# k flattens the contribution of low ranks so no single tier dominates.
RRF_K = 60


@dataclass
class Hit:
    entry: Entry
    score: float
    reasons: list[str] = field(default_factory=list)
    via: list[str] = field(default_factory=list)  # neighbour-of which file names
    tiers: list[str] = field(default_factory=list)  # which tiers matched

    @property
    def is_neighbour(self) -> bool:
        return bool(self.via) and not self.reasons and not self.tiers


def _terms(query: str) -> list[str]:
    return [t for t in re.split(r"\s+", query.lower().strip()) if t]


def _count(haystack: str, term: str) -> int:
    return haystack.lower().count(term)


def _score_entry(entry: Entry, terms: list[str]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    body = entry.body.lower()
    tags_joined = " ".join(entry.tags).lower()
    desc = entry.description.lower()
    name = entry.name.lower()

    for term in terms:
        if term in name:
            score += WEIGHT_NAME
            reasons.append(f"name~{term}")
        if term in tags_joined:
            score += WEIGHT_TAG
            reasons.append(f"tag~{term}")
        if term in desc:
            score += WEIGHT_DESCRIPTION
            reasons.append(f"desc~{term}")
        n = _count(body, term)
        if n:
            score += WEIGHT_BODY * n
            reasons.append(f"body~{term}x{n}")
    return score, reasons


def _structural_ranking(space: Space, terms: list[str]) -> list[tuple[str, list[str]]]:
    """Files ranked by weighted field match. Returns [(name, reasons), ...]."""
    scored = []
    for entry in space.entries:
        score, reasons = _score_entry(entry, terms)
        if score > 0:
            scored.append((score, entry.name, reasons))
    scored.sort(key=lambda t: -t[0])
    return [(name, reasons) for _, name, reasons in scored]


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
    """
    space = Space.load(explicit_root)
    terms = _terms(query)
    if not terms:
        return []

    # Each tier yields an ordered list of note names; fuse by reciprocal rank.
    fused: dict[str, float] = {}
    reasons_by_name: dict[str, list[str]] = {}
    tiers_by_name: dict[str, list[str]] = {}

    def add_tier(tier: str, ranked_names: list[str]) -> None:
        for rank, name in enumerate(ranked_names):
            fused[name] = fused.get(name, 0.0) + 1.0 / (RRF_K + rank)
            tiers_by_name.setdefault(name, [])
            if tier not in tiers_by_name[name]:
                tiers_by_name[name].append(tier)

    # Tier 1: structural (always available).
    structural = _structural_ranking(space, terms)
    add_tier("structural", [name for name, _ in structural])
    for name, reasons in structural:
        reasons_by_name[name] = reasons

    # Tier 2: FTS via DuckDB (skip silently if catalog missing).
    try:
        from . import catalog

        fts = catalog.fts_search(query, explicit_root, limit=max(limit * 2, 20))
        add_tier("fts", [_rel_to_name(space, rel) for rel, _desc, _score in fts])
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
        entry = space.by_name.get(name)
        if entry is None:
            continue
        hits[name] = Hit(
            entry=entry,
            score=score,
            reasons=reasons_by_name.get(name, []),
            tiers=tiers_by_name.get(name, []),
        )

    # Graph expansion: neighbours of the matches, scored below any direct hit.
    if expand and hits:
        from . import catalog

        seeds = list(hits)
        try:
            neigh = catalog.neighbours(seeds, explicit_root, hops=1)
        except Exception:
            neigh = []
        for name, rel, dist, via_seed in neigh:
            if name in hits:
                continue
            target = space.by_name.get(name)
            if target is None:
                continue
            hits[name] = Hit(entry=target, score=0.0, reasons=[], via=[via_seed])

    ranked = sorted(hits.values(), key=lambda h: (-h.score, h.entry.rel))
    return ranked[:limit]


def _rel_to_name(space: Space, rel: str) -> str:
    for e in space.entries:
        if e.rel == rel:
            return e.name
    return rel


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
