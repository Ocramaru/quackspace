"""MCP server: expose the quack knowledge layer to LLMs as typed tools.

Instead of shelling out and parsing text, an MCP-aware assistant (Claude Code,
Kiro, Q) calls these tools directly. The server's `instructions` tell the LLM
how to navigate, so it always knows how to use quack without prior context.

Run with:  uv run quack-mcp   (or `quack mcp` once installed)
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from . import catalog, graph as graph_mod
from .core import find_root
from .search import search as do_search

INSTRUCTIONS = """\
This is quack, the meta layer over a directory of the user's work — any files:
notes, docs, code, configs, assets. Use it to find and read files precisely,
without loading the whole tree into context.

How to navigate (cheapest first):
1. `map` — folder-level overview. Decides which folder is relevant.
2. `search` — auto-hybrid (keyword + full-text + semantic if available + graph
   neighbours). This is the default way to find files; you do not pick a mode.
3. `get_file` — read one file's full content by its path or name.
4. `sql` / `graph_path` / `central` / `clusters` — precise structural queries.

Recording what you know:
- `describe(path, description, tags)` writes metadata into the editable store for
  a file you understand. If you already know this repo, describe its relevant
  files (one call each), then call `reindex()` ONCE so search/sql reflect them.
  This is the intended way to seed quack on a codebase an agent already knows.
- File CONTENTS are read-only here; only metadata (descriptions, tags) is writable.

All paths are RELATIVE to the quack root, which every result reports as `root`.
To open a file, join root + path. Never assume an absolute path.
"""

mcp = FastMCP("quack", instructions=INSTRUCTIONS)


def _root() -> str:
    return str(find_root(None))


@mcp.tool()
def map() -> dict[str, Any]:
    """Folder-level overview of the root: each folder and its file count. Start
    here to decide which folder is relevant."""
    cols, rows = catalog.query(
        "SELECT folder, count(*) AS files FROM files "
        "WHERE folder <> '' GROUP BY folder ORDER BY files DESC"
    )
    return {
        "root": _root(),
        "folders": [dict(zip(cols, r)) for r in rows],
    }


@mcp.tool()
def search(query: str, limit: int = 10, expand: bool = True) -> dict[str, Any]:
    """Auto-hybrid search over all files: fuses keyword, full-text, and semantic
    (if embeddings exist) ranking, then adds graph neighbours. The primary way
    to find files. Returns ranked hits with paths relative to `root`."""
    hits = do_search(query, limit=limit, expand=expand)
    return {
        "root": _root(),
        "hits": [
            {
                "path": h.entry.rel,
                "name": h.entry.name,
                "description": h.entry.description,
                "tags": h.entry.tags,
                "tiers": h.tiers,
                "related_via": h.via,
            }
            for h in hits
        ],
    }


@mcp.tool()
def get_file(path_or_name: str) -> dict[str, Any]:
    """Read one file's full content + metadata. Accepts a root-relative path
    (src/app/main.py) or a bare file name without extension (main)."""
    from .core import Space

    space = Space.load(None)
    entry = space.by_name.get(path_or_name)
    if entry is None:
        entry = next((e for e in space.entries if e.rel == path_or_name), None)
    if entry is None:
        return {"root": _root(), "error": f"No file matching {path_or_name!r}."}
    return {
        "root": _root(),
        "path": entry.rel,
        "name": entry.name,
        "description": entry.description,
        "tags": entry.tags,
        "links": entry.links,
        "is_binary": entry.is_binary,
        "modified": entry.modified,
        "stale": entry.stale,
        "content": entry.body,
    }


@mcp.tool()
def sql(query: str) -> dict[str, Any]:
    """Run read-only SQL against the catalog. Tables: files(name, rel, folder,
    ext, description, tags_csv, n_links, n_inbound, is_orphan, is_binary,
    file_modified, described_at, stale, body), tags(name, tag),
    links(src, dst, dst_exists)."""
    cols, rows = catalog.query(query)
    return {"root": _root(), "columns": cols, "rows": [list(r) for r in rows]}


@mcp.tool()
def graph_path(src: str, dst: str) -> dict[str, Any]:
    """Shortest path of wikilinks between two files (by name). Returns the node
    names on the path, or null if they are not connected."""
    return {"root": _root(), "path": graph_mod.shortest_path(src, dst)}


@mcp.tool()
def central(limit: int = 10) -> dict[str, Any]:
    """Most-connected files (hubs) by link degree."""
    rows = graph_mod.centrality(limit=limit)
    return {
        "root": _root(),
        "hubs": [{"name": n, "path": r, "degree": d} for n, r, d in rows],
    }


@mcp.tool()
def clusters() -> dict[str, Any]:
    """Connected components of the link graph. Singletons are orphan files."""
    return {"root": _root(), "clusters": graph_mod.components()}


@mcp.tool()
def describe(
    path: str, description: str, tags: list[str] | None = None
) -> dict[str, Any]:
    """Record a description + tags for one file you already understand, into the
    editable .index.yaml store (the file itself is not modified). Use this to
    annotate a repo you already know without re-reading every file. `path` is a
    root-relative path or a bare file name. After a batch of describe() calls,
    call reindex() once so the changes show up in search/sql/map."""
    from . import generate

    rel = generate.record(None, path, description, list(tags or []))
    if rel is None:
        return {"root": _root(), "error": f"No file matching {path!r}."}
    return {
        "root": _root(),
        "recorded": rel,
        "description": description,
        "tags": list(tags or []),
        "note": "call reindex() when done to refresh search/sql",
    }


@mcp.tool()
def reindex() -> dict[str, Any]:
    """Rebuild the indexes, map, and catalog so prior describe() calls are
    reflected in search/sql/map. Call once after recording descriptions."""
    from .indexer import reindex as do_reindex

    summary = do_reindex(None)
    return {"root": _root(), "files": summary["files"]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
