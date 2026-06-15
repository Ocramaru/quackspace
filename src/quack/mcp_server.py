"""MCP server: expose the quack knowledge layer to LLMs as typed tools.

Instead of shelling out and parsing text, an MCP-aware assistant (Claude Code,
Kiro, Q) calls these tools directly. The server's `instructions` tell the LLM
how to navigate, so it always knows how to use quack without prior context.

Run with:  uv run quack-mcp   (or `quack mcp` once installed)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import graph as graph_mod
from . import catalog
from .config import (
    Config,
    DEFAULT_CENTRAL_LIMIT,
    DEFAULT_FILE_CHAR_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SQL_ROW_LIMIT,
)
from .core import find_root
from .search import search as do_search

INSTRUCTIONS = """\
This is quack, the meta layer over a directory of the user's work (notes, docs,
code, configs, assets). Use it to find and read files precisely, without loading
the whole tree into context.

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
  Follow that with `embed()` if you want semantic search to reflect the changes.
  This is the intended way to seed quack on a codebase an agent already knows.
- File CONTENTS are read-only here; only metadata (descriptions, tags) is writable.

Semantic search:
- `search()` automatically uses semantic (vector) similarity when embeddings
  exist. The `tiers` field on each hit shows which search modes contributed.
- `embed()` builds or refreshes embeddings. Call it after `reindex()` to keep
  the semantic tier current. It is a no-op if embeddings are not configured.

All paths are RELATIVE to the quack root, which every result reports as `root`.
To open a file, join root + path. Never assume an absolute path.

Call `explain()` for a full architecture and data-flow reference, field semantics,
and the catalog schema.
"""

mcp = FastMCP("quack", instructions=INSTRUCTIONS)

MAX_FILE_CHAR_LIMIT = 100_000
MAX_SQL_ROW_LIMIT = 200
MAX_SEARCH_LIMIT = 20
MAX_CENTRAL_LIMIT = 50


@dataclass
class LimitDefaults:
    search: int = DEFAULT_SEARCH_LIMIT
    file_chars: int = DEFAULT_FILE_CHAR_LIMIT
    sql_rows: int = DEFAULT_SQL_ROW_LIMIT
    central: int = DEFAULT_CENTRAL_LIMIT


LIMITS = LimitDefaults()
SERVER_ROOT: str | None = None


def configure_root(explicit_root: str | None = None) -> str:
    """Set the workspace root used by every MCP tool call."""
    global SERVER_ROOT
    SERVER_ROOT = str(find_root(explicit_root))
    return SERVER_ROOT


def _root_arg() -> str | None:
    return SERVER_ROOT


def configure_limits(
    search_limit: int | None = None,
    file_char_limit: int | None = None,
    sql_row_limit: int | None = None,
    central_limit: int | None = None,
    base: LimitDefaults | None = None,
) -> LimitDefaults:
    """Set process-wide MCP defaults, clamped to safe maximums."""
    global LIMITS
    base = base or LimitDefaults()
    LIMITS = LimitDefaults(
        search=_clamp(search_limit, base.search, MAX_SEARCH_LIMIT),
        file_chars=_clamp(file_char_limit, base.file_chars, MAX_FILE_CHAR_LIMIT),
        sql_rows=_clamp(sql_row_limit, base.sql_rows, MAX_SQL_ROW_LIMIT),
        central=_clamp(central_limit, base.central, MAX_CENTRAL_LIMIT),
    )
    return LIMITS


def configure_limits_from_config(
    explicit_root: str | None = None,
    search_limit: int | None = None,
    file_char_limit: int | None = None,
    sql_row_limit: int | None = None,
    central_limit: int | None = None,
) -> LimitDefaults:
    """Load workspace defaults from config.yaml, then apply flag overrides."""
    cfg = Config.load(explicit_root)
    return configure_limits(
        search_limit=search_limit,
        file_char_limit=file_char_limit,
        sql_row_limit=sql_row_limit,
        central_limit=central_limit,
        base=LimitDefaults(
            search=cfg.defaults.search_limit,
            file_chars=cfg.defaults.file_char_limit,
            sql_rows=cfg.defaults.sql_row_limit,
            central=cfg.defaults.central_limit,
        ),
    )


def _clamp(value: int | None, default: int, maximum: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, maximum))


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _query_limited(query: str, row_limit: int) -> tuple[list[str], list[tuple], bool]:
    # Cached connection (reused across MCP calls); close the cursor, not it.
    cur = catalog.read_cursor(_root_arg())
    try:
        cur.execute(query)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(row_limit + 1)
        return cols, rows[:row_limit], len(rows) > row_limit
    finally:
        cur.close()


def _root() -> str:
    return SERVER_ROOT or configure_root(None)


@mcp.tool()
def map() -> dict[str, Any]:
    """Folder-level overview of the root: each folder with its file count and
    description. Start here to decide which folder is relevant, then call
    get_file or search to go deeper."""
    cols, rows = catalog.query_shared(
        "SELECT f.folder, count(*) AS files, fo.description "
        "FROM files f LEFT JOIN folders fo ON f.folder = fo.folder "
        "WHERE f.folder <> '' GROUP BY f.folder, fo.description ORDER BY files DESC",
        explicit_root=_root_arg(),
    )
    return {
        "root": _root(),
        "folders": [dict(zip(cols, r)) for r in rows],
        "next_steps": (
            "Pick a folder then call search(query) to find files inside it, "
            "or sql(\"SELECT name, description FROM files WHERE folder = '<folder>'\") "
            "to list its contents directly."
        ),
    }


@mcp.tool()
def search(query: str, limit: int | None = None, expand: bool = True) -> dict[str, Any]:
    """Auto-hybrid search over files and folders. Returns up to `limit` results.

    Field semantics:
    - `tiers`: which ranking methods contributed — 'structural' (name/tag/description
      match, always available), 'fts' (body text via BM25, always available),
      'semantic' (vector similarity — only if embeddings were built with `quack embed`).
      A hit missing 'semantic' is normal when embeddings haven't been built.
    - `related_via`: non-empty when the hit was pulled in as a link-graph neighbour
      of a direct match, not matched directly itself.
    - `folders`: populated when the query asks about location/which-folder (separate
      list, never blended with file hits). Call `map()` or `sql()` for more folder detail.

    Call `explain()` for a full architecture and schema reference."""
    from .search import route, search_folders

    limit = _clamp(limit, LIMITS.search, MAX_SEARCH_LIMIT)
    hits = do_search(query, explicit_root=_root_arg(), limit=limit, expand=expand)

    folder_hits: list = []
    routed = route(query)
    if routed in ("folders", "both"):
        folder_hits = search_folders(query, explicit_root=_root_arg(), limit=limit)

    if not hits and not folder_hits:
        next_steps = (
            "No results. Try broader or different terms. "
            "If you recently called describe(), call reindex() first — "
            "the catalog won't reflect new descriptions until then."
        )
    else:
        tiers_seen = {t for h in hits for t in h.tiers}
        tips = ["Call get_file(path) to read any file."]
        if "semantic" not in tiers_seen:
            tips.append(
                "No 'semantic' tier in results — embeddings haven't been built. "
                "Run `quack embed` to enable semantic ranking."
            )
        if any(h.via for h in hits):
            tips.append(
                "'related_via' on a hit means it was pulled in as a link-graph "
                "neighbour of a direct match, not matched directly."
            )
        if any(h.entry.stale for h in hits):
            tips.append(
                "Some hits have stale=true — their description was written before "
                "the file last changed. Run `quack generate --stale` to refresh them."
            )
        next_steps = " ".join(tips)

    return {
        "root": _root(),
        "limit": limit,
        "routed_to": routed,
        "hits": [
            {
                "path": h.entry.rel,
                "name": h.entry.name,
                "description": h.entry.description,
                "tags": h.entry.tags,
                "tiers": h.tiers,
                "related_via": h.via,
                "stale": h.entry.stale,
            }
            for h in hits
        ],
        "folders": [
            {
                "path": f.folder,
                "parent": f.parent,
                "description": f.description,
                "via": f.via,
            }
            for f in folder_hits
        ],
        "next_steps": next_steps,
    }


@mcp.tool()
def get_file(path_or_name: str, char_limit: int | None = None) -> dict[str, Any]:
    """Read one file's full content + metadata. Accepts a root-relative path
    (src/app/main.py) or a bare file name without extension (main). Content is
    truncated by default — pass a higher `char_limit` if you need more.

    Field semantics:
    - `stale`: true when the file was modified after its description was last written.
      Run `quack generate --stale` to refresh stale descriptions.
    - `truncated`: true when content was cut at `content_limit`; pass
      char_limit=content_length to read the full file."""
    from .core import Space

    space = Space.load(_root_arg())
    entry = space.by_name.get(path_or_name) or space.by_rel.get(path_or_name)
    if entry is None:
        return {
            "root": _root(),
            "error": f"No file matching {path_or_name!r}.",
            "next_steps": "Try search(query) to find the file, or sql(\"SELECT rel FROM files WHERE name LIKE '%<term>%'\") to locate it by name.",
        }
    char_limit = _clamp(char_limit, LIMITS.file_chars, MAX_FILE_CHAR_LIMIT)
    content, truncated = _truncate(entry.body, char_limit)
    result: dict[str, Any] = {
        "root": _root(),
        "path": entry.rel,
        "name": entry.name,
        "description": entry.description,
        "tags": entry.tags,
        "links": entry.links,
        "is_binary": entry.is_binary,
        "modified": entry.modified,
        "stale": entry.stale,
        "content": content,
        "content_length": len(entry.body),
        "content_limit": char_limit,
        "truncated": truncated,
    }
    if truncated:
        result["next_steps"] = (
            f"Content truncated at {char_limit} of {len(entry.body)} chars. "
            f"Call get_file('{entry.rel}', char_limit={len(entry.body)}) to read the full file."
        )
    return result


@mcp.tool()
def sql(query: str, row_limit: int | None = None) -> dict[str, Any]:
    """Run read-only SQL against the catalog. Tables: files(name, rel, folder,
    ext, description, tags_csv, n_links, n_inbound, is_orphan, is_binary,
    file_modified, described_at, stale, body), folders(folder, parent,
    description, n_files, diagram, described_at) — the direct subfolders of X
    are WHERE parent = 'X' (root is ''), tags(name, tag),
    links(src, dst, dst_exists). Results are capped by `row_limit`; add SQL
    LIMIT clauses for more precise queries."""
    row_limit = _clamp(row_limit, LIMITS.sql_rows, MAX_SQL_ROW_LIMIT)
    cols, rows, truncated = _query_limited(query, row_limit)
    result: dict[str, Any] = {
        "root": _root(),
        "columns": cols,
        "rows": [list(r) for r in rows],
        "row_limit": row_limit,
        "truncated": truncated,
    }
    if truncated:
        result["next_steps"] = (
            f"Result capped at {row_limit} rows. Add a SQL LIMIT clause "
            f"or pass a higher row_limit (max {MAX_SQL_ROW_LIMIT}) to get more."
        )
    return result


@mcp.tool()
def graph_path(src: str, dst: str) -> dict[str, Any]:
    """Shortest path of wikilinks between two files (by name). Returns the node
    names on the path, or null if they are not connected."""
    return {"root": _root(), "path": graph_mod.shortest_path(src, dst, explicit_root=_root_arg())}


@mcp.tool()
def central(limit: int | None = None) -> dict[str, Any]:
    """Most-connected files (hubs) by link degree."""
    limit = _clamp(limit, LIMITS.central, MAX_CENTRAL_LIMIT)
    rows = graph_mod.centrality(explicit_root=_root_arg(), limit=limit)
    return {
        "root": _root(),
        "limit": limit,
        "hubs": [{"name": n, "path": r, "degree": d} for n, r, d in rows],
    }


@mcp.tool()
def clusters() -> dict[str, Any]:
    """Connected components of the link graph. Singletons are orphan files."""
    return {"root": _root(), "clusters": graph_mod.components(explicit_root=_root_arg())}


@mcp.tool()
def describe(
    path: str, description: str, tags: list[str] | None = None
) -> dict[str, Any]:
    """Record a description + tags for one file you already understand, into the
    editable .index.yaml store (the file itself is not modified). Use this to
    annotate a repo you already know without re-reading every file. `path` is a
    root-relative path or a bare file name. After a batch of describe() calls,
    call reindex() once so the changes show up in search/sql/map, then embed()
    to refresh semantic search."""
    from . import generate

    rel = generate.record(_root_arg(), path, description, list(tags or []))
    if rel is None:
        return {"root": _root(), "error": f"No file matching {path!r}."}
    return {
    "root": _root(),
    "recorded": rel,
    "description": description,
    "tags": list(tags or []),
    "next_steps": (
        "Metadata written to .index.yaml. "
        "Call reindex() once when done annotating, then embed() for semantic search. "
        "search/sql/map won't reflect these changes until the catalog is rebuilt."
    ),
}


@mcp.tool()
def reindex() -> dict[str, Any]:
    """Rebuild the indexes, map, and catalog so prior describe() calls are
    reflected in search/sql/map. Does NOT rebuild embeddings — call embed()
    after reindex() to refresh semantic search."""
    from .indexer import reindex as do_reindex

    summary = do_reindex(_root_arg())
    return {
        "root": _root(),
        "files": summary["files"],
        "catalog": summary.get("catalog", "rebuilt"),
        "next_steps": "Catalog is current. Call search() or sql() to query it.",
    }


@mcp.tool()
def explain() -> dict[str, Any]:
    """Architecture and data-flow reference for quack. Call this when you need to
    understand how the pieces fit together, what a tool result field means, or
    why behavior surprised you. Cheaper than re-reading QUACK.md."""
    return {
        "root": _root(),
        "overview": (
            "quack is a local filesystem meta layer. It indexes a directory of files "
            "into a DuckDB catalog and exposes that catalog over MCP so agents can "
            "navigate precisely without loading the whole tree into context."
        ),
        "data_flow": (
            "describe() writes into .index.yaml (the editable store). "
            "reindex() rebuilds .quack/quack.duckdb from those files. "
            "search/sql/map/get_file all read from the catalog. "
            "The catalog is derived and never authoritative — delete it and "
            "reindex() rebuilds it from scratch."
        ),
        "authoritative_vs_derived": {
            "authored — you edit these": [
                ".index.yaml per folder: descriptions + tags for direct children",
                ".quack/config.yaml: AI assistant choice and limits",
            ],
            "generated — never edit": [
                ".quack/quack.duckdb: the full catalog",
                ".quack/map.yaml: nested folder tree",
                "QUACK.md: navigation anchor",
                "<folder>/_diagrams.md: Mermaid link graph",
            ],
        },
        "search_tiers": {
            "structural": "name, tag, and description match — always available",
            "fts": "full-text body search via DuckDB BM25 — always available",
            "semantic": (
                "vector similarity — only available after `quack embed` has been run. "
                "A result with no 'semantic' tier is normal when embeddings don't exist."
            ),
        },
        "search_field_semantics": {
            "tiers": "which ranking methods contributed to this hit",
            "related_via": (
                "non-empty when the hit was expanded from the link graph as a neighbour "
                "of a direct match, not matched directly itself"
            ),
            "stale": "true when the file changed after its description was last written — refresh with `quack generate --stale`",
            "folders": (
                "populated when the query asks about location/which-folder; "
                "kept separate from file hits, never blended"
            ),
        },
        "catalog_schema": {
            "files": "name, rel, folder, ext, description, tags_csv, n_links, n_inbound, is_orphan, is_binary, file_modified, described_at, stale, body",
            "folders": "folder, parent, description, n_files, diagram, described_at — subfolders of X: WHERE parent = 'X' (root = '')",
            "tags": "name, tag",
            "links": "src, dst, dst_exists",
        },
        "annotation_workflow": (
            "Already know this repo? Call describe(path, description, tags) for each "
            "relevant file, then call reindex() once. No per-file model call needed — "
            "you write what you know and the catalog becomes searchable."
        ),
    }


@mcp.tool()
def embed() -> dict[str, Any]:
    """Build or refresh semantic search embeddings for all files and folders.
    Only re-embeds items whose content changed since the last run (incremental).
    Call after reindex() to keep semantic search current. Returns a summary of
    what was embedded, skipped, and deleted. If no embedding command is
    configured (embed.command in .quack/config.yaml), returns an error key
    instead of raising — run `quack embed init` at the terminal to set one up."""
    from .embed import EmbedNotConfigured, build_embeddings

    try:
        summary = build_embeddings(_root_arg())
    except EmbedNotConfigured:
        return {
            "root": _root(),
            "error": "No embedding command configured. Run `quack embed init` at the terminal first.",
        }
    return {
        "root": _root(),
        "embedded": summary["embedded"],
        "folders": summary["folders"],
        "dim": summary["dim"],
        "updated": summary["updated"],
        "skipped": summary["skipped"],
        "deleted": summary["deleted"],
        "folders_updated": summary["folders_updated"],
        "folders_skipped": summary["folders_skipped"],
        "folders_deleted": summary["folders_deleted"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quack-mcp")
    parser.add_argument("--root", default=None, help="quack root containing .quack/")
    parser.add_argument("--search-limit", type=int, default=None)
    parser.add_argument("--file-char-limit", type=int, default=None)
    parser.add_argument("--sql-row-limit", type=int, default=None)
    parser.add_argument("--central-limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    configure_root(args.root)
    configure_limits_from_config(
        explicit_root=_root_arg(),
        search_limit=args.search_limit,
        file_char_limit=args.file_char_limit,
        sql_row_limit=args.sql_row_limit,
        central_limit=args.central_limit,
    )
    mcp.run()


if __name__ == "__main__":
    main()
