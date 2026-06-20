"""MCP server: expose the quack knowledge layer to LLMs as typed tools.

Instead of shelling out and parsing text, an MCP-aware assistant (Claude Code,
Kiro, Q) calls these tools directly. The server's `instructions` tell the LLM
how to navigate, so it always knows how to use quack without prior context.

Run with:  uv run quack-mcp   (or `quack mcp` once installed)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import graph as graph_mod
from . import catalog
from .config import (
    Config,
    DEFAULT_CENTRAL_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SQL_ROW_LIMIT,
)
from .core import DEFAULT_OPAQUE_DIRS, find_root
from .search import search as do_search

INSTRUCTIONS = """\
This is quack, the meta layer over a directory of the user's work (notes, docs,
code, configs, assets). Use it to find and navigate files precisely.

How to navigate (cheapest first):
1. `map()` — bounded folder-level overview. By default it shows only top-level
   folders (`parent = ''`), not the whole tree. To go deeper, call
   `map(parent='<folder>')` or sql("SELECT folder, n_files FROM folders WHERE
   parent = '<folder>' ORDER BY n_files DESC"). To list files inside a folder,
   use sql("SELECT rel, name, description FROM files WHERE folder = '<folder>'
   ORDER BY name"), then pass any `rel` to `file_meta`.
2. `search` — auto-hybrid (keyword + full-text + semantic if available + graph
   neighbours). This is the default way to find files; you do not pick a mode.
3. `file_meta` — description, tags, links, stale flag, and the absolute path for
   a specific file. quack never returns file content — use the `absolute_path` it
   returns with your host's own file-reading tool so reads go through your normal
   permission flow.
4. `sql` / `graph_path` / `central` / `clusters` — precise structural queries.

Recording what you know:
- `describe(path, description, tags)` writes metadata into the editable store for
  a file you understand. If you already know this repo, describe its relevant
  files (one call each), then call `reindex()` ONCE so search/sql reflect them.
  Follow that with `embed()` if you want semantic search to reflect the changes.
  This is the intended way to seed quack on a codebase an agent already knows.
- File CONTENTS are never read or stored by quack tools; only metadata is writable.

Semantic search:
- `search()` automatically uses semantic (vector) similarity when embeddings
  exist. The `tiers` field on each hit shows which search modes contributed.
- `embed()` builds or refreshes embeddings. Call it after `reindex()` to keep
  the semantic tier current. It is a no-op if embeddings are not configured.

All paths are RELATIVE to the quack root, which every result reports as `root`.
`file_meta` also returns `absolute_path` — pass that directly to your file-reading
tool without constructing paths yourself.

Call `explain()` for a full architecture and data-flow reference, field semantics,
and the catalog schema. On large or vendored-heavy roots, prefer scoped
map/sql/search calls; use `.quackignore`, config `index.opaque_dirs`, or a
narrower quack root/per-project indexing to keep generated dependencies out.
"""

mcp = FastMCP("quack", instructions=INSTRUCTIONS)

MAX_SQL_ROW_LIMIT = 200
MAX_SEARCH_LIMIT = 20
MAX_CENTRAL_LIMIT = 50
MAX_MAP_LIMIT = 100
LARGE_ROOT_FILE_THRESHOLD = 10_000
LARGE_ROOT_FOLDER_THRESHOLD = 1_000
EDITOR_CACHE_DIRS = frozenset({".idea", ".ipynb_checkpoints"})


@dataclass
class LimitDefaults:
    search: int = DEFAULT_SEARCH_LIMIT
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
    sql_row_limit: int | None = None,
    central_limit: int | None = None,
    base: LimitDefaults | None = None,
) -> LimitDefaults:
    """Set process-wide MCP defaults, clamped to safe maximums."""
    global LIMITS
    base = base or LimitDefaults()
    LIMITS = LimitDefaults(
        search=_clamp(search_limit, base.search, MAX_SEARCH_LIMIT),
        sql_rows=_clamp(sql_row_limit, base.sql_rows, MAX_SQL_ROW_LIMIT),
        central=_clamp(central_limit, base.central, MAX_CENTRAL_LIMIT),
    )
    return LIMITS


def configure_limits_from_config(
    explicit_root: str | None = None,
    search_limit: int | None = None,
    sql_row_limit: int | None = None,
    central_limit: int | None = None,
) -> LimitDefaults:
    """Load workspace defaults from config.yaml, then apply flag overrides."""
    cfg = Config.load(explicit_root)
    return configure_limits(
        search_limit=search_limit,
        sql_row_limit=sql_row_limit,
        central_limit=central_limit,
        base=LimitDefaults(
            search=cfg.defaults.search_limit,
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


def _catalog_counts() -> dict[str, int]:
    cur = catalog.read_cursor(_root_arg())
    try:
        files = cur.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        folders = cur.execute("SELECT COUNT(*) FROM folders").fetchone()[0]
        return {
            "files": int(files or 0),
            "folders": int(folders or 0),
        }
    finally:
        cur.close()


def _is_large_catalog(counts: dict[str, int]) -> bool:
    return (
        counts["files"] >= LARGE_ROOT_FILE_THRESHOLD
        or counts["folders"] >= LARGE_ROOT_FOLDER_THRESHOLD
    )


def _configured_opaque_dirs() -> frozenset[str]:
    try:
        cfg = Config.load(_root_arg())
        return DEFAULT_OPAQUE_DIRS | frozenset(cfg.index.opaque_dirs)
    except Exception:
        return DEFAULT_OPAQUE_DIRS


def _path_has_dir(rel: str, names: set[str] | frozenset[str]) -> bool:
    parts = rel.split("/")[:-1]
    return any(p in names for p in parts)


def _scope_guidance() -> str:
    return (
        "For large or dependency-heavy roots, add generated/vendor directories to "
        "`.quackignore` or config `index.opaque_dirs`, or run quack per project with "
        "a narrower root."
    )


def _root() -> str:
    return SERVER_ROOT or configure_root(None)


def _render_folder_tree(
    top_label: str, shown: list, truncated: bool, child_count: int
) -> str:
    """Render one level of child folders as a tree, e.g.

        Projects
        ├── knit_to_knit (14) "A project where we go over..."
        └── sandbox (11)

    Uses ``rich.tree`` for the connectors. File counts follow each entry as
    ``(n)``; the description is appended in quotes only when present, truncated
    so each line stays short. Nodes are added as plain ``Text`` so arbitrary
    folder names/descriptions are never interpreted as console markup.
    """
    import io

    from rich.console import Console
    from rich.text import Text
    from rich.tree import Tree

    tree = Tree(Text(top_label))
    for folder, _row_parent, desc, n_files in shown:
        leaf = folder.rsplit("/", 1)[-1] or folder
        desc = (desc or "").strip()
        if len(desc) > 80:
            desc = desc[:79].rstrip() + "…"
        label = f"{leaf} ({int(n_files or 0)})"
        if desc:
            label += f' "{desc}"'
        tree.add(Text(label))
    if not shown:
        tree.add(Text("(no subfolders)"))
    elif truncated:
        tree.add(Text(
            f"… ({child_count - len(shown)} more folder(s) not shown — "
            "narrow with map(parent='<folder>'))"
        ))

    buf = io.StringIO()
    # Non-terminal Console renders plain text (no ANSI); wide width avoids wrap.
    Console(file=buf, force_terminal=False, no_color=True, width=200).print(tree)
    return buf.getvalue().rstrip("\n")


@mcp.tool()
def map(parent: str = "", limit: int | None = None) -> dict[str, Any]:
    """Bounded folder-level overview, rendered as an indented `tree` string
    (folder name, `(file count)`, and description when present). By default it
    shows only top-level folders (`parent = ''`), never the whole tree. Pass
    `parent='<folder>'` to descend one level. Shows folders, not files. The
    `root`/`child_count`/`total_files`/`total_folders`/`truncated` fields give
    the scale of the catalog so you know when to stay scoped."""
    limit = _clamp(limit, MAX_MAP_LIMIT, MAX_MAP_LIMIT)
    parent = (parent or "").strip("/")
    cur = catalog.read_cursor(_root_arg())
    try:
        rows = cur.execute(
            """
            SELECT folder, parent, description, n_files
            FROM folders
            WHERE parent = ?
            ORDER BY n_files DESC, folder
            LIMIT ?
            """,
            [parent, limit + 1],
        ).fetchall()
        child_count = cur.execute(
            "SELECT COUNT(*) FROM folders WHERE parent = ?", [parent]
        ).fetchone()[0]
    finally:
        cur.close()
    counts = _catalog_counts()
    truncated = len(rows) > limit
    shown = rows[:limit]
    tips = [
        "This is a one-level folder view. Pick a folder and call map(parent='<folder>') "
        "to descend, or use sql(\"SELECT folder, n_files FROM folders WHERE parent = '<folder>' ORDER BY n_files DESC\").",
        "To list files inside a folder, use sql(\"SELECT rel, name, description FROM files WHERE folder = '<folder>' ORDER BY name\"), then call file_meta(path_or_name=rel).",
    ]
    if truncated:
        tips.append(
            f"Only {limit} of {child_count} child folders are shown; use a narrower parent "
            "instead of asking for the whole tree."
        )
    if _is_large_catalog(counts):
        tips.append(
            f"This catalog is large ({counts['files']} files, {counts['folders']} folders); "
            "avoid unbounded folder/file SQL and stay scoped by parent or folder."
        )
        tips.append(_scope_guidance())
    root = _root()
    top_label = parent or (root.rstrip("/").rsplit("/", 1)[-1] or root)
    return {
        "root": root,
        "parent": parent,
        "limit": limit,
        "child_count": child_count,
        "total_files": counts["files"],
        "total_folders": counts["folders"],
        "truncated": truncated,
        "tree": _render_folder_tree(top_label, shown, truncated, child_count),
        "next_steps": " ".join(tips),
    }


@mcp.tool()
def search(query: str, limit: int | None = None, expand: bool = True) -> dict[str, Any]:
    """Auto-hybrid search over files and folders. Returns up to `limit` results.

    Field semantics:
    - `tiers`: which ranking methods contributed — 'structural' (name/tag/description
      match, always available), 'fts' (full-text search over indexed content via BM25,
      always available), 'semantic' (vector similarity — only if embeddings were built
      with `quack embed`).
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
    elif not hits and folder_hits:
        next_steps = (
            "Folder matches found but no file hits. "
            "Use sql(\"SELECT rel, name, description FROM files WHERE folder = '<folder>' ORDER BY name\") "
            "to list files inside a matched folder, then call file_meta(path_or_name=rel)."
        )
    else:
        tiers_seen = {t for h in hits for t in h.tiers}
        tips = ["Call file_meta(path_or_name=path) for metadata and absolute_path, then read with your host file tool."]
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
                "Some hits have stale=true — their description may be outdated. "
                "Call file_meta(path_or_name=path) for absolute_path, then read "
                "with your host file tool before relying on the description."
            )
        next_steps = " ".join(tips)

    counts = _catalog_counts()
    extra_tips = []
    if (hits and len(hits) >= limit) or (folder_hits and len(folder_hits) >= limit):
        extra_tips.append(
            f"Results are capped at limit={limit}; on large roots, narrow the query "
            "with distinctive terms or use map(parent='<folder>')/folder-scoped SQL first."
        )
    if _is_large_catalog(counts) and (hits or folder_hits):
        extra_tips.append(
            f"This catalog is large ({counts['files']} files, {counts['folders']} folders); "
            "prefer scoped search terms and folder filters over broad queries."
        )
    noisy_dirs = _configured_opaque_dirs() | EDITOR_CACHE_DIRS
    noisy_hits = [h.entry.rel for h in hits if _path_has_dir(h.entry.rel, noisy_dirs)]
    if noisy_hits:
        extra_tips.append(
            "Some hits are under editor/cache or opaque dependency directories. "
            + _scope_guidance()
        )
    if extra_tips:
        next_steps = " ".join([next_steps, *extra_tips])

    return {
        "root": _root(),
        "query": query,
        "limit": limit,
        "max_limit": MAX_SEARCH_LIMIT,
        "hit_count": len(hits),
        "folder_count": len(folder_hits),
        "routed_to": routed,
        "searched_files": routed in ("files", "both"),
        "searched_folders": routed in ("folders", "both"),
        "hits": [
            {
                "path": h.entry.rel,
                "name": h.entry.name,
                "description": h.entry.description,
                "tags": h.entry.tags,
                "score": round(h.score, 4),
                "tiers": h.tiers,
                "related_via": h.via,
                "stale": h.entry.stale,
            }
            for h in hits
        ],
        "folders": [
            {
                "folder": f.folder,
                "parent": f.parent,
                "description": f.description,
                "matched_by": f.via,
            }
            for f in folder_hits
        ],
        "next_steps": next_steps,
    }


@mcp.tool()
def file_meta(path_or_name: str) -> dict[str, Any]:
    """Metadata for one file: description, tags, wikilinks, stale flag, modified
    time, and the absolute path to pass to your host file-reading tool.

    quack never reads or returns file content — reads must go through your host
    environment's own tool so they flow through its normal permission checks.
    Use `absolute_path` directly; do not construct it from `root` + `path`.

    Accepts a root-relative path (preferred — unambiguous; src/app/main.py) or a
    bare file name (main). Bare names may match multiple files; prefer the `path`
    field from search() or sql() results to avoid ambiguity.

    Field semantics:
    - `stale`: true when the file was modified after its description was last written.
    - `links`: wikilink targets as {name, path, ambiguous, exists} — `path` is the
      root-relative path when resolved unambiguously, else null; `ambiguous` is true
      when multiple files share the name."""
    root = _root()
    cur = catalog.read_cursor(_root_arg())
    try:
        rows = cur.execute(
            "SELECT rel, name, folder, description, tags_csv, "
            "is_binary, file_modified, stale "
            "FROM files WHERE rel = ? OR name = ? LIMIT 2",
            [path_or_name, path_or_name],
        ).fetchall()

        if not rows:
            return {
                "root": root,
                "error": f"No file matching {path_or_name!r}.",
                "next_steps": (
                    "Try search(query) or "
                    "sql(\"SELECT rel, name FROM files WHERE name LIKE '%<term>%'\") "
                    "to locate the file, then call file_meta(path_or_name=rel)."
                ),
            }

        if len(rows) > 1:
            return {
                "root": root,
                "error": f"Ambiguous name {path_or_name!r} — multiple files match.",
                "candidates": [{"path": r[0], "name": r[1], "folder": r[2]} for r in rows],
                "next_steps": "Call file_meta(path_or_name=<path>) with one of the candidate paths.",
            }

        rel, name, folder, description, tags_csv, is_binary, modified, stale = rows[0]
        link_rows = cur.execute(
            "SELECT dst, dst_exists FROM links WHERE src = ?", [name]
        ).fetchall()
        dst_names = [dst for dst, _ in link_rows]
        paths_by_dst: dict[str, list[str]] = {}
        if dst_names:
            ph = ",".join("?" for _ in dst_names)
            for dst_name, dst_rel in cur.execute(
                f"SELECT name, rel FROM files WHERE name IN ({ph})", dst_names
            ).fetchall():
                paths_by_dst.setdefault(dst_name, []).append(dst_rel)
    finally:
        cur.close()

    tags = [t for t in (tags_csv or "").split(",") if t]
    links = []
    for dst, exists in link_rows:
        dst_paths = paths_by_dst.get(dst, [])
        links.append({
            "name": dst,
            "path": dst_paths[0] if len(dst_paths) == 1 else None,
            "ambiguous": len(dst_paths) > 1,
            "exists": bool(exists),
        })
    absolute_path = str(Path(root) / rel)

    return {
        "root": root,
        "path": rel,
        "absolute_path": absolute_path,
        "name": name,
        "folder": folder,
        "description": description or "",
        "tags": tags,
        "links": links,
        "is_binary": bool(is_binary),
        "modified": str(modified) if modified else None,
        "stale": bool(stale),
        "next_steps": (
            f"Read content with your host file-reading tool at: {absolute_path!r}. "
            "If you learn stable metadata worth recording, call describe(path, description, tags) "
            "then reindex() once after your batch."
        ),
    }


@mcp.tool()
def sql(query: str, row_limit: int | None = None) -> dict[str, Any]:
    """Run read-only SQL against the catalog. Tables: files(name, rel, folder,
    ext, description, tags_csv, n_links, n_inbound, is_orphan, is_binary,
    file_modified, described_at, stale), folders(folder, parent,
    description, n_files, diagram, described_at) — the direct subfolders of X
    are WHERE parent = 'X' (root is ''), tags(name, tag),
    links(src, dst, dst_exists). Results are capped by `row_limit`; add SQL
    LIMIT clauses for more precise queries.

    `files.rel` is the root-relative path — pass it directly to file_meta() as
    path_or_name to get metadata and absolute_path. `files.name` is the bare stem
    without extension and may be ambiguous when multiple files share a name.
    Note: files.body is indexed for full-text search but returning it via sql()
    bypasses host permission controls — use file_meta() + your host tool instead."""
    row_limit = _clamp(row_limit, LIMITS.sql_rows, MAX_SQL_ROW_LIMIT)
    cols, rows, truncated = _query_limited(query, row_limit)
    records = [dict(zip(cols, r)) for r in rows]
    result: dict[str, Any] = {
        "root": _root(),
        "query": query,
        "columns": cols,
        "rows": [list(r) for r in rows],
        "records": records,
        "row_count": len(rows),
        "row_limit": row_limit,
        "truncated": truncated,
    }
    counts = _catalog_counts()
    if truncated:
        result["next_steps"] = (
            f"Result capped at {row_limit} rows and may be too large for MCP context. "
            "Narrow the WHERE clause, use parent-scoped folder queries "
            "(folders.parent = '<folder>'), or use map(parent='<folder>') first. "
            "If rows contain files.rel, pass that value to file_meta(path_or_name=rel) for metadata and absolute_path."
        )
    else:
        result["next_steps"] = (
            "If records contain a `rel` field, pass it to file_meta(path_or_name=rel) "
            "to get metadata and absolute_path for reading with your host file tool."
        )
        if _is_large_catalog(counts):
            result["next_steps"] += (
                f" This catalog is large ({counts['files']} files, {counts['folders']} folders); "
                "avoid unbounded SELECTs and prefer folder/parent filters."
            )
    return result


@mcp.tool()
def graph_path(src: str, dst: str) -> dict[str, Any]:
    """Shortest wikilink path between two files, identified by name. Use after
    search(), file_meta(), or sql() when you already have two specific file names
    and want to understand how they are connected. Returns node names on the path,
    or null if they are not reachable from each other.

    Note: `path` contains file *names* (bare stems), not root-relative paths.
    `nodes` provides the same sequence as `{name, path}` objects for direct use
    with file_meta or your host file tool."""
    path = graph_mod.shortest_path(src, dst, explicit_root=_root_arg())
    nodes = None
    if path:
        _cur = catalog.read_cursor(_root_arg())
        try:
            ph = ",".join("?" for _ in path)
            _path_by_name = dict(
                _cur.execute(
                    f"SELECT name, rel FROM files WHERE name IN ({ph})", path
                ).fetchall()
            )
        finally:
            _cur.close()
        nodes = [{"name": n, "path": _path_by_name.get(n)} for n in path]
    return {
        "root": _root(),
        "src": src,
        "dst": dst,
        "connected": path is not None,
        "path": path,
        "nodes": nodes,
        "next_steps": (
            "Nodes in `path` are file names. Call file_meta(path_or_name=name) for "
            "metadata and the absolute path, then read with your host file tool."
        ) if path else (
            "No wikilink connection found. Try central() to find hub files, or "
            "clusters() to see whether src and dst belong to different topic islands."
        ),
    }


@mcp.tool()
def central(limit: int | None = None) -> dict[str, Any]:
    """Most-connected files (hubs) by wikilink degree. Use when you want
    important or heavily-referenced files rather than keyword-relevant ones —
    hubs are natural starting points for exploring an unfamiliar space.
    Files under configured opaque dependency/cache dirs are excluded."""
    limit = _clamp(limit, LIMITS.central, MAX_CENTRAL_LIMIT)
    opaque_dirs = _configured_opaque_dirs()
    rows = graph_mod.centrality(
        explicit_root=_root_arg(),
        limit=limit,
        exclude_dir_names=opaque_dirs,
    )
    counts = _catalog_counts()
    tips = [
        "Call file_meta(path_or_name=path) on any hub for metadata and absolute_path, "
        "then read with your host file tool. Or call graph_path(src, dst) to trace connections."
    ]
    if _is_large_catalog(counts):
        tips.append(
            "On large roots, centrality can still surface generated/vendor noise if those "
            "dirs were indexed; add them to `.quackignore` or config `index.opaque_dirs`, "
            "then reindex."
        )
    if len(rows) >= limit:
        tips.append(
            f"Hubs are capped at limit={limit}; use search() or folder-scoped SQL when "
            "you already know the area you care about."
        )
    return {
        "root": _root(),
        "limit": limit,
        "max_limit": MAX_CENTRAL_LIMIT,
        "excluded_opaque_dirs": sorted(opaque_dirs),
        "hubs": [{"name": n, "path": r, "degree": d} for n, r, d in rows],
        "next_steps": " ".join(tips),
    }


@mcp.tool()
def clusters() -> dict[str, Any]:
    """Connected components of the wikilink graph. Use to discover topic islands
    (groups of inter-linked files) and isolated files (singletons — files with no
    wikilink edges at all, distinct from files.is_orphan which tracks inbound links).
    Complements search() when you want topological structure rather than relevance."""
    raw_clusters = graph_mod.components(explicit_root=_root_arg())
    all_names = [n for cluster in raw_clusters for n in cluster]
    path_by_name: dict[str, str] = {}
    if all_names:
        _cur = catalog.read_cursor(_root_arg())
        try:
            ph = ",".join("?" for _ in all_names)
            path_by_name = dict(
                _cur.execute(
                    f"SELECT name, rel FROM files WHERE name IN ({ph})", all_names
                ).fetchall()
            )
        finally:
            _cur.close()
    return {
        "root": _root(),
        "clusters": [
            [{"name": n, "path": path_by_name.get(n)} for n in cluster]
            for cluster in raw_clusters
        ],
        "next_steps": (
            "Each cluster is a list of {name, path} objects. "
            "Call file_meta(path_or_name=path) for metadata and absolute_path to read with your host tool. "
            "Use graph_path(src, dst) or sql() to inspect relationships among names in a cluster."
        ),
    }


@mcp.tool()
def describe(
    path: str, description: str, tags: list[str] | None = None
) -> dict[str, Any]:
    """Record a description + tags for one file you already understand, into the
    editable .index.yaml store (the file itself is not modified). Use this to
    annotate a repo you already know without re-reading every file.

    `path` accepts a root-relative path (preferred — unambiguous) or a bare file
    name. Prefer `path` from search()/central() results or `rel` from sql() results
    to avoid name collisions — bare names are accepted only when unique across the
    space. After a batch of describe() calls, call reindex() once so changes show
    up in search/sql/map, then embed() to refresh semantic search."""
    from . import generate

    _cur = catalog.read_cursor(_root_arg())
    try:
        _amb_rows = _cur.execute(
            "SELECT rel, name, folder FROM files WHERE rel = ? OR name = ? LIMIT 2",
            [path, path],
        ).fetchall()
    finally:
        _cur.close()
    if len(_amb_rows) > 1:
        return {
            "root": _root(),
            "error": f"Ambiguous name {path!r} — multiple files match.",
            "candidates": [{"path": r[0], "name": r[1], "folder": r[2]} for r in _amb_rows],
            "next_steps": "Call describe(path=<path>, ...) with one of the candidate `path` values (root-relative, unambiguous).",
        }

    rel = generate.record(_root_arg(), path, description, list(tags or []))
    if rel is None:
        return {
            "root": _root(),
            "error": f"No file matching {path!r}.",
            "next_steps": (
                "Call search(query) or "
                "sql(\"SELECT rel, name FROM files WHERE name LIKE '%<term>%'\") "
                "to find the exact path, then call describe(path=<rel>, ...)."
            ),
        }
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
    described_files = 0
    stale_files = 0
    folder_count = 0
    try:
        _cur = catalog.read_cursor(_root_arg())
        try:
            row = _cur.execute(
                "SELECT "
                "COUNT(*) FILTER (WHERE description IS NOT NULL AND description != '') AS d, "
                "COUNT(*) FILTER (WHERE stale) AS s "
                "FROM files"
            ).fetchone()
            if row:
                described_files, stale_files = row
            folder_count = _cur.execute("SELECT COUNT(*) FROM folders").fetchone()[0]
        finally:
            _cur.close()
    except Exception:
        pass
    return {
        "root": _root(),
        "files": summary["files"],
        "described_files": described_files,
        "stale_files": stale_files,
        "folders": folder_count,
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
            "reindex() scans the filesystem to rebuild .quack/quack.duckdb. "
            "search/sql/map/file_meta all read from the catalog — MCP tool results "
            "never include file content, only indexed metadata. "
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
            "fts": "full-text search over indexed content via DuckDB BM25 — always available",
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
            "stale": "true when the file changed after its description was last written",
            "folders": (
                "populated when the query asks about location/which-folder; "
                "kept separate from file hits, never blended"
            ),
        },
        "catalog_schema": {
            "files": "name, rel, folder, ext, description, tags_csv, n_links, n_inbound, is_orphan, is_binary, file_modified, described_at, stale",
            "folders": "folder, parent, description, n_files, diagram, described_at — subfolders of X: WHERE parent = 'X' (root = '')",
            "tags": "name, tag",
            "links": "src, dst, dst_exists",
        },
        "large_root_guidance": (
            "For large or vendored-heavy roots, start with map()'s top-level view "
            "or folders.parent SQL, then scope file queries by folder. Hide generated "
            "or cache trees with `.quackignore`; record dependency/cache dirs you want "
            "acknowledged but not indexed in `.quack/config.yaml` index.opaque_dirs; "
            "or run quack per project with a narrower root."
        ),
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
    parser.add_argument("--sql-row-limit", type=int, default=None)
    parser.add_argument("--central-limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    configure_root(args.root)
    configure_limits_from_config(
        explicit_root=_root_arg(),
        search_limit=args.search_limit,
        sql_row_limit=args.sql_row_limit,
        central_limit=args.central_limit,
    )
    mcp.run()


if __name__ == "__main__":
    main()
