"""Graph algorithms over the catalog's link structure, all in SQL.

Recursive CTEs give real graph queries (path, reach, components) and window
functions give centrality, with zero extra dependencies. Each query returns
only the slice it computes, never the whole graph, so an LLM can ask a precise
structural question without flooding its context.

If the duckpgq community extension ever ships for this DuckDB version, the same
queries could be expressed as SQL/PGQ MATCH patterns; the capability here does
not depend on it.
"""

from __future__ import annotations

from .catalog import read_cursor

# Undirected edge view used by every traversal below.
_EDGE_CTE = """
    edge(a, b) AS (
        SELECT src, dst FROM links WHERE dst_exists
        UNION ALL
        SELECT dst, src FROM links WHERE dst_exists
    )
"""


def shortest_path(
    src: str, dst: str, explicit_root: str | None = None, max_hops: int = 12
) -> list[str] | None:
    """Return the node names on a shortest path src..dst, or None if none."""
    if src == dst:
        return [src]
    con = read_cursor(explicit_root)
    try:
        row = con.execute(
            f"""
            WITH RECURSIVE {_EDGE_CTE},
            walk(name, dist, path) AS (
                SELECT ?, 0, [?]
                UNION ALL
                SELECT e.b, w.dist + 1, list_append(w.path, e.b)
                FROM walk w JOIN edge e ON e.a = w.name
                WHERE w.dist < ? AND NOT list_contains(w.path, e.b)
            )
            SELECT path FROM walk WHERE name = ? ORDER BY dist LIMIT 1
            """,
            [src, src, max_hops, dst],
        ).fetchone()
        return list(row[0]) if row else None
    finally:
        con.close()


def centrality(
    explicit_root: str | None = None, limit: int = 10
) -> list[tuple[str, str, int]]:
    """Degree centrality: most-connected files. Returns [(name, rel, degree)].

    Degree (inbound + outbound, existing edges) is the cheap, robust signal;
    it is what makes a file a hub. Returns the top `limit`.
    """
    con = read_cursor(explicit_root)
    try:
        return con.execute(
            f"""
            WITH {_EDGE_CTE},
            deg AS (SELECT a AS name, count(*) AS degree FROM edge GROUP BY a)
            SELECT n.name, n.rel, coalesce(d.degree, 0) AS degree
            FROM files n LEFT JOIN deg d ON d.name = n.name
            ORDER BY degree DESC, n.name
            LIMIT ?
            """,
            [limit],
        ).fetchall()
    finally:
        con.close()


def components(explicit_root: str | None = None) -> list[list[str]]:
    """Connected components (clusters) of the undirected link graph.

    Only files that participate in the link graph are included; isolated files
    (the norm once quack indexes whole repos) are omitted rather than reported
    as thousands of singletons. Returns a list of name-lists, largest first.
    """
    con = read_cursor(explicit_root)
    try:
        # Label-propagation seeded from linked nodes only: each node's component
        # id is the min node name reachable from it. Iterate to a fixed point.
        rows = con.execute(
            f"""
            WITH RECURSIVE {_EDGE_CTE},
            reach(name, root) AS (
                SELECT DISTINCT a, a FROM edge
                UNION
                SELECT e.b, r.root FROM reach r JOIN edge e ON e.a = r.name
            )
            SELECT name, min(root) AS comp FROM reach GROUP BY name
            """,
        ).fetchall()
    finally:
        con.close()

    groups: dict[str, list[str]] = {}
    for name, comp in rows:
        groups.setdefault(comp, []).append(name)
    return sorted((sorted(v) for v in groups.values()), key=len, reverse=True)
