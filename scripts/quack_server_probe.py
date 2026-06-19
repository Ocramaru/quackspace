"""
Prototype: test FTS, VSS (HNSW), and graph search over the DuckDB Quack server protocol.

Usage:
    uv run python scripts/quack_server_probe.py [path/to/quack.duckdb]

What this tests:
  1. Can quack_serve() start against an existing catalog with FTS + VSS indexes?
  2. Can a second in-process connection attach via the Quack protocol and run queries?
  3. Do FTS (match_bm25), VSS (array_cosine_distance / HNSW), and graph (recursive CTE) queries
     all work through the client-side ATTACH?
  4. Does the client need to LOAD extensions, or does the server handle them?

Architecture being validated:
    Server conn (R/W, owns file lock, serves quack_serve)
    Client conn (in-memory, ATTACHes to server, runs all queries)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb

# ── helpers ──────────────────────────────────────────────────────────────────

TOKEN = "quacktest"
PORT = 19494  # non-default port to avoid collisions
SERVER_URI = f"quack:localhost:{PORT}"


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def ok(label: str, value: object = "") -> None:
    suffix = f"  →  {value}" if value != "" else ""
    print(f"  ✓  {label}{suffix}")


def fail(label: str, err: Exception) -> None:
    print(f"  ✗  {label}")
    print(f"       {type(err).__name__}: {err}")


# ── server setup ─────────────────────────────────────────────────────────────

def start_server(catalog_path: Path) -> duckdb.DuckDBPyConnection:
    """Open the catalog in R/W mode and start the Quack HTTP server."""
    con = duckdb.connect(str(catalog_path))
    # Extensions must be loaded in the server process before serving.
    con.execute("INSTALL quack; LOAD quack;")
    con.execute("INSTALL fts;   LOAD fts;")
    con.execute("INSTALL vss;   LOAD vss;")
    con.execute("SET hnsw_enable_experimental_persistence = true;")
    result = con.execute(
        f"CALL quack_serve('{SERVER_URI}', token := '{TOKEN}')"
    ).fetchone()
    print(f"\n  Server started: {result}")
    return con


def stop_server(con: duckdb.DuckDBPyConnection) -> None:
    try:
        con.execute(f"CALL quack_stop('{SERVER_URI}')")
    except Exception:
        pass
    con.close()


# ── client setup ─────────────────────────────────────────────────────────────

def make_client() -> duckdb.DuckDBPyConnection:
    """Create an in-memory client connection and ATTACH the Quack server."""
    client = duckdb.connect()
    client.execute("INSTALL quack; LOAD quack;")
    client.execute(f"ATTACH '{SERVER_URI}' AS cat (TOKEN '{TOKEN}')")
    return client


# ── test 1: FTS ──────────────────────────────────────────────────────────────

def test_fts(client: duckdb.DuckDBPyConnection, server_con: duckdb.DuckDBPyConnection) -> None:
    header("TEST 1 — Full-Text Search (BM25 via fts extension)")

    # 1a: client calls match_bm25 directly (extension functions are schema-bound)
    for load_ext in (False, True):
        label = f"match_bm25 direct {'(client LOAD fts)' if load_ext else '(no client LOAD fts)'}"
        try:
            if load_ext:
                client.execute("INSTALL fts; LOAD fts;")
            rows = client.execute(
                """
                SELECT rel, score FROM (
                    SELECT rel,
                           fts_main_files.match_bm25(name, 'catalog') AS score
                    FROM cat.files
                ) WHERE score IS NOT NULL
                ORDER BY score DESC LIMIT 3
                """
            ).fetchall()
            ok(label, rows)
            break
        except Exception as e:
            fail(label, e)

    # 1b: server creates a VIEW wrapping the FTS query; client reads the view
    label_view = "FTS via server-side VIEW (workaround)"
    try:
        server_con.execute(
            """
            CREATE OR REPLACE VIEW fts_catalog AS
            SELECT rel, name, description, score FROM (
                SELECT rel, name, description,
                       fts_main_files.match_bm25(name, 'catalog') AS score
                FROM files
            ) WHERE score IS NOT NULL
            ORDER BY score DESC
            """
        )
        rows = client.execute("SELECT rel, score FROM cat.fts_catalog LIMIT 3").fetchall()
        ok(label_view, rows)
    except Exception as e:
        fail(label_view, e)

    # 1c: parameterised FTS via a server-side MACRO
    label_macro = "FTS via server-side MACRO(terms)"
    try:
        server_con.execute(
            """
            CREATE OR REPLACE MACRO fts_search(terms) AS TABLE
            SELECT rel, name, description, score FROM (
                SELECT rel, name, description,
                       fts_main_files.match_bm25(name, terms) AS score
                FROM files
            ) WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT 10
            """
        )
        rows = client.execute(
            "SELECT rel, score FROM cat.fts_search('reindex') LIMIT 3"
        ).fetchall()
        ok(label_macro, rows)
    except Exception as e:
        fail(label_macro, e)


# ── test 2: VSS / HNSW ───────────────────────────────────────────────────────

def test_vss(client: duckdb.DuckDBPyConnection, server_con: duckdb.DuckDBPyConnection) -> None:
    header("TEST 2 — Vector Search (array_cosine_distance / HNSW)")

    # Grab a real embedding vector from the server to use as query vector.
    sample = server_con.execute(
        "SELECT vec FROM embeddings LIMIT 1"
    ).fetchone()
    if not sample:
        print("  ⚠  No embeddings in catalog — skipping VSS test.")
        return

    vec = sample[0]
    dim = len(vec)

    # Test 2a: raw cosine distance (no HNSW index, pure brute-force)
    label_bf = f"array_cosine_distance brute-force (dim={dim}), no client LOAD vss"
    try:
        rows = client.execute(
            f"""
            SELECT e.rel, array_cosine_distance(e.vec, ?::FLOAT[{dim}]) AS dist
            FROM cat.embeddings e
            ORDER BY dist LIMIT 3
            """,
            [vec],
        ).fetchall()
        ok(label_bf, rows)
    except Exception as e:
        fail(label_bf, e)
        # Try again with client loading vss
        try:
            client.execute("INSTALL vss; LOAD vss;")
            rows = client.execute(
                f"""
                SELECT e.rel, array_cosine_distance(e.vec, ?::FLOAT[{dim}]) AS dist
                FROM cat.embeddings e
                ORDER BY dist LIMIT 3
                """,
                [vec],
            ).fetchall()
            ok(f"array_cosine_distance (client LOAD vss)", rows)
        except Exception as e2:
            fail("array_cosine_distance (client LOAD vss)", e2)

    # Test 2b: HNSW index scan (ORDER BY with LIMIT triggers ANN)
    label_hnsw = "HNSW index ANN scan via ORDER BY dist LIMIT"
    try:
        rows = client.execute(
            f"""
            SELECT e.rel, array_cosine_distance(e.vec, ?::FLOAT[{dim}]) AS dist
            FROM cat.embeddings e
            ORDER BY dist LIMIT 5
            """,
            [vec],
        ).fetchall()
        ok(label_hnsw, rows)
    except Exception as e:
        fail(label_hnsw, e)

    # Test 2c: EXPLAIN to see if HNSW index is actually used
    label_explain = "EXPLAIN shows HNSW_INDEX_SCAN (not seq scan)"
    try:
        plan = client.execute(
            f"""
            EXPLAIN SELECT e.rel, array_cosine_distance(e.vec, ?::FLOAT[{dim}]) AS dist
            FROM cat.embeddings e
            ORDER BY dist LIMIT 5
            """,
            [vec],
        ).fetchall()
        plan_text = "\n".join(str(r) for r in plan)
        if "HNSW" in plan_text.upper():
            ok(label_explain, "HNSW_INDEX_SCAN found in plan")
        else:
            print(f"  ⚠  {label_explain}")
            print(f"       No HNSW_INDEX_SCAN in plan — using seq scan")
            print(f"       Plan excerpt: {plan_text[:300]}")
    except Exception as e:
        fail(label_explain, e)


# ── test 3: Graph (recursive CTE) ────────────────────────────────────────────

def test_graph(client: duckdb.DuckDBPyConnection, server_con: duckdb.DuckDBPyConnection) -> None:
    header("TEST 3 — Graph Search (recursive CTE over links table)")

    # Seed some links so the graph queries have data to traverse.
    server_con.execute("DELETE FROM links WHERE src LIKE '__probe_%'")
    server_con.executemany(
        "INSERT INTO links VALUES (?, ?, true)",
        [
            ("__probe_a__", "__probe_b__"),
            ("__probe_b__", "__probe_c__"),
            ("__probe_c__", "__probe_d__"),
            ("__probe_a__", "__probe_d__"),
        ],
    )

    # Test 3a: simple link count
    label_count = "SELECT count(*) FROM cat.links"
    try:
        count = client.execute("SELECT count(*) FROM cat.links").fetchone()[0]
        ok(label_count, f"{count} links")
    except Exception as e:
        fail(label_count, e)
        return

    # Test 3b: degree centrality — single-table GROUP BY (no JOIN)
    label_deg = "Degree centrality (single-scan GROUP BY, no JOIN)"
    try:
        rows = client.execute(
            """
            WITH edge(a, b) AS (
                SELECT src, dst FROM cat.links WHERE dst_exists
                UNION ALL
                SELECT dst, src FROM cat.links WHERE dst_exists
            )
            SELECT a AS name, count(*) AS degree FROM edge GROUP BY a
            ORDER BY degree DESC LIMIT 5
            """
        ).fetchall()
        ok(label_deg, rows)
    except Exception as e:
        fail(label_deg, e)

    # Test 3c: degree centrality with JOIN to files (two streaming scans)
    label_join = "Degree centrality JOIN files (multi-scan — known Quack limitation)"
    try:
        rows = client.execute(
            """
            WITH edge(a, b) AS (
                SELECT src, dst FROM cat.links WHERE dst_exists
                UNION ALL
                SELECT dst, src FROM cat.links WHERE dst_exists
            ),
            deg AS (SELECT a AS name, count(*) AS degree FROM edge GROUP BY a)
            SELECT n.name, coalesce(d.degree, 0) AS degree
            FROM cat.files n LEFT JOIN deg d ON d.name = n.name
            ORDER BY degree DESC, n.name LIMIT 5
            """
        ).fetchall()
        ok(label_join, rows)
    except Exception as e:
        fail(label_join, e)

    # Test 3d: recursive CTE connected components (correct SQL from graph.py)
    label_rec = "Recursive CTE connected components"
    try:
        rows = client.execute(
            """
            WITH RECURSIVE
            edge(a, b) AS (
                SELECT src, dst FROM cat.links WHERE dst_exists
                UNION ALL
                SELECT dst, src FROM cat.links WHERE dst_exists
            ),
            reach(name, root) AS (
                SELECT DISTINCT a, a FROM edge
                UNION
                SELECT e.b, r.root FROM reach r JOIN edge e ON e.a = r.name
            )
            SELECT name, min(root) AS comp FROM reach GROUP BY name
            ORDER BY comp, name LIMIT 10
            """
        ).fetchall()
        ok(label_rec, rows)
    except Exception as e:
        fail(label_rec, e)

    # Test 3e: server-side VIEW for graph (workaround for multi-scan limit)
    label_view = "Degree centrality via server-side VIEW (workaround)"
    try:
        server_con.execute(
            """
            CREATE OR REPLACE VIEW centrality_view AS
            WITH edge(a, b) AS (
                SELECT src, dst FROM links WHERE dst_exists
                UNION ALL
                SELECT dst, src FROM links WHERE dst_exists
            ),
            deg AS (SELECT a AS name, count(*) AS degree FROM edge GROUP BY a)
            SELECT n.name, n.rel, coalesce(d.degree, 0) AS degree
            FROM files n LEFT JOIN deg d ON d.name = n.name
            ORDER BY degree DESC, n.name
            """
        )
        rows = client.execute(
            "SELECT name, degree FROM cat.centrality_view LIMIT 5"
        ).fetchall()
        ok(label_view, rows)
    except Exception as e:
        fail(label_view, e)

    # Cleanup
    server_con.execute("DELETE FROM links WHERE src LIKE '__probe_%'")


# ── test 4: concurrent read from server con ───────────────────────────────────

def test_concurrent_server_write(
    client: duckdb.DuckDBPyConnection,
    server_con: duckdb.DuckDBPyConnection,
) -> None:
    header("TEST 4 — Server-side write while client reads")

    label = "Client SELECT while server does INSERT + DELETE (MVCC)"
    try:
        # Read from client
        before = client.execute("SELECT count(*) FROM cat.files").fetchone()[0]

        # Server inserts a dummy row, client reads mid-flight snapshot
        server_con.execute(
            "INSERT INTO files(name, rel, folder, ext, is_binary, is_orphan, stale) "
            "VALUES ('__test__', '__test__', '/', '.txt', false, true, false)"
        )
        during = client.execute("SELECT count(*) FROM cat.files").fetchone()[0]

        # Server deletes the dummy row
        server_con.execute("DELETE FROM files WHERE name = '__test__'")
        after = client.execute("SELECT count(*) FROM cat.files").fetchone()[0]

        ok(label, f"before={before} during={during} after={after}")
        if during > before:
            ok("Client saw committed insert immediately (no isolation gap)")
        else:
            print("  ⚠  Client did NOT see the server's committed insert — check Quack isolation level")
    except Exception as e:
        fail(label, e)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    src = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/marcocassar/workspace/projects/quackspace/test-quackspace/.quack/quack.duckdb"
    )

    with tempfile.TemporaryDirectory() as tmp:
        catalog = Path(tmp) / "probe.duckdb"
        shutil.copy2(src, catalog)
        print(f"\nProbing catalog copy: {catalog}")
        print(f"(source: {src})")

        server_con = start_server(catalog)
        time.sleep(0.3)  # let the HTTP listener bind

        try:
            client = make_client()
            test_fts(client, server_con)
            test_vss(client, server_con)
            test_graph(client, server_con)
            test_concurrent_server_write(client, server_con)
            client.close()
        finally:
            stop_server(server_con)

    print(f"\n{'═' * 60}")
    print("  Probe complete.")
    print("═" * 60)


if __name__ == "__main__":
    main()
