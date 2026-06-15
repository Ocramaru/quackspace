"""The meta collection: one DuckDB catalog of all file metadata.

`quack reindex` rebuilds `.quack/quack.duckdb` from the files (+ the editable
.index.yaml store). It is a derived artifact, never the source of truth, so it
can be deleted and regenerated at any time. DuckDB is embedded (no server) and
gives real SQL plus BM25 full-text search over everything, the fast metadata
search `ls` can't do.

Schema:
    files(name, rel, folder, ext, title, description, tags_csv, n_links,
          n_inbound, is_orphan, is_binary, file_modified, described_at, stale,
          body)
    folders(folder, parent, description, n_files, diagram, described_at)
    tags(name, tag)                  -- one row per (file, tag)
    links(src, dst, dst_exists)      -- one row per wikilink edge
A DuckDB FTS index is built over files(name, description, body) for `match_bm25`.
`stale` is true when the file changed after its description was written.

The `folders` table mirrors the per-folder `.index.yaml` `directories:`
sections 1:1 (the direct subfolders of X are `WHERE parent = 'X'`, and the
root's are `WHERE parent = ''`). Type/tag rollups are NOT materialized; they
stay `GROUP BY` queries the YAML caches.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import duckdb

from .core import Space, find_root

DB_NAME = "quack.duckdb"
SCHEMA_VERSION = 2


def resolve_db(explicit_root: str | None = None) -> Path:
    """The catalog path for a root, resolved like ``find_root`` (walk up for the
    ``.quack/`` marker) — WITHOUT loading the whole space. Cheap; safe to call
    on every MCP tool invocation."""
    return find_root(explicit_root) / ".quack" / DB_NAME


def db_path(space: Space) -> Path:
    return space.root / ".quack" / DB_NAME


def _insert_rows(con: duckdb.DuckDBPyConnection, table: str, rows: list[tuple]) -> None:
    """Bulk-insert ``rows`` (list of column-ordered tuples) into *table*.

    Uses a columnar Arrow insert — DuckDB ingests an Arrow table zero-copy, far
    faster than row-by-row ``executemany`` (which binds every value). Falls back
    to ``executemany`` if PyArrow isn't importable. Caller wraps in a txn."""
    if not rows:
        return
    try:
        import pyarrow as pa
    except ImportError:
        placeholders = "(" + ",".join("?" * len(rows[0])) + ")"
        con.executemany(f"INSERT INTO {table} VALUES {placeholders}", rows)
        return
    arrow_tbl = pa.table({f"c{i}": pa.array(col) for i, col in enumerate(zip(*rows))})
    con.register("_bulk_arrow", arrow_tbl)
    try:
        con.execute(f"INSERT INTO {table} SELECT * FROM _bulk_arrow")
    finally:
        con.unregister("_bulk_arrow")


def build(space: Space, folder_infos: "dict | None" = None) -> dict:
    """Rebuild the catalog from scratch over the loaded space. Returns a
    summary. The space already carries effective metadata (authored .index.yaml
    overlaid on each file). *folder_infos* is the shared folder resolver's
    output; it is resolved here when not supplied so the ``folders`` table
    always matches the per-folder indexes and ``map.yaml``."""
    if folder_infos is None:
        from . import folders as _folders

        folder_infos = _folders.resolve_folders(space)

    path = db_path(space)
    # Close any cached read-only connection first: DuckDB won't open a
    # read-write connection while a read-only one to the same file is live.
    invalidate(path)
    if path.exists():
        path.unlink()  # rebuild clean; the files + .index.yaml are the truth

    names = set(space.by_name)
    inbound: dict[str, int] = defaultdict(int)
    for e in space.entries:
        for target in e.links:
            if target in names:
                inbound[target] += 1

    file_rows = []
    tag_rows = []
    link_rows = []
    for e in space.entries:
        n_in = inbound.get(e.name, 0)
        file_rows.append((
            e.name, e.rel, e.folder, e.ext, e.name,
            e.description, ",".join(e.tags),
            len(e.links), n_in, n_in == 0 and len(e.links) == 0,
            e.is_binary, e.modified, e.described_at, e.stale, e.body,
        ))
        tag_rows.extend((e.name, tag) for tag in e.tags)
        link_rows.extend((e.name, dst, dst in names) for dst in e.links)

    # One row per folder (excluding the root); parent "" means the root.
    folder_rows = [
        (i.rel, i.parent, i.description, i.n_files, i.diagram, i.described_at)
        for i in folder_infos.values()
        if not i.is_root
    ]

    con = duckdb.connect(str(path))
    try:
        _create_schema(con)
        _write_metadata(con)
        # One transaction for all inserts: DuckDB auto-commits per statement, so
        # row-by-row executemany without this commits N times and is ~8x slower.
        con.execute("BEGIN TRANSACTION")
        _insert_rows(con, "files", file_rows)
        _insert_rows(con, "folders", folder_rows)
        _insert_rows(con, "tags", tag_rows)
        _insert_rows(con, "links", link_rows)
        con.execute("COMMIT")
        _build_fts(con)
        n_files = con.execute("SELECT count(*) FROM files").fetchone()[0]
        n_folders = con.execute("SELECT count(*) FROM folders").fetchone()[0]
        n_tags = con.execute("SELECT count(*) FROM tags").fetchone()[0]
        n_links = con.execute("SELECT count(*) FROM links").fetchone()[0]
    finally:
        con.close()

    invalidate(path)  # any cached read-only connection now points to a stale file
    return {
        "db": str(path),
        "files": n_files,
        "folders": n_folders,
        "tags": n_tags,
        "links": n_links,
    }


def update_light(space: Space, folder_infos: "dict | None" = None) -> dict:
    """Incremental in-place update for changes that don't touch the full-text
    surface: file tags / described_at / stale / mtime, the tags table, and the
    folders table. Bodies, descriptions, names, links, inbound counts, and the
    FTS index are left untouched — so the existing BM25 index stays valid and is
    NOT rebuilt. Caller guarantees no file was added/removed and no
    name/body/description changed (see ``indexer._compute_dirty``)."""
    if folder_infos is None:
        from . import folders as _folders

        folder_infos = _folders.resolve_folders(space)

    path = db_path(space)
    invalidate(path)  # free any cached read-only connection before writing
    con = duckdb.connect(str(path))
    try:
        stored = {
            rel: (tags_csv or "", da or "", fm or "")
            for rel, tags_csv, da, fm in con.execute(
                "SELECT rel, tags_csv, described_at, file_modified FROM files"
            ).fetchall()
        }
        # One transaction for all writes (see build(): per-statement commits are
        # ~8x slower than batching them).
        con.execute("BEGIN TRANSACTION")
        for e in space.entries:
            current = (",".join(e.tags), e.described_at, e.modified)
            if stored.get(e.rel) != current:
                con.execute(
                    "UPDATE files SET tags_csv = ?, described_at = ?, "
                    "file_modified = ?, stale = ? WHERE rel = ?",
                    [current[0], e.described_at, e.modified, e.stale, e.rel],
                )

        # tags and folders aren't full-text indexed, so rebuilding them wholesale
        # is cheap and keeps the code simple.
        con.execute("DELETE FROM tags;")
        tag_rows = [(e.name, tag) for e in space.entries for tag in e.tags]
        _insert_rows(con, "tags", tag_rows)

        con.execute("DELETE FROM folders;")
        folder_rows = [
            (i.rel, i.parent, i.description, i.n_files, i.diagram, i.described_at)
            for i in folder_infos.values()
            if not i.is_root
        ]
        _insert_rows(con, "folders", folder_rows)
        con.execute("COMMIT")

        n_files = con.execute("SELECT count(*) FROM files").fetchone()[0]
        n_folders = con.execute("SELECT count(*) FROM folders").fetchone()[0]
        n_tags = con.execute("SELECT count(*) FROM tags").fetchone()[0]
        n_links = con.execute("SELECT count(*) FROM links").fetchone()[0]
    finally:
        con.close()

    return {
        "db": str(path),
        "files": n_files,
        "folders": n_folders,
        "tags": n_tags,
        "links": n_links,
    }


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR);
        CREATE TABLE files (
            name        VARCHAR,
            rel         VARCHAR,
            folder      VARCHAR,
            ext         VARCHAR,
            title       VARCHAR,
            description VARCHAR,
            tags_csv    VARCHAR,
            n_links     INTEGER,
            n_inbound   INTEGER,
            is_orphan   BOOLEAN,
            is_binary   BOOLEAN,
            file_modified VARCHAR,
            described_at  VARCHAR,
            stale         BOOLEAN,
            body        VARCHAR
        );
        CREATE TABLE folders (
            folder      VARCHAR,
            parent      VARCHAR,
            description VARCHAR,
            n_files     INTEGER,
            diagram     VARCHAR,
            described_at VARCHAR
        );
        CREATE TABLE tags  (name VARCHAR, tag VARCHAR);
        CREATE TABLE links (src VARCHAR, dst VARCHAR, dst_exists BOOLEAN);
        """
    )


def _build_fts(con: duckdb.DuckDBPyConnection) -> None:
    """Create the BM25 full-text index over the searchable note fields."""
    con.execute("INSTALL fts; LOAD fts;")
    con.execute(
        "PRAGMA create_fts_index('files', 'name', 'name', 'description', 'body', "
        "overwrite=1);"
    )


def _write_metadata(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("INSERT INTO metadata VALUES ('schema_version', ?)", [str(SCHEMA_VERSION)])
    con.execute(
        "INSERT INTO metadata VALUES ('built_at', ?)",
        [datetime.now().isoformat(timespec="seconds")],
    )
    # Nanosecond build timestamp for the stat-only reindex no-op check: it
    # compares marker-file mtimes against this, and second precision would miss
    # an authored edit made in the same second as the build.
    con.execute(
        "INSERT INTO metadata VALUES ('built_at_ns', ?)", [str(time.time_ns())]
    )


def _validate_schema(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    try:
        row = con.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
    except duckdb.Error as e:
        raise RuntimeError(
            f"Catalog at {path} is missing schema metadata. Run `quack reindex` "
            "to rebuild it from source files and .index.yaml metadata."
        ) from e
    version = int(row[0]) if row else 0
    if version != SCHEMA_VERSION:
        raise RuntimeError(
            f"Catalog at {path} has schema version {version}, but quack expects "
            f"{SCHEMA_VERSION}. Run `quack reindex` to rebuild it."
        )


def recover_message(path: Path) -> str:
    return (
        f"Catalog at {path} could not be opened. It is a derived artifact; "
        "delete it or run `quack reindex` to rebuild it from source files and "
        ".index.yaml metadata."
    )


def connect_path(path: Path) -> duckdb.DuckDBPyConnection:
    """Open a specific catalog file read-only. Validates schema. Caller closes."""
    if not path.exists():
        raise RuntimeError(
            f"No catalog at {path}. Run `quack reindex` to build it."
        )
    try:
        con = duckdb.connect(str(path), read_only=True)
        _validate_schema(con, path)
        return con
    except RuntimeError:
        raise
    except duckdb.Error as e:
        raise RuntimeError(recover_message(path)) from e


def connect(explicit_root: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the catalog read-only for querying. Caller closes it. Resolves the
    path without loading the whole space (no filesystem walk per query)."""
    return connect_path(resolve_db(explicit_root))


# ---------------------------------------------------------------------------
# Shared connection cache — for long-lived processes (the MCP server), so the
# catalog file is opened once and reused across many tool calls instead of
# reopened every time. Automatically reopened when the catalog changes (e.g.
# after `reindex`), detected by the file's stat signature. For per-query thread
# safety, call `.cursor()` on the returned connection.
# ---------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()
_SHARED: dict[str, tuple[duckdb.DuckDBPyConnection, tuple]] = {}


def _stat_signature(path: Path) -> tuple:
    st = path.stat()
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def shared_connection(path: Path) -> duckdb.DuckDBPyConnection:
    """A cached, read-only connection reused across calls in one process. The
    caller must NOT close it (use ``con.cursor()`` for an isolated, thread-safe
    query handle). Reopens automatically when the catalog file changes."""
    key = str(path)
    sig = _stat_signature(path)  # raises if the file is gone → caller falls back
    with _CACHE_LOCK:
        cached = _SHARED.get(key)
        if cached is not None:
            con, cached_sig = cached
            if cached_sig == sig:
                return con
            try:
                con.close()
            except Exception:
                pass
        con = connect_path(path)
        _SHARED[key] = (con, sig)
        return con


def invalidate(path: Path | None = None) -> None:
    """Drop cached connection(s) — call after rewriting the catalog so the next
    reader reopens the fresh file. ``None`` clears the whole cache."""
    with _CACHE_LOCK:
        keys = [str(path)] if path is not None else list(_SHARED)
        for k in keys:
            entry = _SHARED.pop(k, None)
            if entry is not None:
                try:
                    entry[0].close()
                except Exception:
                    pass


def query(sql: str, explicit_root: str | None = None) -> tuple[list[str], list[tuple]]:
    """Run a SQL query against the catalog. Returns (column_names, rows)."""
    con = connect(explicit_root)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, cur.fetchall()
    finally:
        con.close()


def _neighbours_query(
    con: duckdb.DuckDBPyConnection, names: list[str], hops: int = 1
) -> list[tuple[str, str, int, str]]:
    placeholders = ",".join("?" for _ in names)
    if hops <= 1:
        # The common case (search expansion): direct 1-hop neighbours need no
        # recursion — a plain join over the bidirectional edge set is much
        # cheaper than spinning up a recursive CTE.
        return con.execute(
            f"""
            WITH edge(a, b) AS (
                SELECT src, dst FROM links WHERE dst_exists
                UNION ALL
                SELECT dst, src FROM links WHERE dst_exists
            ),
            ranked AS (
                SELECT e.b AS name, n.rel, 1 AS dist, e.a AS seed,
                       row_number() OVER (PARTITION BY e.b ORDER BY e.a) AS rn
                FROM edge e JOIN files n ON n.name = e.b
                WHERE e.a IN ({placeholders})
                  AND e.b NOT IN ({placeholders})
            )
            SELECT name, rel, dist, seed FROM ranked WHERE rn = 1
            ORDER BY name
            """,
            [*names, *names],
        ).fetchall()
    return con.execute(
        f"""
        WITH RECURSIVE
        edge(a, b) AS (
            SELECT src, dst FROM links WHERE dst_exists
            UNION ALL
            SELECT dst, src FROM links WHERE dst_exists
        ),
        walk(name, dist, seed) AS (
            SELECT name, 0, name FROM files WHERE name IN ({placeholders})
            UNION
            SELECT e.b, w.dist + 1, w.seed
            FROM walk w JOIN edge e ON e.a = w.name
            WHERE w.dist < ?
        ),
        ranked AS (
            SELECT w.name, n.rel, w.dist, w.seed,
                   row_number() OVER (PARTITION BY w.name ORDER BY w.dist) AS rn
            FROM walk w JOIN files n ON n.name = w.name
            WHERE w.dist > 0
              AND w.name NOT IN ({placeholders})
        )
        SELECT name, rel, dist, seed FROM ranked WHERE rn = 1
        ORDER BY dist, name
        """,
        [*names, hops, *names],
    ).fetchall()


def tag_neighbours_on(
    con: duckdb.DuckDBPyConnection,
    names: list[str],
    limit: int = 10,
    max_tag_freq: int = 25,
) -> list[tuple[str, str, int]]:
    """Files related to the seeds by *shared tags*, ranked by how many tags they
    share. Returns [(name, rel, shared_tag_count), ...], excluding the seeds.

    High-frequency tags are skipped (a tag on more than *max_tag_freq* files —
    e.g. a generic recognition tag like ``python`` on every source file — links
    everything to everything, so it's noise, not a relationship). This makes tag
    relatedness useful for code repos that have few or no [[wikilinks]]."""
    if not names:
        return []
    ph = ",".join("?" for _ in names)
    return con.execute(
        f"""
        WITH common AS (
            SELECT tag FROM tags GROUP BY tag HAVING count(*) > ?
        ),
        seed_tags AS (
            SELECT DISTINCT tag FROM tags
            WHERE name IN ({ph}) AND tag NOT IN (SELECT tag FROM common)
        ),
        shared AS (
            SELECT t.name, count(*) AS shared
            FROM tags t JOIN seed_tags st ON t.tag = st.tag
            WHERE t.name NOT IN ({ph})
            GROUP BY t.name
        )
        SELECT s.name, f.rel, s.shared
        FROM shared s JOIN files f ON f.name = s.name
        ORDER BY s.shared DESC, s.name
        LIMIT ?
        """,
        [max_tag_freq, *names, *names, limit],
    ).fetchall()


def neighbours_path(
    db: Path, names: list[str], hops: int = 1
) -> list[tuple[str, str, int, str]]:
    """Graph traversal against a known catalog path. Returns [(name, rel, distance, via_seed)]."""
    if not names:
        return []
    con = connect_path(db)
    try:
        return _neighbours_query(con, names, hops)
    finally:
        con.close()


def neighbours(
    names: list[str], explicit_root: str | None = None, hops: int = 1
) -> list[tuple[str, str, int, str]]:
    """Graph traversal in SQL: notes within `hops` of any seed name, in either
    link direction. Returns [(name, rel, distance, via_seed), ...], excluding
    the seeds, where via_seed is one seed that reaches the note at min distance.

    Uses a recursive CTE so only the relevant subgraph is materialized, the
    whole point of keeping the graph in DuckDB instead of a flat file.
    """
    if not names:
        return []
    return neighbours_path(resolve_db(explicit_root), names, hops)


def _fts_query(
    con: duckdb.DuckDBPyConnection, terms: str, limit: int
) -> list[tuple[str, str, str, float]]:
    return con.execute(
        """
        SELECT rel, name, description, score FROM (
            SELECT rel, name, description,
                   fts_main_files.match_bm25(name, ?) AS score
            FROM files
        ) WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        [terms, limit],
    ).fetchall()


def fts_search_path(
    db: Path, terms: str, limit: int = 10
) -> list[tuple[str, str, float]]:
    """BM25 full-text search against a known catalog path. Returns [(rel, description, score)]."""
    con = connect_path(db)
    try:
        return [(rel, desc, score) for rel, _name, desc, score in _fts_query(con, terms, limit)]
    finally:
        con.close()


def fts_search(
    terms: str, explicit_root: str | None = None, limit: int = 10
) -> list[tuple[str, str, float]]:
    """BM25 full-text search. Returns [(rel, description, score), ...]."""
    return fts_search_path(resolve_db(explicit_root), terms, limit)


def list_folders_path(db: Path) -> list[tuple[str, str, str]]:
    """All folder rows from a known catalog path: [(folder, parent, description)]."""
    con = connect_path(db)
    try:
        return con.execute(
            "SELECT folder, parent, description FROM folders"
        ).fetchall()
    finally:
        con.close()


def search_docs_path(db: Path) -> list[tuple[str, str, str, str]]:
    """The short fields structural search needs, straight from the catalog (no
    filesystem walk, and crucially NOT the body column — body matching is the
    FTS tier's job): [(name, rel, description, tags_csv), ...]."""
    con = connect_path(db)
    try:
        return docs_on(con)
    finally:
        con.close()


# Connection-based variants, so one search can open the catalog once and run the
# docs/FTS/graph queries on a single connection instead of reopening it per tier.
def docs_on(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str, str]]:
    return con.execute("SELECT name, rel, description, tags_csv FROM files").fetchall()


def structural_candidates_on(
    con: duckdb.DuckDBPyConnection, terms: list[str]
) -> list[tuple[str, str, str]]:
    """Only the files whose short fields (name/tags/description) contain at least
    one term — matched in SQL so we don't pull every row into Python. Returns
    [(name, description, tags_csv), ...]. ``terms`` are already lowercased;
    ``contains`` matches literal substrings (matching the Python scorer)."""
    if not terms:
        return []
    clauses = []
    params: list[str] = []
    for t in terms:
        clauses.append(
            "(contains(lower(name), ?) OR contains(lower(tags_csv), ?) "
            "OR contains(lower(description), ?))"
        )
        params += [t, t, t]
    where = " OR ".join(clauses)
    return con.execute(
        f"SELECT name, description, tags_csv FROM files WHERE {where}", params
    ).fetchall()


def docs_for_names_on(
    con: duckdb.DuckDBPyConnection, names: list[str]
) -> list[tuple[str, str, str, str]]:
    """Full short rows for a bounded set of result names (so we fetch metadata
    only for what we'll actually return). Returns [(name, rel, description,
    tags_csv), ...]."""
    if not names:
        return []
    ph = ",".join("?" for _ in names)
    return con.execute(
        f"SELECT name, rel, description, tags_csv FROM files WHERE name IN ({ph})",
        list(names),
    ).fetchall()


def fts_on(
    con: duckdb.DuckDBPyConnection, terms: str, limit: int = 10
) -> list[tuple[str, str, str, float]]:
    """BM25 on an open connection. Returns [(rel, name, description, score), ...]."""
    return _fts_query(con, terms, limit)


def neighbours_on(
    con: duckdb.DuckDBPyConnection, names: list[str], hops: int = 1
) -> list[tuple[str, str, int, str]]:
    if not names:
        return []
    return _neighbours_query(con, names, hops)
