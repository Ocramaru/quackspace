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
    tags(name, tag)                  -- one row per (file, tag)
    links(src, dst, dst_exists)      -- one row per wikilink edge
A DuckDB FTS index is built over files(name, description, body) for `match_bm25`.
`stale` is true when the file changed after its description was written.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb

from .core import Space

DB_NAME = "quack.duckdb"
SCHEMA_VERSION = 1


def db_path(space: Space) -> Path:
    return space.root / ".quack" / DB_NAME


def build(space: Space) -> dict:
    """Rebuild the catalog from scratch over the loaded space. Returns a
    summary. The space already carries effective metadata (authored .index.yaml
    overlaid on each file)."""
    path = db_path(space)
    if path.exists():
        path.unlink()  # rebuild clean; the files + .index.yaml are the truth

    names = set(space.by_name)
    inbound: dict[str, int] = defaultdict(int)
    for e in space.entries:
        for target in e.links:
            if target in names:
                inbound[target] += 1

    con = duckdb.connect(str(path))
    try:
        _create_schema(con)
        _write_metadata(con)
        for e in space.entries:
            con.execute(
                "INSERT INTO files VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    e.name,
                    e.rel,
                    e.folder,
                    e.ext,
                    e.name,
                    e.description,
                    ",".join(e.tags),
                    len(e.links),
                    inbound.get(e.name, 0),
                    inbound.get(e.name, 0) == 0 and len(e.links) == 0,
                    e.is_binary,
                    e.modified,
                    e.described_at,
                    e.stale,
                    e.body,
                ],
            )
            for tag in e.tags:
                con.execute("INSERT INTO tags VALUES (?, ?)", [e.name, tag])
            for dst in e.links:
                con.execute(
                    "INSERT INTO links VALUES (?, ?, ?)",
                    [e.name, dst, dst in names],
                )
        _build_fts(con)
        n_files = con.execute("SELECT count(*) FROM files").fetchone()[0]
        n_tags = con.execute("SELECT count(*) FROM tags").fetchone()[0]
        n_links = con.execute("SELECT count(*) FROM links").fetchone()[0]
    finally:
        con.close()

    return {"db": str(path), "files": n_files, "tags": n_tags, "links": n_links}


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


def connect(explicit_root: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the catalog read-only for querying. Caller closes it."""
    space = Space.load(explicit_root)
    path = db_path(space)
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


def query(sql: str, explicit_root: str | None = None) -> tuple[list[str], list[tuple]]:
    """Run a SQL query against the catalog. Returns (column_names, rows)."""
    con = connect(explicit_root)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, cur.fetchall()
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
    con = connect(explicit_root)
    try:
        placeholders = ",".join("?" for _ in names)
        rows = con.execute(
            f"""
            WITH RECURSIVE
            -- undirected edge view over existing notes only
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
                  AND w.name NOT IN ({placeholders})  -- a seed is not its own neighbour
            )
            SELECT name, rel, dist, seed FROM ranked WHERE rn = 1
            ORDER BY dist, name
            """,
            [*names, hops, *names],
        ).fetchall()
        return rows
    finally:
        con.close()


def fts_search(
    terms: str, explicit_root: str | None = None, limit: int = 10
) -> list[tuple[str, str, float]]:
    """BM25 full-text search. Returns [(rel, description, score), ...]."""
    con = connect(explicit_root)
    try:
        rows = con.execute(
            """
            SELECT rel, description, score FROM (
                SELECT rel, description,
                       fts_main_files.match_bm25(name, ?) AS score
                FROM files
            ) WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            [terms, limit],
        ).fetchall()
        return rows
    finally:
        con.close()
