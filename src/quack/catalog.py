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

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import duckdb

from .core import Space

DB_NAME = "quack.duckdb"
SCHEMA_VERSION = 2


def db_path(space: Space) -> Path:
    return space.root / ".quack" / DB_NAME


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
        if file_rows:
            con.executemany("INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", file_rows)
        if folder_rows:
            con.executemany("INSERT INTO folders VALUES (?,?,?,?,?,?)", folder_rows)
        if tag_rows:
            con.executemany("INSERT INTO tags VALUES (?,?)", tag_rows)
        if link_rows:
            con.executemany("INSERT INTO links VALUES (?,?,?)", link_rows)
        _build_fts(con)
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
    con = duckdb.connect(str(path))
    try:
        stored = {
            rel: (tags_csv or "", da or "", fm or "")
            for rel, tags_csv, da, fm in con.execute(
                "SELECT rel, tags_csv, described_at, file_modified FROM files"
            ).fetchall()
        }
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
        if tag_rows:
            con.executemany("INSERT INTO tags VALUES (?,?)", tag_rows)

        con.execute("DELETE FROM folders;")
        folder_rows = [
            (i.rel, i.parent, i.description, i.n_files, i.diagram, i.described_at)
            for i in folder_infos.values()
            if not i.is_root
        ]
        if folder_rows:
            con.executemany("INSERT INTO folders VALUES (?,?,?,?,?,?)", folder_rows)

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
    """Open the catalog read-only for querying. Caller closes it."""
    space = Space.load(explicit_root)
    return connect_path(db_path(space))


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
    space = Space.load(explicit_root)
    return neighbours_path(db_path(space), names, hops)


def _fts_query(
    con: duckdb.DuckDBPyConnection, terms: str, limit: int
) -> list[tuple[str, str, float]]:
    return con.execute(
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


def fts_search_path(
    db: Path, terms: str, limit: int = 10
) -> list[tuple[str, str, float]]:
    """BM25 full-text search against a known catalog path. Returns [(rel, description, score)]."""
    con = connect_path(db)
    try:
        return _fts_query(con, terms, limit)
    finally:
        con.close()


def fts_search(
    terms: str, explicit_root: str | None = None, limit: int = 10
) -> list[tuple[str, str, float]]:
    """BM25 full-text search. Returns [(rel, description, score), ...]."""
    space = Space.load(explicit_root)
    return fts_search_path(db_path(space), terms, limit)


def list_folders_path(db: Path) -> list[tuple[str, str, str]]:
    """All folder rows from a known catalog path: [(folder, parent, description)]."""
    con = connect_path(db)
    try:
        return con.execute(
            "SELECT folder, parent, description FROM folders"
        ).fetchall()
    finally:
        con.close()
