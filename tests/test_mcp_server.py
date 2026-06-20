from __future__ import annotations

import duckdb

import quack.mcp_server as mcp_server
from quack import catalog
from quack.indexer import reindex
from quack.scaffold import scaffold_root


def test_mcp_file_meta_returns_metadata(indexed_mcp_space):
    root = indexed_mcp_space

    result = mcp_server.file_meta("projects/note-0.md")

    assert result["root"] == str(root)
    assert result["path"] == "projects/note-0.md"
    assert result["absolute_path"] == str(root / "projects" / "note-0.md")
    assert "name" in result
    assert "description" in result
    assert "tags" in result
    assert "links" in result
    assert "stale" in result
    assert "content" not in result


def test_mcp_file_meta_not_found(indexed_mcp_space):
    result = mcp_server.file_meta("nonexistent_file.md")

    assert "error" in result
    assert "next_steps" in result


def test_mcp_file_meta_includes_next_steps_with_absolute_path(indexed_mcp_space):
    root = indexed_mcp_space

    result = mcp_server.file_meta("projects/note-0.md")

    assert "next_steps" in result
    assert str(root / "projects" / "note-0.md") in result["next_steps"]


def test_mcp_sql_caps_rows(indexed_mcp_space):
    result = mcp_server.sql("SELECT rel FROM files ORDER BY rel", row_limit=2)

    assert result["columns"] == ["rel"]
    assert len(result["rows"]) == 2
    assert result["row_limit"] == 2
    assert result["truncated"] is True


def test_mcp_map_defaults_to_top_level_and_can_descend(tmp_path, monkeypatch):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "alpha" / "nested").mkdir(parents=True)
    (root / "alpha" / "note.md").write_text("# alpha\n")
    (root / "alpha" / "nested" / "deep.md").write_text("# deep\n")
    (root / "beta").mkdir()
    (root / "beta" / "note.md").write_text("# beta\n")
    reindex(str(root))
    monkeypatch.chdir(root)
    mcp_server.configure_root(str(root))
    mcp_server.configure_limits()

    top = mcp_server.map()
    nested = mcp_server.map(parent="alpha")

    # Top level renders a tree of leaf folder names with file counts.
    assert "alpha (" in top["tree"]
    assert "beta (" in top["tree"]
    assert "nested" not in top["tree"]  # nested folder not surfaced at the top
    # Descending one level shows the child's leaf name.
    assert "nested (" in nested["tree"]
    assert nested["parent"] == "alpha"
    assert "parent = '<folder>'" in top["next_steps"]


def test_mcp_map_reports_truncation(tmp_path, monkeypatch):
    root = scaffold_root(str(tmp_path / "space"))
    for i in range(3):
        folder = root / f"dir-{i}"
        folder.mkdir()
        (folder / "note.md").write_text(f"# dir {i}\n")
    reindex(str(root))
    monkeypatch.chdir(root)
    mcp_server.configure_root(str(root))
    mcp_server.configure_limits()

    result = mcp_server.map(limit=1)

    assert result["limit"] == 1
    assert result["truncated"] is True
    assert result["child_count"] > 1
    # Only `limit` folder entries are rendered, plus a "more not shown" line.
    folder_lines = [l for l in result["tree"].splitlines() if "dir-" in l]
    assert len(folder_lines) == 1
    assert "more folder(s) not shown" in result["tree"]
    assert "Only 1" in result["next_steps"]


def test_mcp_search_and_central_clamp_limits(indexed_mcp_space):
    search_result = mcp_server.search("needle", limit=999, expand=False)
    central_result = mcp_server.central(limit=999)

    assert search_result["limit"] == mcp_server.MAX_SEARCH_LIMIT
    assert len(search_result["hits"]) <= mcp_server.MAX_SEARCH_LIMIT
    assert central_result["limit"] == mcp_server.MAX_CENTRAL_LIMIT


def test_mcp_uses_config_defaults_when_tool_args_are_omitted(indexed_mcp_space):
    root = indexed_mcp_space
    config = root / ".quack" / "config.yaml"
    config.write_text(
        config.read_text()
        .replace("search_limit: 10", "search_limit: 3")
        .replace("sql_row_limit: 50", "sql_row_limit: 2")
        .replace("central_limit: 10", "central_limit: 4")
    )

    limits = mcp_server.configure_limits_from_config(str(root))
    sql_result = mcp_server.sql("SELECT rel FROM files ORDER BY rel")
    search_result = mcp_server.search("needle", expand=False)
    central_result = mcp_server.central()

    assert limits.search == 3
    assert sql_result["row_limit"] == 2
    assert search_result["limit"] == 3
    assert central_result["limit"] == 4


def test_mcp_flags_override_config_defaults(indexed_mcp_space):
    root = indexed_mcp_space
    config = root / ".quack" / "config.yaml"
    config.write_text(config.read_text().replace("search_limit: 10", "search_limit: 3"))

    limits = mcp_server.configure_limits_from_config(str(root), search_limit=5)

    assert limits.search == 5
    assert mcp_server.search("needle", expand=False)["limit"] == 5


# ---------------------------------------------------------------------------
# next_steps guidance fields (MAR-159 / GH#6)
# ---------------------------------------------------------------------------

def test_search_with_hits_includes_next_steps(indexed_mcp_space):
    result = mcp_server.search("needle", expand=False)
    assert result["hits"]
    assert "next_steps" in result
    assert "file_meta" in result["next_steps"]


def test_search_no_hits_includes_next_steps(indexed_mcp_space):
    result = mcp_server.search("xyzzy_no_match_ever", expand=False)
    assert result["hits"] == []
    assert "next_steps" in result
    assert "reindex" in result["next_steps"]


def test_search_no_semantic_tier_mentions_embed(indexed_mcp_space):
    result = mcp_server.search("needle", expand=False)
    tiers_seen = {t for h in result["hits"] for t in h["tiers"]}
    if "semantic" not in tiers_seen:
        assert "embed" in result["next_steps"]


def test_file_meta_includes_next_steps_with_describe_guidance(indexed_mcp_space):
    result = mcp_server.file_meta("projects/note-0.md")
    assert "next_steps" in result
    assert "describe" in result["next_steps"]


def test_describe_includes_next_steps(indexed_mcp_space):
    result = mcp_server.describe("projects/note-0.md", "A test note", ["test"])
    assert "next_steps" in result
    assert "reindex" in result["next_steps"]


def test_reindex_includes_next_steps_and_catalog(indexed_mcp_space):
    result = mcp_server.reindex()
    assert "next_steps" in result
    assert "catalog" in result


def test_central_excludes_opaque_dir_hubs(indexed_mcp_space):
    root = indexed_mcp_space
    db = catalog.resolve_db(str(root))
    catalog.invalidate(db)
    con = duckdb.connect(str(db))
    try:
        file_rows = [
            ("vendor", "libs/site-packages/vendor.md", "libs/site-packages", "md"),
            ("dep1", "libs/site-packages/dep1.md", "libs/site-packages", "md"),
            ("dep2", "libs/site-packages/dep2.md", "libs/site-packages", "md"),
            ("real", "projects/real.md", "projects", "md"),
            ("real2", "projects/real2.md", "projects", "md"),
        ]
        for name, rel, folder, ext in file_rows:
            con.execute(
                """
                INSERT INTO files VALUES (
                    ?, ?, ?, ?, ?, '', '', 0, 0, false, false,
                    '2024-01-01T00:00:00', '', false, '', ''
                )
                """,
                [name, rel, folder, ext, name],
            )
        con.executemany(
            "INSERT INTO links VALUES (?, ?, true)",
            [("vendor", "dep1"), ("vendor", "dep2"), ("dep1", "vendor"), ("real", "real2")],
        )
    finally:
        con.close()
    catalog.invalidate(db)

    result = mcp_server.central(limit=5)

    assert "site-packages" in result["excluded_opaque_dirs"]
    assert "vendor" not in {h["name"] for h in result["hubs"]}
    assert "real" in {h["name"] for h in result["hubs"]}


# ---------------------------------------------------------------------------
# explain() tool (MAR-159 / GH#6)
# ---------------------------------------------------------------------------

def test_explain_returns_architecture_reference(indexed_mcp_space):
    result = mcp_server.explain()
    assert "root" in result
    assert "data_flow" in result
    assert "search_tiers" in result
    assert "structural" in result["search_tiers"]
    assert "fts" in result["search_tiers"]
    assert "semantic" in result["search_tiers"]
    assert "catalog_schema" in result
    assert "files" in result["catalog_schema"]
    assert "annotation_workflow" in result
    assert "large_root_guidance" in result
    assert ".quackignore" in result["large_root_guidance"]
