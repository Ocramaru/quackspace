from __future__ import annotations

import quack.mcp_server as mcp_server


def test_mcp_get_file_truncates_content(indexed_mcp_space):
    root = indexed_mcp_space

    result = mcp_server.get_file("projects/note-0.md", char_limit=12)

    assert result["root"] == str(root)
    full = (root / "projects" / "note-0.md").read_text()
    assert result["content"] == full[:12]
    assert len(result["content"]) == 12
    assert result["content_length"] == len(full)
    assert result["content_limit"] == 12
    assert result["truncated"] is True


def test_mcp_sql_caps_rows(indexed_mcp_space):
    result = mcp_server.sql("SELECT rel FROM files ORDER BY rel", row_limit=2)

    assert result["columns"] == ["rel"]
    assert len(result["rows"]) == 2
    assert result["row_limit"] == 2
    assert result["truncated"] is True


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
        .replace("file_char_limit: 20000", "file_char_limit: 9")
        .replace("sql_row_limit: 50", "sql_row_limit: 2")
        .replace("central_limit: 10", "central_limit: 4")
    )

    limits = mcp_server.configure_limits_from_config(str(root))
    file_result = mcp_server.get_file("projects/note-0.md")
    sql_result = mcp_server.sql("SELECT rel FROM files ORDER BY rel")
    search_result = mcp_server.search("needle", expand=False)
    central_result = mcp_server.central()

    assert limits.search == 3
    assert file_result["content_limit"] == 9
    assert file_result["truncated"] is True
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
