from __future__ import annotations

import json
import yaml

from conftest import arg_value, write_yaml
from quack import catalog, diagram, graph, index_store
from quack.core import Space
from quack.doctor import diagnose, format_report
from quack.generate import _parse_meta, record
from quack.indexer import reindex
from quack.scaffold import new_note, scaffold_root
from quack.search import search


def test_scaffold_new_note_reindex_catalog_search_graph_doctor_and_diagram(sample_space):
    root = sample_space

    created = new_note("Gamma Note", folder="projects", description="Gamma desc", tags=["gamma"], explicit_root=str(root))
    assert created.parent == root / "projects"
    assert created.name == "gamma-note.md"
    assert created.read_text().startswith("---")

    summary = reindex(str(root))
    assert summary["files"] == 4
    assert (root / ".quack" / "map.yaml").exists()
    assert (root / ".quack" / "quack.duckdb").exists()
    assert (root / "projects" / ".index.yaml").exists()

    cols, rows = catalog.query("SELECT rel, description FROM files ORDER BY rel", explicit_root=str(root))
    assert cols == ["rel", "description"]
    rels = [row[0] for row in rows]
    assert "projects/alpha.md" in rels
    assert "projects/gamma-note.md" in rels

    hits = search("regex", explicit_root=str(root), expand=False)
    assert hits and hits[0].entry.rel == "projects/alpha.md"

    assert graph.shortest_path("alpha", "beta", explicit_root=str(root)) == ["alpha", "beta"]
    hubs = graph.centrality(explicit_root=str(root), limit=2)
    assert hubs[0][0] in {"alpha", "beta"}
    assert graph.components(explicit_root=str(root)) == [["alpha", "beta"]]

    report = diagnose(str(root))
    assert not report.ok
    assert ("projects/alpha.md", "missing") in report.broken_links
    assert "broken wikilink" in format_report(report)

    result = diagram.diagram(str(root))
    assert result["folder_diagrams"] == 1
    assert (root / ".quack" / "diagram.md").exists()
    assert (root / "projects" / "_diagrams.md").exists()


def test_search_reports_tier_progress(sample_space):
    root = sample_space
    reindex(str(root))
    calls: list[tuple[int, int, str]] = []

    hits = search(
        "regex",
        explicit_root=str(root),
        expand=False,
        progress=lambda done, total, message: calls.append((done, total, message)),
    )

    assert hits and hits[0].entry.rel == "projects/alpha.md"
    messages = [message for _done, _total, message in calls]
    assert "Opening catalog" in messages
    assert "Searching structure" in messages
    assert "Searching full text" in messages
    assert "Searching embeddings" in messages
    assert "Search complete" in messages
    assert calls[-1] == (6, 6, "Search complete")


def test_reindex_fast_noop_reports_file_progress(sample_space):
    root = sample_space
    reindex(str(root))
    calls: list[tuple[int, int, str]] = []

    summary = reindex(
        str(root),
        progress=lambda done, total, message: calls.append((done, total, message)),
    )

    assert summary["folder_indexes"] == 0
    assert summary["files"] == 3
    assert calls[0] == (0, None, "Waddling through files: 0")
    assert (3, None, "Waddled 3 file(s)") in calls
    assert (0, 3, "Checking files") in calls
    assert calls[-1][0] == 3
    assert calls[-1][1] == 3
    assert calls[-1][2].startswith("Checking ")


def test_reindex_reports_post_scan_phases(sample_space):
    root = sample_space
    calls: list[tuple[int, int, str]] = []

    reindex(
        str(root),
        progress=lambda done, total, message: calls.append((done, total, message)),
    )

    assert (0, None, "Waddling through files: 0") in calls
    assert (3, None, "Waddled 3 file(s)") in calls
    assert (0, 1, "Loading file contents") in calls
    assert (0, 1, "Applying metadata") in calls
    assert (1, 1, "Applied metadata") in calls
    assert (0, -1, "Quacking at metadata") in calls
    assert (1, 1, "Quacked at metadata") in calls
    assert (0, -1, "Checking catalog changes") in calls
    assert (1, 1, "Checked catalog changes") in calls


def test_index_store_preserves_authored_fields_and_record_updates_metadata(sample_space):
    root = sample_space
    reindex(str(root))

    rel = record(str(root), "beta", "Beta manual description", ["manual", "beta"] )
    assert rel == "projects/beta.md"
    reindex(str(root))

    authored = index_store.load_authored(root / "projects")
    assert authored["beta.md"]["description"] == "Beta manual description"
    assert authored["beta.md"]["tags"] == ["manual", "beta"]

    space = Space.load(str(root))
    beta = next(e for e in space.entries if e.rel == "projects/beta.md")
    assert beta.description == "Beta manual description"
    assert beta.tags == ["manual", "beta"]


def test_reindex_survives_non_utf8_markdown(sample_space):
    """A markdown file in a legacy encoding (e.g. Windows-1252 smart quotes)
    must not crash reindex; it is indexed with leniently decoded text."""
    root = sample_space
    # 0x93/0x94 are cp1252 curly quotes, invalid as UTF-8.
    (root / "projects" / "legacy.md").write_bytes(
        b"# Legacy\n\nHe said \x93hello\x94 to everyone.\n"
    )
    reindex(str(root))  # must not raise

    _, rows = catalog.query(
        "SELECT rel, body FROM files WHERE rel = 'projects/legacy.md'",
        explicit_root=str(root),
    )
    assert rows and rows[0][0] == "projects/legacy.md"
    assert "hello" in rows[0][1]  # surrounding text preserved


def test_reindex_survives_malformed_frontmatter(sample_space):
    """A markdown file with broken YAML frontmatter must not crash reindex; the
    raw text is kept as the body and the file is still indexed."""
    root = sample_space
    (root / "projects" / "broken.md").write_text(
        "---\nthis: : not: valid: yaml\n  [unclosed\n---\n# Broken\n\nreal content\n"
    )
    reindex(str(root))  # must not raise

    _, rows = catalog.query(
        "SELECT rel, body FROM files WHERE rel = 'projects/broken.md'",
        explicit_root=str(root),
    )
    assert rows and rows[0][0] == "projects/broken.md"
    assert "real content" in rows[0][1]


def test_reindex_can_disable_body_storage(sample_space):
    root = sample_space
    token = "privatebodytoken"
    (root / "projects" / "private.md").write_text(f"# Private\n\n{token}\n")
    reindex(str(root))

    _, rows = catalog.query(
        "SELECT body FROM files WHERE rel = 'projects/private.md'",
        explicit_root=str(root),
    )
    assert token in rows[0][0]
    assert search(token, explicit_root=str(root), expand=False)

    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    config["index"] = {"store_body": False}
    write_yaml(root / ".quack" / "config.yaml", config)

    summary = reindex(str(root))

    assert summary["catalog"] == "full"
    _, rows = catalog.query(
        "SELECT body FROM files WHERE rel = 'projects/private.md'",
        explicit_root=str(root),
    )
    assert rows[0][0] == ""
    assert search(token, explicit_root=str(root), expand=False) == []


def test_parse_meta_accepts_json_and_falls_back_to_text():
    assert _parse_meta('{"description":"A file","tags":["Py", "CLI"]}') == ("A file", ["py", "cli"])
    assert _parse_meta("just a sentence") == ("just a sentence", [])


def test_mcp_json_contains_agent_launch_command_and_limit_flags(tmp_path):
    from quack import mcp_install

    root = scaffold_root(str(tmp_path / "space"))
    path = mcp_install.write_project_config(
        str(root),
        search_limit=7,
        file_char_limit=1234,
        sql_row_limit=8,
        central_limit=9,
    )
    data = json.loads(path.read_text())
    entry = data["mcpServers"]["quack"]

    assert "quack" in data["mcpServers"]
    assert entry["command"]
    assert arg_value(entry["args"], "--root") == str(root.resolve())
    assert arg_value(entry["args"], "--search-limit") == "7"
    assert arg_value(entry["args"], "--file-char-limit") == "1234"
    assert arg_value(entry["args"], "--sql-row-limit") == "8"
    assert arg_value(entry["args"], "--central-limit") == "9"


def test_nested_scaffold_inherits_parent_defaults(tmp_path):
    parent = scaffold_root(str(tmp_path / "parent"))
    parent_config = parent / ".quack" / "config.yaml"
    data = yaml.safe_load(parent_config.read_text())
    data["defaults"]["search_limit"] = 4
    data["defaults"]["file_char_limit"] = 123
    data["defaults"]["sql_row_limit"] = 5
    data["defaults"]["central_limit"] = 6
    write_yaml(parent_config, data)

    child = scaffold_root(str(parent / "projects" / "child"))
    child_data = yaml.safe_load((child / ".quack" / "config.yaml").read_text())

    assert child_data["defaults"] == data["defaults"]
    assert child_data["ai"]["command"] == ""
