from __future__ import annotations

import builtins
import json
import signal
import sys
from pathlib import Path

import pytest
import yaml

from conftest import arg_value
from quack.cli import main
from quack.core import find_root
from quack.mcp_install import launch_command


def test_find_root_walks_up_to_quack_marker(tmp_path, monkeypatch):
    root = tmp_path / "space"
    child = root / "notes"
    (root / ".quack").mkdir(parents=True)
    child.mkdir()

    monkeypatch.chdir(child)

    assert find_root() == root.resolve()


def test_find_root_uses_closest_nested_quack_marker(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    inner = outer / "projects" / "inner"
    child = inner / "notes"
    (outer / ".quack").mkdir(parents=True)
    (inner / ".quack").mkdir(parents=True)
    child.mkdir()

    monkeypatch.chdir(child)

    assert find_root() == inner.resolve()


def test_find_root_fails_outside_quack_space(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="No quack space found"):
        find_root()


def test_find_root_rejects_explicit_root_without_marker(tmp_path):
    with pytest.raises(RuntimeError, match="missing .quack"):
        find_root(str(tmp_path))


def test_where_prints_workspace_state_package_and_command(tmp_path, capsys):
    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)

    assert main(["where", "--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert f"root:     {root.resolve()}" in out
    assert f"state:    {root.resolve() / '.quack'}" in out
    assert "package:" in out
    assert "command:" in out
    assert f"guide:    {root.resolve() / 'QUACK.md'}" in out


def test_init_prints_clean_reindex_summary(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)
    monkeypatch.setattr("quack.cli.reindex", lambda _root, progress=None: {"files": 486099})

    assert main(["init", str(root)]) == 0

    out = capsys.readouterr().out
    assert "✓ scaffolded space at" in out
    assert "✓ reindexed 486,099 file(s)" in out
    assert "  reindexed 486099 file(s)" not in out


def test_init_can_skip_first_reindex(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)

    def fail_reindex(_root, progress=None):
        raise AssertionError("reindex should not run")

    monkeypatch.setattr("quack.cli.reindex", fail_reindex)

    assert main(["init", str(root), "--no-reindex"]) == 0

    out = capsys.readouterr().out
    assert "✓ scaffolded space at" in out
    assert "reindex: skipped (--no-reindex)" in out


def test_init_prompts_before_autokilling_catalog_locker(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    prompts: list[str] = []
    events: list[tuple] = []

    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)
    monkeypatch.setattr(
        "quack.cli._lock_holder_details",
        lambda _root: [(54684, "/usr/bin/python3.13 /tmp/quack/cli.py init")],
    )

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(
        "quack.cli.os.kill",
        lambda pid, sig: events.append(("kill", pid, sig)),
    )
    monkeypatch.setattr(
        "quack.cli.reindex",
        lambda _root, progress=None: events.append(("reindex",)) or {"files": 0, "folder_indexes": 0},
    )

    assert main(["init", str(root)]) == 0

    out = capsys.readouterr().out
    assert "Quack discovered an existing .quack" in out
    assert prompts and "Would you like to autokill this process?" in prompts[0]
    assert events[0] == ("kill", 54684, signal.SIGTERM)
    assert events[-1] == ("reindex",)


def test_init_aborts_when_catalog_locker_is_not_autokilled(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    prompts: list[str] = []
    reindex_called = False

    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)
    monkeypatch.setattr(
        "quack.cli._lock_holder_details",
        lambda _root: [(54684, "/usr/bin/python3.13 /tmp/quack/cli.py init")],
    )

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "n"

    monkeypatch.setattr(builtins, "input", fake_input)

    def fail_reindex(_root, progress=None):
        nonlocal reindex_called
        reindex_called = True
        raise AssertionError("reindex should not run after consent is denied")

    monkeypatch.setattr("quack.cli.reindex", fail_reindex)

    assert main(["init", str(root)]) == 1

    out = capsys.readouterr().out
    assert "init cancelled" in out
    assert prompts and "Would you like to autokill this process?" in prompts[0]
    assert not reindex_called


def test_init_dry_run_lists_writes_without_scaffolding(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    monkeypatch.setenv("QUACK_NO_ANIM", "1")

    assert main(["init", str(root), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "quack init preview" in out
    assert "paths:" in out
    assert str(root.resolve() / ".quack") in out
    assert str(root.resolve() / "QUACK.md") in out
    assert "gitignore: repo .gitignore files checked during init" in out
    assert "reindex: runs during init" in out
    assert not root.exists()


def test_init_dry_run_does_not_scan_gitignore_repos(tmp_path, monkeypatch):
    root = tmp_path / "space"
    monkeypatch.setattr(
        "quack.scaffold.preview_gitignore",
        lambda _root: pytest.fail("dry-run should stay cheap"),
        raising=False,
    )

    assert main(["init", str(root), "--dry-run"]) == 0


def test_init_no_gitignore_skips_repo_gitignore(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)

    assert main(["init", str(repo), "--no-gitignore", "--no-reindex"]) == 0

    out = capsys.readouterr().out
    assert "gitignore: skipped (--no-gitignore)" in out
    assert not (repo / ".gitignore").exists()
    assert not (repo / ".quack" / ".gitignore").exists()
    config = yaml.safe_load((repo / ".quack" / "config.yaml").read_text())
    assert config["gitignore"] is False


def test_init_interactive_choices_update_config_before_writes(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    answers = iter(["n", "n", "n"])
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)
    monkeypatch.setattr(
        "quack.cli.reindex",
        lambda _root, progress=None: {"files": 0, "folder_indexes": 0},
    )

    assert main(["init", str(repo)]) == 0

    out = capsys.readouterr().out
    assert "diagrams: turned off in config" in out
    assert not (repo / ".gitignore").exists()
    assert not (repo / ".quack" / ".gitignore").exists()
    config = yaml.safe_load((repo / ".quack" / "config.yaml").read_text())
    assert config["gitignore"] is False
    assert config["index"]["diagrams"] is False


def test_init_interactive_can_setup_embeddings_without_building(tmp_path, capsys, monkeypatch):
    import yaml

    root = tmp_path / "space"
    script = tmp_path / "embedder.py"
    script.write_text("import json; print(json.dumps([0.1, 0.2, 0.3, 0.4]))\n")
    answers = iter(["y", "y", "y", f"{sys.executable} {script}", "n"])
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)
    monkeypatch.setattr(
        "quack.cli.reindex",
        lambda _root, progress=None: {"files": 0, "folder_indexes": 0},
    )

    assert main(["init", str(root)]) == 0

    out = capsys.readouterr().out
    assert "configured custom embeddings (dim 4)" in out
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"]["command"] == f"{sys.executable} {script}"
    assert config["embed"]["dim"] == 4
    assert config["embed"]["include_body"] is True


def test_init_embedding_setup_failure_does_not_fail_init(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    script = tmp_path / "embedder.py"
    script.write_text("print('not json')\n")
    answers = iter(["y", "y", "y", f"{sys.executable} {script}"])
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)
    monkeypatch.setattr(
        "quack.cli.reindex",
        lambda _root, progress=None: {"files": 0, "folder_indexes": 0},
    )

    assert main(["init", str(root)]) == 0

    out = capsys.readouterr().out
    assert "embeddings: skipped" in out
    assert (root / ".quack" / "config.yaml").exists()


def test_init_preserves_existing_config_without_prompting(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)
    (root / ".quack" / "config.yaml").write_text(
        yaml.safe_dump({"index": {"store_body": False, "diagrams": False}})
    )
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: pytest.fail("should not prompt"))
    monkeypatch.setattr("quack.cli.run_setup", lambda _root: None)

    assert main(["init", str(root), "--no-reindex"]) == 0

    out = capsys.readouterr().out
    assert "config: preserved existing" in out
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["index"] == {"store_body": False, "diagrams": False}


def test_reindex_honors_configured_diagram_skip(tmp_path, capsys, monkeypatch):
    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)
    (root / ".quack" / "config.yaml").write_text(
        yaml.safe_dump({"index": {"diagrams": False}})
    )
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr(
        "quack.cli.reindex",
        lambda _root, progress=None: {
            "files": 1,
            "folder_indexes": 1,
            "map": str(root / ".quack" / "map.yaml"),
            "db": str(root / ".quack" / "quack.duckdb"),
        },
    )
    monkeypatch.setattr("quack.cli.diagram", lambda *args, **kwargs: pytest.fail("should not diagram"))

    assert main(["reindex", "--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert "diagrams: skipped (index.diagrams: false)" in out


def test_mcp_launch_uses_source_checkout_when_command_not_installed(tmp_path, monkeypatch):
    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")

    command, args = launch_command(str(root))

    assert command == "uv"
    assert args[:2] == ["run", "--project"]
    assert (Path(args[2]) / "pyproject.toml").exists()
    assert "quack-mcp" in args
    assert arg_value(args, "--root") == str(root.resolve())


def test_mcp_print_includes_limit_flags(tmp_path, capsys):
    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)

    assert main([
        "mcp",
        "print",
        "--root",
        str(root),
        "--search-limit",
        "6",
        "--sql-row-limit",
        "7",
    ]) == 0

    out = capsys.readouterr().out
    entry = json.loads(out)["mcpServers"]["quack"]
    assert arg_value(entry["args"], "--root") == str(root.resolve())
    assert arg_value(entry["args"], "--search-limit") == "6"
    assert arg_value(entry["args"], "--sql-row-limit") == "7"


def test_write_config_preserves_existing_defaults(tmp_path):
    import yaml
    from quack.config import write_config

    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)
    config = root / ".quack" / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "ai": {"command": "old", "timeout": 1, "skip": False},
                "embed": {"command": "embed", "dim": 3, "timeout": 4},
                "defaults": {
                    "search_limit": 4,
                    "file_char_limit": 123,
                    "sql_row_limit": 5,
                    "central_limit": 6,
                },
                "index": {"store_body": False},
            },
            sort_keys=False,
        )
    )

    write_config("new", explicit_root=str(root), timeout=9, skip=True)

    data = yaml.safe_load(config.read_text())
    assert data["ai"] == {"command": "new", "timeout": 9, "skip": True}
    assert data["embed"] == {
        "command": "embed",
        "dim": 3,
        "timeout": 4,
        "include_body": True,
    }
    assert data["defaults"] == {
        "search_limit": 4,
        "file_char_limit": 123,
        "sql_row_limit": 5,
        "central_limit": 6,
    }
    assert data["index"] == {"store_body": False, "diagrams": True}


def test_config_loads_index_body_storage(tmp_path):
    from quack.config import Config

    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)
    config = root / ".quack" / "config.yaml"
    config.write_text(yaml.safe_dump({"index": {"store_body": "false", "diagrams": "false"}}))

    loaded = Config.load(str(root))

    assert loaded.index.store_body is False
    assert loaded.index.diagrams is False
