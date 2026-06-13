from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_mcp_launch_uses_source_checkout_when_command_not_installed(tmp_path, monkeypatch):
    root = tmp_path / "space"
    (root / ".quack").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")

    command, args = launch_command(str(root))

    assert command == "uv"
    assert args[:2] == ["run", "--project"]
    assert Path(args[2]).name == "QuackSpace"
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
            },
            sort_keys=False,
        )
    )

    write_config("new", explicit_root=str(root), timeout=9, skip=True)

    data = yaml.safe_load(config.read_text())
    assert data["ai"] == {"command": "new", "timeout": 9, "skip": True}
    assert data["embed"] == {"command": "embed", "dim": 3, "timeout": 4}
    assert data["defaults"] == {
        "search_limit": 4,
        "file_char_limit": 123,
        "sql_row_limit": 5,
        "central_limit": 6,
    }
