from __future__ import annotations

import json
import os
from types import SimpleNamespace

import yaml

from quack import cli
from quack.cli import main
from quack.indexer import reindex
from quack.scaffold import scaffold_root
from quack.sync import pending_work


def test_status_reports_missing_catalog(tmp_path, capsys):
    root = scaffold_root(str(tmp_path / "space"))

    assert main(["status", "--root", str(root)]) == 0
    assert "No catalog found" in capsys.readouterr().out


def test_status_reports_stat_only_file_drift(tmp_path, capsys):
    root = scaffold_root(str(tmp_path / "space"))
    note = root / "notes.md"
    note.write_text("one\n")
    reindex(str(root))
    note.write_text("changed and longer\n")
    stat = note.stat()
    os.utime(note, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    (root / "new.md").write_text("new\n")

    assert main(["status", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "new:                1" in out
    assert "modified:           1" in out
    assert "new.md" in out
    assert "notes.md" in out


def test_status_is_current_when_embeddings_are_not_configured(tmp_path, capsys):
    root = scaffold_root(str(tmp_path / "space"))
    config_path = root / ".quack" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["embed"]["command"] = ""
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "note.md").write_text("hello\n")
    reindex(str(root))

    pending = pending_work(str(root))
    assert pending.missing_embeddings == set()

    assert main(["status", "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == "✓ up to date"


def test_sync_reindexes_then_is_noop_without_embed_config(tmp_path, capsys, monkeypatch):
    root = scaffold_root(str(tmp_path / "space"))
    (root / "note.md").write_text("hello\n")
    reindex(str(root))
    (root / "second.md").write_text("second\n")
    calls = []
    real_reindex = cli.reindex

    def tracked_reindex(explicit_root, progress=None):
        calls.append(explicit_root)
        return real_reindex(explicit_root, progress=progress)

    monkeypatch.setattr(cli, "reindex", tracked_reindex)
    assert main(["sync", "--root", str(root)]) == 0
    assert len(calls) == 1
    assert main(["sync", "--root", str(root)]) == 0
    assert len(calls) == 1
    assert "already up to date" in capsys.readouterr().out


def test_sync_uses_shared_pending_and_refreshes_embeddings(monkeypatch, capsys):
    pending = SimpleNamespace(
        catalog_exists=True,
        nothing_to_do=False,
        index_count=2,
        missing_embeddings={"a.md"},
    )
    monkeypatch.setattr("quack.sync.pending_work", lambda _root: pending)
    monkeypatch.setattr(cli, "reindex", lambda _root: {"catalog": "full"})
    monkeypatch.setattr(
        cli.Config,
        "load",
        lambda _root: SimpleNamespace(embed=SimpleNamespace(configured=True, skip=False)),
    )
    monkeypatch.setattr(
        "quack.embed.build_embeddings",
        lambda _root: {"updated": 1, "deleted": 0},
    )

    assert main(["sync", "--root", "/unused"]) == 0
    out = capsys.readouterr().out
    assert "reindexed 2 changed file(s)" in out
    assert "refreshed 1 embedding(s)" in out


class _PyPIResponse:
    def __init__(self, version: str):
        self.payload = json.dumps({"info": {"version": version}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.payload


def test_upgrade_checks_mocked_pypi_and_prints_command(monkeypatch, capsys):
    monkeypatch.setattr(cli, "__version__", "1.0.0")
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: _PyPIResponse("2.0.0"))
    monkeypatch.setattr(cli, "_uses_uv_tool", lambda: True)
    monkeypatch.setattr(cli, "is_interactive", lambda: False)

    assert main(["update"]) == 0
    out = capsys.readouterr().out
    assert "current: 1.0.0" in out
    assert "latest:  2.0.0" in out
    assert "uv tool upgrade quackspace" in out
    assert "defaulting to no" in out


def test_upgrade_handles_network_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline"))
    )
    assert main(["upgrade"]) == 1
    assert "could not check PyPI" in capsys.readouterr().err


def test_uninstall_purges_workspace_removes_mcp_and_prints_package_command(
    tmp_path, monkeypatch, capsys
):
    root = scaffold_root(str(tmp_path / "space"))
    (root / ".mcp.json").write_text("{}\n")
    prompts = iter([True, False])
    cleaned = []
    monkeypatch.setattr(cli, "_safe_confirm", lambda *_args, **_kwargs: next(prompts))
    monkeypatch.setattr("quack.clean.clean", lambda explicit_root, purge=False: cleaned.append((explicit_root, purge)) or {"other": 1})
    monkeypatch.setattr("quack.mcp_install.CLIENTS", [])
    monkeypatch.setattr(cli, "_uses_uv_tool", lambda: False)

    assert main(["uninstall", "--root", str(root)]) == 0
    assert cleaned == [(str(root.resolve()), True)]
    assert not (root / ".mcp.json").exists()
    out = capsys.readouterr().out
    assert "python -m pip uninstall -y quackspace" in out
    assert "package uninstall skipped" in out


def test_uninstall_noninteractive_keeps_workspace(monkeypatch, tmp_path, capsys):
    root = scaffold_root(str(tmp_path / "space"))
    monkeypatch.setattr(cli, "is_interactive", lambda: False)
    monkeypatch.setattr("quack.mcp_install.CLIENTS", [])
    monkeypatch.setattr(cli, "_uses_uv_tool", lambda: False)

    assert main(["uninstall", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "workspace files: kept" in out
    assert out.count("defaulting to no") == 2
