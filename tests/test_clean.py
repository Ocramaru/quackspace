"""Tests for `quack clean` — derived cleanup vs full purge."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

from quack import diagram
from quack.clean import clean
from quack.cli import main
from quack.generate import record
from quack.indexer import reindex
from quack.scaffold import scaffold_root


def _space(tmp_path: Path) -> Path:
    root = scaffold_root(str(tmp_path / "space"))
    notes = root / "notes"
    notes.mkdir(exist_ok=True)
    (notes / "a.md").write_text("# A\n\nbody [[b]]\n")
    (notes / "b.md").write_text("# B\n\nback [[a]]\n")
    reindex(str(root))
    diagram.diagram(str(root))  # creates notes/_diagrams.md + .quack/diagram.md
    return root


def test_clean_derived_removes_artifacts_keeps_authored(tmp_path):
    root = _space(tmp_path)
    record(str(root), "notes/a.md", "authored desc", ["t"])
    reindex(str(root))
    idx = root / "notes" / ".index.yaml"
    assert (root / ".quack" / "quack.duckdb").exists()
    assert (root / "notes" / "_diagrams.md").exists()
    assert "authored desc" in idx.read_text()

    clean(str(root))

    # Derived artifacts gone...
    assert not (root / ".quack" / "quack.duckdb").exists()
    assert not (root / ".quack" / "map.yaml").exists()
    assert not (root / ".quack" / "diagram.md").exists()
    assert not (root / "notes" / "_diagrams.md").exists()
    # ...authored metadata, config, and the anchor kept.
    assert idx.exists() and "authored desc" in idx.read_text()
    assert (root / ".quack" / "config.yaml").exists()
    assert (root / "QUACK.md").exists()

    # And it all rebuilds.
    reindex(str(root))
    assert (root / ".quack" / "quack.duckdb").exists()


def test_clean_all_purges_quack_layer(tmp_path):
    root = _space(tmp_path)
    clean(str(root), purge=True)

    assert not (root / ".quack").exists()
    assert not (root / "QUACK.md").exists()
    assert not (root / "notes" / ".index.yaml").exists()
    # The user's own content is untouched.
    assert (root / "notes" / "a.md").exists()
    assert (root / "notes" / "b.md").exists()


def test_clean_all_requires_yes_via_cli(tmp_path):
    root = _space(tmp_path)

    # Without --yes the destructive purge is refused and nothing is removed.
    assert main(["clean", "--all", "--root", str(root)]) == 1
    assert (root / ".quack").exists()
    assert (root / "notes" / ".index.yaml").exists()

    # With --yes it proceeds.
    assert main(["clean", "--all", "--yes", "--root", str(root)]) == 0
    assert not (root / ".quack").exists()


def test_clean_derived_via_cli(tmp_path):
    root = _space(tmp_path)
    assert main(["clean", "--root", str(root)]) == 0
    assert not (root / ".quack" / "quack.duckdb").exists()
    assert (root / ".quack" / "config.yaml").exists()


def test_clean_dry_run_reports_without_removing(tmp_path, capsys):
    root = _space(tmp_path)

    assert main(["clean", "--dry-run", "--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert "quack clean preview (no deletes)" in out
    assert "catalog:  1" in out
    assert "map:      1" in out
    assert "diagrams: 2" in out
    assert (root / ".quack" / "quack.duckdb").exists()
    assert (root / ".quack" / "map.yaml").exists()
    assert (root / ".quack" / "diagram.md").exists()
    assert (root / "notes" / "_diagrams.md").exists()


def test_clean_can_remove_only_diagrams(tmp_path):
    root = _space(tmp_path)

    assert main(["clean", "--diagrams", "--root", str(root)]) == 0

    assert (root / ".quack" / "quack.duckdb").exists()
    assert (root / ".quack" / "map.yaml").exists()
    assert not (root / ".quack" / "diagram.md").exists()
    assert not (root / "notes" / "_diagrams.md").exists()


def test_clean_interactive_menu_can_pick_catalog_and_map(tmp_path, monkeypatch):
    root = _space(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "3")

    assert main(["clean", "--root", str(root)]) == 0

    assert not (root / ".quack" / "quack.duckdb").exists()
    assert not (root / ".quack" / "map.yaml").exists()
    assert (root / ".quack" / "diagram.md").exists()
    assert (root / "notes" / "_diagrams.md").exists()


def test_clean_all_dry_run_does_not_require_yes_or_remove(tmp_path, capsys):
    root = _space(tmp_path)

    assert main(["clean", "--all", "--dry-run", "--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert "quack clean preview (no deletes)" in out
    assert "mode: full uninstall" in out
    assert (root / ".quack").exists()
    assert (root / "QUACK.md").exists()
    assert (root / "notes" / ".index.yaml").exists()


def test_clean_all_removes_mcp_registration(tmp_path):
    from quack import mcp_install

    root = _space(tmp_path)
    mcp_install.write_project_config(str(root))
    assert (root / ".mcp.json").exists()

    clean(str(root), purge=True)
    assert not (root / ".mcp.json").exists()


def test_clean_catches_stragglers_not_in_catalog(tmp_path):
    """A _diagrams.md in a folder the catalog never indexed is still found and
    removed by the disk-scan catch-all, and reported as an extra."""
    root = _space(tmp_path)
    stray_dir = root / "added-later"
    stray_dir.mkdir()
    stray = stray_dir / "_diagrams.md"
    stray.write_text("# GENERATED stray\n")

    removed = clean(str(root))
    assert not stray.exists()
    assert removed["extras"] >= 1
