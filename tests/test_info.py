from __future__ import annotations

import json

import duckdb
import yaml

from quack import catalog
from quack.cli import main
from quack.indexer import reindex
from quack.scaffold import scaffold_root


def _space(tmp_path, embed_command: str = ""):
    root = scaffold_root(str(tmp_path / "space"))
    config_path = root / ".quack" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["embed"]["command"] = embed_command
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "a.md").write_text("alpha\n")
    (root / "b.md").write_text("beta\n")
    (root / "c.txt").write_text("gamma\n")
    (root / "sub").mkdir()
    (root / "sub" / "d.md").write_text("delta\n")
    reindex(str(root))
    return root


def _snapshot(root):
    return {
        str(path): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_info_missing_catalog(tmp_path, capsys):
    root = scaffold_root(str(tmp_path / "space"))

    assert main(["info", "--root", str(root)]) == 0
    assert "No catalog found" in capsys.readouterr().out

    assert main(["info", "--root", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"catalog": False}


def test_info_json_counts(tmp_path, capsys):
    root = _space(tmp_path)

    assert main(["info", "--root", str(root), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)

    assert info["catalog"] is True
    assert info["files"] == 4
    assert info["embedded"] == 0
    assert info["missing_embeddings"] == 0
    assert info["folders"] >= 2
    assert info["total_size"] == len("alpha\n") + len("beta\n") + len("gamma\n") + len("delta\n")
    types = {entry["type"]: entry["files"] for entry in info["top_types"]}
    assert types["md"] == 3
    assert types["txt"] == 1
    assert info["last_indexed"]
    assert info["last_embedded"] is None
    assert info["embedding"]["enabled"] is False


def test_info_reports_embedding_config(tmp_path, capsys):
    command = "quack embed text --provider ollama --model test-model"
    root = _space(tmp_path, embed_command=command)

    assert main(["info", "--root", str(root), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)

    assert info["embedding"]["enabled"] is True
    assert info["embedding"]["model"] == "test-model"
    assert info["embedded"] == 0
    assert info["missing_embeddings"] > 0


def test_info_embedded_counts_distinct_files(tmp_path, capsys):
    command = "quack embed text --provider ollama --model test-model"
    root = _space(tmp_path, embed_command=command)

    assert main(["info", "--root", str(root), "--json"]) == 0
    before = json.loads(capsys.readouterr().out)
    total = before["embedded"] + before["missing_embeddings"]

    con = duckdb.connect(str(root / ".quack" / catalog.DB_NAME))
    try:
        con.execute(
            "CREATE TABLE embeddings (name VARCHAR, rel VARCHAR, source_hash VARCHAR, vec FLOAT[2])"
        )
        # a.md has two fresh chunk rows, b.md one: two files embedded.
        for rel in ("a.md", "a.md", "b.md"):
            con.execute(
                "INSERT INTO embeddings SELECT rel, rel, sha256(? || chr(0) || embed_source_hash), "
                "[0.0, 1.0] FROM files WHERE rel = ?",
                [command, rel],
            )
        # A stale vector for c.txt and one for a file no longer indexed don't count.
        con.execute("INSERT INTO embeddings VALUES ('c.txt', 'c.txt', 'stale', [0.0, 1.0])")
        con.execute("INSERT INTO embeddings VALUES ('gone.md', 'gone.md', 'h', [0.0, 1.0])")
    finally:
        con.close()

    assert main(["info", "--root", str(root), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)

    assert info["embedded"] == 2
    assert info["missing_embeddings"] == before["missing_embeddings"] - 2
    assert info["embedded"] + info["missing_embeddings"] == total


def test_info_text_output(tmp_path, capsys):
    root = _space(tmp_path)

    assert main(["info", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "files indexed:      4" in out
    assert "missing embeddings: 0" in out
    assert "last embedded:      never" in out


def test_info_writes_nothing(tmp_path, capsys):
    root = _space(tmp_path)
    before = _snapshot(root)

    assert main(["info", "--root", str(root)]) == 0
    assert main(["info", "--root", str(root), "--json"]) == 0
    capsys.readouterr()

    assert _snapshot(root) == before
