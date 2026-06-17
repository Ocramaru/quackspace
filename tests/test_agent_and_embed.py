from __future__ import annotations

import json
import sys

import pytest

from quack.cli import main
from quack.config import AIConfig, Config, EmbedConfig
from quack.embed import (
    DEFAULT_EMBED_COMMAND,
    OLLAMA_EMBED_COMMAND,
    EmbedNotConfigured,
    _embed_text,
    build_embeddings,
    run_embed_setup,
    semantic_search,
)
from quack.embed_provider import embed as builtin_embed
from quack.kiro import hook_definitions
from quack.scaffold import scaffold_root


def test_agent_kiro_install_writes_hooks(tmp_path, capsys):
    root = scaffold_root(str(tmp_path / "space"))

    assert main(["agent", "kiro", "install", "--root", str(root)]) == 0

    out = capsys.readouterr().out
    hook = root / ".kiro" / "hooks" / "quack-reindex-on-save.kiro.hook"
    assert "installed 1 Kiro hook" in out
    assert hook.exists()
    data = json.loads(hook.read_text())
    assert data == hook_definitions()["quack-reindex-on-save"]


def test_embed_not_configured_requires_configured_command(tmp_path):
    import yaml

    root = scaffold_root(str(tmp_path / "space"))
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    config["embed"] = {"command": "", "dim": 0, "timeout": 5}
    (root / ".quack" / "config.yaml").write_text(yaml.safe_dump(config))

    with pytest.raises(EmbedNotConfigured):
        build_embeddings(str(root))


def test_embed_text_rejects_non_json_output(tmp_path):
    script = tmp_path / "embedder.py"
    script.write_text("print('not json')\n")
    cfg = EmbedConfig(command=f"{sys.executable} {script}", timeout=5)

    with pytest.raises(json.JSONDecodeError):
        _embed_text(cfg, "hello")


def test_embed_text_nonzero_with_empty_output_has_actionable_error():
    cfg = EmbedConfig(command=f'{sys.executable} -c "import sys; sys.exit(4)"', timeout=5)

    with pytest.raises(RuntimeError) as exc:
        _embed_text(cfg, "hello")

    message = str(exc.value)
    assert "Embedding command failed (4)" in message
    assert "no output on stderr or stdout" in message


def test_embed_text_replaces_text_after_tokenizing(tmp_path):
    script = tmp_path / "embedder.py"
    script.write_text(
        "import json, sys\n"
        "assert sys.argv[1] == 'hello \"world\"'\n"
        "print(json.dumps([1.0]))\n"
    )
    cfg = EmbedConfig(command=f'{sys.executable} {script} "{{text}}"', timeout=5)

    assert _embed_text(cfg, 'hello "world"') == [1.0]


def test_embed_init_command_validates_and_writes_config(tmp_path, capsys):
    import yaml

    root = scaffold_root(str(tmp_path / "space"))
    script = tmp_path / "embedder.py"
    script.write_text("import json; print(json.dumps([0.1, 0.2, 0.3]))\n")

    assert main([
        "embed",
        "init",
        "--root",
        str(root),
        "--command",
        f"{sys.executable} {script}",
    ]) == 0

    out = capsys.readouterr().out
    assert "configured custom embeddings (dim 3)" in out
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"]["command"] == f"{sys.executable} {script}"
    assert config["embed"]["provider"] == "custom"
    assert config["embed"]["dim"] == 3


def test_embed_command_flag_implies_setup(tmp_path, capsys):
    import yaml

    root = scaffold_root(str(tmp_path / "space"))
    script = tmp_path / "embedder.py"
    script.write_text("import json; print(json.dumps([0.1, 0.2]))\n")

    assert main([
        "embed",
        "--root",
        str(root),
        "--command",
        f"{sys.executable} {script}",
    ]) == 0

    out = capsys.readouterr().out
    assert "configured custom embeddings (dim 2)" in out
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"]["provider"] == "custom"
    assert config["embed"]["dim"] == 2


def test_embed_init_rejects_invalid_command_without_writing(tmp_path):
    import yaml

    root = scaffold_root(str(tmp_path / "space"))
    script = tmp_path / "embedder.py"
    script.write_text("print('not json')\n")

    assert main([
        "embed",
        "init",
        "--root",
        str(root),
        "--command",
        f"{sys.executable} {script}",
    ]) == 1

    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"]["command"] == "quack embed text"
    assert config["embed"]["provider"] == "builtin"


def test_embed_not_configured_cli_points_to_init(tmp_path, capsys):
    import yaml

    root = scaffold_root(str(tmp_path / "space"))
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    config["embed"] = {"command": "", "dim": 0, "timeout": 5}
    (root / ".quack" / "config.yaml").write_text(yaml.safe_dump(config))

    assert main(["embed", "--root", str(root)]) == 1

    out = capsys.readouterr().out
    assert "Run `quack embed init`" in out


def test_semantic_search_configured_but_not_built_is_friendly(tmp_path, capsys):
    import yaml

    root = scaffold_root(str(tmp_path / "space"))
    (root / "projects" / "a.md").write_text("# A\n\nlogin\n")
    from quack.indexer import reindex

    reindex(str(root))
    script = tmp_path / "embedder.py"
    script.write_text("import json; print(json.dumps([0.1, 0.2]))\n")
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    config["embed"] = {"command": f"{sys.executable} {script}", "dim": 2, "timeout": 5}
    (root / ".quack" / "config.yaml").write_text(yaml.safe_dump(config))

    assert main(["search", "login", "--semantic", "--root", str(root)]) == 1

    out = capsys.readouterr().out
    assert "Run `quack embed`" in out


def test_embed_setup_noninteractive_uses_builtin_default(tmp_path, monkeypatch, capsys):
    import yaml

    root = scaffold_root(str(tmp_path / "space"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    result = run_embed_setup(str(root))

    out = capsys.readouterr().out
    assert result.configured is True
    assert result.command == DEFAULT_EMBED_COMMAND
    assert result.provider == "builtin"
    assert "built-in free local embedding command" in out
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"]["command"] == DEFAULT_EMBED_COMMAND
    assert config["embed"]["provider"] == "builtin"
    assert config["embed"]["dim"] == 256


def test_embed_setup_ollama_provider_can_pull(tmp_path, monkeypatch):
    import yaml

    from quack import embed as embed_mod

    root = scaffold_root(str(tmp_path / "space"))
    pulled = []
    monkeypatch.setattr(
        embed_mod,
        "_pull_ollama_model",
        lambda model, timeout: pulled.append((model, timeout)),
    )
    monkeypatch.setattr(embed_mod, "_ensure_ollama_server", lambda timeout: None)
    monkeypatch.setattr(embed_mod, "_embed_text", lambda _cfg, _text: [0.1, 0.2, 0.3])

    result = run_embed_setup(str(root), provider="ollama", pull=True, timeout=9)

    assert result.configured is True
    assert result.provider == "ollama"
    assert result.command == OLLAMA_EMBED_COMMAND
    assert pulled == [("nomic-embed-text", 9)]
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"] == {
        "provider": "ollama",
        "command": OLLAMA_EMBED_COMMAND,
        "dim": 3,
        "timeout": 9,
        "skip": False,
    }


def test_embed_setup_interactive_defaults_to_ollama(tmp_path, monkeypatch):
    import builtins
    import yaml

    from quack import embed as embed_mod

    root = scaffold_root(str(tmp_path / "space"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")
    monkeypatch.setattr(embed_mod, "_ollama_model_exists", lambda _model: True)
    monkeypatch.setattr(embed_mod, "_ensure_ollama_server", lambda timeout: None)
    monkeypatch.setattr(embed_mod, "_embed_text", lambda _cfg, _text: [0.1, 0.2])

    result = run_embed_setup(str(root))

    assert result.provider == "ollama"
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"]["provider"] == "ollama"
    assert config["embed"]["command"] == OLLAMA_EMBED_COMMAND
    assert config["embed"]["dim"] == 2


def test_ollama_pull_skips_existing_model(monkeypatch, capsys):
    from quack import embed as embed_mod

    monkeypatch.setattr(embed_mod, "_ollama_model_exists", lambda _model: True)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("ollama pull should not run")

    monkeypatch.setattr(embed_mod.subprocess, "run", fail_run)

    embed_mod._pull_ollama_model("nomic-embed-text", timeout=9)

    assert "already installed" in capsys.readouterr().out


def test_embed_setup_offers_builtin_when_ollama_install_declines_or_fails(
    tmp_path, monkeypatch
):
    import builtins

    from quack import embed as embed_mod

    root = scaffold_root(str(tmp_path / "space"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")
    monkeypatch.setattr(embed_mod, "_ollama_binary_exists", lambda: False)
    monkeypatch.setattr(embed_mod, "_install_ollama", lambda timeout: False)
    monkeypatch.setattr(embed_mod, "_embed_text", lambda _cfg, _text: [0.1, 0.2])

    result = run_embed_setup(str(root))

    assert result.provider == "builtin"
    assert result.command == DEFAULT_EMBED_COMMAND


def test_embed_setup_installs_ollama_then_pulls_model(tmp_path, monkeypatch):
    import builtins

    from quack import embed as embed_mod

    root = scaffold_root(str(tmp_path / "space"))
    installed = {"ok": False}
    pulled = []

    def install(timeout):
        installed["ok"] = True
        return True

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")
    monkeypatch.setattr(embed_mod, "_ollama_binary_exists", lambda: installed["ok"])
    monkeypatch.setattr(embed_mod, "_ollama_model_exists", lambda _model: False)
    monkeypatch.setattr(embed_mod, "_install_ollama", install)
    monkeypatch.setattr(embed_mod, "_ensure_ollama_server", lambda timeout: None)
    monkeypatch.setattr(
        embed_mod,
        "_pull_ollama_model",
        lambda model, timeout: pulled.append((model, timeout)),
    )
    monkeypatch.setattr(embed_mod, "_embed_text", lambda _cfg, _text: [0.1, 0.2])

    result = run_embed_setup(str(root), timeout=9)

    assert result.provider == "ollama"
    assert pulled == [("nomic-embed-text", 9)]


def test_ensure_ollama_server_starts_when_not_ready(monkeypatch):
    from quack import embed as embed_mod

    checks = iter([False, False, True])
    started = []
    monkeypatch.setattr(embed_mod, "_ollama_server_ready", lambda: next(checks))
    monkeypatch.setattr(embed_mod, "_ollama_binary_exists", lambda: True)
    monkeypatch.setattr(embed_mod.time, "sleep", lambda _seconds: None)

    class FakePopen:
        def __init__(self, cmd, **_kwargs):
            started.append(cmd)

    monkeypatch.setattr(embed_mod.subprocess, "Popen", FakePopen)

    embed_mod._ensure_ollama_server(timeout=10)

    assert started == [["ollama", "serve"]]


def test_embed_custom_provider_requires_command_noninteractive(tmp_path, monkeypatch):
    root = scaffold_root(str(tmp_path / "space"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with pytest.raises(RuntimeError, match="Custom embeddings need `--command`"):
        run_embed_setup(str(root), provider="custom")


def test_embed_provider_flag_implies_setup(tmp_path, monkeypatch, capsys):
    import yaml

    from quack import embed as embed_mod

    root = scaffold_root(str(tmp_path / "space"))
    monkeypatch.setattr(embed_mod, "_embed_text", lambda _cfg, _text: [0.1, 0.2])

    assert main(["embed", "--root", str(root), "--provider", "ollama"]) == 0

    out = capsys.readouterr().out
    assert "configured ollama embeddings (dim 2)" in out
    config = yaml.safe_load((root / ".quack" / "config.yaml").read_text())
    assert config["embed"]["provider"] == "ollama"
    assert config["embed"]["command"] == OLLAMA_EMBED_COMMAND


def test_builtin_embed_provider_returns_normalized_vector():
    vec = builtin_embed("path: src/quack/embed.py\nbody: semantic search")

    assert len(vec) == 256
    assert any(v != 0 for v in vec)
    assert sum(v * v for v in vec) == pytest.approx(1.0)


def test_embed_text_subcommand_prints_json_vector(capsys, monkeypatch):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("path: a.py\nbody: search"))

    assert main(["embed", "text"]) == 0

    vec = json.loads(capsys.readouterr().out)
    assert len(vec) == 256


def test_embed_ollama_text_subcommand_prints_json_vector(capsys, monkeypatch):
    import io

    from quack import embed_ollama

    monkeypatch.setattr(sys, "stdin", io.StringIO("body: search"))
    monkeypatch.setattr(embed_ollama, "embed", lambda text, model: [len(text), len(model)])

    assert main([
        "embed",
        "text",
        "--provider",
        "ollama",
        "--model",
        "nomic-embed-text",
    ]) == 0

    assert json.loads(capsys.readouterr().out) == [12, 16]


def test_embed_refresh_skips_unchanged_updates_stale_and_prunes_deleted(tmp_path):
    import yaml

    from quack import catalog
    from quack.indexer import reindex

    root = scaffold_root(str(tmp_path / "space"))
    (root / "notes").mkdir()
    first = root / "notes" / "first.md"
    second = root / "notes" / "second.md"
    first.write_text("alpha\n")
    second.write_text("beta\n")
    reindex(str(root))

    log = tmp_path / "embed-log.txt"
    script = tmp_path / "embedder.py"
    script.write_text(
        "import hashlib, json, pathlib, sys\n"
        "text = sys.stdin.read()\n"
        "pathlib.Path(sys.argv[1]).open('a').write(text + '\\n---\\n')\n"
        "h = hashlib.sha256(text.encode()).digest()\n"
        "print(json.dumps([b / 255.0 for b in h[:4]]))\n"
    )
    cfg = root / ".quack" / "config.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["embed"] = {"command": f"{sys.executable} {script} {log}", "timeout": 10}
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))

    first_summary = build_embeddings(str(root))
    first_calls = log.read_text().count("\n---\n")
    assert first_calls == first_summary["updated"] + first_summary["folders_updated"]

    second_summary = build_embeddings(str(root))
    assert log.read_text().count("\n---\n") == first_calls
    assert second_summary["updated"] == 0
    assert second_summary["folders_updated"] == 0
    assert second_summary["skipped"] == second_summary["embedded"]
    assert second_summary["folders_skipped"] == second_summary["folders"]

    first.write_text("alpha changed\n")
    reindex(str(root))
    changed_summary = build_embeddings(str(root))
    assert changed_summary["updated"] == 1
    assert changed_summary["folders_updated"] == 0
    assert log.read_text().count("\n---\n") == first_calls + 1

    second.unlink()
    reindex(str(root))
    pruned_summary = build_embeddings(str(root))
    assert pruned_summary["deleted"] == 1
    assert pruned_summary["folders_updated"] == 1
    assert log.read_text().count("\n---\n") == first_calls + 2
    _, rows = catalog.query(
        "SELECT count(*) FROM embeddings WHERE rel = 'notes/second.md'",
        explicit_root=str(root),
    )
    assert rows[0][0] == 0


def test_embed_refresh_updates_when_command_changes(tmp_path):
    import yaml

    from quack.indexer import reindex

    root = scaffold_root(str(tmp_path / "space"))
    (root / "note.md").write_text("same text\n")
    reindex(str(root))

    script = tmp_path / "embedder.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "text = sys.stdin.read()\n"
        "pathlib.Path(sys.argv[1]).open('a').write(text + '\\n---\\n')\n"
        "print(json.dumps([0.1, 0.2, 0.3]))\n"
    )
    cfg = root / ".quack" / "config.yaml"
    data = yaml.safe_load(cfg.read_text())
    first_log = tmp_path / "first-log.txt"
    data["embed"] = {"command": f"{sys.executable} {script} {first_log}", "timeout": 10}
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))

    first_summary = build_embeddings(str(root))

    second_log = tmp_path / "second-log.txt"
    data["embed"]["command"] = f"{sys.executable} {script} {second_log}"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    second_summary = build_embeddings(str(root))

    assert second_summary["updated"] == first_summary["embedded"]
    assert second_summary["folders_updated"] == first_summary["folders"]


def test_semantic_search_filters_stale_vectors_after_reindex(tmp_path):
    import os
    import time
    import yaml

    from quack.indexer import reindex

    root = scaffold_root(str(tmp_path / "space"))
    note = root / "note.md"
    note.write_text("alpha\n")
    reindex(str(root))

    script = tmp_path / "embedder.py"
    script.write_text(
        "import hashlib, json, sys\n"
        "t = sys.stdin.read()\n"
        "h = hashlib.sha256(t.encode()).digest()\n"
        "print(json.dumps([b / 255.0 for b in h[:4]]))\n"
    )
    cfg = root / ".quack" / "config.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["embed"] = {"command": f"{sys.executable} {script}", "timeout": 10}
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))

    build_embeddings(str(root))
    assert [rel for rel, _name, _dist in semantic_search("alpha", str(root))] == [
        "note.md"
    ]

    note.write_text("alpha changed\n")
    now = time.time() + 2
    os.utime(note, (now, now))
    reindex(str(root))
    assert semantic_search("alpha", str(root)) == []

    build_embeddings(str(root))
    assert [rel for rel, _name, _dist in semantic_search("alpha", str(root))] == [
        "note.md"
    ]


def test_embed_cli_rebuild_refreshes_every_current_vector(tmp_path, capsys):
    import yaml

    from quack.indexer import reindex

    root = scaffold_root(str(tmp_path / "space"))
    (root / "one.md").write_text("one\n")
    (root / "two.md").write_text("two\n")
    reindex(str(root))

    log = tmp_path / "embed-log.txt"
    script = tmp_path / "embedder.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "text = sys.stdin.read()\n"
        "pathlib.Path(sys.argv[1]).open('a').write(text + '\\n---\\n')\n"
        "print(json.dumps([0.1, 0.2]))\n"
    )
    cfg = root / ".quack" / "config.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["embed"] = {"command": f"{sys.executable} {script} {log}", "timeout": 10}
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))

    assert main(["embed", "--root", str(root)]) == 0
    first_calls = log.read_text().count("\n---\n")
    assert main(["embed", "--root", str(root)]) == 0
    assert log.read_text().count("\n---\n") == first_calls

    assert main(["embed", "--root", str(root), "--rebuild"]) == 0
    assert log.read_text().count("\n---\n") == first_calls * 2
    out = capsys.readouterr().out
    assert "refreshed:" in out
