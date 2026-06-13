from __future__ import annotations

import json
import sys

import pytest

from quack.cli import main
from quack.config import AIConfig, Config, EmbedConfig
from quack.embed import EmbedNotConfigured, _embed_text, build_embeddings
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
    root = scaffold_root(str(tmp_path / "space"))

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
