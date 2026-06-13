from __future__ import annotations

import sys

from quack import _duck
from quack.cli import DUCK_MESSAGES
from quack.config import AIConfig, Config, DefaultsConfig, EmbedConfig
from quack.generate import fill_descriptions


def test_duck_disabled_for_non_tty(monkeypatch):
    monkeypatch.delenv("QUACK_NO_ANIM", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("QUACK_MCP", raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

    assert _duck.enabled() is False


def test_duck_disabled_by_env(monkeypatch):
    monkeypatch.setenv("QUACK_NO_ANIM", "1")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    assert _duck.enabled() is False


def test_swimming_noops_when_disabled():
    with _duck.swimming("Testing", enabled=False) as progress:
        progress.update(done=1, total=2, message="Still testing")
        progress.advance()

    assert progress.done == 2
    assert progress.total == 2
    assert progress.message == "Still testing"


def test_cli_duck_messages_are_centralized():
    assert DUCK_MESSAGES["reindex"] == "Reindexing workspace"
    assert DUCK_MESSAGES["embed"] == "Building embeddings"
    assert DUCK_MESSAGES["generate"] == "Generating descriptions"


def test_fill_descriptions_reports_progress(sample_space, monkeypatch):
    root = sample_space
    calls: list[tuple[int, int, str]] = []

    cfg = Config(
        ai=AIConfig(command="fake", timeout=1, skip=False),
        embed=EmbedConfig(),
        defaults=DefaultsConfig(),
    )
    monkeypatch.setattr("quack.generate.Config.load", lambda explicit_root=None: cfg)
    monkeypatch.setattr(
        "quack.generate.run_ai",
        lambda config, prompt: '{"description": "Generated description", "tags": ["ai"]}',
    )

    result = fill_descriptions(
        str(root),
        only="projects/beta.md",
        dry_run=True,
        progress=lambda done, total, message: calls.append((done, total, message)),
    )

    assert result.updated == ["projects/beta.md: Generated description  [tags: ai]"]
    assert calls[0] == (0, 1, "Generating projects/beta.md")
    assert calls[-1] == (1, 1, "Generated projects/beta.md")
