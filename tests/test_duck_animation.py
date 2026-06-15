from __future__ import annotations

import sys

from quack import _duck
from quack.config import AIConfig, Config, DefaultsConfig, EmbedConfig
from quack.generate import fill_descriptions


class SequenceRng:
    def __init__(self, values: list[int]):
        self.values = values

    def randrange(self, start: int, stop: int | None = None) -> int:
        assert self.values
        value = self.values.pop(0)
        if stop is None:
            assert 0 <= value < start
        else:
            assert start <= value < stop
        return value


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
    assert progress.refresh_per_second == 10


def test_paddling_noops_when_disabled():
    with _duck.paddling("Searching", enabled=False) as progress:
        progress.update(done=1, total=2, message="Searching full text")

    assert progress.done == 1
    assert progress.total == 2
    assert progress.message == "Searching full text"
    assert progress.refresh_per_second == 10


def test_paddling_frame_preserves_count_when_message_is_truncated():
    frame = _duck._paddling_frame(
        1,
        "Searching embeddings and loading a very long explanation",
        done=4,
        total=6,
        width=34,
    )

    assert "[4/6]" in frame
    assert "..." in frame


def test_family_state_adds_ducks_from_right_after_interval():
    family = _duck._FamilyState.create(
        refresh_per_second=1,
        rng=SequenceRng([1, 15, 3, 15]),
    )

    before = family.specs(14, 80)
    after = family.specs(15, 80)

    assert len(before) == 1
    assert len(after) == 3
    assert after[1][1] > 80
    assert after[2][1] > 80
    assert family.ducks[1].born_frame > family.ducks[0].born_frame
    assert family.ducks[2].born_frame > family.ducks[1].born_frame


def test_family_state_removes_ducks_only_after_exiting_offscreen():
    family = _duck._FamilyState.create(
        refresh_per_second=1,
        rng=SequenceRng([4, 15, 2, 15]),
    )

    changing = family.specs(15, 80)
    exiting = [duck for duck in family.ducks if duck.exiting_since is not None]
    family.next_change_frame = 999
    gone = family.specs(220, 80)

    assert len(changing) == 4
    assert len(exiting) == 2
    assert exiting[0].exiting_since != exiting[1].exiting_since
    assert len(gone) == 2
    assert all(duck.exiting_since is None for duck in family.ducks)


def test_pond_width_tracks_terminal_width():
    assert _duck._pond_width(30) == _duck._MIN_POND_WIDTH
    assert _duck._pond_width(70) == 68
    assert _duck._pond_width(140) == 138


def test_frame_uses_requested_terminal_width():
    narrow = _duck._frame(3, "Working", width=50)
    wide = _duck._frame(3, "Working", width=90)
    narrow_surface = [
        line for line in narrow.splitlines() if line.startswith("[dim magenta]")
    ][-1]
    wide_surface = [
        line for line in wide.splitlines() if line.startswith("[dim magenta]")
    ][-1]

    assert len(narrow_surface) == _duck._pond_width(50) + len("[dim magenta][/]")
    assert len(wide_surface) == _duck._pond_width(90) + len("[dim magenta][/]")


def test_frame_layout_stays_stable_across_animation_frames():
    assert len(_duck._frame(0, "Working").splitlines()) == len(
        _duck._frame(25, "Working").splitlines()
    )


def test_frame_renders_progress_bar_when_total_is_known():
    frame = _duck._frame(0, "Reindexing", done=335965, total=486096, width=80)

    assert "335,965/486,096 69%" in frame
    assert "━━━━" in frame


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
