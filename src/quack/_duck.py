"""Tiny terminal duck animations for human-facing CLI waits.

Pure decoration: this module must never affect command behavior or exit codes.
It auto-disables for non-TTY output, ``NO_COLOR``, ``QUACK_NO_ANIM``, and when
``rich`` is unavailable.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, TypeAlias

_Cell: TypeAlias = tuple[str, str | None]


class _RandomSource(Protocol):
    def randrange(self, start: int, stop: int | None = None) -> int:
        ...


# Tail flick, with a splash mid-stroke.
_TAILS = ["/", "~", "\\"]

# Water is generated from a deterministic per-frame random seed. That gives the
# pond different heights without making tests or terminal recordings flaky.
_POND_WIDTH = 44
_WAVE_SEED = 11
_MIN_POND_WIDTH = 44
_DEFAULT_BABY_COUNT = 3
_MIN_BABY_COUNT = 1
_MAX_BABY_COUNT = 4
_FAMILY_CHANGE_MIN_SECONDS = 15
_FAMILY_CHANGE_MAX_SECONDS = 30
_DUCK_SCENE_ROWS = 7
_DUCK_STAGGER_FRAMES = 8
_DUCK_SPACING = 9
_MAMA_TOPS = (3, 2, 2, 1, 2)
_BABY_COLORS = ("light_goldenrod1", "light_pink1", "pale_turquoise1", "light_salmon1")
_BABY_ART_LEFT = (
    "   __ ",
    " <(o )",
    "  (  )",
    "   ^^ ",
)
_BABY_ART_RIGHT = (
    " __   ",
    "( o)> ",
    " (  ) ",
    "  ^^  ",
)
_BABY_ART = _BABY_ART_LEFT
_BABY_WIDTH = max(
    len(line) for art in (_BABY_ART_LEFT, _BABY_ART_RIGHT) for line in art
)
_BABY_WORDS = ("peep", "quack", "eep", "waddle", "hi")
_STATUS_BOX_MIN_WIDTH = 24
_STATUS_BOX_MAX_WIDTH = 44
DEFAULT_MESSAGES = (
    "quacking around",
    "getting ducks in a row",
    "paddling the catalog",
    "waddling through files",
    "making a splash",
    "preening metadata",
)
MESSAGE_SEPARATOR = " 󰇥 "
_MESSAGE_FRAME_SPAN = 40
_EYES = (
    "o", "o", "o", "o",
    "<",
    "o", "o", "o", "O",
    "o", "o", "o", "o",
    "-",
    "o", "o", "o", "o",
)


def _eye(frame: int) -> str:
    return _EYES[(frame // 2) % len(_EYES)]


def _blank_cells(width: int) -> list[_Cell]:
    return [(" ", None) for _ in range(width)]


@lru_cache(maxsize=256)
def _visible_span(text: str) -> tuple[int, int] | None:
    first = len(text) - len(text.lstrip(" "))
    if first == len(text):
        return None
    return first, len(text.rstrip(" "))


def _place_cells(row: list[_Cell], pos: int, text: str, style: str | None = None) -> None:
    span = _visible_span(text)
    if span is None:
        return
    first, last = span
    for i, char in enumerate(text[first:last], start=first):
        col = pos + i
        if 0 <= col < len(row):
            row[col] = (char, style)


def _render_cells(row: list[_Cell]) -> str:
    while row and row[-1][0] == " " and row[-1][1] is None:
        row.pop()
    rendered: list[str] = []
    segment: list[str] = []
    style: str | None = None

    def flush() -> None:
        if not segment:
            return
        text = "".join(segment)
        rendered.append(f"[{style}]{text}[/]" if style else text)
        segment.clear()

    for char, cell_style in row:
        if cell_style != style:
            flush()
            style = cell_style
        segment.append(char)
    flush()
    return "".join(rendered).rstrip()


def _copy_rows(rows: tuple[tuple[_Cell, ...], ...]) -> list[list[_Cell]]:
    return [list(row) for row in rows]


def _pond_width(console_width: int | None = None) -> int:
    if console_width is None:
        console_width = shutil.get_terminal_size((80, 24)).columns
    return max(_MIN_POND_WIDTH, console_width - 2)


def _mama_wander(frame: int) -> int:
    slow = ((frame // 18) % 5) - 2
    lazy = ((frame // 37) % 3) - 1
    return max(-3, min(3, slow + lazy))


def _mama_left(width: int, frame: int = 0) -> int:
    base = max(2, min(10, width // 5))
    return max(1, min(width - 10, base + _mama_wander(frame)))


def _mama_top(frame: int) -> int:
    drift = (frame // 24 + frame // 53) % 5
    return _MAMA_TOPS[drift]


def _new_baby_count(rng: _RandomSource | None = None) -> int:
    rng = rng or random.SystemRandom()
    return rng.randrange(_MIN_BABY_COUNT, _MAX_BABY_COUNT + 1)


def _baby_count(count: int | None = None) -> int:
    return count if count is not None else _DEFAULT_BABY_COUNT


def _baby_word(frame: int, baby: int) -> str | None:
    phase = (frame + baby * 13) % 54
    if phase >= 12:
        return None
    return _BABY_WORDS[(frame // 54 + baby) % len(_BABY_WORDS)]


def _baby_bob(frame: int, baby: int) -> int:
    return ((frame // 4 + baby * 3) % 4) - 1


def _baby_lane(frame: int, baby: int) -> int:
    return (frame // 23 + baby * 2 + _baby_bob(frame, baby)) % 4


def _baby_looks_right(frame: int, baby: int) -> bool:
    return (frame // 36 + baby) % 4 == 1


def _baby_art(frame: int, baby: int) -> tuple[str, str, str, str]:
    return _BABY_ART_RIGHT if _baby_looks_right(frame, baby) else _BABY_ART_LEFT


def _mama_looks_right(frame: int) -> bool:
    return (frame // 44) % 4 == 2


def _baby_target(frame: int, width: int, baby: int) -> int:
    return _mama_left(width, frame) + 14 + baby * _DUCK_SPACING


def _baby_start(width: int, baby: int) -> int:
    return width + 6 + baby * _DUCK_SPACING


def _baby_wander(frame: int, baby: int) -> int:
    slow = ((frame // 9 + baby * 4) % 7) - 3
    lazy = ((frame // 17 + baby * 5) % 5) - 2
    return max(-5, min(5, slow + lazy))


def _baby_entering_pos(frame: int, width: int, baby: int, born_frame: int) -> int:
    approach = (frame - born_frame) // 2 - baby * 8
    if approach < 0:
        return _baby_start(width, baby)
    return max(_baby_target(frame, width, baby), _baby_start(width, baby) - approach)


def _baby_pos(frame: int, width: int, baby: int, born_frame: int) -> int:
    return _baby_entering_pos(frame, width, baby, born_frame) + _baby_wander(frame, baby)


def _baby_exit_pos(
    frame: int,
    width: int,
    baby: int,
    born_frame: int,
    exiting_since: int,
) -> int:
    if frame < exiting_since:
        return _baby_pos(frame, width, baby, born_frame)
    exit_start = _baby_pos(exiting_since, width, baby, born_frame)
    return exit_start + (frame - exiting_since) // 2


def _baby_specs(
    frame: int, width: int = _POND_WIDTH, baby_count: int | None = None
) -> list[tuple[int, int, tuple[str, str, str, str], str]]:
    specs: list[tuple[int, int, tuple[str, str, str, str], str]] = []
    for baby in range(_baby_count(baby_count)):
        pos = _baby_pos(frame, width, baby, 0)
        row = _baby_lane(frame, baby)
        color = _BABY_COLORS[baby % len(_BABY_COLORS)]
        specs.append((row, pos, _baby_art(frame, baby), color))
    return specs


@dataclass
class _Duckling:
    slot: int
    born_frame: int
    exiting_since: int | None = None


@dataclass
class _FamilyState:
    refresh_per_second: int
    rng: _RandomSource
    ducks: list[_Duckling]
    next_change_frame: int

    @classmethod
    def create(
        cls,
        *,
        refresh_per_second: int,
        rng: _RandomSource | None = None,
        frame: int = 0,
    ) -> "_FamilyState":
        rng = rng or random.SystemRandom()
        ducks = [
            _Duckling(slot=slot, born_frame=frame)
            for slot in range(_new_baby_count(rng))
        ]
        return cls(
            refresh_per_second=refresh_per_second,
            rng=rng,
            ducks=ducks,
            next_change_frame=frame + _family_interval_frames(refresh_per_second, rng),
        )

    def specs(
        self, frame: int, width: int
    ) -> list[tuple[int, int, tuple[str, str, str, str], str]]:
        self._remove_exited(frame, width)
        if frame >= self.next_change_frame:
            self._choose_new_target(frame)
        return [
            _duckling_spec(frame, width, duck)
            for duck in self.ducks
        ]

    def _choose_new_target(self, frame: int) -> None:
        if any(duck.exiting_since is not None for duck in self.ducks):
            self.next_change_frame = frame + _family_interval_frames(
                self.refresh_per_second,
                self.rng,
            )
            return
        current = len(self.ducks)
        target = _new_target_baby_count(current, self.rng)
        self._set_target(frame, target)
        self.next_change_frame = frame + _family_interval_frames(
            self.refresh_per_second,
            self.rng,
        )

    def _set_target(self, frame: int, target: int) -> None:
        staying = sorted(
            [duck for duck in self.ducks if duck.exiting_since is None],
            key=lambda item: item.slot,
        )
        current = len(staying)
        if target > current:
            for slot in range(current, target):
                born_frame = frame + (slot - current) * _DUCK_STAGGER_FRAMES
                self.ducks.append(_Duckling(slot=slot, born_frame=born_frame))
            return
        for index, duck in enumerate(reversed(staying[target:])):
            duck.exiting_since = frame + index * _DUCK_STAGGER_FRAMES

    def _remove_exited(self, frame: int, width: int) -> None:
        self.ducks = [
            duck
            for duck in self.ducks
            if duck.exiting_since is None
            or _baby_exit_pos(frame, width, duck.slot, duck.born_frame, duck.exiting_since)
            < width + _BABY_WIDTH
        ]


def _family_interval_frames(refresh_per_second: int, rng: _RandomSource) -> int:
    min_frame = _FAMILY_CHANGE_MIN_SECONDS * max(refresh_per_second, 1)
    max_frame = _FAMILY_CHANGE_MAX_SECONDS * max(refresh_per_second, 1)
    return rng.randrange(min_frame, max_frame + 1)


def _new_target_baby_count(current: int, rng: _RandomSource) -> int:
    target = _new_baby_count(rng)
    if target != current:
        return target
    if current < _MAX_BABY_COUNT:
        return current + 1
    return current - 1


def _duckling_spec(
    frame: int,
    width: int,
    duck: _Duckling,
) -> tuple[int, int, tuple[str, str, str, str], str]:
    if duck.exiting_since is None:
        pos = _baby_pos(frame, width, duck.slot, duck.born_frame)
    else:
        pos = _baby_exit_pos(frame, width, duck.slot, duck.born_frame, duck.exiting_since)
    row = _baby_lane(frame, duck.slot)
    color = _BABY_COLORS[duck.slot % len(_BABY_COLORS)]
    return row, pos, _baby_art(frame, duck.slot), color


def _baby_duck_rows(
    frame: int, width: int = _POND_WIDTH, baby_count: int | None = None
) -> tuple[str, ...]:
    rows = [_blank_cells(width) for _ in range(_DUCK_SCENE_ROWS)]
    for row, pos, art, color in _baby_specs(frame, width, baby_count):
        for line_no, line in enumerate(art):
            _place_cells(rows[row + line_no], pos, line, color)
    return tuple(_render_cells(row) for row in rows)


@lru_cache(maxsize=64)
def _pond_background_cells(phase: int, width: int) -> tuple[tuple[_Cell, ...], ...]:
    rows = [_blank_cells(width) for _ in range(_DUCK_SCENE_ROWS)]
    for row_no, row in enumerate(rows):
        spacing = 11 + row_no * 2
        offset = (phase * (row_no + 1) + row_no * 5 + _WAVE_SEED) % spacing
        for col in range(offset, width, spacing):
            mark = "~" if (col + phase + row_no) % 3 else "_"
            style = "dim cyan" if (col + row_no + phase) % 2 else "dim magenta"
            _place_cells(row, col, mark, style)
        if row_no in (2, 4):
            extra = (width - 1 - phase * (row_no + 2) - row_no * 7) % max(width, 1)
            _place_cells(row, extra, "~", "dim blue")
    return tuple(tuple(row) for row in rows)


def _pond_background_rows(frame: int, width: int = _POND_WIDTH) -> tuple[str, ...]:
    phase = frame // 3
    return tuple(
        _render_cells(row) for row in _copy_rows(_pond_background_cells(phase, width))
    )


def _place_foreground_bubbles(
    rows: list[list[_Cell]],
    frame: int,
    width: int,
    baby_specs: list[tuple[int, int, tuple[str, str, str, str], str]],
    mama_top: int,
) -> None:
    def place_trail(
        mouth_row: int,
        mouth_pos: int,
        direction: int,
        phase: int,
        span: int,
    ) -> None:
        if phase >= span:
            return
        for puff in range(3):
            age = phase - puff * 2
            if age < 0 or age >= span:
                continue
            bubble_row = max(0, mouth_row - age // 3)
            drift = age // 2 + (age + puff) % 2
            bubble_pos = mouth_pos + direction * (1 + drift)
            bubble = "O" if age < 3 and puff == 0 else ("o" if age < 7 else ".")
            _place_cells(rows[bubble_row], bubble_pos, bubble, "bold cyan")

    for baby, (row, pos, art, _) in enumerate(baby_specs):
        phase = (frame // 2 + baby * 3) % 12
        mouth_row = row + 1
        looks_right = art == _BABY_ART_RIGHT
        mouth_pos = pos + 4 if looks_right else pos + 1
        outward_direction = 1 if looks_right else -1
        place_trail(mouth_row, mouth_pos, outward_direction, phase, 10)

    mama_phase = frame % 18
    mouth_row = mama_top + 1
    looks_right = _mama_looks_right(frame)
    mama_left = _mama_left(width, frame)
    mouth_pos = mama_left + 7 if looks_right else mama_left
    outward_direction = 1 if looks_right else -1
    place_trail(mouth_row, mouth_pos, outward_direction, mama_phase, 13)


def _duck_scene_rows(
    frame: int,
    width: int = _POND_WIDTH,
    baby_count: int | None = None,
    baby_specs: list[tuple[int, int, tuple[str, str, str, str], str]] | None = None,
) -> tuple[str, ...]:
    rows = _copy_rows(_pond_background_cells(frame // 3, width))
    if baby_specs is None:
        baby_specs = _baby_specs(frame, width, baby_count)
    mama_top = _mama_top(frame)
    for baby, (row, pos, art, color) in enumerate(baby_specs):
        for line_no, line in enumerate(art):
            _place_cells(rows[row + line_no], pos, line, color)
        word = _baby_word(frame, baby)
        if word is not None:
            speech_row = max(0, row - 1)
            speech_pos = pos + 6 if row == 0 else pos + 1
            _place_cells(rows[speech_row], speech_pos, word, "dim white")

    left = _mama_left(width, frame)
    tail = _TAILS[(frame // 2) % len(_TAILS)]
    eye = _eye(frame)
    looks_right = _mama_looks_right(frame)
    if looks_right:
        mama_art = (
            ("   __", "orange1"),
            (f"___( {eye})>", "gold1"),
            (f"( {tail} <_. )", "gold1"),
            (" `-----'", "gold1"),
        )
    else:
        mama_art = (
            ("   __", "orange1"),
            (f"<({eye} )___", "gold1"),
            (f"( ._> {tail} )", "gold1"),
            (" `-----'", "gold1"),
        )
    for row, (line, color) in enumerate(mama_art, start=mama_top):
        _place_cells(rows[row], left, line, color)
        if row == mama_top + 1:
            eye_col = 5 if looks_right else 2
            _place_cells(rows[row], left + eye_col, eye, "bold white")

    _place_foreground_bubbles(rows, frame, width, baby_specs, mama_top)

    return tuple(_render_cells(row) for row in rows)


@lru_cache(maxsize=16)
def _base_wave_heights(width: int) -> tuple[int, ...]:
    rng = random.Random(_WAVE_SEED + width)
    return tuple(rng.randrange(2) for _ in range(width))


@lru_cache(maxsize=256)
def _wave_rows(frame: int, width: int = _POND_WIDTH) -> tuple[str, str]:
    heights = list(_base_wave_heights(width))
    ripple_count = max(1, min(4, width // 36))
    for ripple in range(ripple_count):
        speed = ripple + 1
        center = (frame // 2 * speed + ripple * 13 + _WAVE_SEED) % width
        heights[center] = 2

    crest: list[str] = []
    surface: list[str] = []
    for col, height in enumerate(heights):
        crest.append("~" if height == 2 else " ")
        if height == 1:
            surface.append("~")
        elif height == 0 and (col + frame // 3) % 5 == 0:
            surface.append("_")
        else:
            surface.append(" ")
    return "".join(crest).rstrip(), "".join(surface)


def _message(frame: int, job: str | None) -> str:
    pun = DEFAULT_MESSAGES[(frame // _MESSAGE_FRAME_SPAN) % len(DEFAULT_MESSAGES)]
    return f"{job}{MESSAGE_SEPARATOR}{pun}" if job else pun


@lru_cache(maxsize=8)
def _truncate_text(text: str, max_width: int) -> str:
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    return f"{text[: max_width - 3]}..."


def _header_rows(width: int, status: str) -> tuple[str, str, str]:
    rows = [_blank_cells(width) for _ in range(3)]
    box_width = min(_STATUS_BOX_MAX_WIDTH, max(_STATUS_BOX_MIN_WIDTH, width // 2))
    if width < box_width + 2:
        _place_title(rows[1], width)
        return tuple(_render_cells(row) for row in rows)

    left = width - box_width
    _place_title(rows[1], left)
    inner_width = box_width - 4
    status = _truncate_text(status, inner_width)
    top = f".{'-' * (box_width - 2)}."
    bottom = f"`{'-' * (box_width - 2)}'"
    _place_cells(rows[0], left, top, "bold cyan")
    _place_cells(rows[1], left, "|", "bold gold1")
    _place_cells(rows[1], left + 2, status, "bold gold1")
    _place_cells(rows[1], left + box_width - 1, "|", "bold cyan")
    _place_cells(rows[2], left, bottom, "bold gold1")
    return tuple(_render_cells(row) for row in rows)


def _place_title(row: list[_Cell], region_width: int) -> None:
    tagline = ""
    if region_width >= 40:
        tagline = " enjoy this quacky animation"
    elif region_width >= 26:
        tagline = " quacky animation"

    title_width = len("quackspace") + len(tagline)
    left = max(0, (region_width - title_width) // 2)
    _place_cells(row, left, "quack", "bold gold1")
    _place_cells(row, left + 5, "space", "bold cyan")
    if tagline:
        _place_cells(row, left + 10, tagline, "dim")


def enabled(force: bool | None = None) -> bool:
    """Return whether animations should run in this process."""
    if force is not None:
        return force
    if os.environ.get("QUACK_NO_ANIM"):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("QUACK_MCP"):
        return False
    return sys.stderr.isatty()


def _frame(
    i: int,
    message: str,
    *,
    done: int | None = None,
    total: int | None = None,
    width: int | None = None,
    baby_count: int | None = None,
    baby_specs: list[tuple[int, int, tuple[str, str, str, str], str]] | None = None,
) -> str:
    width = _pond_width(width)
    crest, surface = _wave_rows(i, width)
    _, top_surface = _wave_rows(i + 7, width)
    dots = "." * ((i // 2) % 4)
    progress = ""
    if done is not None:
        progress = f" [{done}"
        if total is not None:
            progress += f"/{total}"
        progress += "]"
    status = f"{_message(i, message)}{progress}{dots}"
    header_rows = _header_rows(width, status)
    duck_rows = _duck_scene_rows(i, width, baby_count, baby_specs)
    return (
        f"{chr(10).join(header_rows)}\n"
        f"[dim magenta]{top_surface}[/]\n"
        f"{chr(10).join(duck_rows)}\n"
        f"[dim cyan]{crest}[/]\n"
        f"[dim magenta]{surface}[/]"
    )


@dataclass
class DuckProgress:
    """No-op friendly progress handle returned by :func:`swimming`."""

    message: str
    total: int | None = None
    force: bool | None = None
    refresh_per_second: int = 10

    def __post_init__(self) -> None:
        self.done: int | None = 0 if self.total is not None else None
        self._family = _FamilyState.create(refresh_per_second=self.refresh_per_second)
        self._live = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "DuckProgress":
        if not enabled(self.force):
            return self
        try:
            from rich.console import Console
            from rich.live import Live
        except ImportError:
            return self

        console = Console(stderr=True)
        self._console = console
        self._render_markup = console.render_str
        self._live = Live(
            console=console,
            refresh_per_second=self.refresh_per_second,
            transient=True,
        )
        self._live.__enter__()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            if exc_type is None:
                print("quack!", file=sys.stderr)

    def update(
        self,
        done: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            if done is not None:
                self.done = done
            if total is not None:
                self.total = total
            if message is not None:
                self.message = message

    def advance(self, step: int = 1, message: str | None = None) -> None:
        with self._lock:
            self.done = (self.done or 0) + step
            if message is not None:
                self.message = message

    def _animate(self) -> None:
        assert self._live is not None
        delay = 1 / max(self.refresh_per_second, 1)
        last_frame: str | None = None
        last_width: int | None = None
        for i in itertools.count():
            if self._stop.is_set():
                return
            width = getattr(self._console, "width", None)
            pond_width = _pond_width(width)
            baby_specs = self._family.specs(i, pond_width)
            with self._lock:
                frame = _frame(
                    i,
                    self.message,
                    done=self.done,
                    total=self.total,
                    width=width,
                    baby_specs=baby_specs,
                )
            if frame != last_frame or width != last_width:
                self._live.update(self._render_markup(frame))
                last_frame = frame
                last_width = width
            time.sleep(delay)


def swimming(
    message: str,
    *,
    total: int | None = None,
    enabled: bool | None = None,
) -> DuckProgress:
    """Animate a small duck while a human-facing CLI operation runs."""
    return DuckProgress(message=message, total=total, force=enabled)


@contextlib.contextmanager
def disabled():
    """Temporarily silence duck animations in this process."""
    old = os.environ.get("QUACK_NO_ANIM")
    os.environ["QUACK_NO_ANIM"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("QUACK_NO_ANIM", None)
        else:
            os.environ["QUACK_NO_ANIM"] = old


def play(message: str = "") -> None:
    """Loop the duck animation until the user interrupts it."""
    try:
        import rich  # noqa: F401
    except ImportError:
        print("The duck needs `rich` (pip install rich).")
        return

    try:
        with swimming(message, enabled=True):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("quack!")


if __name__ == "__main__":
    play()
