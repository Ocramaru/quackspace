"""Tiny terminal duck animations for human-facing CLI waits.

Pure decoration: this module must never affect command behavior or exit codes.
It auto-disables for non-TTY output, ``NO_COLOR``, ``QUACK_NO_ANIM``, and when
``rich`` is unavailable.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import sys
import threading
import time
from dataclasses import dataclass

# Tail flick, with a splash mid-stroke.
_TAILS = ["/", "~", "\\"]

# A wider band than we show, so we can slide a window across it for flowing water.
_WAVE = "~~~^~~~^~~~^~~~^~~~^~~~^~~~^~~~^~~~^"

_BUBBLES = ["    ", "  o ", " o  ", "o   "]


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


def _frame(i: int, message: str, *, done: int | None = None, total: int | None = None) -> str:
    tail = _TAILS[i % len(_TAILS)]
    water = _WAVE[i % 6:][:18]
    dots = "." * (i % 4)
    bubble = _BUBBLES[i % len(_BUBBLES)]
    progress = ""
    if done is not None:
        progress = f" [{done}"
        if total is not None:
            progress += f"/{total}"
        progress += "]"
    duck = (
        f"      [orange1]__[/]   [dim cyan]{bubble}[/]\n"
        f"   [gold1]<([/][bold white]o[/][gold1] )___[/]\n"
        f"    [gold1]( ._> {tail} [gold1])[/]\n"
        f"     [gold1]`-----'[/]"
    )
    return (
        "[bold gold1]quack[/][bold cyan]space[/]\n\n"
        f"{duck}\n"
        f"[blue]{water}[/]\n\n"
        f"[dim italic]{message}{progress}{dots}[/]"
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
        self._live = None
        self._panel = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "DuckProgress":
        if not enabled(self.force):
            return self
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.panel import Panel
        except ImportError:
            return self

        self._panel = Panel
        console = Console(stderr=True)
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
        assert self._panel is not None
        delay = 1 / max(self.refresh_per_second, 1)
        for i in itertools.count():
            if self._stop.is_set():
                return
            with self._lock:
                frame = _frame(i, self.message, done=self.done, total=self.total)
            self._live.update(self._panel(frame, border_style="cyan", width=34))
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


def play(message: str = "Paddling the catalog") -> None:
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
