"""Small interactive prompt helpers.

Keeping prompt parsing here avoids each setup flow growing its own subtly
different yes/no and choice handling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    key: str
    label: str
    aliases: tuple[str, ...] = ()


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(prompt + suffix).strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def text(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def choice(prompt: str, choices: list[Choice], default: str) -> str:
    """Select from choices: arrow keys + Enter on a real TTY, or type a number."""
    if is_interactive():
        try:
            return _arrow_choice(prompt, choices, default)
        except Exception:
            pass
    return _numeric_choice(prompt, choices, default)


def _numeric_choice(prompt: str, choices: list[Choice], default: str) -> str:
    """Number-keyed fallback for non-interactive or non-Unix contexts."""
    by_answer: dict[str, str] = {}
    for index, item in enumerate(choices, start=1):
        print(f"  {index}. {item.label}")
        by_answer[str(index)] = item.key
        by_answer[item.key.lower()] = item.key
        for alias in item.aliases:
            by_answer[alias.lower()] = item.key
    try:
        answer = input(f"  {prompt} [{default}]: ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return by_answer.get(answer, answer)


def _arrow_choice(prompt: str, choices: list[Choice], default: str) -> str:
    """Arrow-key + number selection on a real TTY.

    Raises on non-Unix or when stdin is not a real tty — caller falls back
    to _numeric_choice. Up/Down moves highlight, Enter confirms, 1-9 jumps
    directly, Escape accepts the current highlight.
    """
    import select as _sel
    import termios
    import tty as _tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)  # raises termios.error if fd is not a real tty

    no_color = bool(os.environ.get("NO_COLOR"))

    def _hl(s: str) -> str:
        return s if no_color else f"\x1b[1;36m{s}\x1b[0m"

    def _dim(s: str) -> str:
        return s if no_color else f"\x1b[2m{s}\x1b[0m"

    n = len(choices)
    idx = next((i for i, c in enumerate(choices) if c.key == default), 0)

    def _draw() -> None:
        sys.stdout.write(f"  {prompt}\n")
        for i, c in enumerate(choices):
            if i == idx:
                sys.stdout.write(f"  {_hl('▶')}  {_hl(c.label)}\n")
            else:
                sys.stdout.write(f"     {_dim(c.label)}\n")
        sys.stdout.flush()

    def _redraw() -> None:
        for _ in range(n + 1):
            sys.stdout.write("\x1b[1A\x1b[2K")
        _draw()

    _draw()
    try:
        _tty.setcbreak(fd)
        while True:
            ch = os.read(fd, 1)
            if ch == b"\x1b":
                ready, _, _ = _sel.select([fd], [], [], 0.05)
                if ready:
                    rest = os.read(fd, 2)
                    if rest == b"[A":
                        idx = (idx - 1) % n
                        _redraw()
                    elif rest == b"[B":
                        idx = (idx + 1) % n
                        _redraw()
            elif ch in (b"\r", b"\n"):
                break
            elif ch.isdigit():
                num = int(ch)
                if 1 <= num <= n:
                    idx = num - 1
                    _redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    selected = choices[idx]
    sys.stdout.write(f"  {_hl('→')} {selected.label}\n")
    sys.stdout.flush()
    return selected.key
