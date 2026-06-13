"""A tiny animated duck paddling on the quack pond, shown with `quack --duck`.

Pure decoration. Runs until interrupted (Ctrl-C). `rich` ships transitively via
`mcp[cli]`; if it is ever missing we degrade to a clean message, not a traceback.
"""

from __future__ import annotations

import itertools
import time

# Tail flick (gold), with a splash mid-stroke (cyan).
_TAILS = ["[gold1]/[/]", "[cyan]~[/]", "[gold1]\\[/]"]

# A wider band than we show, so we can slide a window across it for flowing water.
_WAVE = "≈≈∿≈≈∿≈≈∿≈≈∿≈≈∿≈≈∿≈≈∿≈≈∿≈≈∿"

# A bubble drifts up every few frames: (row_from_bottom, char).
_BUBBLES = ["    ", "  ° ", " °  ", "°   "]


def _frame(i: int, message: str) -> str:
    tail = _TAILS[i % len(_TAILS)]
    water = _WAVE[i % 6:][:18]
    dots = "." * (i % 4)
    bubble = _BUBBLES[i % len(_BUBBLES)]
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
        f"[dim italic]{message}{dots}[/]"
    )


def play(message: str = "Paddling the catalog") -> None:
    """Loop the duck animation until the user interrupts it."""
    try:
        from rich.live import Live
        from rich.panel import Panel
    except ImportError:
        print("The duck needs `rich` (pip install rich).")
        return

    try:
        with Live(refresh_per_second=10, transient=True) as live:
            for i in itertools.count():
                live.update(
                    Panel(_frame(i, message), border_style="cyan", width=32)
                )
                time.sleep(0.12)
    except KeyboardInterrupt:
        print("quack!")


if __name__ == "__main__":
    play()
