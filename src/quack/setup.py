"""Interactive setup for the AI assistant used to generate descriptions.

`quack setup` shows an arrow-key menu of known assistants (plus a custom and
an opt-out option), probes which are installed, and writes the choice to
.quack/config.yaml. Zero dependencies: it drives the terminal with termios.

The AI is optional. Space works fully without it; the assistant is only used
to generate short descriptions of files and folders for the search index.
"""

from __future__ import annotations

import shutil
import sys
import termios
import tty
from dataclasses import dataclass

from .config import PROVIDERS, write_config

EXPLAINER = (
    "Space uses an AI assistant to write short descriptions of your files and\n"
    "folders. Those descriptions power the search index that lets an LLM find\n"
    "the right note fast. This is optional, you can write descriptions yourself."
)


@dataclass
class SetupResult:
    configured: bool
    command: str
    skipped: bool


def _menu_options() -> list[dict]:
    """Providers plus a 'use without AI' opt-out, with availability probed."""
    opts: list[dict] = []
    for p in PROVIDERS:
        available = p["binary"] is None or shutil.which(p["binary"]) is not None
        suffix = "" if available or p["binary"] is None else "  (not found on PATH)"
        opts.append({**p, "label": p["label"] + suffix, "available": available})
    opts.append(
        {
            "key": "skip",
            "label": "Use Space without AI (write descriptions yourself)",
            "binary": None,
            "command": "",
            "available": True,
        }
    )
    return opts


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_key() -> str:
    """Return 'up', 'down', 'enter', 'quit', or '' for anything else."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("q", "\x03"):  # q or Ctrl-C
            return "quit"
        if ch == "\x1b":  # escape sequence, e.g. arrow keys
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "quit"  # bare Escape
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ""


def _select(options: list[dict]) -> int | None:
    """Arrow-key select over options; returns index or None if cancelled."""
    idx = next((i for i, o in enumerate(options) if o.get("available")), 0)
    print(EXPLAINER + "\n")
    print("Use up/down arrows, Enter to choose, q to cancel.\n")
    rendered = False
    while True:
        if rendered:
            # Move cursor up over the previously drawn lines to redraw in place.
            sys.stdout.write(f"\x1b[{len(options)}A")
        for i, opt in enumerate(options):
            pointer = "›" if i == idx else " "
            line = f" {pointer} {opt['label']}"
            sys.stdout.write("\x1b[2K" + line + "\n")  # clear line, then draw
        sys.stdout.flush()
        rendered = True

        key = _read_key()
        if key == "up":
            idx = (idx - 1) % len(options)
        elif key == "down":
            idx = (idx + 1) % len(options)
        elif key == "enter":
            return idx
        elif key == "quit":
            return None


def run_setup(explicit_root: str | None = None) -> SetupResult:
    """Run the selector and persist the choice. Non-interactive shells get a
    clear message instead of a broken menu."""
    options = _menu_options()

    if not _interactive():
        print(EXPLAINER)
        print(
            "\nNon-interactive shell: edit .quack/config.yaml `ai.command` directly,"
            "\nor run `quack setup` from a terminal."
        )
        return SetupResult(configured=False, command="", skipped=False)

    choice = _select(options)
    if choice is None:
        print("\nCancelled. No changes made.")
        return SetupResult(configured=False, command="", skipped=False)

    chosen = options[choice]

    if chosen["key"] == "skip":
        write_config(command="", explicit_root=explicit_root, skip=True)
        print("\n✓ Set up without AI. Descriptions stay manual.")
        print("  Re-run `quack setup` anytime to add an assistant.")
        return SetupResult(configured=False, command="", skipped=True)

    if chosen["key"] == "custom":
        print("\nEnter the command. Use {prompt} where the prompt goes")
        print("(leave it out to pipe the prompt on stdin):")
        command = input("  command: ").strip()
        if not command:
            print("No command entered. No changes made.")
            return SetupResult(configured=False, command="", skipped=False)
    else:
        command = chosen["command"]
        if not chosen["available"]:
            print(f"\nNote: '{chosen['binary']}' is not on your PATH yet.")
            print("Saved anyway; install it and `quack generate` will work.")

    write_config(command=command, explicit_root=explicit_root)
    print(f"\n✓ Configured: {chosen['label'].split('  (')[0]}")
    print("  Run `quack generate` to fill in missing descriptions.")
    return SetupResult(configured=True, command=command, skipped=False)
