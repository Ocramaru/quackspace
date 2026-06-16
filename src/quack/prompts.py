"""Small interactive prompt helpers.

Keeping prompt parsing here avoids each setup flow growing its own subtly
different yes/no and choice handling.
"""

from __future__ import annotations

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
