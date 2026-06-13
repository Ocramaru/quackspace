"""Small helpers for subprocess-backed provider commands."""

from __future__ import annotations


def failure_message(
    kind: str,
    argv: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    max_detail: int = 800,
) -> str:
    """Format a non-zero subprocess result without dumping huge prompts/output."""
    detail = stderr.strip() or stdout.strip() or "no output on stderr or stdout"
    if len(detail) > max_detail:
        detail = detail[: max_detail - 3] + "..."
    command = argv[0] if argv else "<empty command>"
    return f"{kind} command failed ({returncode}) running {command!r}: {detail}"
