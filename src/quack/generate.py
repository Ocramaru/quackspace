"""AI-backed classification: fill in missing file descriptions **and** tags.

Uses whatever command `.quack/config.yaml` specifies, so the logic is
independent of which assistant runs it. Works on any file — a Markdown note, a
Python module, a config — and writes the result into the editable per-folder
`.index.yaml` store (never into the file itself).
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from . import index_store
from .config import Config
from .core import Space
from .subprocess_utils import failure_message

# CLI assistants wrap output in ANSI color codes and a leading prompt marker.
# Those are terminal-only decoration; in the store they are corruption, so we
# strip them before anything is ever saved.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class AINotConfigured(Exception):
    """Raised when no AI command is set. The CLI offers to run setup."""


def _clean(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.lstrip("> ").strip()




def run_ai(config: Config, prompt: str) -> str:
    """Run the configured AI command and return its cleaned stdout.

    Raises AINotConfigured if no command is set, and a clear RuntimeError if the
    command's binary is missing or it exits non-zero, never a raw traceback.
    """
    if not config.ai.configured:
        raise AINotConfigured()
    cmd = config.ai.command
    argv = (
        shlex.split(cmd) if config.ai.uses_stdin else shlex.split(cmd.replace("{prompt}", prompt))
    )
    stdin = prompt if config.ai.uses_stdin else None
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=config.ai.timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"AI command not found: '{argv[0]}'. Run `quack setup` to choose an "
            "assistant, or fix `ai.command` in .quack/config.yaml."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"AI command timed out after {config.ai.timeout}s running {argv[0]!r}."
        )
    if proc.returncode != 0:
        raise RuntimeError(
            failure_message("AI", argv, proc.returncode, proc.stdout, proc.stderr)
        )
    return _clean(proc.stdout)


def _parse_meta(text: str) -> tuple[str, list[str]]:
    """Parse the model's reply into (description, tags). Tolerates extra prose
    around the JSON; falls back to treating the whole reply as the description."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            desc = str(obj.get("description", "")).strip()
            raw_tags = obj.get("tags") or []
            if isinstance(raw_tags, str):
                raw_tags = raw_tags.split(",")
            tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
            if desc:
                return desc, tags
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return text.strip().strip('"'), []


def _resolve_folder(root, path_or_name: str):
    """Resolve a root-relative path to an existing, non-root content folder, or
    None. Used so ``describe`` can author a *folder* description."""
    if not path_or_name:
        return None
    cand = (root / path_or_name).resolve()
    if cand == root or not cand.is_dir():
        return None
    return cand if root in cand.parents else None


def record(
    explicit_root: str | None,
    path_or_name: str,
    description: str,
    tags: list[str] | None = None,
) -> str | None:
    """Write an authored description + tags for one already-indexed file — or a
    folder — into the relevant `.index.yaml`, stamping `described_at`. A folder
    is authored in its *parent's* `directories:` section. Returns the
    root-relative path, or None if nothing matches. Shared by the `quack
    describe` CLI and the MCP `describe` tool — the way an agent that already
    understands a repo records what it knows."""
    space = Space.load(explicit_root)
    now = datetime.now().isoformat(timespec="seconds")
    entry = space.by_name.get(path_or_name) or next(
        (e for e in space.entries if e.rel == path_or_name), None
    )
    if entry is not None:
        index_store.set_meta(
            entry.path.parent, entry.path.name, description, list(tags or []), now
        )
        return entry.rel

    folder = _resolve_folder(space.root, path_or_name)
    if folder is not None:
        index_store.set_meta(
            folder.parent,
            folder.name,
            description,
            list(tags or []),
            now,
            section=index_store.DIRS_KEY,
        )
        return folder.relative_to(space.root).as_posix()

    return None


@dataclass
class GenResult:
    updated: list[str]
    skipped: list[str]


def fill_descriptions(
    explicit_root: str | None = None,
    only: str | None = None,
    dry_run: bool = False,
    include_stale: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> GenResult:
    """Generate a description + tags for every file missing one (and, when
    `include_stale` is set, every file whose description has gone stale) and
    write them into the per-folder `.index.yaml`.

    `only` restricts to a single file path (used by the file-created hook).
    """
    config = Config.load(explicit_root)
    if not config.ai.configured:
        raise AINotConfigured()
    space = Space.load(explicit_root)
    updated: list[str] = []
    skipped: list[str] = []
    candidates = [
        entry
        for entry in space.entries
        if ((not entry.description) or (include_stale and entry.stale))
        and (not only or entry.rel == only)
    ]
    total = len(candidates)

    for done, entry in enumerate(candidates):
        if progress is not None:
            progress(done, total, f"Generating {entry.rel}")
        content = entry.body.strip()[:4000] or "(empty or binary file)"
        template = config.ai.resolved_generate_prompt
        prompt = (
            template
            .replace("{path}", entry.rel)
            .replace("{ext}", entry.ext or "(none)")
            .replace("{content}", content)
        )
        description, tags = _parse_meta(run_ai(config, prompt))
        if not description:
            skipped.append(entry.rel)
            if progress is not None:
                progress(done + 1, total, f"Skipped {entry.rel}")
            continue
        if dry_run:
            tag_str = f"  [tags: {', '.join(tags)}]" if tags else ""
            updated.append(f"{entry.rel}: {description}{tag_str}")
            if progress is not None:
                progress(done + 1, total, f"Generated {entry.rel}")
            continue
        index_store.set_meta(
            entry.path.parent,
            entry.path.name,
            description,
            tags,
            datetime.now().isoformat(timespec="seconds"),
        )
        updated.append(entry.rel)
        if progress is not None:
            progress(done + 1, total, f"Generated {entry.rel}")

    return GenResult(updated=updated, skipped=skipped)
