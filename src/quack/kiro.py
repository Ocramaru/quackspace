"""Kiro integration, both directions.

Kiro -> quack:  generate `.kiro/hooks/*.kiro.hook` files so Kiro runs
                `quack` automatically (reindex on save, generate on create).
quack -> Kiro:  `send()` hands a prompt to `kiro-cli chat` so quack can ask
                Kiro's agent to do work (e.g. author a description).

The hook JSON shape follows Kiro's agent-hook format:
    { enabled, name, description, version, when: {type, patterns},
      then: {type, prompt|command} }
Confirm the hooks load in Kiro's Agent Hooks panel; the on-disk schema is
not formally published, so field names may need adjusting per Kiro version.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from .core import find_root

HOOK_VERSION = "1"

# kiro-cli invocation used by quack -> Kiro. Kept here (not config.yaml)
# because this is specifically the Kiro path; config.yaml is the generic one.
KIRO_CHAT = ["kiro-cli", "chat", "--no-interactive", "--trust-all-tools"]


def _hook(name: str, description: str, when: dict, then: dict) -> dict:
    return {
        "enabled": True,
        "name": name,
        "description": description,
        "version": HOOK_VERSION,
        "when": when,
        "then": then,
    }


def hook_definitions() -> dict[str, dict]:
    """The hooks quack installs into .kiro/hooks/.

    Just one: keep the AI navigation layer current on every save. Description
    generation is the on-demand `quack generate` command, not a hook, so
    users wire up their own automation for it if they want one.
    """
    return {
        "quack-reindex-on-save": _hook(
            name="quack: reindex on save",
            description="Regenerate indexes, map, catalog, and diagrams when any file is saved.",
            when={"type": "fileEdited", "patterns": ["**/*"]},
            then={"type": "runCommand", "command": "quack reindex"},
        ),
    }


def install_hooks(explicit_root: str | None = None) -> list[Path]:
    """Write the hook files into <vault>/.kiro/hooks/."""
    root = find_root(explicit_root)
    hooks_dir = root / ".kiro" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slug, defn in hook_definitions().items():
        out = hooks_dir / f"{slug}.kiro.hook"
        out.write_text(json.dumps(defn, indent=2) + "\n")
        written.append(out)
    return written


def send(prompt: str, timeout: int = 120) -> str:
    """quack -> Kiro: send a prompt to kiro-cli and return its response."""
    proc = subprocess.run(
        KIRO_CHAT + [prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"kiro-cli failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()
