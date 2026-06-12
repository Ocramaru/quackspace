"""Load Space configuration, including the AI command used for generation.

space stays tool-agnostic: it never hardcodes a model. The `.quack/config.yaml`
file says what command turns a prompt into text, so the same generation code
works with kiro-cli, claude, or anything else.

config.yaml shape:

    ai:
      # {prompt} is substituted with the generation prompt. If omitted, the
      # prompt is piped to the command on stdin instead.
      command: kiro-cli chat --no-interactive --trust-all-tools "{prompt}"
      timeout: 120        # seconds

When no command is set, the AI is "not configured": `quack generate` offers
to run `quack setup` rather than failing with a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .core import find_root

CONFIG_NAME = "config.yaml"
DEFAULT_AI_TIMEOUT = 120

# Known assistants the setup selector offers. `binary` is what we probe for on
# PATH; `command` is written into config.yaml. Order is the menu order.
PROVIDERS: list[dict] = [
    {
        "key": "kiro",
        "label": "Kiro (kiro-cli)",
        "binary": "kiro-cli",
        "command": 'kiro-cli chat --no-interactive --trust-all-tools "{prompt}"',
    },
    {
        "key": "claude",
        "label": "Claude Code (claude)",
        "binary": "claude",
        "command": 'claude -p "{prompt}"',
    },
    {
        "key": "custom",
        "label": "Custom command (edit config.yaml yourself)",
        "binary": None,
        "command": "",
    },
]


@dataclass
class AIConfig:
    command: str = ""
    timeout: int = DEFAULT_AI_TIMEOUT
    skip: bool = False  # user opted out of AI; never prompt to set it up again

    @property
    def configured(self) -> bool:
        return bool(self.command.strip())

    @property
    def uses_stdin(self) -> bool:
        """True when the prompt is piped on stdin rather than substituted."""
        return "{prompt}" not in self.command


@dataclass
class EmbedConfig:
    """Command that turns text into an embedding vector (JSON array of floats).
    {text} is substituted, or the text is piped on stdin if {text} is absent."""

    command: str = ""
    dim: int = 0  # vector dimension; 0 = infer from first embedding
    timeout: int = DEFAULT_AI_TIMEOUT

    @property
    def configured(self) -> bool:
        return bool(self.command.strip())

    @property
    def uses_stdin(self) -> bool:
        return "{text}" not in self.command


@dataclass
class Config:
    ai: AIConfig
    embed: EmbedConfig
    path: Path | None  # the config file that was loaded, if any

    @classmethod
    def load(cls, explicit_root: str | None = None) -> "Config":
        root = find_root(explicit_root)
        path = root / ".quack" / CONFIG_NAME
        data = {}
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
        ai_raw = data.get("ai", {}) or {}
        ai = AIConfig(
            command=str(ai_raw.get("command", "") or ""),
            timeout=int(ai_raw.get("timeout", DEFAULT_AI_TIMEOUT)),
            skip=bool(ai_raw.get("skip", False)),
        )
        emb_raw = data.get("embed", {}) or {}
        embed = EmbedConfig(
            command=str(emb_raw.get("command", "") or ""),
            dim=int(emb_raw.get("dim", 0)),
            timeout=int(emb_raw.get("timeout", DEFAULT_AI_TIMEOUT)),
        )
        return cls(ai=ai, embed=embed, path=path if path.exists() else None)


def write_config(
    command: str,
    explicit_root: str | None = None,
    timeout: int = DEFAULT_AI_TIMEOUT,
    skip: bool = False,
) -> Path:
    """Write .quack/config.yaml with the chosen AI command (or an opt-out)."""
    root = find_root(explicit_root)
    path = root / ".quack" / CONFIG_NAME
    body = (
        "# Space configuration.\n\n"
        "ai:\n"
        "  # Command that turns a prompt into text. {prompt} is substituted with\n"
        "  # the generation prompt; if you omit {prompt}, the prompt is piped on\n"
        "  # stdin. Swap this line to use a different assistant.\n"
        f"  command: {_yaml_scalar(command)}\n"
        f"  timeout: {timeout}\n"
        "  # Set skip: true to use Space without AI; descriptions stay manual and\n"
        "  # `quack generate` will not offer to set up an assistant.\n"
        f"  skip: {'true' if skip else 'false'}\n"
    )
    path.write_text(body)
    return path


def _yaml_scalar(value: str) -> str:
    """Quote a command as a single-line YAML scalar. JSON strings are valid
    YAML flow scalars, so json.dumps handles the embedded quotes and braces."""
    import json

    return json.dumps(value)
