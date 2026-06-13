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

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .core import find_root

CONFIG_NAME = "config.yaml"
DEFAULT_AI_TIMEOUT = 120
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_FILE_CHAR_LIMIT = 20_000
DEFAULT_SQL_ROW_LIMIT = 50
DEFAULT_CENTRAL_LIMIT = 10

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
class DefaultsConfig:
    """Workspace-tunable defaults for agent-facing bounded outputs."""

    search_limit: int = DEFAULT_SEARCH_LIMIT
    file_char_limit: int = DEFAULT_FILE_CHAR_LIMIT
    sql_row_limit: int = DEFAULT_SQL_ROW_LIMIT
    central_limit: int = DEFAULT_CENTRAL_LIMIT


@dataclass
class Config:
    ai: AIConfig
    embed: EmbedConfig
    path: Path | None = None  # the config file that was loaded, if any
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)

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
        defaults_raw = data.get("defaults", {}) or {}
        defaults = DefaultsConfig(
            search_limit=int(defaults_raw.get("search_limit", DEFAULT_SEARCH_LIMIT)),
            file_char_limit=int(defaults_raw.get("file_char_limit", DEFAULT_FILE_CHAR_LIMIT)),
            sql_row_limit=int(defaults_raw.get("sql_row_limit", DEFAULT_SQL_ROW_LIMIT)),
            central_limit=int(defaults_raw.get("central_limit", DEFAULT_CENTRAL_LIMIT)),
        )
        return cls(
            ai=ai,
            embed=embed,
            path=path if path.exists() else None,
            defaults=defaults,
        )


def write_config(
    command: str,
    explicit_root: str | None = None,
    timeout: int = DEFAULT_AI_TIMEOUT,
    skip: bool = False,
) -> Path:
    """Write .quack/config.yaml with the chosen AI command (or an opt-out).

    Existing non-AI settings are user-owned and preserved. In particular,
    `defaults:` controls MCP output limits and should not be reset by setup.
    """
    root = find_root(explicit_root)
    path = root / ".quack" / CONFIG_NAME

    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["ai"] = {"command": command, "timeout": timeout, "skip": skip}
        data.setdefault(
            "defaults",
            {
                "search_limit": DEFAULT_SEARCH_LIMIT,
                "file_char_limit": DEFAULT_FILE_CHAR_LIMIT,
                "sql_row_limit": DEFAULT_SQL_ROW_LIMIT,
                "central_limit": DEFAULT_CENTRAL_LIMIT,
            },
        )
        data.setdefault("gitignore", True)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return path

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
        "\n"
        "defaults:\n"
        "  # Agent-facing output defaults. Tool-call arguments and MCP serve flags\n"
        "  # can override these, but these are the persistent workspace defaults.\n"
        f"  search_limit: {DEFAULT_SEARCH_LIMIT}\n"
        f"  file_char_limit: {DEFAULT_FILE_CHAR_LIMIT}\n"
        f"  sql_row_limit: {DEFAULT_SQL_ROW_LIMIT}\n"
        f"  central_limit: {DEFAULT_CENTRAL_LIMIT}\n"
        "\n"
        "# Set gitignore: false to opt out of quack managing a block in your\n"
        "# repo's .gitignore (and skip .quack/.gitignore creation).\n"
        "gitignore: true\n"
    )
    path.write_text(body)
    return path


def _yaml_scalar(value: str) -> str:
    """Quote a command as a single-line YAML scalar. JSON strings are valid
    YAML flow scalars, so json.dumps handles the embedded quotes and braces."""
    import json

    return json.dumps(value)
