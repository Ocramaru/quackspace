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

    index:
      store_body: true     # set false to keep file text out of DuckDB

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
DEFAULT_HIDDEN_DIR_PENALTY = 1.0
DEFAULT_STORE_BODY = True
DEFAULT_DIAGRAMS = True

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
    provider: str = ""
    dim: int = 0  # vector dimension; 0 = infer from first embedding
    timeout: int = DEFAULT_AI_TIMEOUT
    include_body: bool = True
    skip: bool = False  # user opted out of embeddings; never prompt to set up again

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
    hidden_dir_penalty: float = DEFAULT_HIDDEN_DIR_PENALTY


@dataclass
class IndexConfig:
    """Workspace-tunable indexing behavior."""

    store_body: bool = DEFAULT_STORE_BODY
    diagrams: bool = DEFAULT_DIAGRAMS


@dataclass
class Config:
    ai: AIConfig
    embed: EmbedConfig
    path: Path | None = None  # the config file that was loaded, if any
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

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
            provider=str(emb_raw.get("provider", "") or ""),
            dim=int(emb_raw.get("dim", 0)),
            timeout=int(emb_raw.get("timeout", DEFAULT_AI_TIMEOUT)),
            include_body=_bool_config(emb_raw.get("include_body"), True),
            skip=bool(emb_raw.get("skip", False)),
        )
        defaults_raw = data.get("defaults", {}) or {}
        defaults = DefaultsConfig(
            search_limit=int(defaults_raw.get("search_limit", DEFAULT_SEARCH_LIMIT)),
            file_char_limit=int(defaults_raw.get("file_char_limit", DEFAULT_FILE_CHAR_LIMIT)),
            sql_row_limit=int(defaults_raw.get("sql_row_limit", DEFAULT_SQL_ROW_LIMIT)),
            central_limit=int(defaults_raw.get("central_limit", DEFAULT_CENTRAL_LIMIT)),
            hidden_dir_penalty=float(defaults_raw.get("hidden_dir_penalty", DEFAULT_HIDDEN_DIR_PENALTY)),
        )
        index_raw = data.get("index", {}) or {}
        if not isinstance(index_raw, dict):
            index_raw = {}
        index = IndexConfig(
            store_body=_bool_config(index_raw.get("store_body"), DEFAULT_STORE_BODY),
            diagrams=_bool_config(index_raw.get("diagrams"), DEFAULT_DIAGRAMS),
        )
        return cls(
            ai=ai,
            embed=embed,
            path=path if path.exists() else None,
            defaults=defaults,
            index=index,
        )


def _default_defaults_config() -> dict:
    return {
        "search_limit": DEFAULT_SEARCH_LIMIT,
        "file_char_limit": DEFAULT_FILE_CHAR_LIMIT,
        "sql_row_limit": DEFAULT_SQL_ROW_LIMIT,
        "central_limit": DEFAULT_CENTRAL_LIMIT,
    }


def _default_embed_config() -> dict:
    return {
        "provider": "builtin",
        "command": "quack embed text",
        "dim": 256,
        "timeout": DEFAULT_AI_TIMEOUT,
        "include_body": True,
    }


def _load_config_data(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_config_shape(data: dict) -> dict:
    data.setdefault("ai", {"command": "", "timeout": DEFAULT_AI_TIMEOUT, "skip": False})
    data.setdefault("defaults", _default_defaults_config())
    if not isinstance(data.get("index"), dict):
        data["index"] = {}
    data["index"].setdefault("store_body", DEFAULT_STORE_BODY)
    data["index"].setdefault("diagrams", DEFAULT_DIAGRAMS)
    data.setdefault("embed", _default_embed_config())
    if not isinstance(data.get("embed"), dict):
        data["embed"] = _default_embed_config()
    data["embed"].setdefault("include_body", True)
    data.setdefault("gitignore", True)
    return data


def _write_config_data(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


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
        data = _ensure_config_shape(_load_config_data(path))
        data["ai"] = {"command": command, "timeout": timeout, "skip": skip}
        _write_config_data(path, data)
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
        "index:\n"
        "  # Store file text in DuckDB for body full-text search. Set false to\n"
        "  # keep only path/metadata/links in the catalog; run `quack reindex`\n"
        "  # after changing this so old catalog rows are rebuilt.\n"
        f"  store_body: {'true' if DEFAULT_STORE_BODY else 'false'}\n"
        "  # Generate Mermaid link diagrams during `quack reindex` when folder\n"
        "  # indexes changed. Use `quack reindex --no-diagrams` to skip once.\n"
        f"  diagrams: {'true' if DEFAULT_DIAGRAMS else 'false'}\n"
        "\n"
        "embed:\n"
        "  # Free local default. Run `quack embed init` to choose Ollama or\n"
        "  # another provider, or edit this command directly. The command must\n"
        "  # print one JSON array of floats; if {text} is omitted, text is piped\n"
        "  # on stdin.\n"
        "  provider: builtin\n"
        "  command: quack embed text\n"
        "  dim: 256\n"
        f"  timeout: {DEFAULT_AI_TIMEOUT}\n"
        "  # Set false to embed only path/name/type/tags/description/links,\n"
        "  # without raw file body content.\n"
        "  include_body: true\n"
        "\n"
        "# Set gitignore: false to opt out of quack managing a block in your\n"
        "# repo's .gitignore (and skip .quack/.gitignore creation).\n"
        "gitignore: true\n"
    )
    path.write_text(body)
    return path


def write_embed_config(
    command: str,
    explicit_root: str | None = None,
    dim: int = 0,
    timeout: int = DEFAULT_AI_TIMEOUT,
    provider: str = "custom",
    skip: bool = False,
) -> Path:
    """Write the embedding command, preserving the rest of config.yaml."""
    root = find_root(explicit_root)
    path = root / ".quack" / CONFIG_NAME
    data = _ensure_config_shape(_load_config_data(path))
    data["embed"] = {
        "provider": provider,
        "command": command,
        "dim": dim,
        "timeout": timeout,
        "include_body": _bool_config(data.get("embed", {}).get("include_body"), True),
        "skip": skip,
    }
    _write_config_data(path, data)
    return path


def write_embed_skip(explicit_root: str | None = None) -> Path:
    """Permanently opt out of embedding setup prompts (embed.skip: true)."""
    root = find_root(explicit_root)
    path = root / ".quack" / CONFIG_NAME
    data = _ensure_config_shape(_load_config_data(path))
    data.setdefault("embed", {})["skip"] = True
    _write_config_data(path, data)
    return path


def _bool_config(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _yaml_scalar(value: str) -> str:
    """Quote a command as a single-line YAML scalar. JSON strings are valid
    YAML flow scalars, so json.dumps handles the embedded quotes and braces."""
    import json

    return json.dumps(value)
