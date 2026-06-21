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
      store_body: true            # set false to keep file text out of DuckDB
      dataset_threshold: 10000     # folders over this many files are skipped
      dataset_ext_threshold: 500   # or this many of one bulk-data type (.npy, .png…)

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
DEFAULT_SQL_ROW_LIMIT = 50
DEFAULT_CENTRAL_LIMIT = 10
DEFAULT_MAP_AUTO_ITEMS = 50
DEFAULT_HIDDEN_DIR_PENALTY = 1.0
DEFAULT_LOCAL_DIR_BOOST = 1.5
DEFAULT_STORE_BODY = True
DEFAULT_DIAGRAMS = True
DEFAULT_DATASET_THRESHOLD = 10_000
DEFAULT_DATASET_EXT_THRESHOLD = 500
DEFAULT_BODY_MAX_BYTES = 1_000_000
DEFAULT_TAG_ROLLUP_LIMIT = 5
DEFAULT_EMBED_BODY_CHAR_LIMIT = 4_000
DEFAULT_EMBED_TEXT_CHAR_LIMIT = 20_000
DEFAULT_RRF_K = 60
DEFAULT_WEIGHT_NAME = 10
DEFAULT_WEIGHT_TAG = 6
DEFAULT_WEIGHT_DESCRIPTION = 4
DEFAULT_LAKE_ENABLED = True
DEFAULT_LAKE_SNAPSHOT = True
DEFAULT_LAKE_SIZE_THRESHOLD_MB = 200
DEFAULT_LAKE_ROW_THRESHOLD = 100_000
DEFAULT_DIAGRAM_MAX_DEPTH = 3
DEFAULT_DATASET_EXTENSIONS: frozenset[str] = frozenset({
    "npy", "npz", "pt", "pth", "ckpt", "safetensors", "onnx", "pb",
    "h5", "hdf5", "tfrecord", "mat", "pkl", "pickle", "bin",
    "parquet", "arrow", "feather",
    "png", "jpg", "jpeg", "bmp", "gif", "tiff", "tif", "webp",
    "wav", "flac", "mp3", "ogg", "mp4", "mov", "avi", "mkv",
    "ply", "pcd",
})
# "Bodyless" files DO get an embedding vector, but built from metadata only
# (path/name/type/tags/description/links) — their raw bytes are skipped because
# they are bulky or not prose. This is for text-ish-but-unhelpful-body formats;
# genuinely binary/non-text formats are handled one tier up by
# DEFAULT_NONEMBEDDABLE_* (no vector at all), so they are deliberately NOT
# repeated here.
DEFAULT_BODYLESS_EMBED_TAGS: frozenset[str] = frozenset({
    "data",
})
DEFAULT_BODYLESS_EMBED_EXTENSIONS: frozenset[str] = frozenset({
    "csv", "jsonl", "log", "pdf", "svg", "tsv", "xls", "xlsx",
})
# Stricter than "bodyless": files whose extension is here get NO embedding
# vector at all (not even a metadata-only one). These are genuinely non-text /
# binary formats — or content-free sidecars — so the only signal they could
# contribute is their path, which the structural and full-text tiers already
# cover. Skipping them outright is the difference between embedding a few
# thousand meaningful files and embedding hundreds of thousands of asset blobs.
# A file with an authored/generated description is embedded regardless (the
# description is real signal); see catalog.embeddable.
DEFAULT_NONEMBEDDABLE_EXTENSIONS: frozenset[str] = frozenset({
    # images
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp", "ico", "avif", "ai",
    # audio / video
    "mp3", "wav", "flac", "ogg", "aac", "m4a", "mp4", "mov", "avi", "mkv", "webm",
    # archives
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
    # fonts
    "woff", "woff2", "ttf", "otf", "eot",
    # databases
    "db", "duckdb", "sqlite", "sqlite3",
    # compiled / binary artifacts
    "bin", "exe", "dll", "so", "dylib", "o", "a", "class", "wasm", "pyc", "pyd",
    # ML tensors / serialized binary data
    "npy", "npz", "pt", "pth", "ckpt", "safetensors", "onnx", "pb",
    "h5", "hdf5", "tfrecord", "mat", "pkl", "pickle", "parquet", "arrow", "feather",
    # Unity GUID sidecars: text, but pure per-asset import-settings noise.
    "meta",
})
DEFAULT_NONEMBEDDABLE_TAGS: frozenset[str] = frozenset({
    "assets", "dependencies", "lockfile",
})
# Folders that are walked and indexed — so their files stay findable by name and
# full-text search — but whose files are never embedded. These are large,
# regenerated trees (e.g. a Unity project's import cache) where per-file semantic
# vectors add no value and would dominate the embed run. Matched on any ancestor
# directory name. Contrast with core.DEFAULT_OPAQUE_DIRS, which are not even
# descended into. A described file under one of these is still embedded.
DEFAULT_NONEMBEDDABLE_DIRS: frozenset[str] = frozenset({
    "Library", "Temp", "Logs", "MemoryCaptures", "UserSettings",
})

# Default prompt template for `quack generate`. Use {path}, {ext}, {content} as
# placeholders — all other { } characters are treated literally (no escaping needed).
DEFAULT_GENERATE_PROMPT = """\
Classify the following file for a search index. Output ONLY a JSON object on one
line, no preamble and no code fences:
{"description": "<one sentence, max 25 words>", "tags": ["<lowercase topic>", ...]}
Use 1-5 short topical tags (language, role, domain). If the content is empty
(binary file), infer from the path and type.

Path: {path}
Type: {ext}

Content (may be truncated):
{content}
"""

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
    generate_prompt: str = ""  # empty → DEFAULT_GENERATE_PROMPT at runtime

    @property
    def configured(self) -> bool:
        return bool(self.command.strip())

    @property
    def uses_stdin(self) -> bool:
        """True when the prompt is piped on stdin rather than substituted."""
        return "{prompt}" not in self.command

    @property
    def resolved_generate_prompt(self) -> str:
        return self.generate_prompt.strip() or DEFAULT_GENERATE_PROMPT


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
    body_char_limit: int = DEFAULT_EMBED_BODY_CHAR_LIMIT
    text_char_limit: int = DEFAULT_EMBED_TEXT_CHAR_LIMIT
    bodyless_tags: list[str] = field(default_factory=list)
    bodyless_extensions: list[str] = field(default_factory=list)
    nonembeddable_tags: list[str] = field(default_factory=list)
    nonembeddable_extensions: list[str] = field(default_factory=list)
    nonembeddable_dirs: list[str] = field(default_factory=list)

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
    sql_row_limit: int = DEFAULT_SQL_ROW_LIMIT
    central_limit: int = DEFAULT_CENTRAL_LIMIT
    map_auto_items: int = DEFAULT_MAP_AUTO_ITEMS
    hidden_dir_penalty: float = DEFAULT_HIDDEN_DIR_PENALTY
    local_dir_boost: float = DEFAULT_LOCAL_DIR_BOOST
    rrf_k: int = DEFAULT_RRF_K
    weight_name: int = DEFAULT_WEIGHT_NAME
    weight_tag: int = DEFAULT_WEIGHT_TAG
    weight_description: int = DEFAULT_WEIGHT_DESCRIPTION


@dataclass
class IndexConfig:
    """Workspace-tunable indexing behavior."""

    store_body: bool = DEFAULT_STORE_BODY
    diagrams: bool = DEFAULT_DIAGRAMS
    dataset_threshold: int = DEFAULT_DATASET_THRESHOLD
    dataset_ext_threshold: int = DEFAULT_DATASET_EXT_THRESHOLD
    dataset_extensions: list[str] = field(default_factory=list)
    body_max_bytes: int = DEFAULT_BODY_MAX_BYTES
    tag_rollup_limit: int = DEFAULT_TAG_ROLLUP_LIMIT
    opaque_dirs: list[str] = field(default_factory=list)
    diagram_max_depth: int = DEFAULT_DIAGRAM_MAX_DEPTH


@dataclass
class LakeConfig:
    """Controls DuckLake Parquet-backed catalog snapshots and auto-tiering."""

    enabled: bool = DEFAULT_LAKE_ENABLED
    snapshot_on_reindex: bool = DEFAULT_LAKE_SNAPSHOT
    size_threshold_mb: int = DEFAULT_LAKE_SIZE_THRESHOLD_MB
    row_threshold: int = DEFAULT_LAKE_ROW_THRESHOLD


@dataclass
class Config:
    ai: AIConfig
    embed: EmbedConfig
    path: Path | None = None  # the config file that was loaded, if any
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    lake: LakeConfig = field(default_factory=LakeConfig)

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
            generate_prompt=str(ai_raw.get("generate_prompt", "") or ""),
        )
        emb_raw = data.get("embed", {}) or {}
        embed = EmbedConfig(
            command=str(emb_raw.get("command", "") or ""),
            provider=str(emb_raw.get("provider", "") or ""),
            dim=int(emb_raw.get("dim", 0)),
            timeout=int(emb_raw.get("timeout", DEFAULT_AI_TIMEOUT)),
            include_body=_bool_config(emb_raw.get("include_body"), True),
            skip=bool(emb_raw.get("skip", False)),
            body_char_limit=_int_config(emb_raw.get("body_char_limit"), DEFAULT_EMBED_BODY_CHAR_LIMIT),
            text_char_limit=_int_config(emb_raw.get("text_char_limit"), DEFAULT_EMBED_TEXT_CHAR_LIMIT),
            bodyless_tags=_name_list_config(emb_raw.get("bodyless_tags")),
            bodyless_extensions=_str_list_config(emb_raw.get("bodyless_extensions")),
            nonembeddable_tags=_name_list_config(emb_raw.get("nonembeddable_tags")),
            nonembeddable_extensions=_str_list_config(emb_raw.get("nonembeddable_extensions")),
            nonembeddable_dirs=_name_list_config(emb_raw.get("nonembeddable_dirs")),
        )
        defaults_raw = data.get("defaults", {}) or {}
        defaults = DefaultsConfig(
            search_limit=int(defaults_raw.get("search_limit", DEFAULT_SEARCH_LIMIT)),
            sql_row_limit=int(defaults_raw.get("sql_row_limit", DEFAULT_SQL_ROW_LIMIT)),
            central_limit=int(defaults_raw.get("central_limit", DEFAULT_CENTRAL_LIMIT)),
            map_auto_items=_int_config(defaults_raw.get("map_auto_items"), DEFAULT_MAP_AUTO_ITEMS),
            hidden_dir_penalty=float(defaults_raw.get("hidden_dir_penalty", DEFAULT_HIDDEN_DIR_PENALTY)),
            local_dir_boost=float(defaults_raw.get("local_dir_boost", DEFAULT_LOCAL_DIR_BOOST)),
            rrf_k=_int_config(defaults_raw.get("rrf_k"), DEFAULT_RRF_K),
            weight_name=_int_config(defaults_raw.get("weight_name"), DEFAULT_WEIGHT_NAME),
            weight_tag=_int_config(defaults_raw.get("weight_tag"), DEFAULT_WEIGHT_TAG),
            weight_description=_int_config(defaults_raw.get("weight_description"), DEFAULT_WEIGHT_DESCRIPTION),
        )
        index_raw = data.get("index", {}) or {}
        if not isinstance(index_raw, dict):
            index_raw = {}
        raw_depth = index_raw.get("diagram_max_depth", DEFAULT_DIAGRAM_MAX_DEPTH)
        try:
            diagram_max_depth = max(0, int(raw_depth))
        except (TypeError, ValueError):
            diagram_max_depth = DEFAULT_DIAGRAM_MAX_DEPTH
        index = IndexConfig(
            store_body=_bool_config(index_raw.get("store_body"), DEFAULT_STORE_BODY),
            diagrams=_bool_config(index_raw.get("diagrams"), DEFAULT_DIAGRAMS),
            dataset_threshold=_int_config(
                index_raw.get("dataset_threshold"), DEFAULT_DATASET_THRESHOLD
            ),
            dataset_ext_threshold=_int_config(
                index_raw.get("dataset_ext_threshold"), DEFAULT_DATASET_EXT_THRESHOLD
            ),
            dataset_extensions=_str_list_config(index_raw.get("dataset_extensions")),
            body_max_bytes=_int_config(
                index_raw.get("body_max_bytes"), DEFAULT_BODY_MAX_BYTES
            ),
            tag_rollup_limit=_int_config(
                index_raw.get("tag_rollup_limit"), DEFAULT_TAG_ROLLUP_LIMIT
            ),
            opaque_dirs=_name_list_config(index_raw.get("opaque_dirs")),
            diagram_max_depth=diagram_max_depth,
        )
        lake_raw = data.get("lake", {}) or {}
        if not isinstance(lake_raw, dict):
            lake_raw = {}
        lake = LakeConfig(
            enabled=_bool_config(lake_raw.get("enabled"), DEFAULT_LAKE_ENABLED),
            snapshot_on_reindex=_bool_config(lake_raw.get("snapshot_on_reindex"), DEFAULT_LAKE_SNAPSHOT),
            size_threshold_mb=_int_config(lake_raw.get("size_threshold_mb"), DEFAULT_LAKE_SIZE_THRESHOLD_MB),
            row_threshold=_int_config(lake_raw.get("row_threshold"), DEFAULT_LAKE_ROW_THRESHOLD),
        )
        return cls(
            ai=ai,
            embed=embed,
            path=path if path.exists() else None,
            defaults=defaults,
            index=index,
            lake=lake,
        )


# --- One source of truth for config.yaml -----------------------------------
#
# Each section is an ordered list of knobs (and commented-out hints). This spec
# drives BOTH the generated config.yaml template (with comments) AND the
# default-/shape-ensuring on load, so the create path and the update path can
# never drift — the duplication that previously let keys go missing from one
# place but not another. Typed access still flows through the dataclasses and
# `Config.load`; this only governs defaults and how the file is written.


@dataclass(frozen=True)
class _Knob:
    """One config key: its default value, an optional inline comment, and
    optional comment lines emitted just above it. `quote` forces YAML quoting
    for values that may contain `:`/quotes/braces (the AI command)."""

    key: str
    default: object
    comment: str = ""
    pre: tuple[str, ...] = ()
    quote: bool = False


@dataclass(frozen=True)
class _Hint:
    """Commented-out lines emitted verbatim (an optional/advanced key shown only
    as an example). Contributes nothing to the default shape."""

    lines: tuple[str, ...]


@dataclass(frozen=True)
class _Section:
    name: str
    header: tuple[str, ...]
    items: tuple


_CONFIG_SPEC: tuple[_Section, ...] = (
    _Section("ai", (), (
        _Knob("command", "", quote=True, pre=(
            "Command that turns a prompt into text. {prompt} is substituted with",
            "the generation prompt; if you omit {prompt}, the prompt is piped on",
            "stdin. Swap this line to use a different assistant.",
        )),
        _Knob("timeout", DEFAULT_AI_TIMEOUT),
        _Knob("skip", False, pre=(
            "Set skip: true to use Space without AI; descriptions stay manual and",
            "`quack generate` will not offer to set up an assistant.",
        )),
        _Hint((
            "Customize the prompt `quack generate` sends to the AI. Use {path},",
            "{ext}, and {content} as placeholders; all other { } are literal.",
            "Omit this key to use the built-in default prompt.",
            "generate_prompt: |",
            "  Classify the following file ...",
        )),
    )),
    _Section("defaults", (
        "Agent-facing output defaults. Tool-call arguments and MCP serve flags",
        "can override these, but these are the persistent workspace defaults.",
    ), (
        _Knob("search_limit", DEFAULT_SEARCH_LIMIT),
        _Knob("sql_row_limit", DEFAULT_SQL_ROW_LIMIT),
        _Knob("central_limit", DEFAULT_CENTRAL_LIMIT),
        _Knob("map_auto_items", DEFAULT_MAP_AUTO_ITEMS,
              "`quack map` auto-depth target: expand until ~this many entries"),
        _Knob("hidden_dir_penalty", DEFAULT_HIDDEN_DIR_PENALTY,
              "search rank penalty for hits under hidden/.dot dirs"),
        _Knob("local_dir_boost", DEFAULT_LOCAL_DIR_BOOST,
              "search rank boost for hits under the current working dir"),
        _Knob("rrf_k", DEFAULT_RRF_K,
              "reciprocal rank fusion constant (higher = flatter score spread)"),
        _Knob("weight_name", DEFAULT_WEIGHT_NAME, "structural tier: name match score"),
        _Knob("weight_tag", DEFAULT_WEIGHT_TAG, "structural tier: tag match score"),
        _Knob("weight_description", DEFAULT_WEIGHT_DESCRIPTION,
              "structural tier: description match score"),
    )),
    _Section("index", (), (
        _Knob("store_body", DEFAULT_STORE_BODY, pre=(
            "Store file text in DuckDB for body full-text search. Set false to",
            "keep only path/metadata/links in the catalog; run `quack reindex`",
            "after changing this so old catalog rows are rebuilt.",
        )),
        _Knob("diagrams", DEFAULT_DIAGRAMS, pre=(
            "Generate Mermaid link diagrams during `quack reindex` when folder",
            "indexes changed. Use `quack reindex --no-diagrams` to skip once.",
        )),
        _Knob("dataset_threshold", DEFAULT_DATASET_THRESHOLD),
        _Knob("dataset_ext_threshold", DEFAULT_DATASET_EXT_THRESHOLD),
        _Knob("dataset_extensions", [],
              "extra extensions that count toward dataset_ext_threshold"),
        _Knob("body_max_bytes", DEFAULT_BODY_MAX_BYTES,
              "max bytes read from each file for FTS body"),
        _Knob("tag_rollup_limit", DEFAULT_TAG_ROLLUP_LIMIT,
              "top N tags surfaced in folder rollups"),
        _Knob("opaque_dirs", [],
              "additional dir names to record but not descend (e.g. build outputs)"),
        _Knob("diagram_max_depth", DEFAULT_DIAGRAM_MAX_DEPTH,
              "global link diagram includes folders up to this depth"),
    )),
    _Section("embed", (
        "Free local default. Run `quack embed init` to choose Ollama or",
        "another provider, or edit this command directly. The command must",
        "print one JSON array of floats; if {text} is omitted, text is piped",
        "on stdin.",
    ), (
        _Knob("provider", "builtin"),
        _Knob("command", "quack embed text"),
        _Knob("dim", 256),
        _Knob("timeout", DEFAULT_AI_TIMEOUT),
        _Knob("include_body", True, pre=(
            "Set false to embed only path/name/type/tags/description/links,",
            "without raw file body content.",
        )),
        _Knob("body_char_limit", DEFAULT_EMBED_BODY_CHAR_LIMIT,
              "max body chars in per-file embed text"),
        _Knob("text_char_limit", DEFAULT_EMBED_TEXT_CHAR_LIMIT,
              "max total chars sent to the embedding model"),
        _Knob("bodyless_tags", [],
              "additional tags whose files embed metadata only (no body)"),
        _Knob("bodyless_extensions", [],
              "additional extensions that embed metadata only (no body)"),
        _Knob("nonembeddable_tags", [],
              "additional tags whose files skip embedding entirely", pre=(
            "Stricter than bodyless: files matching these get NO embedding vector",
            "at all (built-in defaults already cover images, audio/video, archives,",
            "fonts, databases, binaries, tensors, and Unity .meta sidecars). A file",
            "with a description is always embedded. These extend the defaults.",
        )),
        _Knob("nonembeddable_extensions", [],
              "additional extensions that skip embedding entirely"),
        _Knob("nonembeddable_dirs", [],
              "folders walked+indexed but never embedded (e.g. Unity Library)"),
    )),
    _Section("lake", (
        "DuckLake Parquet-backed catalog snapshots and auto-tiering.",
        "When enabled, each reindex snapshots files+folders to .quack/lake_data/.",
    ), (
        _Knob("enabled", DEFAULT_LAKE_ENABLED),
        _Knob("snapshot_on_reindex", DEFAULT_LAKE_SNAPSHOT),
        _Knob("size_threshold_mb", DEFAULT_LAKE_SIZE_THRESHOLD_MB, pre=(
            "Tier body text to DuckLake when quack.duckdb exceeds this size (MB).",
        )),
        _Knob("row_threshold", DEFAULT_LAKE_ROW_THRESHOLD, pre=(
            "Or when the files table exceeds this many rows.",
        )),
    )),
)


def _yaml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[]"
    return str(v)


def _canonical_defaults() -> dict:
    """The default value of every config key, by section, derived from the spec.
    Hints contribute nothing. The single source the shape-ensuring uses."""
    out: dict[str, dict] = {}
    for sec in _CONFIG_SPEC:
        section: dict[str, object] = {}
        for item in sec.items:
            if isinstance(item, _Knob):
                section[item.key] = list(item.default) if isinstance(item.default, list) else item.default
        out[sec.name] = section
    return out


def _render_config_yaml(overrides: dict) -> str:
    """Render the full commented config.yaml from the spec. `overrides` supplies
    runtime values (the chosen AI command/timeout/skip) keyed by (section, key)."""
    parts: list[str] = ["# Space configuration.\n\n"]
    for sec in _CONFIG_SPEC:
        parts.append(f"{sec.name}:\n")
        for line in sec.header:
            parts.append(f"  # {line}\n")
        for item in sec.items:
            if isinstance(item, _Hint):
                for line in item.lines:
                    parts.append(f"  # {line}\n")
                continue
            for line in item.pre:
                parts.append(f"  # {line}\n")
            value = overrides.get((sec.name, item.key), item.default)
            rendered = _yaml_scalar(value) if item.quote else _yaml_value(value)
            line = f"  {item.key}: {rendered}"
            if item.comment:
                line += f"  # {item.comment}"
            parts.append(line + "\n")
        parts.append("\n")
    parts.append(
        "# Set gitignore: false to opt out of quack managing a block in your\n"
        "# repo's .gitignore (and skip .quack/.gitignore creation).\n"
        "gitignore: true\n"
    )
    return "".join(parts)


def _load_config_data(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_config_shape(data: dict) -> dict:
    """Fill in any missing default keys without touching values the user set.
    Driven entirely by `_canonical_defaults()` so it can't drift from the
    written template."""
    for section, keys in _canonical_defaults().items():
        if not isinstance(data.get(section), dict):
            data[section] = {}
        for key, default in keys.items():
            data[section].setdefault(key, list(default) if isinstance(default, list) else default)
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
        # Replace only the AI command/timeout/skip; preserve any other ai keys
        # the user set (e.g. a custom generate_prompt).
        ai = data["ai"] if isinstance(data.get("ai"), dict) else {}
        ai.update({"command": command, "timeout": timeout, "skip": skip})
        data["ai"] = ai
        _write_config_data(path, data)
        return path

    overrides = {
        ("ai", "command"): command,
        ("ai", "timeout"): timeout,
        ("ai", "skip"): skip,
    }
    path.write_text(_render_config_yaml(overrides))
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
    old_embed = data.get("embed", {}) or {}
    data["embed"] = {
        "provider": provider,
        "command": command,
        "dim": dim,
        "timeout": timeout,
        "include_body": _bool_config(old_embed.get("include_body"), True),
        "body_char_limit": _int_config(old_embed.get("body_char_limit"), DEFAULT_EMBED_BODY_CHAR_LIMIT),
        "text_char_limit": _int_config(old_embed.get("text_char_limit"), DEFAULT_EMBED_TEXT_CHAR_LIMIT),
        "bodyless_tags": _name_list_config(old_embed.get("bodyless_tags")),
        "bodyless_extensions": _str_list_config(old_embed.get("bodyless_extensions")),
        "nonembeddable_tags": _name_list_config(old_embed.get("nonembeddable_tags")),
        "nonembeddable_extensions": _str_list_config(old_embed.get("nonembeddable_extensions")),
        "nonembeddable_dirs": _name_list_config(old_embed.get("nonembeddable_dirs")),
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


def _int_config(value, default: int) -> int:
    """Coerce a config value to a non-negative int, falling back to *default*
    when it is missing or unparseable (a bad value should never crash a load)."""
    if value is None:
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n >= 0 else default


def _str_list_config(value) -> list[str]:
    """Coerce a config value to a list of strings, stripping leading dots (for extensions)."""
    if not isinstance(value, list):
        return []
    return [str(v).lstrip(".").lower() for v in value if v]


def _name_list_config(value) -> list[str]:
    """Coerce a config value to a list of strings without stripping dots (for dir/tag names)."""
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v]


def _yaml_scalar(value: str) -> str:
    """Quote a command as a single-line YAML scalar. JSON strings are valid
    YAML flow scalars, so json.dumps handles the embedded quotes and braces."""
    import json

    return json.dumps(value)
