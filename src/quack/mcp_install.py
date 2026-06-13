"""Make connecting an LLM to quack trivial.

Every MCP client uses the same JSON shape under an `mcpServers` key, only the
file location differs. So we generate one canonical stdio entry and (a) write
the project-root `.mcp.json` that clients auto-discover, and (b) register with
installed client CLIs (kiro-cli, claude). Nothing is tracked in a side folder:
the clients' own config files are the source of truth, and `quack mcp status`
reads them back live.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .core import find_root

SERVER_NAME = "quack"


def server_limit_args(
    search_limit: int | None = None,
    file_char_limit: int | None = None,
    sql_row_limit: int | None = None,
    central_limit: int | None = None,
) -> list[str]:
    args: list[str] = []
    for flag, value in (
        ("--search-limit", search_limit),
        ("--file-char-limit", file_char_limit),
        ("--sql-row-limit", sql_row_limit),
        ("--central-limit", central_limit),
    ):
        if value is not None:
            args.extend([flag, str(value)])
    return args


def launch_command(
    explicit_root: str | None = None,
    search_limit: int | None = None,
    file_char_limit: int | None = None,
    sql_row_limit: int | None = None,
    central_limit: int | None = None,
) -> tuple[str, list[str]]:
    """The stdio launch command an MCP client runs to start the server.

    Normal installs expose `quack-mcp` on PATH. During development from a source
    checkout, fall back to `uv run --project <checkout> quack-mcp`. The user's
    `.quack/` directory is workspace state, not a vendored copy of the tool.
    """
    root = find_root(explicit_root)
    root_args = ["--root", str(root)]
    limit_args = server_limit_args(
        search_limit=search_limit,
        file_char_limit=file_char_limit,
        sql_row_limit=sql_row_limit,
        central_limit=central_limit,
    )
    if shutil.which("quack-mcp"):
        return "quack-mcp", [*root_args, *limit_args]
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "pyproject.toml").exists():
        return "uv", ["run", "--project", str(package_root), "quack-mcp", *root_args, *limit_args]
    raise RuntimeError(
        "Could not find quack-mcp on PATH or a source checkout to run from. "
        "Install quack first with `uv tool install quackspace`."
    )


def server_entry(explicit_root: str | None = None, **limit_kwargs) -> dict:
    cmd, args = launch_command(explicit_root, **limit_kwargs)
    return {"command": cmd, "args": args}


def mcp_json_snippet(explicit_root: str | None = None, **limit_kwargs) -> str:
    return json.dumps(
        {"mcpServers": {SERVER_NAME: server_entry(explicit_root, **limit_kwargs)}}, indent=2
    )


def write_project_config(explicit_root: str | None = None, **limit_kwargs) -> Path:
    """Write/merge the project-root .mcp.json (the auto-discover convention)."""
    root = find_root(explicit_root)
    path = root / ".mcp.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text()) or {}
        except json.JSONDecodeError:
            data = {}
    data.setdefault("mcpServers", {})[SERVER_NAME] = server_entry(explicit_root, **limit_kwargs)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


@dataclass
class ClientTarget:
    key: str
    label: str
    binary: str

    @property
    def installed(self) -> bool:
        return shutil.which(self.binary) is not None


CLIENTS = [
    ClientTarget("kiro", "Kiro (kiro-cli)", "kiro-cli"),
    ClientTarget("claude", "Claude Code (claude)", "claude"),
]


def register_command(client: str, explicit_root: str | None = None, **limit_kwargs) -> list[str]:
    """The exact CLI command that registers quack with a client (for display
    and execution). Both kiro-cli and claude take `... mcp add ... -- <cmd>`."""
    cmd, args = launch_command(explicit_root, **limit_kwargs)
    if client == "kiro":
        return [
            "kiro-cli", "mcp", "add", "--name", SERVER_NAME,
            "--command", cmd, "--args", json.dumps(args),
        ]
    if client == "claude":
        return ["claude", "mcp", "add", SERVER_NAME, "--", cmd, *args]
    raise ValueError(client)


def run_register(client: str, explicit_root: str | None = None, **limit_kwargs) -> tuple[bool, str]:
    """Execute a client's register command. Returns (ok, output)."""
    argv = register_command(client, explicit_root, **limit_kwargs)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return False, f"{argv[0]} not found"
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out


def status(explicit_root: str | None = None) -> dict:
    """Report where quack is currently registered, read live from each source."""
    root = find_root(explicit_root)
    proj = root / ".mcp.json"
    proj_has = False
    if proj.exists():
        try:
            proj_has = SERVER_NAME in (json.loads(proj.read_text()).get("mcpServers") or {})
        except json.JSONDecodeError:
            proj_has = False

    clients = {}
    for c in CLIENTS:
        if not c.installed:
            clients[c.key] = "not installed"
            continue
        clients[c.key] = "registered" if _client_has_space(c) else "missing"
    return {"project_mcp_json": proj_has, "clients": clients}


def _client_has_space(client: ClientTarget) -> bool:
    try:
        proc = subprocess.run(
            [client.binary, "mcp", "list"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        return False
    out = proc.stdout + proc.stderr
    # Match the server name as a whole token, not a substring: "workspace"
    # contains "space" but is not our server. Clients format the list entry
    # differently (kiro: "• space   uv", claude: "space: ..."), so match the
    # token with a word boundary on both sides and accept any non-word follower.
    return re.search(rf"(?<![\w-]){re.escape(SERVER_NAME)}(?![\w-])", out) is not None
