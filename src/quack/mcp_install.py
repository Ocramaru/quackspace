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


def launch_command(explicit_root: str | None = None) -> tuple[str, list[str]]:
    """The stdio launch command an MCP client runs to start the server.

    Two install shapes, two commands:
    - Vendored checkout (the toolkit's source lives in `<root>/.quack`):
      `uv run --project <toolkit> quack-mcp` — self-contained, uses that
      project's own venv, no PATH install needed.
    - Global install (`uv tool`/`pipx`): `<root>/.quack` is just a marker dir,
      so launch the `quack-mcp` command that's already on PATH.
    """
    toolkit = find_root(explicit_root) / ".quack"
    if (toolkit / "pyproject.toml").exists():
        return "uv", ["run", "--project", str(toolkit), "quack-mcp"]
    if shutil.which("quack-mcp"):
        return "quack-mcp", []
    # No source and nothing on PATH: emit the vendored form as a best guess.
    return "uv", ["run", "--project", str(toolkit), "quack-mcp"]


def server_entry(explicit_root: str | None = None) -> dict:
    cmd, args = launch_command(explicit_root)
    return {"command": cmd, "args": args}


def mcp_json_snippet(explicit_root: str | None = None) -> str:
    return json.dumps(
        {"mcpServers": {SERVER_NAME: server_entry(explicit_root)}}, indent=2
    )


def write_project_config(explicit_root: str | None = None) -> Path:
    """Write/merge the project-root .mcp.json (the auto-discover convention)."""
    root = find_root(explicit_root)
    path = root / ".mcp.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text()) or {}
        except json.JSONDecodeError:
            data = {}
    data.setdefault("mcpServers", {})[SERVER_NAME] = server_entry(explicit_root)
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


def register_command(client: str, explicit_root: str | None = None) -> list[str]:
    """The exact CLI command that registers quack with a client (for display
    and execution). Both kiro-cli and claude take `... mcp add ... -- <cmd>`."""
    cmd, args = launch_command(explicit_root)
    if client == "kiro":
        return [
            "kiro-cli", "mcp", "add", "--name", SERVER_NAME,
            "--command", cmd, "--args", json.dumps(args),
        ]
    if client == "claude":
        return ["claude", "mcp", "add", SERVER_NAME, "--", cmd, *args]
    raise ValueError(client)


def run_register(client: str, explicit_root: str | None = None) -> tuple[bool, str]:
    """Execute a client's register command. Returns (ok, output)."""
    argv = register_command(client, explicit_root)
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
