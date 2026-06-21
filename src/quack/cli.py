"""`quack` CLI, the single entrypoint for the quack knowledge layer.

quack sits on top of a directory of your work — any files — and builds a meta
layer (editable .index.yaml + a DuckDB catalog) so LLMs can navigate all of it
quickly. Plays well with Obsidian but does not require it.

    quack init [dir]         create & scaffold a new space
    quack reindex            regenerate .index.yaml / map.yaml / catalog / diagrams
    quack describe PATH -d "…" [-t tag,tag]   record a file's description + tags
    quack generate [--stale] AI: fill in (or refresh) descriptions + tags
    quack doctor             report broken links + missing/stale descriptions
    quack new "Title" [-f folder] [-d description] [-t tag,tag]
    quack where              show workspace, state, package, and command paths
    quack agent kiro ...      agent integrations (Kiro, later others)

All commands accept --root PATH (defaults to walking up for `.quack/`, then
$QUACK_ROOT / $OBSIDIAN_VAULT).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .catalog import DB_NAME
from .config import Config
from .embed import EMBED_PROVIDER_CHOICES
from .prompts import yes_no

try:
    from rich_argparse import RichHelpFormatter

    _FORMATTER: type[argparse.HelpFormatter] = RichHelpFormatter
except ImportError:  # color is a nicety, never a hard requirement
    _FORMATTER = argparse.HelpFormatter


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that defaults to colored help on every Python version.

    Subparsers are created with this class too (argparse uses ``type(self)``),
    so the whole command tree gets the same formatter without per-parser wiring.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", _FORMATTER)
        super().__init__(*args, **kwargs)

from .diagram import diagram
from .doctor import diagnose, format_report
from .generate import AINotConfigured, fill_descriptions
from .indexer import reindex
from .scaffold import new_note
from .search import format_hits, search
from .setup import run_setup
from .core import find_root
from . import kiro as kiro_mod


def _add_root_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=None, help="quack root (default: walk up for .quack/)")


def _add_mcp_limit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--search-limit", type=int, default=None, help="default MCP search result limit")
    p.add_argument("--sql-row-limit", type=int, default=None, help="default MCP SQL row limit")
    p.add_argument("--central-limit", type=int, default=None, help="default MCP centrality result limit")


def _mcp_limit_kwargs(args) -> dict:
    return {
        "search_limit": getattr(args, "search_limit", None),
        "sql_row_limit": getattr(args, "sql_row_limit", None),
        "central_limit": getattr(args, "central_limit", None),
    }


def _catalog_lock_paths(root: str | Path) -> list[Path]:
    state = Path(root).resolve() / ".quack"
    if not state.exists():
        return []
    paths = [state / DB_NAME]
    paths.extend(sorted(state.glob(f"{DB_NAME}.build-*")))
    return [path for path in paths if path.exists()]


def _locker_command(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def _lock_holder_details(root: str | Path) -> list[tuple[int, str]]:
    lock_paths = _catalog_lock_paths(root)
    if not lock_paths:
        return []
    try:
        result = subprocess.run(
            ["lsof", "-t", "--", *map(str, lock_paths)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    lockers: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pid = int(line)
            command = _locker_command(pid)
            if pid != os.getpid() and "quack" in command.lower():
                lockers.append((pid, command))
    lockers.sort(key=lambda item: item[0])
    return lockers


def _maybe_release_catalog_locks(root: str | Path) -> bool:
    """Ask before terminating quack processes that are holding the catalog locked."""

    lockers = _lock_holder_details(root)
    if not lockers:
        return True

    pid, command = lockers[0]
    summary = command if len(command) <= 120 else f"{command[:117]}..."
    print(
        "Quack discovered an existing .quack but "
        f"process {summary} (pid {pid}) is locking up the .duckdb metastore so the init will fail."
    )
    if len(lockers) > 1:
        print(f"  {len(lockers) - 1} other process(es) are also holding the catalog lock.")
    if not yes_no("Would you like to autokill this process?", default=False):
        print("  init cancelled")
        return False

    for pid, _command in lockers:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _lock_holder_details(root):
            return True
        time.sleep(0.1)

    for pid, _command in _lock_holder_details(root):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="quack",
        description="quack builds a navigable meta layer over your local work "
        "(notes, docs, projects) so LLMs can find anything fast. Document-agnostic.",
        epilog="Run `quack <command> --help` for command-specific options.",
    )
    parser.add_argument("--version", action="version", version=f"quack {__version__}")
    parser.add_argument("--duck", action="store_true", help=argparse.SUPPRESS)
    # Not required: bare `quack` falls through to printing help (see main()).
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_reindex = sub.add_parser("reindex", help="regenerate the AI navigation layer")
    _add_root_arg(p_reindex)
    p_reindex.add_argument(
        "--no-diagrams", action="store_true", help="skip Mermaid diagram generation"
    )

    p_diagram = sub.add_parser("diagram", help="regenerate Mermaid link diagrams only")
    _add_root_arg(p_diagram)

    p_map = sub.add_parser("map", help="print the folder tree (folders + files) for a directory")
    _add_root_arg(p_map)
    p_map.add_argument(
        "parent", nargs="?", default=None,
        help="folder (relative to the root) to list (default: the folder you're in)",
    )
    p_map.add_argument(
        "--at", default=None, metavar="PATH",
        help="map at a filesystem path to a dir or file (a file maps its folder) — "
             "handy for a path from `quack search`",
    )
    p_map.add_argument(
        "-d", "--depth", type=int, default=None,
        help="folder levels to expand (default: auto-fit to ~50 entries)",
    )
    p_map.add_argument(
        "--folders-only", action="store_true", help="list folders only, no files"
    )
    p_map.add_argument(
        "--ext", default="", metavar="EXTS",
        help="only list files with these extensions (comma-separated, e.g. py,md)",
    )
    p_map.add_argument(
        "--min-size", type=int, default=None, metavar="BYTES",
        help="only list files at least this many bytes",
    )
    p_map.add_argument(
        "--max-size", type=int, default=None, metavar="BYTES",
        help="only list files at most this many bytes",
    )
    p_map.add_argument(
        "-n", "--limit", type=int, default=None, help="max entries shown per level"
    )

    p_clean = sub.add_parser(
        "clean", help="remove generated artifacts (catalog/map/diagrams)"
    )
    _add_root_arg(p_clean)
    p_clean.add_argument(
        "--all",
        dest="purge",
        action="store_true",
        help="fully uninstall quack: also remove .index.yaml, QUACK.md, .quack/",
    )
    p_clean.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt for --all"
    )
    p_clean.add_argument(
        "--dry-run", action="store_true", help="show what would be removed without deleting anything"
    )
    p_clean.add_argument(
        "--catalog", action="store_true", help="remove only the DuckDB catalog"
    )
    p_clean.add_argument(
        "--map", action="store_true", help="remove only .quack/map.yaml"
    )
    p_clean.add_argument(
        "--diagrams", action="store_true", help="remove only generated Mermaid diagrams"
    )

    p_doctor = sub.add_parser("doctor", help="health-check the root (files + MCP)")
    _add_root_arg(p_doctor)
    p_doctor.add_argument(
        "--strict", action="store_true", help="exit non-zero if any issue found"
    )
    p_doctor.add_argument(
        "--files", action="store_true", help="only check notes (skip MCP)"
    )
    p_doctor.add_argument(
        "--mcp", action="store_true", help="only check MCP registration (skip notes)"
    )

    p_new = sub.add_parser("new", help="scaffold a markdown note with frontmatter")
    _add_root_arg(p_new)
    p_new.add_argument("title")
    p_new.add_argument("-f", "--folder", default="projects")
    p_new.add_argument("-d", "--description", default="")
    p_new.add_argument("-t", "--tags", default="", help="comma-separated")

    p_describe = sub.add_parser(
        "describe", help="record a description + tags for any file"
    )
    _add_root_arg(p_describe)
    p_describe.add_argument("path", help="root-relative path or bare file name")
    p_describe.add_argument("-d", "--description", required=True)
    p_describe.add_argument("-t", "--tags", default="", help="comma-separated")
    p_describe.add_argument(
        "--no-reindex", action="store_true", help="skip the reindex afterwards"
    )

    p_where = sub.add_parser("where", help="show workspace, state, package, and command paths")
    _add_root_arg(p_where)

    p_setup = sub.add_parser("setup", help="choose the AI assistant interactively")
    _add_root_arg(p_setup)

    p_init = sub.add_parser(
        "init", help="create & scaffold a new space, then choose an assistant"
    )
    _add_root_arg(p_init)
    p_init.add_argument(
        "dir", nargs="?", default=None, help="target directory (created if missing; default: current)"
    )
    p_init.add_argument(
        "--no-reindex",
        action="store_true",
        help="skip the first reindex so you can tune .quackignore before indexing",
    )
    p_init.add_argument(
        "--no-gitignore",
        action="store_true",
        help="do not write or update quack-managed .gitignore files",
    )
    p_init.add_argument(
        "--no-diagrams",
        action="store_true",
        help="turn off diagram generation in this workspace config",
    )
    p_init.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be scaffolded without writing anything",
    )

    p_gen = sub.add_parser(
        "generate", help="use the configured AI to fill in missing descriptions"
    )
    _add_root_arg(p_gen)
    p_gen.add_argument("--only", default=None, help="restrict to one file path")
    p_gen.add_argument(
        "--dry-run", action="store_true", help="print descriptions without writing"
    )
    p_gen.add_argument(
        "--stale",
        action="store_true",
        help="also refresh descriptions whose file changed since they were written",
    )

    p_search = sub.add_parser(
        "search", help="auto-hybrid search (structural + FTS + semantic + graph)"
    )
    _add_root_arg(p_search)
    p_search.add_argument("query", nargs="+", help="search terms")
    p_search.add_argument(
        "-n", "--limit", type=int, default=5,
        help="max results (default: 5; the output tells you how to ask for more)",
    )
    p_search.add_argument(
        "--no-expand", action="store_true", help="skip graph-neighbour expansion"
    )
    p_search.add_argument(
        "--fts", action="store_true", help="force only DuckDB BM25 full-text ranking"
    )
    p_search.add_argument(
        "--semantic", action="store_true", help="force only vss semantic ranking"
    )
    p_search.add_argument(
        "--folders", action="store_true", help="force only folder-level search"
    )
    p_search.add_argument(
        "--with-folders", action="store_true",
        help="also show matching folders alongside file hits (hidden by default)",
    )
    p_search.add_argument(
        "--no-local", action="store_true",
        help="disable current-directory locality boost (search the full workspace equally)",
    )

    p_graph = sub.add_parser("graph", help="graph queries over the link structure")
    graph_sub = p_graph.add_subparsers(dest="graph_command", metavar="<subcommand>")
    g_path = graph_sub.add_parser("path", help="shortest path between two notes")
    _add_root_arg(g_path)
    g_path.add_argument("src")
    g_path.add_argument("dst")
    g_cent = graph_sub.add_parser("central", help="most-connected notes (hubs)")
    _add_root_arg(g_cent)
    g_cent.add_argument("-n", "--limit", type=int, default=10)
    g_comp = graph_sub.add_parser("clusters", help="connected components")
    _add_root_arg(g_comp)

    p_embed = sub.add_parser("embed", help="build or configure semantic embeddings (DuckDB vss)")
    _add_root_arg(p_embed)
    p_embed.add_argument(
        "embed_command",
        nargs="?",
        choices=("init", "setup", "text"),
        help="use `init`/`setup`; provider command: `text`",
    )
    p_embed.add_argument(
        "embed_args",
        nargs="*",
        help=argparse.SUPPRESS,
    )
    p_embed.add_argument(
        "--command",
        dest="embed_setup_command",
        default=None,
        help="embedding command for `quack embed init`; use {text} or stdin",
    )
    p_embed.add_argument(
        "--provider",
        choices=EMBED_PROVIDER_CHOICES,
        default=None,
        help="provider preset for `quack embed init`; also used by `quack embed text`",
    )
    p_embed.add_argument(
        "--model",
        default=None,
        help="embedding model for `quack embed text --provider ollama`",
    )
    p_embed.add_argument(
        "--pull",
        action="store_true",
        help="pull the Ollama embedding model during `quack embed init --provider ollama`",
    )
    p_embed.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="embedding command timeout in seconds",
    )
    p_embed.add_argument(
        "--rebuild",
        action="store_true",
        help="drop existing vectors and rebuild the embedding cache",
    )
    p_embed.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel embedding workers (default: auto for Ollama, otherwise up to 8); use 1 to disable",
    )
    p_embed.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="show full subprocess output on embedding errors",
    )

    p_mcp = sub.add_parser("mcp", help="MCP server for LLM tool access")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command", metavar="<subcommand>")
    p_mcp_serve = mcp_sub.add_parser("serve", help="run the server on stdio (clients launch this)")
    _add_root_arg(p_mcp_serve)
    _add_mcp_limit_args(p_mcp_serve)
    p_mcp_print = mcp_sub.add_parser("print", help="print the mcpServers JSON snippet")
    _add_root_arg(p_mcp_print)
    _add_mcp_limit_args(p_mcp_print)
    p_mcp_install = mcp_sub.add_parser(
        "install", help="register the server with .mcp.json + installed clients"
    )
    _add_root_arg(p_mcp_install)
    _add_mcp_limit_args(p_mcp_install)
    p_mcp_install.add_argument(
        "--yes", action="store_true", help="register global clients without prompting"
    )
    p_mcp_install.add_argument(
        "--project-only", action="store_true", help="only write .mcp.json"
    )
    p_mcp_status = mcp_sub.add_parser("status", help="show where quack is registered")
    _add_root_arg(p_mcp_status)

    p_sql = sub.add_parser("sql", help="run SQL against the DuckDB metadata catalog")
    _add_root_arg(p_sql)
    p_sql.add_argument("query", help="SQL statement (tables: files, tags, links)")
    p_sql.add_argument(
        "--csv", action="store_true", help="output CSV instead of aligned columns"
    )

    p_agent = sub.add_parser("agent", help="agent integrations")
    agent_sub = p_agent.add_subparsers(dest="agent_provider", metavar="<provider>")
    p_agent_kiro = agent_sub.add_parser("kiro", help="Kiro agent integration")
    kiro_agent_sub = p_agent_kiro.add_subparsers(dest="agent_command", metavar="<subcommand>")
    p_agent_kiro_install = kiro_agent_sub.add_parser(
        "install", help="write .kiro/hooks/*.kiro.hook files"
    )
    _add_root_arg(p_agent_kiro_install)
    p_agent_kiro_send = kiro_agent_sub.add_parser("send", help="send a prompt to kiro-cli")
    p_agent_kiro_send.add_argument("prompt")

    return parser


def _format_table(cols: list[str], rows: list[tuple], csv: bool = False) -> str:
    """Render query results as CSV or aligned columns."""
    if not rows:
        return "(no rows)"
    if csv:
        out = [",".join(cols)] if cols else []
        out += [",".join("" if v is None else str(v) for v in r) for r in rows]
        return "\n".join(out)
    str_rows = [["" if v is None else str(v) for v in r] for r in rows]
    widths = [
        max(len(cols[i]) if cols else 0, *(len(r[i]) for r in str_rows))
        for i in range(len(str_rows[0]))
    ]
    lines = []
    if cols:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
        lines.append("  ".join("-" * widths[i] for i in range(len(cols))))
    lines += ["  ".join(v.ljust(widths[i]) for i, v in enumerate(r)) for r in str_rows]
    return "\n".join(lines)


def _run_mcp(args) -> int:
    from . import mcp_install as mi

    cmd = args.mcp_command

    if cmd == "serve":
        from .mcp_server import main as mcp_main

        mcp_main([*(["--root", str(find_root(args.root))] if args.root else []), *mi.server_limit_args(**_mcp_limit_kwargs(args))])
        return 0

    if cmd == "print":
        print(mi.mcp_json_snippet(args.root, **_mcp_limit_kwargs(args)))
        return 0

    if cmd == "status":
        st = mi.status(args.root)
        flag = "yes" if st["project_mcp_json"] else "no"
        print(f"project .mcp.json: {flag}")
        for key, state in st["clients"].items():
            print(f"  {key}: {state}")
        return 0

    if cmd == "install":
        limit_kwargs = _mcp_limit_kwargs(args)
        path = mi.write_project_config(args.root, **limit_kwargs)
        print(f"✓ wrote {path}")
        if args.project_only:
            return 0
        for client in mi.CLIENTS:
            if not client.installed:
                print(f"  {client.label}: not installed, skipping")
                continue
            register = " ".join(mi.register_command(client.key, args.root, **limit_kwargs))
            if not args.yes:
                print(f"\n{client.label} is a global config change. Command:")
                print(f"  {register}")
                if not yes_no(f"Register quack with {client.label}?", default=False):
                    print("  skipped")
                    continue
            ok, out = mi.run_register(client.key, args.root, **limit_kwargs)
            print(f"  {'✓' if ok else '✗'} {client.label}: {out or ('registered' if ok else 'failed')}")
        return 0

    print("usage: quack mcp {serve|install|print|status}")
    return 1


def _run_graph(args) -> int:
    from . import graph as graph_mod

    cmd = getattr(args, "graph_command", None)
    if cmd == "path":
        path = graph_mod.shortest_path(args.src, args.dst, explicit_root=args.root)
        if path is None:
            print(f"No path between {args.src} and {args.dst}.")
            return 1
        print(" → ".join(path) + f"  ({len(path) - 1} hops)")
        return 0
    if cmd == "central":
        rows = graph_mod.centrality(explicit_root=args.root, limit=args.limit)
        for name, rel, degree in rows:
            print(f"{degree:>3}  {rel}")
        return 0
    if cmd == "clusters":
        comps = graph_mod.components(explicit_root=args.root)
        for i, names in enumerate(comps, 1):
            kind = "orphan" if len(names) == 1 else f"{len(names)} notes"
            print(f"cluster {i} ({kind}): {', '.join(names)}")
        return 0
    print("usage: quack graph {path|central|clusters}")
    return 1


def _run_kiro(command: str | None, args) -> int:
    if command == "install":
        written = kiro_mod.install_hooks(args.root)
        print(f"✓ installed {len(written)} Kiro hook(s):")
        for p in written:
            print(f"    {p}")
        return 0
    if command == "send":
        print(kiro_mod.send(args.prompt))
        return 0
    print("usage: quack agent kiro {install|send}")
    return 1


def _run_agent(args) -> int:
    provider = getattr(args, "agent_provider", None)
    if provider == "kiro":
        return _run_kiro(getattr(args, "agent_command", None), args)
    print("usage: quack agent {kiro}")
    return 1


def _run_generate(args) -> bool:
    """Run description generation and print results. Returns True on success."""
    from ._duck import swimming

    with swimming("Generating descriptions") as progress:
        result = fill_descriptions(
            args.root,
            only=args.only,
            dry_run=args.dry_run,
            include_stale=args.stale,
            progress=progress.update,
        )
    if args.dry_run:
        for line in result.updated:
            print(line)
    else:
        print(f"✓ wrote description + tags for {len(result.updated)} file(s)")
        for rel in result.updated:
            print(f"    {rel}")
    if result.skipped:
        print(f"  skipped {len(result.skipped)} file(s) (no usable output)")
    if not args.dry_run and result.updated:
        with swimming("Reindexing") as progress:
            reindex(args.root, progress=progress.update)
        print("  reindexed")
    return True


def _run_embed_provider_command(args) -> int:
    text = " ".join(args.embed_args) if args.embed_args else sys.stdin.read()
    if args.embed_command == "text":
        provider = args.provider or "builtin"
        if provider == "builtin":
            from .embed_provider import embed

            print(json.dumps(embed(text)))
            return 0
        if provider == "ollama":
            from .embed import DEFAULT_AI_TIMEOUT, _ensure_ollama_server
            from .embed_ollama import DEFAULT_MODEL, embed

            try:
                _ensure_ollama_server(DEFAULT_AI_TIMEOUT, out=sys.stderr)
                vec = embed(text, model=args.model or DEFAULT_MODEL)
            except RuntimeError as e:
                print(f"✗ {e}", file=sys.stderr)
                return 1
            print(json.dumps(vec))
            return 0
        print(
            "`quack embed text` supports --provider builtin or --provider ollama.",
            file=sys.stderr,
        )
        return 1
    return 1


def _offer_setup(space_arg) -> bool:
    """No AI configured: offer to run setup. Returns True if now configured."""
    config = Config.load(space_arg)
    if config.ai.skip:
        print("AI is turned off (ai.skip: true in config.yaml).")
        print("Run `quack setup` to enable an assistant.")
        return False
    print("No AI assistant is set up yet.")
    print("The assistant writes short descriptions of your files and folders.")
    if not yes_no("Would you like to set one up now?", default=False):
        print("Skipped. Run `quack setup` later, or write descriptions yourself.")
        return False
    result = run_setup(space_arg)
    return result.configured


def _maybe_setup_embeddings(root: str, *, can_build: bool) -> None:
    if not sys.stdin.isatty():
        return
    config = Config.load(root)
    if config.embed.skip:
        return
    if config.embed.configured and config.embed.provider != "builtin":
        return
    if not yes_no("Choose semantic search embedding provider?", default=True):
        return
    from .embed import run_embed_setup

    try:
        result = run_embed_setup(root)
    except RuntimeError as e:
        print(f"  embeddings: skipped ({e})")
        return
    if not result.configured or not can_build:
        return
    if not yes_no("Build embeddings now?", default=False):
        return
    from ._duck import swimming
    from .embed import build_embeddings

    try:
        with swimming("Embedding") as progress:
            summary = build_embeddings(root, progress=progress.update)
    except RuntimeError as e:
        print(f"  embeddings: skipped ({e})")
        return
    print(
        f"✓ embedded {summary['embedded']:,} file(s) + "
        f"{summary['folders']:,} folder(s) (dim {summary['dim']})"
    )


def _clean_targets_from_args(args) -> set[str] | None:
    targets = set()
    if getattr(args, "catalog", False):
        targets.add("catalog")
    if getattr(args, "map", False):
        targets.add("map")
    if getattr(args, "diagrams", False):
        targets.add("diagrams")
    return targets or None


def _choose_clean_mode() -> tuple[set[str] | None, bool] | None:
    from .prompts import Choice, choice as pick

    selected = pick(
        "What would you like to clean?",
        [
            Choice("derived", "Derived artifacts (catalog, map, diagrams)"),
            Choice("diagrams", "Diagrams only"),
            Choice("catalog", "Catalog and map only"),
            Choice("purge", "Full uninstall (removes authored metadata too)"),
            Choice("cancel", "Cancel"),
        ],
        default="derived",
    )
    if selected == "cancel":
        return None
    if selected == "diagrams":
        return {"diagrams"}, False
    if selected == "catalog":
        return {"catalog", "map"}, False
    if selected == "purge":
        ok = yes_no(
            "Full uninstall deletes authored .index.yaml metadata. Continue?",
            default=False,
        )
        return (None, True) if ok else None
    return None, False  # derived (default)


def _print_clean_report(removed: dict, *, purge: bool, dry_run: bool) -> None:
    title = "quack clean preview" if dry_run else "✓ cleaned quack artifacts"
    print(title)
    mode = "full uninstall" if purge else ", ".join(removed.get("targets", []))
    print(f"  mode: {mode or 'derived artifacts'}")
    print(f"  catalog:  {removed['catalog']:,}")
    print(f"  map:      {removed['map']:,}")
    print(f"  diagrams: {removed['diagrams']:,}")
    if purge or removed.get("indexes"):
        print(f"  indexes:  {removed['indexes']:,}")
    if purge or removed.get("other"):
        print(f"  other:    {removed['other']:,}")
    if removed.get("extras"):
        print(f"  extras:   {removed['extras']:,} stray artifact(s) found by scan")
    if not purge:
        print("  rebuild:  quack reindex")


def main(argv: list[str] | None = None) -> int:
    """Entry point: dispatch, turning expected failures into clean messages
    instead of tracebacks."""
    try:
        return _dispatch(argv)
    except RuntimeError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


def _incomplete_command(parser: argparse.ArgumentParser, args) -> argparse.ArgumentParser | None:
    """Walk the chosen subcommand path. If a parser that expects a subcommand
    was invoked without one (bare `quack`, `quack mcp`, `quack agent kiro`, …),
    return it so the caller can show that parser's own help."""
    current = parser
    while True:
        subparsers = next(
            (a for a in current._actions if isinstance(a, argparse._SubParsersAction)),
            None,
        )
        if subparsers is None:  # reached a real leaf command
            return None
        chosen = getattr(args, subparsers.dest, None)
        if chosen is None:  # group invoked without picking a subcommand
            return current
        current = subparsers.choices[chosen]


def _map_parent_from_args(args, root) -> str:
    """Resolve which folder `quack map` should list, as a rel path under *root*.

    Precedence: --at <fs path> (a file maps its folder) > positional parent
    (a root-relative folder) > the folder you're standing in (cwd)."""
    if getattr(args, "at", None) is not None:
        p = Path(args.at).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        p = p.resolve()
        if p.is_file():
            p = p.parent
        try:
            rel = p.relative_to(root.resolve()).as_posix()
        except ValueError:
            raise RuntimeError(
                f"--at path {args.at!r} is not inside the quack root {root}"
            )
        return "" if rel == "." else rel
    if args.parent is not None:
        return args.parent.strip("/")
    try:  # default: the folder you're standing in
        rel = Path.cwd().resolve().relative_to(root.resolve()).as_posix()
        return "" if rel == "." else rel
    except ValueError:
        return ""  # cwd is outside the root — fall back to top level


def _add_listing_nodes(node, listing: dict, include_files: bool) -> None:
    """Render one `catalog.folder_listing` level onto a rich tree node, recursing
    into nested folders. Subfolders (with `/` + count) first, then loose files."""
    from rich.text import Text

    for f in listing.get("folders", []):
        leaf = f["folder"].rsplit("/", 1)[-1] or f["folder"]
        label = Text(f"{leaf}/ ")
        kind = f.get("kind")
        if kind == "opaque":
            # Recorded but not descended into — its files are not indexed, so a
            # "(0)" count would read as "empty". Say so explicitly instead.
            label.append("ignored", style="yellow")
        elif kind == "dataset":
            label.append("dataset", style="yellow")
        else:
            label.append(f"({f['n_files']})", style="cyan")
        if f.get("description"):
            label.append(f"  {f['description']}", style="dim")
        _add_listing_nodes(node.add(label), f, include_files)
    if listing.get("truncated"):
        node.add(Text("… more folders (raise --limit)", style="dim"))
    if include_files:
        for fl in listing.get("files", []):
            label = Text(fl["rel"].rsplit("/", 1)[-1] or fl["rel"])
            if fl.get("description"):
                label.append(f"  {fl['description']}", style="dim")
            node.add(label)
        if listing.get("files_truncated"):
            node.add(Text("… more files (raise --limit)", style="dim"))


def _run_map(args) -> int:
    """Print a directory listing (folders + files) as a rich tree.

    Human-facing companion to the MCP `map` tool: same catalog data
    (`catalog.folder_listing`), rendered with tree connectors. Supports --depth,
    --folders-only, --ext / --min-size / --max-size filters, and --at <path>.
    """
    from rich.console import Console
    from rich.text import Text
    from rich.tree import Tree

    from . import catalog

    root = find_root(args.root)
    parent = _map_parent_from_args(args, root)
    exts = {
        e.strip().lstrip(".").lower()
        for e in (args.ext or "").split(",")
        if e.strip()
    } or None
    include_files = not args.folders_only

    listing = catalog.folder_listing(
        args.root,
        parent,
        depth=(args.depth if args.depth and args.depth > 0 else None),
        include_files=include_files,
        exts=exts,
        min_size=args.min_size,
        max_size=args.max_size,
        limit=int(args.limit) if args.limit else 1000,
        auto_items=Config.load(args.root).defaults.map_auto_items,
    )

    header = Text(parent or root.name or str(root))
    note = f"  ({listing['files_here']} files here"
    if args.depth is None and listing["depth"] > 1:
        note += f", auto-expanded to depth {listing['depth']}"
    header.append(note + ")", style="dim")
    tree = Tree(header)
    _add_listing_nodes(tree, listing, include_files)
    if not listing["folders"] and not (include_files and listing["files"]):
        tree.add(Text("(empty)", style="dim"))
    Console().print(tree)
    return 0


def _dispatch(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.duck:
        from ._duck import play

        play()
        return 0

    incomplete = _incomplete_command(parser, args)
    if incomplete is not None:  # show the relevant help instead of erroring
        incomplete.print_help()
        return 0

    if args.command == "reindex":
        from ._duck import swimming

        config = Config.load(args.root)
        generate_diagrams = config.index.diagrams and not args.no_diagrams
        with swimming("Reindexing") as progress:
            summary = reindex(args.root, progress=progress.update)
            if generate_diagrams and summary["folder_indexes"]:
                progress.update(message="Generating diagrams")
                d = diagram(args.root, progress=progress.update)
            else:
                d = None
        print(
            f"✓ reindexed {summary['files']:,} files across "
            f"{summary['folder_indexes']:,} folder(s)\n"
            f"  map:     {summary['map']}\n"
            f"  catalog: {summary['db']}"
        )
        datasets = summary.get("datasets") or {}
        if datasets:
            print(f"  datasets: {len(datasets):,} folder(s) recorded but not indexed")
            for rel, reason in sorted(datasets.items()):
                print(f"    - {rel} ({reason})")
        if d is not None:
            print(f"  diagrams: {d['folder_diagrams']} folder(s) + {d['global']}")
        elif generate_diagrams:
            print("  diagrams: skipped (no folder indexes changed)")
        elif args.no_diagrams:
            print("  diagrams: skipped (--no-diagrams)")
        else:
            print("  diagrams: skipped (index.diagrams: false)")
        return 0

    if args.command == "diagram":
        from ._duck import swimming

        with swimming("Generating diagrams") as progress:
            d = diagram(args.root, progress=progress.update)
        print(f"✓ wrote {d['folder_diagrams']} folder diagram(s)\n  global: {d['global']}")
        return 0

    if args.command == "clean":
        from ._duck import swimming
        from .clean import clean

        targets = _clean_targets_from_args(args)
        purge = args.purge
        purge_confirmed = args.yes
        if not purge and targets is None and not args.dry_run and sys.stdin.isatty():
            chosen = _choose_clean_mode()
            if chosen is None:
                print("clean cancelled")
                return 0
            targets, purge = chosen
            purge_confirmed = purge

        if purge and not purge_confirmed and not args.dry_run:
            print(
                "✗ `quack clean --all` fully uninstalls quack and DELETES authored\n"
                "  .index.yaml descriptions, QUACK.md, and .quack/. This cannot be\n"
                "  undone. Re-run with --yes to confirm.",
                file=sys.stderr,
            )
            return 1
        message = "Previewing clean" if args.dry_run else "Cleaning"
        with swimming(message, total=1) as progress:
            progress.update(0, 1, "Checking clean targets")
            removed = clean(args.root, purge=purge, dry_run=args.dry_run, targets=targets)
            progress.update(1, 1, "Clean preview ready" if args.dry_run else "Cleaned")
        _print_clean_report(removed, purge=purge, dry_run=args.dry_run)
        return 0

    if args.command == "doctor":
        # Default checks both; --files or --mcp narrows the scope.
        check_files = args.files or not args.mcp
        check_mcp = args.mcp or not args.files
        ok = True
        if check_files:
            report = diagnose(args.root)
            print(format_report(report))
            ok = ok and report.ok
        if check_mcp:
            from . import mcp_install as mi

            st = mi.status(args.root)
            print("MCP registration:")
            print(f"  project .mcp.json: {'yes' if st['project_mcp_json'] else 'no'}")
            for key, state in st["clients"].items():
                mark = "✓" if state == "registered" else ("-" if state == "not installed" else "✗")
                print(f"  {mark} {key}: {state}")
            # A missing registration on an installed client is the only MCP fault.
            if any(s == "missing" for s in st["clients"].values()):
                ok = False
                print("  (run `quack mcp install` to register)")
        return 1 if (args.strict and not ok) else 0

    if args.command == "new":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        path = new_note(
            title=args.title,
            folder=args.folder,
            description=args.description,
            tags=tags,
            explicit_root=args.root,
        )
        print(f"✓ created {path}")
        print("  next: fill in the description, then run `quack reindex`")
        return 0

    if args.command == "describe":
        from . import generate
        from ._duck import swimming

        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        rel = generate.record(args.root, args.path, args.description, tags)
        if rel is None:
            print(f"✗ no indexed file or folder matching {args.path!r}", file=sys.stderr)
            return 1
        print(f"✓ described {rel}")
        if not args.no_reindex:
            with swimming("Reindexing") as progress:
                reindex(args.root, progress=progress.update)
            print("  reindexed")
        return 0

    if args.command == "where":
        root = find_root(args.root)
        package = Path(__file__).resolve().parents[2]
        print(f"root:     {root}")
        print(f"state:    {root / '.quack'}")
        print(f"package:  {package}")
        print(f"command:  {Path(sys.argv[0]).resolve()}")
        print(f"guide:    {root / 'QUACK.md'}")
        return 0

    if args.command == "search":
        from ._duck import paddling

        terms = " ".join(args.query)
        root_path = find_root(args.root)
        root = str(root_path)

        # Resolve cwd relative to the workspace root for locality boosting.
        # None when --no-local is set, cwd is the root itself, or cwd is outside.
        cwd_rel: str | None = None
        if not args.no_local:
            try:
                rel = Path.cwd().relative_to(root_path).as_posix()
                cwd_rel = rel if rel != "." else None
            except ValueError:
                pass

        if args.fts:
            from . import catalog

            with paddling("Searching full text") as progress:
                progress.update(0, 1, "Searching full text")
                rows = catalog.fts_search(terms, explicit_root=args.root, limit=args.limit)
                progress.update(1, 1, "Search complete")
            if not rows:
                print("No matches.")
                return 1
            print(f"# root: {root}  (paths below are relative to it)")
            for rel, desc, score in rows:
                print(f"{rel}  [bm25 {score:.2f}]")
                if desc:
                    print(f"    {desc}")
            return 0
        if args.semantic:
            from .embed import EmbedNotConfigured, semantic_search

            try:
                with paddling("Searching embeddings") as progress:
                    progress.update(0, 1, "Searching embeddings")
                    rows = semantic_search(terms, explicit_root=args.root, limit=args.limit)
                    progress.update(1, 1, "Search complete")
            except EmbedNotConfigured:
                print("No embeddings. Configure `embed.command` and run `quack embed`.")
                return 1
            except Exception as e:
                print(f"No embeddings built yet ({e}). Run `quack embed`.")
                return 1
            print(f"# root: {root}  (paths below are relative to it)")
            for rel, name, dist in rows:
                print(f"{rel}  [cosine {dist:.3f}]")
            return 0
        if args.folders:
            from .search import format_folder_hits, search_folders

            with paddling("Searching folders") as progress:
                fhits = search_folders(
                    terms,
                    explicit_root=args.root,
                    limit=args.limit,
                    cwd_rel=cwd_rel,
                    progress=progress.update,
                )
            print(f"# root: {root}  (paths below are relative to it)")
            if cwd_rel:
                print(f"# cwd:  {cwd_rel}/")
            print(format_folder_hits(fhits))
            return 0 if fhits else 1

        from .search import format_folder_hits, route, search_folders

        with paddling("Searching") as progress:
            # Fetch one extra hit so we can tell the user whether more exist
            # without paging through them; only `limit` are ever shown.
            hits = search(
                terms,
                explicit_root=args.root,
                limit=args.limit + 1,
                expand=not args.no_expand,
                cwd_rel=cwd_rel,
                progress=progress.update,
            )
            # Folders are noise in a file search, so they stay hidden unless the
            # caller asks for them with --with-folders.
            fhits: list = []
            if args.with_folders and route(terms) in ("folders", "both"):
                fhits = search_folders(
                    terms,
                    explicit_root=args.root,
                    limit=args.limit,
                    cwd_rel=cwd_rel,
                    progress=progress.update,
                )
        has_more = len(hits) > args.limit
        hits = hits[: args.limit]
        if cwd_rel:
            print(f"# cwd: {cwd_rel}/  (local results boosted)")
        print(format_hits(hits, root=root))
        if has_more:
            more = max(args.limit * 2, 30)
            print(
                f"\n# showing the top {len(hits)} — for more results run: "
                f"quack search {terms} --limit {more}"
            )
        if not args.with_folders and route(terms) in ("folders", "both"):
            print("# folders are hidden — add --with-folders to include them")
        if fhits:
            print("\n# folders")
            print(format_folder_hits(fhits))
        return 0 if (hits or fhits) else 1

    if args.command == "sql":
        from . import catalog

        cols, rows = catalog.query(args.query, explicit_root=args.root)
        print(_format_table(cols, rows, csv=args.csv))
        return 0

    if args.command == "map":
        return _run_map(args)

    if args.command == "mcp":
        return _run_mcp(args)

    if args.command == "graph":
        return _run_graph(args)

    if args.command == "agent":
        return _run_agent(args)

    if args.command == "embed":
        from ._duck import swimming
        from .embed import EmbedNotConfigured, build_embeddings, run_embed_setup

        if args.embed_command == "text":
            return _run_embed_provider_command(args)

        if (
            args.embed_command in ("init", "setup")
            or args.embed_setup_command
            or args.provider
            or args.pull
        ):
            try:
                result = run_embed_setup(
                    args.root,
                    command=args.embed_setup_command,
                    provider=args.provider,
                    pull=args.pull,
                    timeout=args.timeout or Config.load(args.root).embed.timeout,
                )
            except RuntimeError as e:
                print(f"✗ {e}", file=sys.stderr)
                return 1
            return 0 if result.configured else 1

        try:
            with swimming("Embedding") as progress:
                result = build_embeddings(
                    args.root,
                    progress=progress.update,
                    rebuild=args.rebuild,
                    timeout=args.timeout,
                    workers=args.workers,
                )
        except EmbedNotConfigured:
            print("No embedding command. Run `quack embed init`,")
            print("or set `embed.command` in .quack/config.yaml.")
            print("The command must print one JSON array of floats.")
            return 1
        except RuntimeError as e:
            from .embed import EmbedSubprocessError
            print(f"✗ {e}", file=sys.stderr)
            if args.verbose and isinstance(e, EmbedSubprocessError):
                print(f"  command: {' '.join(e.argv)}", file=sys.stderr)
                if e.stdout.strip():
                    print(f"  stdout:\n{e.stdout.rstrip()}", file=sys.stderr)
                if e.stderr.strip():
                    print(f"  stderr:\n{e.stderr.rstrip()}", file=sys.stderr)
            elif not args.verbose:
                print("  Run `quack embed --verbose` for full subprocess output.", file=sys.stderr)
            return 1
        print(
            f"✓ embedded {result['embedded']:,} file(s) + "
            f"{result['folders']:,} folder(s) (dim {result['dim']})"
        )
        changed = (
            result["updated"] + result["deleted"]
            + result["folders_updated"] + result["folders_deleted"]
        )
        if changed:
            print(
                f"  refreshed: {result['updated']:,} file(s), "
                f"{result['folders_updated']:,} folder(s); "
                f"pruned: {result['deleted']:,} file(s), "
                f"{result['folders_deleted']:,} folder(s)"
            )
        else:
            print("  refreshed: already up to date")
        failed = result.get("failed", 0) + result.get("folders_failed", 0)
        if failed:
            print(
                f"  skipped: {result.get('failed', 0):,} file(s), "
                f"{result.get('folders_failed', 0):,} folder(s) failed to embed"
            )
            if args.verbose:
                for item in result.get("failed_items", [])[:20]:
                    print(f"    {item}")
                if len(result.get("failed_items", [])) > 20:
                    print(f"    ... {len(result['failed_items']) - 20:,} more")
        return 0

    if args.command == "init":
        from ._duck import swimming
        from .scaffold import preview_scaffold, scaffold_root, update_init_config

        gitignore_summaries = []
        target = args.dir or args.root
        target_root = Path(target).expanduser().resolve() if target else Path.cwd().resolve()
        config_path = target_root / ".quack" / "config.yaml"
        config_existed = config_path.exists()
        manage_gitignore = not args.no_gitignore
        diagrams_enabled = not args.no_diagrams
        if config_existed:
            config = Config.load(str(target_root))
            diagrams_enabled = config.index.diagrams and not args.no_diagrams
        elif not args.dry_run and sys.stdin.isatty():
            if not args.no_gitignore:
                manage_gitignore = yes_no(
                    "Add quack's generated files to .gitignore files?",
                    default=True,
                )
            if not args.no_diagrams:
                diagrams_enabled = yes_no(
                    "Generate Mermaid diagrams during reindex?",
                    default=True,
                )
        if not args.no_reindex and not _maybe_release_catalog_locks(target_root):
            return 1
        if args.dry_run:
            with swimming("Previewing init", total=1) as progress:
                progress.update(0, 1, "Checking init plan")
                paths = preview_scaffold(str(target_root), manage_gitignore=manage_gitignore)
                progress.update(1, 1, "Preview ready")
            print("quack init preview")
            print("paths:")
            for path in paths:
                print(f"  {path}")
            if config_existed:
                print(f"  config: exists, preserved by init: {config_path}")
            if args.no_gitignore:
                print("  gitignore: skipped (--no-gitignore)")
            elif manage_gitignore:
                print("  gitignore: repo .gitignore files checked during init")
            if args.no_reindex:
                print("  reindex: skipped (--no-reindex)")
            else:
                print("  reindex: runs during init")
            return 0
        with swimming("Scaffolding", total=5) as progress:
            root = scaffold_root(
                args.dir or args.root,
                progress=progress.update,
                gitignore_summary=gitignore_summaries,
                manage_gitignore=manage_gitignore,
            )
        if config_existed:
            print(f"  config: preserved existing {config_path}")
            if args.no_gitignore or args.no_diagrams:
                update_init_config(
                    root / ".quack" / "config.yaml",
                    gitignore=False if args.no_gitignore else None,
                    diagrams=False if args.no_diagrams else None,
                )
            if args.no_gitignore:
                print("  gitignore: turned off in config")
            if args.no_diagrams:
                print("  diagrams: turned off in config")
        else:
            update_init_config(
                root / ".quack" / "config.yaml",
                gitignore=manage_gitignore,
                diagrams=diagrams_enabled,
            )
            if not diagrams_enabled:
                print("  diagrams: turned off in config")
        print(f"✓ scaffolded space at {root}")
        if gitignore_summaries:
            print(f"  {gitignore_summaries[-1].format(root)}")
        elif args.no_gitignore:
            print("  gitignore: skipped (--no-gitignore)")
        if args.no_reindex:
            print("  reindex: skipped (--no-reindex)")
        else:
            with swimming("Reindexing") as progress:
                summary = reindex(str(root), progress=progress.update)
                if diagrams_enabled and summary.get("folder_indexes", 0):
                    progress.update(message="Generating diagrams")
                    d = diagram(str(root), progress=progress.update)
                else:
                    d = None
            print(f"✓ reindexed {summary['files']:,} file(s)")
            if d is not None:
                print(f"  diagrams: {d['folder_diagrams']:,} folder(s) + {d['global']}")
            elif not diagrams_enabled:
                print("  diagrams: skipped (index.diagrams: false)")
        print("\nChoose an assistant to auto-write descriptions (optional):\n")
        run_setup(str(root))
        if not config_existed:
            print("\nSemantic search embeddings are optional:\n")
            _maybe_setup_embeddings(str(root), can_build=not args.no_reindex)
        print("\nNext: `cd` in and run `quack mcp install` to connect an LLM.")
        return 0

    if args.command == "setup":
        run_setup(args.root)
        return 0

    if args.command == "generate":
        try:
            result = _run_generate(args)
        except AINotConfigured:
            if not _offer_setup(args.root):
                return 1
            try:
                result = _run_generate(args)
            except AINotConfigured:
                print("No AI configured. Nothing generated.")
                return 1
        return 0 if result else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
