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
    quack where              show where the root, toolkit, and command live

All commands accept --root PATH (defaults to walking up for `.quack/`, then
$QUACK_ROOT / $OBSIDIAN_VAULT).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quack",
        description="quack builds a navigable meta layer over your local work "
        "(notes, docs, projects) so LLMs can find anything fast. Document-agnostic.",
        epilog="Run `quack <command> --help` for command-specific options.",
    )
    parser.add_argument("--version", action="version", version=f"quack {__version__}")
    # Not required: bare `quack` falls through to printing help (see main()).
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_reindex = sub.add_parser("reindex", help="regenerate the AI navigation layer")
    _add_root_arg(p_reindex)
    p_reindex.add_argument(
        "--no-diagrams", action="store_true", help="skip Mermaid diagram generation"
    )

    p_diagram = sub.add_parser("diagram", help="regenerate Mermaid link diagrams only")
    _add_root_arg(p_diagram)

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

    p_where = sub.add_parser("where", help="show root, toolkit, and command paths")
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
    p_search.add_argument("-n", "--limit", type=int, default=10, help="max results")
    p_search.add_argument(
        "--no-expand", action="store_true", help="skip graph-neighbour expansion"
    )
    p_search.add_argument(
        "--fts", action="store_true", help="force only DuckDB BM25 full-text ranking"
    )
    p_search.add_argument(
        "--semantic", action="store_true", help="force only vss semantic ranking"
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

    p_embed = sub.add_parser("embed", help="build semantic embeddings (DuckDB vss)")
    _add_root_arg(p_embed)

    p_mcp = sub.add_parser("mcp", help="MCP server for LLM tool access")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command", metavar="<subcommand>")
    mcp_sub.add_parser("serve", help="run the server on stdio (clients launch this)")
    p_mcp_print = mcp_sub.add_parser("print", help="print the mcpServers JSON snippet")
    _add_root_arg(p_mcp_print)
    p_mcp_install = mcp_sub.add_parser(
        "install", help="register the server with .mcp.json + installed clients"
    )
    _add_root_arg(p_mcp_install)
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

    p_kiro = sub.add_parser("kiro", help="Kiro integration")
    kiro_sub = p_kiro.add_subparsers(dest="kiro_command", metavar="<subcommand>")
    p_kiro_install = kiro_sub.add_parser(
        "install", help="write .kiro/hooks/*.kiro.hook files"
    )
    _add_root_arg(p_kiro_install)
    p_kiro_send = kiro_sub.add_parser("send", help="send a prompt to kiro-cli")
    p_kiro_send.add_argument("prompt")

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

    cmd = getattr(args, "mcp_command", None)

    if cmd in (None, "serve"):
        from .mcp_server import main as mcp_main

        mcp_main()
        return 0

    if cmd == "print":
        print(mi.mcp_json_snippet(args.root))
        return 0

    if cmd == "status":
        st = mi.status(args.root)
        flag = "yes" if st["project_mcp_json"] else "no"
        print(f"project .mcp.json: {flag}")
        for key, state in st["clients"].items():
            print(f"  {key}: {state}")
        return 0

    if cmd == "install":
        path = mi.write_project_config(args.root)
        print(f"✓ wrote {path}")
        if args.project_only:
            return 0
        for client in mi.CLIENTS:
            if not client.installed:
                print(f"  {client.label}: not installed, skipping")
                continue
            register = " ".join(mi.register_command(client.key, args.root))
            if not args.yes:
                print(f"\n{client.label} is a global config change. Command:")
                print(f"  {register}")
                try:
                    ans = input(f"Register quack with {client.label}? [y/N] ").strip().lower()
                except EOFError:
                    ans = ""
                if ans not in ("y", "yes"):
                    print("  skipped")
                    continue
            ok, out = mi.run_register(client.key, args.root)
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


def _run_generate(args) -> bool:
    """Run description generation and print results. Returns True on success."""
    result = fill_descriptions(
        args.root, only=args.only, dry_run=args.dry_run, include_stale=args.stale
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
        reindex(args.root)
        print("  reindexed")
    return True


def _offer_setup(space_arg) -> bool:
    """No AI configured: offer to run setup. Returns True if now configured."""
    config = Config.load(space_arg)
    if config.ai.skip:
        print("AI is turned off (ai.skip: true in config.yaml).")
        print("Run `quack setup` to enable an assistant.")
        return False
    print("No AI assistant is set up yet.")
    print("The assistant writes short descriptions of your files and folders.")
    try:
        answer = input("Would you like to set one up now? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        print("Skipped. Run `quack setup` later, or write descriptions yourself.")
        return False
    result = run_setup(space_arg)
    return result.configured


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


def _dispatch(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:  # bare `quack` → show help instead of erroring
        parser.print_help()
        return 0

    if args.command == "reindex":
        summary = reindex(args.root)
        print(
            f"✓ reindexed {summary['files']} files across "
            f"{summary['folder_indexes']} folder(s)\n"
            f"  map:     {summary['map']}\n"
            f"  catalog: {summary['db']}"
        )
        if not args.no_diagrams:
            d = diagram(args.root)
            print(f"  diagrams: {d['folder_diagrams']} folder(s) + {d['global']}")
        return 0

    if args.command == "diagram":
        d = diagram(args.root)
        print(f"✓ wrote {d['folder_diagrams']} folder diagram(s)\n  global: {d['global']}")
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

        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        rel = generate.record(args.root, args.path, args.description, tags)
        if rel is None:
            print(f"✗ no indexed file matching {args.path!r}", file=sys.stderr)
            return 1
        print(f"✓ described {rel}")
        if not args.no_reindex:
            reindex(args.root)
            print("  reindexed")
        return 0

    if args.command == "where":
        root = find_root(args.root)
        toolkit = Path(__file__).resolve().parents[2]  # .../<root>/.quack
        print(f"root:     {root}")
        print(f"toolkit:  {toolkit}")
        print(f"command:  {Path(sys.argv[0]).resolve()}")
        print(f"guide:    {toolkit / 'GUIDE.md'}")
        return 0

    if args.command == "search":
        terms = " ".join(args.query)
        root = str(find_root(args.root))
        if args.fts:
            from . import catalog

            rows = catalog.fts_search(terms, explicit_root=args.root, limit=args.limit)
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
                rows = semantic_search(terms, explicit_root=args.root, limit=args.limit)
            except EmbedNotConfigured:
                print("No embeddings. Configure `embed.command` and run `quack embed`.")
                return 1
            print(f"# root: {root}  (paths below are relative to it)")
            for rel, name, dist in rows:
                print(f"{rel}  [cosine {dist:.3f}]")
            return 0
        hits = search(
            terms,
            explicit_root=args.root,
            limit=args.limit,
            expand=not args.no_expand,
        )
        print(format_hits(hits, root=root))
        return 0 if hits else 1

    if args.command == "sql":
        from . import catalog

        cols, rows = catalog.query(args.query, explicit_root=args.root)
        print(_format_table(cols, rows, csv=args.csv))
        return 0

    if args.command == "mcp":
        return _run_mcp(args)

    if args.command == "graph":
        return _run_graph(args)

    if args.command == "embed":
        from .embed import EmbedNotConfigured, build_embeddings

        try:
            result = build_embeddings(args.root)
        except EmbedNotConfigured:
            print("No embedding command. Set `embed.command` in .quack/config.yaml,")
            print("then run `quack embed`. It must print a JSON array of floats.")
            return 1
        print(f"✓ embedded {result['embedded']} notes (dim {result['dim']})")
        return 0

    if args.command == "init":
        from .scaffold import scaffold_root

        root = scaffold_root(args.dir or args.root)
        print(f"✓ scaffolded space at {root}")
        summary = reindex(str(root))
        print(f"  reindexed {summary['files']} file(s)")
        print("\nChoose an assistant to auto-write descriptions (optional):\n")
        run_setup(str(root))
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

    if args.command == "kiro":
        if args.kiro_command == "install":
            written = kiro_mod.install_hooks(args.root)
            print(f"✓ installed {len(written)} Kiro hook(s):")
            for p in written:
                print(f"    {p}")
            return 0
        if args.kiro_command == "send":
            print(kiro_mod.send(args.prompt))
            return 0
        print("usage: quack kiro {install|send}")
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
