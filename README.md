# quack

A DuckDB-backed knowledge layer over your local work — any files: notes, docs,
code, configs, assets — that helps LLMs navigate everything. It generates a
cheap, precise meta layer from your files; you author metadata in one editable
place (`.index.yaml`), and everything else is derived. Plays well with Obsidian
but does not require it.

PyPI package: `quackspace` · command: `quack`

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/Ocramaru/quackspace/main/install.sh | bash
```

This installs `uv` (if needed) and the `quack` CLI globally. Then create a space
anywhere:

```bash
quack init my-workspace   # make & scaffold the folder (or `quack init` to use the current one)
cd my-workspace
quack mcp install         # connect an LLM (Claude Code, Kiro, …) over MCP
```

Prefer Python packaging? `uv tool install quackspace` (or `pipx install
quackspace`) does the same as the one-liner's second step.

### Releasing

Releases publish to PyPI via GitHub Actions + Trusted Publishing (no stored
token). One-time: add a PyPI *pending publisher* (project `quackspace`, owner
`Ocramaru`, repo `quackspace`, workflow `publish.yml`, environment `pypi`) and a
GitHub environment named `pypi`. Then bump `version` in `pyproject.toml`, tag,
and publish a GitHub Release — the workflow builds and uploads. Manual fallback:
`uv build && uv publish --token <pypi-token>`.

## Architecture

```
<your-root>/                ← any directory; quack finds it by the .quack/ marker
├── .quack/                 ← the toolkit (and the root marker)
│   ├── GUIDE.md            hand-written: how an AI should search the tree
│   ├── map.yaml            GENERATED: folder tree + folder descriptions
│   ├── quack.duckdb        GENERATED: catalog (files, tags, links, FTS) — the queryable store
│   ├── diagram.md          GENERATED: whole-graph Mermaid link diagram
│   ├── config.yaml         your AI assistant choice
│   └── src/quack/          the CLI + library
├── .quackignore            optional: extra ignore patterns
├── QUACK.md                visible navigation anchor for LLMs
├── src/  docs/  notes/ …   ANY files: code, configs, markdown, assets
│   ├── .index.yaml         EDITABLE: per-file description + tags (links derived)
│   └── _diagrams.md        GENERATED: this folder's Mermaid link graph
└── …
```

**One rule:** the only thing you edit is each folder's `.index.yaml`
(description + tags per file; Markdown may also use frontmatter). `quack reindex`
MERGES it — preserving your text — and regenerates every map, catalog, and
diagram from the files + `[[wikilinks]]`, so the navigation layer can never
drift. The root can be named anything; quack locates it by walking up for
`.quack/` (like git finds `.git`).

## The catalog (DuckDB)

`quack reindex` builds `.quack/quack.duckdb`, a single embedded catalog of all
metadata: `files` (name, rel, folder, ext, description, tags_csv, n_links,
n_inbound, is_orphan, is_binary, file_modified, described_at, stale, body),
`tags(name, tag)`, and `links(src, dst, dst_exists)`, plus a BM25 full-text
index over name/description/body. (`stale` is true when a file changed after its
description was written — see `quack generate --stale`.)

It is the queryable store and **the graph lives here too**: the `links` table
is the edge list, and multi-hop traversal is a recursive CTE behind
`quack search` and `quack graph`. There is no separate graph.json: a query
pulls only the relevant slice into context instead of loading the whole graph,
which matters as the tree grows. The catalog is derived, never authoritative;
delete it and `quack reindex` rebuilds it from the files.

```bash
quack sql "SELECT folder, count(*) FROM files GROUP BY folder"
quack sql "SELECT rel FROM files WHERE stale"                 # descriptions to refresh
quack sql "SELECT src, dst FROM links WHERE NOT dst_exists"   # broken links
quack search "regex" --fts                                    # BM25 ranking
quack graph path file-a file-b                                # shortest link path
```

## The `quack` command

Once installed (see [Install](#install)), `quack` is on your PATH and finds the
root from any directory inside it by walking up for `.quack/` — so commands work
no matter where you invoke them. (Developing on a checkout instead? `uv run
quack` inside `.quack/` is the equivalent.)

```bash
quack init [dir]       # create & scaffold a new space (dir, or the current folder)
quack reindex          # regenerate everything (indexes, map, catalog, diagrams)
quack reindex --no-diagrams
quack diagram          # diagrams only
quack doctor           # check files + MCP registration
quack doctor --files   # only files;  --mcp only MCP;  --strict to fail on issues
quack new "Title" -f projects -d "one-line description" -t tag,tag   # new markdown note
quack describe PATH -d "…" -t tag,tag   # record a description + tags for any file
quack setup            # choose the AI assistant (arrow-key menu)
quack generate         # AI: write description + tags for files missing one
quack generate --stale # also refresh descriptions whose file changed since
quack search "terms"   # auto-hybrid: structural + FTS + semantic + graph
quack search "terms" --fts        # force DuckDB BM25 full-text ranking
quack search "terms" --semantic   # force vss semantic ranking
quack embed            # build semantic embeddings (DuckDB vss)
quack graph path|central|clusters # graph queries
quack sql "SELECT ..." # query the catalog directly
quack mcp install      # register the MCP server with clients
quack where            # show root / toolkit / command paths
```

Root resolution: `--root` > walk up for `.quack/` > `$QUACK_ROOT` >
`$OBSIDIAN_VAULT` > the package location.

## LLM access (MCP)

`quack mcp install` writes a project-root `.mcp.json` (the auto-discover
convention Claude Code and others pick up) and offers to register with installed
client CLIs (kiro-cli, claude). The server exposes typed tools, `map`, `search`,
`get_file`, `sql`, `graph_path`, `central`, `clusters` (read), plus `describe`
and `reindex` (write), each returning `root` so the LLM can join `root` +
relative path. `QUACK.md` at the root is a visible anchor telling any LLM how to
navigate even without MCP.

**Seeding quack on a repo an agent already knows.** Point the MCP server at a
codebase and ask the assistant to annotate it: for each relevant file it calls
`describe(path, description, tags)` (writing into `.index.yaml` — the file
itself is untouched), then `reindex()` once. No per-file model shell-out; the
agent records what it already understands, and the catalog becomes searchable.

## AI is optional

The assistant is used for one thing: writing short descriptions + tags for your
files (`quack generate`) so the search index is rich. quack works fully without
it — you just author descriptions yourself by editing each folder's
`.index.yaml`.

- `quack setup` shows an arrow-key menu of assistants (kiro-cli, claude, a
  custom command, or "use without AI"), probes which are installed, and writes
  the choice to `.quack/config.yaml`. `quack init` is an alias that runs it.
- `quack generate` fills in missing descriptions using that command. If none is
  set up, it explains what the AI is for and offers to run setup.
- Set `ai.skip: true` in `config.yaml` to use quack without AI permanently;
  `generate` then stops offering to set one up.
- Swap assistants anytime by editing `ai.command` (use `{prompt}` for the
  prompt, or omit it to pipe on stdin) or re-running `quack setup`.

## Keeping it in sync

Run `quack reindex` after structural changes. To automate, wire it to one of:
- Obsidian **Shell Commands** plugin (run on save),
- a git **pre-commit** hook (`quack doctor --strict --files && quack reindex`),
- a file-watcher,
- `quack kiro install` (writes a Kiro reindex-on-save hook).
