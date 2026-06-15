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

### Releasing and beta builds

Quack uses uv-native packaging. CI runs `uv sync --locked --dev` and
`uv run pytest tests`. Release and beta workflows use `uv build`, `uv version`,
and `uv publish`, following Astral uv’s documented GitHub Actions flow.

For a local beta wheel:

```bash
uv version --bump patch --bump dev=$(date -u +%Y%m%d%H%M) --no-sync
uv build --wheel
uv tool install --force dist/quackspace-*.whl
```

If the beta version should not stay in the checkout, restore or bump the version
before merging. For mainline releases, set a stable version, tag it as `vX.Y.Z`,
and push the tag. The `Publish release to PyPI` workflow builds, smoke-tests the
wheel and sdist, then publishes with PyPI Trusted Publishing.

One-time PyPI setup: add a trusted publisher for project `quackspace`, owner
`Ocramaru`, repo `quackspace`, workflow `publish.yml`, environment `pypi`, and
create the matching GitHub environment.

For beta artifacts in GitHub, run the `Build beta wheel` workflow manually. It
uses `uv version --bump patch --bump dev=${{ github.run_number }}`, builds a dev
wheel, smoke-tests it, and uploads the wheel as an artifact. To push that dev
build to TestPyPI, enable the workflow input and configure a matching TestPyPI
trusted publisher/environment named `testpypi`.

## Architecture

A **Quack Space** is the workspace root: the directory of files you want agents
to navigate. The installed package provides the `quack` and `quack-mcp`
commands; workspace-local state lives under `.quack/`.

```
<your-root>/                ← the Quack Space; quack finds it by the .quack/ marker
├── .quack/                 ← workspace-local state/config/index data
│   ├── config.yaml         your AI assistant and embedding command choices
│   ├── map.yaml            GENERATED: full nested folder tree + descriptions/rollups
│   └── quack.duckdb        GENERATED: catalog (files, folders, tags, links, FTS) — the queryable store
├── .quackignore            optional: extra ignore patterns
├── QUACK.md                visible navigation anchor for LLMs
├── src/  docs/  notes/ …   ANY files: code, configs, markdown, assets
│   ├── .index.yaml         EDITABLE: this folder's direct files: + directories:
│   └── _diagrams.md        GENERATED: this folder's Mermaid link graph
└── …
```

**One rule:** the only thing you edit is each folder's `.index.yaml`. It
describes the folder's **direct children**: a `files:` section (description +
tags per file; Markdown may also use frontmatter) and a `directories:` section
for its subfolders — so a folder is described by its parent, just like a file.
Well-known files/folders get a recognition default (authored → frontmatter →
recognition → blank). There is one `.index.yaml` per folder, including
subdir-only folders and the root. `quack reindex`
MERGES it — preserving your text — and regenerates every map, catalog, and
diagram from the files + `[[wikilinks]]`, so the navigation layer can never
drift. The root can be named anything; quack locates it by walking up for
`.quack/` (like git finds `.git`). Source code for the tool stays in the
installed package or development checkout, not inside `.quack/`.

## The catalog (DuckDB)

`quack reindex` builds `.quack/quack.duckdb`, a single embedded catalog of all
metadata: `files` (name, rel, folder, ext, description, tags_csv, n_links,
n_inbound, is_orphan, is_binary, file_modified, described_at, stale, body),
`folders(folder, parent, description, n_files, diagram, described_at)` — the
direct subfolders of X are `WHERE parent = 'X'` (root is `''`), mirroring the
`directories:` sections 1:1 — `tags(name, tag)`, and
`links(src, dst, dst_exists)`, plus a BM25 full-text index over
name/description/body. (`stale` is true when a file changed after its
description was written — see `quack generate --stale`.) With `quack embed`,
files and folders get **separate** vector tables (`embeddings` and
`folder_embeddings`); `quack search` routes where/which-folder questions to the
folder space and keeps file vs folder hits distinct.

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
quack agent kiro install # install Kiro agent hooks
quack where            # show workspace / state / package / command paths
```

Root resolution: `--root` > walk up for `.quack/` > `$QUACK_ROOT` >
`$OBSIDIAN_VAULT`. If none points to a directory containing `.quack/`, commands
that need a Quack Space fail clearly and tell you to run `quack init` or pass
`--root`.

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
- `quack agent kiro install` (writes a Kiro reindex-on-save hook).
