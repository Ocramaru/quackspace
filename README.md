# quack

A navigation layer over a local directory of files (notes, docs, code, configs, assets) that lets you waddle at supersonic speeds. LLMs find exactly what they need without scanning everything.

PyPI package: `quackspace` · command: `quack`

## Why it exists

LLMs are good at reading files once they find them. The hard part is knowing which files matter. `quack` solves that: it indexes your workspace into a DuckDB catalog with descriptions, tags, and a link graph, exposes it over MCP, and keeps a human-readable `QUACK.md` as a navigation anchor. Instead of dumping whole directories into context, you waddle straight to the right file with precise retrieval at the file level, graph neighbours pulled in as needed.

It also works great without AI: you can author descriptions yourself and get a searchable, queryable catalog of your own files.

## Quick start

```bash
# Install the quack CLI
curl -fsSL https://raw.githubusercontent.com/Ocramaru/quackspace/main/install.sh | bash

# Create a new workspace (or `quack init` to use the current folder)
quack init my-workspace
cd my-workspace

# Connect an LLM (Claude Code, Kiro, ...) over MCP
quack mcp install

# Index everything
quack reindex
```

Prefer Python packaging?

```bash
uv tool install quackspace   # or: pipx install quackspace
```

## Two things to know

**`quack` (the package)** is the installed CLI and MCP server. It goes in your `PATH` and finds the right workspace by walking up for `.quack/`, just like `git` finds `.git`.

**`.quack/` (workspace state)** is a hidden folder at your workspace root. It holds the DuckDB catalog, the generated map, and your config. This is local to each workspace; the installed package is shared.

## How it works

A **Quack Space** is any directory you run `quack init` in. After that:

```
my-workspace/               <- the Quack Space root
├── .quack/
│   ├── config.yaml         your AI assistant choice
│   ├── map.yaml            GENERATED: folder tree with descriptions
│   └── quack.duckdb        GENERATED: full catalog (files, tags, links, FTS, embeddings)
├── .quackignore            optional extra ignore patterns (dependency trees and
│                           large datasets are skipped automatically — see below)
├── QUACK.md                navigation anchor for LLMs (generated)
├── notes/  docs/  src/ ...   your actual files
│   └── .index.yaml         EDITABLE: descriptions + tags for this folder's children
└── ...
```

**The one rule:** the only file you edit by hand is each folder's `.index.yaml`. It describes that folder's direct children: files and subfolders. Run `quack reindex` after any change and everything else regenerates: the map, the catalog, the diagrams.

`quack reindex` merges your descriptions (it never overwrites what you wrote) and rebuilds the catalog from scratch. Delete `.quack/quack.duckdb` and `reindex` brings it back.

## Commands

```bash
quack init [dir]            # create & scaffold a new space, then index it
quack init --no-reindex     # scaffold only; tune .quackignore before indexing
quack init --no-gitignore   # scaffold without writing quack-managed .gitignore rules
quack init --no-diagrams    # scaffold with diagram generation turned off in config
quack init --dry-run        # show what init would write without changing files
quack reindex               # rebuild everything (catalog, map, diagrams)
quack search "terms"        # auto-hybrid: keyword + FTS + semantic + graph
quack search "terms" --fts  # force BM25 full-text ranking
quack sql "SELECT ..."      # query the catalog directly
quack graph path a b        # shortest link path between two files
quack describe PATH -d "..." -t tag,tag   # record a description for any file
quack generate              # AI: fill in missing descriptions
quack generate --stale      # also refresh stale ones
quack embed init            # choose embeddings (Ollama recommended)
quack embed init --provider ollama --pull  # pull/use local nomic-embed-text
quack embed                 # build semantic embeddings
quack new "Title" -f folder -d "..." -t tag,tag   # new markdown note
quack doctor                # check links, descriptions, MCP registration
quack clean --dry-run       # show generated artifacts clean would remove
quack clean --diagrams      # remove only generated Mermaid diagrams
quack clean --catalog --map # remove only catalog + map
quack setup                 # choose an AI assistant
quack mcp install           # register the MCP server with Claude Code / Kiro
quack where                 # show workspace, state, package, and command paths
```

Root resolution order: `--root` → walk up for `.quack/` → `$QUACK_ROOT` → `$OBSIDIAN_VAULT`. If none resolves to a directory containing `.quack/`, the command tells you to run `quack init` or pass `--root`.

Fresh interactive `quack init` asks a couple setup questions before it writes:
whether quack should manage generated-file `.gitignore` rules, and whether
future `quack reindex` runs should generate Mermaid diagrams. It also offers
to configure optional semantic-search embeddings. Existing `.quack/config.yaml`
files are preserved unless you pass an explicit flag.

Interactive `quack clean` shows a small menu for cleanup scope. In scripts, it
keeps the default safe behavior and removes only derived artifacts unless you
pass flags like `--diagrams`, `--catalog --map`, or `--all`.

## What gets indexed

`reindex` walks the whole root, but skips three kinds of noise so the catalog
stays about your actual work:

- **`.quackignore`** (root) — one pattern per line, matched against each
  file/dir name **and** its root-relative path (globs via fnmatch, e.g.
  `*.lock`). Use it for build outputs and scratch dirs.
- **Dependency trees** — `node_modules`, `site-packages`, `.venv`, `.tox`, … are
  recorded as folders (so an agent knows they exist) but never descended into.
  Caches (`__pycache__`, `.mypy_cache`, …) and `.git`/`.quack` are hidden
  entirely. No config needed.
- **Datasets (by size, not name)** — a folder with more files than
  `index.dataset_threshold` (default 10000, any type), or more than
  `index.dataset_ext_threshold` (default 500) files of one bulk-data type
  (`.npy`, `.png`, tensors, parquet…), is recorded and tagged `dataset` but its
  files aren't indexed one by one — so a 200k-file data dump can't drown the
  catalog. Set either to `0` in `.quack/config.yaml` to disable.

## The catalog

`quack.duckdb` is a DuckDB database built by `reindex`. It's the queryable store for everything:

- **`files`**: name, rel path, folder, ext, description, tags, link counts, stale flag, body
- **`folders`**: folder, parent, description, file count, diagram
- **`tags`**: name to tag index
- **`links`**: src, dst, dst_exists (edge list; multi-hop via recursive CTE)
- **BM25 full-text index** over name, description, and body. Set
  `index.store_body: false` in `.quack/config.yaml` and run `quack reindex` to
  leave `files.body` empty and limit catalog full-text search to
  names/descriptions.
- **Diagrams** are generated during `quack reindex` by default when folder
  indexes change. Set `index.diagrams: false` in `.quack/config.yaml`, or run
  `quack reindex --no-diagrams` to skip them once.
- **`embeddings` / `folder_embeddings`**: separate vector spaces after
  `quack embed init` configures an embedding command and `quack embed` builds
  vectors. The recommended local setup is Ollama with `nomic-embed-text`; run
  `quack embed init --provider ollama --pull` and QuackSpace will skip the pull
  when the model is already installed. Interactive setup can offer to install
  Ollama, start `ollama serve`, and pull the model when needed. QuackSpace also
  ships a free no-setup fallback (`quack embed text`), and you can choose
  `--provider custom --command "..."` with any command that prints a JSON array
  of floats. Files embed labeled path, name, folder, type, tags, links,
  description, and a bounded body for source/prose files. Data and asset files
  embed metadata only; set `embed.include_body: false` in `.quack/config.yaml`
  to make all file embeddings metadata-only. Folders embed labeled path,
  description, type/tag rollups, and direct child names/descriptions. Re-running `quack embed`
  refreshes missing or stale vectors and prunes deleted paths; use
  `quack embed --rebuild` to recreate the vector cache from scratch.

```bash
quack sql "SELECT folder, count(*) FROM files GROUP BY folder"
quack sql "SELECT rel FROM files WHERE stale"              # descriptions to refresh
quack sql "SELECT src, dst FROM links WHERE NOT dst_exists"  # broken links
```

## LLM access (MCP)

`quack mcp install` writes `.mcp.json` at the workspace root (the auto-discover convention Claude Code and Kiro pick up) and optionally registers with installed client CLIs.

The MCP server exposes `map`, `search`, `file_meta`, `sql`, `graph_path`, `central`, `clusters`, and `explain` (read-only) plus `describe` and `reindex` (write). `file_meta` returns a file's metadata and `absolute_path` — never its content, so reads flow through the host's own permission-checked tools. `explain` gives an agent a guided tour of the tools and next-step suggestions. Every result includes `root` so the LLM can construct absolute paths.

**Seeding a repo an agent already knows:** point the MCP server at a codebase and ask the assistant to call `describe(path, description, tags)` for each file it understands, then `reindex()` once. No per-file model shell-out; the agent writes what it already knows, and the catalog becomes searchable.

## AI is optional

The AI does one thing: write short descriptions + tags for your files (`quack generate`). quack works fully without it; just author `.index.yaml` entries yourself.

- `quack setup` shows an arrow-key menu (kiro-cli, claude, a custom command, or "none") and writes the choice to `.quack/config.yaml`.
- `quack generate` fills missing descriptions. Without a configured assistant it explains and offers to run setup.
- Set `ai.skip: true` in `config.yaml` to permanently opt out; `generate` won't prompt again.
- Swap assistants by re-running `quack setup` or editing `ai.command` in `config.yaml` (use `{prompt}` placeholder, or omit to pipe on stdin).

## Keeping it in sync

Run `quack reindex` after structural changes. Automation options:

- Git **pre-commit** hook: `quack doctor --strict --files && quack reindex`
- Obsidian **Shell Commands** plugin (run on vault save)
- A file-watcher pointed at the workspace root
- `quack agent kiro install`: writes a Kiro hook that reindexes on save

## Releasing

Quack uses uv-native packaging. CI runs `uv sync --locked --dev` and `uv run pytest tests`. For mainline releases: set a stable version, tag it `vX.Y.Z`, and push. The `Publish release to PyPI` workflow builds, smoke-tests, and publishes via PyPI Trusted Publishing.

For a local dev/beta wheel:

```bash
uv version --bump patch --bump dev=$(date -u +%Y%m%d%H%M) --no-sync
uv build --wheel
uv tool install --force dist/quackspace-*.whl
```
