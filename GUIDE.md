# quack, AI navigation guide

quack is a meta layer over a directory of your work — any files: notes, docs,
code, configs, assets. Built for cheap, precise AI retrieval. **Read only what
you need, in this order, never scan the whole tree.**

## Source of truth

The authored metadata for every folder's **direct children** lives in its
**editable** `.index.yaml`: a `files:` section (`description` + `tags`; `links`
derived from `[[wikilinks]]`) and a `directories:` section describing its
immediate subfolders. A folder is described by its *parent's* index, the same
way a file is. There is one `.index.yaml` per folder — including folders that
contain only subfolders, and the root.

Well-known files and folders (`.gitignore`, `*.py`, `tests/`, `docs/`) get an
instant **recognition default** description with no AI call. Precedence is
authored `.index.yaml` → Markdown frontmatter → recognition default → blank.
Recognition defaults are non-sticky (blank `described_at`), so they re-derive
each reindex and are freely overridden by `describe`/`generate`.

`quack reindex` MERGES — your descriptions/tags are preserved — and regenerates
everything below, which must never be hand-edited:

| Artifact | Scope | Use it to… |
|---|---|---|
| `.quack/map.yaml` | global nested tree | decide **which folder** is relevant |
| `<folder>/.index.yaml` | one per folder: `files:` + `directories:` | author + pick the **1-3 children** that match |
| `.quack/quack.duckdb` | the catalog (files, folders, tags, links, FTS, embeddings) | **search** metadata and pull only the related slice |
| `.quack/diagram.md` + `<folder>/_diagrams.md` | Mermaid | **visualise** the link graph |

`.index.yaml` is the one file you edit by hand; everything else is generated.

## Drill-down (the whole trick)

1. Read this file + `.quack/map.yaml` (tiny, one line per folder). Choose folder(s).
2. Read that folder's `.index.yaml`. Choose the files that match the query.
3. Read those files. To pull adjacent context, query the catalog instead of
   loading the whole graph, it returns only the slice you need:
   - `quack search "<terms>"` auto-hybrid ranks files (keyword + FTS + semantic
     if available) and adds their graph neighbours. `--fts`/`--semantic` force
     a single tier.
   - `quack sql "<SQL>"` queries the catalog directly. Tables: `files`
     (name, rel, folder, ext, description, tags_csv, n_links, n_inbound,
     is_orphan, is_binary, file_modified, described_at, stale, body),
     `folders(folder, parent, description, n_files, diagram, described_at)`
     — the direct subfolders of X are `WHERE parent = 'X'` (root is `''`) —
     `tags(name, tag)`, `links(src, dst, dst_exists)`.

`quack search` also routes *where/which-folder* questions to folder-level
results (the `folders` list), kept separate from file hits. With embeddings
built (`quack embed`), files and folders have their own vector spaces
(`embeddings` and `folder_embeddings`) so the two never blend.

The graph lives in the catalog's `links` table; multi-hop traversal is a
recursive CTE behind `quack search` and `quack graph`, so you never load the
entire graph to find a few neighbours. Read cost scales with *relevance*.

## Conventions

- Folders are lowercase, topic-based. Add a folder freely; `reindex` discovers it.
- A folder's description resolves: parent `.index.yaml` `directories:` (authored)
  → that folder's `.folder.md` frontmatter → folder recognition default → blank.
  Author one by editing the parent's `directories:` section, or over MCP call
  `describe` then `reindex`.
- Ignore patterns live in `.quackignore` at the root: one pattern per line,
  matched against each file/dir name **and** its root-relative path (globs via
  fnmatch, e.g. `*.lock`). Built-ins are automatic: `.quack`/`.git`/`.obsidian`/
  `.trash` and caches (`__pycache__`, `.mypy_cache`, …) are hidden entirely;
  vendored/dependency trees (`site-packages`, `node_modules`, `.venv`, `.tox`,
  …) are **mentioned** as folders but their contents are not indexed.
- Datasets are detected by size, not name: a folder with more files than
  `index.dataset_threshold` (any type), or more than `index.dataset_ext_threshold`
  files of one bulk-data type (`.npy`, `.png`, tensors…), is recorded and tagged
  `dataset` but its files aren't indexed — so a 200k-file data dump can't drown
  the catalog. Tune or disable (set `0`) both in `.quack/config.yaml`.
- Author metadata by editing a folder's `.index.yaml`, or let the assistant
  classify it: `quack generate` writes a description + tags for every file
  missing one (`--stale` also refreshes ones whose file changed since).
- Already know the repo? Record it directly with `quack describe PATH -d "…"
  -t tag,tag`, or — over MCP — call `describe(path, description, tags)` for each
  file then `reindex()` once. No per-file model call; you write what you know.
- New Markdown note: `quack new "Title" -f folder -d "..." -t tag,tag`.
- After any structural change run `quack reindex` (also refreshes diagrams).
- `quack doctor` reports broken wikilinks (the only hard fault), plus missing or
  stale descriptions.
