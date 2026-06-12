# quack, AI navigation guide

quack is a meta layer over a directory of your work — any files: notes, docs,
code, configs, assets. Built for cheap, precise AI retrieval. **Read only what
you need, in this order, never scan the whole tree.**

## Source of truth

The authored metadata for every file lives in its folder's **editable**
`.index.yaml` (`description` + `tags`; `links` is derived from `[[wikilinks]]`).
Markdown may additionally carry frontmatter, which seeds the store. `quack
reindex` MERGES — your descriptions/tags are preserved — and regenerates
everything below, which must never be hand-edited:

| Artifact | Scope | Use it to… |
|---|---|---|
| `.quack/map.yaml` | folder-level | decide **which folder** is relevant |
| `<folder>/.index.yaml` | file-level, one per folder | author + pick the **1-3 files** that match |
| `.quack/quack.duckdb` | the catalog (files, tags, links, FTS) | **search** metadata and pull only the related slice |
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
     `tags(name, tag)`, `links(src, dst, dst_exists)`.

The graph lives in the catalog's `links` table; multi-hop traversal is a
recursive CTE behind `quack search` and `quack graph`, so you never load the
entire graph to find a few neighbours. Read cost scales with *relevance*.

## Conventions

- Folders are lowercase, topic-based. Add a folder freely; `reindex` discovers it.
- A folder can describe itself with a `.folder.md` note (its `description`
  becomes the folder's line in `map.yaml`).
- Ignore patterns live in `.quackignore` at the root (built-ins like `.quack`,
  `.git`, `node_modules` are always skipped).
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
