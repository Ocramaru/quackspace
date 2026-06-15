"""Scale + performance tests (MAR-142 / MAR-143).

The correctness tests run in the normal suite; the timing benchmark is marked
``perf`` and is opt-in (`pytest -m perf`) so the default suite stays fast.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from quack import catalog
from quack.clean import clean
from quack.core import Space
from quack.generate import record
from quack.indexer import reindex
from quack.scaffold import scaffold_root
from quack.search import search


def _make_space(tmp_path: Path, n: int) -> tuple[Path, set[str]]:
    """A synthetic space of *n* markdown files across nested folders, each with
    frontmatter (description + tags), a unique body token, and a wikilink."""
    root = scaffold_root(str(tmp_path / "space"))
    rels: set[str] = set()
    for i in range(n):
        folder = root / f"area{i % 8}" / f"sub{i % 3}"
        folder.mkdir(parents=True, exist_ok=True)
        rel = f"area{i % 8}/sub{i % 3}/note{i}.md"
        (root / rel).write_text(
            f"---\ndescription: Note {i} about topic{i % 20}\n"
            f"tags: [topic{i % 20}, t{i % 5}]\n---\n"
            f"# Note {i}\n\nUnique body token zql{i}. Links to [[note{(i + 1) % n}]].\n"
        )
        rels.add(rel)
    return root, rels


# ---------------------------------------------------------------------------
# MAR-143 — scale correctness: parallel scan + catalog restructure
# ---------------------------------------------------------------------------

def test_scale_reindex_correctness(tmp_path):
    root, rels = _make_space(tmp_path, 600)
    summary = reindex(str(root))
    assert summary["files"] == len(rels)

    # The parallel scan yields exactly the generated files — no dupes, none lost.
    got = [e.rel for e in Space.load(str(root)).entries]
    assert len(got) == len(set(got)) == len(rels)
    assert set(got) == rels

    # Catalog mirrors that: one row per file, no duplicate rels.
    _, rows = catalog.query(
        "SELECT count(*), count(DISTINCT rel) FROM files", explicit_root=str(root)
    )
    assert rows[0][0] == len(rels)
    assert rows[0][1] == len(rels)

    # folders table parent mapping is correct.
    _, frows = catalog.query(
        "SELECT folder, parent FROM folders", explicit_root=str(root)
    )
    fmap = dict(frows)
    assert fmap["area0"] == ""
    assert fmap["area0/sub0"] == "area0"

    # tags landed.
    _, trows = catalog.query("SELECT count(*) FROM tags", explicit_root=str(root))
    assert trows[0][0] > 0


def test_scale_parallel_build_is_deterministic(tmp_path):
    """Two full builds of the same space produce the identical catalog — the
    thread-pool loading order must not change the result."""
    root, _ = _make_space(tmp_path, 500)
    reindex(str(root))
    snap = lambda: catalog.query(  # noqa: E731
        "SELECT rel, name, description, tags_csv, n_links FROM files ORDER BY rel",
        explicit_root=str(root),
    )[1]
    first = snap()

    clean(str(root))  # drop the catalog, force a fresh full rebuild
    reindex(str(root))
    assert snap() == first


def test_scale_incremental_tiers(tmp_path):
    root, rels = _make_space(tmp_path, 300)
    ordered = sorted(rels)
    assert reindex(str(root))["catalog"] == "full"
    assert reindex(str(root))["catalog"] == "skipped"

    # Authoring tags only (description stays the frontmatter value) → light.
    record(str(root), ordered[0], "", ["hot"])
    assert reindex(str(root))["catalog"] == "light"

    # A body change → full (FTS must be rebuilt) and the new text is searchable.
    p = root / ordered[1]
    p.write_text("# Changed\n\nNEWTOKEN_xyzzy lives here.\n")
    os.utime(p, (time.time() + 3, time.time() + 3))
    assert reindex(str(root))["catalog"] == "full"
    hits = search("NEWTOKEN_xyzzy", explicit_root=str(root), expand=False)
    assert any(h.entry.rel == ordered[1] for h in hits)


# ---------------------------------------------------------------------------
# MAR-142 — timing benchmark (opt-in: `pytest -m perf`)
# ---------------------------------------------------------------------------

@pytest.mark.perf
def test_perf_reindex_and_search(tmp_path, capsys):
    root, _ = _make_space(tmp_path, 1000)

    t = time.perf_counter()
    reindex(str(root))
    full = time.perf_counter() - t

    t = time.perf_counter()
    reindex(str(root))
    noop = time.perf_counter() - t

    search("topic3", explicit_root=str(root))  # warm
    t = time.perf_counter()
    for _ in range(10):
        search("topic3 zql42", explicit_root=str(root))
    srch = (time.perf_counter() - t) / 10

    with capsys.disabled():
        print(
            f"\n[perf] 1000 files — full reindex {full * 1000:.0f}ms, "
            f"no-op {noop * 1000:.0f}ms, search {srch * 1000:.1f}ms"
        )

    # Generous ceilings: catch gross regressions without being flaky on slow CI.
    assert full < 10.0
    assert noop < 3.0
    assert srch < 2.0
