"""Datasets are detected by size, not name: a folder with many files (any type)
or many files of one bulk-data extension is recorded in the meta layer but its
files are not indexed one by one. See core.DatasetPolicy / config.IndexConfig."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from quack import catalog
from quack.config import DEFAULT_DATASET_EXTENSIONS
from quack.core import DatasetPolicy, Space, _dataset_reason
from quack.indexer import reindex
from quack.scaffold import scaffold_root


def _reason(names: list[str], policy: DatasetPolicy) -> str:
    counts = (
        Counter(Path(n).suffix.lower().lstrip(".") for n in names)
        if policy.per_ext
        else None
    )
    return _dataset_reason(len(names), counts, policy)


def _set_thresholds(root: Path, *, total: int | None = None, per_ext: int | None = None) -> None:
    """Override the dataset thresholds in the space's config.yaml."""
    cfg = root / ".quack" / "config.yaml"
    data = yaml.safe_load(cfg.read_text()) or {}
    index = data.setdefault("index", {})
    if total is not None:
        index["dataset_threshold"] = total
    if per_ext is not None:
        index["dataset_ext_threshold"] = per_ext
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))


def test_dataset_reason_per_extension_only_bulk_types():
    policy = DatasetPolicy(total=0, per_ext=3, extensions=DEFAULT_DATASET_EXTENSIONS)
    assert _reason([f"a{i}.npy" for i in range(4)], policy) == "4 .npy files"
    assert _reason([f"m{i}.py" for i in range(9)], policy) == ""
    assert _reason([f"a{i}.npy" for i in range(3)], policy) == ""


def test_dataset_reason_generic_count_any_type():
    policy = DatasetPolicy(total=5, per_ext=0, extensions=DEFAULT_DATASET_EXTENSIONS)
    assert _reason([f"m{i}.py" for i in range(6)], policy) == "6 files"
    assert _reason([f"m{i}.py" for i in range(5)], policy) == ""


def test_dataset_reason_inactive_policy_never_fires():
    assert _reason([f"a{i}.npy" for i in range(1000)], DatasetPolicy()) == ""


def test_bulk_extension_folder_recorded_but_not_indexed(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    _set_thresholds(root, per_ext=3)
    data = root / "projects" / "arrays"
    data.mkdir(parents=True)
    for i in range(5):
        (data / f"arr{i}.npy").write_bytes(b"\x00\x01")
    (root / "projects" / "readme.md").write_text("# Real\n\nan indexed note\n")
    reindex(str(root))

    _, rows = catalog.query(
        "SELECT count(*) FROM files WHERE rel LIKE 'projects/arrays/%'",
        explicit_root=str(root),
    )
    assert rows[0][0] == 0
    _, frows = catalog.query(
        "SELECT description FROM folders WHERE folder = 'projects/arrays'",
        explicit_root=str(root),
    )
    assert frows and frows[0][0] == "Dataset: 5 .npy files, not indexed."
    _, mine = catalog.query(
        "SELECT count(*) FROM files WHERE rel = 'projects/readme.md'",
        explicit_root=str(root),
    )
    assert mine[0][0] == 1


def test_generic_threshold_catches_any_type(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    _set_thresholds(root, total=5, per_ext=0)
    bulk = root / "bulk"
    bulk.mkdir()
    for i in range(7):
        (bulk / f"m{i}.py").write_text(f"x = {i}\n")
    sp = Space.load(str(root))
    assert sp.datasets.get("bulk") == "7 files"
    assert not any(e.rel.startswith("bulk/") for e in sp.entries)


def test_non_bulk_extension_below_generic_is_indexed(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    # Only the per-ext trigger is on; .py is not a bulk-data type.
    _set_thresholds(root, total=0, per_ext=3)
    src = root / "src"
    src.mkdir()
    for i in range(6):
        (src / f"m{i}.py").write_text(f"x = {i}\n")
    sp = Space.load(str(root))
    assert "src" not in sp.datasets
    assert sum(1 for e in sp.entries if e.rel.startswith("src/")) == 6


def test_thresholds_zero_disables_detection(tmp_path):
    root = scaffold_root(str(tmp_path / "space"))
    _set_thresholds(root, total=0, per_ext=0)
    data = root / "data"
    data.mkdir()
    for i in range(10):
        (data / f"a{i}.npy").write_bytes(b"\x00")
    sp = Space.load(str(root))
    assert sp.datasets == {}
    assert sum(1 for e in sp.entries if e.rel.startswith("data/")) == 10


def test_reindex_is_noop_second_time_with_dataset(tmp_path):
    """The stat-only no-op check must apply the same dataset skip as the build,
    or a dataset folder would make every reindex look dirty."""
    root = scaffold_root(str(tmp_path / "space"))
    _set_thresholds(root, per_ext=3)
    imgs = root / "projects" / "imgs"
    imgs.mkdir(parents=True)
    for i in range(5):
        (imgs / f"i{i}.png").write_bytes(b"\x89PNG\x00")
    (root / "projects" / "n.md").write_text("# n\n\nbody\n")
    reindex(str(root))
    summary = reindex(str(root))
    assert summary["catalog"] == "skipped"
