from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from quack.indexer import reindex
from quack.scaffold import scaffold_root


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--perf-files",
        action="store",
        type=int,
        default=None,
        help="number of files for perf benchmarks (default 1000 or $QUACK_PERF_FILES)",
    )


@pytest.fixture
def perf_files(request) -> int:
    """How many files the perf benchmark generates. From --perf-files, else
    $QUACK_PERF_FILES, else 1000."""
    cli = request.config.getoption("--perf-files")
    if cli is not None:
        return cli
    import os

    return int(os.environ.get("QUACK_PERF_FILES", "1000"))


def arg_value(args: list[str], flag: str) -> str:
    idx = args.index(flag)
    return args[idx + 1]


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


@pytest.fixture
def sample_space(tmp_path: Path) -> Path:
    root = scaffold_root(str(tmp_path / "space"))
    projects = root / "projects"
    projects.mkdir(exist_ok=True)
    (projects / "alpha.md").write_text(
        "---\ndescription: Alpha frontmatter description\ntags: [alpha, docs]\n---\n"
        "# Alpha\n\nLinks to [[beta]] and [[missing]]. Regex details live here.\n"
    )
    (projects / "beta.md").write_text("# Beta\n\nBacklink to [[alpha]]. Searchable beta body.\n")
    (root / "resources" / "data.txt").write_text("plain resource body")
    return root


@pytest.fixture
def indexed_mcp_space(tmp_path: Path, monkeypatch) -> Path:
    import quack.mcp_server as mcp_server

    root = scaffold_root(str(tmp_path / "space"))
    notes = root / "projects"
    notes.mkdir(exist_ok=True)
    for i in range(4):
        (notes / f"note-{i}.md").write_text(f"# Note {i}\n\nneedle {i} " + ("x" * 100))
    reindex(str(root))
    monkeypatch.chdir(root)
    mcp_server.configure_root(str(root))
    mcp_server.configure_limits()
    return root
