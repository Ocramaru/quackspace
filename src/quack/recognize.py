"""Zero-cost recognition defaults for well-known files and folders.

A static table maps boilerplate the world already understands — `.gitignore`,
`pyproject.toml`, a `tests/` folder — to a short default description and a few
tags. It is deterministic, instant, and identical across every space, so quack
never spends an AI call describing things that need no judgement.

Recognition is the **lowest** precedence layer: authored `.index.yaml` and
Markdown frontmatter both win over it (see ``core.Entry``). Defaults are written
back with a blank ``described_at`` so they stay *non-sticky* — re-derived each
reindex and freely overridable by ``describe``/``generate``.

Pure data + lookup, no I/O.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

# (description, tags) keyed by exact file name. Highest-priority file match.
EXACT_FILES: dict[str, tuple[str, list[str]]] = {
    ".gitignore": ("Files and paths Git should not track.", ["git", "config"]),
    ".gitattributes": ("Per-path Git attributes.", ["git", "config"]),
    ".editorconfig": ("Editor formatting conventions shared across the repo.", ["config", "editor"]),
    ".python-version": ("Pinned Python interpreter version.", ["python", "config"]),
    ".dockerignore": ("Paths excluded from the Docker build context.", ["docker", "config"]),
    "LICENSE": ("The project's software license.", ["legal", "license"]),
    "LICENSE.md": ("The project's software license.", ["legal", "license"]),
    "LICENSE.txt": ("The project's software license.", ["legal", "license"]),
    "README.md": ("Project overview and entry-point documentation.", ["docs", "readme"]),
    "pyproject.toml": ("Python project metadata, dependencies, and build config.", ["python", "packaging", "config"]),
    "setup.py": ("Legacy Python package build script.", ["python", "packaging"]),
    "setup.cfg": ("Python package configuration.", ["python", "packaging", "config"]),
    "requirements.txt": ("Pinned Python dependencies.", ["python", "dependencies"]),
    "Dockerfile": ("Container image build recipe.", ["docker", "build"]),
    "docker-compose.yml": ("Multi-container Docker service definitions.", ["docker", "config"]),
    "docker-compose.yaml": ("Multi-container Docker service definitions.", ["docker", "config"]),
    "Makefile": ("Build and task automation targets.", ["build", "make"]),
    "package.json": ("Node project metadata, scripts, and dependencies.", ["node", "javascript", "config"]),
    "tsconfig.json": ("TypeScript compiler configuration.", ["typescript", "config"]),
    "CHANGELOG.md": ("Notable changes per release.", ["docs", "changelog"]),
    "CONTRIBUTING.md": ("How to contribute to the project.", ["docs"]),
}

# (description, tags) keyed by glob pattern. Checked after exact names.
GLOB_FILES: list[tuple[str, tuple[str, list[str]]]] = [
    ("*.lock", ("Resolved dependency lockfile.", ["dependencies", "lockfile"])),
    ("*.lockb", ("Resolved dependency lockfile.", ["dependencies", "lockfile"])),
    (".env*", ("Environment variable definitions.", ["config", "env"])),
]

# (description, tags) keyed by lowercase extension (no dot). Lowest file match.
EXTENSIONS: dict[str, tuple[str, list[str]]] = {
    "py": ("Python source file.", ["python", "source"]),
    "pyi": ("Python type stub.", ["python", "types"]),
    "js": ("JavaScript source file.", ["javascript", "source"]),
    "ts": ("TypeScript source file.", ["typescript", "source"]),
    "tsx": ("TypeScript React component.", ["typescript", "react", "source"]),
    "jsx": ("JavaScript React component.", ["javascript", "react", "source"]),
    "sh": ("Shell script.", ["shell", "script"]),
    "bash": ("Bash script.", ["shell", "script"]),
    "toml": ("TOML configuration file.", ["config", "toml"]),
    "yml": ("YAML configuration file.", ["config", "yaml"]),
    "yaml": ("YAML configuration file.", ["config", "yaml"]),
    "json": ("JSON data or configuration file.", ["config", "json"]),
    "ini": ("INI configuration file.", ["config", "ini"]),
    "cfg": ("Configuration file.", ["config"]),
    # Prose extensions (md, rst, txt) are deliberately NOT recognized: their
    # type says nothing about their content, and a generic default would make
    # `quack generate` skip every note. They stay blank so generate/describe
    # fill in real descriptions. (Known doc files like README.md are still
    # recognized by exact name above.)
    "csv": ("Comma-separated tabular data.", ["data", "csv"]),
    "sql": ("SQL script.", ["sql", "database"]),
    "html": ("HTML document.", ["web", "html"]),
    "css": ("Cascading style sheet.", ["web", "css"]),
}

# (description, tags) keyed by folder name. Matched case-insensitively.
FOLDERS: dict[str, tuple[str, list[str]]] = {
    "tests": ("Automated test suite.", ["tests"]),
    "test": ("Automated test suite.", ["tests"]),
    "src": ("Primary source tree.", ["source"]),
    "docs": ("Project documentation.", ["docs"]),
    "doc": ("Project documentation.", ["docs"]),
    "examples": ("Usage examples.", ["docs", "examples"]),
    "scripts": ("Utility and automation scripts.", ["scripts"]),
    "bin": ("Executable scripts and binaries.", ["scripts", "bin"]),
    ".github": ("GitHub configuration and CI workflows.", ["ci", "github", "config"]),
    ".vscode": ("VS Code workspace settings.", ["editor", "config"]),
    # Opaque dirs (see core.DEFAULT_OPAQUE_DIRS): mentioned, never indexed.
    "node_modules": ("Installed Node dependencies (not indexed).", ["node", "dependencies"]),
    "site-packages": ("Installed Python packages (not indexed).", ["python", "dependencies"]),
    "bower_components": ("Installed Bower dependencies (not indexed).", ["dependencies"]),
    ".venv": ("Python virtual environment (not indexed).", ["python", "venv"]),
    "venv": ("Python virtual environment (not indexed).", ["python", "venv"]),
    "virtualenv": ("Python virtual environment (not indexed).", ["python", "venv"]),
    ".tox": ("tox virtual environments (not indexed).", ["python", "tox"]),
    ".eggs": ("Python egg build artifacts (not indexed).", ["python", "build"]),
    "migrations": ("Database schema migrations.", ["database", "migrations"]),
    "assets": ("Static assets.", ["assets"]),
    "static": ("Static web assets.", ["web", "assets"]),
    "templates": ("Template files.", ["templates"]),
}


def recognize_file(path: str | Path) -> tuple[str, list[str]] | None:
    """Default ``(description, tags)`` for a well-known file, else ``None``.

    Precedence: exact name beats glob beats extension. Tags are returned as a
    fresh list so callers can mutate freely.
    """
    p = Path(path)
    name = p.name

    hit = EXACT_FILES.get(name)
    if hit is not None:
        return hit[0], list(hit[1])

    for pattern, value in GLOB_FILES:
        if fnmatch(name, pattern):
            return value[0], list(value[1])

    ext = p.suffix.lower().lstrip(".")
    hit = EXTENSIONS.get(ext)
    if hit is not None:
        return hit[0], list(hit[1])

    return None


def recognize_folder(name: str) -> tuple[str, list[str]] | None:
    """Default ``(description, tags)`` for a well-known folder name, else
    ``None``. Matched on the folder's base name, case-insensitively."""
    hit = FOLDERS.get(name) or FOLDERS.get(name.lower())
    if hit is not None:
        return hit[0], list(hit[1])
    return None
