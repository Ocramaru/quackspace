"""Health checks that keep the system honest.

quack now indexes every file, and descriptions are optional annotations rather
than a requirement, so the only hard fault is a broken wikilink (a link that
resolves to nothing). Missing descriptions are reported as an informational
nudge toward `quack generate`, not a failure. Well-known files and folders get
a recognition default, so boilerplate (a `.gitignore`, a `tests/` folder) is
never flagged as undescribed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import Space


@dataclass
class Report:
    n_files: int
    missing_description: list[str]  # files with no description yet (informational)
    stale: list[str]  # described, but the file changed since (informational)
    broken_links: list[tuple[str, str]]  # (source file, dangling wikilink target)
    folders_missing_description: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # Descriptions are optional; only unresolved links are a real fault.
        return not self.broken_links


def diagnose(explicit_root: str | None = None) -> Report:
    from . import folders as _folders

    space = Space.load(explicit_root)
    names = set(space.by_name)

    # Recognition defaults already populate e.description, so recognized files
    # are not counted as missing.
    missing = sorted(e.rel for e in space.entries if not e.description)
    stale = sorted(e.rel for e in space.entries if e.stale)

    broken: list[tuple[str, str]] = []
    for e in space.entries:
        for target in e.links:
            if target not in names:
                broken.append((e.rel, target))

    infos = _folders.resolve_folders(space)
    folders_missing = sorted(
        i.rel for i in infos.values() if not i.is_root and not i.description
    )

    return Report(
        n_files=len(space.entries),
        missing_description=missing,
        stale=stale,
        broken_links=sorted(broken),
        folders_missing_description=folders_missing,
    )


def format_report(r: Report) -> str:
    lines: list[str] = []
    if r.broken_links:
        lines.append(f"✗ {len(r.broken_links)} broken wikilink(s):")
        lines += [f"    {src} → [[{tgt}]]" for src, tgt in r.broken_links]
    if r.missing_description:
        n = len(r.missing_description)
        lines.append(
            f"⚠ {n} of {r.n_files} file(s) have no description yet "
            "(optional — run `quack generate` to fill them in):"
        )
        lines += [f"    {p}" for p in r.missing_description[:10]]
        if n > 10:
            lines.append(f"    … and {n - 10} more")
    if r.stale:
        n = len(r.stale)
        lines.append(
            f"⚠ {n} description(s) may be stale (file changed since written — "
            "run `quack generate --stale` to refresh):"
        )
        lines += [f"    {p}" for p in r.stale[:10]]
        if n > 10:
            lines.append(f"    … and {n - 10} more")
    if r.folders_missing_description:
        n = len(r.folders_missing_description)
        lines.append(
            f"⚠ {n} folder(s) have no description (neither authored nor "
            "recognized — describe them in the parent's .index.yaml):"
        )
        lines += [f"    {p}/" for p in r.folders_missing_description[:10]]
        if n > 10:
            lines.append(f"    … and {n - 10} more")
    if not lines:
        return f"✓ {r.n_files} files indexed; descriptions present, links resolve."
    return "\n".join(lines)

