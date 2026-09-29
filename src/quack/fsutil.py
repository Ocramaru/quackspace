"""The single choke point for writing/removing quack-*generated* artifacts
(``.index.yaml``, ``_diagrams.md``, ``map.yaml``, the vault diagram).

These live inside folders quack doesn't own — a container-mounted read-only
tree, a vendored ``node_modules``, etc. — so one unwritable folder must not
abort a whole ``reindex``/``diagram`` run. ``write_generated``/
``remove_generated`` swallow ``OSError`` and record the miss instead;
``drain_skipped`` hands the collected misses to the caller for a single
end-of-run summary line instead of a warning per file.

User-*intent* writes (``quack init``, ``config set``, ``quack new``, ...) are
not generated artifacts and must keep raising — they don't go through here.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_skipped: list[tuple[Path, str]] = []


def write_generated(path: Path, text: str, *, if_changed: bool = False) -> bool:
    """Write a generated artifact. Returns whether it wrote (``False`` both
    when unchanged and when skipped) and never raises ``OSError``."""
    try:
        if if_changed and path.exists() and path.read_text() == text:
            return False
        path.write_text(text)
        return True
    except OSError as e:
        _record(path, e)
        return False


def remove_generated(path: Path) -> bool:
    """Remove a stale generated artifact. Returns whether it removed the file
    and never raises ``OSError``."""
    try:
        path.unlink()
        return True
    except OSError as e:
        _record(path, e)
        return False


def _record(path: Path, e: OSError) -> None:
    reason = e.strerror or str(e)
    _skipped.append((path, reason))
    logger.debug("skip unwritable %s: %s", path, reason)


def drain_skipped() -> list[tuple[Path, str]]:
    """Pop and return every path skipped since the last drain."""
    global _skipped
    out, _skipped = _skipped, []
    return out
