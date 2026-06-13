from __future__ import annotations

import subprocess
import sys

import quack


def main() -> int:
    assert quack.__version__
    result = subprocess.run(
        [sys.executable, "-m", "quack.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "quack" in result.stdout.lower()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
