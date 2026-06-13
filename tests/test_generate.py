from __future__ import annotations

import shlex
import sys

import pytest

from quack.config import AIConfig, Config, EmbedConfig
from quack.generate import run_ai


def _config(command: str, timeout: int = 5) -> Config:
    return Config(
        ai=AIConfig(command=command, timeout=timeout),
        embed=EmbedConfig(),
        path=None,
    )


def test_run_ai_nonzero_with_empty_output_has_actionable_error():
    command = f'{sys.executable} -c "import sys; sys.exit(7)"'

    with pytest.raises(RuntimeError) as exc:
        run_ai(_config(command), "prompt")

    message = str(exc.value)
    assert "AI command failed (7)" in message
    assert "no output on stderr or stdout" in message
    assert sys.executable in message


def test_run_ai_nonzero_uses_stdout_when_stderr_empty(tmp_path):
    script = tmp_path / "provider.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write('bad provider')\n"
        "sys.exit(2)\n"
    )
    command = f"{sys.executable} {shlex.quote(str(script))}"

    with pytest.raises(RuntimeError) as exc:
        run_ai(_config(command), "prompt")

    message = str(exc.value)
    assert "AI command failed (2)" in message
    assert "bad provider" in message
