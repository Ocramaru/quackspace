from __future__ import annotations

from quack.subprocess_utils import failure_message


def test_failure_message_prefers_stderr_then_stdout_then_empty_marker():
    assert "stderr detail" in failure_message("AI", ["cmd"], 1, "stdout detail", "stderr detail")
    assert "stdout detail" in failure_message("AI", ["cmd"], 1, "stdout detail", "")
    assert "no output on stderr or stdout" in failure_message("AI", ["cmd"], 1, "", "")


def test_failure_message_truncates_long_detail():
    msg = failure_message("AI", ["cmd"], 1, "x" * 20, "", max_detail=10)

    assert msg.endswith("xxxxxxx...")
