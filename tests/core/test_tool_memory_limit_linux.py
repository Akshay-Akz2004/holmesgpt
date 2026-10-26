"""Runs real commands through the tool memory-protection prefix (Linux only)."""

import os
import sys

import pytest

from holmes.plugins.toolsets.bash.common.bash import execute_bash_command

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="limits are only enforced on Linux"
)

LIMIT_MB = 200

# Reserves address space without touching it, the way the Go runtime does at startup.
RESERVE_4GB = (
    "import ctypes, mmap; "
    "libc = ctypes.CDLL(None); libc.mmap.restype = ctypes.c_void_p; "
    "libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, "
    "ctypes.c_int, ctypes.c_int, ctypes.c_long]; "
    "p = libc.mmap(None, 4 << 30, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0); "
    "print('reserved' if p not in (None, ctypes.c_void_p(-1).value) else 'failed')"
)


@pytest.fixture(autouse=True)
def small_limit(monkeypatch):
    monkeypatch.setattr("holmes.utils.memory_limit.TOOL_MEMORY_LIMIT_MB", LIMIT_MB)


def _python(code: str) -> str:
    return f'{sys.executable} -c "{code}"'


def test_reserved_address_space_is_not_capped():
    result = execute_bash_command(_python(RESERVE_4GB), timeout=30)
    assert result.return_code == 0, result.stdout
    assert result.stdout == "reserved"


def test_reservation_probe_detects_an_address_space_cap():
    result = execute_bash_command(
        f"ulimit -v {LIMIT_MB * 1024}; " + _python(RESERVE_4GB), timeout=30
    )
    assert result.stdout == "failed"


def test_allocation_over_cap_fails_with_oom_hint():
    result = execute_bash_command(
        _python(f"x = bytearray({2 * LIMIT_MB} * 1024 * 1024)"), timeout=30
    )
    assert result.return_code != 0
    assert result.stdout.startswith("[OOM]")
    assert "MemoryError" in result.stdout


def test_allocation_under_cap_succeeds():
    result = execute_bash_command(
        _python(f"x = bytearray({LIMIT_MB // 2} * 1024 * 1024); print('ok')"),
        timeout=30,
    )
    assert result.return_code == 0, result.stdout
    assert result.stdout == "ok"


def test_tool_and_its_children_are_first_oom_victims():
    result = execute_bash_command(
        "cat /proc/self/oom_score_adj; bash -c 'cat /proc/self/oom_score_adj'",
        timeout=30,
    )
    assert result.stdout.split() == ["1000", "1000"]


def test_holmes_oom_score_is_unchanged():
    with open(f"/proc/{os.getpid()}/oom_score_adj") as f:
        before = f.read()
    execute_bash_command("true", timeout=30)
    with open(f"/proc/{os.getpid()}/oom_score_adj") as f:
        assert f.read() == before
