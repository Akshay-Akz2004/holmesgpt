"""
When the container as a whole runs out of memory, the kernel must kill the
tool subprocess, not Holmes.

Runs a stand-in for Holmes inside a memory cgroup (the mechanism behind a pod's
memory limit). It holds more memory than the tool, then runs a tool that keeps
allocating until the cgroup limit is hit. Needs a writable memory cgroup
(root, or a delegated cgroup v2 subtree); skipped otherwise.
"""

import json
import os
import shlex
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Iterator, Optional

import pytest
from tenacity import Retrying, retry_if_exception_type, stop_after_delay, wait_fixed

CGROUP_LIMIT_MB = 600
HOLMES_BALLAST_MB = 300

HOLMES_STAND_IN = textwrap.dedent(
    f"""
    import json, sys

    ballast = bytearray({HOLMES_BALLAST_MB} * 1024 * 1024)
    for i in range(0, len(ballast), 4096):
        ballast[i] = 1

    import holmes.plugins.toolsets.bash.common.bash as bash
    if sys.argv[2] == "without-oom-score":
        prefix = bash.get_ulimit_prefix
        bash.get_ulimit_prefix = lambda: prefix().split("2>/dev/null; ", 1)[1]

    result = bash.execute_bash_command(f"{{sys.executable}} {{sys.argv[1]}}", timeout=120)
    print(json.dumps({{"return_code": result.return_code, "stdout": result.stdout[:200]}}))
    """
)

GROWING_TOOL = textwrap.dedent(
    """
    import time

    chunks = []
    while True:
        chunks.append(bytearray(20 * 1024 * 1024))
        time.sleep(0.01)
    """
)


def _writable_memory_cgroup_root() -> Optional[Path]:
    v2 = Path("/sys/fs/cgroup")
    if (v2 / "cgroup.controllers").exists():
        if "memory" in (v2 / "cgroup.subtree_control").read_text() and os.access(
            v2, os.W_OK
        ):
            return v2
        return None
    v1 = Path("/sys/fs/cgroup/memory")
    if (v1 / "memory.limit_in_bytes").exists() and os.access(v1, os.W_OK):
        return v1
    return None


CGROUP_ROOT = _writable_memory_cgroup_root()

pytestmark = pytest.mark.skipif(
    CGROUP_ROOT is None, reason="needs a writable memory cgroup"
)


@pytest.fixture
def memory_cgroup() -> Iterator[Path]:
    assert CGROUP_ROOT is not None
    cgroup = CGROUP_ROOT / f"holmes-oom-test-{uuid.uuid4().hex[:8]}"
    cgroup.mkdir()
    limit = str(CGROUP_LIMIT_MB * 1024 * 1024)
    if (cgroup / "memory.max").exists():
        (cgroup / "memory.max").write_text(limit)
        if (cgroup / "memory.swap.max").exists():
            (cgroup / "memory.swap.max").write_text("0")
    else:
        (cgroup / "memory.limit_in_bytes").write_text(limit)
        if (cgroup / "memory.memsw.limit_in_bytes").exists():
            (cgroup / "memory.memsw.limit_in_bytes").write_text(limit)
    yield cgroup
    for pid in (cgroup / "cgroup.procs").read_text().split():
        try:
            os.kill(int(pid), 9)
        except ProcessLookupError:
            pass
    for attempt in Retrying(
        retry=retry_if_exception_type(OSError),
        stop=stop_after_delay(10),
        wait=wait_fixed(0.2),
        reraise=True,
    ):
        with attempt:
            cgroup.rmdir()


def _run_holmes_in_cgroup(cgroup: Path, mode: str, tmp_path: Path):
    holmes_script = tmp_path / "holmes_stand_in.py"
    holmes_script.write_text(HOLMES_STAND_IN)
    tool_script = tmp_path / "growing_tool.py"
    tool_script.write_text(GROWING_TOOL)
    command = (
        f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))} && exec "
        + shlex.join([sys.executable, str(holmes_script), str(tool_script), mode])
    )
    env = {**os.environ, "TOOL_MEMORY_LIMIT_MB": str(4 * CGROUP_LIMIT_MB)}
    return subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, timeout=180, env=env
    )


def test_container_oom_kills_the_tool_not_holmes(memory_cgroup, tmp_path):
    proc = _run_holmes_in_cgroup(memory_cgroup, "with-oom-score", tmp_path)

    assert proc.returncode == 0, f"Holmes was killed: {proc.returncode} {proc.stderr}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["return_code"] in (137, -9)
    assert result["stdout"].startswith("[OOM]")


def test_control_without_oom_score_holmes_is_killed(memory_cgroup, tmp_path):
    """Shows the setup is sensitive: without the score, Holmes is the bigger target."""
    proc = _run_holmes_in_cgroup(memory_cgroup, "without-oom-score", tmp_path)

    assert proc.returncode == -9, proc.stdout + proc.stderr
