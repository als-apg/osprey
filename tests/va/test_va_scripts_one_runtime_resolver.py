"""The VA developer scripts take their container runtime from the runtime helper.

Which runtime OSPREY drives is decided once, by
``osprey.deployment.runtime_helper``, from ``CONTAINER_RUNTIME`` and the host.
Each script asks it exactly once and names no runtime anywhere else, so no
script can prefer, probe or refuse a runtime the helper would not.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "scripts/va/build_and_boot_check.sh",
    "scripts/va/live_ca/run_live_ca.sh",
    "scripts/va/probe_pcaspy/run_probe.sh",
]

_HELPER_CALL = re.compile(r"-m osprey\.deployment\.runtime_helper\b")

#: A runtime named as a word. Lowercase only, and not part of a longer name or
#: path, so ``DOCKER_BUILDKIT`` and ``docker/virtual-accelerator`` do not count.
_RUNTIME_WORD = re.compile(r"(?<![\w/.-])(docker|podman)(?![\w/.-])")

#: The retired runtime variable, spelled split so this file does not name it.
_RETIRED_VARIABLE = "OSPREY_VA_" + "RUNTIME"


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_script_asks_the_helper_once_for_its_runtime(script: str) -> None:
    lines = (REPO_ROOT / script).read_text(encoding="utf-8").splitlines()
    calls = [line for line in lines if _HELPER_CALL.search(line)]

    assert len(calls) == 1, calls
    assert calls[0].lstrip().startswith('RUNTIME="$('), calls[0]


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_script_names_no_runtime_outside_the_helper_call(script: str) -> None:
    lines = (REPO_ROOT / script).read_text(encoding="utf-8").splitlines()
    named = [
        f"{number}: {line}"
        for number, line in enumerate(lines, start=1)
        if _RUNTIME_WORD.search(line) and not _HELPER_CALL.search(line)
    ]

    assert named == []


@pytest.mark.skipif(shutil.which("git") is None, reason="reads the tracked tree with git grep")
def test_no_tracked_file_names_the_retired_runtime_variable() -> None:
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-F",
            _RETIRED_VARIABLE,
            "--",
            ".",
            ":(exclude).claude",
            ":(exclude)CHANGELOG.md",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode in (0, 1), result.stderr
    assert result.stdout.splitlines() == []
