"""Tests for the podman-compose pin and probes ci.yml's podman jobs run.

``static-checks`` and ``two-shape-boot-e2e`` both force podman-compose as
podman's compose provider: on a GitHub runner podman would otherwise hand the
work to the docker-compose plugin, and a lane that quietly ran the plugin would
report green having tested nothing new. So the pin, the force and the probe are
each held here on their own. Fake ``pip``, ``podman`` and ``podman-compose`` on
``PATH`` record what ``pin`` calls; the probes run against fake provider infos.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.deployment.runtime_helper import ComposeProvider
from tests._by_path import load_module

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "podman_compose.py"
podman_compose = load_module("podman_compose", _SCRIPT)

_DOCUMENT = (
    "services:\n  event-dispatcher:\n    environment:\n      TOKEN: local-wins\n"
    "  dispatch-worker-1:\n    environment:\n      TOKEN: local-wins\n"
)


_BASH = shutil.which("bash") or "/bin/bash"


def _tool(bin_dir: Path, name: str, log: Path) -> None:
    tool = bin_dir / name
    tool.write_text(f'#!{_BASH}\necho "{name} $*" >> "{log}"\n')
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)


def _pin(tmp_path: Path, *args: str, tools: tuple[str, ...]) -> tuple[int, str, list[str], Path]:
    """Run ``pin`` with only *tools* on PATH: a runner's own podman must not answer."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    log.touch()
    for name in tools:
        _tool(bin_dir, name, log)
    github_env = tmp_path / "github_env"
    github_env.touch()
    env = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path / "home"),
        "GITHUB_ENV": str(github_env),
    }
    done = subprocess.run(
        [sys.executable, str(_SCRIPT), "pin", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, github_env.read_text(), log.read_text().splitlines(), bin_dir


def test_the_pin_reads_its_expected_version_from_the_install_set() -> None:
    """The asserted version is derived from the install line, so the two cannot drift."""
    assert podman_compose.PINNED[0] == "podman-compose==1.0.6"
    assert podman_compose.PINNED_VERSION == (1, 0, 6)


def test_the_install_is_exactly_the_pinned_set() -> None:
    argv = podman_compose.install_argv("pip")
    assert argv[:4] == ["pip", "install", "--user", "--no-deps"]
    assert all("==" in spec for spec in argv[4:]), argv
    assert [spec.split("==")[0] for spec in argv[4:]] == [
        "podman-compose",
        "python-dotenv",
        "PyYAML",
    ]


def test_pin_installs_forces_and_exports_the_provider(tmp_path: Path) -> None:
    code, exported, calls, bin_dir = _pin(tmp_path, tools=("pip", "podman", "podman-compose"))
    assert code == 0
    assert f"pip {' '.join(podman_compose.install_argv('pip')[1:])}" in calls
    provider = str(bin_dir / "podman-compose")
    conf = (tmp_path / "home" / ".config" / "containers" / "containers.conf").read_text()
    assert conf == f'[engine]\ncompose_providers = ["{provider}"]\n'
    # The variable as well as the file: a user-level containers.conf.d drop-in outranks the file.
    assert exported == f"PODMAN_COMPOSE_PROVIDER={provider}\n"
    assert "podman ps" in calls  # rootless storage init, paid before any probe's timeout


def test_pin_fails_when_the_install_leaves_no_provider_on_path(tmp_path: Path) -> None:
    code, exported, _, _ = _pin(tmp_path, tools=("pip", "podman"))
    assert code == 1
    assert exported == ""


def test_pin_without_podman_fails_the_lane(tmp_path: Path) -> None:
    """The lane's pin is unguarded: that is what keeps the Tier 0 skip below from
    hiding a runner that genuinely lost podman."""
    code, exported, calls, _ = _pin(tmp_path, tools=("pip",))
    assert code == 1
    assert (exported, calls) == ("", [])


def test_optional_pin_without_podman_skips_the_parse(tmp_path: Path) -> None:
    code, exported, calls, _ = _pin(tmp_path, "--optional", tools=("pip",))
    assert code == 0
    assert exported == "OSPREY_PARSE_SKIP=1\n"
    assert calls == [], "nothing is installed on a runner without podman"


def test_parse_check_honours_the_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(podman_compose.SKIP_VARIABLE, "1")
    monkeypatch.setattr(podman_compose.subprocess, "run", _unexpected)
    assert podman_compose.parse_check(tmp_path / "parse-check") == 0


def _unexpected(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(f"ran {args}")


@pytest.mark.parametrize(
    ("provider", "version", "problem"),
    [
        (ComposeProvider.PODMAN_COMPOSE, (1, 0, 6), None),
        (ComposeProvider.DOCKER_V2, (1, 0, 6), "must be podman-compose"),
        # The parsed tuple, never the banner text.
        (ComposeProvider.PODMAN_COMPOSE, (1, 1, 0), "must be podman-compose==1.0.6"),
    ],
)
def test_provider_problems(
    provider: ComposeProvider, version: tuple[int, ...], problem: str | None
) -> None:
    info = SimpleNamespace(
        provider=provider, version=version, version_text=".".join(map(str, version))
    )
    problems = podman_compose.provider_problems(info)
    if problem is None:
        assert problems == []
    else:
        assert len(problems) == 1 and problem in problems[0], problems


@pytest.mark.parametrize(
    ("document", "problem"),
    [
        (_DOCUMENT, None),
        (_DOCUMENT.replace("dispatch-worker-1:", "worker:"), "dispatch-worker-1 missing"),
        (_DOCUMENT.replace("local-wins", "${TOKEN}"), "did not interpolate"),
        # Both chain members carry the key: the shared value showing means precedence broke.
        (_DOCUMENT + "      OTHER: shared-loses\n", "outranked .env"),
    ],
)
def test_merged_document_problems(document: str, problem: str | None) -> None:
    problems = podman_compose.merged_document_problems(document)
    if problem is None:
        assert problems == []
    else:
        assert len(problems) == 1 and problem in problems[0], problems


def _fake_osprey(monkeypatch: pytest.MonkeyPatch, *, returncode: int = 0) -> list[list[str]]:
    """Stand OSPREY's compose seams and the provider probe in for a built project."""
    from osprey.deployment import compose_generator, container_lifecycle
    from osprey_connectors import config as connectors_config

    info = SimpleNamespace(provider=ComposeProvider.PODMAN_COMPOSE)
    monkeypatch.setattr(connectors_config, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(compose_generator, "resolve_repo_root", lambda config: ".")
    monkeypatch.setattr(compose_generator, "compose_provider_env", lambda provider, root: {})
    monkeypatch.setattr(
        compose_generator,
        "compose_base_cmd",
        lambda runtime, files, root, env_args, provider: [*runtime, "-f", "merged.yml"],
    )
    monkeypatch.setattr(container_lifecycle, "as_built_compose_files", lambda config, root: [])
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda root, provider: [])
    monkeypatch.setattr(
        podman_compose, "_probe_provider", lambda config: (["podman-compose"], info)
    )
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, _DOCUMENT, "")

    monkeypatch.setattr(podman_compose.subprocess, "run", run)
    return calls


def test_the_parse_runs_osprey_own_argv_through_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """What separates this from a YAML lint: the document podman-compose parses is
    the one OSPREY's own argv writes, and ``config`` (not ``ps``) is what parses and
    interpolates it."""
    calls = _fake_osprey(monkeypatch)
    assert podman_compose.check_merged_topology() == []
    assert calls == [["podman-compose", "-f", "merged.yml", "config"]]


def test_the_parse_fails_when_podman_compose_rejects_the_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_osprey(monkeypatch, returncode=1)
    assert podman_compose.check_merged_topology() == [
        "podman-compose could not parse the merged topology"
    ]


def test_the_parse_fails_when_the_argv_carries_project_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """podman-compose 1.0.6 does not parse ``--project-directory``; an argv that
    grew it would red every podman deploy, so the parse refuses it before running."""
    from osprey.deployment import compose_generator

    calls = _fake_osprey(monkeypatch)
    monkeypatch.setattr(
        compose_generator,
        "compose_base_cmd",
        lambda runtime, files, root, env_args, provider: [*runtime, "--project-directory", "."],
    )
    assert podman_compose.check_merged_topology() == [
        "podman-compose does not parse --project-directory"
    ]
    assert calls == []
