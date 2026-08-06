"""The project-venv install must tolerate a large, slow dependency download.

``osprey build`` installs osprey and its whole transitive dependency tree into
the freshly created project venv. That tree is well over a gigabyte of wheels
(claude-agent-sdk, playwright, llvmlite, pyarrow, scipy, litellm ... measured at
~1.4 GB installed from ~2.2 GB of downloads), so on a cold cache or a slow link
the install is *legitimately* a multi-minute operation.

The cap on that subprocess therefore has to bound a **hung** installer, not a
slow one. A cap tight enough to fire on an ordinary cold-cache download turns a
working build into an abort — and because the download time is a property of the
network rather than of the project, it fires intermittently and looks like
flakiness rather than a misconfigured limit.

When the cap does fire it must also say so in terms the operator can act on:
:func:`_create_project_venv` raising a bare ``subprocess.TimeoutExpired`` lands
in ``build_cmd``'s catch-all and prints "Unexpected error", which names neither
the step that timed out nor anything to do about it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.cli.build_environment import _create_project_venv
from osprey.cli.build_profile import BuildProfile, EnvironmentConfig
from osprey.errors import BuildProfileError

# A full cold-cache install of osprey's dependency tree is a >1 GB download.
# Fifteen minutes is the floor at which the cap is bounding a hung installer
# rather than a slow network; the shipped value may be more generous still.
MIN_INSTALL_TIMEOUT_S = 900


def _profile() -> BuildProfile:
    """A minimal profile that installs osprey from a pinned release spec."""
    return BuildProfile(
        name="test",
        dependencies=[],
        osprey_install="osprey-framework==1.2.3",
        environment=EnvironmentConfig(python=None, packages=[]),
    )


def _is_install(cmd: list[str]) -> bool:
    """True for the dependency-install call, not the venv-creation call."""
    return "install" in cmd


@pytest.fixture
def with_uv(monkeypatch) -> str:
    """Pin the uv branch so the install command shape is deterministic."""
    uv = "/usr/bin/uv"
    monkeypatch.setattr("osprey.cli.build_environment.shutil.which", lambda _: uv)
    monkeypatch.delenv("UV", raising=False)
    return uv


def test_install_timeout_allows_a_full_dependency_download(with_uv, monkeypatch, tmp_path):
    """The install subprocess gets a cap sized for a >1 GB download."""
    seen: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    _create_project_venv(tmp_path, _profile())

    install_calls = [(cmd, kw) for cmd, kw in seen if _is_install(cmd)]
    assert install_calls, f"no install call recorded; saw {[c for c, _ in seen]}"
    _, kwargs = install_calls[0]
    assert kwargs["timeout"] >= MIN_INSTALL_TIMEOUT_S, (
        f"the project-venv install is capped at {kwargs['timeout']}s, which a "
        "cold-cache download of osprey's dependency tree can exceed on a slow "
        "link — the cap must bound a hung installer, not a slow one"
    )


def test_install_timeout_is_reported_as_an_actionable_build_error(with_uv, monkeypatch, tmp_path):
    """A timed-out install raises BuildProfileError naming the step and a remedy."""

    def fake_run(cmd, **kwargs):
        if _is_install(cmd):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    with pytest.raises(BuildProfileError) as excinfo:
        _create_project_venv(tmp_path, _profile())

    message = str(excinfo.value)
    assert "project venv" in message.lower(), message
    # The operator needs to know it was the dependency install that ran long,
    # not merely that "something" timed out.
    assert "install" in message.lower(), message


def test_venv_creation_failure_still_reports_its_own_error(with_uv, monkeypatch, tmp_path):
    """Guard: the install-timeout handling does not swallow venv-creation errors."""

    def fake_run(cmd, **kwargs):
        if _is_install(cmd):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    with pytest.raises(BuildProfileError, match="Failed to create project venv"):
        _create_project_venv(Path(tmp_path), _profile())
