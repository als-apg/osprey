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

Venv creation is bounded the same way, and its timeout is reported the same
way: as a build error naming the step and the check to run, not as an
unexpected error.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.cli import build_environment
from osprey.cli.build_environment import _create_project_venv
from osprey.cli.build_profile import BuildProfile, EnvironmentConfig
from osprey.errors import BuildProfileError

# A full cold-cache install of osprey's dependency tree is a >1 GB download.
# Fifteen minutes is the floor at which the cap is bounding a hung installer
# rather than a slow network; the shipped value may be more generous still.
MIN_INSTALL_TIMEOUT_S = 900

# A tighter cap on venv creation fires on a merely loaded host.
MIN_CREATE_TIMEOUT_S = 300


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


@pytest.fixture
def without_uv(monkeypatch) -> None:
    """Pin the stdlib branch: no uv on PATH."""
    monkeypatch.setattr("osprey.cli.build_environment.shutil.which", lambda _: None)
    monkeypatch.delenv("UV", raising=False)


@pytest.mark.usefixtures("with_uv")
def test_install_timeout_allows_a_full_dependency_download(monkeypatch, tmp_path):
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


@pytest.mark.usefixtures("with_uv")
def test_install_timeout_is_reported_as_an_actionable_build_error(monkeypatch, tmp_path):
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


@pytest.mark.usefixtures("with_uv")
def test_venv_creation_failure_still_reports_its_own_error(monkeypatch, tmp_path):
    """Guard: the install-timeout handling does not swallow venv-creation errors."""

    def fake_run(cmd, **kwargs):
        if _is_install(cmd):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    with pytest.raises(BuildProfileError, match="Failed to create project venv"):
        _create_project_venv(Path(tmp_path), _profile())


@pytest.mark.parametrize("branch", ["with_uv", "without_uv"])
def test_a_venv_creation_timeout_is_reported_as_a_build_error_naming_the_step(
    branch, request, monkeypatch, tmp_path
):
    """A timed-out venv creation raises BuildProfileError naming the step and a check."""
    request.getfixturevalue(branch)
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        if _is_install(cmd):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    with pytest.raises(BuildProfileError) as excinfo:
        _create_project_venv(tmp_path, _profile())

    message = str(excinfo.value)
    assert "creating the project venv" in message, message
    assert "5 minutes" in message, message
    assert "-c pass" in message, message
    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
    assert not [cmd for cmd in seen if _is_install(cmd)], (
        f"the install ran after venv creation timed out: {seen}"
    )


@pytest.mark.usefixtures("with_uv")
def test_the_venv_creation_bound_leaves_a_loaded_host_room(monkeypatch, tmp_path):
    """The venv-creation subprocess is capped by the module bound, never below its floor."""
    seen: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    _create_project_venv(tmp_path, _profile())

    create_calls = [(cmd, kw) for cmd, kw in seen if not _is_install(cmd)]
    assert create_calls, f"no venv-creation call recorded; saw {[c for c, _ in seen]}"
    _, kwargs = create_calls[0]
    assert kwargs["timeout"] == build_environment._VENV_CREATE_TIMEOUT_S
    assert kwargs["timeout"] >= MIN_CREATE_TIMEOUT_S


@pytest.mark.usefixtures("with_uv")
def test_a_failed_install_mentioning_litellm_fails_closed(monkeypatch, tmp_path):
    """A failed install is a build error — never a link to the build env's packages.

    An install that fails is a failed install whatever its output says. Matching
    a package name in that output and recovering by installing ``--no-deps`` and
    writing a ``.pth`` into the project venv would point the project at the
    *build host's* site-packages, so the built project would run against
    whatever happens to be installed beside osprey rather than against its own
    recorded dependency set.
    """
    # The first install is the one whose result decides the outcome; any later
    # one would be a fallback path, and there must not be one.
    installs: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if _is_install(cmd):
            installs.append(cmd)
            if len(installs) == 1:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="error: distribution litellm==1.81.11 is not available",
                )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    with pytest.raises(BuildProfileError) as excinfo:
        _create_project_venv(tmp_path, _profile())

    # The resolver's own output is what tells the operator which package failed.
    assert "litellm" in str(excinfo.value), str(excinfo.value)
    assert len(installs) == 1, f"a second install ran after the first failed: {installs}"
    assert not list(tmp_path.rglob("_osprey_build_env.pth")), (
        "a failed install left a .pth pointing the project venv at the build "
        "environment's site-packages"
    )


@pytest.mark.usefixtures("with_uv")
def test_install_timeout_names_the_mirror_remedy(monkeypatch, tmp_path):
    """A timed-out install ends with the shared re-run and mirror remedy."""

    def fake_run(cmd, **kwargs):
        if _is_install(cmd):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)

    with pytest.raises(BuildProfileError) as excinfo:
        _create_project_venv(tmp_path, _profile())

    message = str(excinfo.value)
    assert message.endswith(build_environment._INSTALL_NETWORK_REMEDY), message
    assert "UV_INDEX_URL" in message, message
    assert "PIP_INDEX_URL" in message, message
