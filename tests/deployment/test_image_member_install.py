"""Every image that installs the framework with pip installs its workspace members locally.

pip cannot see a uv workspace. The framework requires each member listed in
:data:`osprey.deployment.members.WORKSPACE_MEMBERS`, so an image recipe that
installs the framework from a checkout (or a dev-staged wheel) without also
handing pip the matching member would resolve that member from PyPI: a released
snapshot beside checkout framework code, or no match at all for a member that
has never been published.

Two recipe shapes carry this obligation, and both are executed here under a
real ``sh`` with a recording ``pip`` stub rather than grepped, so the tests pin
what pip is actually asked to do:

- the standalone Virtual Accelerator image (``docker/virtual-accelerator/
  Containerfile``) builds every ``./packages/*/`` member from source before the
  framework, under the same ``SETUPTOOLS_SCM_PRETEND_VERSION``;
- the service Dockerfiles whose dev wheel layer resolves the framework wheel
  with an extra pass every staged non-framework wheel in that same pip call.
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import subprocess

import pytest

import osprey
from osprey.deployment.members import WORKSPACE_MEMBERS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTAINERFILE = REPO_ROOT / "docker" / "virtual-accelerator" / "Containerfile"
SERVICES_DIR = pathlib.Path(osprey.__file__).parent / "templates" / "services"

# Service recipes whose wheel layer resolves the dev framework wheel with an
# extra, mapped to that extra.
RESOLVING_SERVICES = {
    "virtual_accelerator": "virtual-accelerator",
    "gchat_bridge": "gchat",
    "teams_bridge": "teams",
}

OSPREY_VERSION = "2026.9.23.dev7"


def _run_bodies(text: str) -> list[str]:
    """Every RUN instruction's shell body, line-continuations joined."""
    joined = re.sub(r"\\\n", " ", text)
    return re.findall(r"^RUN (.+)$", joined, flags=re.MULTILINE)


def _record_pip(tmp_path: pathlib.Path) -> tuple[dict[str, str], pathlib.Path]:
    """An env whose ``pip`` appends ``<SCM version>\\t<argv>`` per call and exits 0."""
    log = tmp_path / "pip-calls.log"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    stub = stub_bin / "pip"
    stub.write_text(
        "#!/bin/sh\n"
        'printf "%s\\t%s\\n" "${SETUPTOOLS_SCM_PRETEND_VERSION:-}" "$*"'
        f" >> {shlex.quote(str(log))}\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    env.pop("SETUPTOOLS_SCM_PRETEND_VERSION", None)
    return env, log


def _calls(log: pathlib.Path) -> list[tuple[str, list[str]]]:
    out = []
    for line in log.read_text().splitlines():
        version, _, argv = line.partition("\t")
        out.append((version, shlex.split(argv)))
    return out


# ── docker/virtual-accelerator/Containerfile ────────────────────────────────


def _containerfile_install_body() -> str:
    bodies = [b for b in _run_bodies(CONTAINERFILE.read_text()) if "[virtual-accelerator]" in b]
    assert len(bodies) == 1, f"expected one framework-install RUN, got {len(bodies)}"
    return bodies[0]


def _run_containerfile_install(
    tmp_path: pathlib.Path, members: tuple[str, ...], *, failing: str | None = None
) -> tuple[subprocess.CompletedProcess, list[tuple[str, list[str]]]]:
    """Execute the install RUN in a staged tree holding *members* under packages/."""
    workdir = tmp_path / "opt-osprey"
    for member in members:
        (workdir / "packages" / member).mkdir(parents=True)
    env, log = _record_pip(tmp_path)
    if failing is not None:
        # A member whose build fails must stop the layer, not be skipped.
        stub = tmp_path / "bin" / "pip"
        stub.write_text(
            stub.read_text().replace(
                "exit 0\n", f'case "$*" in *{failing}*) exit 1;; esac\nexit 0\n'
            )
        )
    env["OSPREY_VERSION"] = OSPREY_VERSION
    result = subprocess.run(
        ["sh", "-c", _containerfile_install_body()],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, (_calls(log) if log.exists() else [])


def test_containerfile_loops_over_every_packages_member():
    """The install layer iterates ``./packages/*/`` rather than naming a member,
    so a member added to the workspace is installed the day it lands."""
    body = _containerfile_install_body()
    assert re.search(r"for \w+ in \./packages/\*/;", body), body
    for member in WORKSPACE_MEMBERS:
        assert f"./packages/{member}" not in body, (
            f"the install layer names {member} explicitly instead of looping over packages/*/"
        )


def test_containerfile_installs_each_member_before_the_framework(tmp_path):
    result, calls = _run_containerfile_install(tmp_path, WORKSPACE_MEMBERS)
    assert result.returncode == 0, result.stderr

    member_calls = {
        member: i
        for i, (_, argv) in enumerate(calls)
        for member in WORKSPACE_MEMBERS
        if any(a.rstrip("/").endswith(f"packages/{member}") for a in argv)
    }
    assert set(member_calls) == set(WORKSPACE_MEMBERS), calls
    framework = [i for i, (_, argv) in enumerate(calls) if ".[virtual-accelerator]" in argv]
    assert len(framework) == 1, calls
    assert max(member_calls.values()) < framework[0], (
        f"a member installs after the framework, so pip resolves it from PyPI first: {calls}"
    )


def test_containerfile_member_builds_carry_the_pretend_version(tmp_path):
    """The staged context has no .git, so every source build — members and
    framework alike — needs the host-resolved version handed to setuptools-scm."""
    _, calls = _run_containerfile_install(tmp_path, WORKSPACE_MEMBERS)
    source_builds = [
        (version, argv)
        for version, argv in calls
        if ".[virtual-accelerator]" in argv or any("packages/" in a for a in argv)
    ]
    assert len(source_builds) == len(WORKSPACE_MEMBERS) + 1, calls
    for version, argv in source_builds:
        assert version == OSPREY_VERSION, f"{argv} built without the pretend version"


def test_containerfile_member_loop_covers_an_unlisted_member(tmp_path):
    """A member present under packages/ but not yet in any list still installs."""
    members = (*WORKSPACE_MEMBERS, "zz-new-member")
    result, calls = _run_containerfile_install(tmp_path, members)
    assert result.returncode == 0, result.stderr
    assert any(
        any(a.rstrip("/").endswith("packages/zz-new-member") for a in argv) for _, argv in calls
    ), calls


def test_containerfile_member_failure_fails_the_layer(tmp_path):
    """A failed member build must not be masked by a later loop iteration."""
    first = sorted(WORKSPACE_MEMBERS)[0]
    result, calls = _run_containerfile_install(tmp_path, WORKSPACE_MEMBERS, failing=first)
    assert result.returncode != 0, f"layer succeeded though {first} failed: {calls}"
    assert not any(".[virtual-accelerator]" in argv for _, argv in calls), (
        "the framework installed after a member build failed"
    )


# ── service Dockerfile wheel layers ─────────────────────────────────────────


def _wheel_body(service: str) -> str:
    text = (SERVICES_DIR / service / "Dockerfile").read_text()
    bodies = [b for b in _run_bodies(text) if "/tmp/ctx/*.whl" in b]
    assert len(bodies) == 1, f"{service}: expected exactly one wheel RUN, got {len(bodies)}"
    return bodies[0]


def _wheel_name(dist: str) -> str:
    return f"{dist.replace('-', '_')}-2026.9.23-py3-none-any.whl"


def _run_wheel_layer(service: str, tmp_path: pathlib.Path, staged: list[str]) -> list[list[str]]:
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    for dist in staged:
        (ctx / _wheel_name(dist)).write_text("")
    env, log = _record_pip(tmp_path)
    result = subprocess.run(
        ["sh", "-c", _wheel_body(service).replace("/tmp/ctx", str(ctx))],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"{service}: wheel RUN failed:\n{result.stderr}"
    return [argv for _, argv in _calls(log)]


def _resolving_call(service: str, calls: list[list[str]]) -> list[str]:
    extra = RESOLVING_SERVICES[service]
    resolving = [c for c in calls if any(a.endswith(f".whl[{extra}]") for a in c)]
    assert len(resolving) == 1, f"{service}: expected one resolving call, got {calls}"
    return resolving[0]


@pytest.mark.parametrize("service", sorted(RESOLVING_SERVICES))
def test_resolving_call_names_every_member_wheel_glob(service):
    """The resolving call takes every staged non-framework wheel, not one by name."""
    body = _wheel_body(service)
    assert "$(ls /tmp/ctx/*.whl | grep -v /osprey_framework-)" in body, body
    assert "osprey_connectors-*.whl" not in body, (
        f"{service}: the resolving call still names the connectors wheel alone"
    )


@pytest.mark.parametrize("service", sorted(RESOLVING_SERVICES))
def test_resolving_call_passes_every_staged_member_wheel(service, tmp_path):
    calls = _run_wheel_layer(service, tmp_path, ["osprey-framework", *WORKSPACE_MEMBERS])
    resolving = _resolving_call(service, calls)
    names = [pathlib.Path(a.split("[")[0]).name for a in resolving if ".whl" in a]
    for member in WORKSPACE_MEMBERS:
        assert _wheel_name(member) in names, (
            f"{service}: {member} wheel missing from the resolving call, so pip "
            f"would fetch it from PyPI: {resolving}"
        )
    assert names.count(_wheel_name("osprey-framework")) == 1, (
        f"{service}: the framework wheel appears other than once (with its extra): {resolving}"
    )


@pytest.mark.parametrize("service", sorted(RESOLVING_SERVICES))
def test_resolving_call_with_only_the_framework_wheel(service, tmp_path):
    """No member wheel staged: the call still runs, carrying just the framework."""
    calls = _run_wheel_layer(service, tmp_path, ["osprey-framework"])
    resolving = _resolving_call(service, calls)
    assert [a for a in resolving if ".whl" in a] == [
        a for a in resolving if a.endswith(f".whl[{RESOLVING_SERVICES[service]}]")
    ], resolving
