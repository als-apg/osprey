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
  framework, each under its own stream's ``SETUPTOOLS_SCM_PRETEND_VERSION``
  (osprey-connectors' own number, everything else the framework's);
- the service Dockerfiles whose dev wheel layer resolves the framework wheel
  with an extra pass every staged non-framework wheel in that same pip call.
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import shutil
import subprocess

import pytest

import osprey
from osprey.deployment.members import WORKSPACE_MEMBERS
from tests.deployment._proxy_idiom import run_instructions

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
OSPREY_CONNECTORS_VERSION = "2026.10.0.post3+g123456789"


def _run_bodies(text: str) -> list[str]:
    """Every RUN instruction's shell body, as the shell receives it."""
    return [instruction.removeprefix("RUN ") for instruction in run_instructions(text)]


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
    tmp_path: pathlib.Path,
    members: tuple[str, ...],
    *,
    failing: str | None = None,
    connectors_version: str | None = OSPREY_CONNECTORS_VERSION,
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
    # A Containerfile ARG with an empty default reaches RUN as an empty variable.
    env["OSPREY_CONNECTORS_VERSION"] = connectors_version or ""
    result = subprocess.run(
        ["sh", "-c", _containerfile_install_body()],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
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


def _pretend_versions(calls: list[tuple[str, list[str]]]) -> dict[str, str]:
    """``{member or "framework": SETUPTOOLS_SCM_PRETEND_VERSION}`` per source build."""
    versions = {}
    for version, argv in calls:
        if ".[virtual-accelerator]" in argv:
            versions["framework"] = version
        for member in WORKSPACE_MEMBERS:
            if any(a.rstrip("/").endswith(f"packages/{member}") for a in argv):
                versions[member] = version
    return versions


def test_containerfile_member_builds_carry_the_pretend_version(tmp_path):
    """The staged context has no .git, so every source build — members and
    framework alike — needs the host-resolved version handed to setuptools-scm.
    osprey-connectors has its own calendar stream, so it gets its own number;
    every other build gets the framework's."""
    _, calls = _run_containerfile_install(tmp_path, WORKSPACE_MEMBERS)
    versions = _pretend_versions(calls)
    assert set(versions) == {*WORKSPACE_MEMBERS, "framework"}, calls
    for build, version in versions.items():
        expected = OSPREY_CONNECTORS_VERSION if build == "osprey-connectors" else OSPREY_VERSION
        assert version == expected, f"{build} built with {version!r}, not {expected!r}"


def test_containerfile_connectors_falls_back_to_the_framework_version(tmp_path):
    """A caller that passes only OSPREY_VERSION (any from before the split)
    still builds: connectors carries the framework's number, as it used to."""
    _, calls = _run_containerfile_install(tmp_path, WORKSPACE_MEMBERS, connectors_version=None)
    assert set(_pretend_versions(calls).values()) == {OSPREY_VERSION}, calls


def test_containerfile_declares_the_connectors_version_arg():
    assert re.search(r"^ARG OSPREY_CONNECTORS_VERSION=$", CONTAINERFILE.read_text(), re.M)


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


# ── docker/virtual-accelerator/Containerfile: the dependency layer ──────────
#
# The third-party closure installs from a requirements file the
# `dependency-set` stage exports from uv.lock, in a layer keyed on that file
# alone, so a source change leaves it cached and the image runs the versions CI
# tests. These pin the parts of that: the stage reads only the lock and the
# manifests with a pinned uv, the layer installs nothing but the export and
# comes before any source, nothing installed afterwards resolves on top of it,
# and the export really is the lock's third-party closure.


def _stages(text: str) -> dict[str, list[tuple[str, str]]]:
    """``{stage name: [(KEYWORD, args), ...]}``; the unnamed final stage is ``""``."""
    stages: dict[str, list[tuple[str, str]]] = {}
    current: list[tuple[str, str]] | None = None
    pending = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        keyword, _, args = (pending + line).partition(" ")
        pending = ""
        if keyword.upper() == "FROM":
            name = re.search(r"\sAS\s+(\S+)\s*$", args, re.I)
            current = stages.setdefault(name.group(1) if name else "", [])
        assert current is not None, f"instruction before the first FROM: {line}"
        current.append((keyword.upper(), args.strip()))
    return stages


def _dependency_set_body() -> str:
    bodies = [b for b in _run_bodies(CONTAINERFILE.read_text()) if "uv export" in b]
    assert len(bodies) == 1, f"expected one dependency-set RUN, got {len(bodies)}"
    return bodies[0]


def test_dependency_set_stage_reads_only_the_lock_and_manifests():
    """Anything else COPYed into that stage would key the dependency layer on it,
    and an unpinned uv could change what the export emits from the same lock."""
    stage = _stages(CONTAINERFILE.read_text())["dependency-set"]
    image = stage[0][1].split()[-3]
    assert re.fullmatch(r"ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+-\S+", image), image
    copies = [args for kw, args in stage if kw == "COPY"]
    assert copies == ["pyproject.toml uv.lock /spec/", "packages/ /spec/packages/"], copies


def test_dependency_set_exports_the_lock_as_committed():
    """--frozen: the lock is read, never re-resolved, so a lock out of step with
    a manifest fails the build. --no-emit-workspace: the workspace's own
    distributions build from source, and pip would fetch them from PyPI here."""
    argv = shlex.split(_dependency_set_body().partition("&&")[0])
    for flag in ("--frozen", "--no-dev", "--all-packages", "--no-emit-workspace"):
        assert flag in argv, f"the export lost {flag}: {argv}"
    assert argv[argv.index("--extra") + 1] == "virtual-accelerator", argv
    assert "--no-hashes" not in argv, "the install runs with --require-hashes"
    assert "test -s /spec/requirements.txt" in _dependency_set_body()


def test_dependency_layer_installs_only_the_export_before_any_source():
    """The closure installs from the export alone -- no hand pin beside it to
    drift from the lock -- before src/ or packages/ land, so a source change
    cannot invalidate it."""
    final = _stages(CONTAINERFILE.read_text())[""]
    install = next(
        i for i, (kw, args) in enumerate(final) if kw == "RUN" and "-r /opt/osprey-deps/" in args
    )
    before = [args for kw, args in final[:install] if kw == "COPY"]
    assert before == [
        "--from=dependency-set /spec/requirements.txt /opt/osprey-deps/requirements.txt"
    ], before
    assert shlex.split(final[install][1]) == [
        "pip",
        "install",
        "--no-cache-dir",
        "--require-hashes",
        "--only-binary",
        "pcaspy",
        "-r",
        "/opt/osprey-deps/requirements.txt",
    ], final[install][1]
    sources = [i for i, (kw, args) in enumerate(final) if kw == "COPY" and "src/" in args]
    assert sources and min(sources) > install


def test_source_builds_resolve_nothing_on_top_of_the_lock(tmp_path):
    """Every member and framework build is --no-deps, so pip cannot upgrade a
    locked dependency past the lock's version, and `pip check` follows them so a
    requirement the export does not satisfy fails the build."""
    result, calls = _run_containerfile_install(tmp_path, WORKSPACE_MEMBERS)
    assert result.returncode == 0, result.stderr
    installs = [argv for _, argv in calls if argv[:1] == ["install"]]
    assert len(installs) == len(WORKSPACE_MEMBERS) + 1, calls
    assert all("--no-deps" in argv for argv in installs), installs
    assert [argv for _, argv in calls][-1] == ["check"], calls


def _staged_lock(tmp_path: pathlib.Path) -> pathlib.Path:
    """What the dependency-set stage COPYs in: the root manifest, the lock and
    every member's manifest (and nothing else, which the export must not need)."""
    root = tmp_path / "spec"
    root.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_bytes((REPO_ROOT / name).read_bytes())
    for manifest in (REPO_ROOT / "packages").glob("*/pyproject.toml"):
        (root / "packages" / manifest.parent.name).mkdir(parents=True)
        (root / "packages" / manifest.parent.name / "pyproject.toml").write_bytes(
            manifest.read_bytes()
        )
    return root


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_dependency_set_is_the_locks_third_party_closure(tmp_path):
    """Run the stage's real command against the repo's lock: every entry is
    pinned to the lock's version and hashed, the PyAT and pcaspy pins the image
    serves from are the lock's, and no workspace distribution appears.

    This runs whichever uv is on PATH, not the image's pinned one; the two agree
    on the lock's revision or the image build fails at --frozen."""
    import tomllib

    root = _staged_lock(tmp_path)
    body = _dependency_set_body().replace("/spec/", "")
    env = dict(os.environ, UV_CACHE_DIR=str(tmp_path / "uv-cache"))
    result = subprocess.run(
        ["sh", "-c", body],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    text = (root / "requirements.txt").read_text()
    pins = {
        (_canonical(name), version)
        for name, version in re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", text, re.M)
    }
    assert pins, text[:500]

    # A name can be locked at more than one version (one per resolution fork),
    # so the comparison is over (name, version) pairs. The workspace's own
    # projects carry no version in the lock (it comes from git at build time).
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked = {
        (_canonical(pkg["name"]), pkg["version"]) for pkg in lock["package"] if "version" in pkg
    }
    assert pins <= locked, f"pinned off the lock: {sorted(pins - locked)}"
    assert not {name for name, _ in pins} & {"osprey-framework", *WORKSPACE_MEMBERS}
    for name in ("accelerator-toolbox", "pcaspy"):
        assert {v for n, v in pins if n == name} == {v for n, v in locked if n == name}, name
    entries = re.split(r"^(?=[A-Za-z0-9])", text, flags=re.M)[1:]
    unhashed = [e.split()[0] for e in entries if "--hash=sha256:" not in e]
    assert not unhashed, f"entries without a hash fail --require-hashes: {unhashed}"


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
    (ctx / ".osprey-layer-anchor").write_text("")
    for dist in staged:
        (ctx / _wheel_name(dist)).write_text("")
    env, log = _record_pip(tmp_path)
    result = subprocess.run(
        ["sh", "-c", _wheel_body(service).replace("/tmp/ctx", str(ctx))],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
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
