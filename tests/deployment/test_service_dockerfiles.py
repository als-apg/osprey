"""Layer-split invariants for the four service Dockerfiles.

`osprey deploy up` builds each managed service (event_dispatcher,
virtual_accelerator, bluesky, bluesky_panels) from a template Dockerfile shipped
under ``osprey/templates/services/<name>/``. This module pins the shared layer
contract those recipes follow so a fast per-project dev rebuild only re-runs the
cheap wheel layer, never the expensive framework/deps install:

- a **deps layer** (one RUN) that primes the pinned framework release + deps,
  with a dev-gated fallback for an unreleased pin, then installs an optional
  dev-staged ``osprey-local-requirements.txt`` manifest (the local dependency
  delta, COPYed via the guaranteed-sibling idiom) before purging the toolchain,
  and
- a separate **wheel layer** that optionally overlays a locally-built wheel via
  the ``COPY .dockerignore *.wh[l]`` idiom, force-reinstalls it ``--no-deps`` and
  runs ``pip check``,

followed by a metadata-only ``ARG OSPREY_PROJECT_NAME`` / ``LABEL
com.osprey.project`` pair kept last so a per-project value never invalidates the
shared deps cache. Each RUN body is parsed under ``sh -n`` so a shell syntax
error in the conditional install fails here rather than at ``docker build``.
"""

import os
import pathlib
import re
import subprocess

import pytest

import osprey

SERVICES_DIR = pathlib.Path(osprey.__file__).parent / "templates" / "services"

# The four managed services and their pinned primer spec (what the deps layer
# installs from PyPI). Only the virtual accelerator carries an extra.
PRIMER_SPEC = {
    "event_dispatcher": "osprey-framework==$OSPREY_VERSION",
    "virtual_accelerator": "osprey-framework[virtual-accelerator]==$OSPREY_VERSION",
    "bluesky": "osprey-framework==$OSPREY_VERSION",
    "bluesky_panels": "osprey-framework==$OSPREY_VERSION",
}
SERVICES = sorted(PRIMER_SPEC)

# The dev-staged local-dependency manifest: COPYed next to .dockerignore (the
# guaranteed sibling keeps the glob matching when absent) and installed inside
# the deps RUN while the C toolchain is still available.
MANIFEST_COPY = "COPY .dockerignore osprey-local-requirements.tx[t] /tmp/deps-ctx/"
MANIFEST_INSTALL = "pip install --no-cache-dir -r /tmp/deps-ctx/osprey-local-requirements.txt"


def _dockerfile(service: str) -> str:
    return (SERVICES_DIR / service / "Dockerfile").read_text()


def _run_bodies(text: str) -> list[str]:
    """Every RUN instruction's shell body, line-continuations joined."""
    joined = re.sub(r"\\\n", " ", text)
    return re.findall(r"^RUN (.+)$", joined, flags=re.MULTILINE)


def _instructions(text: str) -> list[tuple[str, str]]:
    """(keyword, full joined line) for each Dockerfile instruction, in order."""
    joined = re.sub(r"\\\n", " ", text)
    out: list[tuple[str, str]] = []
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append((line.split(maxsplit=1)[0], line))
    return out


def _deps_body(service: str) -> str:
    """The deps-layer RUN body (the one carrying the pinned primer spec)."""
    spec = PRIMER_SPEC[service]
    bodies = [b for b in _run_bodies(_dockerfile(service)) if spec in b]
    assert len(bodies) == 1, f"{service}: expected exactly one deps RUN, got {len(bodies)}"
    return bodies[0]


def _wheel_body(service: str) -> str:
    """The wheel-layer RUN body (the one operating on /tmp/ctx/*.whl)."""
    bodies = [b for b in _run_bodies(_dockerfile(service)) if "/tmp/ctx/*.whl" in b]
    assert len(bodies) == 1, f"{service}: expected exactly one wheel RUN, got {len(bodies)}"
    return bodies[0]


@pytest.mark.parametrize("service", SERVICES)
class TestLayerSplit:
    def test_deps_and_wheel_are_separate_runs(self, service):
        deps, wheel = _deps_body(service), _wheel_body(service)
        assert deps != wheel, f"{service}: deps and wheel share one RUN — not layer-split"
        # The deps layer must not touch the staged wheel, and the wheel layer
        # must not re-prime the framework from PyPI.
        assert "/tmp/ctx" not in deps, f"{service}: deps layer references the wheel context"
        assert PRIMER_SPEC[service] not in wheel, f"{service}: wheel layer re-primes from PyPI"

    def test_wheel_copy_idiom(self, service):
        text = _dockerfile(service)
        assert "COPY .dockerignore *.wh[l] /tmp/ctx/" in text, (
            f"{service}: missing the guaranteed-sibling wheel COPY idiom"
        )

    def test_wheel_install_sequence(self, service):
        wheel = _wheel_body(service)
        if service == "virtual_accelerator":
            # VA's first (deps-resolving) install carries the extra so a dev
            # wheel that changes the extra's deps picks them up.
            assert 'pip install --no-cache-dir "${whl}[virtual-accelerator]"' in wheel
        else:
            assert "pip install --no-cache-dir /tmp/ctx/*.whl" in wheel
        assert "pip install --no-cache-dir --no-deps --force-reinstall /tmp/ctx/*.whl" in wheel
        assert "pip check" in wheel
        # Always clean up the staged context regardless of whether a wheel was
        # present (so a no-op wheel layer still leaves no /tmp/ctx behind).
        assert "rm -rf /tmp/ctx" in wheel

    def test_pinned_primer_spec_present(self, service):
        deps = _deps_body(service)
        spec = PRIMER_SPEC[service]
        assert f'pip install --no-cache-dir "{spec}"' in deps, (
            f"{service}: pinned primer spec {spec!r} missing from deps layer"
        )

    def test_dev_gated_fallback(self, service):
        deps = _deps_body(service)
        # Fallback exists, is gated on OSPREY_DEV=1, and a non-dev pin miss is
        # fatal (explicit exit 1 branch).
        assert '[ "$OSPREY_DEV" = "1" ]' in deps, f"{service}: fallback not gated on OSPREY_DEV=1"
        assert "exit 1" in deps, f"{service}: no fail-loud branch for a non-dev pin miss"
        assert "ARG OSPREY_DEV" in _dockerfile(service), f"{service}: OSPREY_DEV arg not declared"

    def test_trailing_project_label_is_last(self, service):
        instrs = _instructions(_dockerfile(service))
        (kw_a, line_a), (kw_b, line_b) = instrs[-2], instrs[-1]
        assert kw_a == "ARG" and "OSPREY_PROJECT_NAME" in line_a, (
            f"{service}: second-to-last instruction is not `ARG OSPREY_PROJECT_NAME`: {line_a!r}"
        )
        assert kw_b == "LABEL" and "com.osprey.project" in line_b, (
            f"{service}: last instruction is not `LABEL com.osprey.project`: {line_b!r}"
        )
        # The arg name must be OSPREY_PROJECT_NAME, not the OSPREY_PROJECT that
        # names the project *directory* elsewhere.
        assert "OSPREY_PROJECT=" not in line_a

    def test_every_run_parses_under_sh(self, service):
        for body in _run_bodies(_dockerfile(service)):
            result = subprocess.run(["sh", "-n", "-c", body], capture_output=True, text=True)
            assert result.returncode == 0, (
                f"{service}: invalid shell syntax in a RUN body:\n{body}\n{result.stderr}"
            )

    def test_deps_layer_guards_missing_version_arg(self, service):
        """A manual `docker build` without --build-arg OSPREY_VERSION must fail
        with an explicit error, not pip choking on `osprey-framework==`."""
        deps = _deps_body(service)
        guard = (
            '[ -n "$OSPREY_VERSION" ] || '
            '{ echo "ERROR: OSPREY_VERSION build-arg is required" >&2; exit 1; }'
        )
        assert guard in deps, f"{service}: deps layer missing the OSPREY_VERSION guard"

    def test_wheel_layer_propagates_install_failure(self, service, tmp_path):
        """Empirical probe: a failing `pip` inside the wheel layer must fail the
        whole RUN body — the trailing `rm -rf /tmp/ctx` cleanup must not swallow
        the exit status (`fi && rm`, never `fi; rm`)."""
        result, ctx = _probe_wheel_body(_wheel_body(service), tmp_path, with_wheel=True)
        assert result.returncode != 0, (
            f"{service}: wheel-layer RUN exited 0 despite pip failing — a broken "
            f"dev-wheel install would build a stale image silently:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert ctx.exists(), f"{service}: cleanup ran despite the install failing"

    def test_wheel_layer_no_wheel_is_noop_success(self, service, tmp_path):
        """Empirical probe: with no wheel staged the RUN is a successful no-op
        (the untaken `if` exits 0) and still cleans up the staged context."""
        result, ctx = _probe_wheel_body(_wheel_body(service), tmp_path, with_wheel=False)
        assert result.returncode == 0, (
            f"{service}: no-wheel wheel layer must exit 0:\n{result.stdout}\n{result.stderr}"
        )
        assert not ctx.exists(), f"{service}: staged context not cleaned up on the no-wheel path"

    def test_manifest_copy_precedes_deps_run(self, service):
        """The local-requirements manifest is COPYed (guaranteed-sibling glob
        idiom, so an absent manifest never fails the COPY) immediately before
        the deps RUN that installs it."""
        instrs = _instructions(_dockerfile(service))
        copy_idx = [i for i, (_, line) in enumerate(instrs) if line == MANIFEST_COPY]
        assert len(copy_idx) == 1, f"{service}: missing the manifest COPY sibling idiom"
        deps_idx = [
            i for i, (kw, line) in enumerate(instrs) if kw == "RUN" and PRIMER_SPEC[service] in line
        ]
        assert len(deps_idx) == 1, f"{service}: expected exactly one deps RUN instruction"
        assert copy_idx[0] < deps_idx[0], f"{service}: manifest COPY must precede the deps RUN"

    def test_deps_run_installs_manifest_before_toolchain_purge(self, service):
        """The deps RUN conditionally installs the staged manifest AFTER the
        primer (including its dev fallback) and BEFORE the toolchain purge, so
        a native dep in the local delta still compiles; the staged context is
        removed in the same RUN's cleanup."""
        deps = _deps_body(service)
        assert "&& if [ -f /tmp/deps-ctx/osprey-local-requirements.txt ]; then" in deps, (
            f"{service}: manifest install missing or not &&-chained in the deps RUN"
        )
        assert MANIFEST_INSTALL in deps, f"{service}: deps RUN does not install the manifest"
        assert deps.index(MANIFEST_INSTALL) > deps.index("WARNING: pin unreleased"), (
            f"{service}: manifest install must follow the primer's dev-fallback construct"
        )
        assert deps.index(MANIFEST_INSTALL) < deps.index("apt-get purge -y build-essential"), (
            f"{service}: manifest install must precede the toolchain purge"
        )
        assert "rm -rf /var/lib/apt/lists/* /tmp/deps-ctx" in deps, (
            f"{service}: deps RUN must clean up /tmp/deps-ctx with the apt cleanup"
        )

    def test_deps_layer_propagates_manifest_install_failure(self, service, tmp_path):
        """Empirical probe: with a manifest staged and its `pip install -r`
        failing, the whole deps RUN body must exit nonzero (the conditional and
        the trailing cleanup must not swallow the failure)."""
        result, ctx = _probe_deps_body(_deps_body(service), tmp_path, with_manifest=True)
        assert result.returncode != 0, (
            f"{service}: deps RUN exited 0 despite the manifest install failing:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert ctx.exists(), f"{service}: cleanup ran despite the manifest install failing"

    def test_deps_layer_no_manifest_is_success(self, service, tmp_path):
        """Empirical probe: with no manifest staged the conditional is a no-op
        and the deps RUN succeeds, still cleaning up the staged context."""
        result, ctx = _probe_deps_body(_deps_body(service), tmp_path, with_manifest=False)
        assert result.returncode == 0, (
            f"{service}: no-manifest deps RUN must exit 0:\n{result.stdout}\n{result.stderr}"
        )
        assert not ctx.exists(), f"{service}: staged context not cleaned up on the no-manifest path"


def _probe_wheel_body(body: str, tmp_path, *, with_wheel: bool):
    """Execute a wheel-layer RUN body in a sandbox with a real shell.

    ``/tmp/ctx`` is rewritten to a temp dir (optionally holding a fake wheel)
    and a stub ``pip`` that always exits 1 shadows the real one on PATH, so the
    probe exercises the body's failure-propagation shape without a container.
    Returns ``(CompletedProcess, ctx_path)``.
    """
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    if with_wheel:
        (ctx / "osprey_framework-0.0.0-py3-none-any.whl").write_text("")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    stub_pip = stub_bin / "pip"
    stub_pip.write_text("#!/bin/sh\nexit 1\n")
    stub_pip.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    result = subprocess.run(
        ["sh", "-c", body.replace("/tmp/ctx", str(ctx))],
        capture_output=True,
        text=True,
        env=env,
    )
    return result, ctx


def _probe_deps_body(body: str, tmp_path, *, with_manifest: bool):
    """Execute a deps-layer RUN body in a sandbox with a real shell.

    ``/tmp/deps-ctx`` (and the apt lists dir) are rewritten to temp dirs, and
    stub ``apt-get``/``pip`` shadow the real ones on PATH: apt-get is a no-op
    and pip succeeds for the primer but exits 1 for any ``-r`` (manifest)
    install, so the probe exercises the manifest branch's failure propagation
    without a container. Returns ``(CompletedProcess, ctx_path)``.
    """
    ctx = tmp_path / "deps-ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    if with_manifest:
        (ctx / "osprey-local-requirements.txt").write_text("no-such-dep==0.0.0\n")
    apt_lists = tmp_path / "apt-lists"
    apt_lists.mkdir()
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name, script in (
        ("apt-get", "#!/bin/sh\nexit 0\n"),
        ("pip", '#!/bin/sh\ncase " $* " in *" -r "*) exit 1 ;; *) exit 0 ;; esac\n'),
    ):
        stub = stub_bin / name
        stub.write_text(script)
        stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        OSPREY_VERSION="0.0.0",
        OSPREY_DEV="",
    )
    rewritten = body.replace("/tmp/deps-ctx", str(ctx)).replace(
        "/var/lib/apt/lists", str(apt_lists)
    )
    result = subprocess.run(["sh", "-c", rewritten], capture_output=True, text=True, env=env)
    return result, ctx


def test_virtual_accelerator_wheel_extra_placement():
    """The VA extra is primed in the deps layer and re-resolved on the wheel
    layer's FIRST install (so a dev wheel that adds/bumps a dep inside the extra
    gets it — `pip check` cannot detect missing extras), while the second
    `--no-deps --force-reinstall` install stays plain."""
    deps = _deps_body("virtual_accelerator")
    wheel = _wheel_body("virtual_accelerator")
    assert "[virtual-accelerator]" in deps
    assert 'pip install --no-cache-dir "${whl}[virtual-accelerator]"' in wheel
    first_install, _, after_force_reinstall = wheel.partition("--force-reinstall")
    assert "[virtual-accelerator]" in first_install
    assert "[virtual-accelerator]" not in after_force_reinstall


def test_every_service_ships_non_self_excluding_dockerignore():
    """Every service template dir that ships a Dockerfile must ship a
    .dockerignore that does not exclude `.dockerignore` itself — it is the
    guaranteed COPY sibling that keeps the optional-wheel glob matching."""
    dockerfile_dirs = sorted(p.parent for p in SERVICES_DIR.glob("*/Dockerfile"))
    assert dockerfile_dirs, "no service Dockerfiles found"
    for d in dockerfile_dirs:
        di = d / ".dockerignore"
        assert di.exists(), f"{d.name}: ships a Dockerfile but no .dockerignore"
        entries = {
            line.strip()
            for line in di.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        offenders = {e for e in entries if e.strip("/") == ".dockerignore"}
        assert not offenders, f"{d.name}: .dockerignore self-excludes: {offenders}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
