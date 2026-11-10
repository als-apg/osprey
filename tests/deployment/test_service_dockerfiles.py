"""Build invariants for the shipped service Dockerfiles.

`osprey up` builds each managed service from a template Dockerfile shipped
under ``osprey/templates/services/<name>/``. Two sets of checks live here, over
two different sets of files.

The **layer-split contract**, over the four framework-primed services
(event_dispatcher, virtual_accelerator, bluesky, bluesky_web) that
:data:`PRIMER_SPEC` names with the exact spec their deps layer installs — those
recipes are not only parsed but executed, so each has to be named with its spec
rather than discovered. This module pins the shared layer contract they follow
so a fast per-project dev rebuild only re-runs the cheap wheel layer, never the
expensive framework/deps install:

- a **deps layer** (one RUN) that primes the pinned framework release + deps,
  with a dev-gated fallback for an unreleased pin, then installs an optional
  dev-staged ``osprey-local-requirements.txt`` manifest (the local dependency
  delta, COPYed via the guaranteed-sibling idiom) before purging the toolchain,
  and
- a separate **wheel layer** that optionally overlays a locally-built wheel via
  the ``COPY .osprey-layer-anchor *.wh[l]`` idiom, force-reinstalls it
  ``--no-deps`` and runs ``pip check``,

followed by a metadata-only ``ARG OSPREY_PROJECT_NAME`` / ``LABEL
com.osprey.project`` pair kept last so a per-project value never invalidates the
shared deps cache. Each RUN body is parsed under ``sh -n`` so a shell syntax
error in the conditional install fails here rather than at ``docker build``.

The **whole-file invariants**, over every Dockerfile shipped as a literal file —
all seven services plus the web-terminal auth sidecar, discovered by glob rather
than listed: the proxy-delivery idiom on every apt-using RUN, and a
non-self-excluding ``.dockerignore`` beside every service recipe.
"""

import dataclasses
import pathlib
import re
import subprocess

import pytest

import osprey
from tests.deployment._cache_mounts import assert_build_caches_mounted
from tests.deployment._pip_probe import (
    APT_NOOP,
    UV_TO_PIP,
    deps_pip_calls,
    deps_sandbox,
    lock_path_install_calls,
    must_be_held_to_the_lock,
    primer_pip_argv,
    probe_wheel_body,
    run_body,
    stubbed_env,
    wheel_pip_calls,
)
from tests.deployment._proxy_idiom import (
    assert_apt_runs_carry_proxy_idiom,
    assert_site_ca_idiom,
    carries_proxy_idiom,
    run_instructions,
)

TEMPLATES_DIR = pathlib.Path(osprey.__file__).parent / "templates"
SERVICES_DIR = TEMPLATES_DIR / "services"

# Every Dockerfile shipped as a literal file — the managed services plus the
# web-terminal auth sidecar. Discovered rather than listed, so a new service
# recipe is covered by the shared invariants below the day it lands. The
# project image's recipe is a Jinja template and is checked from the rendered
# output instead (tests/cli/test_dockerfile_template.py).
SHIPPED_DOCKERFILES = sorted(TEMPLATES_DIR.rglob("Dockerfile"))
SHIPPED_DOCKERFILE_IDS = [str(p.parent.relative_to(TEMPLATES_DIR)) for p in SHIPPED_DOCKERFILES]


@dataclasses.dataclass(frozen=True)
class SiteBuildShape:
    """What one recipe must carry to honour a site's build settings.

    Attributes:
        trust_vars: The trust variables this image's tools read. Every tool
            family reads a different one and none reads the system store by
            default, so a variable missing here is a tool that keeps trusting
            its own bundle.
        pip_args: The site ARGs this recipe declares. A recipe that runs no pip
            declares only the proxy-bypass list, which is the one name that is
            not pip's own environment variable and has to be mapped by hand.
        deps_fetch: The token selecting the RUN that installs this recipe's
            dependencies — the one that has to carry the bypass map.
        bootstrap_fetches: Network fetches that legitimately precede the CA
            layer; see :func:`assert_site_ca_idiom`.
    """

    trust_vars: tuple[str, ...]
    pip_args: tuple[str, ...]
    deps_fetch: str
    bootstrap_fetches: int = 0


#: The trust variables a Python image's tools read: pip trusts only its bundled
#: certifi without PIP_CERT, and the other two cover Python's ssl module and
#: requests. An image that also carries Node adds NODE_EXTRA_CA_CERTS.
_PYTHON_TRUST_VARS = ("PIP_CERT", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
_NODE_TRUST_VARS = (*_PYTHON_TRUST_VARS, "NODE_EXTRA_CA_CERTS")

#: The three site ARGs a pip-installing recipe declares.
_PIP_SITE_ARGS = ("PIP_NO_PROXY", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL")

#: Per-recipe site-build shape, keyed as :data:`SHIPPED_DOCKERFILE_IDS` spells
#: it. ``services/qmd`` is the one recipe on a base image with no trust store at
#: all, so its CA layer necessarily follows the plain-HTTP bootstrap fetch that
#: installs one; it installs its dependencies with npm and runs no pip.
SITE_BUILD_SHAPE = {
    "modules/web_terminals/auth_sidecar": SiteBuildShape(
        _PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"
    ),
    "services/bluesky": SiteBuildShape(_PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/bluesky_web": SiteBuildShape(_PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/event_dispatcher": SiteBuildShape(_NODE_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/gchat_bridge": SiteBuildShape(_PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/nextcloud_bridge": SiteBuildShape(_PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/optimizer": SiteBuildShape(_PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/teams_bridge": SiteBuildShape(_PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/slack_bridge": SiteBuildShape(_PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"),
    "services/qmd": SiteBuildShape(
        _NODE_TRUST_VARS, ("PIP_NO_PROXY",), "npm install", bootstrap_fetches=1
    ),
    "services/virtual_accelerator": SiteBuildShape(
        _PYTHON_TRUST_VARS, _PIP_SITE_ARGS, "pip install"
    ),
}

# The four services whose layer split is checked here, and the pinned primer
# spec each one's deps layer installs from PyPI. Only the virtual accelerator
# carries an extra.
PRIMER_SPEC = {
    "event_dispatcher": "osprey-framework==$OSPREY_VERSION",
    "virtual_accelerator": "osprey-framework==$OSPREY_VERSION",
    "bluesky": "osprey-framework==$OSPREY_VERSION",
    "bluesky_web": "osprey-framework==$OSPREY_VERSION",
}
SERVICES = sorted(PRIMER_SPEC)

# The dev-staged local-dependency manifest and lock constraints: COPYed next to
# .osprey-layer-anchor (the guaranteed sibling keeps the glob matching when
# absent) and installed inside the deps RUN while the C toolchain is still
# available.
MANIFEST_COPY = (
    "COPY .osprey-layer-anchor osprey-local-requirements.tx[t] "
    "osprey-lock-constraints.tx[t] /tmp/deps-ctx/"
)
MANIFEST_INSTALL = "pip install -r /tmp/deps-ctx/osprey-local-requirements.txt"

#: Where the lock-driven deps layer keeps the lock for the layers after it.
LOCK_IN_IMAGE = "/opt/osprey/lock-constraints.txt"


def _dockerfile(service: str) -> str:
    return (SERVICES_DIR / service / "Dockerfile").read_text()


def _run_bodies(text: str) -> list[str]:
    """Every RUN instruction's shell body, line-continuations joined."""
    joined = re.sub(r"\\\n", " ", text)
    return re.findall(r"^RUN (?:--\S+\s+)*(.+)$", joined, flags=re.MULTILINE)


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
        assert "COPY .osprey-layer-anchor *.wh[l] /tmp/ctx/" in text, (
            f"{service}: missing the guaranteed-sibling wheel COPY idiom"
        )

    def test_wheel_install_sequence(self, service):
        wheel = _wheel_body(service)
        if service == "virtual_accelerator":
            # VA's first (deps-resolving) install carries the extra so a dev
            # wheel that changes the extra's deps picks them up.
            assert 'pip install $lock "${whl}[virtual-accelerator]"' in wheel
        else:
            assert "pip install $lock /tmp/ctx/*.whl" in wheel
        assert "pip install --no-deps --force-reinstall /tmp/ctx/*.whl" in wheel
        assert "pip check" in wheel
        # Always clean up the staged context regardless of whether a wheel was
        # present (so a no-op wheel layer still leaves no /tmp/ctx behind).
        assert "rm -rf /tmp/ctx" in wheel

    def test_pinned_primer_spec_present(self, service):
        deps = _deps_body(service)
        spec = PRIMER_SPEC[service]
        assert f'uv pip install ${{OSPREY_PIP_PRE:+--prerelease=allow}} "{spec}"' in deps, (
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
            result = subprocess.run(
                ["sh", "-n", "-c", body], capture_output=True, text=True, timeout=60
            )
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
        result, ctx = probe_wheel_body(_wheel_body(service), tmp_path, with_wheel=True)
        assert result.returncode != 0, (
            f"{service}: wheel-layer RUN exited 0 despite pip failing — a broken "
            f"dev-wheel install would build a stale image silently:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert ctx.exists(), f"{service}: cleanup ran despite the install failing"

    def test_wheel_layer_no_wheel_is_noop_success(self, service, tmp_path):
        """Empirical probe: with no wheel staged the RUN is a successful no-op
        (the untaken `if` exits 0) and still cleans up the staged context."""
        result, ctx = probe_wheel_body(_wheel_body(service), tmp_path, with_wheel=False)
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
        # Twice: once alone on the lock-driven path, once after the primer on
        # the release path.
        assert deps.rindex(MANIFEST_INSTALL) > deps.index("WARNING: pin unreleased"), (
            f"{service}: manifest install must follow the primer's dev-fallback construct"
        )
        assert deps.rindex(MANIFEST_INSTALL) < deps.index("apt-get purge -y build-essential"), (
            f"{service}: manifest install must precede the toolchain purge"
        )
        assert "rm -rf /tmp/deps-ctx" in deps, f"{service}: deps RUN must clean up /tmp/deps-ctx"

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

    def test_lock_path_primes_no_framework(self, service, tmp_path):
        """With the lock constraints staged the deps layer installs the staged
        manifest alone — nothing named osprey-framework reaches the index — and
        keeps the lock for the layers after it, appended to the setuptools
        constraint every installer here reads."""
        lock = "fwdep==1.0 ; python_version >= '3.11'\n"
        calls = deps_pip_calls(
            _deps_body(service),
            tmp_path,
            {"OSPREY_VERSION": "0.0.0", "OSPREY_DEV": "1"},
            staged={
                "osprey-local-requirements.txt": "fwdep\n",
                "osprey-lock-constraints.txt": lock,
            },
        )
        installs = [c for c in calls if c[:1] == ["install"]]
        assert not [c for c in installs if any("osprey-framework" in a for a in c)], calls
        assert ["install", "-r", str(tmp_path / "deps-ctx/osprey-local-requirements.txt")] in (
            installs
        ), calls
        assert (tmp_path / "opt-osprey/lock-constraints.txt").read_text() == lock

    def test_lock_path_constrains_the_wheel_install(self, service, tmp_path):
        calls = wheel_pip_calls(_wheel_body(service), tmp_path, lock=True)
        first = calls[0]
        assert first[:3] == ["install", "-c", str(tmp_path / "opt-osprey/lock-constraints.txt")]

    def test_release_path_wheel_install_is_unconstrained(self, service, tmp_path):
        calls = wheel_pip_calls(_wheel_body(service), tmp_path, lock=False)
        assert "-c" not in calls[0]


# ── Pre-release pins ─────────────────────────────────────────────────────────
#
# Every recipe that installs the framework by version, discovered by the ARG it
# reads rather than listed, so a new framework-primed recipe is covered the day
# it lands. osprey-connectors releases on its own tags and the framework
# declares only a floor on it, which names no pre-release — and plain pip never
# picks a pre-release for such a requirement, so a beta framework would pass
# over the connectors pre-releases it was tested with. The deps layer therefore
# takes an `OSPREY_PIP_PRE` build arg and, when it is set, resolves with `--pre`,
# free to take the newest connectors pre-release.

FRAMEWORK_PINNED_DOCKERFILES = [
    p for p in SHIPPED_DOCKERFILES if "$OSPREY_VERSION" in p.read_text(encoding="utf-8")
]
FRAMEWORK_PINNED_IDS = [
    str(p.parent.relative_to(TEMPLATES_DIR)) for p in FRAMEWORK_PINNED_DOCKERFILES
]


#: The distributions a deps layer primes by version: the framework, or the
#: workspace package a service image installs in its place (the optimizer).
_PRIMED_DISTRIBUTIONS = ("osprey-framework", "osprey-optimizer")


def _version_pinned_bodies(dockerfile: pathlib.Path) -> list[str]:
    """Every RUN body installing a primed distribution by version.

    The deps layer, and in the VA the layer adding its extra on top of it:
    both resolve against the same pin, so both must admit a pre-release pin.
    """
    text = dockerfile.read_text(encoding="utf-8")
    bodies = [
        b
        for b in _run_bodies(text)
        if any(d in b for d in _PRIMED_DISTRIBUTIONS) and "$OSPREY_VERSION" in b
    ]
    assert bodies, f"{dockerfile}: no RUN installs a primed distribution by version"
    return bodies


@pytest.mark.parametrize("dockerfile", FRAMEWORK_PINNED_DOCKERFILES, ids=FRAMEWORK_PINNED_IDS)
class TestPrereleasePin:
    def test_covers_every_framework_primed_recipe(
        self,
        dockerfile,  # noqa: ARG002 - a column of the shared FRAMEWORK_PINNED_DOCKERFILES table
    ):
        # The discovery above must find the recipes the layer-split contract
        # names — a recipe that pins the framework some other way would slip
        # past both.
        assert {p.parent.name for p in FRAMEWORK_PINNED_DOCKERFILES} >= set(SERVICES)

    def test_declares_the_prerelease_arg(self, dockerfile):
        text = dockerfile.read_text(encoding="utf-8")
        assert re.search(r'^ARG OSPREY_PIP_PRE=""$', text, flags=re.MULTILINE), (
            f"{dockerfile}: missing the OSPREY_PIP_PRE build arg"
        )

    def test_primer_admits_prereleases_when_the_pin_is_one(self, dockerfile, tmp_path):
        for i, body in enumerate(_version_pinned_bodies(dockerfile)):
            sandbox = tmp_path / str(i)
            sandbox.mkdir()
            argv = primer_pip_argv(
                body, sandbox, {"OSPREY_VERSION": "2026.9.0b2", "OSPREY_PIP_PRE": "1"}
            )
            assert "--prerelease=allow" in argv, argv
            assert any(a.endswith("==2026.9.0b2") for a in argv), argv

    def test_primer_stays_strict_for_a_stable_pin(self, dockerfile, tmp_path):
        for i, body in enumerate(_version_pinned_bodies(dockerfile)):
            sandbox = tmp_path / str(i)
            sandbox.mkdir()
            argv = primer_pip_argv(
                body, sandbox, {"OSPREY_VERSION": "2026.9.0", "OSPREY_PIP_PRE": ""}
            )
            assert "--prerelease=allow" not in argv, argv


def _probe_deps_body(body: str, tmp_path, *, with_manifest: bool):
    """Execute a deps-layer RUN body in a sandbox with a real shell.

    ``/tmp/deps-ctx`` (and the apt lists dir) are rewritten to temp dirs, and
    stub ``apt-get``/``pip`` shadow the real ones on PATH: apt-get is a no-op
    and pip succeeds for the primer but exits 1 for any ``-r`` (manifest)
    install, so the probe exercises the manifest branch's failure propagation
    without a container. Returns ``(CompletedProcess, ctx_path)``.
    """
    rewritten, ctx = deps_sandbox(body, tmp_path)
    if with_manifest:
        (ctx / "osprey-local-requirements.txt").write_text("no-such-dep==0.0.0\n")
    pip = '#!/bin/sh\ncase " $* " in *" -r "*) exit 1 ;; *) exit 0 ;; esac\n'
    env = stubbed_env(
        tmp_path,
        {"apt-get": APT_NOOP, "pip": pip, "uv": UV_TO_PIP},
        OSPREY_VERSION="0.0.0",
        OSPREY_DEV="",
    )
    return run_body(rewritten, env), ctx


# ── Node provenance (event_dispatcher) ───────────────────────────────────────
#
# Only the dispatch image ships Node — the worker shells out to the Claude Code
# CLI, which is a Node package. It comes from the base image's own Debian
# release rather than a third-party apt repo, so the image has one package
# provenance and needs no network beyond the Debian mirror to build.


def _run_instruction_source(text: str, needle: str) -> str:
    """The single RUN instruction containing *needle*, as Docker executes it.

    :func:`_run_bodies` joins continuations naively, which folds any comment
    line *inside* a continued RUN into the middle of the body — everything
    after it then reads as inert shell comment, though Docker strips those
    lines and runs what follows. This helper drops them the way Docker does:
    a comment inside a continuation is removed entirely, and so neither ends
    the instruction nor comments out the commands below it. The reconstruction
    itself lives in :func:`tests.deployment._proxy_idiom.run_instructions`,
    which the shared proxy-idiom assertion parses with too.
    """
    matches = [instr for instr in run_instructions(text) if needle in instr]
    assert len(matches) == 1, f"expected exactly one RUN containing {needle!r}, got {len(matches)}"
    return matches[0]


def test_event_dispatcher_node_comes_from_the_base_image_distro():
    node = _run_instruction_source(_dockerfile("event_dispatcher"), "nodejs")
    assert "apt-get install -y --no-install-recommends ca-certificates nodejs npm" in node, node
    # The pinned CLI install and the apt cleanup both follow an inline comment
    # inside the continuation, so they are only reachable in the comment-
    # stripped body above — a naive join would leave them commented out.
    assert "npm install -g @anthropic-ai/claude-code@" in node


def test_event_dispatcher_has_no_third_party_node_apt_repo():
    """Asserted as an absence over the whole file: any of these tokens coming
    back means the vendor setup-script pipeline came back with them."""
    text = _dockerfile("event_dispatcher")
    for token in ("nodesource", "gnupg", "setup_20.x", "apt-key"):
        assert token not in text, f"{token} is back in the Dockerfile — third-party Node repo"
    node = _run_instruction_source(text, "nodejs")
    assert "| bash" not in node and "| sh" not in node, (
        "the node layer pipes a downloaded script into a shell"
    )


def test_event_dispatcher_keeps_npm_in_the_final_image():
    """nodejs and npm stay in the final image, beside the Claude Code CLI npm
    installs there and a headless run executes directly. The deps layer's own
    toolchain purge is asserted elsewhere; what must not appear is a purge in
    *this* layer."""
    text = _dockerfile("event_dispatcher")
    node = _run_instruction_source(text, "nodejs")
    assert "apt-get purge" not in node, "the node layer purges packages"
    assert "autoremove" not in node, "the node layer autoremoves packages"
    assert "nodejs and npm stay in the final image" in text


def _va_extra_body() -> str:
    """The VA's virtual-accelerator layer: the RUN adding the extra on top of the deps layer."""
    spec = "osprey-framework[virtual-accelerator]==$OSPREY_VERSION"
    bodies = [b for b in _run_bodies(_dockerfile("virtual_accelerator")) if spec in b]
    assert len(bodies) == 1, f"expected exactly one RUN adding the extra, got {len(bodies)}"
    return bodies[0]


def test_virtual_accelerator_wheel_extra_placement():
    """The VA extra is added in its own layer after the shared deps layer and
    re-resolved on the wheel layer's FIRST install (so a dev wheel that
    adds/bumps a dep inside the extra gets it — `pip check` cannot detect
    missing extras), while the second `--no-deps --force-reinstall` install
    stays plain."""
    deps = _deps_body("virtual_accelerator")
    extra = _va_extra_body()
    wheel = _wheel_body("virtual_accelerator")
    assert "[virtual-accelerator]" not in deps, "the extra is back in the shared deps layer"
    assert "--only-binary pcaspy" in extra
    assert '[ "$OSPREY_DEV" = "1" ]' in extra and "exit 1" in extra, (
        "the extra layer lost the dev-gated fallback for an unreleased pin"
    )
    joined = re.sub(r"\\\n", " ", _dockerfile("virtual_accelerator"))
    assert joined.index(deps) < joined.index(extra) < joined.index(wheel), (
        "the layers must run deps, then the extra, then the wheel overlay"
    )
    assert 'pip install $lock "${whl}[virtual-accelerator]"' in wheel
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


# ── Shared deps layer ────────────────────────────────────────────────────────
#
# A deploy builds several of these images at once, and the framework install is
# by far the slowest step in each. BuildKit runs a step once and shares the
# layer only when everything up to and including it is identical: the same
# instructions, the same build args, and the same bytes in every file a COPY
# reads. These pin both halves of that for the Python service recipes.

LAYER_ANCHOR = ".osprey-layer-anchor"

#: Python service recipes that legitimately diverge before the deps layer. The
#: optimizer installs its own workspace package, with torch from the PyTorch
#: index, in place of the framework, so its deps layer is its own by design.
_OWN_DEPS_LAYER = {"services/optimizer"}

SHARED_DEPS_IDS = [
    label
    for label, path in zip(SHIPPED_DOCKERFILE_IDS, SHIPPED_DOCKERFILES, strict=True)
    if "FROM python:" in path.read_text(encoding="utf-8") and label not in _OWN_DEPS_LAYER
]


def _final_stage_chain(text: str) -> list[str]:
    """The final image's layer chain as BuildKit keys it, base stage first.

    A stage that starts ``FROM <earlier stage>`` continues that stage's
    layers, so its chain is the earlier stage's followed by its own.
    ``AS <name>`` only names a stage and keys nothing, so it is dropped, and
    global ARGs before the first FROM are not part of any stage.

    An ARG creates no layer, but every ARG in scope is in the environment of
    each RUN after it, and so in that RUN's cache key. ARGs do NOT cross a
    FROM: a stage built on an earlier one inherits its layers and none of its
    ARGs. So ARG lines are not entries of their own; each RUN carries the ARG
    names in scope at that point, which is what makes two recipes with the
    same RUN text but different ARGs in scope compare unequal.

    A stage built on an earlier stage also leaves a boundary entry, so such a
    chain never equals a single-stage one. That is conservative on purpose:
    BuildKit keys a RUN after ``FROM <stage>`` differently from the same RUN
    in a single-stage recipe once the earlier stage sets an ENV, and nothing in
    the instruction text shows it (measured on BuildKit 0.34: the deps RUN of a
    ``FROM va-base`` final stage never reused the bridge image's).
    """
    stages: dict[str, list[str]] = {}
    chains: list[list[str]] = []
    current: list[str] | None = None
    scope: list[str] = []
    for keyword, line in _instructions(text):
        if keyword == "FROM":
            words = line.split()
            base = words[1]
            name = words[3] if len(words) > 3 and words[2].upper() == "AS" else None
            if base in stages:
                current = [*stages[base], f"(stage boundary: FROM {base})"]
            else:
                current = [f"FROM {base}"]
            chains.append(current)
            scope = []
            if name:
                stages[name] = current
        elif current is None:
            continue
        elif keyword == "ARG":
            arg = line.split(maxsplit=1)[1].split("=", 1)[0]
            if arg not in scope:
                scope.append(arg)
        elif keyword == "RUN":
            current.append(f"{line}  [ARGs in scope: {', '.join(scope)}]")
        else:
            current.append(line)
    assert chains, "no FROM in the recipe"
    return chains[-1]


def _shared_prefix(dockerfile: pathlib.Path) -> list[str]:
    """The final image's instructions from FROM through the deps RUN."""
    prefix: list[str] = []
    for line in _final_stage_chain(dockerfile.read_text(encoding="utf-8")):
        prefix.append(line)
        if line.startswith("RUN ") and MANIFEST_INSTALL in line:
            return prefix
    raise AssertionError(f"{dockerfile}: no deps RUN installing the manifest")


def test_every_build_context_ships_the_same_layer_anchor():
    """Each recipe's context carries the anchor, byte-identical to every other.

    The recipes COPY it beside their optional globs, so its content is part of
    every cache key from the site-CA layer down. A copy that drifted would cost
    nothing visible and quietly give that image its own framework install.
    """
    anchors = {
        label: path.parent / LAYER_ANCHOR
        for label, path in zip(SHIPPED_DOCKERFILE_IDS, SHIPPED_DOCKERFILES, strict=True)
    }
    missing = sorted(label for label, anchor in anchors.items() if not anchor.is_file())
    assert not missing, f"no {LAYER_ANCHOR} beside: {missing}"
    contents = {anchor.read_bytes() for anchor in anchors.values()}
    assert len(contents) == 1, f"{LAYER_ANCHOR} copies differ between build contexts"


@pytest.mark.parametrize("dockerfile", SHIPPED_DOCKERFILES, ids=SHIPPED_DOCKERFILE_IDS)
def test_no_context_ignores_the_layer_anchor(dockerfile):
    """The anchor reaches the build: a context that ignored it would fail every COPY."""
    ignore = dockerfile.parent / ".dockerignore"
    entries = {
        line.strip().strip("/")
        for line in ignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert not entries & {LAYER_ANCHOR, ".*", "*"}, f"{ignore} excludes {LAYER_ANCHOR}"


@pytest.mark.parametrize("label", SHARED_DEPS_IDS)
def test_python_services_share_one_deps_layer(label):
    """Every Python service recipe is identical from FROM through the deps RUN.

    Whatever one image needs beyond the framework (an extra, the queueserver
    pin, Node) belongs in a layer after it; one differing instruction above it
    gives that image a framework install of its own, which on a slow index is
    most of a deploy's build time.
    """
    reference = TEMPLATES_DIR / "services" / "bluesky_web" / "Dockerfile"
    dockerfile = TEMPLATES_DIR / label / "Dockerfile"
    assert _shared_prefix(dockerfile) == _shared_prefix(reference), (
        f"{label}: diverges from bluesky_web before the end of the deps layer"
    )


def test_shared_deps_layer_covers_the_compose_built_services():
    """Floor on the discovery: the services a deploy builds together are in it."""
    assert {
        "modules/web_terminals/auth_sidecar",
        "services/bluesky",
        "services/bluesky_web",
        "services/event_dispatcher",
        "services/virtual_accelerator",
    } <= set(SHARED_DEPS_IDS)


def test_python_services_share_one_deps_layer__mutation_extra_in_the_deps_layer():
    """The extra moved back into the VA's deps RUN gives it its own install again."""
    path = TEMPLATES_DIR / "services" / "virtual_accelerator" / "Dockerfile"
    text = path.read_text(encoding="utf-8")
    deps_spec = '"osprey-framework==$OSPREY_VERSION"'
    assert text.count(deps_spec) == 1, "mutation is stale"
    mutated = text.replace(deps_spec, '"osprey-framework[virtual-accelerator]==$OSPREY_VERSION"')
    reference = TEMPLATES_DIR / "services" / "bluesky_web" / "Dockerfile"
    chain = _final_stage_chain(mutated)
    deps = next(i for i, line in enumerate(chain) if MANIFEST_INSTALL in line)
    assert chain[: deps + 1] != _shared_prefix(reference)


def test_final_stage_chain_follows_the_parent_stage():
    """A final stage built FROM an earlier stage continues that stage's layers."""
    chain = _final_stage_chain(
        "ARG TARGETARCH\n"
        "FROM python:3.11-slim AS base\nWORKDIR /app\n"
        "FROM base AS other\nRUN echo other\n"
        "FROM base\nRUN echo final\n"
    )
    assert chain == [
        "FROM python:3.11-slim",
        "WORKDIR /app",
        "(stage boundary: FROM base)",
        "RUN echo final  [ARGs in scope: ]",
    ]


def test_final_stage_chain_drops_args_at_a_from():
    """An ARG declared in the parent stage is not in scope after FROM <parent>."""
    inherited = _final_stage_chain(
        'FROM python:3.11-slim AS base\nARG A=""\nFROM base\nRUN echo deps\n'
    )
    redeclared = _final_stage_chain(
        'FROM python:3.11-slim AS base\nARG A=""\nFROM base\nARG A=""\nRUN echo deps\n'
    )
    assert inherited[-1] == "RUN echo deps  [ARGs in scope: ]"
    assert redeclared[-1] == "RUN echo deps  [ARGs in scope: A]"


def test_python_services_share_one_deps_layer__mutation_va_final_stage_from_va_base():
    """A VA final stage built FROM va-base no longer counts as sharing the deps layer."""
    path = TEMPLATES_DIR / "services" / "virtual_accelerator" / "Dockerfile"
    text = path.read_text(encoding="utf-8")
    final = text.rindex("\nFROM python:3.11-slim\n")
    deps = text.index('ARG OSPREY_VERSION=""', final)
    mutated = text[:final] + "\nFROM va-base\n" + text[deps:]
    reference = TEMPLATES_DIR / "services" / "bluesky_web" / "Dockerfile"
    chain = _final_stage_chain(mutated)
    end = next(i for i, line in enumerate(chain) if MANIFEST_INSTALL in line)
    assert chain[: end + 1] != _shared_prefix(reference)


#: Shipped recipes that install with pip, and so fetch through the build caches.
#: The qmd sidecar installs with npm on a Node base and is not one of them.
PIP_INSTALLING = [p for p in SHIPPED_DOCKERFILES if "pip install" in p.read_text(encoding="utf-8")]


@pytest.mark.parametrize(
    "dockerfile",
    PIP_INSTALLING,
    ids=[str(p.parent.relative_to(TEMPLATES_DIR)) for p in PIP_INSTALLING],
)
def test_installs_fetch_through_the_shared_build_caches(dockerfile):
    """Every pip and apt install mounts the caches the deploy's images share.

    The rule is spelled once, in :mod:`tests.deployment._cache_mounts`, and
    applied to the rendered project template too.
    """
    assert_build_caches_mounted(
        dockerfile.read_text(encoding="utf-8"),
        str(dockerfile.parent.relative_to(TEMPLATES_DIR)),
    )


# ── Proxy delivery ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("dockerfile", SHIPPED_DOCKERFILES, ids=SHIPPED_DOCKERFILE_IDS)
def test_apt_runs_deliver_the_proxy_settings(dockerfile):
    """Every apt-using RUN in every shipped recipe hands apt the proxy.

    The rule, and the idiom that satisfies it, are spelled once in
    :mod:`tests.deployment._proxy_idiom` — here and in the rendered project
    template's copy of this test, so a change of idiom is one edit.
    """
    assert_apt_runs_carry_proxy_idiom(
        dockerfile.read_text(), str(dockerfile.parent.relative_to(TEMPLATES_DIR))
    )


@pytest.mark.parametrize("dockerfile", SHIPPED_DOCKERFILES, ids=SHIPPED_DOCKERFILE_IDS)
def test_manifest_installing_recipes_constrain_setuptools(dockerfile):
    """Every recipe that installs the local manifest pins setuptools below 84.

    setuptools 84.0.0 broke ``setuptools_dso``'s compile-probe error handling,
    so any source build of the EPICS toolchain (p4p pulls
    pvxslibs/epicscorelibs) dies where no binary wheel exists — every
    linux/arm64 image. p4p is a base framework dependency, so it is in the
    manifest, so this is not optional for any recipe that installs it.

    Parametrized over the discovered set rather than a list on purpose: the
    single-recipe guard in :mod:`tests.deployment.test_arm64_p4p_image_build`
    covers only the rendered project image, which is how the sidecar spent a
    release without the constraint and only failed on a cold arm64 build.
    """
    text = dockerfile.read_text()
    if MANIFEST_INSTALL not in text:
        pytest.skip("recipe does not install the local-requirements manifest")

    deps_run = next(body for body in _run_bodies(text) if MANIFEST_INSTALL in body)
    assert "setuptools<84" in deps_run, (
        f"{dockerfile.parent.name}: deps layer does not pin setuptools below 84.0.0, "
        f"which breaks setuptools_dso's compile probe and fails every EPICS "
        f"source build (linux/arm64):\n{deps_run}"
    )
    assert "PIP_CONSTRAINT" in deps_run, (
        f"{dockerfile.parent.name}: the setuptools pin must be exported as "
        f"PIP_CONSTRAINT — a plain requirement pin does not reach pip's "
        f"isolated build environments:\n{deps_run}"
    )


@pytest.mark.parametrize("dockerfile", SHIPPED_DOCKERFILES, ids=SHIPPED_DOCKERFILE_IDS)
def test_every_shipped_recipe_carries_the_site_ca_layer(dockerfile):
    """Every image OSPREY builds accepts the site's CA before it fetches.

    A recipe without the layer is an image that verifies its build-time TLS
    against a trust store the site's re-signing proxy is not in, so the build
    dies at whichever fetch reaches the proxy first — and the auth sidecar,
    which reaches the identity provider at RUN time too, would start green and
    fail every login at the discovery fetch.
    """
    label = str(dockerfile.parent.relative_to(TEMPLATES_DIR))
    recipe = SITE_BUILD_SHAPE[label]
    for i, tree in enumerate(_stage_trees(dockerfile.read_text())):
        assert_site_ca_idiom(
            tree,
            f"{label} (stage tree {i + 1})",
            recipe.trust_vars,
            bootstrap_fetches=recipe.bootstrap_fetches,
        )


def _stage_trees(text: str) -> list[str]:
    """The recipe split where a stage starts from an external image.

    A stage built ``FROM`` an earlier stage (by name, or through a
    ``${...}``-selected name) inherits that stage's trust store, so it belongs
    to the same tree; a stage that starts from a registry image begins a new
    tree, which has to install the site CA itself. A single-stage recipe is one
    tree.
    """
    named: set[str] = set()
    starts: list[int] = []
    for match in re.finditer(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", text, flags=re.M | re.I):
        base, name = match.group(1), match.group(2)
        if base not in named and "${" not in base:
            starts.append(match.start())
        if name:
            named.add(name)
    starts = starts or [0]
    head = text[: starts[0]]
    ends = [*starts[1:], len(text)]
    return [head + text[a:b] for a, b in zip(starts, ends, strict=True)]


def test_stage_trees_split_at_an_external_base():
    """A stage on an earlier stage stays in its tree; a new registry base starts one."""
    trees = _stage_trees(
        "ARG T\nFROM python:3.11-slim AS base\nRUN a\nFROM base AS b\nRUN b\n"
        "FROM b-${T} AS c\nFROM python:3.11-slim\nRUN d\n"
    )
    assert len(trees) == 2
    assert "RUN b" in trees[0] and "RUN d" not in trees[0]
    assert "RUN d" in trees[1] and "RUN a" not in trees[1]


@pytest.mark.parametrize("dockerfile", SHIPPED_DOCKERFILES, ids=SHIPPED_DOCKERFILE_IDS)
def test_every_shipped_recipe_declares_the_site_pip_args(dockerfile):
    """Each build is handed the whole site set, not only the CA.

    Docker drops a ``--build-arg`` no recipe declares — with a warning nobody
    reads — so a missing ARG here is an internal mirror configured once in
    ``config.yml`` that reaches some images and quietly leaves this one
    resolving from PyPI. PIP_INDEX_URL and PIP_EXTRA_INDEX_URL are pip's own
    environment names, so declaring them is all it takes; PIP_NO_PROXY is not
    one, and has to be mapped onto the proxy-bypass names the deps layer's
    fetches read. A recipe that runs no pip declares only the bypass list.
    """
    label = str(dockerfile.parent.relative_to(TEMPLATES_DIR))
    recipe = SITE_BUILD_SHAPE[label]
    text = dockerfile.read_text()

    for arg in recipe.pip_args:
        assert re.search(rf'^ARG {arg}=""$', text, flags=re.M), (
            f"{label} declares no `ARG {arg}` — the builder passes it and Docker drops it"
        )
    deps_run = next(instr for instr in run_instructions(text) if recipe.deps_fetch in instr)
    guarded_map = (
        '[ -z "$PIP_NO_PROXY" ] || export NO_PROXY="$PIP_NO_PROXY" no_proxy="$PIP_NO_PROXY"'
    )
    assert guarded_map in deps_run, (
        f"{label}: the deps layer does not map PIP_NO_PROXY onto the names its "
        f"fetches read under a guard, so the site's bypass list is either inert "
        f"here or — exported unconditionally — wipes a bypass list the build "
        f"environment already carries:\n{deps_run}"
    )


def test_every_discovered_recipe_declares_its_site_build_shape():
    """A recipe with no entry above would silently skip both guards.

    The parametrization is keyed by the discovery glob, so a new Dockerfile
    lands in it the day it is added — and a lookup that quietly defaulted
    would cover it with the wrong trust variables rather than failing.
    """
    assert set(SHIPPED_DOCKERFILE_IDS) == set(SITE_BUILD_SHAPE), (
        "shipped Dockerfiles and their declared site-build shapes have drifted apart"
    )


def test_site_ca_guard_catches_a_recipe_without_the_layer():
    """Meta-test: a recipe that fetches with no CA layer must fail the guard."""
    bare = "FROM python:3.11-slim\nRUN apt-get update\n"
    with pytest.raises(AssertionError, match="OSPREY_SITE_CA"):
        assert_site_ca_idiom(bare, "synthetic", ("SSL_CERT_FILE",))


def test_shipped_dockerfiles_are_all_discovered():
    """Floor on the discovery glob: a typo that finds nothing would make the
    parametrization above vacuous rather than red."""
    assert "modules/web_terminals/auth_sidecar" in SHIPPED_DOCKERFILE_IDS
    assert {f"services/{s}" for s in SERVICES} <= set(SHIPPED_DOCKERFILE_IDS)


def test_proxy_guard_catches_a_bare_apt_run():
    """Meta-test: an apt RUN with no proxy delivery must fail the guard."""
    bare = "RUN apt-get update \\\n && apt-get install -y --no-install-recommends curl\n"
    assert not carries_proxy_idiom(run_instructions(bare)[0])
    with pytest.raises(AssertionError, match="proxy-delivery idiom"):
        assert_apt_runs_carry_proxy_idiom(bare, "synthetic")


def test_proxy_guard_accepts_the_declared_arg_idiom():
    """Meta-test: delivering the settings from a declared build ARG counts too
    — the project template's deps layer follows its bridge with exactly this."""
    from_arg = (
        'RUN export NO_PROXY="$PIP_NO_PROXY" no_proxy="$PIP_NO_PROXY" \\\n && apt-get update\n'
    )
    assert carries_proxy_idiom(run_instructions(from_arg)[0])


# ── Claude Code CLI pin parity ───────────────────────────────────────────────
#
# The framework's default CLI pin is spelled once in code
# (``_DEFAULT_CLAUDE_CLI_VERSION``, which the project Dockerfile.j2 renders
# from) and hand-copied into two places that cannot read it: the
# dispatch-worker Dockerfile (a literal file, not a template) and the
# installation page (three times; RST substitutions do not expand inside code
# blocks or inline literals). Each copy is asserted equal to the constant so a
# bump that misses one fails here instead of shipping two different CLIs.

DOCS_DIR = pathlib.Path(osprey.__file__).parents[2] / "docs" / "source"
INSTALLATION_PAGE = DOCS_DIR / "getting-started" / "installation.rst"
_CLI_PIN_RE = re.compile(r'@anthropic-ai/claude-code@([0-9][\w.\-]*)|cli_version: "([^"]+)"')


def _cli_pins(text: str) -> list[str]:
    """Every spelled-out CLI version in *text*, in order of appearance."""
    return [a or b for a, b in _CLI_PIN_RE.findall(text)]


def _default_cli_pin() -> str:
    from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

    return _DEFAULT_CLAUDE_CLI_VERSION


def test_event_dispatcher_pins_the_default_cli_version():
    pins = _cli_pins(_dockerfile("event_dispatcher"))
    assert pins == [_default_cli_pin()], (
        f"event_dispatcher/Dockerfile pins {pins}, _DEFAULT_CLAUDE_CLI_VERSION is "
        f"{_default_cli_pin()!r} — bump both together"
    )


def test_installation_page_pins_the_default_cli_version():
    pins = _cli_pins(INSTALLATION_PAGE.read_text())
    assert len(pins) >= 3, f"expected the install command and the pin example, found {pins}"
    assert set(pins) == {_default_cli_pin()}, (
        f"{INSTALLATION_PAGE.relative_to(DOCS_DIR)} spells the CLI pin as {sorted(set(pins))}, "
        f"_DEFAULT_CLAUDE_CLI_VERSION is {_default_cli_pin()!r}"
    )


def test_cli_pin_regex_reads_both_spellings():
    """Meta-test: the matcher sees the npm spec and the YAML key, and nothing else."""
    sample = 'npm install -g @anthropic-ai/claude-code@1.2.3 \\\n  cli_version: "4.5.6"  # x.y.z\n'
    assert _cli_pins(sample) == ["1.2.3", "4.5.6"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── Source builds are held to the lock in every layer ───────────────────────
#
# A `--dev` build stages osprey-lock-constraints.txt, the checkout's uv.lock
# exported as constraints, and every layer that installs a Python package must
# then install it at the lock's version: the deps layer, the extras layers, the
# queueserver layer, every wheel layer. Each RUN that installs anything is run
# here, under stubs, as a source build would run it, and every installer call
# that resolves packages has to have read the lock. Nothing may name an osprey
# distribution as a requirement either: on that path the only osprey code is
# the staged wheels, and a name would be resolved from the index.

_LOCK_PATH_ENV = {
    "OSPREY_VERSION": "0.0.0",
    "OSPREY_DEV": "1",
    "OSPREY_PIP_PRE": "",
    "BLUESKY_QUEUESERVER_VERSION": "0.0.25",
    "OPTIMIZER_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cpu",
    "PIP_NO_PROXY": "",
    "PIP_INDEX_URL": "",
    "PIP_EXTRA_INDEX_URL": "",
}

#: Recipes whose RUNs install Python packages (qmd installs with npm alone).
_PIP_RECIPES = [p for p in SHIPPED_DOCKERFILES if "pip install" in p.read_text(encoding="utf-8")]


def _installing_bodies(dockerfile: pathlib.Path) -> list[str]:
    """Each RUN that installs a Python package, minus the arm64 native-wheels
    builder stage, which compiles pcaspy at its hash-pinned version and ships
    nothing but that wheel."""
    return [
        instruction.removeprefix("RUN ")
        for instruction in run_instructions(dockerfile.read_text(encoding="utf-8"))
        if "pip install" in instruction and "pip wheel" not in instruction
    ]


@pytest.mark.parametrize(
    "dockerfile",
    _PIP_RECIPES,
    ids=[str(p.parent.relative_to(TEMPLATES_DIR)) for p in _PIP_RECIPES],
)
def test_every_install_on_a_source_build_is_held_to_the_lock(dockerfile, tmp_path):
    bodies = _installing_bodies(dockerfile)
    assert bodies, f"{dockerfile}: no installing RUN found"
    for i, body in enumerate(bodies):
        sandbox = tmp_path / str(i)
        sandbox.mkdir()
        calls = lock_path_install_calls(body, sandbox, _LOCK_PATH_ENV)
        for held, argv in calls:
            if must_be_held_to_the_lock(argv):
                assert held, f"{dockerfile}: an install not held to the lock: {argv}\n{body}"
            named = [a for a in argv if re.match(r"osprey[-_](framework|optimizer)\b", a)]
            assert not named, f"{dockerfile}: names an osprey distribution on the index: {argv}"


@pytest.mark.parametrize(
    "dockerfile",
    _PIP_RECIPES,
    ids=[str(p.parent.relative_to(TEMPLATES_DIR)) for p in _PIP_RECIPES],
)
def test_a_release_build_reads_no_lock(dockerfile, tmp_path):
    """The other side of the switch: with nothing staged no layer reads a lock,
    so a release build installs exactly what it did before the lock existed."""
    for i, body in enumerate(_installing_bodies(dockerfile)):
        sandbox = tmp_path / str(i)
        sandbox.mkdir()
        calls = lock_path_install_calls(
            body, sandbox, {**_LOCK_PATH_ENV, "OSPREY_DEV": ""}, source_build=False
        )
        assert not [argv for held, argv in calls if held], f"{dockerfile}: {calls}"


def test_the_extras_manifest_mapping_matches_the_recipes():
    """``_DEV_WHEEL_EXTRAS`` and the recipes that COPY the extras manifest name
    the same contexts, and each mapped extra is the one that recipe installs."""
    from osprey.deployment.compose_generator import _DEV_WHEEL_EXTRAS

    copying = {
        p.parent.name
        for p in SHIPPED_DOCKERFILES
        if "osprey-extra-requirements.tx[t]" in p.read_text(encoding="utf-8")
    }
    assert copying == set(_DEV_WHEEL_EXTRAS)
    for context, extras in _DEV_WHEEL_EXTRAS.items():
        text = (SERVICES_DIR / context / "Dockerfile").read_text(encoding="utf-8")
        for extra in extras:
            assert f"[{extra}]" in text, f"{context}: does not install the {extra!r} extra"


def test_the_arm64_pcaspy_wheel_is_the_version_the_lock_holds():
    """The VA's native-wheels stage builds pcaspy at PCASPY_VERSION and the
    virtual-accelerator layer takes it from find-links. On a source build that
    resolve is held to uv.lock, so a pin the lock moved and this ARG did not
    would leave the constrained install no candidate on arm64."""
    import tomllib

    va = _dockerfile("virtual_accelerator")
    pinned = re.search(r"^ARG PCASPY_VERSION=(\S+)$", va, flags=re.MULTILINE)
    assert pinned, "the VA recipe no longer pins PCASPY_VERSION"
    lock = tomllib.loads(
        (pathlib.Path(__file__).resolve().parents[2] / "uv.lock").read_text(encoding="utf-8")
    )
    locked = {p["version"] for p in lock["package"] if p["name"] == "pcaspy"}
    assert locked == {pinned.group(1)}, (
        f"PCASPY_VERSION={pinned.group(1)} but uv.lock resolves pcaspy {locked}"
    )
