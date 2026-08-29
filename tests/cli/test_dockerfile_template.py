"""Tests for the generated reference Dockerfile and .dockerignore.

`osprey build` renders a reference container recipe (Dockerfile +
.dockerignore) into every project root. These tests pin:

- content invariants (base image, port, project path, the site-extension
  ARG contract),
- the security-critical .dockerignore entries (secrets never enter the image),
- the pairing between the repo's hatch exclude list and its .gitignore (what
  must not ship has to be kept out of both),
- that a Claude Code regeneration never touches the Dockerfile (it is
  rendered by `osprey build` and owned by the user in between), and
- the anti-drift guard: every `osprey <cmd>` invocation inside the rendered
  Dockerfile must resolve against the real click command tree, so renaming
  or removing a CLI command/flag fails these tests instead of silently
  shipping a broken recipe.
"""

import fnmatch
import json
import os
import pathlib
import re
import shlex
import subprocess
import tomllib

import click
import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.port_layout import DEFAULT_PORT_BASE, default_port, layout_ports
from tests.deployment._proxy_idiom import assert_apt_runs_carry_proxy_idiom

# The site-extension contract: exactly these quoted build ARGs, with these
# defaults. (CLAUDE_CLI_VERSION is rendered without quotes and is not captured.)
EXPECTED_ARGS = {
    "OSPREY_PIP_SPEC": "osprey-framework",
    "OSPREY_DEV": "",
    "PIP_NO_PROXY": "",
    "OSPREY_OFFLINE": "0",
    "OSPREY_SITE_CA": "",
}


def _render(repo: pathlib.Path, preset: str, *set_pairs: str) -> pathlib.Path:
    """Materialize a deployment repo from *preset* and render its build/ zone.

    Returns the render — ``<repo>/build`` — because that is where the Dockerfile
    lands: it is derived output like everything else the build produces.
    """
    args = ["init", str(repo), "--preset", preset, "--no-git"]
    for pair in set_pairs:
        args += ["--set", pair]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(
        cli, ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]
    )
    assert result.exit_code == 0, result.output
    return repo / "build"


@pytest.fixture(scope="module")
def hello_project(tmp_path_factory):
    """Render a hello-world deployment once, for the content checks."""
    return _render(tmp_path_factory.mktemp("dockerfile-tpl") / "hello-docker", "hello-world")


@pytest.fixture(scope="module")
def deps_project(tmp_path_factory):
    """The same, for a deployment whose profile declares pip dependencies."""
    return _render(
        tmp_path_factory.mktemp("dockerfile-deps") / "deps-docker",
        "hello-world",
        "dependencies=[numpy, pydantic>=2]",
    )


class TestDockerfileContent:
    """Content invariants of the rendered Dockerfile."""

    def test_rendered_at_project_root(self, hello_project):
        assert (hello_project / "Dockerfile").exists()
        assert (hello_project / ".dockerignore").exists()

    def test_base_image_port_and_project_path(self, hello_project):
        text = (hello_project / "Dockerfile").read_text()
        assert "FROM python:3.12-slim" in text
        # The exposed port is the layout's `web` slot at the deployment's base,
        # not a literal — moving `deployment.port_base` moves this line.
        assert f"EXPOSE {default_port('web', base=DEFAULT_PORT_BASE)}" in text
        assert "/app/hello-docker/" in text
        assert "WORKDIR /app/hello-docker" in text

    def test_arg_contract(self, hello_project):
        """Exactly the contract ARGs, with the documented defaults."""
        text = (hello_project / "Dockerfile").read_text()
        declared = dict(re.findall(r'^ARG (\w+)="([^"]*)"$', text, flags=re.MULTILINE))
        assert declared == EXPECTED_ARGS

    def test_no_unrendered_jinja(self, hello_project):
        for name in ("Dockerfile", ".dockerignore"):
            text = (hello_project / name).read_text()
            assert "{{" not in text, f"unrendered Jinja in {name}"
            assert "{%" not in text, f"unrendered Jinja in {name}"

    @classmethod
    def _node_run_body(cls, text: str) -> str:
        """The single apt RUN that installs Node."""
        bodies = [b for b in cls._run_command_bodies(text) if "nodejs" in b]
        assert len(bodies) == 1, f"expected exactly one node-install RUN, got {len(bodies)}"
        return bodies[0]

    def test_node_comes_from_the_base_image_distro(self, hello_project):
        """Node and npm are Debian packages, installed in the same apt RUN as the
        agent's shell tools — one package provenance, and no network beyond the
        Debian mirror. The list is this image's own (curl/git/procps are runtime
        tools the agent's shell reaches for); the dispatch image installs a
        smaller one, so the two are deliberately not asserted as a shared list.
        """
        node = self._node_run_body((hello_project / "Dockerfile").read_text())
        assert (
            "apt-get install -y --no-install-recommends "
            "curl git procps ca-certificates gosu nodejs npm" in node
        ), node
        assert "rm -rf /var/lib/apt/lists/*" in node

    def test_no_third_party_node_apt_repo(self, hello_project):
        """No third-party apt repo, and none of the machinery one needs.

        Fetching a vendor's setup script and piping it to a shell adds a second
        package source to the image, a key to trust, and a network dependency
        that fails the build wherever that host is unreachable. Asserted over
        the whole file as an absence, because that is the shape of the
        regression: any of these tokens reappearing means the pipeline came
        back.
        """
        text = (hello_project / "Dockerfile").read_text()
        for token in ("nodesource", "gnupg", "setup_20.x", "apt-key"):
            assert token not in text, f"{token} is back in the Dockerfile — third-party Node repo"
        node = self._node_run_body(text)
        assert "| bash" not in node and "| sh" not in node, (
            "the node layer pipes a downloaded script into a shell"
        )

    def test_npm_survives_into_the_final_image(self, hello_project):
        """npm is a runtime dependency, not a build-time convenience.

        The agent is launched as ``npx -y @anthropic-ai/claude-code@<version>``
        (claude_launcher.py), so an apt cleanup that treats npm as build-only
        breaks the agent at run time, not at build time. The reason is pinned
        alongside the absence of any purge in this layer, so a future edit has
        to confront it.
        """
        text = (hello_project / "Dockerfile").read_text()
        node = self._node_run_body(text)
        assert "apt-get purge" not in node, "the node layer purges packages"
        assert "autoremove" not in node, "the node layer autoremoves packages"
        assert "npm is a runtime dependency, not a build-time convenience" in text

    def test_apt_runs_deliver_the_proxy_settings(self, hello_project):
        """Every apt-using RUN hands apt the proxy settings before it fetches.

        The rendered project image is the ninth recipe under the same rule as
        the eight on-disk ones in
        :mod:`tests.deployment.test_service_dockerfiles`; the idiom itself is
        spelled once, in :mod:`tests.deployment._proxy_idiom`.
        """
        assert_apt_runs_carry_proxy_idiom(
            (hello_project / "Dockerfile").read_text(), "rendered project template"
        )

    def test_cli_pin_is_its_own_layer(self, hello_project):
        """The pinned CLI install stays a separate RUN from the apt layer, so
        bumping ``CLAUDE_CLI_VERSION`` does not re-run apt."""
        text = (hello_project / "Dockerfile").read_text()
        cli_bodies = [
            b
            for b in self._run_command_bodies(text)
            if "npm install -g" in b and "@anthropic-ai/claude-code" in b
        ]
        assert len(cli_bodies) == 1, f"expected exactly one CLI-install RUN, got {len(cli_bodies)}"
        assert 'npm install -g "@anthropic-ai/claude-code@${CLAUDE_CLI_VERSION}"' in cli_bodies[0]
        assert "nodejs" not in cli_bodies[0], "the CLI pin shares a RUN with the apt install"

    @classmethod
    def _ca_run_body(cls, text: str) -> str:
        """The single site-CA RUN body: installs the staged CA gated on
        ``OSPREY_SITE_CA`` (a no-op when the ARG is unset)."""
        bodies = [b for b in cls._run_command_bodies(text) if "update-ca-certificates" in b]
        assert len(bodies) == 1, f"expected exactly one site-CA RUN, got {len(bodies)}"
        return bodies[0]

    def test_site_ca_copy_idiom_present(self, hello_project):
        """The site CA is staged via the guaranteed-sibling glob idiom — the
        always-present .dockerignore keeps the COPY from failing when nothing
        is staged."""
        text = (hello_project / "Dockerfile").read_text()
        assert re.search(
            r"^COPY \.dockerignore \*\.cr\[t\] \*\.pe\[m\] /tmp/ca-ctx/", text, flags=re.MULTILINE
        ), "missing `COPY .dockerignore *.cr[t] *.pe[m] /tmp/ca-ctx/` CA-staging idiom"

    def test_site_ca_layer_precedes_the_first_network_fetch(self, hello_project):
        """``update-ca-certificates`` must run before any RUN that fetches over
        HTTPS — a CA installed after the first apt/npm/pip fetch is a CA those
        fetches never trusted."""
        text = (hello_project / "Dockerfile").read_text()
        ca = self._ca_run_body(text)
        assert '[ -n "$OSPREY_SITE_CA" ]' in ca, "CA install must be gated on OSPREY_SITE_CA"
        assert "/usr/local/share/ca-certificates/" in ca
        assert text.index("update-ca-certificates") < text.index("apt-get update"), (
            "the site-CA layer must precede the first network fetch"
        )

    def test_site_ca_envs_cover_every_tool_family(self, hello_project):
        """Node, pip, Python ssl and requests each read a different variable;
        all four must point at the merged Debian bundle, so build and runtime
        fetches alike trust a staged site CA."""
        text = (hello_project / "Dockerfile").read_text()
        for var in ("NODE_EXTRA_CA_CERTS", "PIP_CERT", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            assert f"{var}=/etc/ssl/certs/ca-certificates.crt" in text, (
                f"{var} does not point at the system bundle"
            )

    def test_ca_run_unset_arg_is_noop_success(self, hello_project, tmp_path):
        """Empirical probe: with OSPREY_SITE_CA unset the RUN is a successful
        no-op that installs nothing and still cleans up the staged context."""
        ca = self._ca_run_body((hello_project / "Dockerfile").read_text())
        result, ctx, store = _probe_ca_body(ca, tmp_path, staged=False, arg="")
        assert result.returncode == 0, (
            f"no-CA site-CA layer must exit 0:\n{result.stdout}\n{result.stderr}"
        )
        assert not ctx.exists(), "staged context not cleaned up on the no-CA path"
        assert list(store.iterdir()) == [], "a CA was installed with the ARG unset"

    def test_ca_run_installs_the_named_file(self, hello_project, tmp_path):
        """Empirical probe: with the ARG naming a staged file, the RUN lands it
        in the trust-store directory under the fixed .crt name
        update-ca-certificates requires, then cleans up."""
        ca = self._ca_run_body((hello_project / "Dockerfile").read_text())
        result, ctx, store = _probe_ca_body(ca, tmp_path, staged=True, arg="site-ca.crt")
        assert result.returncode == 0, f"site-CA install failed:\n{result.stdout}\n{result.stderr}"
        assert (store / "osprey-site-ca.crt").exists()
        assert not ctx.exists()

    def test_ca_run_missing_staged_file_fails(self, hello_project, tmp_path):
        """Empirical probe: an ARG naming a file that was never staged must fail
        the build loudly — a silently skipped CA surfaces later as an opaque
        cert error in whichever fetch hits the proxy first."""
        ca = self._ca_run_body((hello_project / "Dockerfile").read_text())
        result, _, store = _probe_ca_body(ca, tmp_path, staged=False, arg="site-ca.crt")
        assert result.returncode != 0, (
            f"site-CA layer exited 0 despite the named file missing:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert list(store.iterdir()) == []

    @classmethod
    def _deps_run_body(cls, text: str) -> str:
        """The single deps-layer RUN body: primes ``$OSPREY_PIP_SPEC`` with the
        C toolchain installed and purged in the same layer."""
        bodies = [
            b
            for b in cls._run_command_bodies(text)
            if "build-essential" in b and "$OSPREY_PIP_SPEC" in b
        ]
        assert len(bodies) == 1, f"expected exactly one deps RUN, got {len(bodies)}"
        return bodies[0]

    @classmethod
    def _wheel_run_body(cls, text: str) -> str:
        """The single wheel-layer RUN body: force-reinstalls osprey from a
        staged ``*.whl`` (a no-op when none is present)."""
        bodies = [b for b in cls._run_command_bodies(text) if "--force-reinstall" in b]
        assert len(bodies) == 1, f"expected exactly one wheel RUN, got {len(bodies)}"
        return bodies[0]

    def test_deps_and_wheel_are_separate_layers(self, hello_project):
        """The wheel install (force-reinstall) must not share a RUN with the
        toolchain install/purge — they are distinct cache layers now."""
        text = (hello_project / "Dockerfile").read_text()
        deps = self._deps_run_body(text)
        wheel = self._wheel_run_body(text)
        assert deps is not wheel and deps != wheel
        # The toolchain lives only in the deps layer.
        assert "build-essential" in deps
        assert "build-essential" not in wheel
        # The wheel force-reinstall lives only in the wheel layer.
        assert "--force-reinstall" in wheel
        assert "--force-reinstall" not in deps

    def test_deps_run_primes_pip_spec_with_dev_fallback(self, hello_project):
        """The deps RUN installs ``$OSPREY_PIP_SPEC`` and carries the dev-gated
        fallback (warn + install unpinned) guarded on ``OSPREY_DEV=1``."""
        text = (hello_project / "Dockerfile").read_text()
        deps = self._deps_run_body(text)
        assert '"$OSPREY_PIP_SPEC"' in deps
        assert '[ "$OSPREY_DEV" = "1" ]' in deps, "fallback must be gated on OSPREY_DEV=1"
        assert "WARNING: pin unreleased, priming with latest" in deps
        # Fallback installs the unpinned package; non-dev path fails loudly.
        assert "pip install --no-cache-dir osprey-framework" in deps
        assert "exit 1" in deps

    def test_wheel_run_force_reinstalls_and_checks(self, hello_project):
        """The wheel RUN reinstalls osprey from the staged wheel with
        ``--no-deps --force-reinstall`` and validates with ``pip check``."""
        text = (hello_project / "Dockerfile").read_text()
        wheel = self._wheel_run_body(text)
        assert "--no-deps --force-reinstall" in wheel
        assert "pip check" in wheel
        assert "/tmp/ctx/*.whl" in wheel
        assert "rm -rf /tmp/ctx" in wheel

    def test_wheel_copy_idiom_present(self, hello_project):
        """The wheel is staged via ``COPY .dockerignore *.wh[l]`` — the
        always-present .dockerignore sibling keeps the COPY from failing when no
        wheel exists."""
        text = (hello_project / "Dockerfile").read_text()
        assert re.search(r"^COPY \.dockerignore \*\.wh\[l\] /tmp/ctx/", text, flags=re.MULTILINE), (
            "missing `COPY .dockerignore *.wh[l] /tmp/ctx/` wheel-staging idiom"
        )

    def test_deps_run_has_no_deps_for_hello_world(self, hello_project):
        """hello-world has no profile dependencies — the OSPREY spec is installed
        bare (only whitespace between the spec and the ``||`` fallback)."""
        deps = self._deps_run_body((hello_project / "Dockerfile").read_text())
        assert re.search(r'"\$OSPREY_PIP_SPEC" +\|\|', deps), deps

    def test_profile_dependencies_on_deps_run(self, deps_project):
        """Profile pip deps appear shlex-quoted after ``$OSPREY_PIP_SPEC`` in the
        deps RUN (both the primary install and the dev fallback carry them)."""
        deps = self._deps_run_body((deps_project / "Dockerfile").read_text())
        assert "numpy" in deps
        assert "'pydantic>=2'" in deps, "version-constrained dep must be shell-quoted"
        # Deps trail the OSPREY spec token, not precede it.
        assert "numpy" in deps.split('"$OSPREY_PIP_SPEC"', 1)[1]

    @staticmethod
    def _run_command_bodies(text: str) -> list[str]:
        """Every RUN instruction's shell body, with line-continuations joined."""
        joined = re.sub(r"\\\n", " ", text)
        return re.findall(r"^RUN (.+)$", joined, flags=re.MULTILINE)

    def test_wheel_run_propagates_install_failure(self, hello_project, tmp_path):
        """Empirical probe: a failing `pip` inside the wheel-layer RUN must fail
        the whole body — the trailing `rm -rf /tmp/ctx` cleanup must not swallow
        the exit status (`fi && rm`, never `fi; rm`). Otherwise a broken dev
        wheel would build a stale image silently."""
        wheel = self._wheel_run_body((hello_project / "Dockerfile").read_text())
        result, ctx = _probe_wheel_body(wheel, tmp_path, with_wheel=True)
        assert result.returncode != 0, (
            f"wheel-layer RUN exited 0 despite pip failing:\n{result.stdout}\n{result.stderr}"
        )
        assert ctx.exists(), "cleanup ran despite the install failing"

    def test_wheel_run_no_wheel_is_noop_success(self, hello_project, tmp_path):
        """Empirical probe: with no wheel staged the RUN is a successful no-op
        (the untaken `if` exits 0) and still cleans up the staged context."""
        wheel = self._wheel_run_body((hello_project / "Dockerfile").read_text())
        result, ctx = _probe_wheel_body(wheel, tmp_path, with_wheel=False)
        assert result.returncode == 0, (
            f"no-wheel wheel layer must exit 0:\n{result.stdout}\n{result.stderr}"
        )
        assert not ctx.exists(), "staged context not cleaned up on the no-wheel path"

    def test_manifest_copy_precedes_deps_run(self, hello_project):
        """The dev-staged local-requirements manifest is COPYed via the same
        guaranteed-sibling glob idiom as the wheel (never fails when absent),
        immediately before the deps RUN that installs it."""
        text = (hello_project / "Dockerfile").read_text()
        match = re.search(
            r"^COPY \.dockerignore osprey-local-requirements\.tx\[t\] /tmp/deps-ctx/$",
            text,
            flags=re.MULTILINE,
        )
        assert match, "missing the manifest COPY sibling idiom"
        deps_pos = text.index('pip install --no-cache-dir "$OSPREY_PIP_SPEC"')
        assert match.start() < deps_pos, "manifest COPY must precede the deps RUN"

    def test_deps_run_installs_manifest_before_toolchain_purge(self, hello_project):
        """The deps RUN conditionally installs the staged manifest AFTER the
        primer (including its dev fallback) and BEFORE the toolchain purge, so
        a native dep in the local delta still compiles; the staged context is
        removed in the same RUN's cleanup."""
        deps = self._deps_run_body((hello_project / "Dockerfile").read_text())
        install = "pip install --no-cache-dir -r /tmp/deps-ctx/osprey-local-requirements.txt"
        assert "&& if [ -f /tmp/deps-ctx/osprey-local-requirements.txt ]; then" in deps, (
            "manifest install missing or not &&-chained in the deps RUN"
        )
        assert install in deps
        assert deps.index(install) > deps.index("WARNING: pin unreleased"), (
            "manifest install must follow the primer's dev-fallback construct"
        )
        assert deps.index(install) < deps.index("apt-get purge -y build-essential"), (
            "manifest install must precede the toolchain purge"
        )
        assert "rm -rf /var/lib/apt/lists/* /tmp/deps-ctx" in deps, (
            "deps RUN must clean up /tmp/deps-ctx with the apt cleanup"
        )

    def test_deps_run_propagates_manifest_install_failure(self, hello_project, tmp_path):
        """Empirical probe: with a manifest staged and its `pip install -r`
        failing, the whole deps RUN body must exit nonzero — the conditional
        and the trailing cleanup must not swallow the failure."""
        deps = self._deps_run_body((hello_project / "Dockerfile").read_text())
        result, ctx = _probe_deps_body(deps, tmp_path, with_manifest=True)
        assert result.returncode != 0, (
            f"deps RUN exited 0 despite the manifest install failing:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert ctx.exists(), "cleanup ran despite the manifest install failing"

    def test_deps_run_no_manifest_is_success(self, hello_project, tmp_path):
        """Empirical probe: with no manifest staged the conditional is a no-op
        and the deps RUN succeeds, still cleaning up the staged context."""
        deps = self._deps_run_body((hello_project / "Dockerfile").read_text())
        result, ctx = _probe_deps_body(deps, tmp_path, with_manifest=False)
        assert result.returncode == 0, (
            f"no-manifest deps RUN must exit 0:\n{result.stdout}\n{result.stderr}"
        )
        assert not ctx.exists(), "staged context not cleaned up on the no-manifest path"

    def test_run_commands_are_valid_shell(self, hello_project, deps_project):
        """Every rendered RUN body must parse under ``/bin/sh -n``.

        A rendered-string assertion cannot catch a shell syntax error — e.g. an
        env-assignment prefix (``VAR=x``) placed before an ``if`` compound, which
        parses fine as text but only fails when a real ``docker build`` runs it.
        Parsing each RUN body with ``sh -n`` guards the wheel-drop conditional
        install (and every other RUN) without needing a container build.
        """
        for project in (hello_project, deps_project):
            for body in self._run_command_bodies((project / "Dockerfile").read_text()):
                result = subprocess.run(["sh", "-n", "-c", body], capture_output=True, text=True)
                assert result.returncode == 0, (
                    f"invalid shell syntax in a rendered RUN body:\n{body}\n{result.stderr}"
                )


def _probe_wheel_body(body: str, tmp_path, *, with_wheel: bool):
    """Execute the wheel-layer RUN body in a sandbox with a real shell.

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


def _probe_ca_body(body: str, tmp_path, *, staged: bool, arg: str):
    """Execute the site-CA RUN body in a sandbox with a real shell.

    ``/tmp/ca-ctx`` and the trust-store directory are rewritten to temp dirs
    and a no-op ``update-ca-certificates`` stub shadows any real one on PATH,
    so the probe exercises the gate, the install and the cleanup without a
    container. Returns ``(CompletedProcess, ctx_path, store_path)``.
    """
    ctx = tmp_path / "ca-ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    if staged:
        (ctx / "site-ca.crt").write_text("-----BEGIN CERTIFICATE-----\n")
    store = tmp_path / "ca-store"
    store.mkdir()
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    stub = stub_bin / "update-ca-certificates"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        OSPREY_SITE_CA=arg,
    )
    rewritten = body.replace("/tmp/ca-ctx", str(ctx)).replace(
        "/usr/local/share/ca-certificates", str(store)
    )
    result = subprocess.run(["sh", "-c", rewritten], capture_output=True, text=True, env=env)
    return result, ctx, store


def _probe_deps_body(body: str, tmp_path, *, with_manifest: bool):
    """Execute the deps-layer RUN body in a sandbox with a real shell.

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
        OSPREY_PIP_SPEC="osprey-framework",
        OSPREY_DEV="",
        PIP_NO_PROXY="",
    )
    rewritten = body.replace("/tmp/deps-ctx", str(ctx)).replace(
        "/var/lib/apt/lists", str(apt_lists)
    )
    result = subprocess.run(["sh", "-c", rewritten], capture_output=True, text=True, env=env)
    return result, ctx


class TestDockerignore:
    """Security-critical exclusions."""

    @staticmethod
    def _entries(project) -> set[str]:
        return {
            line.strip()
            for line in (project / ".dockerignore").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

    def test_secrets_and_host_state_excluded(self, hello_project):
        entries = self._entries(hello_project)
        # Secrets and host state are excluded. The env exclusion is a glob:
        # `.env` alone let the deploy-generated `.env.users` into the
        # image. `var/` is the STATE zone the container mounts its own volume
        # over, so the host's copy is dead weight in the image.
        for required in (".env*", ".venv", ".git", "var/"):
            assert required in entries, f"{required} missing from .dockerignore"
        # .env.example is safe and useful inside the image — must be re-included
        assert "!.env.example" in entries

    def test_the_render_itself_is_not_excluded(self, hello_project):
        """``build/`` must NOT be ignored — it is the deployment being shipped.

        The render IS the deployment, and the project image's context is a
        deployment repo, so an image built with ``build/`` ignored would carry
        no config.yml, no .mcp.json and no Claude Code artifacts — and would
        fail only at runtime, as an agent with nothing configured.

        Asserted as an absence, which is the shape of the mistake: this is the
        one entry whose presence is the bug.
        """
        assert "build/" not in self._entries(hello_project)

    def test_auth_and_production_env_are_excluded(self, hello_project):
        """Named regression guard for the two files that carry live credentials.

        ``.env.auth`` holds the web-terminal password hashes and the sidecar's
        session-signing secrets; ``.env.users`` holds the provider secrets
        a multi-user deploy generates at the project root — the same root the
        persona images are then built from, so an unexcluded one is baked into
        every agent image.

        Asserted by **matching** rather than by literal line, and separately
        from :meth:`test_secrets_and_host_state_excluded`, which pins the
        template's own spelling. Today one ``.env*`` glob covers both; a future
        rewrite to explicit entries, or to a narrower pattern, keeps this test
        meaningful either way, and the names stay written down where the reason
        for excluding them is.

        Last-match-wins with negation, the way both git and Docker resolve it —
        a later ``!.env.auth`` would re-expose the file, so polarity is carried
        through the whole scan rather than returning on the first hit. That
        makes ORDER load-bearing, which is why this reads the file's lines
        directly instead of reusing :meth:`_entries`: that helper returns a
        *set*, and resolving a last-match rule over an unordered collection
        would decide ``.env.example``'s fate at random.
        """
        lines = [
            line.strip()
            for line in (hello_project / ".dockerignore").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

        def _ignored(name: str) -> bool:
            ignored = False
            for entry in lines:
                pattern = entry[1:] if entry.startswith("!") else entry
                if fnmatch.fnmatch(name, pattern.strip("/")):
                    ignored = not entry.startswith("!")
            return ignored

        for secret in (".env.auth", ".env.users"):
            assert _ignored(secret), f"{secret} would be baked into the image"
        # Positive control: the negation that keeps the documented variable
        # list in the image must still win over the glob that covers it.
        assert not _ignored(".env.example"), ".env.example should stay in the image"

    def test_dockerfile_excluded_but_not_dockerignore(self, hello_project):
        """The wheel layer's ``COPY .dockerignore *.wh[l]`` needs .dockerignore to
        be a guaranteed-present sibling, so it must NOT self-exclude. The
        Dockerfile itself is still excluded (the image needs no build recipe)."""
        entries = self._entries(hello_project)
        assert "Dockerfile" in entries
        assert ".dockerignore" not in entries, (
            ".dockerignore must not self-exclude — the wheel-staging COPY relies on it"
        )

    def test_staged_models_excluded(self, hello_project):
        """Model files staged by hand are qmd build inputs, not image content.

        Pinned with its ``**/`` prefix, which is the one deviation from the
        root-anchored spelling the rest of the list uses: the staged directory
        sits beside the qmd service's Dockerfile inside the render, so a
        root-anchored pattern would name nothing and gigabytes of GGUF would
        ride into the project image.
        """
        assert "**/prefetched-models/" in self._entries(hello_project)


class TestPackagingExcludePairing:
    """The hatch exclude list and .gitignore are one list, spelled twice.

    Hatchling packages the working tree rather than what git tracks, so a
    directory that must never ship needs both halves: the
    ``[tool.hatch.build]`` exclude keeps it out of the sdist and wheel, the
    ``.gitignore`` entry keeps it out of the repo. Either half alone leaves a
    way in, and the failure is quiet at the moment it is introduced — a staged
    model directory that was gitignored but not excluded produced a
    multi-gigabyte wheel, and nothing said so until someone installed it.

    Asserted as a pairing over the whole list rather than entry by entry, so a
    future exclude added with no ``.gitignore`` sibling fails here too.
    """

    @staticmethod
    def _repo_root() -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[2]

    @classmethod
    def _hatch_excludes(cls) -> list[str]:
        with (cls._repo_root() / "pyproject.toml").open("rb") as handle:
            return tomllib.load(handle)["tool"]["hatch"]["build"]["exclude"]

    @classmethod
    def _gitignore_entries(cls) -> set[str]:
        text = (cls._repo_root() / ".gitignore").read_text(encoding="utf-8")
        return {
            line.strip().rstrip("/")
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        }

    def test_every_hatch_exclude_is_gitignored(self):
        # Compared with the trailing slash normalized away: .gitignore spells a
        # directory `foo/` and hatch spells it `foo`, and the pairing is about
        # the path, not the punctuation.
        gitignored = self._gitignore_entries()
        for pattern in self._hatch_excludes():
            assert pattern.rstrip("/") in gitignored, (
                f"{pattern} is excluded from the build but not gitignored — "
                "add the matching .gitignore entry"
            )

    def test_staged_models_are_excluded_from_the_distribution(self):
        """Named guard for the pair that motivated the rule."""
        assert "**/prefetched-models" in self._hatch_excludes()
        assert "**/prefetched-models" in self._gitignore_entries()


class TestRegenOwnership:
    """The Dockerfile is build output; a Claude Code regeneration never touches it."""

    def test_regen_never_touches_dockerfile(self, hello_project):
        from osprey.cli.templates.manager import TemplateManager

        dockerfile = hello_project / "Dockerfile"
        original = dockerfile.read_text()
        try:
            dockerfile.write_text("# USER-CUSTOMIZED RECIPE\n")
            result = TemplateManager().regenerate_claude_code(hello_project)

            assert dockerfile.read_text() == "# USER-CUSTOMIZED RECIPE\n"
            touched = set(result["changed"]) | set(result["unchanged"])
            assert "Dockerfile" not in touched
            assert not any("Dockerfile" in f for f in touched)
        finally:
            dockerfile.write_text(original)


# ── The container privilege split ────────────────────────────────────────────


def _instructions(dockerfile_text: str) -> list[tuple[str, str]]:
    """``(instruction, body)`` for every Dockerfile instruction, in file order.

    Line continuations are joined first, so a multi-line ``RUN`` is one entry
    and the sequence is the LAYER sequence — which is what the cache-order
    assertions below are actually about.
    """
    joined = re.sub(r"\\\n", " ", dockerfile_text)
    out: list[tuple[str, str]] = []
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if parts[0].isupper() and len(parts[0]) > 1:
            out.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return out


def _render_template(**context) -> str:
    """Render ``Dockerfile.j2`` directly, against a synthetic context.

    The persona-conditional line depends on a context key whose real value comes
    from the profile's permission overrides, and no shipped preset denies the
    setup tool yet (the base floor lands with the tiered presets). Rendering the
    template itself is how both sides of the conditional can be pinned now,
    rather than only the side today's presets happen to produce.
    """
    from osprey.cli.templates.manager import TemplateManager

    context.setdefault("project_name", "synthetic")
    context.setdefault("claude_code_cli_version", "1.2.3")
    # EXPOSE/CMD render from the layout table the real render builds in
    # TemplateManager._project_context.
    context.setdefault("port_base", DEFAULT_PORT_BASE)
    context.setdefault("osprey_ports", layout_ports(DEFAULT_PORT_BASE))
    template = TemplateManager().jinja_env.get_template("project/Dockerfile.j2")
    return template.render(**context)


class TestPrivilegeSplit:
    """The image starts as root, drops to `osprey`, and owns the render itself.

    The rendered project is what decides what the agent may do — ``config.yml``,
    ``.mcp.json``, the ``.claude/`` artifacts — so the image leaves it
    root-owned and hands the agent's user only the mutable state. These pin the
    parts of that split that live in the Dockerfile; the entrypoint's own
    behaviour is pinned by ``tests/cli/test_entrypoint_script.py``.
    """

    def test_gosu_is_installed_in_the_apt_layer(self, hello_project):
        """The privilege-drop tool ships in the layer that changes least.

        Not beside the entrypoint that uses it: an apt install in the tail of
        the file would re-run on every render change, and the entrypoint refuses
        to start without gosu rather than running the agent as root.
        """
        node = TestDockerfileContent._node_run_body((hello_project / "Dockerfile").read_text())
        assert " gosu " in node, node

    def test_no_user_instruction(self, hello_project):
        """A ``USER`` line would make the entrypoint's root-only startup steps
        silent no-ops — it is the entrypoint that drops privileges here."""
        text = (hello_project / "Dockerfile").read_text()
        assert re.search(r"^USER\s", text, flags=re.MULTILINE) is None, (
            "USER instruction present; the entrypoint's gosu drop is the only privilege change"
        )

    def test_entrypoint_names_the_rendered_script(self, hello_project):
        """JSON-array form (no shell wrapper between PID 1 and the script), and
        the path is where the render's own copy lands inside the image."""
        text = (hello_project / "Dockerfile").read_text()
        entrypoints = [body for instr, body in _instructions(text) if instr == "ENTRYPOINT"]
        assert len(entrypoints) == 1, entrypoints
        assert json.loads(entrypoints[0]) == ["/app/hello-docker/build/entrypoint.sh"]

    def test_cmd_is_still_the_web_server(self, hello_project):
        """CMD survives as the entrypoint's ``"$@"``, so the command the image
        runs is unchanged and an override still goes through the drop."""
        text = (hello_project / "Dockerfile").read_text()
        cmds = [body for instr, body in _instructions(text) if instr == "CMD"]
        assert len(cmds) == 1
        assert json.loads(cmds[0])[:2] == ["osprey", "web"]

    def test_runtime_posture_envs(self, hello_project):
        """Both markers the runtime reads out of the image."""
        text = (hello_project / "Dockerfile").read_text()
        assert "ENV OSPREY_RENDER_ZONE_READONLY=1" in text
        assert "ENV OSPREY_RUNTIME_UID=1000:1000" in text

    def test_declared_runtime_uid_matches_the_user_it_creates(self, hello_project):
        """The declared pair is not a second, independently-editable fact.

        Host-side per-user volume seeding chowns to whatever
        ``OSPREY_RUNTIME_UID`` says (deployment/web_terminals/seeding.py), so a
        drift between it and the account the image creates would silently chown
        every seeded volume to a user that does not exist in the image.
        """
        text = (hello_project / "Dockerfile").read_text()
        declared = re.search(r"^ENV OSPREY_RUNTIME_UID=(\d+):(\d+)$", text, flags=re.MULTILINE)
        assert declared, "OSPREY_RUNTIME_UID must be declared as numeric uid:gid"
        user_run = [b for _, b in _instructions(text) if "useradd" in b]
        assert len(user_run) == 1, user_run
        assert f"--uid {declared.group(1)}" in user_run[0]
        assert f"--gid {declared.group(2)}" in user_run[0]
        assert f"groupadd --gid {declared.group(2)}" in user_run[0]

    def test_state_directories_are_precreated(self, hello_project):
        """Every directory the unprivileged user writes to at runtime exists in
        the image before the chown — one created later is created root-owned."""
        text = (hello_project / "Dockerfile").read_text()
        mkdir = [b for _, b in _instructions(text) if b.startswith("mkdir")]
        assert len(mkdir) == 1, mkdir
        for relpath in (
            "var/agent_data/api_calls",
            "var/agent_data/config-backups",
            "var/audit",
        ):
            assert f"/app/hello-docker/{relpath}" in mkdir[0], relpath

    def test_chown_is_narrowed_to_the_mutable_state(self, hello_project):
        """The blanket `chown -R` of the whole project is gone.

        That single line is what used to hand the agent its own render. What
        replaces it is `var/` plus the knowledge bundle — and the bundle chown
        is guarded, because `osprey build` renders that directory only for a
        deployment that names a bundle path.
        """
        text = (hello_project / "Dockerfile").read_text()
        bodies = [b for _, b in _instructions(text) if "chown" in b]
        assert bodies, "no chown at all — the state zone would be unwritable"
        for body in bodies:
            assert "chown -R osprey:osprey /app/hello-docker " not in body + " ", (
                f"blanket chown of the whole project is back: {body}"
            )
        joined = "\n".join(bodies)
        assert "chown -R osprey:osprey /app/hello-docker/var" in joined
        assert "if [ -d /app/hello-docker/build/data/facility_knowledge ]" in joined
        assert "chown -R osprey:osprey /app/hello-docker/build/data/facility_knowledge" in joined

    def test_render_zone_is_not_chowned(self, hello_project):
        """Nothing in the render is handed to the agent's user — the shipped
        presets leave the setup capability in place today, so `config.yml` IS
        chowned; every other path under `build/` is not."""
        text = (hello_project / "Dockerfile").read_text()
        chowned = "\n".join(b for _, b in _instructions(text) if "chown" in b)
        for reserved in (".claude", ".mcp.json", "CLAUDE.md"):
            assert reserved not in chowned, f"{reserved} must stay root-owned"


class TestSetupCapabilityConditional:
    """`build/config.yml` is chowned only for a persona that can still edit it.

    The tier boundary is a filesystem fact rather than a permission list: a
    readonly or readwrite render cannot write the file that says what it may do,
    and the admin tier — whose whole purpose is the Config panel — can. These
    render the template against both contexts directly, which is what pins the
    conditional itself; `TestTieredPresetConfigChown` below asks the same
    question of the real `control-assistant` presets that produce each side.
    """

    #: The one line the conditional renders. Matched exactly rather than by the
    #: filename, which the COPY layer's prose mentions for unrelated reasons.
    _CHOWN = "chown osprey:osprey /app/synthetic/build/config.yml"

    def test_no_config_chown_without_the_capability(self):
        text = _render_template(is_setup_patch_capable=False)
        assert self._CHOWN not in text, (
            "config.yml handed to the agent's user for a persona that cannot patch it"
        )

    def test_config_chown_with_the_capability(self):
        text = _render_template(is_setup_patch_capable=True)
        assert f"RUN {self._CHOWN}" in text

    def test_absent_context_key_fails_closed(self):
        """A context that never sets the key renders the SAFE side.

        The key is added by one call site; a render path that forgets it must
        produce a locked-down image rather than a quietly writable config.
        """
        text = _render_template()
        assert self._CHOWN not in text

    def test_both_override_spellings_are_read(self):
        """Dotted and nested reach the same rendered key, so both must count.

        The presets write dotted overrides and say so in a comment, but a
        nested `claude_code:` mapping lands on the same key in `config.yml`. A
        privilege check that saw only one spelling would call a profile capable
        no matter what the other spelling denied.
        """
        from osprey.cli.build_cmd import _profile_setup_patch_capable

        tool = "mcp__osprey_workspace__setup_patch"

        class _Profile:
            def __init__(self, config):
                self.config = config

        nested_deny = _Profile({"claude_code": {"permissions": {"deny": [tool]}}})
        assert _profile_setup_patch_capable(nested_deny) is False

        nested_lift = _Profile(
            {"claude_code": {"permissions": {"deny": [tool], "remove_deny": [tool]}}}
        )
        assert _profile_setup_patch_capable(nested_lift) is True

        # A block spelling one `claude_code` path two ways cannot be built —
        # parsing it is refused by
        # `build_profile_load._reject_mixed_claude_code_spellings`, because
        # `config_update_fields` would discard one of the two silently (pinned
        # in tests/cli/test_build_profile.py). The helper still reads every
        # spelling, which is what makes its union exact rather than merely
        # broad: each list is at most one spelling's value.
        mixed = _Profile(
            {
                "claude_code.permissions.deny": [tool],
                "claude_code": {"permissions": {"remove_deny": [tool]}},
            }
        )
        assert _profile_setup_patch_capable(mixed) is True

    @pytest.mark.parametrize(
        "spelling",
        [
            pytest.param(lambda key, value: {f"claude_code.permissions.{key}": value}, id="dotted"),
            pytest.param(
                lambda key, value: {"claude_code.permissions": {key: value}}, id="middle-split"
            ),
            pytest.param(
                lambda key, value: {"claude_code": {"permissions": {key: value}}}, id="nested"
            ),
        ],
    )
    @pytest.mark.parametrize("key,capable", [("deny", False), ("remove_deny", True)])
    def test_the_container_and_the_guard_read_every_spelling_the_same(
        self, spelling, key: str, capable: bool
    ):
        """One reader for one question, across all three split points.

        `claude_code.permissions.deny` can be written as one dotted key, as a
        `claude_code.permissions:` mapping, or fully nested, and all three
        render the same leaf. This helper used to read the first and the third:
        a profile that denied the setup tool through the middle one was reported
        CAPABLE and the image chowned `build/config.yml` to a persona whose
        settings.json denies the tool. Both sides now come from
        `personas.persona_capability_document`, so the two verdicts are the same
        verdict.
        """
        from osprey.cli.build_cmd import _profile_setup_patch_capable
        from osprey.cli.profile_conventions import is_setup_patch_capable
        from osprey.deployment.web_terminals.personas import persona_capability_document

        tool = "mcp__osprey_workspace__setup_patch"
        config = {"claude_code.permissions.deny": [tool]} if key == "remove_deny" else {}
        config.update(spelling(key, [tool]))

        class _Profile:
            def __init__(self, config):
                self.config = config

        guard_verdict = is_setup_patch_capable(persona_capability_document(config))
        assert _profile_setup_patch_capable(_Profile(config)) is guard_verdict is capable

    def test_the_floor_does_not_deny_the_setup_tool(self):
        """The parity this whole derivation rests on.

        The capability check reads the PROFILE's deny; it never sees the
        framework's `DENY_DEFAULTS` floor. While the floor is silent about the
        setup tool the two agree — and if the deny ever moved down into the
        floor, every persona would read as capable and every image would chown
        its config.yml to the agent.
        """
        from fnmatch import fnmatchcase

        from osprey.cli.profile_conventions import SETUP_PATCH_TOOL
        from osprey.cli.templates.claude_code import DENY_DEFAULTS

        assert not [e for e in DENY_DEFAULTS if fnmatchcase(SETUP_PATCH_TOOL, e)]

    def test_the_build_refuses_a_floor_that_denies_the_setup_tool(self, monkeypatch):
        """And the build says so rather than shipping the widened chown."""
        from osprey.cli.templates import claude_code
        from osprey.errors import BuildProfileError

        monkeypatch.setattr(
            claude_code, "DENY_DEFAULTS", (*claude_code.DENY_DEFAULTS, "mcp__osprey_workspace__*")
        )
        with pytest.raises(BuildProfileError, match="chown build/config.yml"):
            claude_code._lint_write_tools_are_gated({}, [])

    def test_capability_is_derived_from_the_profile_permissions(self):
        """The value the real render context carries, over the two spellings a
        profile uses: a tier adds the deny, and a higher tier lifts it with
        `remove_deny`. Reading the deny alone would call the admin tier denied.
        """
        from osprey.cli.build_cmd import _profile_setup_patch_capable

        class _Profile:
            def __init__(self, config):
                self.config = config

        tool = "mcp__osprey_workspace__setup_patch"
        assert _profile_setup_patch_capable(_Profile({})) is True
        assert (
            _profile_setup_patch_capable(_Profile({"claude_code.permissions.deny": [tool]}))
            is False
        )
        assert (
            _profile_setup_patch_capable(
                _Profile(
                    {
                        "claude_code.permissions.deny": [tool],
                        "claude_code.permissions.remove_deny": [tool],
                    }
                )
            )
            is True
        )
        assert (
            _profile_setup_patch_capable(
                _Profile({"claude_code.permissions.deny": ["mcp__osprey_workspace__*"]})
            )
            is False
        )


@pytest.fixture(scope="module")
def tiered_render(tmp_path_factory):
    """Render the `control-assistant` preset once — the deployment's own project
    plus one per persona delta, all from a single `osprey build`.

    Module-scoped: the tier question below is asked of five Dockerfiles out of
    the same render, and rendering per test would pay for the same build five
    times over.
    """
    return _render(tmp_path_factory.mktemp("dockerfile-tiers") / "ca-docker", "control-assistant")


class TestTieredPresetConfigChown:
    """The conditional above, asked of the presets that actually ship.

    `TestSetupCapabilityConditional` pins both sides of the template branch
    against a synthetic context; this pins that the real preset family lands one
    tier on each side. It is the end of the chain the tiered presets exist for:
    the base preset denies `setup_patch`, `control-assistant-admin` lifts that
    deny with `remove_deny`, `_profile_setup_patch_capable` composes the two into
    the render context, and the difference comes out here as which image hands
    `build/config.yml` to the agent's user.

    Asserted on the rendered Dockerfiles rather than on the context value,
    because the file is what a `docker build` reads.
    """

    @staticmethod
    def _chown_line(project_name: str) -> str:
        return f"RUN chown osprey:osprey /app/{project_name}/build/config.yml"

    @staticmethod
    def _dockerfile(render: pathlib.Path, project_name: str) -> str:
        """The Dockerfile of one project in the shared render.

        The deployment's own project IS the render directory; each persona's
        sits beside it under the name `osprey init` wrote into the catalog,
        `<repo>-<persona>`.
        """
        root = render if project_name == render.parent.name else render / project_name
        dockerfile = root / "Dockerfile"
        assert dockerfile.exists(), sorted(p.name for p in render.iterdir())
        return dockerfile.read_text()

    def test_admin_render_hands_config_yml_to_the_agent(self, tiered_render):
        """The admin tier keeps the setup capability, so the one file that
        capability exists to edit is chowned to the user the entrypoint drops
        to. Without this line the Config panel and `setup_patch` both fail on a
        root-owned file at runtime — the tier would be admin in name only."""
        project = f"{tiered_render.parent.name}-admin"
        assert self._chown_line(project) in self._dockerfile(tiered_render, project)

    @pytest.mark.parametrize("persona", ["readonly", "readwrite", "ariel"])
    def test_no_other_persona_hands_over_config_yml(self, tiered_render, persona):
        """Every tier the base preset's floor still applies to renders the image
        WITHOUT the chown: it cannot rewrite the file that says what it may do,
        whatever else its permission list allows."""
        project = f"{tiered_render.parent.name}-{persona}"
        text = self._dockerfile(tiered_render, project)
        assert self._chown_line(project) not in text
        assert "/build/config.yml" not in "\n".join(
            b for _, b in _instructions(text) if "chown" in b
        )

    def test_the_hosting_deployment_does_not_hand_over_config_yml(self, tiered_render):
        """The base render is the single-user image every deployment starts
        from, and it sits at its own floor: the deny it declares is not lifted
        by anything, so its config.yml stays root-owned too."""
        project = tiered_render.parent.name
        text = self._dockerfile(tiered_render, project)
        assert self._chown_line(project) not in text
        assert "/build/config.yml" not in "\n".join(
            b for _, b in _instructions(text) if "chown" in b
        )


class TestLayerOrder:
    """A rebuild after a render change must not redo the expensive layers.

    Every project change — a config key, an artifact, a rule — lands in the
    `COPY .` of the deployment, so everything that fetches from a network has to
    sit above it. The privilege split adds instructions to the tail, which is
    exactly where they cost nothing; this pins that they stayed there.
    """

    #: Instructions whose cache key is the build context's content.
    _CONTEXT_FETCHERS = ("apt-get install", "npm install", "pip install", "osprey vendor fetch")

    def test_network_layers_precede_the_deployment_copy(self, hello_project):
        text = (hello_project / "Dockerfile").read_text()
        instructions = _instructions(text)
        copy_at = next(
            i for i, (instr, body) in enumerate(instructions) if body.startswith(". /app/")
        )
        for i, (instr, body) in enumerate(instructions):
            if any(fetch in body for fetch in self._CONTEXT_FETCHERS):
                assert i < copy_at, (
                    f"{instr} {body[:60]!r} fetches from the network after the deployment "
                    "COPY — every render change would re-run it"
                )

    def test_the_tail_after_the_copy_is_only_cheap_local_steps(self, hello_project):
        """What follows the COPY is metadata and two local filesystem steps.

        Named as an allow-list rather than a count so that adding an ENV or a
        LABEL stays free, while an install or a fetch sneaking into the tail
        fails here.
        """
        instructions = _instructions((hello_project / "Dockerfile").read_text())
        copy_at = next(
            i for i, (instr, body) in enumerate(instructions) if body.startswith(". /app/")
        )
        for instr, body in instructions[copy_at + 1 :]:
            assert instr in {"RUN", "ENV", "WORKDIR", "EXPOSE", "ENTRYPOINT", "CMD"}, instr
            if instr == "RUN":
                assert body.split()[0] in {"mkdir", "groupadd", "chown"}, body


# ── Anti-drift guard ─────────────────────────────────────────────────────────


def _extract_osprey_invocations(dockerfile_text: str) -> list[list[str]]:
    """Extract argv-after-`osprey` for every osprey call in the Dockerfile.

    Handles RUN shell-form (incl. `\\` continuations, `&&`/`;` compounds,
    `if ...; then osprey ...; fi`) and CMD/ENTRYPOINT JSON-array form.
    """
    text = dockerfile_text.replace("\\\n", " ")
    invocations: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in {"RUN", "CMD", "ENTRYPOINT"}:
            continue
        body = parts[1].strip()
        if body.startswith("["):
            token_groups = [json.loads(body)]
        else:
            token_groups = []
            for piece in re.split(r"&&|\|\||;", body):
                try:
                    token_groups.append(shlex.split(piece))
                except ValueError:
                    continue
        for tokens in token_groups:
            if "osprey" in tokens:
                i = tokens.index("osprey")
                invocations.append(tokens[i + 1 :])
    return invocations


def _assert_resolves_in_cli(args: list[str]) -> None:
    """Walk the real click tree: subcommand chain and flags must all exist."""
    ctx = click.Context(cli)
    cmd: click.Command = cli
    i = 0
    chain = ["osprey"]
    while i < len(args) and isinstance(cmd, click.Group):
        name = args[i]
        if name.startswith("-"):
            break
        sub = cmd.get_command(ctx, name)
        assert sub is not None, f"Dockerfile references unknown command: {' '.join(chain)} {name}"
        cmd = sub
        chain.append(name)
        i += 1

    valid_flags: set[str] = set()
    for param in cmd.params:
        valid_flags.update(getattr(param, "opts", []))
        valid_flags.update(getattr(param, "secondary_opts", []))

    for token in args[i:]:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            assert flag in valid_flags, (
                f"Dockerfile references unknown flag {flag} for `{' '.join(chain)}` "
                f"(valid: {sorted(valid_flags)})"
            )


class TestCliCrossCheck:
    """Every osprey invocation in the rendered Dockerfile must resolve."""

    def test_all_osprey_invocations_resolve(self, hello_project):
        text = (hello_project / "Dockerfile").read_text()
        invocations = _extract_osprey_invocations(text)
        # Sanity: the template still calls osprey at all. Two today — the
        # offline `vendor fetch` and the `web` entrypoint — and the floor is
        # what keeps a template that stopped calling osprey from passing this
        # by having nothing to check.
        assert len(invocations) >= 2, f"expected >=2 osprey calls, got: {invocations}"
        for args in invocations:
            _assert_resolves_in_cli(args)

    def test_guard_catches_unknown_flag(self):
        """The guard itself must fail on a bogus flag (meta-test)."""
        with pytest.raises(AssertionError, match="unknown flag"):
            _assert_resolves_in_cli(["vendor", "fetch", "--no-such-flag"])

    def test_guard_catches_unknown_command(self):
        with pytest.raises(AssertionError, match="unknown command"):
            _assert_resolves_in_cli(["vendor", "fetch-everything"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
