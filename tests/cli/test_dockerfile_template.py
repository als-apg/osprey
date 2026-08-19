"""Tests for the generated reference Dockerfile and .dockerignore.

`osprey build` renders a reference container recipe (Dockerfile +
.dockerignore) into every project root. These tests pin:

- content invariants (base image, port, project path, the 3-ARG site
  extension contract),
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
from tests.deployment._proxy_idiom import assert_apt_runs_carry_proxy_idiom

# The site-extension contract: exactly these quoted build ARGs, with these
# defaults. (CLAUDE_CLI_VERSION is rendered without quotes and is not captured.)
EXPECTED_ARGS = {
    "OSPREY_PIP_SPEC": "osprey-framework",
    "OSPREY_DEV": "",
    "PIP_NO_PROXY": "",
    "OSPREY_OFFLINE": "0",
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
        assert "EXPOSE 8087" in text
        assert "/app/hello-docker/" in text
        assert "WORKDIR /app/hello-docker" in text

    def test_arg_contract(self, hello_project):
        """Exactly the 3 contract ARGs, with the documented defaults."""
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
            "curl git procps ca-certificates nodejs npm" in node
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
