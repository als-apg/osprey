"""Tests for the generated reference Dockerfile and .dockerignore.

`osprey build` renders a reference container recipe (Dockerfile +
.dockerignore) into every project root. These tests pin:

- content invariants (base image, port, project path, the 3-ARG site
  extension contract),
- the security-critical .dockerignore entries (secrets never enter the image),
- that `osprey claude regen` never touches the Dockerfile (it is rendered
  once and then owned by the user), and
- the anti-drift guard: every `osprey <cmd>` invocation inside the rendered
  Dockerfile must resolve against the real click command tree, so renaming
  or removing a CLI command/flag fails these tests instead of silently
  shipping a broken recipe.
"""

import json
import re
import shlex
import subprocess

import click
import pytest
from click.testing import CliRunner

from osprey.cli.main import cli

# The site-extension contract: exactly these build ARGs, with these defaults.
EXPECTED_ARGS = {
    "OSPREY_PIP_SPEC": "osprey-framework",
    "PIP_NO_PROXY": "",
    "OSPREY_OFFLINE": "0",
}


@pytest.fixture(scope="module")
def hello_project(tmp_path_factory):
    """Build a hello-world preset project once for content checks."""
    out = tmp_path_factory.mktemp("dockerfile-tpl")
    result = CliRunner().invoke(
        cli,
        [
            "build",
            "hello-docker",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    return out / "hello-docker"


@pytest.fixture(scope="module")
def deps_project(tmp_path_factory):
    """Build a project from a profile with pip dependencies."""
    out = tmp_path_factory.mktemp("dockerfile-deps")
    profile = out / "profile.yml"
    profile.write_text("extends: hello-world\ndependencies:\n  - numpy\n  - pydantic>=2\n")
    result = CliRunner().invoke(
        cli,
        [
            "build",
            "deps-docker",
            str(profile),
            "--skip-deps",
            "--skip-lifecycle",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    return out / "deps-docker"


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

    @staticmethod
    def _pip_install_lines(text: str) -> list[str]:
        return [
            line
            for line in text.splitlines()
            if "pip install" in line and not line.strip().startswith("#")
        ]

    @staticmethod
    def _spec_and_wheel_lines(pip_lines: list[str]) -> tuple[str, str]:
        """Split the wheel-drop install's two pip branches.

        The OSPREY install is a ``--dev`` wheel-drop conditional: a local-wheel
        branch (``/tmp/ctx/*.whl``) and the default ``$OSPREY_PIP_SPEC`` branch.
        Both carry the same profile-dependency args.
        """
        assert len(pip_lines) == 2, f"expected wheel + spec pip branches, got {pip_lines}"
        spec_line = next(line for line in pip_lines if "$OSPREY_PIP_SPEC" in line)
        wheel_line = next(line for line in pip_lines if "*.whl" in line)
        return spec_line, wheel_line

    def test_pip_line_has_no_deps_for_hello_world(self, hello_project):
        """hello-world has no profile dependencies — bare OSPREY install on both
        branches of the wheel-drop conditional."""
        pip_lines = self._pip_install_lines((hello_project / "Dockerfile").read_text())
        spec_line, wheel_line = self._spec_and_wheel_lines(pip_lines)
        # Nothing but the shell continuation follows the OSPREY spec (no deps).
        assert re.search(r'"\$OSPREY_PIP_SPEC"\s*;?\s*\\?\s*$', spec_line), spec_line
        assert re.search(r"/tmp/ctx/\*\.whl\s*;?\s*\\?\s*$", wheel_line), wheel_line

    def test_profile_dependencies_on_pip_line(self, deps_project):
        """Profile pip deps appear shlex-quoted after the OSPREY spec on BOTH
        branches of the wheel-drop conditional (dev wheel and PyPI spec)."""
        pip_lines = self._pip_install_lines((deps_project / "Dockerfile").read_text())
        spec_line, wheel_line = self._spec_and_wheel_lines(pip_lines)
        for line in (spec_line, wheel_line):
            assert "numpy" in line
            assert "'pydantic>=2'" in line, "version-constrained dep must be shell-quoted"
        # Deps must trail the OSPREY spec token, not precede it.
        assert "numpy" in spec_line.split('"$OSPREY_PIP_SPEC"', 1)[1]

    @staticmethod
    def _run_command_bodies(text: str) -> list[str]:
        """Every RUN instruction's shell body, with line-continuations joined."""
        joined = re.sub(r"\\\n", " ", text)
        return re.findall(r"^RUN (.+)$", joined, flags=re.MULTILINE)

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


class TestDockerignore:
    """Security-critical exclusions."""

    def test_secrets_and_host_state_excluded(self, hello_project):
        entries = {
            line.strip()
            for line in (hello_project / ".dockerignore").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        for required in (".env", ".venv", ".git", "_agent_data/"):
            assert required in entries, f"{required} missing from .dockerignore"
        # .env.example is safe and useful inside the image — must NOT be excluded
        assert ".env.example" not in entries


class TestRegenOwnership:
    """The Dockerfile is rendered once by build; regen never touches it."""

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
        # Sanity: the template contains at least regen, vendor fetch, and web
        assert len(invocations) >= 3, f"expected >=3 osprey calls, got: {invocations}"
        for args in invocations:
            _assert_resolves_in_cli(args)

    def test_guard_catches_unknown_flag(self):
        """The guard itself must fail on a bogus flag (meta-test)."""
        with pytest.raises(AssertionError, match="unknown flag"):
            _assert_resolves_in_cli(["claude", "regen", "--no-such-flag"])

    def test_guard_catches_unknown_command(self):
        with pytest.raises(AssertionError, match="unknown command"):
            _assert_resolves_in_cli(["claude", "regenerate-everything"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
