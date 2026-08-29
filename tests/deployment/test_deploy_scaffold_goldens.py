"""The deploy-scaffolding templates, rendered against a registry deployment.

``goldens/exemplar-profile/`` is a second reference deployment, and it earns its
place by being the one the feature's own exemplar is not: it names a registry,
declares a service with a Dockerfile, and therefore exercises the images stage
and the registry credential that the exemplar's ``image_source: local`` never
reaches.

The byte specification lives elsewhere: the three-zone exemplar in
``tests/fixtures/lifecycle_repo.py``, which
``tests/cli/test_emitted_artifacts_clean.py`` holds the templates to byte for
byte. A second, hand-built reference deployment beside this file would mean
keeping two specifications, which is one more than a specification can be.

So what is asserted here is behaviour rather than bytes: the branches a
registry turns on, and the security properties that hold whatever the profile
says. A pipeline that assembled ``.env.users`` itself, from a heredoc of
masked CI variables, would need tests policing which tokens the heredoc may
name. This pipeline assembles nothing: the deploy host runs
``osprey users env`` against its own ``.env``, and the single
allowlist in ``osprey.deployment.web_terminals.env_production`` decides what
lands in that file. What this module pins is the absence half — no secret may
reach a file, or the deploy host's command line, from here.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey
from osprey.cli.build_profile_deploy import (
    SUPPORTED_CI_PLATFORMS,
    DeployConfig,
    parse_deploy_block,
)
from osprey.cli.deploy_scaffold_templates import (
    CI_TEMPLATES,
    VERIFY_TEMPLATE,
    build_ci_context,
    build_verify_context,
    render,
    service_image_names,
)

GOLDENS = Path(__file__).parent / "goldens"
EXEMPLAR_DIR = GOLDENS / "exemplar-profile"

#: Passed in place of the installed version so a render does not change with the
#: release the test happens to run under.
FROZEN_VERSION = "OSPREY_VERSION"

#: The deployment repo's directory name. It is the deployment's name, and the
#: pipeline's title falls back to it; no emitted path keys off it any more.
REPO_NAME = "demo-facility"


@pytest.fixture(scope="module")
def exemplar() -> dict[str, Any]:
    """The exemplar profile, as the scaffolder reads it."""
    return yaml.safe_load((EXEMPLAR_DIR / "profile.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exemplar_deploy(exemplar: dict[str, Any]) -> DeployConfig:
    """The exemplar's parsed ``deploy:`` block."""
    deploy = parse_deploy_block(exemplar)
    assert deploy is not None, "the exemplar profile must declare a deploy block"
    return deploy


def render_ci(
    profile: dict[str, Any],
    deploy: DeployConfig,
    profile_dir: Path = EXEMPLAR_DIR,
    version: str = FROZEN_VERSION,
) -> str:
    """Render the pipeline for a profile."""
    context = build_ci_context(profile, deploy, profile_dir, REPO_NAME, version)
    return render(CI_TEMPLATES[deploy.ci], context)


@pytest.fixture(scope="module")
def rendered_ci(exemplar: dict[str, Any], exemplar_deploy: DeployConfig) -> str:
    """The pipeline the exemplar renders."""
    return render_ci(exemplar, exemplar_deploy)


@pytest.fixture(scope="module")
def rendered_verify(exemplar: dict[str, Any]) -> str:
    """The health check the exemplar renders."""
    return render(VERIFY_TEMPLATE, build_verify_context(exemplar, FROZEN_VERSION))


@pytest.fixture(scope="module")
def remote_script(rendered_ci: str) -> str:
    """The body of the deploy job's ``<<REMOTE`` heredoc.

    The heredoc is unquoted, so every ``$NAME`` inside it is expanded by the
    CI runner and travels to the deploy host as literal text on a command
    line. That makes it the one place in the pipeline where a masked variable
    could leak, which is why the security tests read it on its own.
    """
    match = re.search(r"<<REMOTE\n(.*?)\n\s*REMOTE\n", rendered_ci, re.DOTALL)
    assert match, "could not locate the deploy job's remote heredoc"
    return match.group(1)


# ── The provenance stamp ─────────────────────────────────────────────────────


def test_installed_version_is_the_only_value_that_moves_with_a_release(
    exemplar: dict[str, Any], exemplar_deploy: DeployConfig, rendered_ci: str, rendered_verify: str
) -> None:
    """Two renders of one profile differ in exactly the ``osprey-version:`` line.

    The scaffolder normalizes that line away before deciding whether a re-emit
    would change anything, so an upgrade rewrites neither file. A second
    release-dependent value anywhere in either template would be masked by that
    same normalization and would silently stop re-emission from noticing a real
    change.
    """
    live = (
        (rendered_ci, render_ci(exemplar, exemplar_deploy, version=osprey.__version__)),
        (
            rendered_verify,
            render(VERIFY_TEMPLATE, build_verify_context(exemplar, osprey.__version__)),
        ),
    )
    for frozen, current in live:
        differing = [
            (left, right)
            for left, right in zip(frozen.splitlines(), current.splitlines(), strict=True)
            if left != right
        ]
        assert differing == [
            (f"# osprey-version: {FROZEN_VERSION}", f"# osprey-version: {osprey.__version__}")
        ]


def test_the_byte_specification_lives_with_the_exemplar() -> None:
    """No hand-built goldens live here, and none may come back.

    They belong to the retired ``profile/`` layout, and a repo carrying both
    them and the three-zone exemplar would carry two specifications free to
    disagree. The bytes are pinned in ``tests/cli/test_emitted_artifacts_clean.py``
    against ``tests/fixtures/lifecycle_repo.py``; this asserts the pair has not
    quietly reappeared beside a suite that does not read it.
    """
    assert not (GOLDENS / "gitlab-ci.yml").exists()
    assert not (GOLDENS / "verify.sh").exists()
    assert EXEMPLAR_DIR.is_dir(), "the registry-flavoured profile is still read here"


# ── Template lookup ──────────────────────────────────────────────────────────


def test_every_supported_platform_has_a_template() -> None:
    """``deploy.ci`` accepts exactly the platforms with a shipped pipeline."""
    assert set(CI_TEMPLATES) == set(SUPPORTED_CI_PLATFORMS)
    assert CI_TEMPLATES["gitlab"] == "gitlab-ci.yml.j2"


# ── Per-service image jobs ───────────────────────────────────────────────────


def test_profile_owned_service_gets_an_image_job(rendered_ci: str) -> None:
    """The facility's own Dockerfile-carrying service earns a build job."""
    assert "image:facility-mcp:" in rendered_ci
    assert "extends: .service-image" in rendered_ci
    # The build context is the source zone's own services/, at the repo root —
    # not a copy under build/, which the render would have to succeed first to
    # produce.
    assert '"services/$SERVICE"' in rendered_ci


def test_packaged_service_directory_gets_no_image_job(
    exemplar: dict[str, Any], exemplar_deploy: DeployConfig, tmp_path: Path
) -> None:
    """A copied packaged service earns no job, Dockerfile or not.

    A materialized profile carries the virtual accelerator's service directory,
    Dockerfile included, but the framework builds that image from its own
    upstream. Only what the profile's ``services:`` block declares is the
    facility's to build.
    """
    profile_dir = tmp_path / "profile"
    (profile_dir / "services" / "facility-mcp").mkdir(parents=True)
    (profile_dir / "services" / "facility-mcp" / "Dockerfile").write_text("FROM scratch\n")
    (profile_dir / "services" / "virtual_accelerator").mkdir(parents=True)
    (profile_dir / "services" / "virtual_accelerator" / "Dockerfile").write_text("FROM scratch\n")

    assert service_image_names(exemplar, profile_dir) == ["facility-mcp"]
    rendered = render_ci(exemplar, exemplar_deploy, profile_dir=profile_dir)
    assert "image:facility-mcp:" in rendered
    assert "virtual_accelerator" not in rendered


# ── Pipeline contract ────────────────────────────────────────────────────────


def test_pipeline_runs_the_expected_osprey_commands(rendered_ci: str) -> None:
    """The ``osprey`` invocations are the ones the deployment story promises.

    Every one is bare. The repo IS the deployment, so each verb walks up to the
    profile from wherever it is run and needs no argument to say which one it
    means — which is also what makes the pipeline and a laptop run the same
    commands.
    """
    invocations = re.findall(r"^\s*(?:- )?(osprey .+)$", rendered_ci, re.MULTILINE)
    assert invocations == [
        "osprey validate",
        "osprey build --skip-lifecycle --skip-deps",
        "osprey build",
        "osprey users env --output .env.users",
        "osprey up -d",
    ]
    assert " -o " not in rendered_ci


def test_ci_extra_include_is_guarded_by_exists(rendered_ci: str) -> None:
    """The facility's own jobs are included, but a deleted file is not fatal."""
    pipeline = yaml.safe_load(rendered_ci)
    assert pipeline["include"] == [
        {"local": "ci-extra.yml", "rules": [{"exists": ["ci-extra.yml"]}]}
    ]


def test_pipeline_pins_the_version_floor_the_profile_declares(rendered_ci: str) -> None:
    """CI can never run an OSPREY that does not understand the profile."""
    assert 'pip install --no-cache-dir "osprey-framework>=2026.8.0"' in rendered_ci


# ── External projects ────────────────────────────────────────────────────────


def test_exemplar_render_carries_no_trace_of_external_projects(rendered_ci: str) -> None:
    """A profile declaring none must render as though the feature did not exist."""
    assert "external" not in rendered_ci


def test_declared_external_projects_get_their_own_jobs(exemplar: dict[str, Any]) -> None:
    """Each external project is proved pullable while there is time to rotate."""
    profile = copy.deepcopy(exemplar)
    profile["deploy"]["external_projects"] = [
        {
            "name": "beamline-tools",
            "url": "registry.example.org/beamline",
            "image": "tools:latest",
            "token_env_var": "BEAMLINE_PULL_TOKEN",
        }
    ]
    deploy = parse_deploy_block(profile)
    assert deploy is not None
    rendered = render_ci(profile, deploy)

    assert "external:beamline-tools:" in rendered
    assert 'docker pull "registry.example.org/beamline/tools:latest"' in rendered
    assert "BEAMLINE_PULL_TOKEN" in rendered
    yaml.safe_load(rendered)


# ── The health check ─────────────────────────────────────────────────────────


def test_probe_group_filter_is_not_named_groups(rendered_verify: str) -> None:
    """The filter variable must not be ``GROUPS``.

    Bash owns ``GROUPS`` — assigning to it is silently ignored, so the rename
    would leave every ``wants`` call false. The script exits 0 either way, and
    a health check that runs no probes reports perfect health.
    """
    assert 'PROBE_GROUPS="${*:-services web}"' in rendered_verify
    assert not re.search(r"^\s*GROUPS=", rendered_verify, re.MULTILINE)
    assert not re.search(r"\$\{?GROUPS\b", rendered_verify)


def test_health_check_probes_every_deployed_service(rendered_verify: str) -> None:
    """The three services this facility runs each get a probe.

    The exemplar names no ``port_base``, so two of the three are probed at
    their slots in the default block — the telemetry store at + 50, the
    facility's own MCP server at the first port of the facility band — and
    Channel Access at 5064 is the one port a block never moves.
    """
    assert "probe_tcp  'virtual-accelerator: Channel Access on 5064'  localhost 5064" in (
        rendered_verify
    )
    assert "'openobserve: telemetry store on 10050'" in rendered_verify
    assert "'facility-mcp: machine status on 10900'" in rendered_verify


def test_health_check_drops_the_web_group_when_there_are_no_terminals(
    exemplar: dict[str, Any],
) -> None:
    """A single-user deployment gets a script with nothing to explain away."""
    profile = copy.deepcopy(exemplar)
    profile["config"]["modules.web_terminals"]["enabled"] = False
    rendered = render(VERIFY_TEMPLATE, build_verify_context(profile, FROZEN_VERSION))

    assert 'PROBE_GROUPS="${*:-services}"' in rendered
    assert "Web terminals" not in rendered
    assert "nginx" not in rendered
    assert "./scripts/verify.sh services" in rendered


def test_health_check_is_advisory(rendered_verify: str) -> None:
    """It always exits 0 — a failed probe never fails a deploy."""
    assert rendered_verify.rstrip().endswith("exit 0")
    assert not re.search(r"^\s*exit [1-9]", rendered_verify, re.MULTILINE)
    assert not re.search(r"^set -e", rendered_verify, re.MULTILINE)


# ── Security: no secret reaches a file or a remote command line ──────────────


def test_pipeline_never_assembles_env_production(rendered_ci: str, remote_script: str) -> None:
    """CI writes no secrets file of its own.

    The legacy pipeline built ``.env.users`` from a heredoc of masked CI
    variables and COPYed the result into the runtime image. Nothing here may
    do that again: the only thing that produces the file is the deploy host,
    running ``osprey users env`` against its own ``.env``.

    The file is therefore *named* exactly once, as that command's destination
    on the host, and nowhere else — no CI-side assembly, no artifact, no COPY.
    """
    assert "ENVEOF" not in rendered_ci
    assert "cat > .env.users" not in rendered_ci

    mentions = [line.strip() for line in rendered_ci.splitlines() if ".env.users" in line]
    assert mentions == ["osprey users env --output .env.users"]
    assert mentions[0] in remote_script


def test_the_host_render_never_streams_secrets_to_the_job_log(remote_script: str) -> None:
    """``users env`` must write to a file, not to stdout.

    Without ``--output`` the command echoes the assembled subset — every
    credential the deployment runs on — and in this job stdout is the CI log,
    which is retained, searchable, and visible to everyone with read access to
    the project. The flag is also what creates the file at mode 0600 from its
    first byte; a shell redirect would leave it world-readable on the host.

    Pinned as a positive assertion rather than as the absence of a redirect:
    dropping the flag is a one-token edit whose damage is invisible until a
    real pipeline runs with real secrets in it.
    """
    render_lines = [
        line.strip()
        for line in remote_script.splitlines()
        if "users env" in line and "osprey" in line
    ]
    assert len(render_lines) == 1, render_lines
    assert "--output .env.users" in render_lines[0]
    assert ">" not in render_lines[0]

    # The written file must be the one `osprey up` then reads: both resolve it
    # against the repo root they are run in, so the render has to happen in the
    # same directory — any other destination deploys a stack whose containers
    # have no environment.
    assert "osprey up -d" in remote_script


def test_registry_token_is_never_written_to_a_file(
    rendered_ci: str, exemplar_deploy: DeployConfig
) -> None:
    """The registry credential is passed to ``docker login`` and nowhere else."""
    token = exemplar_deploy.registry.token_env_var
    uses = [line.strip() for line in rendered_ci.splitlines() if f"${token}" in line]
    assert uses == [f'- docker login -u "$CI_REGISTRY_USER" -p "${token}" "$REGISTRY_HOST"']


def test_no_secret_is_expanded_into_the_remote_command_line(
    remote_script: str, exemplar: dict[str, Any]
) -> None:
    """Nothing the profile declares as a secret crosses the SSH boundary.

    The heredoc is unquoted, so a ``$NAME`` here would be expanded by the
    runner and land in the deploy host's process arguments — visible to every
    user on the box. The host reads its own repo-root ``.env`` instead.
    """
    declared = exemplar["env"]["required"]
    # A profile that declared nothing would pass this loop without checking
    # anything, and the exemplar's list is small enough for that to happen by
    # accident on an edit.
    assert declared, "the exemplar declares no secrets, so this test proves nothing"
    for name in declared:
        assert f"${name}" not in remote_script


def test_native_service_tokens_are_absent(rendered_ci: str) -> None:
    """Tokens the framework mints per deployment never appear in CI.

    The event dispatcher's bearer token and its sidecar counterpart are minted
    on the deploy host and belong to the running stack, not to the pipeline.
    """
    assert "EVENT_DISPATCHER" not in rendered_ci
    assert "event_dispatcher" not in rendered_ci
    assert "sidecar_token" not in rendered_ci


def test_ssh_key_is_ci_only_and_lands_outside_the_project(rendered_ci: str) -> None:
    """The deploy key authenticates the job; it is not deployment environment."""
    assert '- cp "$DEPLOY_SSH_KEY" ~/.ssh/id_ed25519 && chmod 600 ~/.ssh/id_ed25519' in (
        rendered_ci
    )
    assert "DEPLOY_SSH_KEY" not in yaml.safe_load(rendered_ci)["variables"]
