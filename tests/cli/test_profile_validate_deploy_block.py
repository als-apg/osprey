"""The ``deploy:`` block schema, as ``osprey profile validate`` enforces it.

The block carries the coordinates a pipeline consumes — CI platform, registry,
deploy host, image source, external projects — and nothing the profile already
answers. Keys that belong elsewhere are rejected *by name* rather than ignored,
so a facility transcribing its deployment learns where each fact actually goes
instead of shipping a deployment silently missing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_deploy import (
    IMAGE_SOURCES,
    SUPPORTED_CI_PLATFORMS,
    parse_deploy_block,
)
from osprey.cli.build_profile_load import _KNOWN_PROFILE_KEYS, load_profile
from osprey.cli.profile_cmd import profile
from osprey.errors import BuildProfileError, ConfigurationError

VALID_DEPLOY: dict[str, Any] = {
    "ci": "gitlab",
    "registry": {
        "url": "git.example.org:5050/physics/production/facility-profiles",
        "token_env_var": "FACILITY_REGISTRY_TOKEN",
    },
    "host": {
        "name": "appsdev2",
        "fqdn": "appsdev2.example.org",
        "user": "operator",
        "project_path": "/home/operator/projects/facility-profiles",
    },
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _deploy(**overrides: Any) -> dict[str, Any]:
    """A valid deploy block with *overrides* applied at the top level."""
    block = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in VALID_DEPLOY.items()
    }
    for key, value in overrides.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    return block


def _write_profile(tmp_path: Path, deploy: Any) -> Path:
    body: dict[str, Any] = {"name": "Facility", "data_bundle": "hello_world"}
    if deploy is not None:
        body["deploy"] = deploy
    path = tmp_path / "profile.yml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _refusal(deploy: Any) -> str:
    """The accumulated problem report for an invalid block."""
    with pytest.raises(BuildProfileError) as excinfo:
        parse_deploy_block({"name": "Facility", "deploy": deploy})
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# The key is part of the schema, and the block is optional
# ---------------------------------------------------------------------------


def test_deploy_is_a_known_profile_key() -> None:
    assert "deploy" in _KNOWN_PROFILE_KEYS


def test_a_profile_without_a_deploy_block_parses_to_none(tmp_path: Path) -> None:
    """Deployment coordinates are opt-in — a laptop-only profile needs none."""
    assert load_profile(_write_profile(tmp_path, None)).deploy is None


def test_a_valid_block_parses_into_typed_coordinates(tmp_path: Path) -> None:
    deploy = load_profile(_write_profile(tmp_path, VALID_DEPLOY)).deploy

    assert deploy is not None
    assert deploy.ci == "gitlab"
    assert deploy.image_source == "registry"
    assert deploy.registry is not None
    assert deploy.registry.url == "git.example.org:5050/physics/production/facility-profiles"
    assert deploy.registry.token_env_var == "FACILITY_REGISTRY_TOKEN"
    assert deploy.host.name == "appsdev2"
    assert deploy.host.fqdn == "appsdev2.example.org"
    assert deploy.host.user == "operator"
    assert deploy.host.project_path == "/home/operator/projects/facility-profiles"
    assert deploy.external_projects == []


def test_the_block_must_be_a_mapping() -> None:
    assert "must be a mapping" in _refusal(["gitlab"])


def test_every_refusal_is_a_configuration_error() -> None:
    """`BuildProfileError` subclasses it — the CLI's error handling depends on
    that, and so does every caller that catches the broader class."""
    with pytest.raises(ConfigurationError):
        parse_deploy_block({"deploy": {}})


# ---------------------------------------------------------------------------
# ci — the platform name, and only platforms with a template
# ---------------------------------------------------------------------------


def test_ci_is_required() -> None:
    message = _refusal(_deploy(ci=None))

    assert "'ci' is required" in message
    assert "'gitlab'" in message


def test_an_unsupported_ci_platform_names_the_supported_ones() -> None:
    message = _refusal(_deploy(ci="jenkins"))

    assert "'jenkins'" in message
    assert "no pipeline template" in message
    for platform in SUPPORTED_CI_PLATFORMS:
        assert repr(platform) in message


def test_gitlab_is_the_only_platform_with_a_template_today() -> None:
    """Pinned so adding a platform to the tuple is a deliberate act that has to
    come with the template the scaffolding verbs look up by this name."""
    assert SUPPORTED_CI_PLATFORMS == ("gitlab",)


def test_the_legacy_ci_mapping_shape_is_rejected() -> None:
    """`ci:` names the platform, not a mapping of pipeline coordinates:
    everything else comes from the CI environment at run time."""
    message = _refusal(_deploy(ci={"provider": "gitlab", "project_id": 951}))

    assert "names the platform as a string" in message
    assert "not a mapping" in message


def test_the_removed_gitlab_block_is_rejected_by_name() -> None:
    """A `gitlab:` block is not a deploy key: the platform is named
    through `ci:` instead."""
    message = _refusal(_deploy(gitlab={"host": "git.example.org", "project_id": 951}))

    assert "'gitlab' is not a deploy key" in message
    assert 'ci: "gitlab"' in message


# ---------------------------------------------------------------------------
# image_source and registry
# ---------------------------------------------------------------------------


def test_image_source_defaults_to_pulling_what_ci_built() -> None:
    parsed = parse_deploy_block({"deploy": _deploy()})

    assert parsed is not None
    assert parsed.image_source == "registry"


def test_image_source_local_needs_no_registry() -> None:
    """A host that builds its own images has nothing to pull from."""
    parsed = parse_deploy_block({"deploy": _deploy(image_source="local", registry=None)})

    assert parsed is not None
    assert parsed.registry is None


def test_registry_is_required_when_images_are_pulled() -> None:
    message = _refusal(_deploy(registry=None))

    assert "'registry' is required when image_source is 'registry'" in message


def test_an_unknown_image_source_names_the_valid_ones() -> None:
    message = _refusal(_deploy(image_source="rsync"))

    assert "'rsync'" in message
    for source in IMAGE_SOURCES:
        assert repr(source) in message


def test_registry_url_is_required() -> None:
    assert "'registry.url' is required" in _refusal(_deploy(registry={"url": ""}))


def test_a_scheme_prefixed_registry_url_is_rejected() -> None:
    """An image reference built from a scheme-prefixed URL is unpullable, and
    the failure would otherwise surface only on the deploy host."""
    message = _refusal(_deploy(registry={"url": "https://git.example.org:5050/physics"}))

    assert "'registry.url' must not include a scheme" in message


def test_registry_rejects_an_unknown_subkey() -> None:
    message = _refusal(_deploy(registry={"url": "git.example.org:5050/p", "provider": "gitlab"}))

    assert "'registry' has unknown key(s): 'provider'" in message


# ---------------------------------------------------------------------------
# host — where the deployment actually runs
# ---------------------------------------------------------------------------


def test_host_is_required() -> None:
    message = _refusal(_deploy(host=None))

    assert "'host' is required" in message


@pytest.mark.parametrize("missing", ["name", "user", "project_path"])
def test_each_host_coordinate_is_required(missing: str) -> None:
    host = {key: value for key, value in VALID_DEPLOY["host"].items() if key != missing}

    assert f"'host.{missing}' is required" in _refusal(_deploy(host=host))


def test_host_fqdn_is_optional() -> None:
    host = {key: value for key, value in VALID_DEPLOY["host"].items() if key != "fqdn"}
    parsed = parse_deploy_block({"deploy": _deploy(host=host)})

    assert parsed is not None
    assert parsed.host.fqdn is None


def test_host_project_path_must_be_absolute() -> None:
    """It is resolved on the deploy host, where nobody's working directory is
    knowable from here."""
    host = {**VALID_DEPLOY["host"], "project_path": "projects/facility-profiles"}

    assert "'host.project_path' must be absolute" in _refusal(_deploy(host=host))


def test_host_rejects_an_unknown_subkey() -> None:
    host = {**VALID_DEPLOY["host"], "port": 22}

    assert "'host' has unknown key(s): 'port'" in _refusal(_deploy(host=host))


# ---------------------------------------------------------------------------
# external_projects
# ---------------------------------------------------------------------------


def test_external_projects_parse_into_typed_entries() -> None:
    parsed = parse_deploy_block(
        {
            "deploy": _deploy(
                external_projects=[
                    {
                        "name": "beam-viewer",
                        "url": "git.example.org:5050/physics/beam-viewer",
                        "image": "beam-viewer:latest",
                        "token_env_var": "BEAM_VIEWER_DEPLOY_TOKEN",
                    }
                ]
            )
        }
    )

    assert parsed is not None
    assert [entry.name for entry in parsed.external_projects] == ["beam-viewer"]
    assert parsed.external_projects[0].token_env_var == "BEAM_VIEWER_DEPLOY_TOKEN"


def test_an_external_project_needs_name_url_and_image() -> None:
    message = _refusal(_deploy(external_projects=[{"name": "beam-viewer"}]))

    assert "'external_projects[0].url' is required" in message
    assert "'external_projects[0].image' is required" in message


def test_two_external_projects_may_not_share_a_name() -> None:
    entry = {
        "name": "beam-viewer",
        "url": "git.example.org:5050/physics/beam-viewer",
        "image": "beam-viewer:latest",
    }
    message = _refusal(_deploy(external_projects=[entry, dict(entry)]))

    assert "repeats 'beam-viewer'" in message


def test_external_projects_must_be_a_list() -> None:
    message = _refusal(_deploy(external_projects={"beam-viewer": {}}))

    assert "'external_projects' must be a list" in message


# ---------------------------------------------------------------------------
# Keys the profile owns are rejected by name, not ignored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value", "expected_home"),
    [
        ("llm", {"provider": "anthropic"}, "`provider:` and `model:`"),
        ("timezone", "America/Los_Angeles", "system.timezone"),
        ("control_system", {"type": "epics"}, "control_system.type"),
        ("users", ["operator"], "modules.web_terminals.users"),
        ("personas", {"analysis": {}}, "modules.web_terminals.personas"),
        ("default_persona", "assistant", "modules.web_terminals.default_persona"),
    ],
)
def test_a_profile_owned_key_is_rejected_naming_its_real_home(
    key: str, value: Any, expected_home: str
) -> None:
    message = _refusal(_deploy(**{key: value}))

    assert f"'{key}' is not a deploy key" in message
    assert expected_home in message


@pytest.mark.parametrize("key", ["env", "environment"])
def test_the_deploy_block_declares_no_environment_variables(key: str) -> None:
    """It names them (`registry.token_env_var`); the profile's own
    `env.required` / `env.defaults` is the one channel that declares them."""
    message = _refusal(_deploy(**{key: {"required": ["FACILITY_REGISTRY_TOKEN"]}}))

    assert f"'{key}' is not a deploy key" in message
    assert "`env.required` / `env.defaults`" in message


def test_a_token_value_written_where_a_name_belongs_is_rejected() -> None:
    """The block references variable names; a value here would land a secret in
    a version-controlled file."""
    fake_value = "glpat-XxYyZz123"  # gitleaks:allow — refused by the schema
    registry = {**VALID_DEPLOY["registry"], "token_env_var": fake_value}
    message = _refusal(_deploy(registry=registry))

    assert "must be an environment variable NAME" in message


def test_an_unknown_deploy_key_names_its_closest_spelling() -> None:
    message = _refusal(_deploy(registries={"url": "git.example.org:5050/p"}))

    assert "unknown key 'registries'" in message
    assert "did you mean 'registry'?" in message


# ---------------------------------------------------------------------------
# The commented stub an emitted profile carries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_an_emitted_profile_offers_the_deploy_block_as_a_comment_only(preset: str) -> None:
    """Guards the emitter's double-emit foot-gun: a commented template that
    also lands as an active key would give the profile two ``deploy:`` homes,
    the second silently winning. It must appear exactly once, commented."""
    from osprey.cli.build_profile_emit import emit_standalone_profile_yaml

    text = emit_standalone_profile_yaml(preset, (), (), "Emitted")

    assert "deploy" not in (yaml.safe_load(text) or {})
    assert [line for line in text.splitlines() if line.strip() == "# deploy:"] == ["# deploy:"]
    assert "deploy:" not in [line.strip() for line in text.splitlines()]


def test_the_emitted_stub_is_a_block_the_schema_accepts() -> None:
    """A template that documents a shape the validator refuses would teach the
    facility the wrong one, and it would only find out after uncommenting."""
    from osprey.cli.build_profile_emit import emit_standalone_profile_yaml

    text = emit_standalone_profile_yaml("control-assistant", (), (), "Emitted")
    lines = text.splitlines()
    start = lines.index("# deploy:")
    stub = []
    for line in lines[start:]:
        if not line.startswith("#"):
            break
        stub.append(line[2:] if line.startswith("# ") else line[1:])
    block = yaml.safe_load("\n".join(stub))

    assert parse_deploy_block(block) is not None


# ---------------------------------------------------------------------------
# Every problem in the block is reported at once
# ---------------------------------------------------------------------------


def test_the_refusal_reports_every_problem_at_once() -> None:
    """A facility filling this in for the first time is transcribing a handful
    of coordinates; one round-trip per mistake is the slowest way to learn it."""
    message = _refusal({"ci": "jenkins", "llm": {"provider": "anthropic"}})

    assert "'jenkins'" in message
    assert "'llm' is not a deploy key" in message
    assert "'registry' is required" in message
    assert "'host' is required" in message
    assert message.count("\n  - ") == 4


# ---------------------------------------------------------------------------
# `osprey profile validate` is where a facility meets all of this
# ---------------------------------------------------------------------------


def test_validate_accepts_a_profile_carrying_a_deploy_block(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(profile, ["validate", str(_write_profile(tmp_path, VALID_DEPLOY))])

    assert result.exit_code == 0, result.output


def test_validate_reports_the_deploy_target(runner: CliRunner, tmp_path: Path) -> None:
    """ "Valid" alone cannot say whether the coordinates were read at all."""
    result = runner.invoke(profile, ["validate", str(_write_profile(tmp_path, VALID_DEPLOY))])

    assert "gitlab CI" in result.output
    assert "operator@appsdev2" in result.output


def test_validate_says_nothing_about_deploy_when_the_block_is_absent(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(profile, ["validate", str(_write_profile(tmp_path, None))])

    assert result.exit_code == 0, result.output
    assert "Deploy:" not in result.output


def test_validate_exits_nonzero_and_names_the_problem(runner: CliRunner, tmp_path: Path) -> None:
    path = _write_profile(tmp_path, _deploy(ci="jenkins"))
    result = runner.invoke(profile, ["validate", str(path)])

    assert result.exit_code != 0
    assert "jenkins" in result.output
