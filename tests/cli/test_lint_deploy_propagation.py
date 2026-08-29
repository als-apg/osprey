"""The web-stack lint reads the config a BUILD renders, not the raw ``config:``.

``image_source`` lives in the ``deploy:`` block and reaches the rendered config
through ``deploy_config_overrides``. A lint run against the raw ``config:``
block therefore judges a profile on a view no deployment ever runs: a facility
that correctly states ``image_source: local`` once, in the deploy block, gets
told its defaulted ``registry`` mode is missing a ``registry.url`` it will never
read. That is not hypothetical — it is what the hand-built exemplar profile hit.

Both surfaces that gate on the lint — ``osprey profile validate`` and ``osprey
build``'s pre-check — go through ``deploy_aware_config_errors``, so a profile
that validates builds. The parity tests at the bottom are what keep that true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_deploy import (
    IMAGE_SOURCE_CONFIG_KEY,
    deploy_aware_config_errors,
    parse_deploy_block,
)
from osprey.cli.profile_cmd import profile as profile_group
from osprey.deployment.web_terminals.lint import profile_config_errors

#: The frozen reference deployment. Its ``deploy:`` block says ``image_source:
#: local`` and its ``config:`` block deliberately does NOT — one fact, one home.
EXEMPLAR = Path(__file__).resolve().parents[1] / "deployment" / "goldens" / "exemplar-profile"

DEPLOY_BLOCK: dict[str, Any] = {
    "ci": "gitlab",
    "image_source": "local",
    "host": {"name": "demo-deploy", "user": "osprey", "project_path": "/opt/demo"},
}

#: A roster that stands up the persona stack — which is what makes the
#: mode-coherence check apply at all (a config with no catalog resolves through
#: the pre-catalog path and is never asked about ``registry.url``).
WEB_TERMINALS: dict[str, Any] = {
    "enabled": True,
    "users": [{"name": "operator", "index": 0, "persona": "readwrite"}],
    "default_persona": "readwrite",
    "personas": {"readwrite": {"build_profile": "hello-world", "project": "demo-readwrite"}},
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _config(**extra: Any) -> dict[str, Any]:
    return {"facility.prefix": "demo", "modules.web_terminals": dict(WEB_TERMINALS), **extra}


def _deploy(**extra: Any) -> Any:
    return parse_deploy_block({"deploy": {**DEPLOY_BLOCK, **extra}})


def _reported(errors: list[str]) -> str:
    return " | ".join(errors)


# ---------------------------------------------------------------------------
# The merged view
# ---------------------------------------------------------------------------


def test_a_deploy_block_answers_a_question_the_raw_config_cannot(runner: CliRunner) -> None:
    """The whole bug in one assertion: the same profile, judged two ways."""
    config = _config()

    raw = profile_config_errors(config)
    merged = deploy_aware_config_errors(_deploy(), config)

    assert any("registry.url is not set" in message for message in raw), _reported(raw)
    assert merged == [], _reported(merged)


def test_a_profile_with_no_deploy_block_lints_exactly_as_before() -> None:
    """The deploy block contributes nothing when there is none, so a profile
    that never adopted deployment coordinates must see the identical verdict —
    including the identical failure."""
    config = _config()

    assert deploy_aware_config_errors(None, config) == profile_config_errors(config)
    assert any("registry.url is not set" in message for message in profile_config_errors(config))


def test_a_config_block_image_source_is_still_linted() -> None:
    """No coverage is traded away: a facility that spells the fact in the
    ``config:`` block (the only spelling available without a deploy block) is
    checked exactly as it always was."""
    stated_registry = _config(**{IMAGE_SOURCE_CONFIG_KEY: "registry"})
    stated_local = _config(**{IMAGE_SOURCE_CONFIG_KEY: "local"})

    errors = deploy_aware_config_errors(None, stated_registry)

    assert any("registry.url is not set" in message for message in errors), _reported(errors)
    assert deploy_aware_config_errors(None, stated_local) == []


def test_a_profile_that_runs_no_web_stack_is_unaffected() -> None:
    """``deploy_config_overrides`` writes the leaf only into a module the
    facility declared, so a profile without one merges nothing and lints the
    same either way."""
    config = {"control_system.type": "epics"}

    assert deploy_aware_config_errors(_deploy(), config) == profile_config_errors(config)


def test_the_deploy_registrys_url_is_not_the_one_the_lint_reads() -> None:
    """The propagated value is not a blanket pass, and this pins the edge of
    what it covers.

    ``deploy.registry.url`` is where CI pushes and the host pulls;
    ``registry.url`` in the rendered config is what names each persona's image.
    They are the same registry in practice, but only ``image_source`` is
    propagated between them — so a registry-mode facility still has to state
    the URL in the ``config:`` block, and the lint still says so. Documented
    here rather than quietly fixed: propagating the second value is a schema
    decision, not a lint one.
    """
    pulls_images = _deploy(image_source="registry", registry={"url": "registry.example.org/demo"})

    errors = deploy_aware_config_errors(pulls_images, _config())
    satisfied = deploy_aware_config_errors(pulls_images, _config(**{"registry.url": "r.example"}))

    assert any("registry.url is not set" in message for message in errors), _reported(errors)
    assert satisfied == [], _reported(satisfied)


# ---------------------------------------------------------------------------
# The command surfaces
# ---------------------------------------------------------------------------


def test_the_exemplar_golden_profile_validates(runner: CliRunner) -> None:
    """The acceptance evidence. The hand-built reference deployment is the
    specification this whole feature is derived from; it failing its own
    validator was the bug."""
    result = runner.invoke(profile_group, ["validate", str(EXEMPLAR)])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output


def test_a_malformed_config_block_is_refused_by_name_not_by_traceback(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every reader of ``config:`` assumes a mapping, so a list reached one of
    them as an unhandled ``TypeError``. The profile schema now rejects the shape
    up front, which is both surfaces' answer: ``validate()`` runs before the
    lint on the validate path and during resolution on the build path.
    """
    path = tmp_path / "profile" / "profile.yml"
    path.parent.mkdir(parents=True)
    path.write_text("name: Bad\ndata_bundle: hello_world\nconfig: []\n", encoding="utf-8")

    validated = runner.invoke(profile_group, ["validate", str(path)])
    with caplog.at_level("ERROR"):
        built = _build(runner, path)

    assert validated.exit_code == 2, validated.output
    assert "config must be a mapping" in validated.output
    assert built.exit_code != 0, built.output
    assert "config must be a mapping" in (caplog.text + built.output)
    for result in (validated, built):
        assert not isinstance(result.exception, TypeError), result.exception


def test_an_empty_config_block_is_not_a_malformed_one(runner: CliRunner, tmp_path: Path) -> None:
    """``config:`` with nothing under it — every entry commented out — parses to
    None and means "no config entries", which is a mapping with none in it.

    Pinned beside the rejection above because the obvious normalization (``or
    {}``) would also swallow an empty LIST, quietly accepting the very shape the
    schema is there to name.
    """
    path = tmp_path / "profile" / "profile.yml"
    path.parent.mkdir(parents=True)
    path.write_text("name: Empty\ndata_bundle: hello_world\nconfig:\n", encoding="utf-8")

    result = runner.invoke(profile_group, ["validate", str(path)])

    assert result.exit_code == 0, result.output


def test_the_duplicate_home_refusal_still_fires(runner: CliRunner, tmp_path: Path) -> None:
    """Merging the deploy value in must not become a way to tolerate the second
    home. A profile stating ``image_source`` in both places is still refused
    during resolution, before the lint is ever reached."""
    target = _write_profile(
        tmp_path, deploy=dict(DEPLOY_BLOCK), config=_config(**{IMAGE_SOURCE_CONFIG_KEY: "local"})
    )

    result = runner.invoke(profile_group, ["validate", str(target)])

    assert result.exit_code == 2
    assert "one fact, two homes" in result.output


# ---------------------------------------------------------------------------
# Parity: what validates, builds
# ---------------------------------------------------------------------------


def _write_profile(
    directory: Path, *, deploy: dict[str, Any] | None, config: dict[str, Any]
) -> Path:
    """A minimal buildable profile carrying the multi-user stack."""
    raw: dict[str, Any] = {
        "name": "Demo Facility",
        "data_bundle": "hello_world",
        "provider": "anthropic",
        "model": "haiku",
        "config": config,
    }
    if deploy is not None:
        raw["deploy"] = deploy
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "profile.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _build(runner: CliRunner, profile_path: Path):
    """Build the deployment repo *profile_path* sits at the root of.

    ``profile.yml`` must already be at the repo root — every caller here
    writes it there directly, with no ``osprey init`` in between, so this
    just points ``osprey build --repo`` at that directory.
    """
    return runner.invoke(
        build,
        ["--repo", str(profile_path.parent), "--skip-deps", "--skip-lifecycle"],
    )


def test_a_profile_the_deploy_block_rescues_both_validates_and_builds(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The direction that was broken: the raw ``config:`` block fails the lint,
    the deploy block answers it, and BOTH surfaces have to see that."""
    path = _write_profile(tmp_path / "profile", deploy=dict(DEPLOY_BLOCK), config=_config())

    validated = runner.invoke(profile_group, ["validate", str(path)])
    built = _build(runner, path)

    assert validated.exit_code == 0, validated.output
    assert built.exit_code == 0, built.output


def test_a_profile_neither_can_rescue_is_refused_by_both_with_one_message(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The other direction, and the anti-vacuous half: with no deploy block the
    same lint failure has to stop both surfaces, with the same words. If the two
    ever drift, this is what notices."""
    path = _write_profile(tmp_path / "profile", deploy=None, config=_config())

    validated = runner.invoke(profile_group, ["validate", str(path)])
    with caplog.at_level("ERROR"):
        built = _build(runner, path)

    assert validated.exit_code == 2, validated.output
    assert built.exit_code != 0, built.output
    assert "registry.url is not set" in validated.output
    # CliRunner rich-wraps long lines, so the build's refusal is read off the
    # log record as well as the rendered output.
    assert "registry.url is not set" in (caplog.text + built.output)
