"""The web-stack lint reads the config a BUILD renders, not the raw ``config:``.

``image_source`` lives in the ``deploy:`` block and reaches the rendered config
through ``deploy_config_overrides``, and so does ``registry.url`` in registry
mode when the ``config:`` block names none. A lint run against the raw ``config:``
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

from osprey.cli.build_profile_deploy import (
    IMAGE_SOURCE_CONFIG_KEY,
    deploy_aware_config_errors,
    deploy_config_overrides,
    parse_deploy_block,
)
from osprey.cli.profile_cmd import profile as profile_group
from osprey.deployment.web_terminals.lint import profile_config_errors
from tests.cli._lifecycle_build import invoke_build, rendered_config

#: The frozen reference deployment. Its ``deploy:`` block says ``image_source:
#: local`` and its ``config:`` block deliberately does NOT — one fact, one home.
EXEMPLAR = Path(__file__).resolve().parents[1] / "deployment" / "goldens" / "exemplar-profile"

DEPLOY_BLOCK: dict[str, Any] = {
    "ci": "gitlab",
    "image_source": "local",
    "host": {"name": "demo-deploy", "user": "osprey", "project_path": "/opt/demo"},
}

#: The posture keys a build refuses to guess at. Stated once here because they
#: are the price of admission to a build, not a subject of any test in this file
#: — the deploy/lint parity these tests are about is judged past them. The
#: profiles select the approval hook, so they state its policy.
POSTURE_FLOOR: dict[str, Any] = {
    "control_system.type": "virtual_accelerator",
    "control_system.connector.virtual_accelerator.serving": "in_process",
    "archiver.type": "mock_archiver",
    "approval.enabled": False,
    "approval.default_policy": "always",
    "hooks.debug": False,
}

#: A roster that stands up the render stack. Registry mode names every
#: web-terminal image under ``registry.url``. Each tier it names is a
#: ``tiers/<name>.yml`` file :func:`_write_profile` lays down beside it.
WEB_TERMINALS: dict[str, Any] = {
    "enabled": True,
    "users": [{"name": "console", "index": 0, "tier": "readwrite"}],
    "default_tier": "readwrite",
}


def _config(**extra: Any) -> dict[str, Any]:
    return {"modules.web_terminals": dict(WEB_TERMINALS), **extra}


def _deploy(**extra: Any) -> Any:
    return parse_deploy_block({"deploy": {**DEPLOY_BLOCK, **extra}})


@pytest.fixture
def tier_root(tmp_path: Path) -> Path:
    """A profile root holding the one tier :data:`WEB_TERMINALS` names.

    A profile states its tiers as files, so a lint judging the roster at
    profile altitude reads the catalog from the root it is handed.
    """
    tier_file = tmp_path / "tiers" / "readwrite.yml"
    tier_file.parent.mkdir(parents=True)
    tier_file.write_text("name: Demo Facility (readwrite)\n", encoding="utf-8")
    return tmp_path


def _reported(errors: list[str]) -> str:
    return " | ".join(errors)


# ---------------------------------------------------------------------------
# The merged view
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("runner")
def test_a_deploy_block_answers_a_question_the_raw_config_cannot(tier_root: Path) -> None:
    """The whole bug in one assertion: the same profile, judged two ways."""
    config = _config()

    raw = profile_config_errors(config, profile_root=tier_root)
    merged = deploy_aware_config_errors(_deploy(), config, profile_root=tier_root)

    assert any("registry.url is not set" in message for message in raw), _reported(raw)
    assert merged == [], _reported(merged)


def test_a_profile_with_no_deploy_block_lints_exactly_as_before() -> None:
    """The deploy block contributes nothing when there is none, so a profile
    that never adopted deployment coordinates must see the identical verdict —
    including the identical failure."""
    config = _config()

    assert deploy_aware_config_errors(None, config) == profile_config_errors(config)
    assert any("registry.url is not set" in message for message in profile_config_errors(config))


def test_a_config_block_image_source_is_still_linted(tier_root: Path) -> None:
    """No coverage is traded away: a facility that spells the fact in the
    ``config:`` block (the only spelling available without a deploy block) is
    checked exactly as it always was."""
    stated_registry = _config(**{IMAGE_SOURCE_CONFIG_KEY: "registry"})
    stated_local = _config(**{IMAGE_SOURCE_CONFIG_KEY: "local"})

    errors = deploy_aware_config_errors(None, stated_registry, profile_root=tier_root)

    assert any("registry.url is not set" in message for message in errors), _reported(errors)
    assert deploy_aware_config_errors(None, stated_local, profile_root=tier_root) == []


def test_a_profile_that_runs_no_web_stack_is_unaffected() -> None:
    """``deploy_config_overrides`` writes the leaf only into a module the
    facility declared, so a profile without one merges nothing and lints the
    same either way."""
    config = {"control_system.type": "epics"}

    assert deploy_aware_config_errors(_deploy(), config) == profile_config_errors(config)


def test_the_deploy_registry_satisfies_the_lint_and_an_explicit_config_url_wins(
    tier_root: Path,
) -> None:
    """The deploy block names the registry once: in registry mode its
    ``registry.url`` reaches the rendered ``registry.url`` the lint reads. A
    ``config:`` value points the web tier elsewhere, and the build then
    contributes nothing for that key."""
    pulls_images = _deploy(image_source="registry", registry={"url": "registry.example.org/demo"})
    pointed_elsewhere = _config(**{"registry.url": "r.example"})

    filled = deploy_aware_config_errors(pulls_images, _config(), profile_root=tier_root)
    explicit = deploy_aware_config_errors(pulls_images, pointed_elsewhere, profile_root=tier_root)

    assert filled == [], _reported(filled)
    assert explicit == [], _reported(explicit)
    assert "registry.url" not in deploy_config_overrides(pulls_images, pointed_elsewhere)


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
    (path.parent / "data").mkdir()
    path.write_text("name: Bad\ndata: data\nconfig: []\n", encoding="utf-8")

    validated = runner.invoke(profile_group, ["validate", str(path)])
    with caplog.at_level("ERROR"):
        built = invoke_build(path.parent)

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
    (path.parent / "data").mkdir()
    path.write_text("name: Empty\ndata: data\nconfig:\n", encoding="utf-8")

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
    """A minimal buildable profile carrying the multi-user stack.

    "Minimal" now means the posture floor as well. ``data``,
    ``channel_finder_mode``, ``hooks`` and :data:`POSTURE_FLOOR` are here to
    make the profile buildable at all, not because these tests are about any of
    them: a repo profile must name a data tree, the channel-finder agent the
    default artifact set selects refuses to render without a paradigm, the
    posture keys are the ones a build refuses to guess, ``memory-guard`` is
    what gates ``Write`` — a profile that could write with no gate is refused —
    and ``writes-check`` is what refuses the mixed read/write python tools'
    writes per call under the ``never`` write posture, which a build also
    refuses to render without; ``approval`` is what puts a governed tool such
    as ``execute`` to a human, which the write switch does in every render
    that serves one, so a build refuses to render without it either.
    A caller stating one of them keeps its own value.

    Every tier the roster names gets its ``tiers/<name>.yml``, the file a
    render is resolved from.
    """
    raw: dict[str, Any] = {
        "name": "Demo Facility",
        "project_name": "demo-facility",
        "data": "data",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "channel_finder_mode": "hierarchical",
        "hooks": ["memory-guard", "writes-check", "approval"],
        "config": {**POSTURE_FLOOR, **config},
    }
    if deploy is not None:
        raw["deploy"] = deploy
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data").mkdir(exist_ok=True)
    path = directory / "profile.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    roster = config.get("modules.web_terminals") or {}
    tiers = {roster.get("default_tier")} | {
        user.get("tier") for user in roster.get("users", []) if isinstance(user, dict)
    }
    for tier in sorted(t for t in tiers if t):
        tier_file = directory / "tiers" / f"{tier}.yml"
        tier_file.parent.mkdir(exist_ok=True)
        tier_file.write_text(yaml.safe_dump({"name": f"Demo Facility ({tier})"}), encoding="utf-8")
    return path


def test_a_profile_the_deploy_block_rescues_both_validates_and_builds(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The direction that was broken: the raw ``config:`` block fails the lint,
    the deploy block answers it, and BOTH surfaces have to see that."""
    path = _write_profile(tmp_path / "profile", deploy=dict(DEPLOY_BLOCK), config=_config())

    validated = runner.invoke(profile_group, ["validate", str(path)])
    built = invoke_build(path.parent)

    assert validated.exit_code == 0, validated.output
    assert built.exit_code == 0, built.output


def test_a_registry_mode_profile_names_its_registry_once_on_both_surfaces(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A registry-mode profile that names its registry only in the deploy block
    validates, builds, and renders that registry for the web tier."""
    repo = tmp_path / "profile"
    path = _write_profile(
        repo,
        deploy={
            **DEPLOY_BLOCK,
            "image_source": "registry",
            "registry": {"url": "registry.example.org/demo"},
        },
        config=_config(),
    )

    validated = runner.invoke(profile_group, ["validate", str(path)])
    built = invoke_build(path.parent)

    assert validated.exit_code == 0, validated.output
    assert built.exit_code == 0, built.output
    rendered = rendered_config(repo)
    assert rendered["registry"] == {"url": "registry.example.org/demo"}


def test_an_explicit_config_registry_url_reaches_the_render(tmp_path: Path) -> None:
    """A ``config:`` registry wins over the deploy block's."""
    repo = tmp_path / "profile"
    path = _write_profile(
        repo,
        deploy={
            **DEPLOY_BLOCK,
            "image_source": "registry",
            "registry": {"url": "registry.example.org/demo"},
        },
        config=_config(**{"registry": {"url": "mirror.example.org/demo"}}),
    )

    built = invoke_build(path.parent)

    assert built.exit_code == 0, built.output
    rendered = rendered_config(repo)
    assert rendered["registry"] == {"url": "mirror.example.org/demo"}


def test_a_profile_neither_can_rescue_is_refused_by_both_with_one_message(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The other direction, and the anti-vacuous half: with no deploy block the
    same lint failure has to stop both surfaces, with the same words. If the two
    ever drift, this is what notices."""
    path = _write_profile(tmp_path / "profile", deploy=None, config=_config())

    validated = runner.invoke(profile_group, ["validate", str(path)])
    with caplog.at_level("ERROR"):
        built = invoke_build(path.parent)

    assert validated.exit_code == 2, validated.output
    assert built.exit_code != 0, built.output
    assert "registry.url is not set" in validated.output
    # CliRunner rich-wraps long lines, so the build's refusal is read off the
    # log record as well as the rendered output.
    assert "registry.url is not set" in (caplog.text + built.output)
