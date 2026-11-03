"""``image_source`` has one home in the profile and one value in the render.

A facility says how its deploy host gets images once, in the ``deploy:`` block.
The multi-user web stack reads that answer from
``modules.web_terminals.image_source``, so the build writes it there rather than
making the facility repeat itself — and a profile that says it in both places is
refused, because two homes for one fact are free to disagree and the
disagreement decides whether the host builds images or pulls them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_deploy import (
    IMAGE_SOURCE_CONFIG_KEY,
    REGISTRY_URL_CONFIG_KEY,
    config_image_source_spelling,
    declares_web_terminals,
    deploy_config_overrides,
    parse_deploy_block,
)
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import invoke_build, rendered_config

DEPLOY_BLOCK: dict[str, Any] = {
    "ci": "gitlab",
    "registry": {"url": "registry.example.org/accelerator/demo"},
    "host": {
        "name": "demo-deploy",
        "user": "osprey",
        "project_path": "/opt/demo",
    },
}

WEB_TERMINALS: dict[str, Any] = {
    "enabled": True,
    "nginx_port": 20000,
    "web_base_port": 20100,
    "users": ["console"],
}

# The same stack, deployable in LOCAL mode. `image_source: local` builds each
# render's image, so it requires a tier and a default to build them from —
# `resolve_cards` raises without one at deploy time, and since the build's
# lint reads the deploy-propagated config (`deploy_aware_config_errors`) it
# refuses the same profile earlier. Every end-to-end case below names a local
# deploy block, so they all build from this one, beside a `tiers/` file for
# each name in LOCAL_TIERS.
WEB_TERMINALS_LOCAL: dict[str, Any] = {
    **WEB_TERMINALS,
    "default_tier": "readwrite",
}
LOCAL_TIERS: tuple[str, ...] = ("readwrite",)


def _profile(deploy: dict[str, Any] | None, config: dict[str, Any] | None) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "Demo",
        "project_name": "demo",
        "extends": "hello-world",
        "data": "data",
    }
    if deploy is not None:
        raw["deploy"] = deploy
    if config is not None:
        raw["config"] = config
    return raw


def _parsed(deploy: dict[str, Any], config: dict[str, Any] | None = None) -> Any:
    return parse_deploy_block(_profile(deploy, config))


# ---------------------------------------------------------------------------
# Where the propagation applies
# ---------------------------------------------------------------------------


def test_the_deploy_value_becomes_the_rendered_web_terminals_leaf() -> None:
    deploy = _parsed({**DEPLOY_BLOCK, "image_source": "local"})
    config = {"modules.web_terminals": dict(WEB_TERMINALS)}

    assert deploy_config_overrides(deploy, config) == {IMAGE_SOURCE_CONFIG_KEY: "local"}


def test_the_registry_default_is_propagated_just_as_explicitly() -> None:
    """An unstated `image_source` still means something — 'pull what CI built'
    — so the rendered config says it rather than leaning on a reader default."""
    deploy = _parsed(DEPLOY_BLOCK)
    config = {"modules.web_terminals": dict(WEB_TERMINALS)}

    assert deploy_config_overrides(deploy, config) == {
        IMAGE_SOURCE_CONFIG_KEY: "registry",
        REGISTRY_URL_CONFIG_KEY: DEPLOY_BLOCK["registry"]["url"],
    }


def test_a_profile_with_no_deploy_block_contributes_nothing() -> None:
    assert deploy_config_overrides(None, {"modules.web_terminals": dict(WEB_TERMINALS)}) == {}


def test_nothing_is_written_when_the_profile_runs_no_web_terminal_stack() -> None:
    """The leaf lives inside a module the facility has to have declared; adding
    it to a config without one would invent a module nobody asked for."""
    deploy = _parsed({**DEPLOY_BLOCK, "image_source": "local"})

    assert deploy_config_overrides(deploy, {"control_system.type": "epics"}) == {}
    assert deploy_config_overrides(deploy, {}) == {}


@pytest.mark.parametrize(
    "config",
    [
        {"modules.web_terminals": {"enabled": True}},
        {"modules.web_terminals.enabled": True},
        {"modules": {"web_terminals": {"enabled": True}}},
    ],
    ids=["subtree-key", "dotted-leaf", "nested-mapping"],
)
def test_every_spelling_of_the_module_counts_as_declaring_it(config: dict[str, Any]) -> None:
    assert declares_web_terminals(config) is True


def test_a_config_without_the_module_does_not_count() -> None:
    assert declares_web_terminals({"modules.event_dispatcher": {"enabled": True}}) is False


# ---------------------------------------------------------------------------
# The second home is refused, in every spelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "named"),
    [
        ({IMAGE_SOURCE_CONFIG_KEY: "local"}, IMAGE_SOURCE_CONFIG_KEY),
        (
            {"modules.web_terminals": {**WEB_TERMINALS, "image_source": "local"}},
            "modules.web_terminals: image_source",
        ),
        (
            {"modules": {"web_terminals": {"image_source": "local"}}},
            "modules: web_terminals: image_source",
        ),
    ],
    ids=["dotted-leaf", "subtree-key", "nested-mapping"],
)
def test_stating_it_in_both_homes_is_refused(config: dict[str, Any], named: str) -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        parse_deploy_block(_profile({**DEPLOY_BLOCK, "image_source": "local"}, config))

    message = str(excinfo.value)
    assert "one fact, two homes" in message
    assert named in message


def test_the_refusal_names_the_value_so_moving_it_cannot_change_the_mode() -> None:
    """Deleting the config entry flips the mode unless the deploy block already
    says the same thing, so the message states what the deploy block says."""
    with pytest.raises(BuildProfileError) as excinfo:
        parse_deploy_block(_profile(DEPLOY_BLOCK, {IMAGE_SOURCE_CONFIG_KEY: "local"}))

    assert "image_source: registry" in str(excinfo.value)
    assert IMAGE_SOURCE_CONFIG_KEY in str(excinfo.value)


def test_the_config_spelling_stays_legal_without_a_deploy_block() -> None:
    """A profile with no deploy block has nowhere else to say it — including
    every bundled preset, which cannot know a facility's CI or registry."""
    config = {"modules.web_terminals": {**WEB_TERMINALS, "image_source": "local"}}

    assert parse_deploy_block(_profile(None, config)) is None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, None),
        ({"modules.web_terminals": dict(WEB_TERMINALS)}, None),
        ({IMAGE_SOURCE_CONFIG_KEY: "local"}, IMAGE_SOURCE_CONFIG_KEY),
    ],
)
def test_the_spelling_probe_reports_only_a_real_duplicate(
    config: dict[str, Any], expected: str | None
) -> None:
    assert config_image_source_spelling(config) == expected


# ---------------------------------------------------------------------------
# End to end: what a built project's config.yml actually says
# ---------------------------------------------------------------------------


def _build_project(
    tmp_path: Path,
    profile_body: dict[str, Any],
    tiers: tuple[str, ...] = (),
) -> dict:
    """Write *profile_body* at the repo root, one file per tier, and build in place."""
    profile = tmp_path / "profile.yml"
    (tmp_path / "data").mkdir(exist_ok=True)
    profile.write_text(yaml.safe_dump(profile_body, sort_keys=False), encoding="utf-8")
    for tier in tiers:
        tier_file = tmp_path / "tiers" / f"{tier}.yml"
        tier_file.parent.mkdir(exist_ok=True)
        tier_file.write_text(yaml.safe_dump({"name": f"Demo ({tier})"}), encoding="utf-8")
    result = invoke_build(tmp_path)
    assert result.exit_code == 0, result.output
    return rendered_config(tmp_path)


def test_a_built_project_reads_the_mode_the_deploy_block_named(tmp_path: Path) -> None:
    body = _profile({**DEPLOY_BLOCK, "image_source": "local"}, None)
    body["provider"] = "anthropic"
    body["model"] = "claude-haiku-4-5"
    body["config"] = {
        "modules.web_terminals": dict(WEB_TERMINALS_LOCAL),
    }

    config = _build_project(tmp_path, body, LOCAL_TIERS)

    assert config["modules"]["web_terminals"]["image_source"] == "local"


def test_the_propagated_leaf_does_not_displace_the_rest_of_the_module(tmp_path: Path) -> None:
    """The derived entry is applied after the profile's own, so it has to land
    inside the subtree those wrote rather than replacing it."""
    body = _profile({**DEPLOY_BLOCK, "image_source": "local"}, None)
    body["provider"] = "anthropic"
    body["model"] = "claude-haiku-4-5"
    body["config"] = {
        "modules.web_terminals": dict(WEB_TERMINALS_LOCAL),
    }

    web_terminals = _build_project(tmp_path, body, LOCAL_TIERS)["modules"]["web_terminals"]

    assert web_terminals["nginx_port"] == 20000
    assert web_terminals["web_base_port"] == 20100
    assert web_terminals["users"] == ["console"]


def test_a_project_with_no_web_stack_gains_no_web_terminals_module(tmp_path: Path) -> None:
    body = _profile({**DEPLOY_BLOCK, "image_source": "local"}, None)
    body["provider"] = "anthropic"
    body["model"] = "claude-haiku-4-5"

    config = _build_project(tmp_path, body)

    assert "web_terminals" not in config.get("modules", {})
