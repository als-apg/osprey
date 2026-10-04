"""A registry-mode deploy block names the web tier's registry once.

The web tier names every terminal image from the rendered top-level
``registry.url``. A profile whose ``deploy:`` block pulls images from a registry
already says where that registry is, in ``deploy.registry.url``, so the build
writes that value into the rendered ``registry.url`` whenever the profile's
``config:`` block names none. A ``config:`` block that spells ``registry.url``
in any form keeps its own value: the fill never overwrites and never refuses.
Local mode reads no registry, so nothing is filled there.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_deploy import (
    IMAGE_SOURCE_CONFIG_KEY,
    REGISTRY_URL_CONFIG_KEY,
    config_registry_url_spelling,
    deploy_aware_config_errors,
    deploy_aware_config_warnings,
    deploy_config_overrides,
    parse_deploy_block,
)
from osprey.deployment.web_terminals.lint import profile_config_errors

REGISTRY_URL = "registry.example.org/accelerator/demo"

DEPLOY_BLOCK: dict[str, Any] = {
    "ci": "gitlab",
    "image_source": "registry",
    "registry": {"url": REGISTRY_URL},
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
    "users": ["operator"],
}

#: The same stack with a persona catalog.
WEB_TERMINALS_CATALOG: dict[str, Any] = {
    **WEB_TERMINALS,
    "default_persona": "readwrite",
    "personas": {"readwrite": {"build_profile": "hello-world", "project": "demo-readwrite"}},
}


def _parsed(deploy: dict[str, Any]) -> Any:
    return parse_deploy_block({"name": "Demo", "data": "data", "deploy": deploy})


def _web_stack(**extra: Any) -> dict[str, Any]:
    return {"modules.web_terminals": dict(WEB_TERMINALS), **extra}


def test_registry_mode_fills_the_rendered_registry_url() -> None:
    overrides = deploy_config_overrides(_parsed(DEPLOY_BLOCK), _web_stack())

    assert overrides == {
        IMAGE_SOURCE_CONFIG_KEY: "registry",
        REGISTRY_URL_CONFIG_KEY: REGISTRY_URL,
    }


@pytest.mark.parametrize(
    "spelled",
    [
        {"registry.url": "mirror.example.org/x"},
        {"registry": {"url": "mirror.example.org/x"}},
        {"registry": None},
    ],
    ids=["dotted-leaf", "nested-mapping", "nested-null"],
)
def test_an_explicit_config_registry_url_wins(spelled: dict[str, Any]) -> None:
    overrides = deploy_config_overrides(_parsed(DEPLOY_BLOCK), _web_stack(**spelled))

    assert REGISTRY_URL_CONFIG_KEY not in overrides
    assert overrides[IMAGE_SOURCE_CONFIG_KEY] == "registry"


def test_a_registry_mapping_without_a_url_is_filled() -> None:
    config = _web_stack(registry={"token_env_var": "X"})

    overrides = deploy_config_overrides(_parsed(DEPLOY_BLOCK), config)

    assert overrides[REGISTRY_URL_CONFIG_KEY] == REGISTRY_URL


def test_local_mode_fills_nothing_and_draws_no_unused_url_warning() -> None:
    deploy = _parsed({**DEPLOY_BLOCK, "image_source": "local"})
    config = {"modules.web_terminals": dict(WEB_TERMINALS_CATALOG)}

    assert deploy_config_overrides(deploy, config) == {IMAGE_SOURCE_CONFIG_KEY: "local"}
    warnings = deploy_aware_config_warnings(deploy, config)
    assert not any("registry.url" in message for message in warnings), warnings


def test_no_deploy_block_or_no_web_stack_contributes_nothing() -> None:
    deploy = _parsed(DEPLOY_BLOCK)

    assert deploy_config_overrides(None, _web_stack()) == {}
    assert deploy_config_overrides(deploy, {"control_system.type": "epics"}) == {}
    assert deploy_config_overrides(deploy, {}) == {}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, None),
        ({"registry.url": "r"}, "registry.url"),
        ({"registry": {"url": "r"}}, "registry: url"),
        ({"registry": None}, "registry"),
        ({"registry": {"token_env_var": "X"}}, None),
        ([], None),
    ],
    ids=[
        "absent",
        "dotted-leaf",
        "nested-mapping",
        "nested-null",
        "mapping-without-url",
        "not-a-dict",
    ],
)
def test_the_spelling_probe(config: Any, expected: str | None) -> None:
    assert config_registry_url_spelling(config) == expected


def test_the_filled_url_satisfies_the_lint() -> None:
    deploy = _parsed(DEPLOY_BLOCK)
    config = {"modules.web_terminals": dict(WEB_TERMINALS_CATALOG)}

    merged = deploy_aware_config_errors(deploy, config)
    raw = profile_config_errors(config)

    assert merged == [], merged
    assert any("registry.url is not set" in message for message in raw), raw
