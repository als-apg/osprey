"""A built service template carries ``build:`` only while it runs the image OSPREY builds.

Compose tags a build with the service's ``image:``. A template that renders a
``build:`` block beside an operator's own ``services.<key>.image`` would build
OSPREY's recipe and tag it with the operator's name, so every template OSPREY
builds an image for gates its build on that image: unset, a null block, or a pin
equal to the packaged ``osprey_images.<key>`` keeps the build; any other image
renders none, and compose pulls and runs it as named.

The table below is checked against the default renders, so a newly built
template has to join it, and with it every case here.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest
import yaml
from test_render_defaults_golden import (
    TEMPLATES,
    _golden_name,
    _pinned_context,
    _probe_repo,
    _raw_default_config,
    _render_templates,
)

#: Service directory of every template OSPREY builds an image for, mapped to
#: that image's key in ``osprey_images``.
_BUILT_IMAGE_KEYS = {
    "bluesky": "bluesky_bridge",
    "bluesky_web": "bluesky_web",
    "event_dispatcher": "dispatch",
    "gchat_bridge": "gchat_bridge",
    "nextcloud_bridge": "nextcloud_bridge",
    "qmd": "qmd",
    "teams_bridge": "teams_bridge",
    "virtual_accelerator": "va",
}


def _template_for(key: str) -> str:
    """The templates-root-relative path of the template in directory *key*."""
    (rel,) = [rel for rel in TEMPLATES if PurePosixPath(rel).parent.name == key]
    return rel


def _services(rendered: str) -> dict:
    """The ``services`` mapping of one rendered compose document."""
    return (yaml.safe_load(rendered) or {}).get("services") or {}


def _render(key: str, block: dict | None, image_of=None) -> dict:
    """Render the template in directory *key* with ``services[key]`` set to *block*.

    *image_of*, when given, is called with the injected context and its result
    becomes the block's ``image`` (for a pin that has to name the context's own
    ``osprey_images`` value).
    """
    rel = _template_for(key)
    with _probe_repo() as repo_root:
        context = _pinned_context(_raw_default_config(repo_root))
        if image_of is not None:
            block = {**(block or {}), "image": image_of(context)}
        context["services"][key] = block
        return _services(_render_templates(context, [rel])[_golden_name(rel)])


def _builders(services: dict) -> list[str]:
    """The names of the rendered services that carry ``build``."""
    return [name for name, service in services.items() if "build" in (service or {})]


def test_the_built_set_is_every_template_whose_default_render_builds() -> None:
    with _probe_repo() as repo_root:
        rendered = _render_templates(_pinned_context(_raw_default_config(repo_root)), TEMPLATES)
    building = {
        PurePosixPath(rel).parent.name
        for rel in TEMPLATES
        if _builders(_services(rendered[_golden_name(rel)]))
    }
    assert building == set(_BUILT_IMAGE_KEYS)


@pytest.mark.parametrize("key", sorted(_BUILT_IMAGE_KEYS))
def test_a_pinned_foreign_image_renders_no_build(key: str) -> None:
    pinned = f"registry.example.org/site/{key}:pinned"
    services = _render(key, {"image": pinned})
    assert _builders(services) == []
    assert any(pinned in str((service or {}).get("image", "")) for service in services.values())


@pytest.mark.parametrize("key", sorted(_BUILT_IMAGE_KEYS))
def test_a_pin_naming_the_built_image_keeps_the_build(key: str) -> None:
    image_key = _BUILT_IMAGE_KEYS[key]
    services = _render(key, {}, image_of=lambda context: context["osprey_images"][image_key])
    assert len(_builders(services)) == 1


@pytest.mark.parametrize("key", sorted(_BUILT_IMAGE_KEYS))
def test_a_null_block_keeps_the_build(key: str) -> None:
    services = _render(key, None)
    assert len(_builders(services)) == 1
