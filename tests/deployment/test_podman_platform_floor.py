"""The ``osprey up`` preflight refusing a stated platform on an old podman-compose.

podman-compose passes a service's ``platform:`` to ``podman build`` only from
1.3.0; older releases apply it to ``podman create`` alone, so a compose build
would silently produce host-native images while the ``FROM`` names another
platform. The preflight refuses that pairing before anything is built.
"""

from __future__ import annotations

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.runtime_helper import (
    ComposeProvider,
    ComposeProviderInfo,
    UnsupportedComposeProviderError,
)

PLATFORM_CONFIG = {"images": {"platform": "linux/amd64"}}


@pytest.fixture
def podman_compose_host(monkeypatch):
    """A host whose ``podman compose`` reports the podman-compose version set on it."""
    state = {"version": "1.2.0"}

    def detect(*_args, **_kwargs):
        text = state["version"]
        return ComposeProviderInfo(
            provider=ComposeProvider.PODMAN_COMPOSE,
            version=tuple(int(part) for part in text.split(".")),
            version_text=text,
            banner=f"podman-compose version {text}",
        )

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config=None: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "detect_compose_provider", detect)
    monkeypatch.setattr(container_lifecycle, "_resolve_prebuilt_images", lambda config: False)
    return state


@pytest.mark.usefixtures("podman_compose_host")
def test_old_podman_compose_with_a_platform_is_refused() -> None:
    with pytest.raises(UnsupportedComposeProviderError, match="1.3.0"):
        container_lifecycle._preflight_podman_compose_platform(
            PLATFORM_CONFIG, ComposeProvider.PODMAN_COMPOSE
        )


def test_podman_compose_that_builds_for_the_platform_is_silent(podman_compose_host) -> None:
    podman_compose_host["version"] = "1.3.0"
    container_lifecycle._preflight_podman_compose_platform(
        PLATFORM_CONFIG, ComposeProvider.PODMAN_COMPOSE
    )


def test_unset_platform_is_silent_and_asks_the_host_nothing(monkeypatch) -> None:
    def probe(*_args, **_kwargs):
        raise AssertionError("an unset platform must not probe the compose provider")

    monkeypatch.setattr(container_lifecycle, "detect_compose_provider", probe)
    monkeypatch.setattr(container_lifecycle, "_resolve_prebuilt_images", lambda config: False)
    container_lifecycle._preflight_podman_compose_platform({}, ComposeProvider.PODMAN_COMPOSE)


@pytest.mark.usefixtures("podman_compose_host")
@pytest.mark.parametrize("provider", [ComposeProvider.DOCKER_V2, None])
def test_other_providers_are_silent(provider) -> None:
    container_lifecycle._preflight_podman_compose_platform(PLATFORM_CONFIG, provider)


@pytest.mark.usefixtures("podman_compose_host")
def test_prebuilt_images_are_silent(monkeypatch) -> None:
    monkeypatch.setattr(container_lifecycle, "_resolve_prebuilt_images", lambda config: True)
    container_lifecycle._preflight_podman_compose_platform(
        PLATFORM_CONFIG, ComposeProvider.PODMAN_COMPOSE
    )
