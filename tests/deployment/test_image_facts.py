"""The image facts: one Python, and the platform the profile names or this host's."""

from __future__ import annotations

import pytest

from osprey.deployment import image_facts
from osprey.deployment.image_facts import (
    IMAGE_PLATFORMS,
    fit_platform,
    stated_image_platform,
    wheel_arch,
)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_an_unstated_platform_is_none(value: object) -> None:
    assert stated_image_platform(value) is None


@pytest.mark.parametrize("value", IMAGE_PLATFORMS)
def test_an_accepted_platform_is_stated(value: str) -> None:
    assert stated_image_platform(value) == value


@pytest.mark.parametrize("value", ["linux/riscv64", "amd64", "linux/x86_64", 64])
def test_any_other_platform_is_refused_naming_the_key_and_the_accepted_values(
    value: object,
) -> None:
    with pytest.raises(ValueError) as refused:
        stated_image_platform(value)

    message = str(refused.value)
    assert "images.platform" in message
    assert all(accepted in message for accepted in IMAGE_PLATFORMS)


def test_a_stated_platform_is_the_fit_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_facts.platform, "machine", lambda: "arm64")

    assert fit_platform("linux/amd64") == "linux/amd64"


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "linux/amd64"),
        ("amd64", "linux/amd64"),
        ("AMD64", "linux/amd64"),
        ("aarch64", "linux/arm64"),
        ("arm64", "linux/arm64"),
        ("ppc64le", "linux/ppc64le"),
    ],
)
def test_unstated_the_fit_platform_is_this_hosts(
    monkeypatch: pytest.MonkeyPatch, machine: str, expected: str
) -> None:
    monkeypatch.setattr(image_facts.platform, "machine", lambda: machine)

    assert fit_platform(None) == expected


@pytest.mark.parametrize(
    ("image_platform", "arch"),
    [("linux/amd64", "x86_64"), ("linux/arm64", "aarch64"), ("linux/ppc64le", "ppc64le")],
)
def test_the_wheel_arch_is_the_platforms(image_platform: str, arch: str) -> None:
    assert wheel_arch(image_platform) == arch
