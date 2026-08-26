"""Tests for the podman + Docker-Compose-v2 registry-auth diagnosis.

``podman compose`` is a dispatcher: it hands the work to whichever external
provider the host has configured. When that provider is the Docker Compose v2
CLI plugin, a ``build`` that has to fetch a base image reaches the registry with
an EMPTY credential rather than with no credential at all, and Docker Hub
answers ``401 incorrect username or password``. An anonymous pull would have
succeeded; supplying ``""``/``""`` is what turns it into a refusal.

Nothing in the resulting message names podman, the provider, or the fact that a
credential was involved, so the failure reads as a wrong password nobody typed.
These tests cover the two surfaces that make it legible: an up-front advisory
naming the provider pairing, and a translation of the raw registry error into
the two things that actually resolve it.
"""

from __future__ import annotations

import pytest

from osprey.deployment import runtime_helper
from osprey.deployment.runtime_helper import (
    ComposeProvider,
    ComposeProviderInfo,
    UnsupportedComposeProviderError,
    diagnose_build_failure,
    podman_compose_provider_advisory,
)

# The failure exactly as podman renders it, from a real `osprey up` on a macOS
# host whose `podman compose` resolved to Docker Desktop's compose plugin.
REAL_FAILURE = (
    "STEP 1/14: FROM --platform=linux/amd64 python:3.11-slim\n"
    "Trying to pull docker.io/library/python:3.11-slim...\n"
    "creating build container: unable to copy from source "
    "docker://python:3.11-slim: initializing source docker://python:3.11-slim: "
    "fetching manifest 3.11-slim in docker.io/library/python: unable to "
    "retrieve auth token: invalid username/password: unauthorized: incorrect "
    "username or password\n"
)


def _provider(provider: ComposeProvider) -> ComposeProviderInfo:
    """A minimal ComposeProviderInfo carrying just the provider identity."""
    return ComposeProviderInfo(
        provider=provider,
        version=(2, 24, 5),
        version_text="2.24.5",
        banner="probe banner",
    )


def _pin(monkeypatch, runtime: str, provider: ComposeProvider | None) -> None:
    """Pin the resolved runtime and, unless None, the detected provider."""
    monkeypatch.setattr(
        runtime_helper, "get_runtime_command", lambda config=None: [runtime, "compose"]
    )
    if provider is not None:
        monkeypatch.setattr(
            runtime_helper,
            "detect_compose_provider",
            lambda cmd=None, config=None: _provider(provider),
        )


# ---------------------------------------------------------------------------
# diagnose_build_failure -- translating the raw registry refusal


def test_diagnoses_the_real_registry_auth_refusal() -> None:
    """The captured 401 is recognised and answered with both real remedies."""
    remedy = diagnose_build_failure(REAL_FAILURE)

    assert remedy is not None
    # Names the mechanism rather than restating the registry's wording.
    assert "empty" in remedy.lower()
    # Both escapes an operator actually has.
    assert "podman-compose" in remedy
    assert "docker" in remedy.lower()


def test_diagnosis_ignores_unrelated_build_failures() -> None:
    """A build that failed for any other reason gets no opinion."""
    assert diagnose_build_failure("STEP 4/13: RUN pip install\nERROR: no matching dist") is None
    assert diagnose_build_failure("") is None


def test_diagnosis_ignores_a_genuine_bad_credential() -> None:
    """A real `podman login` rejection is not this bug and must not claim to be.

    The registry says the same words when a human typed the wrong password. The
    signature therefore requires the build-time shape -- a manifest fetch during
    an image pull -- not merely the 401 text.
    """
    assert (
        diagnose_build_failure('Error: logging into "docker.io": invalid username/password') is None
    )


def test_diagnosis_survives_a_partial_or_wrapped_log() -> None:
    """Matching is substring-based, so log decoration does not defeat it."""
    noisy = "\x1b[0m " + REAL_FAILURE.replace("\n", "\r\n") + "\nError: exit status 1\n"
    assert diagnose_build_failure(noisy) is not None


# ---------------------------------------------------------------------------
# podman_compose_provider_advisory -- the up-front warning


def test_advises_on_podman_with_the_docker_compose_provider(monkeypatch) -> None:
    """The broken pairing is named before any image is built."""
    _pin(monkeypatch, "podman", ComposeProvider.DOCKER_V2)

    advisory = podman_compose_provider_advisory({})

    assert advisory is not None
    assert "podman-compose" in advisory


def test_no_advisory_for_podman_with_podman_compose(monkeypatch) -> None:
    """The supported podman pairing is silent -- this is the CI configuration."""
    _pin(monkeypatch, "podman", ComposeProvider.PODMAN_COMPOSE)
    assert podman_compose_provider_advisory({}) is None


def test_no_advisory_for_docker(monkeypatch) -> None:
    """Docker Compose v2 behind the docker runtime is the ordinary case."""
    _pin(monkeypatch, "docker", ComposeProvider.DOCKER_V2)
    assert podman_compose_provider_advisory({}) is None


@pytest.mark.parametrize(
    "boom",
    [RuntimeError("no runtime"), UnsupportedComposeProviderError("unsupported")],
)
def test_advisory_stays_silent_when_the_host_cannot_be_interrogated(monkeypatch, boom) -> None:
    """Total, like _podman_network_backend: no answer means no opinion.

    An advisory that raised would become the outage it exists to describe, and
    the callers that own these two failures say them far better than a
    provider note can.
    """

    def _raise(*_args, **_kwargs):
        raise boom

    monkeypatch.setattr(runtime_helper, "get_runtime_command", _raise)
    assert podman_compose_provider_advisory({}) is None

    _pin(monkeypatch, "podman", None)
    monkeypatch.setattr(runtime_helper, "detect_compose_provider", _raise)
    assert podman_compose_provider_advisory({}) is None


# ---------------------------------------------------------------------------
# diagnose_captured_failure -- the one seam both building verbs share


def test_captured_failure_translates_a_spooled_registry_refusal(tmp_path) -> None:
    """A CapturedProcessError's spool is read and answered."""
    from osprey.deployment.errors import CapturedProcessError
    from osprey.deployment.subprocess_capture import diagnose_captured_failure

    spool = tmp_path / "build-services.log"
    spool.write_text(REAL_FAILURE, encoding="utf-8")

    exc = CapturedProcessError(["podman", "compose", "build"], 1, spool)
    assert diagnose_captured_failure(exc) == diagnose_build_failure(REAL_FAILURE)


def test_captured_failure_is_silent_without_a_spool() -> None:
    """A --verbose run has no spool, and any other exception has no attribute."""
    from osprey.deployment.errors import CapturedProcessError
    from osprey.deployment.subprocess_capture import diagnose_captured_failure

    assert diagnose_captured_failure(CapturedProcessError(["x"], 1)) is None
    assert diagnose_captured_failure(RuntimeError("unrelated")) is None


def test_captured_failure_is_silent_on_an_unreadable_spool(tmp_path) -> None:
    """A spool that has since been pruned must not take down the error path."""
    from osprey.deployment.errors import CapturedProcessError
    from osprey.deployment.subprocess_capture import diagnose_captured_failure

    exc = CapturedProcessError(["x"], 1, tmp_path / "vanished.log")
    assert diagnose_captured_failure(exc) is None


# ---------------------------------------------------------------------------
# the preflight itself


def test_preflight_warns_once_on_the_broken_pairing(monkeypatch) -> None:
    """The advisory reaches the operator through the lifecycle's warn seam."""
    from osprey.deployment import container_lifecycle

    warned: list[tuple] = []
    monkeypatch.setattr(
        container_lifecycle,
        "_warn_fact",
        lambda summary, detail=None, remedy=None: warned.append((summary, detail, remedy)),
    )
    monkeypatch.setattr(
        container_lifecycle, "podman_compose_provider_advisory", lambda config: "the advisory"
    )

    container_lifecycle._preflight_podman_compose_provider({})

    assert len(warned) == 1
    summary, detail, remedy = warned[0]
    assert "podman-compose" in summary
    assert detail == "the advisory"
    assert "containers.conf" in remedy


def test_preflight_is_silent_when_there_is_no_advisory(monkeypatch) -> None:
    """Every host but the broken pairing sees nothing at all."""
    from osprey.deployment import container_lifecycle

    warned: list[tuple] = []
    monkeypatch.setattr(
        container_lifecycle,
        "_warn_fact",
        lambda *a, **k: warned.append(a),
    )
    monkeypatch.setattr(
        container_lifecycle, "podman_compose_provider_advisory", lambda config: None
    )

    container_lifecycle._preflight_podman_compose_provider({})
    assert warned == []
