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

from osprey.deployment.runtime_helper import (
    ComposeProvider,
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
# podman_compose_provider_advisory -- a pure function of the resolved pairing


def test_advises_on_podman_with_the_docker_compose_provider() -> None:
    """The broken pairing is named before any image is built."""
    advisory = podman_compose_provider_advisory("podman", ComposeProvider.DOCKER_V2)

    assert advisory is not None
    assert "podman-compose" in advisory


def test_no_advisory_for_podman_with_podman_compose() -> None:
    """The supported podman pairing is silent -- this is the CI configuration."""
    assert podman_compose_provider_advisory("podman", ComposeProvider.PODMAN_COMPOSE) is None


def test_no_advisory_for_docker() -> None:
    """Docker Compose v2 behind the docker runtime is the ordinary case."""
    assert podman_compose_provider_advisory("docker", ComposeProvider.DOCKER_V2) is None
    assert podman_compose_provider_advisory("docker", ComposeProvider.PODMAN_COMPOSE) is None


def test_the_advisory_never_probes() -> None:
    """It must not shell out: the lifecycle site that calls it forbids that.

    The deploy has already resolved both values by the time the preflight runs,
    and a probe from here would reach past the module-bound `get_runtime_command`
    the lifecycle tests patch -- which is exactly how this function's first
    version broke twenty-three of them.
    """
    import subprocess

    def _forbidden(*args, **kwargs):
        raise AssertionError(f"the advisory ran a subprocess: {args!r}")

    original = subprocess.run
    subprocess.run = _forbidden
    try:
        assert podman_compose_provider_advisory("podman", ComposeProvider.DOCKER_V2) is not None
        assert podman_compose_provider_advisory("docker", ComposeProvider.DOCKER_V2) is None
    finally:
        subprocess.run = original


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
# the preflight itself -- resolution through the seam the lifecycle tests patch


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
        container_lifecycle, "get_runtime_command", lambda config=None: ["podman", "compose"]
    )

    container_lifecycle._preflight_podman_compose_provider({}, ComposeProvider.DOCKER_V2)

    assert len(warned) == 1
    summary, detail, remedy = warned[0]
    assert "podman-compose" in summary
    assert "empty credential" in detail
    assert "containers.conf" in remedy


def test_preflight_is_silent_on_the_supported_pairing(monkeypatch) -> None:
    """podman served by podman-compose -- the CI configuration -- says nothing."""
    from osprey.deployment import container_lifecycle

    warned: list[tuple] = []
    monkeypatch.setattr(container_lifecycle, "_warn_fact", lambda *a, **k: warned.append(a))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config=None: ["podman", "compose"]
    )

    container_lifecycle._preflight_podman_compose_provider({}, ComposeProvider.PODMAN_COMPOSE)
    assert warned == []


def test_preflight_defers_to_the_refusal_that_owns_a_dead_runtime(monkeypatch) -> None:
    """A host with no usable runtime gets no advisory, and no exception.

    verify_runtime_is_running reports that far better than a provider aside
    could, and raising from here would bury the refusal that stops the deploy.
    """
    from osprey.deployment import container_lifecycle

    warned: list[tuple] = []
    monkeypatch.setattr(container_lifecycle, "_warn_fact", lambda *a, **k: warned.append(a))

    def _raise(*_args, **_kwargs):
        raise RuntimeError("no usable runtime")

    monkeypatch.setattr(container_lifecycle, "get_runtime_command", _raise)

    container_lifecycle._preflight_podman_compose_provider({}, ComposeProvider.DOCKER_V2)
    assert warned == []
