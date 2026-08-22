"""The Bluesky stack refuses a podman host still on the legacy ``cni`` backend.

``templates/services/bluesky`` puts the bridge, the RE Manager and Redis on an
``internal: true`` network and dual-homes the queueserver. Rootless podman on
``cni`` ships no aardvark-dns, so a dual-homed container gets only its first
network's resolver: ``bluesky-queueserver`` can never resolve ``bluesky-redis``,
goes unhealthy, and ``osprey up`` aborts before the web slice renders. The whole
deployment is down and nothing osprey prints says why.

Refusing before any container-touching command turns that into one line naming
the host setting to change. Everything the preflight cannot establish -- a
docker host, a podman too old to report the field, a probe that errors or times
out -- lets the deploy through: a preflight that refused because it could not
interrogate the host would cause the outage it exists to prevent.
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.container_lifecycle import _preflight_bluesky_network_backend


def _config(*services: str) -> dict:
    """A deploy config deploying exactly ``services``."""
    return {"deployed_services": list(services)}


def _probe(backend: str | None, *, returncode: int = 0, raises: Exception | None = None):
    """Stand in for the ``podman info`` probe, answering ``backend``."""

    def fake_run(cmd, **kwargs):
        assert cmd[1:] == ["info", "--format", "{{.Host.NetworkBackend}}"], cmd
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout=(backend or ""), stderr="")

    return fake_run


@pytest.fixture
def podman(monkeypatch):
    """Resolve the container runtime to podman."""
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )


@pytest.mark.unit
def test_refuses_bluesky_on_cni(podman, monkeypatch):
    """The one case this preflight exists for: bluesky + podman + cni."""
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _probe("cni\n"))

    with pytest.raises(RuntimeError) as excinfo:
        _preflight_bluesky_network_backend(_config("bluesky", "jupyter"))

    message = str(excinfo.value)
    assert "netavark" in message
    assert "containers.conf" in message
    # The refusal has to name the symptom too, or an operator who has already
    # watched the queueserver go unhealthy cannot connect it to this line.
    assert "bluesky-redis" in message


@pytest.mark.unit
def test_allows_bluesky_on_netavark(podman, monkeypatch):
    """The supported host passes silently."""
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _probe("netavark\n"))

    _preflight_bluesky_network_backend(_config("bluesky"))


@pytest.mark.unit
def test_skips_when_bluesky_is_not_deployed(podman, monkeypatch):
    """Every other stack in this project runs fine on cni, so cni alone is not a refusal."""

    def unexpected(cmd, **kwargs):
        raise AssertionError(f"probed the host for a bluesky-less deploy: {cmd}")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", unexpected)

    _preflight_bluesky_network_backend(_config("jupyter", "openobserve"))
    _preflight_bluesky_network_backend({})


@pytest.mark.unit
def test_skips_on_docker(monkeypatch):
    """`podman info` is not a question a docker host is asked."""

    def unexpected(cmd, **kwargs):
        raise AssertionError(f"probed a docker host: {cmd}")

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", unexpected)

    _preflight_bluesky_network_backend(_config("bluesky"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"backend": "", "returncode": 1}, id="probe-failed"),
        pytest.param({"backend": "<no value>"}, id="podman-too-old-for-the-field"),
        pytest.param({"backend": ""}, id="empty-answer"),
        pytest.param(
            {"backend": None, "raises": subprocess.TimeoutExpired("podman", 10)},
            id="probe-timed-out",
        ),
        pytest.param(
            {"backend": None, "raises": OSError("podman vanished")}, id="runtime-disappeared"
        ),
    ],
)
def test_unreadable_backend_lets_the_deploy_through(podman, monkeypatch, kwargs):
    """No answer means no opinion: never block a deploy on a host we cannot read."""
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _probe(**kwargs))

    _preflight_bluesky_network_backend(_config("bluesky"))


@pytest.mark.unit
def test_unknown_backend_warns_rather_than_refuses(podman, monkeypatch):
    """A backend nobody here has seen gets a warning, not a guess dressed as a refusal."""
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _probe("slirp-of-the-future\n"))
    warnings = []
    monkeypatch.setattr(
        container_lifecycle,
        "_warn_fact",
        lambda summary, detail=None, remedy=None: warnings.append((summary, detail)),
    )

    _preflight_bluesky_network_backend(_config("bluesky"))

    assert len(warnings) == 1
    assert "slirp-of-the-future" in warnings[0][0]


@pytest.mark.unit
def test_no_runtime_defers_to_the_runtime_check(monkeypatch):
    """A host with no usable runtime is diagnosed by verify_runtime_is_running, not here."""

    def no_runtime(config):
        raise RuntimeError("No container runtime found")

    monkeypatch.setattr(container_lifecycle, "get_runtime_command", no_runtime)

    _preflight_bluesky_network_backend(_config("bluesky"))
