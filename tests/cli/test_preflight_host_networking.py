"""The preflight word about a Docker Desktop that cannot forward a host port.

Why this check exists at all is a timing argument, not a correctness one. The
post-up probe in ``web_terminals/postup_hooks.py`` already finds this state, and
finds it by testing the port rather than a setting, so it is the authority. What
it cannot be is early: it runs after every image is built and every container is
up. On a first deploy that is a quarter of an hour spent building images for a
web tier that was never going to be reachable, when the operator could have
fixed it with one checkbox beforehand.

So what is pinned here is the discipline that makes an early warning bearable:
it speaks only when Docker Desktop has been asked and has said no, and it never
refuses a deploy.
"""

from __future__ import annotations

import pytest

from osprey.cli import deploy_cmd
from osprey.deployment import docker_desktop

pytestmark = pytest.mark.unit

#: A config with web terminals rendered, which is what a deployment that owns a
#: reachable landing page looks like.
_WITH_WEB_TERMINALS = {"modules": {"web_terminals": {"nginx_port": 9080}}}


@pytest.fixture
def warned(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None, str | None]]:
    """Collect the promoted warning instead of printing it."""
    calls: list[tuple[str, str | None, str | None]] = []

    def record(summary: str, detail: str | None = None, remedy: str | None = None) -> None:
        calls.append((summary, detail, remedy))

    monkeypatch.setattr(deploy_cmd, "_warn_fact", record)
    return calls


@pytest.fixture
def on_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A macOS/Windows host running Docker Desktop."""
    monkeypatch.setattr(docker_desktop, "on_docker_desktop", lambda config: True)


def _setting(monkeypatch: pytest.MonkeyPatch, value: bool | None) -> None:
    """What Docker Desktop reports, including "cannot tell"."""
    monkeypatch.setattr(docker_desktop, "host_networking_enabled", lambda: value)


class TestWhenItSpeaks:
    def test_a_disabled_forwarder_is_reported_before_anything_is_built(
        self, monkeypatch, warned, on_desktop
    ):
        """The whole point: the operator hears this while it is still cheap."""
        _setting(monkeypatch, False)

        deploy_cmd._warn_if_host_networking_is_off(_WITH_WEB_TERMINALS)

        assert len(warned) == 1
        summary, detail, remedy = warned[0]
        assert "web terminals" in summary
        assert "9080" in detail
        assert "Enable host networking" in remedy

    def test_the_warning_says_the_rest_of_the_deployment_is_fine(
        self, monkeypatch, warned, on_desktop
    ):
        """Only the host-networked tier is affected. Everything else publishes
        its ports and works, and an operator deciding whether to stop and fix
        this now needs to know that."""
        _setting(monkeypatch, False)

        deploy_cmd._warn_if_host_networking_is_off(_WITH_WEB_TERMINALS)

        assert "unaffected" in warned[0][1]


class TestWhenItStaysQuiet:
    def test_an_enabled_forwarder_says_nothing(self, monkeypatch, warned, on_desktop):
        _setting(monkeypatch, True)

        deploy_cmd._warn_if_host_networking_is_off(_WITH_WEB_TERMINALS)

        assert warned == []

    def test_a_setting_that_cannot_be_read_says_nothing(self, monkeypatch, warned, on_desktop):
        """The discipline that keeps this warning worth reading. Guessing here
        would warn on every host this reader cannot interrogate, and the post-up
        probe still catches the real thing by testing the port."""
        _setting(monkeypatch, None)

        deploy_cmd._warn_if_host_networking_is_off(_WITH_WEB_TERMINALS)

        assert warned == []

    def test_a_host_that_is_not_docker_desktop_says_nothing(self, monkeypatch, warned):
        """On Linux the port is bound on the machine itself and this setting does
        not exist."""
        monkeypatch.setattr(docker_desktop, "on_docker_desktop", lambda config: False)
        _setting(monkeypatch, False)

        deploy_cmd._warn_if_host_networking_is_off(_WITH_WEB_TERMINALS)

        assert warned == []

    def test_docker_desktop_is_not_even_asked_off_desktop(self, monkeypatch, warned):
        """The read dials a socket, so it is not work a Linux host should do."""
        monkeypatch.setattr(docker_desktop, "on_docker_desktop", lambda config: False)

        def fail() -> bool:
            raise AssertionError("the setting was read on a host it cannot apply to")

        monkeypatch.setattr(docker_desktop, "host_networking_enabled", fail)

        deploy_cmd._warn_if_host_networking_is_off(_WITH_WEB_TERMINALS)

        assert warned == []

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="no-modules"),
            pytest.param({"modules": {}}, id="no-web-terminals"),
            pytest.param({"modules": {"web_terminals": {}}}, id="no-nginx-port"),
            pytest.param({"modules": None}, id="null-modules"),
        ],
    )
    def test_a_deployment_without_web_terminals_says_nothing(
        self, monkeypatch, warned, on_desktop, config
    ):
        """Nothing in such a deployment binds a host-networked port, so the
        setting cannot affect it."""
        _setting(monkeypatch, False)

        deploy_cmd._warn_if_host_networking_is_off(config)

        assert warned == []


class TestItRefusesNothing:
    def test_the_check_returns_rather_than_raising(self, monkeypatch, warned, on_desktop):
        """An operator who wants the backend services has a working deploy even
        with the web terminals dark, so this is advisory like the probe it
        front-runs. Pinned because turning a warning into a refusal is a
        one-word change that would strand exactly those operators."""
        _setting(monkeypatch, False)

        assert deploy_cmd._warn_if_host_networking_is_off(_WITH_WEB_TERMINALS) is None
