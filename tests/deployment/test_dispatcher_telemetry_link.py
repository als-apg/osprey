"""The dispatcher dashboard's telemetry link follows the deployment's origin.

The link is rendered into the compose file at build time and followed in the
OPERATOR's browser, which sits outside the compose network. It used to be
``http://localhost:<port>`` unconditionally — correct only where the operator's
browser runs on the host that publishes the store. Anywhere else it named the
operator's own machine, and the link either failed or, worse, reached a store
that is not this deployment's.

The rule these tests hold it to:

* a loopback-bound store keeps ``localhost`` — a browser reaching that dashboard
  is on the host anyway, so it is the correct name for it;
* an exposed store takes the host of the deployment's declared external origin
  (``modules.web_terminals.external_origin``, else ``deploy.fqdn``), the same
  authority every other browser-facing URL comes from;
* an exposed store with no derivable origin renders NOTHING. An unset variable
  already means "hide the link", which beats a link nothing serves.
"""

from __future__ import annotations

import pytest

from osprey.deployment.compose_generator import _telemetry_link_host

TELEMETRY_ON = {
    "deployed_services": ["openobserve"],
    "claude_code": {"telemetry": {"enabled": True}},
}


def _config(bind: str, **root: object) -> dict:
    return {**TELEMETRY_ON, "deployment": {"bind_address": bind}, **root}


class TestLinkHost:
    """What the build-time derivation resolves the link's host to."""

    @pytest.mark.parametrize("bind", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_store_keeps_localhost(self, bind: str) -> None:
        assert _telemetry_link_host(_config(bind)) == "localhost"

    def test_unset_bind_is_loopback(self) -> None:
        assert _telemetry_link_host(dict(TELEMETRY_ON)) == "localhost"

    def test_exposed_store_takes_the_declared_external_origin(self) -> None:
        config = _config(
            "0.0.0.0",
            modules={"web_terminals": {"external_origin": "https://terminals.example.org:8443"}},
        )
        assert _telemetry_link_host(config) == "terminals.example.org"

    def test_exposed_store_falls_back_to_the_deploy_fqdn(self) -> None:
        config = _config("0.0.0.0", deploy={"fqdn": "ctl-01.example.org"})
        assert _telemetry_link_host(config) == "ctl-01.example.org"

    def test_exposed_store_with_no_derivable_origin_emits_nothing(self) -> None:
        """Unset already means "hide the link"; a guess would be worse."""
        assert _telemetry_link_host(_config("0.0.0.0")) is None

    def test_pinned_interface_is_still_not_loopback(self) -> None:
        config = _config("10.0.0.7", deploy={"fqdn": "ctl-01.example.org"})
        assert _telemetry_link_host(config) == "ctl-01.example.org"


class TestRender:
    """What the dispatcher compose template does with it."""

    def _render(self, host: str | None) -> str:
        from tests.deployment.test_compose_generator import (
            _dispatcher_context,
            _packaged_compose_template,
        )

        context = _dispatcher_context()
        context.update(
            TELEMETRY_ON,
            services={**context["services"], "openobserve": {"port": 15080}},
            osprey_telemetry_host=host,
        )
        return _packaged_compose_template("services/event_dispatcher/docker-compose.yml.j2").render(
            **context
        )

    def test_host_is_rendered_into_the_url(self) -> None:
        assert 'OSPREY_TELEMETRY_URL: "http://ctl-01.example.org:15080"' in self._render(
            "ctl-01.example.org"
        )

    def test_no_host_renders_no_variable(self) -> None:
        assert "OSPREY_TELEMETRY_URL" not in self._render(None)
