"""Tests for the core ``web_panels`` health category.

Drives the category's concurrent panel probes through an injected
:class:`httpx.MockTransport`, exercising target resolution (built-in panels via
the web-server registry, custom panels via their configured url), the
health-endpoint vs reachability-stand-in distinction, port overrides from
config, and the status mapping for reachable / degraded / unreachable panels.
"""

from __future__ import annotations

import httpx
import pytest

from osprey.health.core.web_panels import CATEGORY, web_panels
from osprey.health.models import CheckResult, Status


async def _run(config, *, handler=None) -> dict[str, CheckResult]:
    """Run the category against a mock transport, keyed by row name."""
    transport = httpx.MockTransport(handler or (lambda r: httpx.Response(200, text="ok")))
    results = await web_panels(config, transport=transport)()
    assert isinstance(results, list)
    assert all(r.category == CATEGORY for r in results)
    return {r.name: r for r in results}


def _cfg(panels: dict, **extra) -> dict:
    cfg: dict = {"web": {"panels": panels}}
    cfg.update(extra)
    return cfg


class TestTargetResolution:
    async def test_no_web_block_contributes_no_rows(self):
        """No `web:` block ⇒ no web terminal in this build ⇒ nothing to report."""
        assert await _run({}) == {}

    @pytest.mark.parametrize("web", [{}, {"panels": {}}])
    async def test_artifacts_is_probed_whenever_the_terminal_is_configured(self, web):
        """`artifacts` is a UNIVERSAL panel — on whether or not it is listed.

        Mirrors ``web_terminal.app._load_panel_config``, which seeds `enabled`
        with ``UNIVERSAL_PANELS`` before reading ``web.panels`` at all, so a
        bare ``web:`` block still runs the workspace panel.
        """
        rows = await _run({"web": web})
        assert set(rows) == {"web_panels.artifacts"}

    async def test_builtin_panels_probe_their_registry_health_url(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        await _run(_cfg({"ariel": {"enabled": True}, "okf": True}), handler=handler)

        assert "http://127.0.0.1:8085/health" in seen  # ariel port_default
        assert "http://127.0.0.1:8093/health" in seen  # okf port_default
        assert "http://127.0.0.1:8086/health" in seen  # artifacts (universal)

    async def test_disabled_builtin_is_not_probed(self):
        rows = await _run(_cfg({"ariel": {"enabled": False}}))
        assert "web_panels.ariel" not in rows

    async def test_config_port_override_is_honoured(self):
        """A facility that moves a panel's port is probed where it actually runs."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        cfg = _cfg({"ariel": True}, ariel={"web": {"port": 9999, "host": "10.0.0.5"}})
        await _run(cfg, handler=handler)

        assert "http://10.0.0.5:9999/health" in seen

    async def test_custom_panel_uses_its_health_endpoint(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        await _run(
            _cfg(
                {
                    "events": {
                        "label": "EVENTS",
                        "url": "http://localhost:8020",
                        "path": "/dashboard",
                        "health_endpoint": "/health",
                    }
                }
            ),
            handler=handler,
        )
        assert "http://localhost:8020/health" in seen

    async def test_custom_panel_without_health_endpoint_falls_back_to_its_path(self):
        """plan/results declare no health endpoint — probe the entry path instead."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        rows = await _run(
            _cfg({"plan": {"label": "PLAN", "url": "http://localhost:8095", "path": "/plan/"}}),
            handler=handler,
        )
        assert "http://localhost:8095/plan/" in seen
        assert rows["web_panels.plan"].status is Status.OK
        assert "no health endpoint configured" in rows["web_panels.plan"].message

    async def test_custom_panel_without_url_is_skipped(self):
        rows = await _run(_cfg({"broken": {"label": "BROKEN"}}))
        assert "web_panels.broken" not in rows

    async def test_row_names_are_category_prefixed_with_the_panel_id(self):
        """`<category>.<id>` so the dashboard's fmtName() strips the prefix."""
        rows = await _run(_cfg({"channel-finder": True}))
        assert "web_panels.channel_finder" in rows


class TestStatusMapping:
    async def test_reachable_panel_is_ok_with_latency(self):
        rows = await _run(_cfg({"ariel": True}))
        row = rows["web_panels.ariel"]
        assert row.status is Status.OK
        assert row.value == "up"
        assert row.latency_ms is not None
        assert "ARIEL" in row.message

    async def test_unreachable_panel_warns_and_never_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        rows = await _run(_cfg({"ariel": True}), handler=handler)
        row = rows["web_panels.ariel"]
        assert row.status is Status.WARNING
        assert row.value == "offline"
        assert "unreachable" in row.message
        assert "web.panels" in row.details

    @pytest.mark.parametrize("code", [500, 503])
    async def test_server_error_is_degraded_for_both_probe_kinds(self, code):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(code)

        rows = await _run(
            _cfg(
                {
                    "ariel": True,
                    "plan": {"url": "http://localhost:8095", "path": "/plan/"},
                }
            ),
            handler=handler,
        )
        assert rows["web_panels.ariel"].status is Status.WARNING
        assert rows["web_panels.plan"].status is Status.WARNING
        assert rows["web_panels.ariel"].value == "degraded"

    async def test_health_endpoint_must_answer_2xx_but_a_path_probe_need_not(self):
        """A 404 fails a declared health contract; a panel path may legitimately 404-free.

        The stand-in only asserts "something is listening", so any non-5xx
        answer counts — that is the whole difference between the two kinds.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        rows = await _run(
            _cfg(
                {
                    "ariel": True,
                    "plan": {"url": "http://localhost:8095", "path": "/plan/"},
                }
            ),
            handler=handler,
        )
        assert rows["web_panels.ariel"].status is Status.WARNING
        assert rows["web_panels.plan"].status is Status.OK


class TestBuiltinAddressResolution:
    """The probe must resolve addresses through the one canonical resolver.

    Re-deriving host/port from the config section alone made the multi-user
    deployment unprobeable: compose exports ``OSPREY_<CONFIG_KEY>_PORT`` per user
    (per-user containers share the host network namespace), so the panel listens
    on the env port while the probe knocked on the config port.
    """

    async def test_port_env_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("OSPREY_ARIEL_PORT", "9391")
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        cfg = _cfg({"ariel": True}, ariel={"web": {"port": 8085}})
        await _run(cfg, handler=handler)

        assert "http://127.0.0.1:9391/health" in seen
        assert "http://127.0.0.1:8085/health" not in seen

    async def test_port_env_override_applies_to_flat_sections_too(self, monkeypatch):
        """`okf` reads its port straight off `facility_knowledge`, not a subkey."""
        monkeypatch.setenv("OSPREY_FACILITY_KNOWLEDGE_PORT", "9691")
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        await _run(_cfg({"okf": True}, facility_knowledge={"port": 8093}), handler=handler)

        assert "http://127.0.0.1:9691/health" in seen

    async def test_panel_id_namespace_is_read_from_the_registry(self):
        """Every built-in panel id resolves to a registry entry — no local table.

        The health category's own copy of the panel-id → registry-key map had
        already drifted from the CLI's copy (it carried an `artifacts` entry the
        CLI omitted). Derived from the registry, the two cannot disagree.
        """
        from osprey.profiles.web_panels import BUILTIN_PANELS
        from osprey.registry.web import PANEL_ID_TO_REGISTRY_KEY

        assert set(PANEL_ID_TO_REGISTRY_KEY) == BUILTIN_PANELS


class TestMisplacedConfigKeys:
    """A key at the wrong nesting depth is reported, never silently ignored."""

    async def test_wrong_depth_port_is_reported_not_silently_ignored(self):
        """`ariel.port` (correct: `ariel.web.port`) must not read back as 8085."""
        rows = await _run(_cfg({"ariel": True}, ariel={"port": 9999}))
        row = rows["web_panels.ariel"]
        assert row.status is Status.WARNING
        assert row.value == "misconfigured"
        assert "ariel.port" in row.details
        assert "ariel.web.port" in row.details

    async def test_wrong_depth_on_a_flat_section_is_reported(self):
        """`artifact_server.web.port` is inert — the section is read flat."""
        rows = await _run(_cfg({}, artifact_server={"web": {"port": 9999}}))
        row = rows["web_panels.artifacts"]
        assert row.value == "misconfigured"
        assert "artifact_server.web.port" in row.details

    async def test_a_misconfigured_panel_does_not_suppress_the_others(self):
        rows = await _run(_cfg({"ariel": True, "okf": True}, ariel={"auto_launch": False}))
        assert rows["web_panels.ariel"].value == "misconfigured"
        assert rows["web_panels.okf"].value == "up"


class TestRegistration:
    def test_category_is_registered_and_config_dependent(self):
        """Registered as a core category, and degrades when config is unavailable."""
        from osprey.health.core import CORE_CATEGORY_NAMES, get_core_category_factory
        from osprey.health.records import CONFIG_DEPENDENT

        assert "web_panels" in CORE_CATEGORY_NAMES
        assert get_core_category_factory("web_panels") is web_panels
        assert "web_panels" in CONFIG_DEPENDENT

    async def test_none_config_yields_no_rows(self):
        assert await _run(None) == {}
