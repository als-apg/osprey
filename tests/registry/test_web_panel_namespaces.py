"""The panel-id ↔ registry-key relation exists once, and every consumer reads it.

Registry keys (``artifact``, ``channel_finder``, ``lattice_dashboard``,
``system_health``) and panel ids (``artifacts``, ``channel-finder``, ``lattice``,
``system-health``) are two namespaces. They used to be bridged by four
hand-maintained tables — in the ``osprey web`` port pre-flight, the core
``web_panels`` health category, the web terminal's panel proxy, and
``profiles.web_panels`` — and two of them had already drifted: the health
category carried an ``artifacts`` entry the CLI's copy omitted entirely.

These tests hold the relation to a single declaration
(``WebServerDefinition.panel_id``) and pin the two conventions derived from it,
so a new companion server cannot be registered half-wired.
"""

from __future__ import annotations

import inspect

from osprey.profiles.web_panels import (
    BUILTIN_PANEL_LABELS,
    BUILTIN_PANELS,
    DEFAULT_PANEL_FALLBACK,
    UNIVERSAL_PANELS,
)
from osprey.registry.web import (
    FRAMEWORK_WEB_SERVERS,
    PANEL_ID_TO_REGISTRY_KEY,
    panel_url_state_attr,
)


class TestOneDeclarationPerPanel:
    def test_panel_ids_are_unique_across_servers(self):
        """Two servers claiming one panel id would silently shadow each other."""
        panel_ids = [defn.panel_id for defn in FRAMEWORK_WEB_SERVERS.values()]
        assert len(set(panel_ids)) == len(panel_ids)
        assert len(PANEL_ID_TO_REGISTRY_KEY) == len(FRAMEWORK_WEB_SERVERS)

    def test_every_registry_key_is_reachable_from_its_panel_id(self):
        for key, defn in FRAMEWORK_WEB_SERVERS.items():
            assert PANEL_ID_TO_REGISTRY_KEY[defn.panel_id] == key

    def test_builtin_panels_is_exactly_the_registered_panel_ids(self):
        """``BUILTIN_PANELS`` is derived, so registering a server registers its panel."""
        assert BUILTIN_PANELS == set(PANEL_ID_TO_REGISTRY_KEY)

    def test_the_gallery_is_registry_key_artifact_and_panel_artifacts(self):
        """The plural/singular pair that drifted, pinned in both directions.

        ``artifact`` is the registry key — the section is ``artifact_server`` and
        the env override ``OSPREY_ARTIFACT_SERVER_PORT``. ``artifacts`` is the
        panel id — it is what ``web.panels`` keys on, what the frontend tab is
        called, what ``/panel/artifacts/`` proxies, and the default tab.
        """
        assert FRAMEWORK_WEB_SERVERS["artifact"].panel_id == "artifacts"
        assert PANEL_ID_TO_REGISTRY_KEY["artifacts"] == "artifact"
        assert "artifacts" in BUILTIN_PANELS
        assert "artifact" not in BUILTIN_PANELS
        assert DEFAULT_PANEL_FALLBACK == "artifacts"

    def test_panel_ids_use_the_frontend_spelling_not_the_registry_spelling(self):
        """Hyphens, never underscores — the two-spelling trap these tables were for."""
        assert all("_" not in panel_id for panel_id in BUILTIN_PANELS)

    def test_universal_panels_and_labels_are_stated_in_panel_ids(self):
        assert UNIVERSAL_PANELS <= BUILTIN_PANELS
        assert set(BUILTIN_PANEL_LABELS) == BUILTIN_PANELS


class TestDerivedConsumers:
    def test_no_consumer_keeps_its_own_translation_table(self):
        """A literal panel-id → registry-key dict anywhere is the defect returning."""
        from osprey.cli import web_cmd
        from osprey.health.core import web_panels as health_web_panels
        from osprey.interfaces.web_terminal.routes import proxy

        for module in (web_cmd, health_web_panels, proxy):
            source = inspect.getsource(module)
            assert '"channel-finder": "channel_finder"' not in source, module.__name__
            assert '"channel_finder": "channel-finder"' not in source, module.__name__

    def test_proxy_reads_the_attribute_the_launcher_publishes(self):
        """The proxy's state-attr map is ``panel_url_state_attr``, not a copy of it.

        The attribute names are a convention shared between the web terminal's
        launcher (which writes them) and the panel proxy (which reads them), and
        the two modules cannot import each other. A second f-string spelling the
        convention out again would agree only until one side was edited, and the
        symptom — a panel reporting unavailable — is indistinguishable from a
        server that is legitimately switched off.

        That the launcher really writes these attributes is asserted
        behaviourally in ``tests/interfaces/test_panel_launch_gating.py``.
        """
        from osprey.interfaces.web_terminal.routes.proxy import _PANEL_STATE_MAP

        assert set(_PANEL_STATE_MAP) == BUILTIN_PANELS
        for panel_id, attr in _PANEL_STATE_MAP.items():
            assert attr == panel_url_state_attr(PANEL_ID_TO_REGISTRY_KEY[panel_id])
