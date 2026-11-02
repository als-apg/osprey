"""Everything the browser is told about a sidecar panel comes from its registry entry.

``osprey.profiles.web_panels.SIDECAR_PANELS`` describes each panel the terminal
starts itself: its label, its sidebar icon and tooltip sentence, the path its
health is polled at. Three surfaces carry that description outward, and none
of them restates it:

- :data:`BUILTIN_PANEL_LABELS`, which ``/api/panels`` and the health category
  word their rows with;
- the descriptor list the page stamps on ``<html data-sidecar-panels>``, which
  ``panel-catalog.js`` builds its sidecar rows from;
- ``GET /api/panels/<id>/server``, one config route for every sidecar.

So a sidecar added to the registry needs no route, no catalog entry and no
icon-map row of its own.
"""

from __future__ import annotations

import json
import re

import pytest

from osprey.profiles import web_panels
from osprey.profiles.web_panels import (
    BUILTIN_PANEL_LABELS,
    JUPYTER_PANEL_ID,
    SIDECAR_PANELS,
    SidecarPanel,
    sidecar_panel_descriptors,
)
from osprey.registry.web import panel_url_state_attr

PROBE_ID = "probe"


@pytest.fixture
def probe_sidecar(monkeypatch):
    """A second sidecar in the registry, declaring nothing beyond what it must."""
    monkeypatch.setitem(
        web_panels.SIDECAR_PANELS, PROBE_ID, SidecarPanel("a probe", "module:Class")
    )


class TestRegistryEntry:
    def test_the_two_positional_fields_still_build_an_entry(self):
        panel = SidecarPanel("One line", "module:Class")

        assert (panel.name, panel.factory_path) == ("One line", "module:Class")
        assert (panel.label, panel.icon, panel.description) == ("", "", "")
        assert panel.health_endpoint is None
        assert panel.operator_secret is True
        assert panel.mcp_path is None
        assert panel.agent_hop is False
        assert (panel.mcp_allow, panel.mcp_ask, panel.mcp_writes) == ((), (), ())
        assert (panel.entry_pages, panel.image_checks) == ((), ())

    def test_an_mcp_path_alone_does_not_opt_the_panel_into_the_agent_hop(self):
        panel = SidecarPanel("One line", "module:Class", mcp_path="/mcp")

        assert panel.mcp_path == "/mcp"
        assert panel.agent_hop is False

    def test_the_agent_fields_are_declared_by_keyword(self):
        panel = SidecarPanel(
            "One line",
            "module:Class",
            mcp_path="/mcp",
            agent_hop=True,
            mcp_allow=("read_one",),
            mcp_ask=("write_one", "write_two"),
            mcp_writes=("write_one",),
        )

        assert panel.agent_hop is True
        assert panel.mcp_allow == ("read_one",)
        assert panel.mcp_ask == ("write_one", "write_two")
        assert panel.mcp_writes == ("write_one",)

    def test_an_entry_is_hashable_with_its_tuples(self):
        """Tuples, not lists: the entry is frozen and one registry object is
        shared by every reader."""
        panel = SidecarPanel("One line", "module:Class", entry_pages=("/lab",))

        assert hash(panel) == hash(SidecarPanel("One line", "module:Class", entry_pages=("/lab",)))

    def test_the_notebook_entry_describes_its_own_row(self):
        panel = SIDECAR_PANELS[JUPYTER_PANEL_ID]

        assert panel.label == "JUPYTER"
        assert panel.icon == "notebook"
        assert panel.description
        assert panel.health_endpoint == "/api/status"
        assert panel.operator_secret is True
        assert panel.mcp_path is None

    @pytest.mark.parametrize("panel_id", sorted(SIDECAR_PANELS))
    def test_the_label_table_takes_a_sidecar_label_from_its_entry(self, panel_id):
        assert BUILTIN_PANEL_LABELS[panel_id] == SIDECAR_PANELS[panel_id].label


class TestDescriptors:
    def test_the_registered_descriptors(self):
        assert sidecar_panel_descriptors() == [
            {
                "id": "jupyter",
                "label": "JUPYTER",
                "icon": "notebook",
                "description": SIDECAR_PANELS["jupyter"].description,
                # The route that predates the generic one, which still answers.
                "configEndpoint": "/api/jupyter-server",
                "healthEndpoint": "/api/status",
                "startEndpoint": "/api/panels/jupyter/start",
            },
        ]

    @pytest.mark.usefixtures("probe_sidecar")
    def test_a_new_sidecar_is_described_without_a_route_of_its_own(self):
        described = {entry["id"]: entry for entry in sidecar_panel_descriptors()}

        assert described[PROBE_ID] == {
            "id": PROBE_ID,
            "label": "PROBE",
            "icon": None,
            "description": None,
            "configEndpoint": f"/api/panels/{PROBE_ID}/server",
            "healthEndpoint": None,
            "startEndpoint": f"/api/panels/{PROBE_ID}/start",
        }

    def test_the_declared_fields_reach_the_descriptor(self, monkeypatch):
        monkeypatch.setitem(
            web_panels.SIDECAR_PANELS,
            PROBE_ID,
            SidecarPanel(
                "a probe",
                "module:Class",
                label="SCREENS",
                icon="hexagon",
                description="Operator screens",
                health_endpoint="/health",
            ),
        )
        described = {entry["id"]: entry for entry in sidecar_panel_descriptors()}

        assert described[PROBE_ID]["label"] == "SCREENS"
        assert described[PROBE_ID]["icon"] == "hexagon"
        assert described[PROBE_ID]["description"] == "Operator screens"
        assert described[PROBE_ID]["healthEndpoint"] == "/health"

    def test_no_credential_posture_reaches_the_browser(self, monkeypatch):
        """``operator_secret``, ``mcp_path`` and ``entry_pages`` are the proxy's
        to act on, the agent's tool lists are the build's, and so are the
        image's checks."""
        monkeypatch.setitem(
            web_panels.SIDECAR_PANELS,
            PROBE_ID,
            SidecarPanel(
                "a probe",
                "module:Class",
                operator_secret=False,
                mcp_path="/mcp",
                agent_hop=True,
                mcp_allow=("read_one",),
                mcp_ask=("write_one",),
                entry_pages=("/lab",),
                image_checks=("python -I -m probe --check",),
            ),
        )
        described = {entry["id"]: entry for entry in sidecar_panel_descriptors()}

        assert set(described[PROBE_ID]) == {
            "id",
            "label",
            "icon",
            "description",
            "configEndpoint",
            "healthEndpoint",
            "startEndpoint",
        }
        stamped = json.dumps(described[PROBE_ID])
        for declared in ("/mcp", "read_one", "write_one", "/lab", "probe --check", "operator"):
            assert declared not in stamped


def _stamp(body: str) -> list[dict]:
    """The descriptors on the opening ``<html>`` tag of a rendered page."""
    start = body.index("<html ")
    html_tag = body[start : body.index(">", start)]
    match = re.search(r"data-sidecar-panels='([^']*)'", html_tag)
    assert match, "the sidecar panels are not stamped on <html>"
    return json.loads(match.group(1))


class TestPageStamp:
    @pytest.mark.parametrize("page", ["/", "/static/session.html"])
    def test_the_page_carries_the_registry_descriptors(self, isolated_app, page):
        with isolated_app() as client:
            resp = client.get(page)

        assert resp.status_code == 200
        assert _stamp(resp.text) == sidecar_panel_descriptors()

    @pytest.mark.usefixtures("probe_sidecar")
    def test_the_stamp_follows_the_registry(self, isolated_app):
        with isolated_app() as client:
            stamped = _stamp(client.get("/").text)

        assert [entry["id"] for entry in stamped] == ["jupyter", PROBE_ID]


class TestGenericServerRoute:
    def test_it_answers_what_the_notebook_route_answers(self, isolated_app):
        with isolated_app() as client:
            setattr(client.app.state, panel_url_state_attr("jupyter"), "http://127.0.0.1:9500")

            generic = client.get("/api/panels/jupyter/server")
            legacy = client.get("/api/jupyter-server")

        assert generic.status_code == 200
        assert generic.json() == legacy.json()
        assert generic.json()["url"] == "/panel/jupyter"
        assert generic.json()["available"] is True
        assert set(generic.json()) == {"url", "available", "state", "message"}

    @pytest.mark.usefixtures("probe_sidecar")
    def test_a_sidecar_that_is_not_running_answers_unavailable(self, isolated_app):
        with isolated_app() as client:
            resp = client.get(f"/api/panels/{PROBE_ID}/server")

        assert resp.status_code == 200
        assert resp.json()["url"] is None
        assert resp.json()["available"] is False

    @pytest.mark.usefixtures("probe_sidecar")
    def test_a_running_sidecar_answers_its_proxy_url(self, isolated_app):
        with isolated_app() as client:
            setattr(client.app.state, panel_url_state_attr(PROBE_ID), "http://127.0.0.1:9501")
            resp = client.get(f"/api/panels/{PROBE_ID}/server")

        assert resp.json()["url"] == f"/panel/{PROBE_ID}"
        assert resp.json()["available"] is True

    @pytest.mark.parametrize("panel_id", ["ariel", "artifacts", "nonesuch"])
    def test_an_id_that_names_no_sidecar_is_not_found(self, isolated_app, panel_id):
        with isolated_app() as client:
            assert client.get(f"/api/panels/{panel_id}/server").status_code == 404


class TestPanelsRouteLabels:
    def test_an_enabled_sidecar_is_labelled_from_its_entry(self, isolated_app):
        with isolated_app() as client:
            client.app.state.enabled_panels = {"artifacts", "jupyter"}
            labels = client.get("/api/panels").json()["labels"]

        assert labels["jupyter"] == SIDECAR_PANELS["jupyter"].label
