"""Sidecar panels are built in, and are in none of the companion-server tables.

A sidecar is a panel the web terminal starts itself: no config section with a
host and a port, no port band, no ``auto_launch`` key, no ``/health`` address.
It is reached only through the terminal's own panel proxy. Every one of those is
a thing ``registry.web`` hands a companion server, so registering a sidecar
there would give it four facts that mean nothing — while leaving it out of
``BUILTIN_PANELS`` would make ``web.panels.jupyter`` an unknown panel id the
build refuses.

These tests hold both halves at once: the sidecar id is a first-class built-in
everywhere panel ids are read, and absent everywhere registry keys are. The
health category's ``skip`` row is pinned in
``tests/health/core/test_web_panels.py``, and the build interview's menu entry
in ``tests/cli/test_build_profile_panels.py``, beside the fixtures each needs.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from osprey.deployment.web_terminals.ports import FAMILY_BASE_FIELDS
from osprey.profiles.web_panels import (
    BUILTIN_PANEL_LABELS,
    BUILTIN_PANELS,
    SIDECAR_PANELS,
    UNIVERSAL_PANELS,
)
from osprey.registry.web import PANEL_ID_TO_REGISTRY_KEY, panel_url_state_attr

SIDECAR_IDS = sorted(SIDECAR_PANELS)


def test_the_notebooks_panel_is_the_registered_sidecar():
    """The one entry, pinned so the id and its label cannot drift apart."""
    assert SIDECAR_IDS == ["jupyter"]
    assert BUILTIN_PANEL_LABELS["jupyter"] == "JUPYTER"


@pytest.mark.parametrize("panel_id", SIDECAR_IDS)
class TestSidecarIsABuiltinPanel:
    """Everything keyed on a panel id covers a sidecar without a second lookup."""

    def test_it_is_a_builtin_panel(self, panel_id):
        assert panel_id in BUILTIN_PANELS
        assert panel_id not in UNIVERSAL_PANELS

    def test_it_has_a_display_label(self, panel_id):
        """An unlabelled builtin renders its raw id in the rail."""
        assert BUILTIN_PANEL_LABELS[panel_id]

    def test_the_proxy_resolves_it(self, panel_id):
        """Without a state-attr entry the proxy treats it as a custom panel and
        looks for a config-declared url that does not exist — a dead tab."""
        from osprey.interfaces.web_terminal.routes.proxy import _PANEL_STATE_MAP

        assert _PANEL_STATE_MAP[panel_id] == panel_url_state_attr(panel_id)

    def test_the_factory_is_a_dotted_path_not_an_import(self, panel_id):
        """``module:attribute``, resolved by the launcher, so reading the panel
        registry never pulls the sidecar's dependencies into the process."""
        module, sep, attribute = SIDECAR_PANELS[panel_id].factory_path.partition(":")
        assert sep, "expected 'module:attribute'"
        assert module and attribute


@pytest.mark.parametrize("panel_id", SIDECAR_IDS)
class TestSidecarIsNotACompanionServer:
    """The registry-key tables stay scoped to what ``ServerLauncher`` runs."""

    def test_it_has_no_registry_key(self, panel_id):
        assert panel_id not in PANEL_ID_TO_REGISTRY_KEY

    def test_it_gets_no_port_family(self, panel_id):
        """A deployment allocates a port band per family. A sidecar binds an
        ephemeral loopback port the terminal picks, so a band would reserve
        ports nothing ever listens on."""
        assert f"{panel_id}_base_port" not in FAMILY_BASE_FIELDS
        assert panel_id not in FAMILY_BASE_FIELDS.values()


def test_importing_the_package_does_not_build_the_web_application():
    """A sidecar process imports a sibling module, not FastAPI and uvicorn.

    Checked in a subprocess: an ``app`` already imported by an earlier test in
    this session would make an in-process ``sys.modules`` assertion pass no
    matter what the package ``__init__`` does.
    """
    probe = (
        "import sys; import osprey.interfaces.web_terminal as pkg; "
        "print('osprey.interfaces.web_terminal.app' in sys.modules); "
        "print(pkg.run_web.__module__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    imported, run_web_module = result.stdout.split()

    assert imported == "False", "importing the package built the web application"
    # …and the lazy export still resolves to the real function.
    assert run_web_module == "osprey.interfaces.web_terminal.app"
