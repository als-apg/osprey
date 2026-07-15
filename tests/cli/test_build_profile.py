"""Tests for the ``scan_panels:`` block of the build-profile schema.

Covers the :class:`ScanPanelsConfig` dataclass, the ``BuildProfile.validate``
exemption that lets the three non-builtin scan-panel web_panels ids
(``scan-plan``, ``scan-results``, ``scan-health``) validate without a
pre-existing ``web.panels.<id>.url`` when a ``scan_panels`` block is present
(their urls are derived post-build by ``_inject_scan_panels``), and a
regression guard that the shipped control-assistant preset/profile still
validates — the gate task 3.3 (tutorial-config) re-runs after adding the
scan panels to that preset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile as bp
from osprey.cli.build_cmd import build
from osprey.cli.build_profile import (
    BuildProfile,
    ScanPanelsConfig,
    _parse_profile,
)
from osprey.errors import BuildProfileError


def test_no_scan_panels_validates(tmp_path: Path) -> None:
    """A profile with no scan_panels block validates without raising."""
    BuildProfile(name="x").validate(tmp_path)


def test_scan_panels_ids_validate_without_url_when_scan_panels_present(
    tmp_path: Path,
) -> None:
    """The three scan-panel ids need no manual web.panels.<id>.url override
    when a scan_panels block is present — the urls are derived post-build by
    _inject_scan_panels, which runs after this validator."""
    profile = BuildProfile(
        name="x",
        web_panels=["scan-plan", "scan-results", "scan-health"],
        scan_panels=ScanPanelsConfig(),
    )
    profile.validate(tmp_path)  # must not raise


def test_scan_panels_ids_without_scan_panels_still_require_url(tmp_path: Path) -> None:
    """The escape hatch is narrow: the scan-panel ids with no scan_panels
    block and no url override are still rejected (nothing would derive their
    URL)."""
    profile = BuildProfile(name="x", web_panels=["scan-plan"])
    with pytest.raises(BuildProfileError, match="scan-plan"):
        profile.validate(tmp_path)


def test_unbacked_custom_panel_still_requires_url_even_with_scan_panels(
    tmp_path: Path,
) -> None:
    """The scan_panels escape hatch applies only to the three known ids — any
    other url-less custom panel is still rejected even when a scan_panels
    block is present."""
    profile = BuildProfile(
        name="x",
        web_panels=["grafana"],
        scan_panels=ScanPanelsConfig(),
    )
    with pytest.raises(BuildProfileError, match="grafana"):
        profile.validate(tmp_path)


def test_scan_panels_port_overflow_raises(tmp_path: Path) -> None:
    """An out-of-range scan_panels.port fails validation."""
    profile = BuildProfile(name="x", scan_panels=ScanPanelsConfig(port=70000))
    with pytest.raises(BuildProfileError, match="scan_panels.port"):
        profile.validate(tmp_path)


def test_scan_panels_default_port() -> None:
    """ScanPanelsConfig defaults to port 8095, matching the compose template
    and the sidecar's default uvicorn bind."""
    assert ScanPanelsConfig().port == 8095


def test_scan_panels_not_a_mapping_raises() -> None:
    """A non-mapping 'scan_panels' block raises during parsing."""
    with pytest.raises(BuildProfileError, match="scan_panels"):
        _parse_profile({"name": "x", "scan_panels": "not-a-mapping"})


def test_scan_panels_is_known_key() -> None:
    """'scan_panels' is a recognized top-level profile key (no unknown-key warning)."""
    assert "scan_panels" in bp._KNOWN_PROFILE_KEYS


def test_scan_panels_parse_round_trip() -> None:
    """A scan_panels block parses its port field through _parse_profile."""
    raw = {"name": "x", "scan_panels": {"port": 9100}}
    profile = _parse_profile(raw)
    assert profile.scan_panels is not None
    assert profile.scan_panels.port == 9100


def test_scan_panels_parse_defaults_when_empty_mapping() -> None:
    """An empty scan_panels mapping (`scan_panels: {}`) parses to defaults."""
    raw = {"name": "x", "scan_panels": {}}
    profile = _parse_profile(raw)
    assert profile.scan_panels is not None
    assert profile.scan_panels.port == 8095


def test_control_assistant_profile_validates() -> None:
    """The shipped control-assistant preset/profile validates cleanly.

    This is a regression guard task 3.3 (tutorial-config) re-runs after
    adding the scan_panels block + the three scan-panel web_panels ids to
    this preset — it must keep validating once that wiring lands.
    """
    presets_dir = bp._presets_dir()
    raw = yaml.safe_load((presets_dir / "control-assistant.yml").read_text(encoding="utf-8"))
    profile = _parse_profile(raw)

    profile.validate(presets_dir)  # raises BuildProfileError on any issue


# ── Task 3.3: turn-key VA-backed scan stack render ───────────────────────────
#
# The control-assistant preset now bakes the bluesky/virtual_accelerator/
# scan_panels injector blocks in directly (no --set/--override flags needed),
# so `osprey build` on the bare preset renders the full scan stack + the
# three scan panels turn-key. These tests build the preset in-process
# (CliRunner, --skip-deps --skip-lifecycle -- Docker-free, mirroring
# tests/cli/test_va_default_config.py's scaffolded_project fixture and
# tests/e2e/_orm_stack.py's build_via_cli_runner) and assert on the rendered
# project's config.yml.


@pytest.fixture(scope="module")
def turnkey_scan_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the bare control-assistant preset (no overrides) into a tmp dir.

    Module-scoped: the build is the slow part (template render + service
    template copies) and every test in this section only reads the resulting
    config.yml, so one build is shared across assertions.
    """
    tmp_path = tmp_path_factory.mktemp("turnkey-scan")
    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "turnkey-scan",
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "turnkey-scan"
    assert (project_dir / "config.yml").exists()
    return project_dir


@pytest.fixture(scope="module")
def turnkey_scan_config(turnkey_scan_project: Path) -> dict:
    return yaml.safe_load((turnkey_scan_project / "config.yml").read_text(encoding="utf-8"))


class TestControlAssistantTurnkeyScanServices:
    """The three injected services render with the preset's baked-in ports."""

    def test_control_assistant_bluesky_service_rendered(self, turnkey_scan_config: dict) -> None:
        bluesky = turnkey_scan_config["services"]["bluesky"]
        assert bluesky["port"] == 8090
        assert bluesky["tiled_enabled"] is True
        assert bluesky["tiled_port"] == 8091

    def test_control_assistant_virtual_accelerator_service_rendered(
        self, turnkey_scan_config: dict
    ) -> None:
        va = turnkey_scan_config["services"]["virtual_accelerator"]
        assert va["port"] == 5064

    def test_control_assistant_scan_panels_service_rendered(
        self, turnkey_scan_config: dict
    ) -> None:
        scan_panels = turnkey_scan_config["services"]["scan_panels"]
        assert scan_panels["port"] == 8095

    def test_control_assistant_deployed_services_includes_all_three(
        self, turnkey_scan_config: dict
    ) -> None:
        deployed = turnkey_scan_config["deployed_services"]
        assert "bluesky" in deployed
        assert "virtual_accelerator" in deployed
        assert "scan_panels" in deployed


class TestControlAssistantTurnkeyScanPanels:
    """The three scan-panel web.panels entries are registered with a url."""

    def test_control_assistant_scan_plan_panel(self, turnkey_scan_config: dict) -> None:
        panel = turnkey_scan_config["web"]["panels"]["scan-plan"]
        assert panel["path"] == "/plan/"
        assert panel["url"]

    def test_control_assistant_scan_results_panel(self, turnkey_scan_config: dict) -> None:
        panel = turnkey_scan_config["web"]["panels"]["scan-results"]
        assert panel["path"] == "/results/"
        assert panel["url"]

    def test_control_assistant_scan_health_panel(self, turnkey_scan_config: dict) -> None:
        panel = turnkey_scan_config["web"]["panels"]["scan-health"]
        assert panel["path"] == "/health-panel/"
        assert panel["health_endpoint"] == "/health"
        assert panel["url"]


class TestControlAssistantTurnkeyScanControlSystem:
    """The preset's config overrides land: mock-by-default + container execution.

    The VA soft-IOC ships and is deployed as part of the turn-key scan stack,
    but control_system.type stays "mock" so a fresh tutorial project remains
    safe out of the box -- flipping the one config line to
    "virtual_accelerator" is what engages the deployed VA end-to-end (covered
    by tests/cli/test_va_default_config.py).
    """

    def test_control_assistant_control_system_type_is_mock(
        self, turnkey_scan_config: dict
    ) -> None:
        assert turnkey_scan_config["control_system"]["type"] == "mock"

    def test_control_assistant_execution_method_is_container(
        self, turnkey_scan_config: dict
    ) -> None:
        assert turnkey_scan_config["execution"]["execution_method"] == "container"

    def test_control_assistant_scan_mcp_server_enabled(
        self, turnkey_scan_config: dict
    ) -> None:
        assert turnkey_scan_config["claude_code"]["servers"]["scan"]["enabled"] is True


def test_control_assistant_turnkey_scan_preset_validates(turnkey_scan_project: Path) -> None:
    """BuildProfile.validate() passes for the preset as shipped (bare, no
    overrides) -- the three non-builtin scan-panel ids are accepted because
    the preset's own scan_panels block is present."""
    presets_dir = bp._presets_dir()
    raw = yaml.safe_load((presets_dir / "control-assistant.yml").read_text(encoding="utf-8"))
    profile = _parse_profile(raw)
    profile.validate(presets_dir)  # raises BuildProfileError on any issue
