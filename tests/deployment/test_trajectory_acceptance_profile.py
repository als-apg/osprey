"""The trajectory acceptance stack renders as its profile says, and only inside its port block.

``tests/e2e/trajectory_acceptance/shape.py`` is the one render of the acceptance
stack: CI builds its images from it, the walk deploys it, the deployed e2e lane
runs it. These tests render it once, with no container started, and check the
render against what the acceptance contract stands on -- the stand-in target and
its sequence module, writes on the stand-in, the archive recording the stand-in,
the image registry, fixed timeouts that admit the walk's largest acquisition,
and no host port outside the profile's block.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment import host_ports
from osprey.deployment.compose_generator import resolve_image_axes
from osprey.deployment.container_lifecycle import as_built_compose_files
from osprey.port_layout import block_range, default_port
from osprey.services.trajectory.acquisition import chain_declaration, n_shots_max
from osprey.services.trajectory.constants import VA_SEQUENCE_DIR, connector_call_timeout_s
from osprey.services.trajectory.startup import TrajectorySettings, _facts
from tests.e2e._orm_stack import VA_ARCHIVER_CI_KNOBS
from tests.e2e.trajectory_acceptance import shape

#: The repo's own shipped scenarios, where :data:`shape.ACTIVE_SET` must name files.
SCENARIOS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "facilities"
    / "example"
    / "scenarios"
)


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return shape.render(tmp_path_factory.mktemp("trajectory_acceptance"))


@pytest.fixture(scope="module")
def config(repo: Path) -> dict[str, Any]:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


def _compose(repo: Path, service_dir: str) -> dict[str, Any]:
    path = repo / "build" / "services" / service_dir / "docker-compose.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestTheCommittedProfile:
    def test_it_adopts_the_shared_archive_knobs(self) -> None:
        assert shape.profile()["va_archiver"] == VA_ARCHIVER_CI_KNOBS["va_archiver"]

    def test_it_points_at_no_preset(self) -> None:
        assert "extends" not in shape.profile()

    def test_it_pins_both_va_protocol_ports_inside_the_port_window(self) -> None:
        va = shape.profile()["virtual_accelerator"]
        assert va["port"] in shape.PORT_WINDOW
        assert va["pva_port"] in shape.PORT_WINDOW
        assert va["port"] != va["pva_port"]

    def test_its_port_window_is_the_first_hundred_ports_of_its_block(self) -> None:
        base = shape.profile()["port_base"]
        assert shape.PORT_WINDOW == range(base, base + 100)

    def test_the_active_set_names_shipped_scenarios(self) -> None:
        assert shape.ACTIVE_SET == ("txt-injection",)
        for name in shape.ACTIVE_SET:
            assert (SCENARIOS_DIR / f"{name}.yaml").is_file(), name

    def test_the_kick_stays_inside_the_corrector_bound(self) -> None:
        assert shape.VCM_SETPOINT.fullmatch(shape.KICK["record"])
        assert 0 < shape.KICK["value_a"] < shape.VCM_BOUND_A


class TestTheRenderedRepo:
    def test_the_materialised_profile_is_explicit(self, repo: Path) -> None:
        materialised = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
        assert "extends" not in materialised

    def test_one_roster_user_on_one_persona_that_writes_to_the_stand_in(self, repo: Path) -> None:
        materialised = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
        terminals = materialised["config"]["modules.web_terminals"]
        assert terminals["users"] == [{"name": shape.USER, "index": 0, "persona": shape.PERSONA}]
        assert sorted(p.name for p in (repo / "personas").glob("*.yml")) == [f"{shape.PERSONA}.yml"]
        delta = yaml.safe_load((repo / "personas" / f"{shape.PERSONA}.yml").read_text())
        assert delta["web_panels"] == ["trajectory"]
        assert delta["config"]["control_system.connector.live_standin.writes_enabled"] is True

    def test_every_vertical_corrector_is_bounded(self, repo: Path) -> None:
        limits = yaml.safe_load((repo / "data" / "facility" / "limits.yaml").read_text())
        kick = [r for r in limits["records"] if r["address"] == shape.KICK["record"]]
        assert kick == [
            {
                "address": shape.KICK["record"],
                "min_value": -shape.VCM_BOUND_A,
                "max_value": shape.VCM_BOUND_A,
            }
        ]

    def test_the_env_file_is_private(self, repo: Path) -> None:
        env = repo / ".env"
        assert env.stat().st_mode & 0o777 == 0o600
        assert shape.PLACEHOLDER_MODEL_KEY_LINE in env.read_text(encoding="utf-8")


class TestTheRenderedConfig:
    def test_the_trajectory_service_commands_the_stand_in(self, config: dict[str, Any]) -> None:
        trajectory = config["services"]["trajectory"]
        assert trajectory["target"] == "standin"
        assert trajectory["sequence_dir"] == VA_SEQUENCE_DIR

    def test_the_sequence_module_is_where_the_service_looks(
        self, repo: Path, config: dict[str, Any]
    ) -> None:
        sequence_dir = repo / config["services"]["trajectory"]["sequence_dir"]
        assert (sequence_dir / "sequence.py").is_file()

    def test_the_stand_in_is_served_and_writable(self, config: dict[str, Any]) -> None:
        base = config["deployment"]["port_base"]
        assert config["services"]["live_standin"]["port"] == default_port("va_standin", base=base)
        assert config["control_system"]["connector"]["live_standin"]["writes_enabled"] is True

    def test_the_simulation_state_dir_is_explicit(self, config: dict[str, Any]) -> None:
        assert config["simulation"]["state_dir"] == "./var/agent_data/simulation"

    def test_the_archive_records_the_stand_in(self, repo: Path, config: dict[str, Any]) -> None:
        assert config["archiver"]["type"] == "mongodb_archiver"
        recorder = _compose(repo, "archiver_recorder")["services"]["archiver-recorder"]
        standin_port = config["services"]["live_standin"]["port"]
        assert recorder["environment"]["EPICS_CA_NAME_SERVERS"] == f"live-standin:{standin_port}"

    def test_the_images_come_from_the_acceptance_registry_at_the_checkout_tag(
        self, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OSPREY_IMAGE_REGISTRY", raising=False)
        monkeypatch.setenv(shape.IMAGE_TAG_ENV, "0123abc")
        assert config["images"]["registry"] == shape.IMAGE_REGISTRY
        assert resolve_image_axes(config) == (f"{shape.IMAGE_REGISTRY}/", "0123abc")

    def test_the_timeouts_are_the_profiles(self, config: dict[str, Any]) -> None:
        trajectory = config["services"]["trajectory"]
        va = config["control_system"]["connector"]["virtual_accelerator"]
        assert trajectory["armed_timeout_s"] == 10.0
        assert trajectory["action_timeout_s"] == 570.0
        assert va["timeout_s"] == 5.0
        assert va["step_read_timeout_s"] == 2.0

    def test_the_action_timeout_admits_the_largest_acquisition(
        self, repo: Path, config: dict[str, Any]
    ) -> None:
        trajectory = config["services"]["trajectory"]
        sequence = repo / trajectory["sequence_dir"] / "sequence.py"
        spec = importlib.util.spec_from_file_location("acceptance_sequence", sequence)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        max_pause_s = module.SEQUENCE_METADATA["max_pause_s"]
        call_timeout_s = connector_call_timeout_s(config, trajectory["target"])
        slots = _facts(config)["trajectory_slots"]
        (model,) = slots
        declaration = chain_declaration(slots[model])
        settings = TrajectorySettings(
            armed_timeout_s=trajectory["armed_timeout_s"],
            action_timeout_s=trajectory["action_timeout_s"],
        )
        admitted = n_shots_max(
            declaration, settings, max_pause_s=max_pause_s, call_timeout_s=call_timeout_s
        )
        assert admitted >= shape.N_SHOTS_MAX


def test_no_host_port_lies_outside_the_block(repo: Path, config: dict[str, Any]) -> None:
    base = config["deployment"]["port_base"]
    assert base == shape.PORT_WINDOW.start
    first, last = block_range(base)
    compose_files = [repo / path for path in as_built_compose_files(config, repo)]
    bindings = [
        *host_ports.parse_host_port_bindings(compose_files),
        *host_ports.derive_host_network_bindings(config),
        *host_ports.derive_web_terminal_bindings(config),
        *host_ports.derive_reader_origin_bindings(config),
    ]
    published = {b.service for b in bindings}
    assert {"virtual-accelerator", "live-standin", "trajectory"} <= published
    outside = sorted((b.service, b.host_port) for b in bindings if not first <= b.host_port <= last)
    assert outside == []
    va = config["services"]["virtual_accelerator"]
    assert {va["port"], va["pva_port"]} <= set(shape.PORT_WINDOW)
