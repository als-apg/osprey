"""The virtual accelerator's entrypoint: what it reads, what it refuses, what it hands on.

The entrypoint serves the simulator view under ``$VA_DATA_DIR/simulator/``
through one composite and one model runner. The runner reaches the Channel
Access server extension at import, so every case here replaces its module
with a recorder: what is asserted is the arguments the entrypoint hands the
composite and the runner, never a served wire (``tests/va/test_apply_fault.py``
boots the entrypoint for real in the live venue).
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from osprey.services.virtual_accelerator import entrypoint
from osprey_connectors.simulation import DEFAULT_TICK_S

#: The documents of a minimal simulator view, by file name.
VIEW: dict[str, dict[str, Any]] = {
    "served_models.json": {"models": ["SR", "texture"]},
    "addresses.json": {"channels": ["SR:A", "SR:B", "SR:C"], "status": ["ca:SIM:SR:STATUS"]},
    "variables.json": {"code": "ca", "models": [], "channels": []},
}

RUNNER_MODULE = "osprey.services.virtual_accelerator.serving.runner"


class _Recorded:
    """What the stubbed composite and runner were constructed with."""

    composite: dict[str, Any]
    runner: dict[str, Any]
    ran: bool = False
    first_pass_error: str | None = None

    def __init__(self) -> None:
        self.events: list[str] = []


def _write_view(data_dir: Path) -> Path:
    view = data_dir / entrypoint.SIMULATOR_DIR
    view.mkdir(parents=True)
    for name, document in VIEW.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    return view


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> _Recorded:
    """Stub the composite and the runner; return what they were built with."""
    record = _Recorded()

    class Composite:
        def __init__(self, view_dir: Path, **kwargs: Any) -> None:
            record.composite = {"view_dir": view_dir, **kwargs}

    class ModelRunner:
        def __init__(self, composite: Any, view: Any, addresses_json: Any, **kwargs: Any) -> None:
            record.runner = {
                "composite": composite,
                "view": view,
                "addresses_json": addresses_json,
                **kwargs,
            }

        def first_pass(self) -> str | None:
            record.events.append("first_pass")
            return record.first_pass_error

        def run(self) -> None:
            record.events.append("run")
            record.ran = True

    from osprey_connectors.simulation import composite

    monkeypatch.setattr(composite, "Composite", Composite)
    runner = types.ModuleType(RUNNER_MODULE)
    runner.ModelRunner = ModelRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, runner)
    # The entrypoint installs its own stop handlers once it serves; the
    # test process keeps its own.
    monkeypatch.setattr(
        entrypoint, "_install_shutdown_signals", lambda: record.events.append("signals")
    )
    ready_line = entrypoint._ready_line

    def _recorded_ready_line(channel_count: int) -> str:
        record.events.append("ready")
        return ready_line(channel_count)

    monkeypatch.setattr(entrypoint, "_ready_line", _recorded_ready_line)
    monkeypatch.setattr(entrypoint, "_configure_logging", lambda: None)
    return record


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data root holding a simulator view, named by the environment."""
    _write_view(tmp_path)
    monkeypatch.setenv("VA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VA_INSTANCE", "virtual_accelerator")
    for name in ("VA_STATE_DIR", "VA_POLL_INTERVAL_S", "VA_MODEL_WRITE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class TestTheView:
    def test_unset_data_dir_resolves_the_addresses_under_data_simulator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VA_DATA_DIR", raising=False)

        assert entrypoint.view_dir() / "addresses.json" == Path("/data/simulator/addresses.json")

    def test_the_data_dir_names_the_root_the_view_sits_under(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VA_DATA_DIR", str(tmp_path))

        assert entrypoint.view_dir() == tmp_path / "simulator"

    def test_the_composite_and_the_runner_are_built_over_the_view(
        self, served: Path, recorded: _Recorded
    ) -> None:
        entrypoint.main()

        assert recorded.composite["view_dir"] == served / "simulator"
        assert recorded.runner["view"] == VIEW["variables.json"]
        assert recorded.runner["addresses_json"] == VIEW["addresses.json"]
        assert recorded.ran

    @pytest.mark.usefixtures("recorded")
    def test_a_missing_view_is_refused_by_its_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VA_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("VA_INSTANCE", "virtual_accelerator")

        with pytest.raises(SystemExit, match=str(tmp_path / "simulator" / "served_models.json")):
            entrypoint.main()

    @pytest.mark.usefixtures("served", "recorded")
    def test_the_ready_line_counts_the_served_channels(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        entrypoint.main()

        assert f"{entrypoint.READY_MARKER}: 3 channels" in capsys.readouterr().out.splitlines()

    @pytest.mark.usefixtures("served")
    def test_the_ready_line_follows_the_first_pass(self, recorded: _Recorded) -> None:
        entrypoint.main()

        assert recorded.events == ["first_pass", "signals", "ready", "run"]

    @pytest.mark.usefixtures("served")
    def test_a_failed_first_pass_exits_without_the_ready_line(
        self, recorded: _Recorded, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recorded.first_pass_error = "the deck has no stable orbit"

        with pytest.raises(SystemExit) as caught:
            entrypoint.main()

        assert "the deck has no stable orbit" in str(caught.value.code)
        assert caught.value.code not in (0, None)
        assert entrypoint.READY_MARKER not in capsys.readouterr().out
        assert not recorded.ran
        assert recorded.events == ["first_pass"]

    @pytest.mark.usefixtures("served", "recorded")
    def test_the_served_models_are_named_at_boot(self, capsys: pytest.CaptureFixture[str]) -> None:
        entrypoint.main()

        assert "Serving models: SR, texture" in capsys.readouterr().out


class TestTheInstance:
    @pytest.mark.parametrize("instance", ["virtual_accelerator", "live_standin"])
    @pytest.mark.usefixtures("served")
    def test_the_instance_reaches_the_composite_and_the_runner(
        self, recorded: _Recorded, monkeypatch: pytest.MonkeyPatch, instance: str
    ) -> None:
        monkeypatch.setenv("VA_INSTANCE", instance)

        entrypoint.main()

        assert recorded.composite["instance"] == instance
        assert recorded.runner["instance"] == instance

    @pytest.mark.usefixtures("served")
    def test_the_model_logs_are_appended_under_the_mounted_simulator_dir(
        self, recorded: _Recorded
    ) -> None:
        entrypoint.main()

        assert recorded.composite["log_dir"] == Path("/var/simulator")

    @pytest.mark.usefixtures("served")
    def test_the_runner_writes_its_health_to_the_healthcheck_file(
        self, recorded: _Recorded
    ) -> None:
        entrypoint.main()

        assert recorded.runner["health_file"] == entrypoint.HEALTH_FILE
        assert entrypoint.HEALTH_FILE == Path("/run/osprey-va/health.json")

    @pytest.mark.usefixtures("served", "recorded")
    def test_a_missing_instance_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VA_INSTANCE")

        with pytest.raises(SystemExit, match="VA_INSTANCE"):
            entrypoint.main()

    @pytest.mark.parametrize("value", ["", "inprocess", "virtual-accelerator", "sandbox"])
    @pytest.mark.usefixtures("served", "recorded")
    def test_an_unknown_instance_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("VA_INSTANCE", value)

        with pytest.raises(SystemExit, match="VA_INSTANCE"):
            entrypoint.main()

    def test_the_instances_are_the_composites_served_ones(self) -> None:
        from osprey_connectors.simulation.composite import INSTANCES

        assert set(entrypoint.VA_INSTANCES) == set(INSTANCES) - {"inprocess"}


class TestTheTick:
    @pytest.mark.usefixtures("served")
    def test_a_stated_poll_interval_is_the_runners_tick(
        self, recorded: _Recorded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VA_POLL_INTERVAL_S", "0.5")

        entrypoint.main()

        assert recorded.runner["tick_interval_s"] == 0.5

    @pytest.mark.parametrize("value", [None, ""])
    @pytest.mark.usefixtures("served")
    def test_an_unset_poll_interval_ticks_at_the_default(
        self, recorded: _Recorded, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        if value is not None:
            monkeypatch.setenv("VA_POLL_INTERVAL_S", value)

        entrypoint.main()

        assert recorded.runner["tick_interval_s"] == DEFAULT_TICK_S

    @pytest.mark.parametrize("value", ["0", "-1", "soon"])
    @pytest.mark.usefixtures("served", "recorded")
    def test_a_poll_interval_that_is_not_a_positive_number_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("VA_POLL_INTERVAL_S", value)

        with pytest.raises(SystemExit, match="VA_POLL_INTERVAL_S"):
            entrypoint.main()


class TestTheStateAndTheToken:
    @pytest.mark.usefixtures("served")
    def test_an_unset_state_dir_serves_nominal_alone(self, recorded: _Recorded) -> None:
        entrypoint.main()

        assert recorded.composite["state_dir"] is None

    @pytest.mark.usefixtures("served")
    def test_a_stated_state_dir_is_the_composites(
        self, recorded: _Recorded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VA_STATE_DIR", "/state/simulation")

        entrypoint.main()

        assert recorded.composite["state_dir"] == Path("/state/simulation")

    @pytest.mark.parametrize("value", [None, "", "   "])
    @pytest.mark.usefixtures("served")
    def test_no_token_refuses_every_model_write(
        self, recorded: _Recorded, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        if value is not None:
            monkeypatch.setenv("VA_MODEL_WRITE_TOKEN", value)

        entrypoint.main()

        assert recorded.runner["model_write_token"] is None

    @pytest.mark.usefixtures("served")
    def test_a_token_is_handed_on_and_never_printed(
        self,
        recorded: _Recorded,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("VA_MODEL_WRITE_TOKEN", " s3cret ")

        entrypoint.main()

        assert recorded.runner["model_write_token"] == " s3cret "
        assert "s3cret" not in capsys.readouterr().out


def test_importing_the_entrypoint_loads_neither_the_server_nor_the_composite() -> None:
    probe = (
        "import sys\n"
        "import osprey.services.virtual_accelerator.entrypoint\n"
        "heavy = {'pcaspy', 'p4p', 'lume_pva_apg', 'osprey_connectors.simulation.composite'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert loaded == "[]"
