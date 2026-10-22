"""The sequence module loads once from its directory, or the service refuses to start.

The loader tests run against the fixture module under ``sequences/fixture`` and
against modules written into a temporary directory, one defect each. The
handle tests run the fixture's ``inject`` through a :class:`SequenceHandle`
on a recording connector, under the product's own run lock and
``command_write``. A module has one entry point and one metadata key: the
settings and command steps once declared here are the measurement file's.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests._guarded_run import reset_run_state, write_config

from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    ENV_CONTROL_IDENTITY,
    ActionContext,
    OspreyWriteRefused,
    lock,
)
from osprey.services.trajectory import startup
from osprey.services.trajectory.constants import MAX_PAUSE_CEILING_S
from osprey.services.trajectory.sequence import (
    SEQUENCE_MODULE_FILE,
    SEQUENCE_MODULE_PREFIX,
    SequenceHandle,
    SequenceModule,
    SequenceRefused,
    SequenceStopped,
    load_sequence,
)
from osprey.services.trajectory.startup import (
    StartupRefused,
    TrajectoryCommands,
    TrajectoryDevice,
    TrajectorySettings,
)
from osprey.utils.module_loading import module_name_for
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import (
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)

FIXTURE_DIR = Path(__file__).parent / "sequences" / "fixture"

ARM = "SR:BPM:ARM"
REQUEST = "SR:INJ:REQUEST"
COMMANDS = TrajectoryCommands(
    model="SR",
    devices=(
        TrajectoryDevice(
            name="SR/BPM",
            x_row="SR:BPM:X",
            y_row="SR:BPM:Y",
            sum_row="SR:BPM:SUM",
            arm=ARM,
            armed="SR:BPM:ARMED",
        ),
    ),
    injection_request=REQUEST,
    injection_busy="SR:INJ:BUSY",
    trigger_sample=10,
)

_ENTRY = "async def inject(handle):\n    pass\n"


def _module(tmp_path: Path, body: str) -> Path:
    directory = tmp_path / "sequence"
    directory.mkdir(exist_ok=True)
    (directory / SEQUENCE_MODULE_FILE).write_text(textwrap.dedent(body))
    return directory


def _with_metadata(tmp_path: Path, metadata: str, entries: str = _ENTRY) -> Path:
    return _module(tmp_path, f"SEQUENCE_METADATA = {metadata}\n\n{entries}")


def _refusal(directory: Path) -> str:
    with pytest.raises(StartupRefused) as caught:
        load_sequence(directory)
    message = str(caught.value)
    assert message.startswith("trajectory service refused to start: sequence module ")
    assert str(directory / SEQUENCE_MODULE_FILE) in message
    return message


# -- the fixture module ---------------------------------------------------------


def test_the_loader_loads_the_fixture_module_with_its_inject_and_pause() -> None:
    loaded = load_sequence(FIXTURE_DIR)

    assert isinstance(loaded, SequenceModule)
    assert loaded.path == FIXTURE_DIR / SEQUENCE_MODULE_FILE
    assert loaded.metadata.max_pause_s == 0.5
    assert loaded.inject is loaded.module.inject


def test_the_loader_registers_the_module_under_its_own_prefix() -> None:
    loaded = load_sequence(FIXTURE_DIR)

    name = module_name_for(FIXTURE_DIR / SEQUENCE_MODULE_FILE, SEQUENCE_MODULE_PREFIX)
    assert loaded is not None
    assert loaded.module.__name__ == name
    assert sys.modules[name] is loaded.module


def test_the_loader_accepts_a_minimal_module(tmp_path: Path) -> None:
    directory = _with_metadata(tmp_path, "{'max_pause_s': 1}")

    loaded = load_sequence(directory)

    assert loaded is not None
    assert loaded.metadata.max_pause_s == 1.0
    assert loaded.inject is loaded.module.inject


def test_the_loader_accepts_max_pause_at_the_ceiling(tmp_path: Path) -> None:
    directory = _with_metadata(tmp_path, f"{{'max_pause_s': {MAX_PAUSE_CEILING_S}}}")

    loaded = load_sequence(directory)

    assert loaded is not None and loaded.metadata.max_pause_s == MAX_PAUSE_CEILING_S


# -- no module -------------------------------------------------------------------


def test_the_loader_gives_no_module_for_an_unset_directory() -> None:
    assert load_sequence(None) is None


def test_the_loader_gives_no_module_for_a_missing_directory(tmp_path: Path) -> None:
    assert load_sequence(tmp_path / "absent") is None


def test_the_loader_gives_no_module_for_a_directory_holding_none(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "other.py").write_text(_ENTRY)

    assert load_sequence(tmp_path) is None


# -- refused modules -------------------------------------------------------------


def test_the_loader_refuses_a_module_that_raises_while_loading(tmp_path: Path) -> None:
    directory = _module(tmp_path, "raise RuntimeError('broken deck')\n")

    message = _refusal(directory)

    assert "RuntimeError: broken deck" in message
    name = module_name_for(directory / SEQUENCE_MODULE_FILE, SEQUENCE_MODULE_PREFIX)
    assert name not in sys.modules


def test_the_loader_refuses_a_module_with_a_syntax_error(tmp_path: Path) -> None:
    assert "SyntaxError" in _refusal(_module(tmp_path, "def inject(:\n"))


def test_the_loader_refuses_a_module_with_no_inject(tmp_path: Path) -> None:
    directory = _with_metadata(
        tmp_path, "{'max_pause_s': 1}", "async def fire(handle):\n    pass\n"
    )

    assert "no 'inject' entry point" in _refusal(directory)


@pytest.mark.parametrize(
    "entries",
    [
        "def inject(handle):\n    pass\n",
        "inject = 3\n",
        "async def _gen(handle):\n    yield\ninject = _gen\n",
    ],
    ids=["sync-inject", "not-callable", "async-generator"],
)
def test_the_loader_refuses_a_non_coroutine_entry_point(tmp_path: Path, entries: str) -> None:
    directory = _with_metadata(tmp_path, "{'max_pause_s': 1}", entries)

    assert "is not a coroutine function" in _refusal(directory)


def test_the_loader_refuses_a_module_with_no_metadata(tmp_path: Path) -> None:
    assert "defines no SEQUENCE_METADATA" in _refusal(_module(tmp_path, _ENTRY))


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ("[]", "not a mapping"),
        ("{}", "has no max_pause_s"),
        ("{'max_pause_s': 1, 'extra': 1}", "unknown keys ['extra']"),
        ("{'max_pause_s': 'soon'}", "not a number of seconds"),
        ("{'max_pause_s': True}", "not a number of seconds"),
        ("{'max_pause_s': float('nan')}", "not a number of seconds"),
        ("{'max_pause_s': 0}", "must be positive"),
        ("{'max_pause_s': -1}", "must be positive"),
        (f"{{'max_pause_s': {MAX_PAUSE_CEILING_S + 0.1}}}", "above the ceiling"),
        ("{'max_pause_s': 1, 'slots': {}}", "unknown keys ['slots']"),
        ("{'max_pause_s': 1, 'writes': {}}", "unknown keys ['writes']"),
    ],
)
def test_the_loader_refuses_malformed_metadata(
    tmp_path: Path, metadata: str, expected: str
) -> None:
    assert expected in _refusal(_with_metadata(tmp_path, metadata))


def test_a_metadata_key_the_contract_dropped_names_where_it_is_declared_now(
    tmp_path: Path,
) -> None:
    directory = _with_metadata(tmp_path, "{'max_pause_s': 1, 'slots': {'a': 'SR:X'}}")

    assert "measurement file's trajectory block" in _refusal(directory)


# -- the start() slot -------------------------------------------------------------


class _Connector:
    async def disconnect(self) -> None:
        return None


def _stub_start(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, settings: TrajectorySettings
) -> None:
    """Stub every other startup check, so the sequence slot is all that is under test."""
    import osprey_connectors.workspace as workspace

    async def _connect(*_: Any) -> _Connector:
        return _Connector()

    monkeypatch.delenv(startup.SEQUENCE_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    monkeypatch.setattr(workspace, "resolve_project_root", lambda config: project_root)
    monkeypatch.setattr(startup, "service_settings", lambda config: ("va", None, settings))
    monkeypatch.setattr(startup, "check_shared_dirs", lambda target: project_root / "lock")
    monkeypatch.setattr(startup, "check_limits_file", lambda section, target: None)
    monkeypatch.setattr(startup, "_facts", lambda config: {})
    monkeypatch.setattr(startup, "select_commands", lambda facts, model: COMMANDS)
    monkeypatch.setattr(startup, "arm_raw_put_block", lambda: None)
    monkeypatch.setattr(startup, "_connect", _connect)


async def test_the_loader_fills_the_started_slot_from_the_resolved_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _with_metadata(tmp_path, "{'max_pause_s': 2}")
    _stub_start(monkeypatch, tmp_path, TrajectorySettings(sequence_dir="sequence"))

    started = await startup.start()

    assert started.sequence_dir == directory
    assert isinstance(started.sequence, SequenceModule)
    assert started.sequence.path == directory / SEQUENCE_MODULE_FILE
    assert started.sequence.metadata.max_pause_s == 2.0


async def test_the_loader_leaves_the_started_slot_empty_when_the_dir_holds_no_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sequence").mkdir()
    _stub_start(monkeypatch, tmp_path, TrajectorySettings(sequence_dir="sequence"))

    started = await startup.start()

    assert started.sequence_dir == tmp_path / "sequence"
    assert started.sequence is None


async def test_the_loader_refuses_start_before_connecting_on_an_invalid_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_metadata(tmp_path, f"{{'max_pause_s': {MAX_PAUSE_CEILING_S * 2}}}")
    _stub_start(monkeypatch, tmp_path, TrajectorySettings(sequence_dir="sequence"))
    connected: list[str] = []

    async def _connect(_section: Any, target: str) -> _Connector:
        connected.append(target)
        return _Connector()

    monkeypatch.setattr(startup, "_connect", _connect)

    with pytest.raises(StartupRefused, match="above the ceiling"):
        await startup.start()
    assert connected == []


# -- the handle ------------------------------------------------------------------

OTHER = "SR:OTHER:SETPOINT"
ALICE_LIVE = ActionContext(target="live", owner="alice", approved_target="live")


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host repo on a live baseline, stamped ``live``, whose write path reads owner alice."""
    reset_run_state(monkeypatch)
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(ENV_CONTROL_IDENTITY, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    return root


class _Machine:
    """A connector holding channel values; it logs every read, setting write and command batch."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._control_target = "live"
        self._limits_validator = LimitsValidator(
            {
                ARM: ChannelLimitsConfig(ARM, 0, 1, writable=True),
                REQUEST: ChannelLimitsConfig(REQUEST, 0, 1000, writable=True),
            },
            {"mode": "exclusive", "on_violation": "error"},
            {
                ARM: {"min_value": 0, "max_value": 1, "confirm": False},
                REQUEST: {"min_value": 0, "max_value": 1000, "confirm": False},
            },
        )
        self.values: dict[str, Any] = {OTHER: 0, REQUEST: 0, **(values or {})}
        self.log: list[tuple[Any, ...]] = []

    def _current_value_reader(self) -> None:
        return None

    async def read_channel(
        self,
        address: str,
        timeout: float | None = None,  # noqa: ARG002 - the connector signature
    ) -> ChannelValue:
        self.log.append(("read", address))
        return ChannelValue(value=self.values[address], timestamp=datetime.now(UTC))

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> None:
        self.log.append(("set", address, value))

    async def write_multiple_channels(
        self, operations: list[tuple[str, Any]], **kwargs: Any
    ) -> list[ChannelWriteResult]:
        self.log.append(("command", list(operations), kwargs))
        return [
            ChannelWriteResult(
                channel_address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED
            )
            for address, value in operations
        ]


class _Clock:
    """A clock that moves only when the handle sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def _fixture() -> SequenceModule:
    loaded = load_sequence(FIXTURE_DIR)
    assert loaded is not None
    return loaded


def _handle(
    module: SequenceModule,
    machine: _Machine,
    *,
    stop: asyncio.Event | None = None,
    clock: _Clock | None = None,
) -> SequenceHandle:
    clock = clock or _Clock()
    return SequenceHandle(
        machine,
        commands=COMMANDS,
        metadata=module.metadata,
        stop=stop or asyncio.Event(),
        sleep=clock.sleep,
        clock=clock,
    )


async def _run(module: SequenceModule, handle: SequenceHandle) -> None:
    with lock(None, context=ALICE_LIVE):
        await module.inject(handle)


@pytest.mark.usefixtures("live")
async def test_the_handle_fires_the_injection_request_from_inject_as_a_command() -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, machine)

    await _run(module, handle)

    assert machine.log == [("command", [(REQUEST, 1)], {"confirm": False})]
    assert handle.commands_fired == [REQUEST]


@pytest.mark.usefixtures("live")
async def test_the_handle_sends_the_request_through_command_write_and_its_guards() -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, machine)

    with pytest.raises(OspreyWriteRefused, match="run lock"):
        await module.inject(handle)

    assert machine.log == []
    assert handle.commands_fired == []


@pytest.mark.parametrize("name", [ARM, REQUEST, "arm", "kicker_enable", OTHER])
@pytest.mark.usefixtures("live")
async def test_the_handle_refuses_any_write_but_the_injection_request(name: str) -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, machine)

    with lock(None, context=ALICE_LIVE), pytest.raises(SequenceRefused) as caught:
        await handle.write(name, 1)

    assert caught.value.code == "sequence_refused"
    assert "the injection request and nothing else" in caught.value.detail
    assert machine.log == []
    assert handle.commands_fired == []


@pytest.mark.usefixtures("live")
async def test_the_handle_refuses_the_request_when_the_build_has_none() -> None:
    module, machine = _fixture(), _Machine()
    handle = SequenceHandle(machine, commands=None, metadata=module.metadata, stop=asyncio.Event())

    with pytest.raises(SequenceRefused, match="no injection request"):
        await handle.write("injection_request", 1)
    assert machine.log == []


@pytest.mark.usefixtures("live")
async def test_a_set_stop_event_ends_inject_at_its_first_call() -> None:
    module, machine, stop = _fixture(), _Machine(), asyncio.Event()
    stop.set()
    handle = _handle(module, machine, stop=stop)

    with pytest.raises(SequenceStopped):
        await _run(module, handle)
    assert machine.log == []


@pytest.mark.usefixtures("live")
async def test_a_stop_during_a_pause_ends_it_at_once() -> None:
    module, machine, stop = _fixture(), _Machine(), asyncio.Event()
    forever = asyncio.Event()

    async def never(_seconds: float) -> None:
        await forever.wait()

    handle = SequenceHandle(
        machine,
        commands=COMMANDS,
        metadata=module.metadata,
        stop=stop,
        sleep=never,
    )
    pausing = asyncio.ensure_future(handle.sleep(0.5))
    await asyncio.sleep(0)
    stop.set()

    with pytest.raises(SequenceStopped):
        await asyncio.wait_for(pausing, timeout=1.0)


@pytest.mark.parametrize("seconds", [-0.1, 0.6, float("nan"), True])
@pytest.mark.usefixtures("live")
async def test_a_pause_outside_zero_to_max_pause_is_refused(seconds: Any) -> None:
    module, machine, clock = _fixture(), _Machine(), _Clock()
    handle = _handle(module, machine, clock=clock)

    with pytest.raises(SequenceRefused, match="max_pause_s"):
        await handle.sleep(seconds)
    assert clock.sleeps == []


@pytest.mark.usefixtures("live")
async def test_read_gives_the_request_by_name_or_any_address() -> None:
    module = _fixture()
    machine = _Machine({REQUEST: 7, "SR:INJ:BUSY": 0})
    handle = _handle(module, machine)

    assert await handle.read("injection_request") == 7
    assert await handle.read("SR:INJ:BUSY") == 0
    assert machine.log == [("read", REQUEST), ("read", "SR:INJ:BUSY")]


@pytest.mark.usefixtures("live")
async def test_wait_for_returns_the_first_value_that_satisfies_the_condition() -> None:
    module, clock = _fixture(), _Clock()
    machine = _Machine({"SR:INJ:BUSY": 1})
    reads = 0
    original = machine.read_channel

    async def read_channel(address: str, timeout: float | None = None) -> ChannelValue:
        nonlocal reads
        reads += 1
        if reads == 3:
            machine.values[address] = 0
        return await original(address, timeout)

    machine.read_channel = read_channel  # type: ignore[method-assign]
    handle = _handle(module, machine, clock=clock)

    assert await handle.wait_for("SR:INJ:BUSY", 0, timeout=1.0) == 0
    assert reads == 3
    assert clock.sleeps == [0.1, 0.1]
    assert await handle.wait_for("SR:INJ:BUSY", lambda value: value < 1, timeout=1.0) == 0


@pytest.mark.usefixtures("live")
async def test_wait_for_times_out_on_its_clock() -> None:
    module, clock = _fixture(), _Clock()
    handle = _handle(module, _Machine({"SR:INJ:BUSY": 1}), clock=clock)

    with pytest.raises(TimeoutError, match="SR:INJ:BUSY"):
        await handle.wait_for("SR:INJ:BUSY", 0, timeout=0.25)
    assert sum(clock.sleeps) == pytest.approx(0.25)


@pytest.mark.usefixtures("live")
async def test_wait_for_raises_sequence_stopped_when_the_stop_event_is_set() -> None:
    module, stop = _fixture(), asyncio.Event()
    clock = _Clock()

    async def stopping_sleep(_seconds: float) -> None:
        stop.set()
        await asyncio.sleep(0)

    handle = SequenceHandle(
        _Machine({"SR:INJ:BUSY": 1}),
        commands=COMMANDS,
        metadata=module.metadata,
        stop=stop,
        sleep=stopping_sleep,
        clock=clock,
    )

    with pytest.raises(SequenceStopped):
        await handle.wait_for("SR:INJ:BUSY", 0, timeout=5.0)


@pytest.mark.usefixtures("live")
async def test_wait_for_refuses_a_bad_timeout() -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, machine)

    with pytest.raises(SequenceRefused, match="positive"):
        await handle.wait_for(OTHER, 1, timeout=0)
    assert machine.log == []
