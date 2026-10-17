"""The sequence module loads once from its directory, or the service refuses to start.

The loader tests run against the fixture module under ``sequences/fixture`` and
against modules written into a temporary directory, one defect each. The
handle tests run the fixture's entry points through a :class:`SequenceHandle`
on a recording connector, under the product's own run lock and
``command_write``.
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
    SequenceSlot,
    SequenceStopped,
    load_sequence,
)
from osprey.services.trajectory.startup import (
    StartupRefused,
    TrajectoryCommands,
    TrajectorySettings,
)
from osprey.services.trajectory.state import Moved
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
from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError

FIXTURE_DIR = Path(__file__).parent / "sequences" / "fixture"

ARM = "SR:BPM:ARM"
REQUEST = "SR:INJ:REQUEST"
COMMANDS = TrajectoryCommands(
    model="SR",
    arm=(ARM,),
    injection_request=REQUEST,
    injection_busy="SR:INJ:BUSY",
    armed=("SR:BPM:ARMED",),
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
        load_sequence(directory, COMMANDS)
    message = str(caught.value)
    assert message.startswith("trajectory service refused to start: sequence module ")
    assert str(directory / SEQUENCE_MODULE_FILE) in message
    return message


# -- the fixture module ---------------------------------------------------------


def test_the_loader_loads_the_fixture_module_with_its_slots_writes_and_pause() -> None:
    loaded = load_sequence(FIXTURE_DIR, COMMANDS)

    assert isinstance(loaded, SequenceModule)
    assert loaded.path == FIXTURE_DIR / SEQUENCE_MODULE_FILE
    assert dict(loaded.metadata.slots) == {
        "acquisition_mode": SequenceSlot(name="acquisition_mode", tag="bpm_acquisition_mode"),
        "kicker_enable": SequenceSlot(name="kicker_enable", address="SR:BUNCH:KICKER:ENABLE"),
    }
    assert loaded.metadata.slots["acquisition_mode"].per_bpm
    assert not loaded.metadata.slots["kicker_enable"].per_bpm
    assert dict(loaded.metadata.writes) == {
        "bpm_mode": ("acquisition_mode",),
        "bunch_clean": ("kicker_enable",),
    }
    assert loaded.metadata.max_pause_s == 0.5
    assert list(loaded.entry_points) == ["inject", "bunch_clean", "bpm_mode"]
    assert loaded.inject is loaded.module.inject
    assert loaded.entry_point("bunch_clean") is loaded.module.bunch_clean


def test_the_loader_registers_the_module_under_its_own_prefix() -> None:
    loaded = load_sequence(FIXTURE_DIR, COMMANDS)

    name = module_name_for(FIXTURE_DIR / SEQUENCE_MODULE_FILE, SEQUENCE_MODULE_PREFIX)
    assert loaded is not None
    assert loaded.module.__name__ == name
    assert sys.modules[name] is loaded.module


def test_the_loader_accepts_a_minimal_module_without_a_command_set(tmp_path: Path) -> None:
    directory = _with_metadata(tmp_path, "{'max_pause_s': 1}")

    loaded = load_sequence(directory, None)

    assert loaded is not None
    assert dict(loaded.metadata.slots) == {}
    assert dict(loaded.metadata.writes) == {}
    assert loaded.metadata.max_pause_s == 1.0
    assert list(loaded.entry_points) == ["inject"]
    assert loaded.entry_point("bpm_mode") is None


def test_the_loader_accepts_max_pause_at_the_ceiling(tmp_path: Path) -> None:
    directory = _with_metadata(tmp_path, f"{{'max_pause_s': {MAX_PAUSE_CEILING_S}}}")

    loaded = load_sequence(directory, COMMANDS)

    assert loaded is not None and loaded.metadata.max_pause_s == MAX_PAUSE_CEILING_S


# -- no module -------------------------------------------------------------------


def test_the_loader_gives_no_module_for_an_unset_directory() -> None:
    assert load_sequence(None, COMMANDS) is None


def test_the_loader_gives_no_module_for_a_missing_directory(tmp_path: Path) -> None:
    assert load_sequence(tmp_path / "absent", COMMANDS) is None


def test_the_loader_gives_no_module_for_a_directory_holding_none(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "other.py").write_text(_ENTRY)

    assert load_sequence(tmp_path, COMMANDS) is None


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
        tmp_path, "{'max_pause_s': 1}", "async def bpm_mode(handle):\n    pass\n"
    )

    assert "no 'inject' entry point" in _refusal(directory)


@pytest.mark.parametrize(
    "entries",
    [
        "def inject(handle):\n    pass\n",
        "inject = 3\n",
        _ENTRY + "def bunch_clean(handle):\n    pass\n",
        _ENTRY + "async def _gen(handle):\n    yield\nbpm_mode = _gen\n",
    ],
    ids=["sync-inject", "not-callable", "sync-bunch-clean", "async-generator"],
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
        ("{'max_pause_s': 1, 'slots': ['a']}", "slots is list, not a mapping"),
        ("{'max_pause_s': 1, 'slots': {'': 'X'}}", "slot name ''"),
        ("{'max_pause_s': 1, 'slots': {'a': 3}}", "slot 'a' is 3"),
        ("{'max_pause_s': 1, 'slots': {'a': ''}}", "slot 'a' is ''"),
        ("{'max_pause_s': 1, 'slots': {'a': {'tag': ''}}}", "slot 'a' is"),
        ("{'max_pause_s': 1, 'slots': {'a': {'tag': 'x', 'other': 1}}}", "slot 'a' is"),
        ("{'max_pause_s': 1, 'writes': []}", "writes is list, not a mapping"),
        ("{'max_pause_s': 1, 'writes': {'fire': []}}", "'fire', not an entry point"),
        ("{'max_pause_s': 1, 'writes': {'bpm_mode': []}}", "the module does not define"),
    ],
)
def test_the_loader_refuses_malformed_metadata(
    tmp_path: Path, metadata: str, expected: str
) -> None:
    assert expected in _refusal(_with_metadata(tmp_path, metadata))


@pytest.mark.parametrize(
    ("writes", "expected"),
    [
        ("'a'", "not a list of slot names"),
        ("['b']", "undeclared slots ['b']"),
        ("['a', 'a']", "names a slot twice"),
    ],
)
def test_the_loader_refuses_writes_naming_anything_but_declared_slots(
    tmp_path: Path, writes: str, expected: str
) -> None:
    directory = _with_metadata(
        tmp_path,
        f"{{'max_pause_s': 1, 'slots': {{'a': 'SR:X'}}, 'writes': {{'bunch_clean': {writes}}}}}",
        _ENTRY + "async def bunch_clean(handle):\n    pass\n",
    )

    assert expected in _refusal(directory)


def test_the_loader_refuses_a_writes_entry_for_inject(tmp_path: Path) -> None:
    directory = _with_metadata(
        tmp_path, "{'max_pause_s': 1, 'slots': {'a': 'SR:X'}, 'writes': {'inject': ['a']}}"
    )

    assert "writes the injection request and nothing else" in _refusal(directory)


@pytest.mark.parametrize("address", [ARM, REQUEST])
def test_the_loader_refuses_a_module_slot_naming_a_service_command(
    tmp_path: Path, address: str
) -> None:
    directory = _with_metadata(tmp_path, f"{{'max_pause_s': 1, 'slots': {{'a': {address!r}}}}}")

    message = _refusal(directory)

    assert f"slot 'a' names {address!r}" in message
    assert "the service alone writes" in message


def test_the_loader_refuses_a_module_slot_tagging_the_arm_signal(tmp_path: Path) -> None:
    directory = _with_metadata(
        tmp_path, "{'max_pause_s': 1, 'slots': {'a': {'tag': 'bpm_arm_command'}}}"
    )

    assert "tags 'bpm_arm_command'" in _refusal(directory)


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

KICKER = "SR:BUNCH:KICKER:ENABLE"
ACQUISITION_TAG = "bpm_acquisition_mode"
MODES = ("SR01:BPM1:MODE", "SR01:BPM2:MODE")
TAGS = {ACQUISITION_TAG: MODES}
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
        self.values: dict[str, Any] = {KICKER: 0, **dict.fromkeys(MODES, 0), **(values or {})}
        self.log: list[tuple[Any, ...]] = []
        self.setting_kwargs: list[dict[str, Any]] = []
        self.refuse: set[str] = set()
        self.fail: set[str] = set()
        self.on_write: Any = None

    def _current_value_reader(self) -> None:
        return None

    async def read_channel(
        self,
        address: str,
        timeout: float | None = None,  # noqa: ARG002 - the connector signature
    ) -> ChannelValue:
        self.log.append(("read", address))
        return ChannelValue(value=self.values[address], timestamp=datetime.now(UTC))

    async def write_channel_checked(
        self, address: str, value: Any, **kwargs: Any
    ) -> ChannelWriteResult:
        self.log.append(("set", address, value))
        self.setting_kwargs.append(kwargs)
        if self.on_write is not None:
            await self.on_write(address)
        if address in self.refuse:
            raise ChannelWriteBlockedError(address, "LIMITS")
        if address in self.fail:
            raise ChannelWriteFailedError(
                address, "UNCONFIRMED", outcome=WriteOutcome.UNCONFIRMED, value_written=value
            )
        self.values[address] = value
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )

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
    loaded = load_sequence(FIXTURE_DIR, COMMANDS)
    assert loaded is not None
    return loaded


def _handle(
    module: SequenceModule,
    entry_point: str,
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
        entry_point=entry_point,
        stop=stop or asyncio.Event(),
        tag_addresses=TAGS,
        sleep=clock.sleep,
        clock=clock,
    )


async def _run(module: SequenceModule, entry_point: str, handle: SequenceHandle) -> None:
    entry = module.entry_point(entry_point)
    assert entry is not None
    with lock(None, context=ALICE_LIVE):
        await entry(handle)


@pytest.mark.usefixtures("live")
async def test_the_handle_fires_the_injection_request_from_inject_as_a_command() -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, "inject", machine)

    await _run(module, "inject", handle)

    assert machine.log == [("command", [(REQUEST, 1)], {"confirm": False})]
    assert handle.commands_fired == [REQUEST]
    assert handle.moved == []


@pytest.mark.usefixtures("live")
async def test_the_handle_sends_the_request_through_command_write_and_its_guards() -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, "inject", machine)

    with pytest.raises(OspreyWriteRefused, match="run lock"):
        await module.inject(handle)

    assert machine.log == []
    assert handle.commands_fired == []


@pytest.mark.usefixtures("live")
async def test_bpm_mode_writes_its_per_bpm_slot_at_every_bpm_with_per_channel_confirm() -> None:
    module, machine = _fixture(), _Machine({MODES[0]: 3, MODES[1]: 4})
    handle = _handle(module, "bpm_mode", machine)

    await _run(module, "bpm_mode", handle)

    assert machine.log == [
        ("read", MODES[0]),
        ("set", MODES[0], 1),
        ("read", MODES[1]),
        ("set", MODES[1], 1),
    ]
    assert machine.setting_kwargs == [{}, {}]  # confirm left to each channel's record
    assert handle.moved == [
        Moved(name=MODES[0], before=3, outcome="confirmed"),
        Moved(name=MODES[1], before=4, outcome="confirmed"),
    ]
    assert handle.commands_fired == []


@pytest.mark.usefixtures("live")
async def test_bunch_clean_pulses_its_slot_and_pauses_on_the_injected_clock() -> None:
    module, machine, clock = _fixture(), _Machine(), _Clock()
    handle = _handle(module, "bunch_clean", machine, clock=clock)

    await _run(module, "bunch_clean", handle)

    assert [entry for entry in machine.log if entry[0] == "set"] == [
        ("set", KICKER, 1),
        ("set", KICKER, 0),
    ]
    assert clock.sleeps == [0.1]
    assert handle.moved == [
        Moved(name=KICKER, before=0, outcome="confirmed"),
        Moved(name=KICKER, before=1, outcome="confirmed"),
    ]


@pytest.mark.usefixtures("live")
async def test_the_handle_refuses_a_request_write_from_bpm_mode(tmp_path: Path) -> None:
    directory = _module(
        tmp_path,
        """
        SEQUENCE_METADATA = {"max_pause_s": 1}

        async def inject(handle):
            pass

        async def bpm_mode(handle):
            await handle.write("injection_request", 1)
        """,
    )
    module = load_sequence(directory, COMMANDS)
    assert module is not None
    machine = _Machine()
    handle = _handle(module, "bpm_mode", machine)

    with pytest.raises(SequenceRefused, match="only inject requests an injection"):
        await _run(module, "bpm_mode", handle)

    assert machine.log == []
    assert handle.commands_fired == []


@pytest.mark.parametrize("entry_point", ["inject", "bpm_mode", "bunch_clean"])
@pytest.mark.parametrize("name", [ARM, REQUEST, "arm", KICKER, "SR:OTHER:SETPOINT"])
@pytest.mark.usefixtures("live")
async def test_the_handle_refuses_arm_addresses_and_undeclared_names_from_any_entry_point(
    entry_point: str, name: str
) -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, entry_point, machine)

    with lock(None, context=ALICE_LIVE), pytest.raises(SequenceRefused) as caught:
        await handle.write(name, 1)

    assert caught.value.code == "sequence_refused"
    assert machine.log == []


@pytest.mark.parametrize(
    ("entry_point", "slot"),
    [
        ("inject", "kicker_enable"),
        ("bpm_mode", "kicker_enable"),
        ("bunch_clean", "acquisition_mode"),
    ],
)
@pytest.mark.usefixtures("live")
async def test_the_handle_refuses_a_slot_the_entry_point_does_not_declare(
    entry_point: str, slot: str
) -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, entry_point, machine)

    with pytest.raises(SequenceRefused, match=f"may not write slot {slot!r}"):
        await handle.write(slot, 1)

    assert machine.log == []


@pytest.mark.usefixtures("live")
async def test_a_refused_setting_write_is_not_listed_and_a_failed_one_is() -> None:
    module, machine = _fixture(), _Machine()
    machine.refuse.add(MODES[0])
    refused = _handle(module, "bpm_mode", machine)

    with pytest.raises(ChannelWriteBlockedError):
        await refused.write("acquisition_mode", 1)
    assert refused.moved == []

    machine.refuse.clear()
    machine.fail.add(MODES[0])
    failed = _handle(module, "bpm_mode", machine)
    with pytest.raises(ChannelWriteFailedError):
        await failed.write("acquisition_mode", 1)
    assert failed.moved == [Moved(name=MODES[0], before=0, outcome="unconfirmed")]


@pytest.mark.usefixtures("live")
async def test_a_per_bpm_slot_whose_signal_no_bpm_carries_is_refused() -> None:
    module, machine = _fixture(), _Machine()
    handle = SequenceHandle(
        machine,
        commands=COMMANDS,
        metadata=module.metadata,
        entry_point="bpm_mode",
        stop=asyncio.Event(),
    )

    with pytest.raises(SequenceRefused, match=ACQUISITION_TAG):
        await handle.write("acquisition_mode", 1)
    assert machine.log == []


@pytest.mark.usefixtures("live")
async def test_a_set_stop_event_ends_the_entry_point_at_its_first_call() -> None:
    module, machine, stop = _fixture(), _Machine(), asyncio.Event()
    stop.set()

    for entry_point in ("inject", "bpm_mode", "bunch_clean"):
        handle = _handle(module, entry_point, machine, stop=stop)
        with pytest.raises(SequenceStopped):
            await _run(module, entry_point, handle)
    assert machine.log == []


@pytest.mark.usefixtures("live")
async def test_a_stop_between_two_bpm_writes_ends_after_the_first_and_lists_it() -> None:
    module, machine, stop = _fixture(), _Machine(), asyncio.Event()

    async def stop_on_write(_address: str) -> None:
        stop.set()

    machine.on_write = stop_on_write
    handle = _handle(module, "bpm_mode", machine, stop=stop)

    with pytest.raises(SequenceStopped):
        await _run(module, "bpm_mode", handle)

    assert [entry for entry in machine.log if entry[0] == "set"] == [("set", MODES[0], 1)]
    assert handle.moved == [Moved(name=MODES[0], before=0, outcome="confirmed")]


@pytest.mark.usefixtures("live")
async def test_a_cancelled_setting_write_still_lands_in_moved() -> None:
    module, machine = _fixture(), _Machine()
    started, release = asyncio.Event(), asyncio.Event()

    async def block(_address: str) -> None:
        started.set()
        await release.wait()

    machine.on_write = block
    handle = _handle(module, "bunch_clean", machine)
    task = asyncio.ensure_future(handle.write("kicker_enable", 1))
    await started.wait()
    task.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handle.moved == [Moved(name=KICKER, before=0, outcome="confirmed")]


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
        entry_point="bunch_clean",
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
    handle = _handle(module, "bunch_clean", machine, clock=clock)

    with pytest.raises(SequenceRefused, match="max_pause_s"):
        await handle.sleep(seconds)
    assert clock.sleeps == []


@pytest.mark.usefixtures("live")
async def test_read_gives_a_slot_by_name_or_any_address() -> None:
    module = _fixture()
    machine = _Machine({MODES[0]: 1, MODES[1]: 2, "SR:INJ:BUSY": 0})
    handle = _handle(module, "inject", machine)

    assert await handle.read("acquisition_mode") == [1, 2]
    assert await handle.read("kicker_enable") == 0
    assert await handle.read("SR:INJ:BUSY") == 0


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
    handle = _handle(module, "inject", machine, clock=clock)

    assert await handle.wait_for("SR:INJ:BUSY", 0, timeout=1.0) == 0
    assert reads == 3
    assert clock.sleeps == [0.1, 0.1]
    assert await handle.wait_for("SR:INJ:BUSY", lambda value: value < 1, timeout=1.0) == 0


@pytest.mark.usefixtures("live")
async def test_wait_for_times_out_on_its_clock() -> None:
    module, clock = _fixture(), _Clock()
    handle = _handle(module, "inject", _Machine({"SR:INJ:BUSY": 1}), clock=clock)

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
        entry_point="inject",
        stop=stop,
        sleep=stopping_sleep,
        clock=clock,
    )

    with pytest.raises(SequenceStopped):
        await handle.wait_for("SR:INJ:BUSY", 0, timeout=5.0)


@pytest.mark.usefixtures("live")
async def test_wait_for_refuses_a_per_bpm_slot_and_a_bad_timeout() -> None:
    module, machine = _fixture(), _Machine()
    handle = _handle(module, "inject", machine)

    with pytest.raises(SequenceRefused, match="one per BPM"):
        await handle.wait_for("acquisition_mode", 1, timeout=1.0)
    with pytest.raises(SequenceRefused, match="positive"):
        await handle.wait_for(KICKER, 1, timeout=0)
    assert machine.log == []


def test_a_handle_names_an_entry_point() -> None:
    with pytest.raises(ValueError, match="not an entry point"):
        SequenceHandle(
            _Machine(),
            commands=COMMANDS,
            metadata=_fixture().metadata,
            entry_point="setup",
            stop=asyncio.Event(),
        )
