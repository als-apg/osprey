"""Setup: the declared settings as a server-checked fact.

**The declaration.** A model's measurement file declares its settings
(``trajectory_slots[model]["settings"]``, :func:`declared_settings`): each has
an id, a scope, the address written, the address read back (the written one
when no readback is paired), a tolerance (exact match when none) and the
``timeout_s`` its readback has to settle in. A ``device`` setting is held once
per BPM device, a ``global`` one once for the model.

**A setup** (:func:`setup_action`) is one holder action and a partial update:
only the setting ids its body names are applied, every other setting is kept
as it reads. Inside the action:

1. ``excluded_bpms``, when the body carries them, are handed to the one
   SolveConfig writer (``cause="setup"``) and the step is brought in line;
   setup never stores exclusions itself;
2. every declared setting of every included BPM (device scope) and every
   global setting is read in one ``read_multiple_channels``: the written
   address, for the value a put-back restores, and the readback;
3. a named setting is written only where its readback differs from the asked
   value, one checked write each, every write listed as a
   :class:`~.state.Moved` in put-back shape;
4. the written settings' readbacks are polled until each is within its
   tolerance or its ``timeout_s`` passed; those that missed are *unsettled*.

When every write landed and settled the setup **takes**: the value expected
of every readback (the asked value of a named setting, the value read of every
other) is held as the current :class:`~.state.SetupState`, whose
``settings_sha`` is the sha256 of those values keyed by written address, and
it is persisted per target and model beside the SolveConfig overrides
(``<state dir>/setup/<model>.json``). A setup that moved settings and did not
take voids the previous one. Any setup that moved a setting voids the step.

**The re-check** (:class:`SetupKeeper`) reads every held readback of the
included BPMs and compares it with the value expected of it: at start, and
before every inject and readout turn of a chain on either door (the chain
folds the readbacks into its pre-inject batch read). A divergence, or a
readback that could not be read, voids the setup and the step with a notice,
and the turn's shot carries no ``settings_sha``. A void setup admits no panel
chain; the agent door is never refused for setup.

Refusals: 400 ``unknown_setting`` for a setting id the model does not declare,
409 ``continuous_running`` while a continuous run is in progress (as the
sessions refuse it), 502 ``setup_failed`` when a write failed part-way (its
moves are listed), 503 ``setup_unreadable`` when the settings cannot be read,
503 ``setup_unsaved`` when the setup cannot be persisted.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.services.trajectory.calls import guarded_call
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.state import (
    Moved,
    SettingReadback,
    SetupBody,
    SetupReadbacks,
    SetupResult,
    SetupState,
    SolveConfigPartial,
    Step,
)

if TYPE_CHECKING:
    from osprey.services.trajectory.admit import Admission
    from osprey.services.trajectory.sessions import Sessions
    from osprey.services.trajectory.store import TrajectoryStore

__all__ = [
    "SETUP",
    "SETUP_DIR",
    "SETUP_FAILED",
    "SETUP_UNREADABLE",
    "SETUP_UNSAVED",
    "UNKNOWN_SETTING",
    "DeclaredSetting",
    "Expected",
    "SetupFailed",
    "SetupKeeper",
    "SetupOutcome",
    "SetupRefused",
    "declared_settings",
    "settings_sha",
    "setup_action",
]

logger = logging.getLogger(__name__)

UNKNOWN_SETTING = "unknown_setting"
SETUP_FAILED = "setup_failed"
SETUP_UNREADABLE = "setup_unreadable"
SETUP_UNSAVED = "setup_unsaved"

#: The name a setup action carries in the service state and its records.
SETUP = "setup"

#: The directory, under the store's, holding one setup file per model.
SETUP_DIR = "setup"

#: The declared setting whose value is the turns every BPM records.
TURNS_TRACKED = "turns_tracked"

#: The settle time of a setting that declares none, seconds.
DEFAULT_SETTLE_S = 5.0

#: Seconds between two readback polls of the settle.
SETTLE_POLL_S = 0.1


class SetupRefused(Refused):
    """A setup refused before it wrote, or a setup state that cannot be kept."""

    statuses = {
        UNKNOWN_SETTING: 400,
        SETUP_FAILED: 502,
        SETUP_UNREADABLE: 503,
        SETUP_UNSAVED: 503,
    }


class SetupFailed(Exception):
    """A setup write failed part-way; ``moved`` lists what was written before it."""

    def __init__(self, reason: str, moved: list[Moved]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.moved = moved


# -- the declaration ------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredSetting:
    """One declared setting with its addresses resolved.

    Attributes:
        id: The setting's declared id.
        scope: ``device`` or ``global``.
        setting: The address written: one per device id for device scope.
        readback: The address read back, the written one when none is paired.
        tolerance: The readback's tolerance, ``None`` for an exact match.
        timeout_s: How long the readback has to settle after a write.
    """

    id: str
    scope: str
    setting: Mapping[str, str] | str
    readback: Mapping[str, str] | str
    tolerance: Mapping[str, float | None] | float | None
    timeout_s: float


@dataclass(frozen=True)
class Expected:
    """One held setting address and the value expected of its readback.

    Attributes:
        id: The declared setting's id.
        device: The BPM device it belongs to, ``None`` for a global setting.
        setting: The address written.
        readback: The address the re-check reads.
        value: The value the readback is expected to hold.
        tolerance: The readback's tolerance, ``None`` for an exact match.
        timeout_s: How long the readback has to settle after a write.
    """

    id: str
    device: str | None
    setting: str
    readback: str
    value: Any
    tolerance: float | None
    timeout_s: float

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "device": self.device,
            "setting": self.setting,
            "readback": self.readback,
            "value": self.value,
            "tolerance": self.tolerance,
            "timeout_s": self.timeout_s,
        }


def _address(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _tolerance(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _mapping_or_empty(value: Any) -> Mapping[Any, Any]:
    return value if isinstance(value, Mapping) else {}


def declared_settings(slots: Mapping[str, Any] | None) -> tuple[DeclaredSetting, ...]:
    """The settings a model's ``trajectory_slots`` entry declares, in declared order.

    A setting without an id, or one that resolves to no address, is skipped.
    """
    declared = slots.get("settings") if isinstance(slots, Mapping) else None
    settings: list[DeclaredSetting] = []
    for raw in declared or ():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            continue
        stated_timeout = _tolerance(raw.get("timeout_s"))
        timeout_s = stated_timeout if stated_timeout is not None else DEFAULT_SETTLE_S
        if raw.get("scope") == "device":
            written = _mapping_or_empty(raw.get("setting"))
            paired = _mapping_or_empty(raw.get("readback"))
            tolerances = _mapping_or_empty(raw.get("tolerance"))
            setting = {
                str(device): address
                for device, value in written.items()
                if (address := _address(value)) is not None
            }
            readback = {
                device: _address(paired.get(device)) or address
                for device, address in setting.items()
            }
            tolerance = {device: _tolerance(tolerances.get(device)) for device in setting}
            settings.append(
                DeclaredSetting(raw["id"], "device", setting, readback, tolerance, timeout_s)
            )
            continue
        address = _address(raw.get("setting")) or _address(raw.get("address"))
        if address is None:
            continue
        settings.append(
            DeclaredSetting(
                raw["id"],
                "global",
                address,
                _address(raw.get("readback")) or address,
                _tolerance(raw.get("tolerance")),
                timeout_s,
            )
        )
    return tuple(settings)


def _held(
    declared: Sequence[DeclaredSetting], devices: Collection[str]
) -> list[tuple[DeclaredSetting, str | None, str, str, float | None]]:
    """Every held address: each device setting over *devices*, each global one once."""
    held: list[tuple[DeclaredSetting, str | None, str, str, float | None]] = []
    for setting in declared:
        if setting.scope == "device":
            assert isinstance(setting.setting, Mapping)
            assert isinstance(setting.readback, Mapping)
            assert isinstance(setting.tolerance, Mapping)
            for device, address in setting.setting.items():
                if device in devices:
                    held.append(
                        (
                            setting,
                            device,
                            address,
                            setting.readback[device],
                            setting.tolerance.get(device),
                        )
                    )
        else:
            assert isinstance(setting.setting, str) and isinstance(setting.readback, str)
            tolerance = setting.tolerance if not isinstance(setting.tolerance, Mapping) else None
            held.append((setting, None, setting.setting, setting.readback, tolerance))
    return held


# -- values ----------------------------------------------------------------------------


def _plain(reading: Any) -> Any:
    """A connector reading's value as a plain Python scalar, ``None`` when unread."""
    if reading is None:
        return None
    value = getattr(reading, "value", reading)
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def matches(read: Any, wanted: Any, tolerance: float | None) -> bool:
    """Whether a readback *read* holds *wanted*: within *tolerance* for numbers."""
    if read is None:
        return False
    left, right = _number(read), _number(wanted)
    if left is not None and right is not None:
        if not (math.isfinite(left) and math.isfinite(right)):
            return False
        return abs(left - right) <= (tolerance or 0.0)
    return bool(read == wanted)


def settings_sha(expected: Iterable[Expected]) -> str:
    """The identity of a setup: sha256 of its expected values keyed by written address."""
    document = {item.setting: item.value for item in expected}
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _turns(expected: Iterable[Expected]) -> int:
    """The turns tracked a setup holds; 0 when the model declares none."""
    for item in expected:
        if item.id == TURNS_TRACKED and item.device is None:
            number = _number(item.value)
            if number is not None and math.isfinite(number):
                return int(round(number))
    return 0


# -- the keeper -------------------------------------------------------------------------


class SetupKeeper:
    """The current setup of one target and model: held, persisted and re-checked.

    Attributes:
        declared: The model's declared settings.
        model: The model, ``None`` without one.
        state: The current setup, ``None`` when none was applied.
        expected: Every held readback and the value expected of it.
        notices: What a re-check voided, for the operator.
        on_void: Called once a re-check voided the setup.
    """

    def __init__(
        self,
        declared: Sequence[DeclaredSetting] = (),
        *,
        model: str | None = None,
        directory: Path | None = None,
        devices: Mapping[str, str] | None = None,
        excluded: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        """Hold no setup yet.

        Args:
            declared: The model's declared settings.
            model: The model the setup is for.
            directory: The store's directory; the setup is persisted under it.
            devices: Each BPM's device id, keyed by BPM name.
            excluded: Answers the BPMs the SolveConfig excludes, by name.
        """
        self.declared = tuple(declared)
        self.model = model
        self.state: SetupState | None = None
        self.expected: tuple[Expected, ...] = ()
        self.notices: list[str] = []
        self.on_void: Callable[[], Any] | None = None
        self._devices = dict(devices or {})
        self._excluded = excluded or (lambda: ())
        self._path = (
            None
            if directory is None or model is None
            else Path(directory) / SETUP_DIR / f"{model}.json"
        )

    # -- the included BPMs -------------------------------------------------------

    def included_devices(self, excluded_bpms: Iterable[str] | None = None) -> list[str]:
        """The device ids of the BPMs not excluded, in device-table order."""
        excluded = set(self._excluded() if excluded_bpms is None else excluded_bpms)
        return [device for bpm, device in self._devices.items() if bpm not in excluded]

    def held(
        self, excluded_bpms: Iterable[str] | None = None
    ) -> list[tuple[DeclaredSetting, str | None, str, str, float | None]]:
        return _held(self.declared, set(self.included_devices(excluded_bpms)))

    # -- the re-check, as the chain calls it ---------------------------------------

    def _checked(self) -> list[Expected]:
        """The expected readbacks a re-check reads: the included BPMs' and every global."""
        included = set(self.included_devices())
        return [item for item in self.expected if item.device is None or item.device in included]

    def addresses(self) -> list[str]:
        """The readbacks a re-check reads; none while no current setup holds."""
        if self.state is None or self.state.void:
            return []
        return list(dict.fromkeys(item.readback for item in self._checked()))

    async def check(self, values: Mapping[str, Any]) -> str | None:
        """Re-check the current setup against *values*, a batch read of :meth:`addresses`.

        Returns:
            The setup's ``settings_sha`` while it holds; ``None`` when none
            holds or this re-check voided it.
        """
        state = self.state
        if state is None or state.void:
            return None
        diverged = [
            item.readback
            for item in self._checked()
            if not matches(_plain(values.get(item.readback)), item.value, item.tolerance)
        ]
        if not diverged:
            return state.settings_sha
        await self.void(
            f"The setup was voided: {', '.join(diverged[:5])}"
            f"{' and more' if len(diverged) > 5 else ''} no longer read as set up."
        )
        return None

    async def recheck(self, connector: Any, call_timeout_s: float) -> str | None:
        """Re-check the current setup against the machine in one batch read.

        A read that fails voids the setup: it can no longer be vouched for.
        """
        addresses = self.addresses()
        if not addresses:
            return self.state.settings_sha if self.state and not self.state.void else None
        try:
            values = await guarded_call(
                lambda: connector.read_multiple_channels(addresses, timeout=call_timeout_s),
                record=lambda _: None,
            )
        except Exception as exc:
            await self.void(f"The setup was voided: its readbacks could not be read ({exc}).")
            return None
        return await self.check(values)

    async def void(self, notice: str) -> None:
        """Void the current setup with *notice*, persist it, and tell :attr:`on_void`."""
        if self.state is None or self.state.void:
            return
        self.state = self.state.model_copy(update={"void": True})
        self.notices.append(notice)
        logger.warning("trajectory service: %s", notice)
        try:
            await self.persist()
        except SetupRefused:
            logger.warning("trajectory service: the void setup was not persisted", exc_info=True)
        if self.on_void is not None:
            self.on_void()

    # -- persistence ---------------------------------------------------------------

    def _document(self) -> bytes:
        return json.dumps(
            {
                "model": self.model,
                "setup": self.state.model_dump(mode="json") if self.state is not None else None,
                "expected": [item.as_json() for item in self.expected],
            },
            indent=2,
            sort_keys=True,
            default=str,
        ).encode()

    def _write(self, payload: bytes) -> None:
        assert self._path is not None
        from osprey.services.trajectory.store import FILE_MODE
        from osprey_connectors.safety_file import atomic_replace

        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace(self._path, payload, mode=FILE_MODE, sync="required")

    async def persist(self) -> None:
        """Write the current setup beside the SolveConfig overrides; nothing without a store.

        Raises:
            SetupRefused: ``setup_unsaved`` when the file cannot be written.
        """
        if self._path is None:
            return
        try:
            await asyncio.to_thread(self._write, self._document())
        except OSError as exc:
            raise SetupRefused(SETUP_UNSAVED, f"the setup cannot be written: {exc}") from exc

    def load(self) -> None:
        """Read the persisted setup; a file that cannot be read holds none, with a notice."""
        if self._path is None or not self._path.is_file():
            return
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
            raw = document.get("setup")
            state = SetupState.model_validate(raw) if raw is not None else None
            expected = tuple(
                Expected(
                    id=str(item["id"]),
                    device=item.get("device"),
                    setting=str(item["setting"]),
                    readback=str(item["readback"]),
                    value=item.get("value"),
                    tolerance=_tolerance(item.get("tolerance")),
                    timeout_s=float(item.get("timeout_s") or DEFAULT_SETTLE_S),
                )
                for item in document.get("expected") or ()
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self.notices.append(
                f"The setup of model {self.model!r} could not be read ({exc}); no setup is current."
            )
            return
        self.state = state
        self.expected = expected

    # -- the readbacks -------------------------------------------------------------

    async def readbacks(self, connector: Any, call_timeout_s: float) -> SetupReadbacks:
        """Every declared setting as read back over the included BPMs, in one batch.

        Raises:
            SetupRefused: ``setup_unreadable`` when the batch read failed.
        """
        held = self.held()
        addresses = list(dict.fromkeys(readback for _s, _d, _w, readback, _t in held))
        values: Mapping[str, Any] = {}
        if addresses:
            try:
                values = await guarded_call(
                    lambda: connector.read_multiple_channels(addresses, timeout=call_timeout_s),
                    record=lambda _: None,
                )
            except Exception as exc:
                raise SetupRefused(
                    SETUP_UNREADABLE, f"the settings could not be read: {exc}"
                ) from exc
        read_at = time.time()
        settings: list[SettingReadback] = []
        for declared in self.declared:
            mine = [entry for entry in held if entry[0] is declared]
            if declared.scope == "device":
                settings.append(
                    SettingReadback(
                        id=declared.id,
                        scope="device",
                        values={
                            str(device): _plain(values.get(readback))
                            for _s, device, _w, readback, _t in mine
                        },
                    )
                )
            else:
                value = _plain(values.get(mine[0][3])) if mine else None
                settings.append(SettingReadback(id=declared.id, scope="global", value=value))
        return SetupReadbacks(settings=settings, read_at=read_at, setup=self.state)


# -- the action ---------------------------------------------------------------------------


@dataclass
class SetupOutcome:
    """What a setup action did: its result, whether it was stopped, the step after it."""

    result: SetupResult
    stopped: bool = False
    step: Step | None = None
    excluded_changed: bool = False
    moved: list[Moved] = field(default_factory=list)


def check_body(keeper: SetupKeeper, body: SetupBody) -> None:
    """Refuse a body naming a setting id the model does not declare.

    Raises:
        SetupRefused: ``unknown_setting``.
    """
    known = {setting.id for setting in keeper.declared}
    unknown = sorted(set(body.settings) - known)
    if unknown:
        raise SetupRefused(
            UNKNOWN_SETTING,
            f"the model declares no setting {', '.join(map(repr, unknown))}; "
            f"it declares {', '.join(sorted(known)) or 'none'}",
        )


def setup_action(
    keeper: SetupKeeper,
    sessions: Sessions,
    store: TrajectoryStore,
    body: SetupBody,
    *,
    owner: str | None,
    client_id: str,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
    poll_s: float = SETTLE_POLL_S,
) -> Callable[[Admission, asyncio.Event], Awaitable[SetupOutcome]]:
    """The held setup action for :meth:`~.sessions.Sessions.call`.

    Raises (from the action):
        SetupRefused: ``setup_unreadable``, ``setup_unsaved``.
        SetupFailed: A write failed part-way.
        SolveConfigRefused, StepRefused: As the SolveConfig writer refuses the
            exclusions.
    """
    connector = sessions.layer.startup.connector
    call_timeout_s = sessions.call_timeout_s

    async def read(addresses: list[str]) -> Mapping[str, Any]:
        try:
            return await guarded_call(
                lambda: connector.read_multiple_channels(addresses, timeout=call_timeout_s),
                record=lambda _: None,
            )
        except Exception as exc:
            raise SetupRefused(SETUP_UNREADABLE, f"the settings could not be read: {exc}") from exc

    async def action(_admission: Admission, stop: asyncio.Event) -> SetupOutcome:
        from osprey.services.trajectory.step import TOOL_UNAVAILABLE, StepRefused
        from osprey_connectors.errors import ChannelWriteBlockedError

        excluded_changed = False
        excluded: Iterable[str] | None = None
        if body.excluded_bpms is not None:
            solve = sessions.solve
            if solve is None:
                raise StepRefused(TOOL_UNAVAILABLE, "the service has no SolveConfig: no tool")
            update = await solve.update(
                SolveConfigPartial(excluded_bpms=list(body.excluded_bpms)), SETUP
            )
            await sessions.after_solve(update, store, client_id)
            excluded_changed = "excluded_bpms" in update.changed
            excluded = solve.values["excluded_bpms"]
        held = keeper.held(excluded)
        addresses = list(
            dict.fromkeys(
                address for _s, _d, written, readback, _t in held for address in (written, readback)
            )
        )
        before = await read(addresses) if addresses else {}

        expected: list[Expected] = []
        writes: list[Expected] = []
        for setting, device, written, readback, tolerance in held:
            asked = setting.id in body.settings
            value = body.settings[setting.id] if asked else _plain(before.get(readback))
            item = Expected(
                setting.id, device, written, readback, value, tolerance, setting.timeout_s
            )
            expected.append(item)
            if asked and not matches(_plain(before.get(readback)), value, tolerance):
                writes.append(item)
        unread = [item.setting for item in writes if _plain(before.get(item.setting)) is None]
        if unread:
            raise SetupRefused(
                SETUP_UNREADABLE,
                f"nothing was written: {', '.join(unread)} could not be read, "
                "so the write could not be put back",
            )

        moved: list[Moved] = []

        def outcome(stopped: bool, unsettled: list[str], setup: SetupState | None) -> SetupOutcome:
            return SetupOutcome(
                result=SetupResult(moved=list(moved), unsettled=unsettled, setup=setup),
                stopped=stopped,
                step=sessions.step.step,
                excluded_changed=excluded_changed,
                moved=list(moved),
            )

        async def did_not_take(reason: str) -> None:
            if moved:
                sessions.step.void()
                await keeper.void(f"The setup was voided: {reason}.")

        for index, item in enumerate(writes):
            if index and stop.is_set():
                await did_not_take(f"a setup was stopped after {index} of {len(writes)} writes")
                return outcome(True, [], None)
            prior = _plain(before.get(item.setting))

            def record(result: Any, item: Expected = item, prior: Any = prior) -> None:
                if isinstance(result, ChannelWriteBlockedError):
                    return
                if isinstance(result, BaseException):
                    word = getattr(result, "outcome", None) or "failed"
                else:
                    word = getattr(result, "outcome", "written")
                moved.append(
                    Moved(
                        name=item.setting,
                        before=prior,
                        after=item.value,
                        outcome=str(getattr(word, "value", word)),
                    )
                )

            try:
                await guarded_call(
                    functools.partial(connector.write_channel_checked, item.setting, item.value),
                    record=record,
                )
            except Exception as exc:
                await did_not_take(f"a setup write failed at {item.setting}")
                raise SetupFailed(f"{item.setting}: {exc}", list(moved)) from exc

        unsettled = await _settle(writes, read, stop, sleep=sleep, clock=clock, poll_s=poll_s)
        if stop.is_set() and unsettled:
            await did_not_take("a setup was stopped before its settings settled")
            return outcome(True, unsettled, None)
        if unsettled:
            await did_not_take(f"{', '.join(unsettled)} did not settle")
            return outcome(False, unsettled, None)

        if moved:
            sessions.step.void()
        keeper.state = SetupState(
            applied_at=wall(),
            by=owner,
            settings_sha=settings_sha(expected),
            turns_tracked=_turns(expected),
        )
        keeper.expected = tuple(expected)
        await keeper.persist()
        return outcome(False, [], keeper.state)

    return action


async def _settle(
    writes: Sequence[Expected],
    read: Callable[[list[str]], Awaitable[Mapping[str, Any]]],
    stop: asyncio.Event,
    *,
    sleep: Callable[[float], Awaitable[Any]],
    clock: Callable[[], float],
    poll_s: float,
) -> list[str]:
    """Poll the written readbacks until each holds its value or its timeout passed.

    Returns:
        The written addresses whose readback missed, in write order.
    """
    start = clock()
    pending = list(writes)
    while pending:
        values = await read(list(dict.fromkeys(item.readback for item in pending)))
        pending = [
            item
            for item in pending
            if not matches(_plain(values.get(item.readback)), item.value, item.tolerance)
        ]
        now = clock()
        late = [item for item in pending if now - start >= item.timeout_s]
        if not pending or len(late) == len(pending) or stop.is_set():
            break
        await sleep(poll_s)
    return [item.setting for item in pending]
