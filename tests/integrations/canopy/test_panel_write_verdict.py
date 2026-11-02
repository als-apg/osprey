"""A panel write is judged per write and per tick on the one execution verdict.

Nothing about writes is rendered into DATA's environment. DATA reads the
execution verdict (:func:`osprey_connectors.posture_store.execution_verdict`)
at every write and on every heartbeat tick, with the hold of every session the
terminal runs: the sidecar names them in ``sessions.json`` in DATA's working
directory and names the hold store in DATA's environment. No session, no hold.

* the claim: :func:`~osprey.integrations.canopy.backend.writes_claim` for each
  row of OSPREY's claim table (``writes_claims`` in
  ``fixtures/backend-statement.json``), and the real backend re-reading it when
  the switch, the position or a hold changes, with no exit;
* the refusal per PUT: the toast a refused write answers is the claim's sentence;
* the sessions file: written 0600 at spawn, rewritten when the set changes, kept
  when the reader raises; the hold store only when one is given;
* the budget, FRONT's write names, the keys DATA accepts and the panel's stop grace.

The sidecar is a real :class:`CanopySidecar` whose two children are a script
that sleeps, so every case reads what a child would be started with.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from canopy.core import security
from tests._renders import tier_render
from tests.integrations.canopy import test_backend_write as backend_write

from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.deployment.web_terminals.roster import config_stop_grace, config_stop_period
from osprey.integrations.canopy import (
    CONDITIONAL_DATA_CANOPY_VARS,
    CONDITIONAL_FRONT_CANOPY_VARS,
    EXPECTED_DATA_CANOPY_VARS,
    EXPECTED_FRONT_CANOPY_VARS,
    SESSIONS_FILENAME,
)
from osprey.integrations.canopy import backend as backend_module
from osprey.integrations.canopy import sidecar as sidecar_module
from osprey.integrations.canopy.backend import PVWriteRefused, read_session_keys, writes_claim
from osprey.integrations.canopy.sidecar import CanopySidecar, write_budget_text
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID, SIDECAR_PANELS
from osprey.runtime import budget as budget_module
from osprey.runtime.budget import write_budget
from osprey.runtime.session import EXIT_BINDING_MOVED
from osprey.runtime.session_outcomes import refusal_message
from osprey_connectors import control_context, posture_store, session_hold
from osprey_connectors.identity import AUDIT_IDENTITY_ENV
from osprey_connectors.posture_store import WriteVerdict
from osprey_connectors.refusals import refusal_phrase

# The backend's fixtures and helpers: a real session over a recording machine,
# bound to ``va@3`` with its switch allowed, the exit recorded.
armed = backend_write.armed
audit_dir = backend_write.audit_dir
connector = backend_write.connector
exits = backend_write.exits
machine = backend_write.machine
state_dir = backend_write.state_dir
make_connector = backend_write.make_connector
write_record = backend_write.write_record
BINDING = backend_write.BINDING
CHANNEL = backend_write.CHANNEL

#: The three names FRONT writes with, and the one DATA's budget is rendered in.
FRONT_WRITE_NAMES = {
    "CANOPY_PUT_RELAY",
    "CANOPY_PUT_RELAY_TIMEOUT_S",
    "CANOPY_WRITE_API_KEY_FILE",
}
DATA_WRITE_NAMES = {"OSPREY_CANOPY_WRITE_BUDGET"}

SLEEPER = "import time\nwhile True:\n    time.sleep(1)\n"

#: OSPREY's writes claim table.
CLAIMS = json.loads(
    (Path(__file__).parent / "fixtures" / "backend-statement.json").read_text(encoding="utf-8")
)["writes_claims"]
CLAIM_ROWS = {row["id"]: row for row in CLAIMS["rows"]}

#: The detail a hold raised by ``watchdog:limits`` adds to its verdict.
HOLD_DETAIL = "Raised by: watchdog:limits."


def _writes(value: Any) -> dict[str, Any]:
    return {"control_system": {"writes": value}}


class Deployment:
    """A config, a record directory and a payload in ``tmp_path``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_path = root / "config.yml"
        self.state_dir = root / "control_target"
        self.state_dir.mkdir()
        self.payload = root / "canopy_payload"
        (self.payload / "viewer").mkdir(parents=True)
        (self.payload / "viewer" / "index.html").write_text("<html></html>", encoding="utf-8")
        (self.payload / "viewer" / "ir-map.json").write_text("{}", encoding="utf-8")
        self.sleeper = root / "sleeper.py"
        self.sleeper.write_text(SLEEPER, encoding="utf-8")
        self.write_config({})
        self.sidecars: list[CanopySidecar] = []

    def write_config(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def write_text(self, text: str) -> None:
        self.config_path.write_text(text, encoding="utf-8")

    def sidecar(self, *, hold_root: Path | None = None, live_sessions=None) -> CanopySidecar:
        sleeper = [sys.executable, str(self.sleeper)]

        class _Sleeping(CanopySidecar):
            def _front_command(self, fd):  # noqa: ARG002 - the stub ignores the socket
                return sleeper

            def _data_command(self, fd):  # noqa: ARG002
                return sleeper

        built = _Sleeping(
            self.root / "agent-data" / "alice",
            "",
            None,
            payload_dir=self.payload,
            hold_root=hold_root,
            live_sessions=live_sessions,
        )
        self.sidecars.append(built)
        built.spawn()
        return built


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("CANOPY_", "OSPREY_CANOPY_", "OSPREY_CONTROL_TARGET")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        session_hold.HOLD_DIR_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    built = Deployment(tmp_path)
    monkeypatch.setenv("OSPREY_CONFIG", str(built.config_path))
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(built.state_dir))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.chdir(tmp_path)
    control_context.invalidate_cache()
    yield built
    for sidecar in built.sidecars:
        sidecar.stop()
    control_context.invalidate_cache()


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.01)


# -- the claim, per row --------------------------------------------------------------


def _verdict_for(row: dict) -> WriteVerdict:
    """The execution verdict *row* names, for the simulator; a hold names its source."""
    verdict = row["verdict"]
    return WriteVerdict(
        allowed=verdict["allowed"],
        reason=verdict["reason"],
        target="va",
        kind="simulator",
        position=verdict["position"],
        detail=HOLD_DETAIL if verdict["reason"] == "session_held" else None,
    )


def _expected(row: dict, verdict: WriteVerdict) -> dict:
    claim = dict(row["claim"])
    if claim["reason"] == "<refusal_message>":
        claim["reason"] = refusal_message(verdict)
    elif claim["reason"] == "<phrase:switching>":
        claim["reason"] = refusal_phrase("switching", verdict.kind)
    return claim


class TestTheClaimTable:
    @pytest.mark.parametrize("row_id", list(CLAIM_ROWS))
    def test_each_row_is_the_claim_its_state_makes(self, row_id):
        row = CLAIM_ROWS[row_id]
        verdict = _verdict_for(row)
        session = SimpleNamespace(
            target_moving=row["target_moving"],
            binding=SimpleNamespace(state=row["binding_state"]),
            verdict=verdict,
        )
        assert writes_claim(session) == _expected(row, verdict)

    def test_a_hold_names_its_source(self):
        row = CLAIM_ROWS["session_held"]
        claim = _expected(row, _verdict_for(row))
        assert claim["reason"] == f"{refusal_phrase('session_held', 'simulator')} {HOLD_DETAIL}"

    def test_the_record_moving_outranks_every_verdict(self):
        verdict = _verdict_for(CLAIM_ROWS["session_held"])
        session = SimpleNamespace(
            target_moving=True, binding=SimpleNamespace(state="bound"), verdict=verdict
        )
        assert writes_claim(session)["cause"] == "switching"

    def test_the_binding_state_outranks_the_verdict(self):
        verdict = _verdict_for(CLAIM_ROWS["session_held"])
        session = SimpleNamespace(
            target_moving=False, binding=SimpleNamespace(state="fallback"), verdict=verdict
        )
        assert writes_claim(session)["cause"] == "control_context_unavailable"

    @pytest.mark.parametrize("reason", ["owner_unknown", "owner_foreign", "launch_pinned"])
    def test_any_other_refusal_is_locked_with_its_reason(self, reason):
        verdict = WriteVerdict(allowed=False, reason=reason, target="va", kind="simulator")
        session = SimpleNamespace(
            target_moving=False, binding=SimpleNamespace(state="bound"), verdict=verdict
        )
        assert writes_claim(session) == {
            "state": "locked",
            "reason": refusal_message(verdict),
            "cause": reason,
        }


# -- the claim and the refusal, per tick and per PUT, on the real backend ---------------


def _hold_store(tmp_path: Path, monkeypatch, *keys: str) -> Path:
    root = tmp_path / "holds"
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    for key in keys:
        session_hold.ensure_session(key, root=root)
    (tmp_path / SESSIONS_FILENAME).write_text(json.dumps(sorted(keys)), encoding="utf-8")
    return root


def _raise(root: Path, key: str) -> None:
    assert session_hold.open_flag(
        key,
        flag_id="flag-1",
        source="watchdog:limits",
        reason="a limit was refused",
        prompt_ref="turn-7",
        prompt_excerpt="set the current to 900",
        root=root,
    )


async def _claim_becomes(connector_, cause: str | None, timeout: float = 5.0) -> dict:
    """Wait until the claim's cause is *cause*: the move watcher re-read the verdict."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while connector_.writes["cause"] != cause:
        assert loop.time() < end, f"the claim never became {cause!r}: {connector_.writes}"
        await asyncio.sleep(0.01)
    return connector_.writes


class TestTheClaimPerTick:
    """A switch flip, a position change or a hold changes the claim; nothing exits.

    The simulator has no gateways to choose between. A switch move that asks a
    connector for its other gateway is a move (``ControlSession._gateway_lags``,
    proven on the real EPICS connector in ``tests/runtime/test_control_session.py``),
    and the last test here shows the backend leaves on it.
    """

    async def test_switch_position_and_back(self, state_dir, connector, exits):
        assert connector.writes == {"state": "on", "reason": None, "cause": None}
        write_record(state_dir, switch={})
        off = await _claim_becomes(connector, "switch_off")
        assert off["state"] == "off"
        write_record(state_dir, switch={"va": "asked"})
        asked = await _claim_becomes(connector, "asked")
        assert asked == {"state": "asked", "reason": None, "cause": "asked"}
        write_record(state_dir, switch={"va": "allowed"})
        await _claim_becomes(connector, None)
        assert exits == []
        assert not connector._watcher.done()
        assert connector._session.binding.id == BINDING

    @pytest.mark.usefixtures("machine")
    async def test_a_hold_raised_and_answered(self, tmp_path, monkeypatch, exits):
        root = _hold_store(tmp_path, monkeypatch, "alice-1")
        built = make_connector()
        await built.start()
        try:
            assert built.writes["state"] == "on"
            _raise(root, "alice-1")
            held = await _claim_becomes(built, "session_held")
            assert held["state"] == "held"
            assert held["reason"].endswith(HOLD_DETAIL)
            session_hold.answer_flags(
                "alice-1", ["flag-1"], answer="continue", operator="alice", root=root
            )
            await _claim_becomes(built, None)
            assert exits == []
            assert not built._watcher.done()
        finally:
            await built.close()

    @pytest.mark.usefixtures("machine")
    async def test_a_session_named_or_dropped(self, tmp_path, monkeypatch, exits):
        root = _hold_store(tmp_path, monkeypatch, "alice-1", "alice-2")
        _raise(root, "alice-2")
        (tmp_path / SESSIONS_FILENAME).write_text('["alice-1"]', encoding="utf-8")
        built = make_connector()
        await built.start()
        try:
            assert built.writes["state"] == "on"
            (tmp_path / SESSIONS_FILENAME).write_text('["alice-1", "alice-2"]', encoding="utf-8")
            await _claim_becomes(built, "session_held")
            (tmp_path / SESSIONS_FILENAME).unlink()
            await _claim_becomes(built, None)
            assert exits == []
        finally:
            await built.close()

    async def test_a_switch_move_asking_for_the_other_gateway_exits_75(
        self, state_dir, connector, exits, monkeypatch
    ):
        """DATA follows the gateway: off -> asked on a two-gateway block replaces the process."""
        from osprey.runtime.session import ControlSession

        lags = {"now": False}
        monkeypatch.setattr(ControlSession, "_gateway_lags", lambda self: lags["now"])
        write_record(state_dir, switch={})
        await _claim_becomes(connector, "switch_off")
        assert exits == []
        lags["now"] = True
        write_record(state_dir, switch={"va": "asked"})
        loop = asyncio.get_running_loop()
        end = loop.time() + 10.0
        while not exits:
            assert loop.time() < end, "the backend never left on the gateway move"
            await asyncio.sleep(0.01)
        assert exits == [EXIT_BINDING_MOVED]


class TestTheRefusalPerPut:
    """A refused write answers the sentence the claim states: one sentence, two places."""

    async def test_off(self, state_dir, connector, machine):
        write_record(state_dir, switch={})
        claim = await _claim_becomes(connector, "switch_off")
        with pytest.raises(PVWriteRefused) as caught:
            await connector.put(CHANNEL, 1.0, binding=BINDING, confirmed=True)
        assert caught.value.detail == claim["reason"]
        assert machine.writes == []

    async def test_held(self, tmp_path, monkeypatch, machine):
        root = _hold_store(tmp_path, monkeypatch, "alice-1")
        _raise(root, "alice-1")
        built = make_connector()
        await built.start()
        try:
            claim = built.writes
            assert claim["state"] == "held"
            with pytest.raises(PVWriteRefused) as caught:
                await built.put(CHANNEL, 1.0, binding=BINDING, confirmed=True)
        finally:
            await built.close()
        assert caught.value.detail == claim["reason"]
        assert machine.writes == []

    async def test_record_unavailable(self, tmp_path, machine):
        (tmp_path / SESSIONS_FILENAME).write_text("{", encoding="utf-8")
        built = make_connector()
        await built.start()
        try:
            claim = built.writes
            assert (claim["state"], claim["cause"]) == ("locked", "record_unavailable")
            with pytest.raises(PVWriteRefused) as caught:
                await built.put(CHANNEL, 1.0, binding=BINDING, confirmed=True)
        finally:
            await built.close()
        assert caught.value.detail == claim["reason"]
        assert machine.writes == []

    async def test_asked_refuses_only_the_unconfirmed(self, state_dir, connector, machine):
        write_record(state_dir, switch={"va": "asked"})
        await _claim_becomes(connector, "asked")
        with pytest.raises(PVWriteRefused) as caught:
            await connector.put(CHANNEL, 1.0, binding=BINDING)
        assert caught.value.detail == "This write needs your confirmation; nothing written."
        await connector.put(CHANNEL, 2.0, binding=BINDING, confirmed=True)
        assert machine.writes == [(CHANNEL, 2.0)]


# -- what the sidecar renders and writes for writes --------------------------------------


class TestNothingAboutWritesIsRendered:
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="absent"),
            pytest.param(_writes("allowed"), id="allowed"),
            pytest.param(_writes("never"), id="never"),
            pytest.param({"control_system": None}, id="null-section"),
        ],
    )
    def test_no_switch_reaches_data_and_no_confirm_flag_reaches_front(self, deployment, config):
        deployment.write_config(config)
        sidecar = deployment.sidecar()
        data, front = sidecar.data_env(), sidecar.front_env()
        assert not [name for name in data if name.startswith("OSPREY_CANOPY_WRITES")]
        assert "CANOPY_CONFIRM_WRITES" not in front

    def test_the_old_switch_is_gone_from_both_modules(self):
        for name in ("WRITES_ENV", "WRITES_OFF_ENV", "_writes_off_reason"):
            assert not hasattr(backend_module, name)
        for name in ("writes_off_reason", "WRITES_ENV", "WRITES_OFF_ENV"):
            assert not hasattr(sidecar_module, name)
        assert sidecar_module.WRITE_BUDGET_ENV == backend_module.WRITE_BUDGET_ENV


class TestTheHoldStore:
    def test_data_is_told_the_hold_store_when_there_is_one(self, deployment, tmp_path):
        root = tmp_path / "holds"
        env = deployment.sidecar(hold_root=root).data_env()
        assert env[session_hold.HOLD_DIR_ENV_VAR] == str(root)

    def test_without_one_data_is_told_none(self, deployment):
        """No store, nothing held."""
        assert session_hold.HOLD_DIR_ENV_VAR not in deployment.sidecar().data_env()

    def test_the_store_is_named_at_every_data_start(self, deployment, tmp_path):
        sidecar = deployment.sidecar(hold_root=tmp_path / "holds")
        deployment.write_config(_writes("never"))
        assert sidecar.data_env()[session_hold.HOLD_DIR_ENV_VAR] == str(tmp_path / "holds")


class TestTheSessionsFile:
    @pytest.fixture(autouse=True)
    def fast_poll(self, monkeypatch):
        monkeypatch.setattr(sidecar_module, "SESSIONS_POLL_S", 0.02)

    @staticmethod
    def _path(sidecar: CanopySidecar) -> Path:
        return sidecar.key_dir / SESSIONS_FILENAME

    def test_spawn_writes_the_live_sessions_sorted_0600(self, deployment):
        sidecar = deployment.sidecar(live_sessions=lambda: ["bob-2", "alice-1", "bob-2"])
        path = self._path(sidecar)
        assert json.loads(path.read_text(encoding="utf-8")) == ["alice-1", "bob-2"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert read_session_keys(path) == ["alice-1", "bob-2"]

    def test_no_reader_names_no_session(self, deployment):
        sidecar = deployment.sidecar()
        assert json.loads(self._path(sidecar).read_text(encoding="utf-8")) == []

    def test_a_change_is_rewritten_whole(self, deployment):
        live = ["alice-1"]
        sidecar = deployment.sidecar(live_sessions=lambda: list(live))
        path = self._path(sidecar)
        live.append("alice-2")
        _wait_for(lambda: json.loads(path.read_text(encoding="utf-8")) == ["alice-1", "alice-2"])
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        live.clear()
        _wait_for(lambda: json.loads(path.read_text(encoding="utf-8")) == [])
        # Replaced whole: no staging file is left beside it.
        assert sorted(p.name for p in sidecar.key_dir.iterdir() if "sessions" in p.name) == [
            SESSIONS_FILENAME
        ]

    def test_an_unchanged_set_is_not_rewritten(self, deployment):
        sidecar = deployment.sidecar(live_sessions=lambda: ["alice-1"])
        path = self._path(sidecar)
        before = path.stat().st_ino, path.stat().st_mtime_ns
        time.sleep(10 * sidecar_module.SESSIONS_POLL_S)
        assert (path.stat().st_ino, path.stat().st_mtime_ns) == before

    def test_a_reader_that_raises_keeps_the_file(self, deployment):
        state = {"keys": ["alice-1"], "broken": False}
        calls = []

        def live():
            calls.append(1)
            if state["broken"]:
                raise RuntimeError("the registry changed under the read")
            return list(state["keys"])

        sidecar = deployment.sidecar(live_sessions=live)
        path = self._path(sidecar)
        state["broken"] = True
        state["keys"] = []
        seen = len(calls)
        _wait_for(lambda: len(calls) >= seen + 3)
        assert json.loads(path.read_text(encoding="utf-8")) == ["alice-1"]
        state["broken"] = False
        _wait_for(lambda: json.loads(path.read_text(encoding="utf-8")) == [])

    def test_stop_ends_the_watch(self, deployment):
        calls = []
        sidecar = deployment.sidecar(live_sessions=lambda: calls.append(1) or [])
        thread = sidecar._sessions_thread
        assert thread is not None and thread.is_alive()
        sidecar.stop()
        assert not thread.is_alive()
        after = len(calls)
        time.sleep(5 * sidecar_module.SESSIONS_POLL_S)
        assert len(calls) == after


# -- the write budget --------------------------------------------------------------


class TestWriteBudget:
    @pytest.mark.parametrize(
        "connectors",
        [
            pytest.param({}, id="defaults"),
            pytest.param({"epics": {"timeout_s": 60}}, id="clamped"),
            pytest.param({"epics": {"step_read_timeout_s": 90}}, id="step-read-capped"),
            pytest.param({"epics": {"timeout_s": 3, "step_read_timeout_s": 4}}, id="below"),
        ],
    )
    def test_data_is_told_the_spawn_budget_and_parses_it_back(self, deployment, connectors):
        config = {"control_system": {"connector": connectors}}
        deployment.write_config(config)
        expected = write_budget(config)
        env = deployment.sidecar().data_env()
        text = env["OSPREY_CANOPY_WRITE_BUDGET"]
        assert text == write_budget_text(expected) == f"{expected.T},{expected.S},{expected.D}"
        assert backend_module._parse_budget(text) == (expected.T, expected.S, expected.D)
        # DATA's own budget from the same config, lowered by the rendered text:
        # equal, as both read one config.
        assert backend_module.process_write_budget(expected, env) == expected

    def test_a_smaller_rendered_budget_lowers_datas_componentwise(self, deployment):
        """DATA uses the componentwise minimum of its own budget and the rendered one."""
        deployment.write_config({"control_system": {"connector": {"epics": {"timeout_s": 3}}}})
        env = deployment.sidecar().data_env()
        larger = write_budget({"control_system": {"connector": {"epics": {"timeout_s": 8}}}})
        lowered = backend_module.process_write_budget(larger, env)
        assert (lowered.T, lowered.S, lowered.D) == (5.0, 2.0, 12.0)
        assert lowered.grace_s == larger.grace_s

    def test_the_budget_stays_the_spawns_when_the_config_changes(self, deployment):
        deployment.write_config({"control_system": {"connector": {"epics": {"timeout_s": 60}}}})
        sidecar = deployment.sidecar()
        before = sidecar.data_env()["OSPREY_CANOPY_WRITE_BUDGET"]
        deployment.write_config({})
        assert sidecar.data_env()["OSPREY_CANOPY_WRITE_BUDGET"] == before == "11.0,2.0,24.0"


# -- FRONT's write names -----------------------------------------------------------


class TestFrontWriteEnv:
    def test_the_relay_is_always_on_with_the_write_key_file(self, deployment):
        sidecar = deployment.sidecar()
        front = sidecar.front_env()
        assert front["CANOPY_PUT_RELAY"] == "1"
        key_file = Path(front["CANOPY_WRITE_API_KEY_FILE"])
        assert key_file == sidecar.key_dir / "front.write"
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
        assert front["CANOPY_PUT_RELAY_TIMEOUT_S"] == "15.0"

    def test_the_relay_timeout_is_the_budgets(self, deployment):
        config = {"control_system": {"connector": {"epics": {"timeout_s": 60}}}}
        deployment.write_config(config)
        front = deployment.sidecar().front_env()
        assert front["CANOPY_PUT_RELAY_TIMEOUT_S"] == str(write_budget(config).relay_timeout_s)
        assert float(front["CANOPY_PUT_RELAY_TIMEOUT_S"]) == budget_module.MAX_RELAY_TIMEOUT_S

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"approval": {"enabled": False}}, id="approval-off"),
            pytest.param({"approval": {"enabled": True, "default_policy": "always"}}, id="on"),
        ],
    )
    def test_front_is_not_told_whether_to_ask(self, deployment, config):
        """Whether the viewer asks is DATA's ``asked`` claim, per tick; FRONT has no flag."""
        deployment.write_config(config)
        assert "CANOPY_CONFIRM_WRITES" not in deployment.sidecar().front_env()


# -- the exported sets -------------------------------------------------------------


class TestExportedSets:
    def test_front_has_three_write_names_data_one(self):
        assert FRONT_WRITE_NAMES <= EXPECTED_FRONT_CANOPY_VARS
        assert not FRONT_WRITE_NAMES & CONDITIONAL_FRONT_CANOPY_VARS
        assert DATA_WRITE_NAMES <= EXPECTED_DATA_CANOPY_VARS
        assert not DATA_WRITE_NAMES & CONDITIONAL_DATA_CANOPY_VARS

    def test_no_writes_switch_or_confirm_flag_is_expected(self):
        everything = EXPECTED_DATA_CANOPY_VARS | EXPECTED_FRONT_CANOPY_VARS
        assert not {"OSPREY_CANOPY_WRITES", "OSPREY_CANOPY_WRITES_OFF"} & everything
        assert "CANOPY_CONFIRM_WRITES" not in everything
        assert CONDITIONAL_DATA_CANOPY_VARS == {"CANOPY_THEME_FILE"}

    @pytest.mark.parametrize("config", [{}, {"control_system": {"writes": "allowed"}}])
    def test_data_renders_every_unconditional_name_at_any_tier(self, deployment, config):
        deployment.write_config(config)
        sidecar = deployment.sidecar()
        data = {name for name in sidecar.data_env() if name in EXPECTED_DATA_CANOPY_VARS}
        front = {name for name in sidecar.front_env() if name in EXPECTED_FRONT_CANOPY_VARS}
        assert data == EXPECTED_DATA_CANOPY_VARS - CONDITIONAL_DATA_CANOPY_VARS
        assert front == EXPECTED_FRONT_CANOPY_VARS - CONDITIONAL_FRONT_CANOPY_VARS

    def test_the_home_screen_is_conditional_and_no_write_name(self):
        assert "CANOPY_HOME_SCREEN" in CONDITIONAL_FRONT_CANOPY_VARS
        assert "CANOPY_HOME_SCREEN" in EXPECTED_FRONT_CANOPY_VARS
        assert "CANOPY_HOME_SCREEN" not in FRONT_WRITE_NAMES


# -- the keys DATA accepts ---------------------------------------------------------


class TestDataKeys:
    def test_canopy_loads_both_keys_with_their_scopes(self, deployment):
        key_dir = deployment.sidecar().key_dir
        read_key = (key_dir / "front.read").read_text().strip()
        write_key = (key_dir / "front.write").read_text().strip()
        registry = security.load_keys_file(key_dir / "data.keys")
        lines = (key_dir / "data.keys").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        read = asyncio.run(registry.get(read_key))
        assert read is not None and read.read and not read.write and not read.admin
        assert read.modules == frozenset(sidecar_module.FRONT_MODULES)

        write = asyncio.run(registry.get(write_key))
        assert write is not None and write.read and write.write and not write.admin
        assert write.modules == frozenset({"hare"})
        assert write.allows_module("hare")
        for module in ("fox", "mole", "vole", "converter"):
            assert not write.allows_module(module)


# -- the panel's registry entry and the terminal's stop grace -----------------------


def _shows(*panel_ids: str) -> dict[str, Any]:
    return {"web": {"panels": {panel_id: {"enabled": True} for panel_id in panel_ids}}}


def _render_project(repo: Path, name: str, config: dict[str, Any]) -> str:
    project = repo / "build" / name
    project.mkdir(parents=True)
    (project / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, **config}), encoding="utf-8"
    )
    return f"build/{name}"


def _deploy_config(repo: Path, readwrite: dict[str, Any], readonly: dict[str, Any]) -> dict:
    """A two-tier deployment: alice is ``readwrite``, bob ``readonly``."""
    return {
        "facility": {"name": "Demo Light Source"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "renders": {
                    "readwrite": tier_render(
                        "readwrite",
                        "dls-readwrite",
                        project_path=_render_project(repo, "dls-readwrite", readwrite),
                    ),
                    "readonly": tier_render(
                        "readonly",
                        "dls-readonly",
                        project_path=_render_project(repo, "dls-readonly", readonly),
                    ),
                },
                "default_tier": "readonly",
                "users": [
                    {"name": "alice", "index": 0, "tier": "readwrite"},
                    {"name": "bob", "index": 1, "tier": "readonly"},
                ],
            }
        },
    }


class TestPanelRegistry:
    def test_the_entry_pages_are_canopys_application_and_never_a_draft(self):
        pages = SIDECAR_PANELS[CANOPY_PANEL_ID].entry_pages
        assert pages == ("/", "/s/*", "/embed/*", "/trace/*")
        assert not any(page.startswith("/drafts") for page in pages)

    def test_the_stop_grace_is_front_then_data_at_the_ceiling_then_the_relay_margin(self):
        grace = SIDECAR_PANELS[CANOPY_PANEL_ID].stop_grace_s
        assert grace == sidecar_module.PANEL_STOP_GRACE_S == 34
        assert grace == (
            sidecar_module.FRONT_GRACE_S
            + budget_module.MAX_WRITE_BOUND_S
            + budget_module.GRACE_MARGIN_S
            + budget_module.RELAY_MARGIN_S
        )

    def test_the_grace_covers_the_largest_data_grace_any_config_gives(self):
        worst = write_budget({"control_system": {"connector": {"epics": {"timeout_s": 600}}}})
        grace = SIDECAR_PANELS[CANOPY_PANEL_ID].stop_grace_s
        assert grace >= sidecar_module.FRONT_GRACE_S + worst.grace_s

    def test_a_project_showing_canopy_needs_its_grace(self):
        assert config_stop_grace(_shows(CANOPY_PANEL_ID)) == 34
        assert config_stop_grace(_shows("jupyter")) == SIDECAR_PANELS["jupyter"].stop_grace_s < 34

    def test_the_terminal_showing_canopy_renders_its_stop_period(self, tmp_path):
        """CANOPY's 34 s grace, then the terminal's own stop on top of it."""
        config = _deploy_config(tmp_path, _shows(CANOPY_PANEL_ID), _shows("jupyter"))
        inputs = resolve_render_inputs(config, tmp_path)
        artifacts = render_web_terminals(config, **inputs)
        compose = artifacts["docker-compose.web.yml"]
        services = yaml.safe_load(compose)["services"]
        period = config_stop_period(_shows(CANOPY_PANEL_ID))

        assert period > 34
        assert inputs["stop_grace_renders"] == {"readwrite": period}
        assert f"    stop_grace_period: {period}s\n" in compose
        assert services["web-alice"]["stop_grace_period"] == f"{period}s"
        assert config_stop_period(_shows("jupyter")) == 0
        assert "stop_grace_period" not in services["web-bob"]
