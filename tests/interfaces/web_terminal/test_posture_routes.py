"""Tests for ``POST /api/terminal/posture`` — the person's write switch (C-POST).

The switch is the **login's**: one ``on``/``off`` per machine in the ``switch``
field of that login's control-context record, written only by
:func:`~osprey_connectors.control_context.set_switch` and read at write time by
every write path through the write verdict.

What the tests below pin:

* **On is locked only by what no chip can lift.** A turn-on is refused 403 for
  a ``never`` tier (an app render is one) or a read-only run; a stop is never
  refused for those, for a follower terminal, or for a run in flight.
* **The epoch is a compare-and-swap.** A chip that read no epoch sends null;
  an epoch-less record is stamped and answers 409 ``stamped``, the retry lands;
  a turn-on that raced a stop stays off.
* **No usable record is a 503, never a 409 loop or a 200** — and a stop that
  did not take says writes are still on, never that they are refused.
* **One envelope, one audit line.** Every non-200 is exactly
  ``{detail: {reason, phrase?, epoch?, stamped?, taken?}}``, and every outcome
  files one ledger line.
* **Runs keep what they started with; off reaches them.** A turn-on lands
  while a sandbox run is going and that run's next write stays pinned; the
  next run writes; a stop refuses the running sandbox's next write.

The GET roster's fields are pinned by its own suite, not here.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from osprey.audit import posture as audit_posture
from osprey.interfaces.web_terminal import control_context_owner
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_eligibility, target_state
from osprey.mcp_server.python_executor import executor as host_executor
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.identity import acting_identity
from osprey_connectors.refusals import REASONS, refusal_phrase
from osprey_connectors_tests._harness import (
    owner,
    pin_identity,
    state_dir_under,
    write_control_context,
    write_payload,
    write_server_report,
)
from osprey_connectors_tests._write_fakes import RecordingConnector, config_reader
from tests.interfaces.web_terminal._control_context import only_alive, read_record
from tests.interfaces.web_terminal._started_app import terminal_client

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
CHAT_A = "cccccccc-1111-2222-3333-444444444444"

STANDIN_PORT = 5074

#: The terminal that owns the context in the follower cases. ``os.getppid()``
#: is a real, live process, so the record names a holder that is not a corpse.
OTHER_TERMINAL_PID = os.getppid()
OTHER_TERMINAL_PORT = 8090

#: A controls server's pid for the target-switch case, reported alive by patch.
SERVER_PID = 5150

#: C-WORDS' sentence for a stop the record did not take.
STOP_SENTENCE = (
    "The stop did not take effect — writes are still on. Retry; if it fails again, "
    "ask the deployment admin."
)

#: Every key a C-POST refusal may carry.
ENVELOPE_KEYS = {"reason", "phrase", "epoch", "stamped", "taken"}


# -- render shapes ----------------------------------------------------------


def _gateways(port):
    row = {"address": "localhost", "port": port, "use_name_server": True}
    return {"read_only": dict(row), "write_access": dict(row)}


def control_system_section(*, writes=None, with_live=True):
    """The three-target render every test here starts from.

    ``epics`` is the facility's own machine, ``live_standin`` the co-deployed
    stand-in, ``virtual_accelerator`` the simulator — three connector blocks,
    therefore three configured targets. *writes* is the tier key: ``None``
    leaves it out, which with no ``writes_enabled`` leaf anywhere is the
    ``never`` tier an app render answers; ``"allowed"`` arms the tier.
    *with_live* ``False`` drops the ``epics`` block, so ``live`` is a machine
    this render does not configure.
    """
    connector = {
        "live_standin": {"gateways": _gateways(STANDIN_PORT)},
        "virtual_accelerator": {"gateways": _gateways(5064)},
    }
    if with_live:
        connector["epics"] = {"gateways": _gateways(5064)}
    section = {"type": "live_standin", "connector": connector}
    if writes is not None:
        section["writes"] = writes
    return section


#: The tier-allowed render, as the write verdict and the executor read it.
ALLOWED = control_system_section(writes="allowed")


def write_config(tmp_path, section=None, *, name="config.yml"):
    """Write a ``config.yml`` carrying *section* (default: the tier-allowed shape)."""
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": ALLOWED if section is None else section,
                "services": {
                    "live_standin": {"port": STANDIN_PORT},
                    "virtual_accelerator": {"port": 5064},
                },
                "deployed_services": ["virtual_accelerator", "live_standin"],
            }
        ),
        encoding="utf-8",
    )
    return path


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def make_client(agent_data_root, workspace_dir, tmp_path):
    """Build an app + TestClient, repeatably, over the same stamped root.

    The record is written **before** the first app starts, so the lifespan
    claim merges into it. The owner task's loop is stubbed out: a loop ticking
    underneath the assertions would be a race rather than coverage.
    """
    write_control_context(agent_data_root, target="live", generation=1)

    @contextmanager
    def _make(config_path=None):
        with patch.object(
            control_context_owner.ControlContextOwnerTask, "start", lambda self: None
        ):
            with terminal_client(workspace_dir) as test_client:
                test_client.app.state.config_path = (
                    write_config(tmp_path) if config_path is None else config_path
                )
                yield test_client

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


class _RecordingChatPool:
    """An operator registry that remembers what was asked of it."""

    def __init__(self) -> None:
        self.terminated: list[str] = []

    def has_chat_key(self, key: str) -> bool:
        return key == CHAT_A

    def get_chat_session(self, key: str):
        return object() if key == CHAT_A else None

    def terminate_chat_session(self, key: str) -> None:
        self.terminated.append(key)

    async def cleanup_all(self) -> None:
        """Not recorded: it runs when the lifespan exits, after the assertion."""


# -- harness ----------------------------------------------------------------


def switch_now():
    """This login's switch, straight off the record."""
    record = read_record()
    return {} if record is None else dict(record.switch)


def epoch_now():
    record = read_record()
    return None if record is None else record.switch_epoch


def wire_now():
    """The record's epoch as the route carries it: a decimal string, or ``None``."""
    epoch = epoch_now()
    return None if epoch is None else str(epoch)


def make_epochless():
    """Rewrite the record with no ``switch_epoch``, as a file from before epochs."""
    path = control_context.record_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["switch_epoch"] = None
    write_payload(path, payload)
    control_context.invalidate_cache()


def post_switch(client, *, session_id=SESSION_A, target="standin", state="on", epoch="read"):
    """POST C-POST's body. ``epoch="read"`` sends the record's epoch, as the chip does."""
    body = {"target": target, "state": state, "epoch": wire_now() if epoch == "read" else epoch}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/api/terminal/posture", json=body)


def turn_on(client, target="standin", **kwargs):
    """The chip's turn-on: send the epoch read, retry once when the 409 stamped."""
    response = post_switch(client, target=target, state="on", **kwargs)
    if response.status_code == 409 and response.json()["detail"]["stamped"]:
        response = post_switch(
            client, target=target, state="on", epoch=response.json()["detail"]["epoch"]
        )
    return response


def assert_envelope(response, status, **expected):
    """The response is *status* with exactly the C-POST refusal *expected*."""
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert set(detail) <= ENVELOPE_KEYS
    assert detail == expected


def write_inflight_marker(root: Path, *, pid, target="standin"):
    """Plant one execution marker where the executor writes it."""
    directory = state_dir_under(root)
    path = (
        directory / f"{target_state.INFLIGHT_FILE_PREFIX}{pid}{target_state.INFLIGHT_FILE_SUFFIX}"
    )
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "target": target,
                "generation": 1,
                "session": "another-session",
                "surface": "python_executor",
                "kernel_id": None,
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return path


# -- the request body -------------------------------------------------------


class TestGrammar:
    @pytest.mark.parametrize(
        "bad_id",
        ["../../etc/passwd", "operator-deadbeef", "", "AAAAAAAA-1111-2222-3333-444444444444"],
    )
    def test_an_id_outside_the_closed_grammar_is_400_in_the_c_post_envelope(self, client, bad_id):
        assert_envelope(post_switch(client, session_id=bad_id), 400, reason="invalid_session_id")
        assert switch_now() == {}

    @pytest.mark.parametrize("bad", ["sandbox", "writes", "ON", "", "true"])
    def test_only_on_and_off_are_states(self, client, bad):
        response = post_switch(client, state=bad)
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)

    @pytest.mark.parametrize(
        "bad", [True, 5, 1.5, [], "", "-1", "+1", "01", "1e3", " 5", "5 ", "0x1f", "1" * 21]
    )
    def test_the_epoch_is_a_decimal_string_or_null(self, client, bad):
        """Never coerced: a number, ``true`` or any other spelling is a 422 that says so."""
        response = post_switch(client, epoch=bad)
        assert response.status_code == 422, response.text
        assert websocket_routes.EPOCH_FORMAT_PHRASE in response.text
        assert switch_now() == {}

    def test_the_old_posture_body_is_422(self, client):
        response = client.post(
            "/api/terminal/posture",
            json={"session_id": SESSION_A, "target": "standin", "posture": "writes"},
        )
        assert response.status_code == 422
        assert switch_now() == {}

    def test_all_with_on_is_400_bulk_on_refused(self, client):
        """A machine is turned on one at a time; the route refuses before the record."""
        with patch.object(control_context, "set_switch") as setter:
            assert_envelope(post_switch(client, target="all"), 400, reason="bulk_on_refused")
        setter.assert_not_called()

    @pytest.mark.parametrize("state", ["on", "off"])
    def test_a_target_this_render_does_not_configure_is_400(self, client, state):
        assert_envelope(
            post_switch(client, target="banana", state=state), 400, reason="unknown_target"
        )

    def test_an_unreadable_render_configures_no_named_target(self, client):
        client.app.state.config_path = None
        assert_envelope(post_switch(client, state="off"), 400, reason="unknown_target")

    def test_route_outcomes_are_not_verdict_reasons(self):
        """The closed set stays closed: these answer the request, not a write."""
        for reason in ("bulk_on_refused", "unknown_target", "not_taken", "invalid_session_id"):
            assert reason not in REASONS


# -- on, off, all off -------------------------------------------------------


class TestTheGesture:
    def test_on_arms_that_machine_and_answers_exactly_c_post(self, client):
        response = turn_on(client, "standin")

        assert response.status_code == 200, response.text
        assert response.json() == {
            "target": "standin",
            "state": "on",
            "switch_epoch": wire_now(),
        }
        assert isinstance(response.json()["switch_epoch"], str)
        assert switch_now() == {"standin": "asked"}

    def test_off_disarms_only_that_machine(self, client):
        assert turn_on(client, "standin").status_code == 200
        assert turn_on(client, "va").status_code == 200

        response = post_switch(client, target="standin", state="off")

        assert response.status_code == 200, response.text
        assert response.json() == {"target": "standin", "state": "off", "switch_epoch": wire_now()}
        assert switch_now() == {"va": "asked"}

    def test_all_off_clears_every_machine_and_moves_the_epoch(self, client):
        assert turn_on(client, "standin").status_code == 200
        assert turn_on(client, "live").status_code == 200
        before = epoch_now()

        response = post_switch(client, target="all", state="off")

        assert response.status_code == 200, response.text
        assert response.json()["target"] == "all"
        assert switch_now() == {}
        assert epoch_now() > before

    def test_all_off_clears_machines_this_render_does_not_configure(
        self, client, tmp_path, agent_data_root
    ):
        """``any_on`` counts every machine in the record; "Turn all off" must reach them."""
        write_control_context(agent_data_root, target="live", generation=1, switch={"live": "on"})
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(writes="allowed", with_live=False), name="two.yml"
        )
        assert_envelope(
            post_switch(client, target="live", state="off"), 400, reason="unknown_target"
        )

        assert post_switch(client, target="all", state="off").status_code == 200
        assert switch_now() == {}

    def test_all_off_needs_no_readable_render(self, client, agent_data_root):
        write_control_context(agent_data_root, target="live", generation=1, switch={"va": "on"})
        client.app.state.config_path = None
        assert post_switch(client, target="all", state="off").status_code == 200
        assert switch_now() == {}

    def test_a_session_less_gesture_is_accepted(self, client):
        """The Lab page's bar posts with no session; the switch is the login's."""
        assert turn_on(client, "standin", session_id=None).status_code == 200
        assert switch_now() == {"standin": "asked"}

    def test_the_gesture_moves_neither_target_nor_generation(self, client):
        before = read_record()
        assert turn_on(client, "standin").status_code == 200
        after = read_record()
        assert (after.target, after.generation, after.owner) == (
            before.target,
            before.generation,
            before.owner,
        )

    def test_each_gesture_pushes_the_context_frame(self, client):
        broadcaster = MagicMock()
        client.app.state.broadcaster = broadcaster
        assert turn_on(client, "standin").status_code == 200
        assert post_switch(client, target="standin", state="off").status_code == 200
        frames = [call.args[0] for call in broadcaster.broadcast.call_args_list]
        assert frames
        assert all(frame == {"type": "control_context"} for frame in frames)
        assert len(frames) >= 2


class TestOffIsNeverRefused:
    """A stop always lands — the gesture an operator needs most."""

    @pytest.fixture
    def on(self, client):
        assert turn_on(client, "standin").status_code == 200
        return client

    def test_off_lands_on_a_never_tier(self, on, tmp_path):
        on.app.state.config_path = write_config(tmp_path, control_system_section(), name="n.yml")
        assert post_switch(on, target="standin", state="off").status_code == 200
        assert switch_now() == {}

    def test_off_lands_on_a_readonly_run(self, on, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        assert post_switch(on, target="standin", state="off").status_code == 200
        assert switch_now() == {}

    def test_off_lands_from_a_follower_terminal(self, on, agent_data_root):
        """The switch is the login's, so a follower acts exactly like the holder."""
        holder = owner(pid=OTHER_TERMINAL_PID, port=OTHER_TERMINAL_PORT)
        write_control_context(
            agent_data_root, target="live", generation=1, owned_by=holder, switch={"standin": "on"}
        )
        on.app.state.control_context_follows = holder

        assert post_switch(on, target="standin", state="off").status_code == 200
        assert switch_now() == {}
        assert read_record().owner == holder

    def test_on_lands_from_a_follower_terminal_too(self, client, agent_data_root):
        holder = owner(pid=OTHER_TERMINAL_PID, port=OTHER_TERMINAL_PORT)
        write_control_context(agent_data_root, target="live", generation=1, owned_by=holder)
        client.app.state.control_context_follows = holder

        assert turn_on(client, "va").status_code == 200
        assert switch_now() == {"va": "asked"}
        assert read_record().owner == holder

    def test_neither_gesture_needs_the_owner_primitive(self, client):
        """No owner on ``app.state`` is no 503: ``set_switch`` is called directly."""
        del client.app.state.control_context_owner
        assert turn_on(client, "standin").status_code == 200
        assert post_switch(client, target="standin", state="off").status_code == 200

    @pytest.mark.parametrize("state", ["on", "off"])
    def test_a_run_in_flight_refuses_neither(self, client, agent_data_root, state):
        write_inflight_marker(agent_data_root, pid=4242, target="standin")
        with only_alive(4242, os.getpid()):
            response = (
                turn_on(client, "standin") if state == "on" else post_switch(client, state="off")
            )
        assert response.status_code == 200, response.text


class TestTurnOnLocks:
    """403 ``{reason, phrase}`` for the two locks no chip gesture can lift."""

    def test_a_never_tier_refuses_on(self, client, tmp_path):
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(writes="never"), name="never.yml"
        )
        assert_envelope(
            post_switch(client, target="standin"),
            403,
            reason="tier_never",
            phrase=refusal_phrase("tier_never", "real"),
        )
        assert switch_now() == {}

    def test_an_app_render_refuses_on_as_tier_never(self, client, tmp_path):
        """No tier key and no writes leaf: the verdict answers ``tier_never``."""
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(), name="app.yml"
        )
        verdict = posture_store.write_verdict(control_system_section(), "va")
        assert verdict.reason == "tier_never"
        assert_envelope(
            post_switch(client, target="va"),
            403,
            reason="tier_never",
            phrase=refusal_phrase("tier_never", verdict.kind),
        )
        assert switch_now() == {}

    def test_a_readonly_run_refuses_on(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        assert_envelope(
            post_switch(client, target="live"),
            403,
            reason="readonly_run",
            phrase=refusal_phrase("readonly_run", "real"),
        )
        assert switch_now() == {}

    def test_the_lock_is_the_verdict_for_this_login_with_the_pin_ignored(self, client, monkeypatch):
        seen = []
        real = posture_store.write_verdict

        def recording(section, target, **kwargs):
            seen.append((target, kwargs))
            return real(section, target, **kwargs)

        monkeypatch.setattr(posture_store, "write_verdict", recording)
        assert turn_on(client, "standin").status_code == 200
        assert seen
        assert all(
            kwargs == {"owner": acting_identity(), "ignore_launch_pin": True}
            for target, kwargs in seen
        )


class TestTheLoginsOwnTierLock:
    """``locks.tier`` on a terminal: its own render's tier, never ``unknown`` (BC-1).

    A terminal-served render carries no ``control_system.login_tiers`` and the
    terminal binds no control-context tree, so the owner rung answers ``skip``
    and the render's tier governs the card's own login.
    """

    def _locks(self, client):
        resp = client.get("/api/terminal/posture")
        assert resp.status_code == 200, resp.text
        return resp.json()["locks"]

    @pytest.mark.usefixtures("agent_data_root")
    def test_the_terminal_render_carries_no_map_and_binds_no_tree(self):
        assert posture_store.LOGIN_TIERS_KEY.rsplit(".", 1)[-1] not in ALLOWED
        assert "OSPREY_CONTROL_CONTEXT_TREE" not in os.environ
        assert posture_store.owner_tier_term(ALLOWED, acting_identity()) == "skip"

    def test_an_allowed_render_locks_its_own_login_allowed(self, client):
        assert self._locks(client)["tier"] == "allowed"

    def test_a_never_render_locks_its_own_login_never(self, client, tmp_path):
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(writes="never"), name="never.yml"
        )
        assert self._locks(client)["tier"] == "never"

    @pytest.mark.usefixtures("agent_data_root")
    @pytest.mark.parametrize("login", ["alice", "someone-else", ""])
    def test_no_login_reads_unknown_on_a_map_free_terminal_render(self, login):
        assert websocket_routes._login_tier(ALLOWED, login) == "allowed"

    @pytest.mark.usefixtures("agent_data_root")
    def test_a_section_carrying_the_map_reads_the_logins_entry(self):
        """The shared-container shape, for contrast: the map decides, absent is unknown."""
        section = {**ALLOWED, "login_tiers": {"alice": "allowed", "bob": "never"}}
        assert websocket_routes._login_tier(section, "alice") == "allowed"
        assert websocket_routes._login_tier(section, "bob") == "never"
        assert websocket_routes._login_tier(section, "carol") == websocket_routes.TIER_UNKNOWN


# -- the epoch --------------------------------------------------------------


class TestEpoch:
    def test_a_null_epoch_on_an_epochless_record_stamps_then_the_retry_lands(self, client):
        make_epochless()
        assert epoch_now() is None

        first = post_switch(client, epoch=None)

        assert first.status_code == 409, first.text
        detail = first.json()["detail"]
        assert set(detail) == {"reason", "epoch", "stamped"}
        assert detail["reason"] == "stopped_since"
        assert detail["stamped"] is True
        assert isinstance(detail["epoch"], str)
        assert detail["epoch"] == wire_now()
        assert switch_now() == {}

        second = post_switch(client, epoch=detail["epoch"])
        assert second.status_code == 200, second.text
        assert switch_now() == {"standin": "asked"}

    def test_a_stale_epoch_is_an_unstamped_409_carrying_the_records(self, client):
        on_disk = epoch_now()
        assert on_disk is not None
        assert_envelope(
            post_switch(client, epoch=str(on_disk - 1)),
            409,
            reason="stopped_since",
            epoch=str(on_disk),
            stamped=False,
        )
        assert switch_now() == {}

    def test_a_null_epoch_on_a_record_with_one_is_unstamped(self, client):
        """The sentinel 0 never matches a real epoch: the chip re-reads."""
        assert_envelope(
            post_switch(client, epoch=None),
            409,
            reason="stopped_since",
            epoch=wire_now(),
            stamped=False,
        )

    def test_a_turn_on_that_raced_a_stop_leaves_nothing_on(self, client):
        """The chip read an epoch; a stop of everything landed; the turn-on stays off."""
        seen = wire_now()
        assert post_switch(client, target="all", state="off").status_code == 200

        response = post_switch(client, epoch=seen)

        assert response.status_code == 409
        assert response.json()["detail"]["stamped"] is False
        assert switch_now() == {}

    def test_a_nanosecond_epoch_round_trips_from_the_read_to_the_turn_on(self, client, monkeypatch):
        """An epoch with nanosecond digits comes back from the read exactly as stored.

        Above 2**53 a JSON number reaches a browser rounded, and the turn-on
        it sends back no longer matches: every gesture answered 409. A clock
        with non-zero low digits, as on Linux, pins the string on every host.
        """
        ns = 4102444800123456789
        assert int(float(ns)) != ns, "the pin needs an epoch a double cannot hold"
        monkeypatch.setattr(control_context.time, "time_ns", lambda: ns)
        assert post_switch(client, target="all", state="off").status_code == 200
        assert epoch_now() == ns

        read = client.get("/api/terminal/posture", params={"session_id": SESSION_A})
        assert read.status_code == 200, read.text
        assert read.json()["switch_epoch"] == str(ns)

        response = post_switch(client, epoch=read.json()["switch_epoch"])

        assert response.status_code == 200, response.text
        assert response.json()["switch_epoch"] == str(ns)
        assert switch_now() == {"standin": "asked"}

    def test_the_server_never_retries(self, client, monkeypatch):
        calls = []
        real = control_context.set_switch

        def counting(*args, **kwargs):
            calls.append(kwargs.get("epoch"))
            return real(*args, **kwargs)

        monkeypatch.setattr(control_context, "set_switch", counting)
        make_epochless()
        assert post_switch(client, epoch=None).status_code == 409
        assert calls == [0]


# -- no usable record -------------------------------------------------------


class TestNoUsableRecord:
    """503 for on and for off — never a 409 loop, never a 200."""

    @pytest.fixture
    def missing(self, client):
        control_context.record_path().unlink()
        control_context.invalidate_cache()
        return client

    @pytest.fixture
    def unreadable(self, client):
        control_context.record_path().write_text("{not json", encoding="utf-8")
        control_context.invalidate_cache()
        return client

    @pytest.mark.parametrize("which", ["missing", "unreadable"])
    def test_on_answers_record_unavailable(self, request, which):
        client = request.getfixturevalue(which)
        for epoch in (None, "0", "12345"):
            assert_envelope(
                post_switch(client, epoch=epoch),
                503,
                reason="record_unavailable",
                phrase=refusal_phrase("record_unavailable"),
            )

    @pytest.mark.parametrize("which", ["missing", "unreadable"])
    @pytest.mark.parametrize("target", ["standin", "all"])
    def test_off_answers_not_taken_with_the_stop_sentence(self, request, which, target):
        client = request.getfixturevalue(which)
        response = post_switch(client, target=target, state="off")
        assert_envelope(response, 503, reason="not_taken", taken=False, phrase=STOP_SENTENCE)
        assert "writes are refused" not in response.json()["detail"]["phrase"].lower()

    def test_the_stop_sentence_is_the_routes_constant(self):
        assert websocket_routes.STOP_NOT_TAKEN_PHRASE == STOP_SENTENCE

    @pytest.mark.parametrize("error", [OSError("read-only volume"), RuntimeError("lock wait")])
    def test_a_failed_write_of_off_is_not_taken_and_writes_stay_on(self, client, error):
        assert turn_on(client, "standin").status_code == 200
        with patch.object(control_context, "_write_record_unlocked", side_effect=error):
            response = post_switch(client, target="standin", state="off")
        assert_envelope(response, 503, reason="not_taken", taken=False, phrase=STOP_SENTENCE)
        assert switch_now() == {"standin": "asked"}

    def test_a_failed_write_of_on_is_not_taken_with_the_record_phrase(self, client):
        with patch.object(
            control_context, "_write_record_unlocked", side_effect=OSError("read-only volume")
        ):
            response = post_switch(client)
        assert_envelope(
            response,
            503,
            reason="not_taken",
            taken=False,
            phrase=refusal_phrase("record_unavailable"),
        )
        assert switch_now() == {}

    def test_a_record_lost_after_the_status_read_is_still_503(self, client):
        """``set_switch`` answers no record with ``StoppedSince(epoch=None)``: not a 409."""
        with patch.object(
            control_context, "set_switch", side_effect=control_context.StoppedSince(epoch=None)
        ):
            assert_envelope(
                post_switch(client),
                503,
                reason="record_unavailable",
                phrase=refusal_phrase("record_unavailable"),
            )
        with patch.object(control_context, "set_switch", return_value=None):
            assert_envelope(
                post_switch(client, state="off"),
                503,
                reason="not_taken",
                taken=False,
                phrase=STOP_SENTENCE,
            )


# -- the envelope -----------------------------------------------------------


class TestEnvelope:
    def test_the_posture_refusal_helper_admits_only_c_post_keys(self):
        with pytest.raises(TypeError):
            websocket_routes._refuse_posture(
                None, status_code=400, gesture="g", reason="unknown_target", message="no"
            )

    def test_the_target_route_keeps_error_and_message(self, client):
        response = client.post(
            "/api/terminal/target", json={"session_id": SESSION_A, "target": "banana"}
        )
        assert response.status_code == 400
        assert set(response.json()["detail"]) == {"error", "message"}


# -- audit ------------------------------------------------------------------


def _assert_one_line(ledger, *, decision, reason, state, target="standin"):
    assert len(ledger) == 1, ledger
    line = ledger[0]
    assert line["subject"] == "session_posture_set"
    assert line["decision"] == decision
    assert line["reason"] == reason
    for part in (f"login={acting_identity()}", f"target={target}", f"state={state}"):
        assert part in line["detail"]
    assert f"outcome={reason}" in line["detail"]
    return line


class TestAudit:
    """One ledger line per outcome, whichever outcome it was."""

    def test_an_accepted_on_files_one_allowed_line(self, client, ledger):
        assert post_switch(client).status_code == 200
        line = _assert_one_line(ledger, decision="allowed", reason="switch_set", state="on")
        assert line["session"] == SESSION_A
        assert "on=standin" in line["detail"]

    def test_an_accepted_all_off_files_one_line(self, client, ledger):
        assert post_switch(client, target="all", state="off").status_code == 200
        _assert_one_line(ledger, decision="allowed", reason="switch_set", state="off", target="all")

    def test_the_stamped_409_files_one_gesture_line(self, client, ledger):
        make_epochless()
        assert post_switch(client, epoch=None).status_code == 409
        line = _assert_one_line(ledger, decision="refused", reason="stopped_since", state="on")
        assert "stamped=true" in line["detail"]

    @pytest.mark.parametrize(
        ("kwargs", "status", "reason"),
        [
            ({"target": "all", "state": "on"}, 400, "bulk_on_refused"),
            ({"target": "banana", "state": "on"}, 400, "unknown_target"),
            ({"epoch": "1"}, 409, "stopped_since"),
        ],
    )
    def test_each_refusal_files_one_refused_line(self, client, ledger, kwargs, status, reason):
        response = post_switch(client, **kwargs)
        assert response.status_code == status
        _assert_one_line(
            ledger,
            decision="refused",
            reason=reason,
            state=kwargs.get("state", "on"),
            target=kwargs.get("target", "standin"),
        )

    def test_the_403_files_one_refused_line(self, client, ledger, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        assert post_switch(client).status_code == 403
        _assert_one_line(ledger, decision="refused", reason="readonly_run", state="on")

    def test_the_503_files_one_refused_line(self, client, ledger):
        control_context.record_path().unlink()
        control_context.invalidate_cache()
        assert post_switch(client, state="off").status_code == 503
        _assert_one_line(ledger, decision="refused", reason="not_taken", state="off")

    def test_a_malformed_session_id_still_leaves_exactly_one_line(self, client, ledger):
        """The middleware files it: there is no legitimate key to name."""
        assert post_switch(client, session_id="../../etc/passwd").status_code == 400
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"

    def test_a_session_less_gesture_is_filed_on_the_lab_surface(self, client, ledger):
        assert post_switch(client, session_id=None).status_code == 200
        assert len(ledger) == 1
        assert ledger[0]["session"] is None
        assert ledger[0]["surface"] == websocket_routes.LAB_MUTATION_SURFACE


# -- logins, restarts, target switches --------------------------------------


class TestTheSwitchIsTheLogins:
    def test_two_logins_are_isolated(self, client, agent_data_root, monkeypatch):
        pin_identity(monkeypatch, "alice")
        write_control_context(agent_data_root, target="live", generation=1)
        pin_identity(monkeypatch, "bob")
        write_control_context(agent_data_root, target="live", generation=1)

        pin_identity(monkeypatch, "alice")
        assert turn_on(client, "standin").status_code == 200

        pin_identity(monkeypatch, "bob")
        assert switch_now() == {}
        assert turn_on(client, "va").status_code == 200
        assert post_switch(client, target="all", state="off").status_code == 200

        pin_identity(monkeypatch, "alice")
        assert switch_now() == {"standin": "asked"}

    def test_on_survives_a_restart(self, make_client):
        with make_client() as first:
            assert turn_on(first, "standin").status_code == 200
        with make_client() as second:
            assert switch_now() == {"standin": "asked"}
            assert post_switch(second, target="standin", state="off").status_code == 200
        assert switch_now() == {}

    def test_on_survives_a_target_switch(self, client, agent_data_root, tmp_path):
        """The target route writes through the owner primitive and keeps the switch.

        A render the switch gate lets through to ``va`` (a probe channel, its
        own gateway, a real archiver, the acknowledged live gateway).
        """
        row = {"address": "gw.example.org", "use_name_server": False}
        va_row = {"address": "localhost", "port": 5074, "use_name_server": True}
        render = {
            "control_system": {
                "type": "epics",
                "writes": "allowed",
                "connector": {
                    "epics": {
                        "probe_channel": "LIVE:PROBE:CHANNEL",
                        "gateways": {
                            "read_only": {**row, "port": 5064},
                            "write_access": {**row, "port": 5084},
                        },
                    },
                    "virtual_accelerator": {
                        "probe_channel": "VA:PROBE:CHANNEL",
                        "gateways": {"read_only": dict(va_row), "write_access": dict(va_row)},
                    },
                },
                "target_switch": {target_eligibility.ACK_LEAF: "gw.example.org"},
            },
            "archiver": {"type": "epics_archiver"},
        }
        path = tmp_path / "switchable.yml"
        path.write_text(yaml.safe_dump(render), encoding="utf-8")
        client.app.state.config_path = path
        assert turn_on(client, "live").status_code == 200
        before = read_record()
        epoch = epoch_now()
        write_server_report(
            agent_data_root,
            SERVER_PID,
            reachability={
                "targets": {
                    "va": {
                        "read_only": {
                            "state": target_eligibility.REACHABILITY_REACHED,
                            "probed_at": datetime.now(UTC).isoformat(),
                        }
                    }
                }
            },
        )
        with only_alive(SERVER_PID, os.getpid()):
            moved = client.post(
                "/api/terminal/target", json={"session_id": SESSION_A, "target": "va"}
            )
        assert moved.status_code == 202, moved.text
        record = read_record()
        assert (record.target, record.generation) == ("va", before.generation + 1)
        assert switch_now() == {"live": "asked"}
        assert epoch_now() == epoch


class TestNoTermination:
    """The POST never respawns a session: every write path reads the switch live."""

    def test_the_live_child_survives_a_gesture(self, client):
        registry = client.app.state.pty_registry
        registry.get_or_create_session(SESSION_A, "echo")
        with (
            patch.object(registry, "detach_session") as detach,
            patch.object(registry, "terminate_session") as terminate,
            patch.object(registry, "terminate_session_if_owner") as terminate_if_owner,
        ):
            assert turn_on(client, "standin").status_code == 200
            assert post_switch(client, target="all", state="off").status_code == 200
        detach.assert_not_called()
        terminate.assert_not_called()
        terminate_if_owner.assert_not_called()
        assert registry.get_session(SESSION_A) is not None

    def test_the_chat_child_survives_a_gesture(self, client):
        recorder = _RecordingChatPool()
        client.app.state.operator_registry = recorder
        assert turn_on(client, "standin", session_id=CHAT_A).status_code == 200
        assert recorder.terminated == []


# -- what running and later writes see --------------------------------------


class TestRunsAndTheSwitch:
    """On reaches the next run and the next connector write; off reaches running ones.

    The sandbox's reference monitor asks
    :func:`~osprey_connectors.posture_store.write_verdict` on every write with
    the launch pin stamped by :func:`host_executor._launch_posture`, exactly as
    the executor stamps it; the connector asks it for its stamped machine.
    """

    @pytest.fixture(autouse=True)
    def _tier(self, monkeypatch):
        monkeypatch.setattr(audit_posture, "control_system_section", lambda: dict(ALLOWED))

    @staticmethod
    def _launch(monkeypatch, target):
        monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
        posture_store.invalidate_cache()
        stamp = host_executor._launch_posture(target)
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)
        posture_store.invalidate_cache()
        return stamp

    @staticmethod
    def _next_write(target):
        read_record()
        return posture_store.write_verdict(ALLOWED, target)

    def test_a_turn_on_lands_under_a_live_run_which_stays_pinned(
        self, client, agent_data_root, monkeypatch
    ):
        assert self._launch(monkeypatch, "standin") == "standin=sandbox"
        write_inflight_marker(agent_data_root, pid=4242, target="standin")

        with only_alive(4242, os.getpid()):
            response = turn_on(client, "standin")

        assert response.status_code == 200, response.text
        verdict = self._next_write("standin")
        assert verdict.allowed is False
        assert verdict.reason == "launch_pinned"

    def test_a_run_launched_after_the_turn_on_writes(self, client, monkeypatch):
        assert turn_on(client, "standin").status_code == 200
        assert self._launch(monkeypatch, "standin") == "standin=writes"
        assert self._next_write("standin").allowed is True

    def test_off_refuses_the_running_sandboxs_next_write(self, client, monkeypatch):
        assert turn_on(client, "standin").status_code == 200
        assert self._launch(monkeypatch, "standin") == "standin=writes"
        assert self._next_write("standin").allowed is True

        assert post_switch(client, target="standin", state="off").status_code == 200

        verdict = self._next_write("standin")
        assert verdict.allowed is False
        assert verdict.reason == "switch_off"

    @pytest.mark.asyncio
    async def test_a_connector_write_for_this_login_passes_at_its_next_call(
        self, client, monkeypatch
    ):
        monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(ALLOWED))
        connector = RecordingConnector()
        connector._connector_type = "epics"
        connector._control_target = "va"

        refused = await connector.write_channel("VA:CORR:1:SP", 0.5)
        assert refused.outcome is WriteOutcome.REFUSED
        assert connector.writes == []

        assert turn_on(client, "va").status_code == 200
        read_record()

        landed = await connector.write_channel("VA:CORR:1:SP", 0.5)
        assert landed.outcome is WriteOutcome.CONFIRMED
        assert connector.writes == [("VA:CORR:1:SP", 0.5)]
