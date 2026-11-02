"""A held session's writes are refused where they execute, and nowhere else.

``posture_store.session_hold_refusal`` answers one question: is the session
named by the environment held? ``execution_verdict`` orders that answer after
the read-only run and the tier and before the rest of ``write_verdict``, and the
connector's write wrappers ask it after ``_writes_enabled``. ``write_verdict``
and ``backstop_verdict`` stay blind to the hold, because they also choose the
gateway a connector child connects on: a hold that reached them would respawn
children and change the gateway on every flag and release.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from osprey_connectors import posture_store, session_hold
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.ipc._pool import ConnectorHostPool
from osprey_connectors.posture_store import (
    NO_OWNER,
    WriteVerdict,
    backstop_verdict,
    execution_verdict,
    session_hold_refusal,
    write_verdict,
)
from osprey_connectors.refusals import REASONS, refusal_phrase, refusal_text
from osprey_connectors.types import TARGET_VA
from osprey_connectors_tests._harness import REFUSAL_BAN_LIST
from osprey_connectors_tests._harness._control_context_fixtures import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
)
from osprey_connectors_tests._paths import PACKAGE_SRC
from osprey_connectors_tests._write_fakes import (
    ARMED_SECTION,
    RecordingConnector,
    arm_switch,
    writes_enabled_config,
)
from osprey_connectors_tests.ipc._soft_ioc import SoftIOC

FIXTURES = Path(__file__).parent / "fixtures" / "session_hold"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))["cases"]

SESSION_ID = "OSPREY_SESSION_ID"
KEY = "3f2b8c1e-6a0d-4c55-9a1e-0b7f2d4e9c11"  # gitleaks:allow - a test session id
OTHER = "7c1d9e2a-0b3f-4e66-8d2c-5a9e1f0b3c22"  # gitleaks:allow - a second test session id
PHRASE = "Writes are paused: this session has an unanswered hold the operator must answer."

CA_TIMEOUT_S = 5.0


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    """No stamped run state and no hold store; the host record under scratch."""
    isolate_verdict_environment(monkeypatch, tmp_path / "data")
    monkeypatch.delenv(SESSION_ID, raising=False)
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    yield
    reset_verdict_state()


@pytest.fixture
def hold_root(tmp_path, monkeypatch) -> Path:
    """The hold store stamped and this session's record present, nothing open."""
    root = tmp_path / "session_holds"
    root.mkdir()
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    monkeypatch.setenv(SESSION_ID, KEY)
    session_hold.ensure_session(KEY)
    return root


def _flag(flag_id: str = "f1", source: str = "watchdog:scope") -> None:
    assert session_hold.open_flag(
        KEY,
        flag_id=flag_id,
        source=source,
        reason="off the plan",
        prompt_ref=f"record:{flag_id}",
        prompt_excerpt="move the corrector",
    )


def _flag_for(key: str, flag_id: str = "f1", source: str = "watchdog:scope") -> None:
    session_hold.ensure_session(key)
    assert session_hold.open_flag(
        key,
        flag_id=flag_id,
        source=source,
        reason="off the plan",
        prompt_ref=f"record:{flag_id}",
        prompt_excerpt="move the corrector",
    )


def _release(*flag_ids: str) -> None:
    session_hold.answer_flags(KEY, list(flag_ids), answer="continue", operator="op")


def _corrupt(root: Path) -> None:
    (root / KEY / session_hold.RECORD_NAME).write_text("{not json", encoding="utf-8")


def _armed(monkeypatch, tmp_path) -> None:
    """The simulator's switch on, and a tier that allows writes."""
    arm_switch(monkeypatch, tmp_path / "data", TARGET_VA)
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)


# -- the words ---------------------------------------------------------------------


def test_the_reason_follows_the_tier_in_the_term_order():
    assert REASONS[:3] == ("readonly_run", "tier_never", "session_held")


def test_the_phrase_is_source_neutral_and_ends_with_who_answers_it():
    assert refusal_phrase("session_held") == PHRASE
    assert not REFUSAL_BAN_LIST.search(PHRASE)
    assert "watchdog" not in PHRASE


@pytest.mark.usefixtures("hold_root")
def test_the_sources_follow_the_phrase_and_the_trailer_ends_the_text():
    _flag(source="watchdog:scope")
    refusal = session_hold_refusal()
    text = refusal_text(refusal.reason, refusal.kind, detail=refusal.detail)
    assert text.startswith(PHRASE)
    assert "watchdog:scope" in text
    assert text.endswith("[osprey-refusal: session_held]")


# -- session_hold_refusal ----------------------------------------------------------


def test_no_hold_store_holds_nothing(monkeypatch):
    monkeypatch.setenv(SESSION_ID, KEY)
    assert session_hold_refusal() is None


@pytest.mark.usefixtures("hold_root")
def test_a_process_with_no_session_is_never_held(monkeypatch):
    _flag()
    monkeypatch.delenv(SESSION_ID)
    assert session_hold_refusal() is None


@pytest.mark.usefixtures("hold_root")
def test_an_empty_session_key_is_no_session(monkeypatch):
    monkeypatch.setenv(SESSION_ID, "")
    assert session_hold_refusal() is None


@pytest.mark.usefixtures("hold_root")
def test_an_open_flag_refuses_and_a_release_lifts_it():
    assert session_hold_refusal() is None
    _flag()
    refusal = session_hold_refusal()
    assert refusal == WriteVerdict(allowed=False, reason="session_held", target=None, kind=None)
    _release("f1")
    assert session_hold_refusal() is None


@pytest.mark.usefixtures("hold_root")
def test_each_source_is_named_once_in_the_order_opened():
    _flag("f1", source="watchdog:b")
    _flag("f2", source="watchdog:a")
    _flag("f3", source="watchdog:b")
    detail = session_hold_refusal().detail
    assert detail.count("watchdog:b") == 1
    assert detail.index("watchdog:b") < detail.index("watchdog:a")


@pytest.mark.usefixtures("hold_root")
def test_a_closed_flags_source_is_not_named():
    _flag("f1", source="watchdog:gone")
    _flag("f2", source="watchdog:open")
    _release("f1")
    detail = session_hold_refusal().detail
    assert "watchdog:open" in detail
    assert "watchdog:gone" not in detail


def test_a_corrupt_record_refuses_in_words_an_operator_can_read(hold_root):
    _corrupt(hold_root)
    refusal = session_hold_refusal()
    assert refusal.reason == "session_held"
    assert not REFUSAL_BAN_LIST.search(refusal.detail)


def test_an_explicit_root_needs_no_stamp(tmp_path, monkeypatch):
    root = tmp_path / "explicit"
    monkeypatch.setenv(SESSION_ID, KEY)
    session_hold.ensure_session(KEY, root=root)
    session_hold.open_flag(
        KEY, flag_id="f", source="s", reason="r", prompt_ref="p", prompt_excerpt="", root=root
    )
    assert session_hold_refusal() is None
    assert session_hold_refusal(root=root).reason == "session_held"


@pytest.mark.usefixtures("hold_root")
def test_a_reader_failure_refuses_and_never_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(session_hold, "open_flags", _boom)
    assert session_hold_refusal().reason == "session_held"


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_refusal_agrees_with_every_shared_fixture(case, tmp_path, monkeypatch):
    spec = CASES[case]
    root = tmp_path / "root"
    shutil.copytree(FIXTURES / case / "root", root)
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    monkeypatch.setenv(SESSION_ID, spec["session_key"])
    refusal = session_hold_refusal()
    assert (refusal is not None) is spec["held"]
    assert session_hold.is_held(spec["session_key"]) is spec["held"]
    if spec["held"]:
        assert refusal.reason == "session_held"


# -- several sessions at once ---------------------------------------------------


@pytest.mark.usefixtures("hold_root")
def test_no_session_keys_belong_to_no_session():
    _flag()  # the environment's session is held, and is not asked
    assert session_hold_refusal(session_keys=[]) is None
    assert session_hold_refusal(session_keys=[""]) is None
    assert execution_verdict(ARMED_SECTION, TARGET_VA, session_keys=[]) == write_verdict(
        ARMED_SECTION, TARGET_VA
    )


@pytest.mark.usefixtures("hold_root")
def test_one_session_key_is_asked_in_place_of_the_environment(monkeypatch):
    _flag_for(OTHER)
    monkeypatch.delenv(SESSION_ID)
    assert session_hold_refusal() is None
    assert session_hold_refusal(session_keys=[OTHER]).reason == "session_held"
    assert session_hold_refusal(session_keys=[KEY]) is None


@pytest.mark.usefixtures("hold_root")
def test_any_held_session_holds_with_its_own_flags():
    session_hold.ensure_session(OTHER)
    assert session_hold_refusal(session_keys=[KEY, OTHER]) is None

    _flag_for(OTHER, source="watchdog:other")
    refusal = session_hold_refusal(session_keys=[KEY, OTHER])
    assert refusal == WriteVerdict(allowed=False, reason="session_held", target=None, kind=None)
    assert refusal.detail == "Raised by: watchdog:other."

    _flag("f2", source="watchdog:mine")
    assert session_hold_refusal(session_keys=[KEY, OTHER]).detail == "Raised by: watchdog:mine."
    assert session_hold_refusal(session_keys=[OTHER, KEY]).detail == "Raised by: watchdog:other."

    session_hold.answer_flags(OTHER, ["f1"], answer="continue", operator="op")
    _release("f2")
    assert session_hold_refusal(session_keys=[KEY, OTHER]) is None


@pytest.mark.usefixtures("hold_root")
def test_an_unsafe_key_among_several_holds():
    session_hold.ensure_session(OTHER)
    assert session_hold_refusal(session_keys=[OTHER, "../escape"]).reason == "session_held"


@pytest.mark.usefixtures("hold_root")
def test_execution_verdict_asks_the_named_sessions(monkeypatch, tmp_path):
    _armed(monkeypatch, tmp_path)
    _flag()  # the environment's session
    session_hold.ensure_session(OTHER)
    allowed = execution_verdict(ARMED_SECTION, TARGET_VA, session_keys=[OTHER])
    assert allowed.allowed is True
    assert allowed == write_verdict(ARMED_SECTION, TARGET_VA)

    held = execution_verdict(ARMED_SECTION, TARGET_VA, session_keys=[OTHER, KEY])
    assert held.reason == "session_held"
    assert (held.target, held.kind) == (TARGET_VA, "simulator")
    assert execution_verdict(ARMED_SECTION, TARGET_VA).reason == "session_held"


# -- execution_verdict -------------------------------------------------------------


@pytest.mark.usefixtures("hold_root")
def test_held_with_the_switch_off_refuses_session_held():
    _flag()
    assert write_verdict(ARMED_SECTION, TARGET_VA).reason == "switch_off"
    verdict = execution_verdict(ARMED_SECTION, TARGET_VA)
    assert verdict.reason == "session_held"
    assert verdict.allowed is False


@pytest.mark.usefixtures("hold_root")
def test_the_held_answer_keeps_the_target_and_kind_of_the_verdict():
    _flag()
    plain = write_verdict(ARMED_SECTION, TARGET_VA)
    verdict = execution_verdict(ARMED_SECTION, TARGET_VA)
    assert (verdict.target, verdict.kind) == (plain.target, plain.kind)
    assert "watchdog:scope" in verdict.detail


@pytest.mark.usefixtures("hold_root")
def test_a_readonly_run_comes_before_the_hold(monkeypatch):
    _flag()
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    assert execution_verdict(ARMED_SECTION, TARGET_VA).reason == "readonly_run"


@pytest.mark.usefixtures("hold_root")
def test_the_section_tier_comes_before_the_hold():
    _flag()
    assert execution_verdict({"writes": "never"}, TARGET_VA).reason == "tier_never"


@pytest.mark.usefixtures("hold_root")
def test_the_hold_comes_before_the_target():
    _flag()
    assert execution_verdict(ARMED_SECTION, "no-such-machine").reason == "session_held"


def test_the_hold_comes_before_the_owner_rung(hold_root, monkeypatch, tmp_path):
    _armed(monkeypatch, tmp_path)
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(hold_root))
    monkeypatch.setenv(SESSION_ID, KEY)
    _flag()
    monkeypatch.setattr(posture_store, "owner_tier_term", lambda *a, **k: "never")
    assert write_verdict(ARMED_SECTION, TARGET_VA).reason == "tier_never"
    assert execution_verdict(ARMED_SECTION, TARGET_VA).reason == "session_held"


@pytest.mark.usefixtures("hold_root")
def test_a_session_not_held_gets_the_write_verdict_unchanged(monkeypatch, tmp_path):
    for target in (TARGET_VA, "no-such-machine", None):
        assert execution_verdict(ARMED_SECTION, target) == write_verdict(ARMED_SECTION, target)
    _armed(monkeypatch, tmp_path)
    verdict = execution_verdict(ARMED_SECTION, TARGET_VA)
    assert verdict.allowed is True
    assert verdict == write_verdict(ARMED_SECTION, TARGET_VA)


def test_the_keywords_reach_the_write_verdict(monkeypatch, tmp_path):
    _armed(monkeypatch, tmp_path)
    assert execution_verdict(ARMED_SECTION, TARGET_VA, owner=NO_OWNER).reason == "owner_unknown"
    monkeypatch.setattr(posture_store, "launch_pin_term", lambda *a, **k: "pinned")
    assert execution_verdict(ARMED_SECTION, TARGET_VA).reason == "launch_pinned"
    assert execution_verdict(ARMED_SECTION, TARGET_VA, ignore_launch_pin=True).allowed is True


def test_the_gateway_verdicts_stay_blind_to_the_hold(hold_root, monkeypatch, tmp_path):
    _armed(monkeypatch, tmp_path)
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(hold_root))
    monkeypatch.setenv(SESSION_ID, KEY)
    _flag()
    assert write_verdict(ARMED_SECTION, TARGET_VA).allowed is True
    assert backstop_verdict(ARMED_SECTION).allowed is True


# -- the connector's write wrappers ------------------------------------------------


@pytest.fixture
def armed_held(monkeypatch, tmp_path, hold_root) -> Path:
    """An armed connector's world, with the hold store stamped; nothing open yet."""
    _armed(monkeypatch, tmp_path)
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(hold_root))
    monkeypatch.setenv(SESSION_ID, KEY)
    return hold_root


@pytest.mark.usefixtures("armed_held")
async def test_write_channel_refuses_while_held_and_writes_after_release():
    connector = RecordingConnector()
    _flag()
    refused = await connector.write_channel("PV:A", 1.0)
    assert refused.outcome is WriteOutcome.REFUSED
    assert refused.refusal_reason == "WRITES_DISABLED"
    assert PHRASE in refused.error_message
    assert "watchdog:scope" in refused.error_message
    assert refused.error_message.endswith("[osprey-refusal: session_held]")
    assert connector.writes == []

    _release("f1")
    landed = await connector.write_channel("PV:A", 1.0)
    assert landed.outcome is WriteOutcome.CONFIRMED
    assert connector.writes == [("PV:A", 1.0)]


@pytest.mark.usefixtures("armed_held")
async def test_a_batch_refuses_every_operation_with_one_answer():
    connector = RecordingConnector()
    _flag()
    results = await connector.write_multiple_channels([("PV:A", 1.0), ("PV:B", 2.0)])
    assert [r.channel_address for r in results] == ["PV:A", "PV:B"]
    assert {r.outcome for r in results} == {WriteOutcome.REFUSED}
    assert all(r.error_message.endswith("[osprey-refusal: session_held]") for r in results)
    assert connector.writes == []

    _release("f1")
    results = await connector.write_multiple_channels([("PV:A", 1.0), ("PV:B", 2.0)])
    assert {r.outcome for r in results} == {WriteOutcome.CONFIRMED}
    assert connector.writes == [("PV:A", 1.0), ("PV:B", 2.0)]


async def test_a_corrupt_record_refuses_the_write(armed_held):
    connector = RecordingConnector()
    _corrupt(armed_held)
    refused = await connector.write_channel("PV:A", 1.0)
    assert refused.outcome is WriteOutcome.REFUSED
    assert refused.error_message.endswith("[osprey-refusal: session_held]")
    assert connector.writes == []


@pytest.mark.usefixtures("armed_held")
async def test_the_hold_never_changes_what_writes_enabled_reports():
    connector = RecordingConnector()
    assert connector._writes_enabled is True
    _flag()
    assert connector._writes_enabled is True


@pytest.mark.usefixtures("armed_held")
async def test_the_monitor_answers_first_when_it_refuses():
    connector = RecordingConnector()
    connector._control_target = None
    _flag()
    refused = await connector.write_channel("PV:A", 1.0)
    assert refused.error_message.endswith("[osprey-refusal: target_unknown]")


@pytest.mark.usefixtures("armed_held")
async def test_a_subclass_overriding_the_monitor_is_still_held():
    class _Open(RecordingConnector):
        @property
        def _writes_enabled(self) -> bool:
            return True

    connector = _Open()
    _flag()
    refused = await connector.write_channel("PV:A", 1.0)
    assert refused.error_message.endswith("[osprey-refusal: session_held]")
    assert connector.writes == []


@pytest.mark.usefixtures("armed_held")
async def test_a_process_with_no_session_writes_past_a_held_store(monkeypatch):
    connector = RecordingConnector()
    _flag()
    monkeypatch.delenv(SESSION_ID)
    assert (await connector.write_channel("PV:A", 1.0)).outcome is WriteOutcome.CONFIRMED


# -- a real connector child ---------------------------------------------------------


async def test_a_child_spawned_held_connects_for_writes_and_survives_the_release(
    tmp_path, monkeypatch, ioc_prefix
):
    """Selection and verification are blind to the hold; only the write is refused."""
    setpoint = f"{ioc_prefix}:SP"
    ioc = SoftIOC(tmp_path, "machine", ioc_prefix, 1.0)
    pool = None
    try:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        monkeypatch.setenv("PYTHONPATH", str(PACKAGE_SRC))
        data = tmp_path / "agent-data"
        arm_switch(monkeypatch, data, None, stamp=False)
        write_control_context(data, target="live", owned_by=None, switch={"live": "on"})

        root = tmp_path / "session_holds"
        monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
        monkeypatch.setenv(SESSION_ID, KEY)
        session_hold.ensure_session(KEY)
        _flag()

        endpoint = {"address": "127.0.0.1", "port": ioc.port, "use_name_server": False}
        section = {
            "type": "epics",
            "writes": "allowed",
            "connector": {
                "epics": {
                    "timeout_s": CA_TIMEOUT_S,
                    "gateways": {"read_only": dict(endpoint), "write_access": dict(endpoint)},
                }
            },
        }
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump({"control_system": section}))
        pool = ConnectorHostPool(section, config_file=config_file)

        # ``pool.connector`` raises when the child's report fails verification.
        live = await pool.connector("live")
        assert live.report["selected_role"] == "write_access"
        assert live.report["writes_enabled"] is True
        pids = pool.pids()
        assert pids

        refused = await live.write_channel(setpoint, 5.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert refused.outcome is WriteOutcome.REFUSED
        assert refused.error_message.endswith("[osprey-refusal: session_held]")
        assert (await live.read_channel(setpoint, timeout=CA_TIMEOUT_S)).value == 1.0

        _release("f1")
        landed = await live.write_channel(setpoint, 5.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert landed.outcome is WriteOutcome.CONFIRMED
        assert (await live.read_channel(setpoint, timeout=CA_TIMEOUT_S)).value == 5.0

        again = await pool.connector("live")
        assert again.report["selected_role"] == "write_access"
        assert pool.pids() == pids
    finally:
        if pool is not None:
            await pool.close()
        ioc.stop()
