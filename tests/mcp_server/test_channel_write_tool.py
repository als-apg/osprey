"""Tests for the channel_write MCP tool.

Covers: the six-word ``outcome`` contract, the report-vs-raise boundary of the
returned envelope, ``confirm`` passthrough, the bounded projection of an
oversize observed value, limits violations, connection errors and error-format
compliance.

Every assertion about an outcome is made on the PARSED tool output —
``summary.results[]`` — never on an internal dict. That projection is the only
thing an agent ever sees, so a field added to the tool's private bookkeeping and
not to the summary would be invisible in production while looking correct in a
test.

The results the connector hands the tool are real ``ChannelWriteResult``
objects, not ``MagicMock``s: a mock answers every attribute truthily, so a
projection reading a field no connector populates would pass here and misreport
in the field.

Every call here runs where the tool's own stage 1 lets it through: a write
tier that allows, the session target's switch on in the record, and the ask
stamp the approval hook would have filed for the call
(``tests/_approval_stamp_fixture.py``). The stage-1 refusals themselves — the
write verdict first, then the approval record — have their own section below.
The tool does its own limits validation via LimitsValidator.
"""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors.errors import ChannelUnreachableError
from osprey_connectors_tests._harness import (
    FIXTURE_IDENTITY,
    isolate_verdict_environment,
    write_control_context,
)
from tests._approval_stamp_fixture import ApprovalStamps
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

#: Every outcome word a result can carry, in the order the contract documents
#: them. Pinned as a list so a word added, dropped or reworded on one side of
#: the contract is visible here.
EXPECTED_OUTCOMES = [
    "refused",
    "failed",
    "confirmed",
    "mismatch",
    "unconfirmed",
    "unrequested",
]

#: Distinguishes "argument not given" from an explicit ``None``, which is itself
#: a meaningful value for ``observed_value``.
_UNSET = object()


def _make_write_result(
    channel="TEST:PV",
    value=1.0,
    outcome="confirmed",
    refusal_reason=None,
    error_message=None,
    observed_value=_UNSET,
    alarm_status=None,
    alarm_severity=None,
    notes=None,
):
    """A real ``ChannelWriteResult``, as a connector would return it.

    ``observed_value`` defaults to the value sent on a confirmed write and to
    ``None`` on every other outcome, which is what a connector reports: only a
    re-read that came back has something to show.

    ``outcome`` is coerced to the enum member here as well as in the dataclass,
    so a word this contract does not have fails in the test that wrote it rather
    than reaching the projection as a plausible-looking string.
    """
    outcome = WriteOutcome(outcome)
    if observed_value is _UNSET:
        observed_value = value if outcome == WriteOutcome.CONFIRMED else None
    return ChannelWriteResult(
        channel_address=channel,
        value_written=value,
        outcome=outcome,
        refusal_reason=refusal_reason,
        error_message=error_message,
        observed_value=observed_value,
        alarm_status=alarm_status,
        alarm_severity=alarm_severity,
        notes=notes,
    )


def _outcomes(data):
    """``{channel: outcome}`` from a parsed tool response."""
    return {r["channel"]: r["outcome"] for r in data["summary"]["results"]}


def _get_channel_write():
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    return get_tool_fn(channel_write)


#: The project every test here writes from: the simulator in process, under a
#: write tier that allows.
_CONFIG = (
    "control_system:\n  type: virtual_accelerator\n  writes: allowed\n"
    "  connector:\n    virtual_accelerator:\n      serving: in_process\n"
)


def _arm(tmp_path, monkeypatch, *, switch=True):
    """The login's switch on for the simulator, in a record under *tmp_path*.

    ``switch=False`` writes the same record with every switch off.
    """
    root = tmp_path / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    write_control_context(root, target="va", owned_by=None, switch={"va": "on"} if switch else None)
    return root


@contextmanager
def _approved():
    """Both stamps the approval hook files for an approved call, bound for the body.

    The ask stamp is filed now. The write-approval stamp is keyed by the call's
    operations, which the body passes to the tool, so it is filed when the tool
    asks for it — on the binding the call entered on, as the hook renders it
    just before the call — and then read back by the tool's own reader. The
    approval window itself is ``test_write_target_binding``'s subject.

    A test that already opened a call scope keeps it — its facts are what it
    asserts on — and the stamp is filed for that call's tool-use id.
    """
    from osprey.audit.call import current_tool_use_id
    from osprey.mcp_server.control_system.tools import channel_write as cw

    state = target_state.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    stamps = ApprovalStamps(state)
    read = cw._read_approval_stamp

    def rendered_then_read(operations, confirm):
        binding = cw._read_target_binding()
        target, generation = binding if binding is not None else (None, None)
        stamps.write_approval(operations, confirm, target=target, generation=generation)
        return read(operations, confirm)

    with patch.object(cw, "_read_approval_stamp", rendered_then_read):
        in_scope = current_tool_use_id()
        if in_scope:
            stamps.ask("channel_write", tool_use_id=in_scope)
            yield
            return
        with stamps.approved("channel_write"):
            yield


def _prepare(tmp_path, monkeypatch, config=_CONFIG):
    """Minimal project + server context the tool needs to run, armed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(config)
    _arm(tmp_path, monkeypatch)
    initialize_server_context()


@contextmanager
def _patched(connector, validator=None):
    """Patch the connector factory and the limits validator together.

    Both have to be patched for every call: the tool builds its own validator
    from config, and an unpatched one would read whatever limits database the
    working directory happens to have.
    """
    with (
        patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch(
            "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
            return_value=validator,
        ),
        _approved(),
    ):
        yield


async def _run_batch(results, validator=None, **kwargs):
    """Run the tool over a batch and return (parsed response, connector)."""
    connector = AsyncMock()
    connector.write_multiple_channels.return_value = results
    with _patched(connector, validator):
        fn = _get_channel_write()
        raw = await fn(
            operations=[{"channel": r.channel_address, "value": r.value_written} for r in results],
            **kwargs,
        )
    return extract_response_dict(raw), connector


async def _run_single(result, validator=None, **kwargs):
    """Run the tool over one operation and return (parsed response, connector)."""
    connector = AsyncMock()
    connector.write_channel.return_value = result
    with _patched(connector, validator):
        fn = _get_channel_write()
        raw = await fn(
            operations=[{"channel": result.channel_address, "value": result.value_written}],
            **kwargs,
        )
    return extract_response_dict(raw), connector


async def test_channel_write_success(tmp_path, monkeypatch):
    """A confirmed write returns a success envelope naming the channel."""
    _prepare(tmp_path, monkeypatch)

    data, _ = await _run_single(_make_write_result(channel="TEST:PV", value=42.0))

    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 1
    assert data["summary"]["outcomes"] == {"confirmed": 1}
    assert data["summary"]["results"][0]["channel"] == "TEST:PV"


async def test_channel_write_multiple_operations(tmp_path, monkeypatch):
    """Multiple write operations are all processed."""
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
    ]
    data, _ = await _run_batch(results)

    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 2
    assert data["summary"]["outcomes"] == {"confirmed": 2}


async def test_channel_write_limits_violation(tmp_path, monkeypatch):
    """Write exceeding channel limits (via inline validator) returns structured error."""
    from osprey_connectors.errors import ChannelLimitsViolationError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_CONFIG)
    _arm(tmp_path, monkeypatch)

    from osprey_connectors.control_system._limits_validator import LimitsValidator

    # Spec'd, so the mock carries exactly the entry points the real validator
    # does: the tool asks for `validate_without_step_check` — the checks a
    # caller that has not yet resolved a connector can make — and an unspec'd
    # mock would answer that with a silent stand-in instead.
    mock_validator = MagicMock(spec=LimitsValidator)
    mock_validator.validate_without_step_check.side_effect = ChannelLimitsViolationError(
        channel_address="TEST:PV",
        value=9999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 9999.0 above maximum 100.0",
        min_value=0.0,
        max_value=100.0,
    )

    with (
        patch(
            "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
            return_value=mock_validator,
        ),
        _approved(),
    ):
        fn = _get_channel_write()
        with assert_raises_error(error_type="limits_violation") as _exc_ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 9999.0}])

    data = _exc_ctx["envelope"]
    # Error message includes the channel, value, reason, and allowed range
    assert "TEST:PV" in data["error_message"]
    assert "9999.0" in data["error_message"]
    assert "100.0" in data["error_message"]
    # Structured details include machine-readable limits
    assert "details" in data
    details = data["details"]
    assert details[0]["channel"] == "TEST:PV"
    assert details[0]["min_value"] == 0.0
    assert details[0]["max_value"] == 100.0
    assert details[0]["violation_type"] == "MAX_EXCEEDED"
    # Suggestions are actionable guidance, not the violation banner
    assert any("Do NOT" in s for s in data["suggestions"])


async def test_channel_write_leaves_max_step_to_the_connector(tmp_path, monkeypatch):
    """A max_step channel is not refused by the tool's own pre-check.

    The step needs a fresh read of the channel over the client the write goes
    through, and the connector is resolved after this check on purpose. The
    connector makes the step check itself; refusing here for want of a reader
    would take max_step off the mediated write path rather than enforcing it.
    """
    from osprey_connectors.control_system._limits_validator import (
        ChannelLimitsConfig,
        LimitsValidator,
    )

    _prepare(tmp_path, monkeypatch)

    validator = LimitsValidator(
        {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV",
                min_value=0.0,
                max_value=100.0,
                max_step=1.0,
                writable=True,
            )
        },
        {"mode": "exclusive"},
    )

    data, connector = await _run_single(
        _make_write_result(channel="TEST:PV", value=99.0), validator=validator
    )

    assert data["summary"]["outcomes"] == {"confirmed": 1}
    connector.write_channel.assert_awaited_once()


async def test_channel_write_still_denies_a_bound_violation_on_a_max_step_channel(
    tmp_path, monkeypatch
):
    """Deferring the step check defers nothing else."""
    from osprey_connectors.control_system._limits_validator import (
        ChannelLimitsConfig,
        LimitsValidator,
    )

    _prepare(tmp_path, monkeypatch)

    validator = LimitsValidator(
        {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV",
                min_value=0.0,
                max_value=100.0,
                max_step=1.0,
                writable=True,
            )
        },
        {"mode": "exclusive"},
    )

    connector = AsyncMock()
    with _patched(connector, validator):
        fn = _get_channel_write()
        with assert_raises_error(error_type="limits_violation") as _exc_ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 9999.0}])

    assert _exc_ctx["envelope"]["details"][0]["violation_type"] == "MAX_EXCEEDED"
    connector.write_channel.assert_not_awaited()


async def test_channel_write_connection_error(tmp_path, monkeypatch):
    """Connection error during write returns standard error format."""
    _prepare(tmp_path, monkeypatch)

    mock_connector = AsyncMock()
    mock_connector.write_channel.side_effect = ConnectionError("IOC unreachable")

    with _patched(mock_connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="connection_error") as _exc_ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 1.0}])

    data = _exc_ctx["envelope"]
    assert "error_message" in data
    assert "suggestions" in data


async def test_channel_write_connector_limits_violation(tmp_path, monkeypatch):
    """ChannelLimitsViolationError from the connector stays a limits_violation."""
    from osprey_connectors.errors import ChannelLimitsViolationError

    _prepare(tmp_path, monkeypatch)

    mock_connector = AsyncMock()
    mock_connector.write_channel.side_effect = ChannelLimitsViolationError(
        channel_address="TEST:PV",
        value=999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 999.0 above maximum 100.0",
        min_value=0.0,
        max_value=100.0,
    )

    with _patched(mock_connector):
        fn = _get_channel_write()
        with assert_raises_error() as _exc_ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 999.0}])

    data = _exc_ctx["envelope"]
    assert data["error_type"] == "limits_violation", (
        f"Expected limits_violation but got {data['error_type']} — "
        "ChannelLimitsViolationError from connector must not be misclassified as internal_error"
    )
    # Connector-level catch should also provide structured details
    assert "100.0" in data["error_message"]
    assert "details" in data
    assert data["details"]["channel"] == "TEST:PV"
    assert data["details"]["max_value"] == 100.0


async def test_channel_write_empty_operations(tmp_path, monkeypatch):
    """Empty operations list returns validation error."""
    _prepare(tmp_path, monkeypatch)

    fn = _get_channel_write()
    with _approved(), assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(operations=[])

    _exc_ctx["envelope"]


async def test_channel_write_missing_channel_key(tmp_path, monkeypatch):
    """Operation missing 'channel' key returns validation error."""
    _prepare(tmp_path, monkeypatch)

    with (
        patch(
            "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
        _approved(),
    ):
        fn = _get_channel_write()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(operations=[{"value": 42.0}])

    _exc_ctx["envelope"]


# ---------------------------------------------------------------------------
# outcome — the closed-set word the agent keys on
# ---------------------------------------------------------------------------


#: (outcome, kwargs for the subject result). One row per documented word, in the
#: documented order, pinned 1:1 against ``EXPECTED_OUTCOMES`` by the meta-test
#: below. The word is the connector's; the tool's job is to carry it through
#: unchanged, so each row also carries the fields that word travels with.
_STATE_CASES = [
    (
        "refused",
        {
            "outcome": "refused",
            "refusal_reason": "WRITES_DISABLED",
            "error_message": "Write to 'PV:SUBJECT' refused: writes are disabled.",
        },
    ),
    ("failed", {"outcome": "failed", "error_message": "caput failed: timeout"}),
    # A MAJOR alarm does not downgrade a confirmed write: the channel holds what
    # was sent, and the alarm is reported beside it.
    (
        "confirmed",
        {
            "outcome": "confirmed",
            "observed_value": 2.0,
            "alarm_status": "HIHI",
            "alarm_severity": 2,
        },
    ),
    ("mismatch", {"outcome": "mismatch", "observed_value": 1.9}),
    ("unconfirmed", {"outcome": "unconfirmed", "error_message": "readback raised: timeout"}),
    ("unrequested", {"outcome": "unrequested"}),
]


@pytest.mark.parametrize(
    "expected_outcome,result_kwargs", _STATE_CASES, ids=[c[0] for c in _STATE_CASES]
)
async def test_outcome_reaches_the_shipped_summary(
    tmp_path, monkeypatch, expected_outcome, result_kwargs
):
    """Each outcome word arrives in the shipped summary as the connector set it.

    Run as a two-op batch with a confirmed first write so the refused and failed
    subjects do not trip the all-negative top-level errors, which are a separate
    contract.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:OK", value=1.0),
        _make_write_result(channel="PV:SUBJECT", value=2.0, **result_kwargs),
    ]
    data, _ = await _run_batch(results)

    assert _outcomes(data)["PV:SUBJECT"] == expected_outcome


def test_state_cases_cover_every_documented_outcome():
    """The parametrisation above exercises the whole closed set, in order.

    Pinned against the enum the connectors set as well as the list documented
    here: a seventh word added to one and not the other would otherwise ship
    with nothing in this file to notice it.
    """
    assert [case[0] for case in _STATE_CASES] == EXPECTED_OUTCOMES
    assert [member.value for member in WriteOutcome] == EXPECTED_OUTCOMES


async def test_result_entry_carries_the_documented_fields(tmp_path, monkeypatch):
    """One result entry, nine keys — the whole agent-visible projection."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(
        channel="TEST:PV",
        value=42.0,
        outcome="mismatch",
        observed_value=41.5,
        alarm_status="HIGH",
        alarm_severity=1,
        notes="observed 41.5, sent 42.0",
    )
    data, _ = await _run_single(result)

    entry = data["summary"]["results"][0]
    assert set(entry) == {
        "channel",
        "value",
        "outcome",
        "refusal_reason",
        "error",
        "observed_value",
        "alarm_status",
        "alarm_severity",
        "notes",
    }
    assert entry["channel"] == "TEST:PV"
    assert entry["value"] == 42.0
    assert entry["outcome"] == "mismatch"
    assert entry["observed_value"] == 41.5
    assert entry["alarm_status"] == "HIGH"
    assert entry["alarm_severity"] == 1
    assert entry["notes"] == "observed 41.5, sent 42.0"
    # Absent fields are reported as null rather than omitted, so a consumer can
    # tell "not reported" from a value.
    assert entry["refusal_reason"] is None
    assert entry["error"] is None


async def test_confirmed_result_projects_its_observed_value(tmp_path, monkeypatch):
    """Whatever the confirming re-read returned reaches the agent."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(
        channel="TEST:PV", value=42.0, outcome="confirmed", observed_value=42.0
    )
    data, _ = await _run_single(result)

    entry = data["summary"]["results"][0]
    assert entry["outcome"] == "confirmed"
    assert entry["observed_value"] == 42.0


async def test_healthy_alarm_severity_zero_survives_the_projection(tmp_path, monkeypatch):
    """A REPORTED healthy severity of 0 is not collapsed into "not reported".

    0 is a channel saying it is fine; None is a channel that said nothing. An
    agent that cannot tell them apart reports an alarm state it was never given.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:ZERO", value=1.0, alarm_status="NO_ALARM", alarm_severity=0),
        _make_write_result(channel="PV:SILENT", value=2.0),
    ]
    data, _ = await _run_batch(results)

    entries = {r["channel"]: r for r in data["summary"]["results"]}
    assert entries["PV:ZERO"]["alarm_severity"] == 0
    assert entries["PV:ZERO"]["alarm_status"] == "NO_ALARM"
    assert entries["PV:SILENT"]["alarm_severity"] is None
    assert entries["PV:SILENT"]["alarm_status"] is None


async def test_outcome_ignores_notes_text(tmp_path, monkeypatch):
    """Notes are display-only: rewording them cannot change the outcome.

    The narration bug this whole contract exists to close came from deriving
    meaning out of prose, so the word an agent keys on must be untouched by it.
    """
    _prepare(tmp_path, monkeypatch)

    outcomes = []
    for note in ("", "Readback matched.", "MISMATCH: readback failed, value not applied!"):
        results = [
            _make_write_result(channel="PV:A", value=1.0, outcome="confirmed", notes=note),
            _make_write_result(channel="PV:B", value=2.0),
        ]
        data, _ = await _run_batch(results)
        outcomes.append(_outcomes(data)["PV:A"])

    assert outcomes == ["confirmed", "confirmed", "confirmed"]


async def test_mixed_batch_reports_one_outcome_per_channel(tmp_path, monkeypatch):
    """A batch of unlike outcomes reports each channel's own word and counts."""
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:OK", value=1.0),
        _make_write_result(
            channel="PV:MISMATCH", value=2.0, outcome="mismatch", observed_value=0.0
        ),
        _make_write_result(
            channel="PV:REFUSED",
            value=3.0,
            outcome="refused",
            refusal_reason="WRITES_DISABLED",
            error_message="Write to 'PV:REFUSED' refused by the reference monitor (WRITES_DISABLED)",
        ),
        _make_write_result(channel="PV:NOCHECK", value=4.0, outcome="unrequested"),
    ]
    data, _ = await _run_batch(results)

    assert _outcomes(data) == {
        "PV:OK": "confirmed",
        "PV:MISMATCH": "mismatch",
        "PV:REFUSED": "refused",
        "PV:NOCHECK": "unrequested",
    }
    assert data["summary"]["outcomes"] == {
        "confirmed": 1,
        "mismatch": 1,
        "refused": 1,
        "unrequested": 1,
    }
    assert data["summary"]["total_writes"] == 4


# ---------------------------------------------------------------------------
# the envelope — what returns and what raises
# ---------------------------------------------------------------------------


async def test_lone_mismatch_returns_rather_than_raising(tmp_path, monkeypatch):
    """A single write that came back different is REPORTED, not raised.

    This is the asymmetry the contract is built on: the tool tells the agent
    what the channel holds and lets it tell the operator, while the Python path
    raises on the same result. A raise here would cost the agent the observed
    value at exactly the moment it matters most.
    """
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(
        channel="TEST:PV", value=42.0, outcome="mismatch", observed_value=41.0
    )
    data, _ = await _run_single(result)

    assert data["status"] == "success"
    entry = data["summary"]["results"][0]
    assert entry["outcome"] == "mismatch"
    assert entry["observed_value"] == 41.0
    assert data["summary"]["outcomes"] == {"mismatch": 1}


async def test_an_enum_member_projects_to_its_word(tmp_path, monkeypatch):
    """The production type, not a look-alike string, reaches the projection.

    Connectors set ``outcome`` to a :class:`WriteOutcome` member. The tool
    renders it with ``str()``, which is the word only because the enum is a
    ``StrEnum`` — swap that base class and every envelope would start shipping
    ``WriteOutcome.MISMATCH``. Feeding the member itself is what pins it.
    """
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(
        channel="TEST:PV", value=42.0, outcome=WriteOutcome.MISMATCH, observed_value=41.0
    )
    assert result.outcome is WriteOutcome.MISMATCH
    data, _ = await _run_single(result)

    assert data["summary"]["results"][0]["outcome"] == "mismatch"
    assert data["summary"]["outcomes"] == {"mismatch": 1}


async def test_a_connector_returning_too_few_results_fails_loudly(tmp_path, monkeypatch):
    """A dropped row is a write whose fate nobody reports, so refuse the report.

    The envelope names exactly the rows the connector handed back. If one is
    missing the response still looks complete — same shape, same status — while
    a channel the operator approved simply goes unmentioned. On the
    hardware-write surface that is worse than a failure, so the tool names the
    connector and raises instead of shipping it.
    """
    _prepare(tmp_path, monkeypatch)

    connector = AsyncMock()
    connector.write_multiple_channels.return_value = [_make_write_result(channel="PV:A", value=1.0)]
    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="internal_error") as ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    message = ctx["envelope"]["error_message"]
    assert "1 write result(s) for 2 operation(s)" in message
    assert "unreported" in message


async def test_a_connector_returning_too_many_results_fails_loudly(tmp_path, monkeypatch):
    """An extra row is a write nobody asked for — the same loss of correspondence.

    Only the batch path can drift: a single write wraps the one result the
    connector returned, so its count is one by construction.
    """
    _prepare(tmp_path, monkeypatch)

    connector = AsyncMock()
    connector.write_multiple_channels.return_value = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
        _make_write_result(channel="PV:C", value=3.0),
    ]
    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="internal_error") as ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    assert "3 write result(s) for 2 operation(s)" in ctx["envelope"]["error_message"]


@pytest.mark.parametrize("outcome", ["unconfirmed", "unrequested"])
async def test_lone_unconfirmed_write_returns(tmp_path, monkeypatch, outcome):
    """An unconfirmed or unchecked write reached the channel, so it reports."""
    _prepare(tmp_path, monkeypatch)

    message = "Write to 'TEST:PV' not acknowledged" if outcome == "unconfirmed" else None
    result = _make_write_result(
        channel="TEST:PV", value=42.0, outcome=outcome, error_message=message
    )
    data, _ = await _run_single(result)

    assert data["status"] == "success"
    assert data["summary"]["results"][0]["outcome"] == outcome


async def test_all_refused_is_write_refused(tmp_path, monkeypatch):
    """All-refused batch yields a typed write_refused envelope.

    The connector refused every write (nothing was sent to the control system).
    The tool raises the nothing-answered carrier with the refusal as its cause,
    so the error handler answers with the refusal's own write_refused envelope,
    the rows added under details.results.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(
            channel="PV:A",
            value=1.0,
            outcome="refused",
            refusal_reason="WRITES_DISABLED",
            error_message="Write to 'PV:A' refused: writes are disabled.",
        ),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            outcome="refused",
            refusal_reason="WRITES_DISABLED",
            error_message="Write to 'PV:B' refused: writes are disabled.",
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    with _patched(mock_connector):
        fn = _get_channel_write()
        with assert_raises_error() as _exc_ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    data = _exc_ctx["envelope"]
    assert data["error_type"] == "write_refused", (
        f"Expected write_refused but got {data['error_type']} — an all-refused "
        "batch is a policy refusal, not an internal_error"
    )
    # Both refused channels are named in the summary message.
    assert "PV:A" in data["error_message"]
    assert "PV:B" in data["error_message"]
    # Structured details carry the refusal reason discriminator.
    assert data["details"]["reason"] == "WRITES_DISABLED"


async def test_control_system_refusal_names_the_control_system(tmp_path, monkeypatch):
    """Who refused decides the wording of an all-refused envelope."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(
        channel="PV:A",
        value=1.0,
        outcome="refused",
        refusal_reason="CONTROL_SYSTEM_REFUSED",
        error_message="Write to 'PV:A' refused by the control system.",
    )
    connector = AsyncMock()
    connector.write_channel.return_value = result

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="write_refused") as _exc_ctx:
            await fn(operations=[{"channel": "PV:A", "value": 1.0}])

    assert "the control system" in _exc_ctx["envelope"]["error_message"]


async def test_a_single_refusal_envelope_says_what_the_refusal_said(tmp_path, monkeypatch):
    """The raised envelope replaces the results, so it must carry their reason.

    A refusal's ``error`` field is the only place the monitor names WHICH
    posture refused and where it lifts — the deployment's read-only setting for
    one control target and the header chip that set it, or the config key for a
    deployment refusal. Naming the channel and stopping there is a dead end,
    and the single-channel write is exactly the common case where the agent
    never sees the per-result field at all.
    """
    _prepare(tmp_path, monkeypatch)

    refusal = (
        "Write to 'PV:A' blocked: writes are off for the 'standin' control target "
        "— turned off from the control-target chip in the header; applies "
        "deployment-wide."
    )
    result = _make_write_result(
        channel="PV:A",
        value=1.0,
        outcome="refused",
        refusal_reason="WRITES_DISABLED",
        error_message=refusal,
    )
    connector = AsyncMock()
    connector.write_channel.return_value = result

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="write_refused") as _exc_ctx:
            await fn(operations=[{"channel": "PV:A", "value": 1.0}])

    message = _exc_ctx["envelope"]["error_message"]
    assert "the reference monitor" in message
    assert "control-target chip in the header" in message
    assert "standin" in message


def _row_facts(rows):
    return [(row["channel"], row["outcome"], row["refusal_reason"], row["error"]) for row in rows]


def _result_facts(results):
    return [(r.channel_address, str(r.outcome), r.refusal_reason, r.error_message) for r in results]


async def test_all_failed_is_nothing_answered(tmp_path, monkeypatch):
    """Every write was sent and none taken: nothing answered, each row's text kept.

    An I/O failure is not a policy refusal, so it is not write_refused; the
    rows carry what the control system said about each write.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(
            channel="PV:A", value=1.0, outcome="failed", error_message="caput failed: timeout"
        ),
        _make_write_result(
            channel="PV:B", value=2.0, outcome="failed", error_message="caput failed: no connection"
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    with _patched(mock_connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="nothing_answered") as _exc_ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    details = _exc_ctx["envelope"]["details"]
    assert _row_facts(details["results"]) == _result_facts(results)
    assert "missing" not in details


async def test_refused_and_failed_batch_is_nothing_answered(tmp_path, monkeypatch):
    """Nothing reached a channel, but one write was attempted: not a pure refusal.

    Calling a batch that contains a real I/O failure a policy refusal would send
    the operator to the limits database for a problem that lives on the wire;
    both rows say what happened to their own write.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(
            channel="PV:A",
            value=1.0,
            outcome="refused",
            refusal_reason="WRITES_DISABLED",
            error_message="Write to 'PV:A' refused: writes are disabled.",
        ),
        _make_write_result(
            channel="PV:B", value=2.0, outcome="failed", error_message="caput failed: timeout"
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    with _patched(mock_connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="nothing_answered") as _exc_ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    details = _exc_ctx["envelope"]["details"]
    assert _row_facts(details["results"]) == _result_facts(results)
    assert "missing" not in details


async def test_partial_refusal_reports_per_op(tmp_path, monkeypatch):
    """One refusal beside one confirmed write returns, and reports both.

    A value did reach a channel, so raising would throw away the report of the
    write that landed.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            outcome="refused",
            refusal_reason="WRITES_DISABLED",
            error_message="Write to 'PV:B' refused: writes are disabled.",
        ),
    ]
    data, _ = await _run_batch(results)

    assert data["status"] == "success"
    assert data["summary"]["outcomes"] == {"confirmed": 1, "refused": 1}
    by_channel = {r["channel"]: r for r in data["summary"]["results"]}
    assert by_channel["PV:B"]["refusal_reason"] == "WRITES_DISABLED"
    assert by_channel["PV:B"]["error"] is not None
    assert by_channel["PV:A"]["refusal_reason"] is None


async def test_executed_channels_name_only_what_reached_a_channel(tmp_path, monkeypatch):
    """The activity highlight names every write that got a value onto a channel.

    ``mismatch``, ``unconfirmed`` and ``unrequested`` all put a value on the
    wire — leaving them out would tell the operator less than happened —
    while ``refused`` and ``failed`` did not.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:CONFIRMED", value=1.0),
        _make_write_result(
            channel="PV:MISMATCH", value=2.0, outcome="mismatch", observed_value=0.0
        ),
        _make_write_result(
            channel="PV:UNCONFIRMED",
            value=3.0,
            outcome="unconfirmed",
            error_message="Write to 'PV:UNCONFIRMED' sent; the confirming re-read timed out",
        ),
        _make_write_result(channel="PV:UNREQUESTED", value=4.0, outcome="unrequested"),
        _make_write_result(
            channel="PV:REFUSED",
            value=5.0,
            outcome="refused",
            refusal_reason="WRITES_DISABLED",
            error_message="Write to 'PV:REFUSED' refused by the reference monitor (WRITES_DISABLED)",
        ),
        _make_write_result(
            channel="PV:FAILED", value=6.0, outcome="failed", error_message="caput failed"
        ),
    ]

    with patch(
        "osprey.mcp_server.control_system.tools.channel_write.notify_agent_activity_async",
        new_callable=AsyncMock,
    ) as notify:
        await _run_batch(results)

    detail = notify.call_args.kwargs["detail"]
    assert detail == "PV:CONFIRMED, PV:MISMATCH, PV:UNCONFIRMED, PV:UNREQUESTED"


async def test_no_activity_highlight_when_nothing_executed(tmp_path, monkeypatch):
    """An all-negative call raises before it can claim a write happened."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(
        channel="PV:A",
        value=1.0,
        outcome="refused",
        refusal_reason="WRITES_DISABLED",
        error_message="Write to 'PV:A' refused by the reference monitor (WRITES_DISABLED)",
    )
    connector = AsyncMock()
    connector.write_channel.return_value = result

    with (
        patch(
            "osprey.mcp_server.control_system.tools.channel_write.notify_agent_activity_async",
            new_callable=AsyncMock,
        ) as notify,
        _patched(connector),
    ):
        fn = _get_channel_write()
        with assert_raises_error(error_type="write_refused"):
            await fn(operations=[{"channel": "PV:A", "value": 1.0}])

    notify.assert_not_called()


# ---------------------------------------------------------------------------
# confirm — the caller's opinion, or none at all
# ---------------------------------------------------------------------------


async def test_omitted_confirm_leaves_the_keyword_absent(tmp_path, monkeypatch):
    """Omission is a sentinel: the keyword is left off, not passed as None.

    Resolution belongs to the connector, per channel. Forwarding None would
    override a deployment's own per-channel setting with "no opinion".
    """
    _prepare(tmp_path, monkeypatch)

    _, connector = await _run_single(_make_write_result(channel="TEST:PV", value=42.0))

    assert "confirm" not in connector.write_channel.call_args.kwargs


async def test_explicit_confirm_false_is_forwarded(tmp_path, monkeypatch):
    """``confirm=False`` is a decision and must cross, not be read as omission.

    The guard is ``if confirm is not None``; an ``if confirm:`` would swallow
    exactly this call and silently confirm a write the operator asked not to.
    """
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0, outcome="unrequested")
    _, connector = await _run_single(result, confirm=False)

    assert connector.write_channel.call_args.kwargs["confirm"] is False


async def test_explicit_confirm_true_is_forwarded(tmp_path, monkeypatch):
    """A caller asking for confirmation gets it forwarded to the connector."""
    _prepare(tmp_path, monkeypatch)

    _, connector = await _run_single(
        _make_write_result(channel="TEST:PV", value=42.0), confirm=True
    )

    assert connector.write_channel.call_args.kwargs["confirm"] is True


async def test_confirm_is_forwarded_on_a_batch(tmp_path, monkeypatch):
    """One confirm setting applies to every channel in the batch."""
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0, outcome="unrequested"),
        _make_write_result(channel="PV:B", value=2.0, outcome="unrequested"),
    ]
    _, connector = await _run_batch(results, confirm=False)

    assert connector.write_multiple_channels.call_args.kwargs["confirm"] is False


async def test_omitted_confirm_leaves_the_batch_keyword_absent(tmp_path, monkeypatch):
    """With no opinion named, each channel resolves its own setting."""
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
    ]
    _, connector = await _run_batch(results)

    assert "confirm" not in connector.write_multiple_channels.call_args.kwargs


async def test_access_details_confirm_is_null_when_omitted(tmp_path, monkeypatch):
    """access_details reports what the caller asked for, not what was resolved."""
    _prepare(tmp_path, monkeypatch)

    data, _ = await _run_single(_make_write_result(channel="TEST:PV", value=42.0))

    assert data["access_details"]["confirm"] is None


async def test_access_details_confirm_echoes_an_explicit_request(tmp_path, monkeypatch):
    """An explicit setting is echoed back verbatim, False included."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0, outcome="unrequested")
    data, _ = await _run_single(result, confirm=False)

    assert data["access_details"]["confirm"] is False


# ---------------------------------------------------------------------------
# observed_value — bounded like a read
# ---------------------------------------------------------------------------


async def test_oversize_observed_value_is_summarised(tmp_path, monkeypatch):
    """A waveform readback too large to inline arrives as a bounded summary.

    A confirming re-read of a waveform channel returns the whole waveform. The
    write tool is bound by the same inline budget as the read tool, and reports
    the withheld value the same way, so an agent meets one shape for "too big
    to show you".
    """
    np = pytest.importorskip("numpy")
    _prepare(
        tmp_path,
        monkeypatch,
        config=_CONFIG + "  read_inline_max_elements: 4\n",
    )

    result = _make_write_result(
        channel="TEST:WAVEFORM",
        value=1.0,
        outcome="mismatch",
        observed_value=np.arange(10, dtype=float),
    )
    data, _ = await _run_single(result)

    observed = data["summary"]["results"][0]["observed_value"]
    assert observed["value_withheld"] is True
    assert observed["shape"] == [10]
    assert observed["element_count"] == 10
    assert observed["dtype"] == "float64"
    assert observed["min"] == 0.0
    assert observed["max"] == 9.0


async def test_observed_value_within_the_budget_stays_inline(tmp_path, monkeypatch):
    """A short waveform is reported as the values themselves."""
    np = pytest.importorskip("numpy")
    _prepare(
        tmp_path,
        monkeypatch,
        config=_CONFIG + "  read_inline_max_elements: 4\n",
    )

    result = _make_write_result(
        channel="TEST:WAVEFORM",
        value=1.0,
        outcome="mismatch",
        observed_value=np.array([1.0, 2.0, 3.0]),
    )
    data, _ = await _run_single(result)

    assert data["summary"]["results"][0]["observed_value"] == [1.0, 2.0, 3.0]


async def test_long_string_observed_value_stays_inline(tmp_path, monkeypatch):
    """A long string is one channel value, not a thousand elements.

    Summarising it would lose the only thing it says, so str/bytes are inline
    whatever their length — the same rule the read tool applies.
    """
    _prepare(
        tmp_path,
        monkeypatch,
        config=_CONFIG + "  read_inline_max_elements: 4\n",
    )

    reading = "OPEN" * 50
    result = _make_write_result(
        channel="TEST:STATE", value="CLOSED", outcome="mismatch", observed_value=reading
    )
    data, _ = await _run_single(result)

    assert data["summary"]["results"][0]["observed_value"] == reading


# ---------------------------------------------------------------------------
# the key the rules name and the key the tool emits are one string
# ---------------------------------------------------------------------------


async def test_emitted_key_is_the_constant_the_rules_name(tmp_path, monkeypatch):
    """The tool emits the exact key its constant names.

    Two files, one word: the generated safety rules tell the agent which key to
    read, and if either side is renamed on its own the agent is told to key on
    something the payload does not contain — silent, because the tool still
    returns a valid result and the agent simply has nothing to go on. The half
    of this pin that renders the rules templates lives with those templates.
    """
    from osprey.mcp_server.control_system.tools.channel_write import OUTCOME_KEY

    _prepare(tmp_path, monkeypatch)

    assert OUTCOME_KEY == "outcome"

    data, _ = await _run_single(_make_write_result(channel="TEST:PV", value=42.0))
    assert OUTCOME_KEY in data["summary"]["results"][0]


# ---------------------------------------------------------------------------
# the limits posture is the one the recorded control target runs under
# ---------------------------------------------------------------------------

#: A deployment that relaxed unlisted channels for its simulator alone: the
#: deployment-wide block refuses them, and only the ``virtual_accelerator``
#: block allows them, so two postures genuinely coexist in one config. The
#: baseline type is per-test: the serving target must be the baseline here,
#: because a deployment serving anything else is switch-capable by
#: construction and runs the connector-host path, where the host republishes
#: its own target record over the one this fixture writes.
_SPLIT_POSTURE_CONFIG = """\
control_system:
  type: {cs_type}
  writes: allowed
  limits_checking:
    enabled: true
    mode: exclusive
    database_path: {db_path}
  connector:
    virtual_accelerator:
      limits_checking:
        enabled: true
        mode: optional
"""

#: Display metadata as the server's single writer records it — irrelevant to
#: the posture, written anyway so the fixture record is the shape a reader meets.
_SPLIT_POSTURE_TARGETS_META = {
    "live": {"label": "LIVE MACHINE", "endpoint": "gateway.example.com:5064", "real_machine": True},
    "va": {
        "label": "virtual accelerator (simulation)",
        "endpoint": "localhost:5074",
        "real_machine": False,
    },
}


def _prepare_split_posture(tmp_path, monkeypatch, target):
    """A split-posture deployment serving *target*, with real files on disk.

    Nothing here is a stand-in for the thing under test: the config is loaded
    off disk, the limits database is a real file, and the target arrives the way
    the tool actually learns it — from the deployment's control-context record.
    The one rebinding is the state root, so these files land in a directory this
    test owns.
    """
    db_path = tmp_path / "limits.json"
    db_path.write_text(
        json.dumps({"LISTED:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}})
    )
    monkeypatch.chdir(tmp_path)
    cs_type = "virtual_accelerator" if target == "va" else "epics"
    config_text = _SPLIT_POSTURE_CONFIG.format(cs_type=cs_type, db_path=db_path)
    if cs_type == "virtual_accelerator":
        # A simulated machine may not pair with the synthesizing mock archiver
        # (the invented-history startup guard); name the store the preset deploys.
        config_text += "archiver:\n  type: mongodb_archiver\n"
    (tmp_path / "config.yml").write_text(config_text)
    root = tmp_path / "var" / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    # The target the tool builds its validator from is the deployment's, read
    # off the control-context record; the report carries the display metadata.
    # Its switch is on, so the write verdict lets the call reach the limits.
    write_control_context(root, target=target, generation=0, switch={target: "on"})
    target_state.write_server_record(_SPLIT_POSTURE_TARGETS_META)
    initialize_server_context()


@contextmanager
def _real_validator(connector):
    """Serve *connector* while the tool builds its own validator from config.

    The opposite of :func:`_patched`: here the validator is exactly the one the
    tool constructs, because which posture it constructs it from is the whole
    question.
    """
    with patch(
        "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=connector,
    ):
        yield


async def test_unlisted_write_passes_on_a_target_whose_block_allows_it(tmp_path, monkeypatch):
    """Serving ``va``, the tool reads the simulator's own permissive block.

    The deployment-wide key still refuses unlisted channels; reading it here
    would refuse a write the operator relaxed for the simulator on purpose.
    """
    _prepare_split_posture(tmp_path, monkeypatch, "va")

    connector = AsyncMock()
    connector.write_channel.return_value = _make_write_result(channel="UNLISTED:PV", value=1.0)

    with _real_validator(connector), _approved():
        fn = _get_channel_write()
        raw = await fn(operations=[{"channel": "UNLISTED:PV", "value": 1.0}])

    data = extract_response_dict(raw)
    assert data["status"] == "success"
    assert _outcomes(data) == {"UNLISTED:PV": "confirmed"}
    assert connector.write_channel.await_count == 1


async def test_unlisted_write_is_refused_on_a_target_the_deployment_block_governs(
    tmp_path, monkeypatch
):
    """Serving ``live``, the same write is refused and the refusal names the key.

    ``live`` resolves to ``epics``, which wrote no block, so the deployment-wide
    key is the one that answered — and it is the line an operator would have to
    edit, so it is the line the refusal quotes. The connector is never reached.
    """
    _prepare_split_posture(tmp_path, monkeypatch, "live")

    connector = AsyncMock()
    connector.write_channel.return_value = _make_write_result(channel="UNLISTED:PV", value=1.0)

    with _real_validator(connector), _approved():
        fn = _get_channel_write()
        with assert_raises_error(error_type="limits_violation") as _exc_ctx:
            await fn(operations=[{"channel": "UNLISTED:PV", "value": 1.0}])

    data = _exc_ctx["envelope"]
    assert data["details"][0]["violation_type"] == "UNLISTED_CHANNEL"
    assert "control_system.limits_checking.mode" in data["details"][0]["reason"], data["details"][
        0
    ]["reason"]
    assert connector.write_channel.await_count == 0


# ---------------------------------------------------------------------------
# The tool_call record: the limits verdict and the old values
# ---------------------------------------------------------------------------


async def test_the_limits_verdict_is_noted(tmp_path, monkeypatch):
    from osprey.audit.call import call_scope
    from osprey_connectors.control_system._limits_validator import LimitsValidator
    from osprey_connectors.errors import ChannelLimitsViolationError

    _prepare(tmp_path, monkeypatch)

    passing = MagicMock(spec=LimitsValidator)
    with call_scope("toolu_ok", None) as call:
        await _run_single(_make_write_result(channel="TEST:PV", value=1.0), validator=passing)
    assert call.facts["limits"] == {"verdict": "passed"}

    with call_scope("toolu_none", None) as call:
        await _run_single(_make_write_result(channel="TEST:PV", value=1.0), validator=None)
    assert call.facts["limits"] == {"verdict": "not_checked"}

    violating = MagicMock(spec=LimitsValidator)
    violating.validate_without_step_check.side_effect = ChannelLimitsViolationError(
        channel_address="TEST:PV",
        value=9999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 9999.0 above maximum 100.0",
        min_value=0.0,
        max_value=100.0,
    )
    with call_scope("toolu_bad", None) as call:
        with assert_raises_error(error_type="limits_violation"):
            await _run_single(_make_write_result(channel="TEST:PV", value=9999.0), violating)
    assert call.facts["limits"]["verdict"] == "violated"
    assert call.facts["limits"]["violations"][0]["channel"] == "TEST:PV"


async def test_old_values_are_read_before_the_window_check(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from osprey.audit.call import call_scope
    from osprey.mcp_server.control_system.tools import channel_write as cw
    from osprey_connectors.control_system._base import ChannelValue

    _prepare(tmp_path, monkeypatch)
    order: list[str] = []
    real_window = cw._check_execution_window

    def window(binding):
        order.append("window")
        return real_window(binding)

    monkeypatch.setattr(cw, "_check_execution_window", window)

    connector = AsyncMock()
    stamp = datetime(2026, 1, 1, tzinfo=UTC)

    async def read(channels, **_kwargs):
        order.append("read")
        return {channel: ChannelValue(value=3.5, timestamp=stamp) for channel in channels}

    connector.read_multiple_channels.side_effect = read
    connector.write_channel.return_value = _make_write_result(channel="TEST:PV", value=4.0)
    with call_scope("toolu_old", None) as call:
        with _patched(connector):
            await _get_channel_write()(operations=[{"channel": "TEST:PV", "value": 4.0}])

    assert order == ["read", "window"]
    assert call.facts["old_values"] == {"TEST:PV": {"value": 3.5, "timestamp": stamp.isoformat()}}


async def test_an_unreadable_old_value_never_blocks_the_write(tmp_path, monkeypatch):
    from osprey.audit.call import call_scope

    _prepare(tmp_path, monkeypatch)
    connector = AsyncMock()
    connector.read_multiple_channels.side_effect = ChannelUnreachableError("TEST:PV", "dead IOC")
    connector.write_channel.return_value = _make_write_result(channel="TEST:PV", value=4.0)
    with call_scope("toolu_dead", None) as call:
        with _patched(connector):
            raw = await _get_channel_write()(operations=[{"channel": "TEST:PV", "value": 4.0}])

    assert extract_response_dict(raw)["status"] == "success"
    connector.write_channel.assert_awaited_once()
    assert call.facts["old_values"] == {"TEST:PV": {"error": "ChannelUnreachableError"}}


async def test_the_verdict_and_old_values_need_no_setting(tmp_path, monkeypatch):
    """With no ``audit.tool_call`` key anywhere, a write still notes both facts."""
    from osprey.audit import tool_call
    from osprey.audit.call import call_scope

    assert not hasattr(tool_call, "settings")
    _prepare(tmp_path, monkeypatch)
    with call_scope("toolu_nokey", None) as call:
        _data, connector = await _run_single(_make_write_result(channel="TEST:PV", value=1.0))

    connector.read_multiple_channels.assert_awaited_once()
    assert "TEST:PV" in call.facts["old_values"]
    assert call.facts["limits"]["verdict"] in {"passed", "not_checked"}


# ---------------------------------------------------------------------------
# Attribution stamps: who the write went out as, on both record surfaces
# ---------------------------------------------------------------------------


async def _stamped_facts(tmp_path, monkeypatch, owner: str | None = None) -> dict:
    """The facts one approved write notes, with ``OSPREY_CONTROL_OWNER`` set to *owner*.

    The variable is set after the deployment is armed, which clears it; ``None``
    leaves it unset.
    """
    from osprey.audit.call import call_scope

    _prepare(tmp_path, monkeypatch)
    if owner is not None:
        monkeypatch.setenv("OSPREY_CONTROL_OWNER", owner)
    with call_scope("toolu_stamp", None) as call:
        data, connector = await _run_single(_make_write_result(channel="TEST:PV", value=1.0))
    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()
    return call.facts


async def test_the_record_carries_the_writing_process_account_and_host(tmp_path, monkeypatch):
    import os
    import pwd
    import socket

    facts = await _stamped_facts(tmp_path, monkeypatch)

    assert facts["ca_user"] == pwd.getpwuid(os.getuid()).pw_name
    assert facts["ca_host"] == socket.gethostname()


async def test_the_dispatch_owner_is_noted_from_the_env_var(tmp_path, monkeypatch):
    # The owner is the login the switch was armed under: any other is refused
    # by the write verdict as foreign before anything is noted.
    facts = await _stamped_facts(tmp_path, monkeypatch, owner=FIXTURE_IDENTITY)

    assert facts["owner"] == FIXTURE_IDENTITY


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
async def test_no_owner_is_noted_without_the_env_var(tmp_path, monkeypatch, value):
    facts = await _stamped_facts(tmp_path, monkeypatch, owner=value)

    assert "owner" not in facts
    assert "ca_user" in facts


async def test_the_owner_ladder_is_never_consulted(tmp_path, monkeypatch):
    """The ladder's last rung is the card's own identity; it must not attribute.

    The write verdict asks the ladder whose switch governs the write, and the
    ladder answers here with the armed login. That answer must not reach the
    record as the write's owner: only the dispatch job's stamp does.
    """
    from osprey_connectors import posture_store

    ladder = MagicMock(return_value=FIXTURE_IDENTITY)
    monkeypatch.setattr(posture_store, "current_owner", ladder)
    facts = await _stamped_facts(tmp_path, monkeypatch)

    assert "owner" not in facts


async def test_an_unreadable_account_costs_the_stamp_not_the_write(tmp_path, monkeypatch):
    import pwd

    def no_entry(_uid):
        raise KeyError("uid not in passwd")

    monkeypatch.setattr(pwd, "getpwuid", no_entry)
    facts = await _stamped_facts(tmp_path, monkeypatch)

    assert "ca_user" not in facts
    assert facts["ca_host"]


async def test_the_stamps_reach_the_default_record_detail(tmp_path, monkeypatch):
    """The middleware's value-free detail carries the stamps as ``key=value`` tokens."""
    from osprey.audit.call import call_scope
    from osprey.mcp_server.audit_middleware import _call_detail

    _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("OSPREY_CONTROL_OWNER", FIXTURE_IDENTITY)
    with call_scope("toolu_detail", None) as call:
        await _run_single(_make_write_result(channel="TEST:PV", value=1.0))
        detail = _call_detail()

    assert detail is not None
    assert f"ca_user={call.facts['ca_user']}" in detail
    assert f"ca_host={call.facts['ca_host']}" in detail
    assert f"owner={FIXTURE_IDENTITY}" in detail


# ---------------------------------------------------------------------------
# Stage 1: the write verdict, then the approval record, before anything else
# ---------------------------------------------------------------------------


async def _unstamped_write(connector):
    """One write with no ask stamp on disk and no call bound to one."""
    with patch(
        "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=connector,
    ):
        return await _get_channel_write()(operations=[{"channel": "TEST:PV", "value": 1.0}])


async def test_a_write_the_prompt_left_no_record_of_is_refused(tmp_path, monkeypatch):
    """Approval applies (no ``approval:`` section asks), and nothing was stamped."""
    _prepare(tmp_path, monkeypatch)
    connector = AsyncMock()

    with assert_raises_error(error_type="approval_missing") as ctx:
        await _unstamped_write(connector)

    assert ctx["envelope"]["error_message"].startswith("not run:")
    connector.write_channel.assert_not_awaited()


async def test_a_stamp_for_another_tool_is_no_record_of_this_write(tmp_path, monkeypatch):
    """The record names the tool it was asked for: a queue_start stamp vouches for nothing here."""
    from osprey.audit.call import call_scope

    _prepare(tmp_path, monkeypatch)
    state = target_state.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    ApprovalStamps(state).ask("queue_start", tool_use_id="toolu_other")
    connector = AsyncMock()

    with call_scope("toolu_other", None), assert_raises_error(error_type="approval_missing"):
        await _unstamped_write(connector)

    connector.write_channel.assert_not_awaited()


async def test_the_write_runs_unchanged_with_approval_switched_off(tmp_path, monkeypatch):
    """``approval.enabled: false``: no record is asked for, and the envelope is the usual one."""
    _prepare(tmp_path, monkeypatch, config=_CONFIG + "approval:\n  enabled: false\n")
    connector = AsyncMock()
    connector.write_channel.return_value = _make_write_result(channel="TEST:PV", value=1.0)

    with patch(
        "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
        return_value=None,
    ):
        data = extract_response_dict(await _unstamped_write(connector))

    assert data["status"] == "success"
    assert _outcomes(data) == {"TEST:PV": "confirmed"}
    connector.write_channel.assert_awaited_once()


async def test_the_write_runs_without_a_record_where_its_policy_skips(tmp_path, monkeypatch):
    """A tool whose policy does not ask needs no record."""
    _prepare(
        tmp_path,
        monkeypatch,
        config=_CONFIG + "approval:\n  enabled: true\n  tools:\n    channel_write: skip\n",
    )
    connector = AsyncMock()
    connector.write_channel.return_value = _make_write_result(channel="TEST:PV", value=1.0)

    with patch(
        "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
        return_value=None,
    ):
        data = extract_response_dict(await _unstamped_write(connector))

    assert data["status"] == "success"


async def test_a_malformed_approval_switch_refuses_every_write(tmp_path, monkeypatch):
    """An ``approval.enabled`` that is not a bool refuses even a stamped write."""
    _prepare(tmp_path, monkeypatch, config=_CONFIG + "approval:\n  enabled: 'yes'\n")
    connector = AsyncMock()

    with _approved(), assert_raises_error(error_type="approval_config_invalid"):
        await _unstamped_write(connector)

    connector.write_channel.assert_not_awaited()


async def test_the_verdict_refuses_before_the_record_is_asked_for(tmp_path, monkeypatch):
    """Switch off and no record: the answer is the verdict's, in its own words."""
    from osprey.mcp_server.control_system.tools import channel_write as cw
    from osprey_connectors.refusals import refusal_phrase

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_CONFIG)
    _arm(tmp_path, monkeypatch, switch=False)
    initialize_server_context()
    asked = MagicMock(side_effect=AssertionError("the record must not be asked for"))
    monkeypatch.setattr(cw, "require_approval_record", asked)
    connector = AsyncMock()

    with assert_raises_error(error_type="writes_disabled") as ctx:
        await _unstamped_write(connector)

    envelope = ctx["envelope"]
    assert envelope["details"] == {"reason": "switch_off", "target": "va"}
    assert envelope["error_message"].startswith(refusal_phrase("switch_off", "simulator"))
    assert "writes_enabled" not in envelope["error_message"]
    asked.assert_not_called()
    connector.write_channel.assert_not_awaited()


async def test_the_verdict_refuses_a_stamped_write_too(tmp_path, monkeypatch):
    """A record of the prompt does not arm a machine whose tier allows nothing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_CONFIG.replace("writes: allowed", "writes: never"))
    _arm(tmp_path, monkeypatch)
    initialize_server_context()
    connector = AsyncMock()

    with _approved(), assert_raises_error(error_type="writes_disabled") as ctx:
        await _unstamped_write(connector)

    assert ctx["envelope"]["details"]["reason"] == "tier_never"
    connector.write_channel.assert_not_awaited()


async def test_a_record_naming_an_unknown_target_is_refused(tmp_path, monkeypatch):
    """A record on a target this deployment cannot resolve refuses as ``target_unknown``."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_CONFIG)
    root = tmp_path / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    write_control_context(root, target="live", owned_by=None, switch={"live": "on"})
    initialize_server_context()
    connector = AsyncMock()

    with _approved(), assert_raises_error(error_type="writes_disabled") as ctx:
        await _unstamped_write(connector)

    assert ctx["envelope"]["details"] == {"reason": "target_unknown", "target": "live"}
    connector.write_channel.assert_not_awaited()


async def test_a_write_no_owner_can_be_named_for_is_refused(tmp_path, monkeypatch):
    """The owner ladder yields nobody: ``owner_unknown``, even for a stamped write."""
    from osprey_connectors import posture_store

    _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(posture_store, "current_owner", lambda *_a, **_k: posture_store.NO_OWNER)
    connector = AsyncMock()

    with _approved(), assert_raises_error(error_type="writes_disabled") as ctx:
        await _unstamped_write(connector)

    assert ctx["envelope"]["details"] == {"reason": "owner_unknown", "target": "va"}
    connector.write_channel.assert_not_awaited()


async def test_a_write_for_a_foreign_owner_is_refused(tmp_path, monkeypatch):
    """A host holds only its own login's switch: another owner is ``owner_foreign``."""
    _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("OSPREY_CONTROL_OWNER", f"{FIXTURE_IDENTITY}-not")
    connector = AsyncMock()

    with _approved(), assert_raises_error(error_type="writes_disabled") as ctx:
        await _unstamped_write(connector)

    assert ctx["envelope"]["details"] == {"reason": "owner_foreign", "target": "va"}
    connector.write_channel.assert_not_awaited()


async def test_an_asked_write_without_its_target_stamp_is_refused(tmp_path, monkeypatch):
    """Stage 1 passed on the ask stamp, but the prompt's target stamp is absent.

    The hook files the two stamps one after the other; without the second the
    target the operator was shown cannot be compared with the one the call
    entered on, so a switch between them would go unseen. Refused.
    """
    _prepare(tmp_path, monkeypatch)
    state = target_state.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    connector = AsyncMock()

    with (
        ApprovalStamps(state).approved("channel_write"),
        assert_raises_error(error_type="approval_missing") as ctx,
    ):
        await _unstamped_write(connector)

    assert ctx["envelope"]["error_message"].startswith("not run:")
    connector.write_channel.assert_not_awaited()


async def test_an_asked_write_with_both_stamps_runs(tmp_path, monkeypatch):
    """The ask stamp and the target stamp, filed by the fixture itself, let the write out."""
    from osprey.mcp_server.control_system.tools import channel_write as cw

    _prepare(tmp_path, monkeypatch)
    state = target_state.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    operations = [{"channel": "TEST:PV", "value": 1.0}]
    binding = cw._read_target_binding()
    assert binding is not None
    connector = AsyncMock()
    connector.write_channel.return_value = _make_write_result(channel="TEST:PV", value=1.0)

    with (
        ApprovalStamps(state).approved(
            "channel_write", operations=operations, target=binding[0], generation=binding[1]
        ),
        patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch(
            "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    ):
        data = extract_response_dict(await _get_channel_write()(operations=operations))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()
