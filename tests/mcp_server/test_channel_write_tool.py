"""Tests for the channel_write MCP tool.

Covers: successful write with readback, limits violation (inline validator),
verification levels, connection errors, error format compliance, and the
``write_state`` contract the agent keys on.

Every assertion about ``write_state`` is made on the PARSED tool output —
``summary.results[]`` — never on an internal dict. That projection is the only
thing an agent ever sees, so a field added to the tool's private bookkeeping and
not to the summary would be invisible in production while looking correct in a
test.

The verification object in the fixture below is a real ``WriteVerification``, not
a ``MagicMock``: a mock answers every attribute truthily, so a state predicate
reading a field no connector populates would pass here and misclassify in the
field.

Note: writes_enabled check is handled by the PreToolUse hook, not the tool itself.
The tool does its own limits validation via LimitsValidator.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.connectors.control_system.base import WriteVerification
from osprey.mcp_server.control_system.server_context import initialize_server_context
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

#: Every state the tool can emit, in the order its predicates decide them. Pinned
#: as a list so a reordering that changes which predicate wins is visible here.
EXPECTED_WRITE_STATES = [
    "blocked",
    "write_failed",
    "verification_not_reported",
    "verification_not_requested",
    "verified_with_alarm",
    "verified",
    "readback_failed",
    "unverified_alarm",
    "verification_failed",
]

#: Distinguishes "argument not given" from an explicit ``None``, which is itself
#: a meaningful value for ``verification`` and ``readback_value``.
_UNSET = object()


def _make_write_result(
    channel="TEST:PV",
    value=1.0,
    success=True,
    error_message=None,
    verification_level="callback",
    blocked=False,
    refusal_reason=None,
    verified=True,
    readback_value=_UNSET,
    tolerance_used=0.1,
    notes="",
    readback_alarm_status=None,
    readback_alarm_severity=None,
    failure_kind=None,
    verification=_UNSET,
):
    """A write result whose verification is a real ``WriteVerification``.

    ``verification`` may be passed explicitly — including ``None`` for a
    connector that reported none, or a duck-typed stand-in for a custom
    connector — otherwise one is built from the keyword arguments.
    """
    result = MagicMock()
    result.channel_address = channel
    result.value_written = value
    result.success = success
    result.error_message = error_message
    result.blocked = blocked
    result.refusal_reason = refusal_reason
    if verification is _UNSET:
        verification = WriteVerification(
            level=verification_level,
            verified=verified,
            readback_value=value if readback_value is _UNSET else readback_value,
            tolerance_used=tolerance_used,
            notes=notes,
            readback_alarm_status=readback_alarm_status,
            readback_alarm_severity=readback_alarm_severity,
            failure_kind=failure_kind,
        )
    result.verification = verification
    return result


def _write_states(data):
    """``{channel: write_state}`` from a parsed tool response."""
    return {r["channel"]: r["write_state"] for r in data["summary"]["results"]}


def _get_channel_write():
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    return get_tool_fn(channel_write)


def _prepare(tmp_path, monkeypatch):
    """Minimal project + server context the tool needs to run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
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
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=validator,
        ),
    ):
        yield


@pytest.mark.unit
async def test_channel_write_success(tmp_path, monkeypatch):
    """Successful write returns result with verification in summary."""
    _prepare(tmp_path, monkeypatch)

    write_result = _make_write_result(channel="TEST:PV", value=42.0)
    mock_connector = AsyncMock()
    mock_connector.write_channel.return_value = write_result

    with _patched(mock_connector):
        fn = _get_channel_write()
        result = await fn(operations=[{"channel": "TEST:PV", "value": 42.0}])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 1
    assert data["summary"]["failed"] == 0
    assert data["summary"]["results"][0]["channel"] == "TEST:PV"


@pytest.mark.unit
async def test_channel_write_with_readback(tmp_path, monkeypatch):
    """Write with readback verification returns verification details in summary."""
    _prepare(tmp_path, monkeypatch)

    write_result = _make_write_result(channel="TEST:PV", value=42.0, verification_level="readback")
    write_result.verification.readback_value = 42.01
    mock_connector = AsyncMock()
    mock_connector.write_channel.return_value = write_result

    with _patched(mock_connector):
        fn = _get_channel_write()
        result = await fn(
            operations=[{"channel": "TEST:PV", "value": 42.0}],
            verification_level="readback",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["results"][0]["verification"]["level"] == "readback"
    assert data["summary"]["results"][0]["verification"]["readback_value"] == 42.01


@pytest.mark.unit
async def test_channel_write_limits_violation(tmp_path, monkeypatch):
    """Write exceeding channel limits (via inline validator) returns structured error."""
    from osprey.errors import ChannelLimitsViolationError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")

    mock_validator = MagicMock()
    mock_validator.validate.side_effect = ChannelLimitsViolationError(
        channel_address="TEST:PV",
        value=9999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 9999.0 above maximum 100.0",
        min_value=0.0,
        max_value=100.0,
    )

    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=mock_validator,
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


@pytest.mark.unit
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


@pytest.mark.unit
async def test_channel_write_multiple_operations(tmp_path, monkeypatch):
    """Multiple write operations are all processed."""
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    with _patched(mock_connector):
        fn = _get_channel_write()
        result = await fn(
            operations=[
                {"channel": "PV:A", "value": 1.0},
                {"channel": "PV:B", "value": 2.0},
            ]
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 2
    assert data["summary"]["failed"] == 0


@pytest.mark.unit
async def test_channel_write_connector_limits_violation(tmp_path, monkeypatch):
    """ChannelLimitsViolationError from connector is classified as limits_violation with structured details."""
    from osprey.errors import ChannelLimitsViolationError

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


@pytest.mark.unit
async def test_channel_write_empty_operations(tmp_path, monkeypatch):
    """Empty operations list returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_channel_write()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(operations=[])

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_channel_write_missing_channel_key(tmp_path, monkeypatch):
    """Operation missing 'channel' key returns validation error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")

    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=None,
    ):
        fn = _get_channel_write()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(operations=[{"value": 42.0}])

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_channel_write_all_refused_is_write_refused(tmp_path, monkeypatch):
    """All-refused batch (every op blocked) yields a typed write_refused envelope.

    The connector refuses every write (blocked=True, never sent to the control
    system). The tool must raise ChannelWriteBlockedError so the error handler
    classifies it as write_refused — NOT the generic internal_error that a bare
    RuntimeError would produce.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(
            channel="PV:A",
            value=1.0,
            success=False,
            error_message="Write to 'PV:A' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            success=False,
            error_message="Write to 'PV:B' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
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
    assert data["error_type"] != "internal_error"
    # Both refused channels are named in the summary message.
    assert "PV:A" in data["error_message"]
    assert "PV:B" in data["error_message"]
    # Structured details carry the refusal reason discriminator.
    assert data["details"]["reason"] == "WRITES_DISABLED"


@pytest.mark.unit
async def test_channel_write_partial_refusal_reports_per_op(tmp_path, monkeypatch):
    """Partial-rejection batch returns success JSON with per-op refusal reporting.

    One op succeeds, one is refused. The tool must NOT raise (writes did happen);
    it returns a success-status result whose per-op entries carry blocked /
    refusal_reason and whose summary reports refused == 1, successful == 1.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0, success=True),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            success=False,
            error_message="Write to 'PV:B' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    with _patched(mock_connector):
        fn = _get_channel_write()
        result = await fn(
            operations=[
                {"channel": "PV:A", "value": 1.0},
                {"channel": "PV:B", "value": 2.0},
            ]
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    summary = data["summary"]
    assert summary["successful"] == 1
    assert summary["refused"] == 1
    assert summary["failed"] == 1
    by_channel = {r["channel"]: r for r in summary["results"]}
    assert by_channel["PV:A"]["success"] is True
    assert by_channel["PV:A"]["blocked"] is False
    assert by_channel["PV:B"]["success"] is False
    assert by_channel["PV:B"]["blocked"] is True
    assert by_channel["PV:B"]["refusal_reason"] == "WRITES_DISABLED"


@pytest.mark.unit
async def test_channel_write_all_failed_caput_is_internal_error(tmp_path, monkeypatch):
    """All-failed caput (attempted, not refused) preserves internal_error.

    The writes were sent to the control system and failed (blocked=False). This
    is an I/O failure, not a policy refusal, so it must keep the RuntimeError ->
    internal_error classification rather than becoming write_refused.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(
            channel="PV:A",
            value=1.0,
            success=False,
            error_message="caput failed: timeout",
            blocked=False,
        ),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            success=False,
            error_message="caput failed: no connection",
            blocked=False,
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    with _patched(mock_connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    data = _exc_ctx["envelope"]
    assert data["error_type"] == "internal_error"
    assert data["error_type"] != "write_refused"


# ---------------------------------------------------------------------------
# write_state — the closed-set word the agent keys on
# ---------------------------------------------------------------------------


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


#: (state, kwargs for the subject result). Each row is one predicate in the
#: ordered chain, chosen so the row above it does NOT also match — that is what
#: makes this a precedence test and not nine independent smoke tests.
_STATE_CASES = [
    # Refused: never sent. Also success=False, so this pins that `blocked` wins.
    ("blocked", {"success": False, "blocked": True, "refusal_reason": "WRITES_DISABLED"}),
    # Attempted and failed. Verification present and "verified" — a failed write
    # must not be reported by its verification.
    ("write_failed", {"success": False, "error_message": "caput failed: timeout"}),
    ("verification_not_reported", {"verification": None}),
    # Level "none" with verified False: nothing was asked for, so this is not a
    # verification failure.
    ("verification_not_requested", {"verification_level": "none", "verified": False}),
    # MAJOR alarm on a readback that matched.
    (
        "verified_with_alarm",
        {
            "verification_level": "readback",
            "verified": True,
            "readback_alarm_severity": 2,
            "readback_alarm_status": "HIHI",
        },
    ),
    # MINOR alarm on a readback that matched stays plain "verified".
    (
        "verified",
        {
            "verification_level": "readback",
            "verified": True,
            "readback_alarm_severity": 1,
            "readback_alarm_status": "HIGH",
        },
    ),
    # The readback read itself raised. Severity is set too, so this pins that
    # failure_kind outranks the alarm predicate.
    (
        "readback_failed",
        {
            "verification_level": "readback",
            "verified": False,
            "failure_kind": "readback_failed",
            "readback_alarm_severity": 3,
        },
    ),
    (
        "unverified_alarm",
        {
            "verification_level": "readback",
            "verified": False,
            "readback_alarm_severity": 1,
            "readback_alarm_status": "HIGH",
        },
    ),
    ("verification_failed", {"verification_level": "readback", "verified": False}),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "expected_state,result_kwargs", _STATE_CASES, ids=[c[0] for c in _STATE_CASES]
)
async def test_write_state_precedence(tmp_path, monkeypatch, expected_state, result_kwargs):
    """Each ordered predicate produces its state in the shipped summary.

    Run as a two-op batch with a healthy first write so the refused and failed
    subjects do not trip the all-refused / all-failed top-level errors, which
    are a separate contract.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:OK", value=1.0),
        _make_write_result(channel="PV:SUBJECT", value=2.0, **result_kwargs),
    ]
    data, _ = await _run_batch(results)

    assert _write_states(data)["PV:SUBJECT"] == expected_state


@pytest.mark.unit
def test_state_cases_cover_every_documented_state():
    """The parametrisation above exercises the whole closed set, in order."""
    assert [case[0] for case in _STATE_CASES] == EXPECTED_WRITE_STATES


@pytest.mark.unit
async def test_readback_healthy_severity_zero_is_still_verification_failed(tmp_path, monkeypatch):
    """A REPORTED healthy severity of 0 does not save an unverified readback.

    Kept out of ``_STATE_CASES`` because that table is one row per documented
    state, pinned 1:1 against ``EXPECTED_WRITE_STATES`` above; this pins the
    0-vs-None distinction on the SAME ``verification_failed`` predicate instead
    of adding a tenth state. 0 is a reported healthy severity, distinct from
    None ("not reported"); the ``> 0`` check in ``_alarm_severity`` correctly
    excludes it, so it falls through to ``verification_failed`` rather than
    being read as an alarm.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:OK", value=1.0),
        _make_write_result(
            channel="PV:SUBJECT",
            value=2.0,
            verification_level="readback",
            verified=False,
            readback_alarm_severity=0,
        ),
    ]
    data, _ = await _run_batch(results)

    assert _write_states(data)["PV:SUBJECT"] == "verification_failed"


@pytest.mark.unit
async def test_write_state_ignores_notes_text(tmp_path, monkeypatch):
    """Notes are display-only: rewording them cannot change the state.

    The narration bug this whole contract exists to close came from deriving
    meaning out of prose, so the state must be a pure function of the
    structured fields.
    """
    _prepare(tmp_path, monkeypatch)

    states = []
    for note in ("", "Readback matched.", "MISMATCH: readback failed, value not applied!"):
        results = [
            _make_write_result(
                channel="PV:A",
                value=1.0,
                verification_level="readback",
                verified=True,
                notes=note,
            ),
            _make_write_result(channel="PV:B", value=2.0),
        ]
        data, _ = await _run_batch(results)
        states.append(_write_states(data)["PV:A"])

    assert states == ["verified", "verified", "verified"]


@pytest.mark.unit
async def test_duck_typed_verification_yields_a_state(tmp_path, monkeypatch):
    """A custom connector's verification object must classify, not raise.

    Out-of-tree connectors may return anything with ``level`` and ``verified``;
    the new alarm and failure_kind fields are read with ``getattr`` defaults so
    an object that predates them still produces a result.
    """
    _prepare(tmp_path, monkeypatch)

    class _MinimalVerification:
        level = "callback"
        verified = True

    class _FloatSeverityVerification:
        level = "callback"
        verified = True
        readback_alarm_severity = 2.0

    class _BoolSeverityVerification:
        level = "callback"
        verified = True
        readback_alarm_severity = True

    results = [
        _make_write_result(channel="PV:A", value=1.0, verification=_MinimalVerification()),
        _make_write_result(channel="PV:B", value=2.0),
        _make_write_result(
            channel="PV:FLOAT", value=3.0, verification=_FloatSeverityVerification()
        ),
        _make_write_result(channel="PV:BOOL", value=4.0, verification=_BoolSeverityVerification()),
    ]
    data, _ = await _run_batch(results)

    entries = {r["channel"]: r for r in data["summary"]["results"]}
    entry = entries["PV:A"]
    assert entry["write_state"] == "verified"
    # The absent fields are reported as null rather than omitted, so a consumer
    # can tell "not reported" from a value.
    assert entry["verification"]["readback_alarm_status"] is None
    assert entry["verification"]["readback_alarm_severity"] is None
    assert entry["verification"]["failure_kind"] is None
    # A float or bool severity fails `_alarm_severity`'s `isinstance(int)` type
    # guard and degrades to "not reported" (None), so both fall through to the
    # plain "verified" state rather than "verified_with_alarm" — degrade, don't
    # raise.
    assert entries["PV:FLOAT"]["write_state"] == "verified"
    assert entries["PV:BOOL"]["write_state"] == "verified"


@pytest.mark.unit
async def test_mixed_batch_reports_one_state_per_channel(tmp_path, monkeypatch):
    """A batch of unlike outcomes reports each channel's own state."""
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:OK", value=1.0),
        _make_write_result(
            channel="PV:MISMATCH", value=2.0, verification_level="readback", verified=False
        ),
        _make_write_result(
            channel="PV:REFUSED",
            value=3.0,
            success=False,
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
        _make_write_result(channel="PV:NOVERIF", value=4.0, verification=None),
    ]
    data, _ = await _run_batch(results)

    assert _write_states(data) == {
        "PV:OK": "verified",
        "PV:MISMATCH": "verification_failed",
        "PV:REFUSED": "blocked",
        "PV:NOVERIF": "verification_not_reported",
    }


@pytest.mark.unit
async def test_alarm_fields_reach_the_shipped_verification_projection(tmp_path, monkeypatch):
    """The alarm name, severity and failure_kind are in summary.results[]."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(
        channel="TEST:PV",
        value=42.0,
        verification_level="readback",
        verified=True,
        readback_value=42.0,
        readback_alarm_status="HIHI",
        readback_alarm_severity=2,
    )
    data, _ = await _run_single(result)

    verification = data["summary"]["results"][0]["verification"]
    assert verification["readback_alarm_status"] == "HIHI"
    assert verification["readback_alarm_severity"] == 2
    assert verification["failure_kind"] is None
    assert data["summary"]["results"][0]["write_state"] == "verified_with_alarm"


# ---------------------------------------------------------------------------
# summary.verification_failed
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_verification_failed_counts_only_executed_unverified(tmp_path, monkeypatch):
    """Only writes that executed, asked for verification, and lacked it count.

    A refused write never reached the machine and a level-"none" write never
    asked, so counting either would overstate how much of the batch is in doubt.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        # Counts: executed, readback requested, not verified.
        _make_write_result(
            channel="PV:MISMATCH", value=1.0, verification_level="readback", verified=False
        ),
        # Counts: executed, callback requested, not confirmed.
        _make_write_result(
            channel="PV:NOCALLBACK", value=2.0, verification_level="callback", verified=False
        ),
        # Does not count: refused, never executed.
        _make_write_result(
            channel="PV:REFUSED",
            value=3.0,
            success=False,
            blocked=True,
            refusal_reason="WRITES_DISABLED",
            verification_level="readback",
            verified=False,
        ),
        # Does not count: no verification was asked for.
        _make_write_result(channel="PV:NONE", value=4.0, verification_level="none", verified=False),
        # Does not count: verified.
        _make_write_result(channel="PV:OK", value=5.0),
    ]
    data, _ = await _run_batch(results)

    assert data["summary"]["verification_failed"] == 2


# ---------------------------------------------------------------------------
# verification_level dispatch — explicit vs omitted, single vs batch
# ---------------------------------------------------------------------------


def _validator_returning(level, tolerance):
    """A limits validator whose database answers with *level* / *tolerance*."""
    validator = MagicMock()
    validator.get_verification_config.return_value = (level, tolerance)
    return validator


@pytest.mark.unit
async def test_explicit_level_beats_the_limits_database_on_a_single_write(tmp_path, monkeypatch):
    """An explicitly requested level is forwarded unchanged.

    Regression: the validator used to overwrite the caller's level
    unconditionally, so naming a level did nothing.
    """
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0, verification_level="readback")
    _, connector = await _run_single(
        result,
        validator=_validator_returning("callback", None),
        verification_level="readback",
    )

    assert connector.write_channel.call_args.kwargs["verification_level"] == "readback"


@pytest.mark.unit
async def test_limits_database_tolerance_applies_to_an_explicit_level(tmp_path, monkeypatch):
    """A per-channel tolerance survives an explicit level.

    The tolerance and the level are separate decisions: dropping the per-channel
    tolerance would compare a setpoint against the connector's absolute default
    instead of the percentage the limits database configured.
    """
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0, verification_level="readback")
    _, connector = await _run_single(
        result,
        validator=_validator_returning("callback", 0.42),
        verification_level="readback",
    )

    kwargs = connector.write_channel.call_args.kwargs
    assert kwargs["verification_level"] == "readback"
    assert kwargs["tolerance"] == 0.42


@pytest.mark.unit
async def test_omitted_level_forwards_the_limits_database_level_and_tolerance(
    tmp_path, monkeypatch
):
    """With no level named, a single write takes the limits database's."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0, verification_level="readback")
    _, connector = await _run_single(result, validator=_validator_returning("readback", 0.05))

    kwargs = connector.write_channel.call_args.kwargs
    assert kwargs["verification_level"] == "readback"
    assert kwargs["tolerance"] == 0.05


@pytest.mark.unit
async def test_omitted_level_with_silent_database_leaves_the_keyword_absent(tmp_path, monkeypatch):
    """Omission is a sentinel: the keyword is left off, not passed as None.

    Passing None would override a legacy custom connector's own declared
    default instead of letting it apply.
    """
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0)
    _, connector = await _run_single(result, validator=_validator_returning(None, None))

    kwargs = connector.write_channel.call_args.kwargs
    assert "verification_level" not in kwargs
    assert "tolerance" not in kwargs


@pytest.mark.unit
async def test_explicit_level_is_forwarded_on_a_batch(tmp_path, monkeypatch):
    """A named level applies to every channel in the batch."""
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
    ]
    _, connector = await _run_batch(
        results,
        validator=_validator_returning("none", None),
        verification_level="readback",
    )

    assert connector.write_multiple_channels.call_args.kwargs["verification_level"] == "readback"


@pytest.mark.unit
async def test_omitted_level_leaves_the_batch_keyword_absent(tmp_path, monkeypatch):
    """An omitted level is not resolved for the batch as a whole.

    A batch carries one scalar level for every channel in it, so resolving here
    would pin one channel's entry onto all of them. Leaving the keyword off lets
    each channel resolve its own, which is what makes batch writes behave like
    single writes.
    """
    _prepare(tmp_path, monkeypatch)

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
    ]
    _, connector = await _run_batch(results, validator=_validator_returning("readback", 0.05))

    assert "verification_level" not in connector.write_multiple_channels.call_args.kwargs


# ---------------------------------------------------------------------------
# access_details
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_access_details_level_is_null_when_omitted(tmp_path, monkeypatch):
    """access_details reports what the caller asked for, not what was resolved.

    Null means "the deployment decided"; the level actually used is reported
    per result under verification.level.
    """
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0, verification_level="readback")
    data, _ = await _run_single(result, validator=_validator_returning("readback", None))

    assert data["access_details"]["verification_level"] is None
    assert data["summary"]["results"][0]["verification"]["level"] == "readback"


@pytest.mark.unit
async def test_access_details_level_echoes_an_explicit_request(tmp_path, monkeypatch):
    """An explicit level is echoed back verbatim."""
    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0, verification_level="none")
    data, _ = await _run_single(result, verification_level="none")

    assert data["access_details"]["verification_level"] == "none"


# ---------------------------------------------------------------------------
# the key the rules name and the key the tool emits are one string
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_emitted_state_key_is_the_key_the_generated_rules_name(tmp_path, monkeypatch):
    """The rules tell the agent to read `write_state`; the tool must emit it.

    Two files, one word. If either is renamed on its own the agent is told to
    key on something the payload does not contain, which is silent — the tool
    still returns a valid result and the agent simply has nothing to go on.
    """
    from osprey.cli.templates.manager import TemplateManager
    from osprey.mcp_server.control_system.tools.channel_write import WRITE_STATE_KEY

    _prepare(tmp_path, monkeypatch)

    result = _make_write_result(channel="TEST:PV", value=42.0)
    data, _ = await _run_single(result)
    assert WRITE_STATE_KEY in data["summary"]["results"][0]

    manager = TemplateManager()
    safety = (manager.template_root / "claude_code/claude/rules/safety.md").read_text()
    routing = manager.jinja_env.get_template(
        "claude_code/claude/rules/control-system-safety.md.j2"
    ).render(enabled_servers=[])

    assert WRITE_STATE_KEY in safety
    assert WRITE_STATE_KEY in routing
