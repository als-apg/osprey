"""The connector's reference monitor decides every write by the write verdict.

``ControlSystemConnector`` wraps every subclass's ``write_channel`` and
``write_multiple_channels``; the wrapper asks ``_writes_enabled``, which is
:func:`osprey_connectors.posture_store.write_verdict` for the connector's own
machine — the one the factory stamped as ``_control_target``. A refused write
never reaches the transport and comes back ``REFUSED`` / ``WRITES_DISABLED``,
its message the refusals contract's phrase for the verdict's reason.

The tests drive ``write_channel`` rather than the property alone, because the
refusal's words are half of what an operator meets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system import _base as connector_base
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.refusals import REASONS, refusal_phrase
from osprey_connectors.types import TARGET_LIVE, TARGET_STANDIN, TARGET_VA, VIRTUAL_ACCELERATOR
from osprey_connectors_tests._harness import (
    REFUSAL_BAN_LIST,
    isolated_connector_registries,
    write_control_context,
    write_payload,
)
from osprey_connectors_tests._harness._write_verdict_cases import case_environment
from osprey_connectors_tests._write_fakes import RecordingConnector, config_reader

__all__ = ["case_environment"]

EXECUTION_MODE = "OSPREY_EXECUTION_MODE"

#: A deployment whose real machine is EPICS and whose write tier allows writes.
ALLOWED: dict[str, Any] = {"type": "epics", "writes": "allowed"}

#: The same deployment with the tier at ``never``.
NEVER: dict[str, Any] = {"type": "epics", "writes": "never"}


@pytest.fixture(autouse=True)
def clean_environment(case_environment):
    """No stamped owner, tree, pin or run mode; the host record under scratch."""


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def tier(monkeypatch):
    """Serve one ``control_system`` section through ``get_config_value``."""

    def _install(section: Any) -> None:
        monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(section))

    return _install


def _switch(root: Path, *targets: str, posture: dict[str, str] | None = None) -> None:
    """The login's record with exactly *targets* switched on.

    *posture* is an old narrowing laid over the mirror verbatim, as a file
    written before the mirror would carry it.
    """
    path = write_control_context(root, owned_by=None, switch=dict.fromkeys(targets, "asked"))
    if posture is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["posture"] = dict(posture)
        write_payload(path, payload)
        control_context.invalidate_cache()


def _built(target: str | None, connector_type: str | None = "epics") -> RecordingConnector:
    """A connector stamped the way ``ConnectorFactory`` stamps one."""
    connector = RecordingConnector()
    connector._connector_type = connector_type
    connector._control_target = target
    return connector


def _count_record_reads(monkeypatch) -> list[int]:
    """Count calls of the host record reader the verdict uses."""
    calls = [0]
    real = control_context.read_record_status

    def _counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(control_context, "read_record_status", _counting)
    return calls


def _refused(result, reason: str, kind: str | None = None) -> None:
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason == "WRITES_DISABLED"
    assert refusal_phrase(reason, kind) in result.error_message
    assert REFUSAL_BAN_LIST.search(result.error_message) is None


class TestTheSwitchDecides:
    """With the tier allowing, the login's switch for the machine decides."""

    @pytest.mark.asyncio
    async def test_switch_off_is_refused_with_the_switch_off_phrase(self, tier, root):
        # Arrange: a record whose switch is on for another machine only.
        tier(ALLOWED)
        _switch(root, TARGET_VA)
        connector = _built(TARGET_LIVE)

        # Act
        result = await connector.write_channel("SR:CORR:1:SP", 0.5)

        # Assert
        _refused(result, "switch_off", "real")
        assert result.error_message.startswith("Write to 'SR:CORR:1:SP' blocked: ")
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_no_record_is_every_switch_off(self, tier):
        tier(ALLOWED)
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("SR:CORR:1:SP", 0.5)

        _refused(result, "switch_off", "real")
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_switch_on_reaches_the_transport(self, tier, root):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("SR:CORR:1:SP", 0.5)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector.writes == [("SR:CORR:1:SP", 0.5)]

    @pytest.mark.asyncio
    async def test_one_machine_on_leaves_the_others_off(self, tier, root):
        tier(ALLOWED)
        _switch(root, TARGET_VA)
        simulator, live = _built(TARGET_VA, VIRTUAL_ACCELERATOR), _built(TARGET_LIVE)

        allowed = await simulator.write_channel("A:SP", 1.0)
        refused = await live.write_channel("A:SP", 1.0)

        assert allowed.outcome is WriteOutcome.CONFIRMED
        _refused(refused, "switch_off", "real")
        assert live.writes == []

    @pytest.mark.asyncio
    async def test_a_switch_flip_lands_on_a_connector_already_writing(self, tier, root):
        """Re-read on every write: no respawn, no config edit."""
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        connector = _built(TARGET_LIVE)
        assert (await connector.write_channel("A:SP", 1.0)).outcome is WriteOutcome.CONFIRMED

        control_context.set_switch(
            [TARGET_LIVE], "off", path=control_context._record_path_under(root)
        )
        control_context.invalidate_cache()
        result = await connector.write_channel("A:SP", 2.0)

        _refused(result, "switch_off", "real")
        assert connector.writes == [("A:SP", 1.0)]

    @pytest.mark.asyncio
    async def test_an_old_sandbox_narrowing_does_not_refuse_with_the_switch_on(self, tier, root):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE, posture={TARGET_LIVE: posture_store.POSTURE_SANDBOX})
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector.writes == [("A:SP", 1.0)]


class TestTheEarlierTermsRefuseFirst:
    """Run mode, tier and target refuse ahead of the switch."""

    @pytest.mark.asyncio
    async def test_an_unstamped_connector_is_refused_as_an_unknown_target(self, tier, root):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE, TARGET_VA, TARGET_STANDIN)
        connector = _built(None, connector_type=None)

        result = await connector.write_channel("A:SP", 1.0)

        _refused(result, "target_unknown")
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_an_unstamped_batch_is_refused_as_an_unknown_target(self, tier, root):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        connector = _built(None, connector_type=None)

        results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

        for result in results:
            _refused(result, "target_unknown")
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_a_tier_that_does_not_allow_refuses_with_the_switch_on(self, tier, root):
        tier(NEVER)
        _switch(root, TARGET_LIVE)
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        _refused(result, "tier_never", "real")
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_a_config_that_cannot_be_read_arms_nothing(self, monkeypatch, root):
        def _broken(*_args, **_kwargs):
            raise FileNotFoundError("no config")

        monkeypatch.setattr("osprey_connectors.config.get_config_value", _broken)
        _switch(root, TARGET_LIVE)
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert refusal_phrase("tier_never") in result.error_message
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_a_readonly_run_refuses_with_the_tier_allowing_and_the_switch_on(
        self, tier, root, monkeypatch
    ):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        monkeypatch.setenv(EXECUTION_MODE, "readonly")
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        _refused(result, "readonly_run")
        # The executor's stderr matcher looks for this; it rides beside the phrase.
        assert "readonly execution mode" in result.error_message
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_a_readonly_run_reads_neither_config_nor_record(self, monkeypatch, root):
        """A run that refuses whatever they say has no business opening them."""

        def _no_config(key, *_args, **_kwargs):
            raise AssertionError(f"config read under a read-only run: {key}")

        monkeypatch.setattr("osprey_connectors.config.get_config_value", _no_config)
        _switch(root, TARGET_LIVE)
        reads = _count_record_reads(monkeypatch)
        monkeypatch.setenv(EXECUTION_MODE, "readonly")
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert reads[0] == 0

    @pytest.mark.asyncio
    async def test_a_launch_pin_refuses_with_the_switch_on(self, tier, root, monkeypatch):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        _refused(result, "launch_pinned", "real")
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_an_unreadable_launch_pin_refuses_too(self, tier, root, monkeypatch):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "garbage")
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        _refused(result, "launch_pinned", "real")

    @pytest.mark.asyncio
    async def test_an_unreadable_record_is_refused_as_unavailable(self, tier, root):
        tier(ALLOWED)
        write_payload(control_context._record_path_under(root), "not a record")
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        _refused(result, "record_unavailable", "real")
        assert connector.writes == []


class TestTheMonitorIsTheVerdict:
    """The property answers exactly what ``write_verdict`` answers."""

    @pytest.mark.parametrize("section", [ALLOWED, NEVER], ids=["allowed", "never"])
    @pytest.mark.parametrize("switched", [(), (TARGET_LIVE,), (TARGET_VA,)], ids=str)
    @pytest.mark.parametrize("target", [TARGET_LIVE, TARGET_VA, None], ids=str)
    def test_the_two_answers_match(self, tier, root, section, switched, target):
        tier(section)
        _switch(root, *switched)
        connector = _built(target)

        allowed = connector._writes_enabled

        verdict = posture_store.write_verdict(section, target)
        assert allowed is verdict.allowed
        assert connector._last_verdict == verdict

    @pytest.mark.asyncio
    async def test_the_factory_stamps_the_target_the_verdict_is_asked_for(self, tier, root):
        tier(ALLOWED)
        _switch(root, TARGET_VA)
        with isolated_connector_registries(clear=True):
            ConnectorFactory.register_control_system(VIRTUAL_ACCELERATOR, RecordingConnector)
            on = await ConnectorFactory.create_control_system_connector(
                {"type": VIRTUAL_ACCELERATOR}, control_target=TARGET_VA
            )
            off = await ConnectorFactory.create_control_system_connector(
                {"type": VIRTUAL_ACCELERATOR}, control_target=TARGET_LIVE
            )

        assert (await on.write_channel("A:SP", 1.0)).outcome is WriteOutcome.CONFIRMED
        _refused(await off.write_channel("A:SP", 1.0), "switch_off", "real")

    @pytest.mark.parametrize(
        "reason", [reason for reason in REASONS if reason not in ("switching", "stopped_since")]
    )
    def test_every_monitor_refusal_names_no_key_file_or_variable(self, reason):
        """The monitor's words are the contract's, plus the channel and the marker."""
        verdict = posture_store.WriteVerdict(
            allowed=False, reason=reason, target=TARGET_LIVE, kind="real"
        )

        result = connector_base._verdict_refusal("A:SP", 1.0, verdict)

        assert refusal_phrase(reason, "real") in result.error_message
        assert REFUSAL_BAN_LIST.search(result.error_message) is None


class TestOneRecordReadPerWrite:
    """One write, or one batch, reads the record once; its refusal reads nothing more."""

    @pytest.mark.asyncio
    async def test_a_refused_write_reads_the_record_once(self, tier, root, monkeypatch):
        tier(ALLOWED)
        _switch(root, TARGET_VA)
        reads = _count_record_reads(monkeypatch)
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert reads[0] == 1

    @pytest.mark.asyncio
    async def test_a_permitted_write_reads_the_record_once(self, tier, root, monkeypatch):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        reads = _count_record_reads(monkeypatch)
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert reads[0] == 1

    @pytest.mark.asyncio
    async def test_a_refused_batch_reads_the_record_once_for_the_whole_batch(
        self, tier, root, monkeypatch
    ):
        tier(ALLOWED)
        _switch(root, TARGET_VA)
        reads = _count_record_reads(monkeypatch)
        connector = _built(TARGET_LIVE)

        results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

        assert reads[0] == 1
        assert len({result.error_message.split(": ", 1)[1] for result in results}) == 1
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_a_pinned_write_reads_the_record_once(self, tier, root, monkeypatch):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
        reads = _count_record_reads(monkeypatch)
        connector = _built(TARGET_LIVE)

        result = await connector.write_channel("A:SP", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert reads[0] == 1


class TestTheMemo:
    """The guard words the refusal from the verdict the property memoised."""

    @pytest.mark.asyncio
    async def test_the_refusal_is_worded_from_the_decision_not_a_second_read(
        self, tier, root, monkeypatch
    ):
        """A switch turned on between decision and wording must not change the words."""
        tier(ALLOWED)
        connector = _built(TARGET_LIVE)
        real_verdict = connector_base._monitor_verdict
        calls = []

        def _once(target):
            calls.append(target)
            verdict = real_verdict(target)
            _switch(root, TARGET_LIVE)  # flips after the decision
            return verdict

        monkeypatch.setattr(connector_base, "_monitor_verdict", _once)

        result = await connector.write_channel("A:SP", 1.0)

        _refused(result, "switch_off", "real")
        assert calls == [TARGET_LIVE]

    @pytest.mark.asyncio
    async def test_the_guard_clears_the_memo_before_each_write(self, tier, root):
        tier(ALLOWED)
        connector = _built(TARGET_LIVE)
        await connector.write_channel("A:SP", 1.0)
        assert connector._last_verdict is not None
        assert connector._last_verdict.reason == "switch_off"

        _switch(root, TARGET_LIVE)
        result = await connector.write_channel("A:SP", 2.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector._last_verdict.allowed is True


class _OverridingConnector(RecordingConnector):
    """A subclass that owns its own answer to ``_writes_enabled``."""

    armed = True

    @property
    def _writes_enabled(self) -> bool:
        return self.armed


class TestAnOverriddenMonitor:
    """The guard asks the property, not the base class's implementation of it."""

    @pytest.mark.asyncio
    async def test_an_override_that_arms_writes_is_honoured(self, tier):
        tier(NEVER)
        connector = _OverridingConnector()
        connector._control_target = TARGET_LIVE

        result = await connector.write_channel("A:SP", 0.5)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector.writes == [("A:SP", 0.5)]

    @pytest.mark.asyncio
    async def test_an_override_that_refuses_is_worded_from_a_verdict_of_its_own(self, tier):
        """The override sets no memo, so the refusal asks the verdict itself."""
        tier(ALLOWED)
        connector = _OverridingConnector()
        connector.armed = False
        connector._control_target = TARGET_LIVE

        result = await connector.write_channel("A:SP", 0.5)

        _refused(result, "switch_off", "real")
        assert connector.writes == []

    @pytest.mark.asyncio
    async def test_an_override_that_refuses_what_the_verdict_allows_claims_no_reason(
        self, tier, root
    ):
        tier(ALLOWED)
        _switch(root, TARGET_LIVE)
        connector = _OverridingConnector()
        connector.armed = False
        connector._control_target = TARGET_LIVE

        result = await connector.write_channel("A:SP", 0.5)

        assert result.outcome is WriteOutcome.REFUSED
        assert all(refusal_phrase(reason) not in result.error_message for reason in REASONS)
        assert REFUSAL_BAN_LIST.search(result.error_message) is None
        assert connector.writes == []
