"""The operator demo's plan and its audit-line grammar, pinned without a stack.

``scripts/demo_target_switch.py`` is driven for real against two containers, and
that run is its actual gate. What is pinned here is everything a reviewer or a
CI job can check in milliseconds:

* ``--dry-run`` really is dry — it prints the plan and drives nothing, so it is
  safe to run anywhere;
* the plan states its two scope limits (name-server only, archiver asserted from
  config) rather than leaving them to the module docstring nobody greps;
* the audit line grammar is stable. The trail is the deliverable an operator
  reads and a pipeline greps, so ``STEP <n> [target=<t>] …`` is a contract:
  a refactor that renamed the fields would leave every consumer reading a trail
  it can no longer parse, and would otherwise fail nowhere.

The script is loaded by path because ``scripts/`` is not an importable package —
the same ``importlib`` load ``tests/va/e2e/conftest.py`` uses for
``sweep_check.py``. Loading it is itself an assertion: the module must be
importable with nothing from ``osprey`` in hand (every framework import in it is
function-local, and behind the branch that needs it), which is what lets
``--dry-run`` work in a bare checkout.

Unlike that conftest, the load here happens in a **fixture** rather than at
import time. Those modules register themselves in ``sys.modules`` while
executing because their dataclasses resolve ``cls.__module__`` during class
creation; this script's dataclasses are ``from __future__ import annotations``
and are only instantiated inside tests, so per-test registration through
``monkeypatch`` does the job — and leaves ``sys.modules`` exactly as collection
found it (``tests/infrastructure/test_import_time_audit.py`` is the rule).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "demo_target_switch.py"
MODULE_NAME = "osprey_demo_target_switch"

#: The script under test, bound per test by :func:`_demo_module` below. Module
#: scope so the tests can read it as an ordinary import; never loaded at import
#: time, which is what the mutation audit is about.
demo = None

#: The ten claims, in the order the demo makes them. Pinned as a count and by
#: their numbering: a plan that silently lost a step would still print a tidy
#: trail, and that is exactly the regression this catches.
EXPECTED_STEPS = 10


@pytest.fixture(autouse=True)
def _demo_module(monkeypatch):
    """Load the script, registered in ``sys.modules`` only for this test.

    ``monkeypatch.setitem`` puts the entry back the way it found it, so nothing
    this module does is visible to a test that runs after it — the property an
    import-time registration cannot offer, because it happens during collection
    before any fixture exists to undo it.
    """
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    spec.loader.exec_module(module)
    # Bound through monkeypatch as well, so the module-level name is restored
    # to None after the test and cannot outlive the registration it names.
    monkeypatch.setitem(globals(), "demo", module)
    return module


@pytest.fixture
def dry_run(capsys):
    """Run the dry plan and hand back its exit code and its lines."""

    def run(*arguments: str) -> tuple[int, list[str]]:
        code = demo.main(["--dry-run", *arguments])
        return code, capsys.readouterr().out.splitlines()

    return run


class TestTheDryRunPrintsThePlanAndDrivesNothing:
    def test_it_exits_zero_and_numbers_every_step_once(self, dry_run):
        code, lines = dry_run("--self-provision")

        assert code == 0
        planned = [line for line in lines if line.startswith("PLAN ")]
        assert [line.split()[1] for line in planned] == [
            str(number) for number in range(1, EXPECTED_STEPS + 1)
        ]
        assert lines[-1] == "RESULT PASS"

    def test_nothing_was_driven(self, dry_run):
        """No STEP line, because a STEP line is a claim about a real machine."""
        _, lines = dry_run("--self-provision")

        assert not [line for line in lines if line.startswith("STEP ")]

    def test_the_plan_names_the_channels_it_would_touch(self, dry_run):
        code, lines = dry_run(
            "--self-provision", "--probe-channel", "X:PROBE", "--write-channel", "X:SETPOINT:SP"
        )
        joined = "\n".join(lines)

        assert code == 0
        assert "X:PROBE" in joined
        assert "X:SETPOINT:SP" in joined

    def test_the_plan_states_both_scope_limits(self, dry_run):
        """The two things the demo does NOT prove, said in its own output.

        A scope limit that lives only in a docstring is a scope limit the
        operator reading the trail never sees.
        """
        _, lines = dry_run("--self-provision")
        notes = "\n".join(line for line in lines if line.startswith("NOTE "))

        assert "name-server mode only" in notes
        assert "UDP" in notes
        assert "ASSERTED FROM CONFIG" in notes
        assert "no mongod is booted" in notes

    def test_the_plan_names_the_block_an_operator_would_edit(self, dry_run):
        """The refusal wording rule, in the plan an operator reads first.

        A trail that told them to set the deployment-wide key would be telling
        them to arm every target the deployment has, which is the opposite of
        what step 9 demonstrates.
        """
        _, lines = dry_run("--self-provision")
        joined = "\n".join(lines)

        assert "control_system.connector.<type>.writes_enabled" in joined
        assert "control_system.writes_enabled" not in joined


class TestTheAuditGrammarIsStable:
    """The trail's line shapes, which operators read and pipelines grep."""

    def test_a_step_line_names_the_active_target(self, capsys):
        audit = demo.Audit()

        audit.step(3, "live", "switching to 'va' via control_target_set")
        audit.ok(3, "va", "generation 1")
        audit.failed(4, "va", "both targets served the same value")

        assert capsys.readouterr().out.splitlines() == [
            "STEP 3 [target=live] switching to 'va' via control_target_set",
            "STEP 3 [target=va] OK generation 1",
            "STEP 4 [target=va] FAILED both targets served the same value",
        ]

    def test_a_verdict_line_names_the_failing_step_and_its_target(self, capsys):
        audit = demo.Audit()

        audit.result_fail(5, "the sandbox read the other target's value", "va")

        assert capsys.readouterr().out.strip() == (
            "RESULT FAIL step 5 [target=va]: the sandbox read the other target's value"
        )

    def test_a_broken_claim_carries_the_target_it_broke_on(self):
        """The failure object, not just the printed line — main() reads this."""
        claims = demo.Claims(lambda: "va")

        with pytest.raises(demo.StepFailed) as raised:
            claims.require(False, 4, "both targets served the same value")

        assert raised.value.number == 4
        assert raised.value.target == "va"

    def test_a_claim_that_breaks_before_a_session_exists_still_names_something(self):
        """An unreadable target degrades to a word, never to a crash in the trail."""

        def exploding() -> str:
            raise RuntimeError("no manager yet")

        claims = demo.Claims(exploding)

        with pytest.raises(demo.StepFailed) as raised:
            claims.require(False, 1, "the roster could not be read")

        assert raised.value.target == demo.UNKNOWN_TARGET


class TestTheDemoRefusesToGuessAWriteChannel:
    """A demo that guessed would write an unreviewed setpoint on their stack."""

    def test_an_operator_posture_must_name_the_setpoint(self, tmp_path, capsys):
        config = tmp_path / "config.yml"
        # A posture complete enough that the write channel is the only thing
        # missing — otherwise this would pin the probe-channel refusal instead.
        config.write_text(
            "control_system:\n"
            "  type: epics\n"
            "  connector:\n"
            "    virtual_accelerator:\n"
            "      probe_channel: X:PROBE\n",
            encoding="utf-8",
        )

        code = demo.main(["--config", str(config)])

        assert code == 1
        output = capsys.readouterr().out
        assert "--write-channel is required" in output
        assert output.strip().splitlines()[-1].startswith("RESULT FAIL")

    def test_the_refusal_lands_before_the_plan_is_printed(self, tmp_path, capsys):
        """A dry run must refuse what the real run would refuse.

        Otherwise it hands the operator a plan for an invocation that cannot
        run, which is worse than no plan at all.
        """
        config = tmp_path / "config.yml"
        config.write_text("control_system:\n  type: epics\n", encoding="utf-8")

        code = demo.main(["--dry-run", "--config", str(config)])

        assert code == 1
        output = capsys.readouterr().out
        assert "--write-channel is required" in output
        assert "PLAN " not in output


class TestThePlanNeverInventsAnOperatorsChannels:
    """What the plan may print for a posture this file did not build."""

    def test_an_operator_posture_renders_placeholders(self):
        arguments = demo.build_parser().parse_args(
            ["--config", "/somewhere/config.yml", "--write-channel", "X:SETPOINT:SP"]
        )

        probe, write = demo.planned_channels(arguments)

        assert probe == demo.PLACEHOLDER_PROBE
        assert write == "X:SETPOINT:SP"

    def test_the_write_placeholder_is_used_when_nothing_named_one(self):
        arguments = demo.build_parser().parse_args(["--config", "/somewhere/config.yml"])

        assert demo.planned_channels(arguments)[1] == demo.PLACEHOLDER_WRITE

    def test_a_self_provisioned_posture_names_the_channels_this_file_chose(self):
        arguments = demo.build_parser().parse_args(["--self-provision"])

        assert demo.planned_channels(arguments) == (
            demo.DEFAULT_PROBE_CHANNEL,
            demo.DEFAULT_WRITE_CHANNEL,
        )

    def test_an_operator_plan_prints_no_tutorial_channel(self, capsys):
        demo.main(
            ["--dry-run", "--config", "/somewhere/config.yml", "--write-channel", "X:SETPOINT:SP"]
        )
        output = capsys.readouterr().out

        assert demo.DEFAULT_PROBE_CHANNEL not in output
        assert demo.PLACEHOLDER_PROBE in output


class TestTheRefusalNamesTheBlockAnOperatorEdits:
    """Which key step 9 puts in front of an operator, per target."""

    def test_each_target_names_its_own_connector_block(self):
        section = {"type": "epics", "connector": {"epics": {}, "virtual_accelerator": {}}}

        assert demo.posture_key(section, "live") == "control_system.connector.epics.writes_enabled"
        assert (
            demo.posture_key(section, "va")
            == "control_system.connector.virtual_accelerator.writes_enabled"
        )

    def test_a_target_that_names_no_machine_falls_back_to_the_deployment_wide_key(self):
        """A mock deployment's 'live' resolves to no connector type, so there is
        no block to send anyone to — and that deployment only ever had the one
        posture anyway."""
        assert demo.posture_key({"type": "mock"}, "live") == "control_system.writes_enabled"


class TestARefusalIsReadFromItsEnvelope:
    """Step 9 reads a raised tool error; it must never fail on the plumbing."""

    def test_the_standard_envelope_is_parsed(self):
        raised = RuntimeError(
            '{"error": true, "error_type": "write_refused", '
            '"details": {"reason": "WRITES_DISABLED"}}'
        )

        envelope = demo._error_envelope(raised)

        assert envelope["error_type"] == "write_refused"
        assert envelope["details"]["reason"] == "WRITES_DISABLED"

    def test_anything_else_is_handed_back_with_its_text_intact(self):
        envelope = demo._error_envelope(TimeoutError("the connector host never answered"))

        assert envelope["error_type"] == "unparsed"
        assert "the connector host never answered" in envelope["error_message"]


class TestTheLimitsNoteIsReadRatherThanAsserted:
    """The strict-limits line states what the database actually contains."""

    def test_the_demo_defaults_are_confirmed_present(self):
        note = demo.limits_note(demo.DEFAULT_PROBE_CHANNEL, demo.DEFAULT_WRITE_CHANNEL)

        assert demo.DEFAULT_PROBE_CHANNEL in note
        assert demo.DEFAULT_WRITE_CHANNEL in note
        assert "does NOT list" not in note

    def test_an_overridden_channel_that_is_absent_is_called_out(self):
        note = demo.limits_note(demo.DEFAULT_PROBE_CHANNEL, "X:NOT:IN:THE:DATABASE")

        assert "does NOT list X:NOT:IN:THE:DATABASE" in note
        assert "refused" in note


class TestTheWriteIsRestoredOrDisclosed:
    """Step 6 never leaves a machine changed without saying so.

    The restore path itself needs two live endpoints and is exercised by the
    real run; what is pinned here is the decision table in front of it, which is
    where a silent leftover would come from.
    """

    @staticmethod
    def _posture(*, self_provisioned: bool) -> demo.Posture:
        return demo.Posture(
            config_path=Path("/nowhere/config.yml"),
            raw={},
            probe_channel="X:PROBE",
            write_channel="X:SETPOINT:SP",
            write_value=1.0,
            self_provisioned=self_provisioned,
        )

    def test_a_run_that_never_wrote_has_nothing_to_settle(self):
        ledger = demo.WriteLedger(channel="X:SETPOINT:SP")

        assert (
            demo.settle_action(self._posture(self_provisioned=True), ledger) == demo.SETTLE_NOTHING
        )
        assert (
            demo.settle_action(self._posture(self_provisioned=False), ledger) == demo.SETTLE_NOTHING
        )

    def test_a_write_that_may_not_have_landed_is_still_settled(self):
        """``attempted`` is the trigger, not ``landed``: a write whose readback
        never happened is exactly the one an operator must be told about."""
        ledger = demo.WriteLedger(channel="X:SETPOINT:SP", original=0.0, attempted=True)

        assert (
            demo.settle_action(self._posture(self_provisioned=False), ledger) == demo.SETTLE_RESTORE
        )
        assert "may have reached" in demo.leftover_sentence(
            self._posture(self_provisioned=False), ledger
        )

    def test_a_scratch_stack_discloses_the_value_it_leaves_behind(self):
        """Nothing to restore — the container is about to be removed — but the
        operator is still told what was on it and at what value."""
        ledger = demo.WriteLedger(
            channel="X:SETPOINT:SP", original=0.0, attempted=True, landed=True
        )
        posture = self._posture(self_provisioned=True)

        assert demo.settle_action(posture, ledger) == demo.SETTLE_DISCLOSE
        note = demo.scratch_disclosure(posture, ledger)
        assert "X:SETPOINT:SP" in note
        assert "NOT restored" in note
        assert "removes it on the way out" in note

    def test_the_leftover_sentence_names_the_value_to_put_back(self):
        """What every failed-restore line carries, so none of them can be vague."""
        ledger = demo.WriteLedger(
            channel="X:SETPOINT:SP", original=0.25, attempted=True, landed=True
        )

        sentence = demo.leftover_sentence(self._posture(self_provisioned=False), ledger)

        assert "2.500000e-01" in sentence  # the value it read before writing
        assert "put it back by hand" in sentence
        assert "landed on" in sentence

    def test_a_value_that_was_never_read_is_said_to_be_unread(self):
        """No invented original: "unknown" and "zero" are different claims."""
        ledger = demo.WriteLedger(channel="X:SETPOINT:SP", attempted=True)

        assert "an unread value" in demo.leftover_sentence(
            self._posture(self_provisioned=False), ledger
        )


class TestASmallSeparationIsReportedAsSuch:
    """Step 4's distinctness, when this demo did not seed the difference."""

    @staticmethod
    def _operator_posture() -> demo.Posture:
        return demo.Posture(
            config_path=Path("/nowhere/config.yml"),
            raw={},
            probe_channel="X:PROBE",
            write_channel="X:SETPOINT:SP",
            write_value=1.0,
            self_provisioned=False,
        )

    def test_a_hair_of_difference_carries_the_jitter_caveat(self):
        caveat = demo._separation_caveat(self._operator_posture(), 1.0, 1.0 + 1e-9)

        assert "CAVEAT" in caveat
        assert "step 7" in caveat

    def test_a_clear_difference_carries_none(self):
        assert demo._separation_caveat(self._operator_posture(), 1.0, 2.0) == ""

    def test_a_seeded_posture_is_judged_by_its_seed_instead(self):
        """The self-provisioned run holds the reading to the separation it made,
        so it has no use for a caveat about magnitudes it did not choose."""
        seeded = demo.Posture(
            config_path=Path("/nowhere/config.yml"),
            raw={},
            probe_channel="X:PROBE",
            write_channel="X:SETPOINT:SP",
            write_value=1.0,
            self_provisioned=True,
            expected_separation=demo.SEEDED_OFFSET_X,
        )

        assert demo._separation_caveat(seeded, 1.0, 1.0 + 1e-9) == ""
