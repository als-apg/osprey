"""Tests for the target identity the `osprey_approval` hook renders.

Every approval prompt carries one line naming the control-system target the
deployment is pointed at — `Target: Live — writes move hardware
(<endpoint>)`, `Target: Rehearsal — nothing moves (<endpoint>)`, `Target: Simulator
— nothing moves`, or, when the context cannot be resolved, the explicit
`Target: deployment baseline (state unavailable)`. The third spelling is the
whole point of the feature: a prompt with no target line at all reads as "not
the machine", so the line is emitted from `build_approval_output` — the single
funnel every ask goes through — rather than assembled per branch.

The identity comes from two files the deployment's own processes write, never
from the rendered config.yml this hook also holds: config states what the
deployment STARTS as, and after a run-time switch that would be a confident,
stale, wrong safety claim. The control-context record
(`control_target/<identity>/control_context.json`) says WHICH target the
deployment is on; the live controls servers' reports
(`control_target/<identity>/server_<pid>.json`) say what that target IS. `read_target_view` folds them into one read, which is what keeps
the "where you are" line and the "where you would be" line from straddling a
switch. The WORDS a target is named by come from its machine kind as the
reader's `published_machine_kind` answers it — never from the label a writer
minted or its `real_machine` claim — so a `standin` reads as the rehearsal only
while a live server publishes it as this deployment's stand-in.

Two levels of test, deliberately:

* in-process, through the `conftest.import_hook` seam, driving the REAL reader
  module (`osprey_target_state`) against real files in `tmp_path`. Only the
  reader's documented path seam — `resolve_state_dir` — is replaced, so
  "unreadable" really is a corrupt record and a dead server's metadata really is
  filtered by the liveness rule, not by a stubbed return value; and
* end to end, running the hook as a subprocess the way Claude Code does, with a
  record and a report under the repo root the hook derives from its own `cwd`,
  so the root anchor, the report read and the prompt assembly are all exercised
  for real.
"""

from __future__ import annotations

import json
import os

import pytest

from osprey_connectors_tests._harness import (
    pin_identity,
    state_dir_under,
    write_control_context,
    write_payload,
    write_server_report,
)
from tests.hooks.conftest import ask_reason

#: The exact line an approver must see whenever the target cannot be resolved.
BASELINE_LINE = "Target: deployment baseline (state unavailable)"

#: The owner-facing words for each machine (C-WORDS, D4): the package's
#: ``KIND_BANNERS``, restated in the hook.
REAL_WORDS = "Live — writes move hardware"
REHEARSAL_WORDS = "Rehearsal — nothing moves"
SIM_WORDS = "Simulator — nothing moves"

#: The warning a switch towards the real machine carries.
LIVE_WARNING = "THIS SWITCH POINTS THE SESSION AT THE REAL MACHINE"

LIVE_ENDPOINT = "pva://live-gw.example.org:5075"
VA_ENDPOINT = "pva://127.0.0.1:5074"
STANDIN_ENDPOINT = "pva://127.0.0.1:5077"

#: The PID of the controls server whose report publishes the display metadata.
#: Alive by construction in every test that asks for `alive_everything`.
SERVER_PID = 5150

#: The generation the record carries. Any non-negative int parses; a value other
#: than 0 keeps "the target moved" distinguishable from "never set".
GENERATION = 4

#: The metadata a writer records for the stand-in slot. Written into the report
#: by the tests that need it rather than into :func:`published_targets`, because
#: a deployment without a ``live_standin`` block publishes no such slot at all.
#: ``machine_kind`` is what the controls server publishes while the slot really
#: is this deployment's stand-in.
STANDIN_META = {
    "label": REHEARSAL_WORDS,
    "endpoint": STANDIN_ENDPOINT,
    "real_machine": False,
    "machine_kind": "rehearsal",
    "probe_channel": "SR:BEAM:CURRENT",
}

HOOK_CONFIG = {
    "server_prefixes": ["mcp__controls__", "mcp__workspace__"],
    "approval_prefixes": ["mcp__controls__", "mcp__workspace__"],
}


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def deployment(tmp_path, reader, monkeypatch):
    """An empty agent-data root the reader is pointed at.

    Returns the ROOT, which is what the shared writers take; the directory the
    two files land in is this identity's under ``<root>/control_target``, and
    the reader is aimed at that same directory through its one path seam. Both
    halves come from :func:`state_dir_under` rather than from a join here: a
    seam pointed one directory above the writers would leave every test in this
    module reading the baseline, which is a real answer and would look like a
    rendering bug rather than a fixture one.

    The identity is pinned so the directory does not depend on the account
    running pytest. These tests drive the reader IN PROCESS; the end-to-end
    tests below run the hook as a subprocess, which resolves its own identity
    from a curated environment, and take neither this fixture nor the pin.
    """
    pin_identity(monkeypatch)
    state_dir_under(tmp_path).mkdir(parents=True)
    monkeypatch.setattr(
        reader, "resolve_state_dir", lambda hook_input=None: str(state_dir_under(tmp_path))
    )
    return tmp_path


@pytest.fixture
def alive_everything(reader, monkeypatch):
    """Treat every PID as alive unless a test says otherwise.

    The reports carry the display metadata and are filtered by the liveness of
    the PID in their filename, so a suite writing a report for a PID it made up
    has to say that PID is running.
    """
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: True)


def published_targets(**overrides):
    """The per-target display metadata a live controls server publishes."""
    targets = {
        "live": {
            "label": REAL_WORDS,
            "endpoint": LIVE_ENDPOINT,
            "real_machine": True,
            "probe_channel": "RING:BEAM:CURRENT",
        },
        "va": {
            "label": "Virtual accelerator",
            "endpoint": VA_ENDPOINT,
            "real_machine": False,
            "probe_channel": "VA:BEAM:CURRENT",
        },
    }
    targets.update(overrides)
    return targets


def write_state(root, target="va", targets=None, server_pid=SERVER_PID):
    """Write the record and one server report that publishes *targets*.

    The pair is what a running deployment has on disk, and what
    ``read_target_view`` reads: the record names the target, the report names
    the machines. Written through the shared fixture writers so a schema change
    fails every suite downstream of the control context at once.
    """
    write_control_context(root, target=target, generation=GENERATION)
    write_server_report(
        root,
        server_pid,
        applied_target=target,
        applied_generation=GENERATION,
        targets=published_targets() if targets is None else targets,
    )


# ---------------------------------------------------------------------------
# the read rules stay in the reader
# ---------------------------------------------------------------------------


def test_a_stale_report_that_sorts_first_never_answers_for_the_live_one(
    approval, reader, deployment, monkeypatch
):
    """A crashed server's leftover report must not shadow the running server's.

    Both reports publish metadata for the same target, so the ONLY thing
    separating them is the liveness of the PID in their filename — and the
    stale one sorts first, which is the order a directory walk sees them in.
    The hook reads the metadata through the reader's own view precisely so that
    this filter cannot be left out: the prompt must show the live endpoint, not
    the endpoint the crashed server was pointed at.

    Reports are never deleted by a reader, so a deployment that has been
    running for a while genuinely does have both files on disk.
    """
    stale_pid, live_pid = 1111, 9999
    assert f"server_{stale_pid}.json" < f"server_{live_pid}.json"
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: int(pid) != stale_pid)

    stale_targets = published_targets()
    stale_targets["live"] = {
        "label": "Crashed session's ring",
        "endpoint": "pva://stale-gw.invalid:5075",
        "real_machine": True,
        "probe_channel": "STALE:BEAM:CURRENT",
    }
    write_state(deployment, server_pid=stale_pid, target="live", targets=stale_targets)
    write_server_report(deployment, live_pid, applied_target="live", targets=published_targets())

    assert approval._target_line() == f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})"

    lines = approval._describe_control_target_set({"target": "live"}, {})
    assert f"Destination: {REAL_WORDS} ({LIVE_ENDPOINT})" in lines
    assert not any("stale-gw" in line or "Crashed" in line for line in lines)


@pytest.mark.usefixtures("alive_everything")
def test_the_hook_reads_the_record_through_the_readers_own_view(approval, reader, deployment):
    """`read_target_view` is the seam; the hook holds no copy of the rules."""
    write_state(deployment, target="live")

    assert approval._session_state_record() == reader.read_target_view()


# ---------------------------------------------------------------------------
# _target_line: the resolved cases
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("alive_everything")
def test_live_target_names_the_machine_and_its_endpoint(approval, deployment):
    """A real-machine target renders LOUD, with the endpoint the writer selected.

    The endpoint is whichever role the writer selects under the session's
    effective posture, republished whenever that posture changes, and reaches
    the prompt verbatim: picking a role here would be a second opinion about
    which gateway the session holds.
    """
    write_state(deployment, target="live")

    assert approval._target_line() == f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})"


@pytest.mark.usefixtures("alive_everything")
def test_a_label_never_softens_the_real_machine_words(approval, deployment):
    """The words come from the machine kind, never from the label a writer minted.

    A ``live`` target whose label says the rehearsal's words is still the real machine by
    kind, so the line names it as one: a label is display text, and a prompt
    that let it soften the claim would be reading the one source C-KIND exists
    to replace (CF-7).
    """
    targets = published_targets()
    targets["live"] = {
        "label": REHEARSAL_WORDS,
        "endpoint": "127.0.0.1:5074",
        "real_machine": False,
        "machine_kind": "rehearsal",
        "probe_channel": "SR:BEAM:CURRENT",
    }
    write_state(deployment, target="live", targets=targets)

    assert approval._target_line() == f"Target: {REAL_WORDS} (127.0.0.1:5074)"

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert f"Destination: {REAL_WORDS} (127.0.0.1:5074)" in lines
    assert any(LIVE_WARNING in line for line in lines)
    assert not any("stand-in" in line or REHEARSAL_WORDS in line for line in lines)


@pytest.mark.usefixtures("alive_everything")
def test_the_live_standin_reads_as_the_rehearsal(approval, deployment):
    """A ``standin`` a live server publishes as this deployment's stand-in.

    The rehearsal words, with the endpoint beside them, on the identity line.
    """
    targets = published_targets(standin=dict(STANDIN_META))
    write_state(deployment, target="standin", targets=targets)

    assert approval._target_line() == f"Target: {REHEARSAL_WORDS} ({STANDIN_ENDPOINT})"


@pytest.mark.parametrize(
    "meta",
    [
        dict(STANDIN_META, machine_kind="real"),
        {k: v for k, v in STANDIN_META.items() if k != "machine_kind"},
        dict(STANDIN_META, machine_kind="odd"),
    ],
    ids=["published-as-real", "no-kind", "odd-kind"],
)
@pytest.mark.usefixtures("alive_everything")
def test_a_standin_that_is_not_this_deployments_reads_as_the_real_machine(
    approval, deployment, meta
):
    """Only the real stand-in says "nothing moves".

    A ``standin`` target pointing anywhere else is published as ``real``; a
    report that states no kind, or one this hook cannot read, reads the same.
    Its label still says the rehearsal's words and its ``real_machine`` claim still says
    ``False`` — neither is consulted, so neither can soften the words.
    """
    write_state(deployment, target="standin", targets=published_targets(standin=meta))

    assert approval._target_line() == f"Target: {REAL_WORDS} ({STANDIN_ENDPOINT})"

    lines = approval._describe_control_target_set({"target": "standin"}, {})

    assert f"Destination: {REAL_WORDS} ({STANDIN_ENDPOINT})" in lines
    assert any(LIVE_WARNING in line for line in lines)
    assert not any(REHEARSAL_WORDS in line or "nothing moves" in line for line in lines)


def test_a_standin_with_no_live_server_renders_the_baseline_line(
    approval, reader, deployment, monkeypatch
):
    """A dead server's rehearsal claim is no claim: the reader ignores it.

    The record still names ``standin``, but nothing alive publishes it, so the
    line degrades to the baseline rather than saying "nothing moves" for a
    process that is gone.
    """
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: False)
    write_state(deployment, target="standin", targets=published_targets(standin=STANDIN_META))

    assert approval._target_line() == BASELINE_LINE
    assert reader.published_machine_kind("standin") == "real"


@pytest.mark.usefixtures("alive_everything")
def test_a_live_record_without_a_label_still_names_the_machine(approval, deployment):
    """An older writer recorded no label — the words do not depend on one."""
    targets = published_targets()
    targets["live"] = {"endpoint": LIVE_ENDPOINT, "real_machine": True}
    write_state(deployment, target="live", targets=targets)

    assert approval._target_line() == f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})"


@pytest.mark.usefixtures("alive_everything")
def test_virtual_target_names_the_simulation(approval, deployment):
    """A simulation target says so in words, without an endpoint to misread."""
    write_state(deployment, target="va")

    assert approval._target_line() == f"Target: {SIM_WORDS}"


@pytest.mark.usefixtures("alive_everything")
def test_live_target_without_a_recorded_endpoint_says_so(approval, deployment):
    """An endpoint the writer never recorded must not render as empty parentheses."""
    targets = published_targets()
    targets["live"] = {"label": REAL_WORDS, "endpoint": "", "real_machine": True}
    write_state(deployment, target="live", targets=targets)

    assert approval._target_line() == f"Target: {REAL_WORDS} (endpoint not recorded)"


# ---------------------------------------------------------------------------
# _target_line: the fallback is explicit, for every reason
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("deployment", "alive_everything")
def test_no_state_at_all_renders_the_explicit_baseline_line(approval, reader):
    """No record at all — no switch capability, or nothing started yet.

    This is the ordinary case on most deployments, and the line still renders:
    the plan's "never silently missing" applies to the boring reason as much as
    to the alarming ones.
    """
    assert reader.read_target()["reason"] == reader.REASON_NO_STATE

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.usefixtures("alive_everything")
def test_a_record_with_no_live_server_renders_the_baseline_line(approval, reader, deployment):
    """The record says WHICH target; only a live server says what it IS.

    A deployment whose record survives a shutdown knows the target it was left
    on, and nothing at all about the machine behind it. Naming that target
    anyway would put a label on the prompt that no running process stands
    behind, so the line degrades to the baseline instead.
    """
    write_control_context(deployment, target="live", generation=GENERATION)

    assert reader.read_target()["target"] == "live"

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.usefixtures("alive_everything")
def test_corrupt_state_renders_the_baseline_line(approval, reader, deployment):
    """A truncated or corrupt record resolves to the baseline, not to silence."""
    record = state_dir_under(deployment) / reader.RECORD_FILENAME
    record.write_text("{not json", encoding="utf-8")

    assert reader.read_target()["reason"] == reader.REASON_UNREADABLE

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.usefixtures("alive_everything")
def test_a_record_outside_the_target_vocabulary_renders_the_baseline_line(
    approval, reader, deployment
):
    """A target name this reader does not know is no target at all.

    The identity fields are all-or-nothing: a name outside the vocabulary comes
    from a writer this hook cannot read, and there is no safe guess between
    "the simulator" and "the machine" to make on its behalf.
    """
    write_payload(
        state_dir_under(deployment) / reader.RECORD_FILENAME,
        {"schema": 1, "target": "staging", "generation": GENERATION},
    )

    assert reader.read_target()["reason"] == reader.REASON_UNREADABLE

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.parametrize(
    "meta",
    [
        {"label": "Virtual accelerator", "endpoint": LIVE_ENDPOINT},
        {"label": "Virtual accelerator", "endpoint": LIVE_ENDPOINT, "real_machine": None},
        {"label": "Virtual accelerator", "endpoint": LIVE_ENDPOINT, "real_machine": False},
        {"label": "Virtual accelerator", "endpoint": LIVE_ENDPOINT, "real_machine": 0},
    ],
    ids=["absent", "null", "false", "int"],
)
@pytest.mark.usefixtures("alive_everything")
def test_the_real_machine_claim_is_not_consulted(approval, deployment, meta):
    """The words come from the machine kind, so no claim can soften them.

    A ``live`` target is the real machine by kind whatever its ``real_machine``
    claim says — absent, null, ``False`` or a falsy ``0`` — and whatever its
    label says. Before C-KIND these rendered the baseline (or, for ``False``,
    "simulation"); a second source beside the kind is exactly what lets two
    surfaces disagree about one machine.
    """
    write_state(deployment, target="live", targets={"live": meta})

    assert approval._target_line() == f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})"


@pytest.mark.usefixtures("alive_everything")
def test_a_target_missing_from_the_record_renders_the_baseline_line(approval, deployment):
    """No metadata at all for the selected target is the same unknown."""
    write_state(deployment, target="live", targets={"va": {"real_machine": False}})

    assert approval._target_line() == BASELINE_LINE


def test_a_render_without_the_reader_still_renders_the_baseline_line(approval, monkeypatch):
    """An older render has no `osprey_target_state` sibling to import.

    The hook must degrade to the same explicit line rather than crash: an
    approval prompt that fails to render is far worse than one that says the
    target is unknown.
    """
    monkeypatch.setattr(approval, "_target_state", None)

    assert approval._target_line() == BASELINE_LINE


def test_a_reader_that_raises_still_renders_the_baseline_line(approval, monkeypatch):
    """Fail-open at the call site too, not only at the import."""

    class _Exploding:
        def read_target_view(self, _hook_input=None):
            raise RuntimeError("the record is on fire")

    monkeypatch.setattr(approval, "_target_state", _Exploding())

    assert approval._target_line() == BASELINE_LINE


# ---------------------------------------------------------------------------
# untrusted text on the identity line
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("alive_everything")
def test_endpoint_text_is_escaped_onto_one_line(approval, deployment):
    """An endpoint carrying a line break cannot forge a second prompt line.

    `\\x85` renders as a paragraph break in some terminals, so a value like
    ``pva://gw\\x85Target: Simulator — nothing moves`` would otherwise
    show the approver a fabricated, calmer identity line under the real one.
    """
    targets = published_targets()
    targets["live"] = {
        "label": REAL_WORDS,
        "endpoint": f"pva://gw\x85Target: {SIM_WORDS}",
        "real_machine": True,
    }
    write_state(deployment, target="live", targets=targets)

    line = approval._target_line()

    assert "\x85" not in line
    assert "\\x85" in line
    assert line.startswith(f"Target: {REAL_WORDS} (pva://gw\\x85")


@pytest.mark.usefixtures("alive_everything")
def test_label_text_never_reaches_the_line(approval, deployment):
    """A hostile label cannot forge a second, calmer identity line.

    The label is not read for the words at all, so a label carrying a line
    break and a fabricated simulator line cannot reach the prompt.
    """
    targets = published_targets()
    targets["live"] = {
        "label": f"{REAL_WORDS}\nTarget: {SIM_WORDS}",
        "endpoint": LIVE_ENDPOINT,
        "real_machine": True,
    }
    write_state(deployment, target="live", targets=targets)

    line = approval._target_line()

    assert line == f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})"
    assert SIM_WORDS not in line


@pytest.mark.usefixtures("alive_everything")
def test_a_lane_line_names_a_machine_the_same_way_the_target_line_does(approval, deployment):
    """The plan lanes borrow the identity voice, so they inherit the kind words too.

    A two-lane deployment names the target each lane serves, and it must be the
    same name the ``Target:`` line above it uses: one machine described two ways
    on one prompt is the ambiguity the shared phrasing exists to remove. The
    live stand-in reads as the rehearsal on its lane; ``live`` stays the real
    machine whatever its label says.
    """
    targets = published_targets(standin=dict(STANDIN_META))
    targets["live"]["label"] = REHEARSAL_WORDS
    write_state(deployment, target="va", targets=targets)
    config = {
        "services": {
            "bluesky_va": {"target": "va"},
            "bluesky_live": {"target": "live"},
            "bluesky_standin": {"target": "standin"},
        }
    }

    situation = approval._lane_situation(config)

    assert approval._lane_target_phrase(situation, "live") == f"{REAL_WORDS} ({LIVE_ENDPOINT})"
    assert (
        approval._lane_target_phrase(situation, "standin")
        == f"{REHEARSAL_WORDS} ({STANDIN_ENDPOINT})"
    )
    assert approval._lane_target_phrase(situation, "va") == SIM_WORDS


# ---------------------------------------------------------------------------
# the line reaches every prompt, through the one envelope builder
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("alive_everything")
def test_every_ask_envelope_carries_the_target_line(approval, deployment):
    """The identity line sits under the headline, above the tool detail.

    Placement matters: a long enrichment block (a whole queue listing, a plan's
    source) must not be able to push the target off the approver's screen.
    """
    write_state(deployment, target="live")

    reason = approval.build_approval_output("Tool: channel_write")["hookSpecificOutput"][
        "permissionDecisionReason"
    ]

    assert f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})" in reason
    assert reason.index(REAL_WORDS) < reason.index("Tool: channel_write")
    assert reason.endswith("Review the operation above and approve to proceed.")


def test_the_ask_envelope_carries_the_baseline_line_when_state_is_absent(approval, monkeypatch):
    """No prompt is ever emitted without a target line of some kind."""
    monkeypatch.setattr(approval, "_target_state", None)

    reason = approval.build_approval_output("Tool: setup_patch")["hookSpecificOutput"][
        "permissionDecisionReason"
    ]

    assert BASELINE_LINE in reason


# ---------------------------------------------------------------------------
# control_target_set: previewing the destination of a prospective switch
# ---------------------------------------------------------------------------


def test_the_switch_describer_is_registered_under_its_short_tool_name(approval):
    """Registration is what makes the describer reachable once the tool ships."""
    assert (
        approval._TARGET_DESCRIBERS["control_target_set"] is approval._describe_control_target_set
    )


@pytest.mark.usefixtures("alive_everything")
def test_switch_to_live_renders_destination_endpoint_and_probe_channel(approval, deployment):
    """The destination is read out of the same state file as the current target.

    The deployment is on the VA here, so the destination's metadata is precisely
    what the identity line cannot answer for.
    """
    write_state(deployment, target="va")

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert any(LIVE_WARNING in line for line in lines)
    assert f"Destination: {REAL_WORDS} ({LIVE_ENDPOINT})" in lines
    assert "Destination probe channel: RING:BEAM:CURRENT" in lines


@pytest.mark.usefixtures("alive_everything")
def test_switch_to_the_simulation_carries_no_live_machine_warning(approval, deployment):
    """Switching away from the machine is not the alarming direction."""
    write_state(deployment, target="live")

    lines = approval._describe_control_target_set({"target": "va"}, {})

    assert f"Destination: {SIM_WORDS} ({VA_ENDPOINT})" in lines
    assert "Destination probe channel: VA:BEAM:CURRENT" in lines
    assert not any(LIVE_WARNING in line or REAL_WORDS in line for line in lines)


@pytest.mark.parametrize(
    ("machine_kind", "words"),
    [("rehearsal", REHEARSAL_WORDS), ("real", REAL_WORDS)],
    ids=["the-stand-in", "pointing-elsewhere"],
)
@pytest.mark.usefixtures("alive_everything")
def test_switch_to_the_standin_names_its_published_kind(approval, deployment, machine_kind, words):
    """The destination's words come from the kind the controls server publishes.

    ``standin`` is the rehearsal only while the server publishes it as this
    deployment's stand-in; pointing anywhere else it is published ``real`` and
    reads, and warns, as the real machine. The label (the rehearsal's words)
    and the ``real_machine`` claim (``False``) are the same in both cases, so
    the only thing that can move the words is the kind.
    """
    targets = published_targets(standin=dict(STANDIN_META, machine_kind=machine_kind))
    write_state(deployment, target="va", targets=targets)

    lines = approval._describe_control_target_set({"target": "standin"}, {})

    assert f"Destination: {words} ({STANDIN_ENDPOINT})" in lines
    assert "Destination probe channel: SR:BEAM:CURRENT" in lines
    warned = any(LIVE_WARNING in line for line in lines)
    assert warned is (machine_kind == "real")
    assert not any("stand-in" in line for line in lines)


@pytest.mark.usefixtures("alive_everything")
def test_a_standin_destination_on_a_record_that_has_no_such_slot_is_reported(approval, deployment):
    """A deployment that stood up no stand-in records no slot for one.

    The switch would be refused by the tool itself; the prompt's job is to say
    it cannot show an endpoint rather than to invent one from the target name.
    """
    write_state(deployment, target="va")

    lines = approval._describe_control_target_set({"target": "standin"}, {})

    assert lines == [
        "Destination: standin — no controls server records metadata for it. "
        "Approval is not blocked; the endpoint cannot be shown."
    ]


def test_a_standin_baseline_deployment_names_its_lanes_for_the_standin(approval):
    """The lane baseline follows the deployment's own connector type.

    A lane block that declares no ``target`` serves the deployment baseline, and
    a ``live_standin`` deployment's baseline is ``standin``. Answering ``live``
    here would name the facility's own machine over a deployment that is not
    wired to it — the one direction a wrong answer must never go.
    """
    standin = {"control_system": {"type": "live_standin"}}
    facility = {"control_system": {"type": "epics"}}

    assert approval._baseline_target(standin) == "standin"
    assert approval._baseline_target(facility) == "live"
    assert approval._rendered_lanes(standin) == [("bluesky", "standin")]


@pytest.mark.usefixtures("alive_everything")
def test_a_destination_without_a_probe_channel_simply_omits_the_line(approval, deployment):
    """Schema tolerance: a writer that records no probe channel is not an error.

    The live block ships its probe channel commented out on purpose — a
    facility's probe channel cannot be guessed — so a record lacking the key is
    the shipped default, not corruption.
    """
    targets = published_targets()
    targets["live"].pop("probe_channel")
    write_state(deployment, target="va", targets=targets)

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert f"Destination: {REAL_WORDS} ({LIVE_ENDPOINT})" in lines
    assert not any("probe channel" in line for line in lines)


@pytest.mark.usefixtures("alive_everything")
def test_a_destination_that_makes_no_machine_claim_still_reads_its_kind(approval, deployment):
    """Silence about ``real_machine`` no longer matters: the kind answers.

    ``live`` is the real machine by kind, so the destination warns and reads as
    one even when its writer recorded no claim.
    """
    targets = published_targets()
    targets["live"].pop("real_machine")
    write_state(deployment, target="va", targets=targets)

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert any(LIVE_WARNING in line for line in lines)
    assert not any("records whether this destination" in line for line in lines)
    assert f"Destination: {REAL_WORDS} ({LIVE_ENDPOINT})" in lines
    assert "Destination probe channel: RING:BEAM:CURRENT" in lines


@pytest.mark.usefixtures("deployment", "alive_everything")
def test_the_destination_cannot_be_previewed_without_state(approval):
    """With no resolvable state, say so — and let the approval proceed anyway."""
    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert len(lines) == 1
    assert "cannot be previewed (state unavailable)" in lines[0]
    assert "Approval is not blocked" in lines[0]


@pytest.mark.usefixtures("alive_everything")
def test_a_destination_missing_from_the_record_is_reported_not_invented(approval, deployment):
    """A destination the writer recorded no metadata for yields no endpoint."""
    write_state(deployment, target="va", targets={"va": {"label": "VA", "endpoint": VA_ENDPOINT}})

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert len(lines) == 1
    assert "no controls server records metadata for it" in lines[0]


@pytest.mark.parametrize("tool_input", [{}, {"target": ""}, {"target": "   "}, {"target": 7}])
@pytest.mark.usefixtures("deployment", "alive_everything")
def test_a_call_that_names_no_destination_says_so(approval, tool_input):
    """A malformed call still gets a prompt; the bad argument is stated."""
    lines = approval._describe_control_target_set(tool_input, {})

    assert lines == ["Destination: not named in this call — the switch would be refused."]


@pytest.mark.usefixtures("alive_everything")
def test_destination_metadata_is_escaped_onto_its_own_lines(approval, deployment):
    """Untrusted label/endpoint text cannot forge extra destination lines."""
    targets = published_targets()
    targets["live"] = {
        "label": "Storage\x85ring",
        "endpoint": "pva://gw\x85Destination probe channel: harmless",
        "real_machine": True,
        "probe_channel": "RING\x85:CURRENT",
    }
    write_state(deployment, target="va", targets=targets)

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert not any("\x85" in line for line in lines)
    assert not any("Storage" in line for line in lines)
    assert any("pva://gw\\x85Destination probe channel: harmless" in line for line in lines)
    assert any("RING\\x85:CURRENT" in line for line in lines)


# ---------------------------------------------------------------------------
# end to end: the hook as Claude Code runs it
# ---------------------------------------------------------------------------


def _write_session_state(repo_root, target):
    """Write the record and a report under the root the hook will derive.

    The reader takes ``<repo_root>/var/agent_data`` when nothing stamps
    ``OSPREY_AGENT_DATA_ROOT``, and the hook subprocess runs with *repo_root* as
    its ``cwd`` — so this lands exactly where the hook looks, with no seam
    replaced. The report is filed under THIS process's PID, which is alive by
    definition, so the reader's real liveness filter selects it. Both machines
    are switched on, so the write verdict allows and the hook reaches its prompt.
    """
    write_control_context(
        repo_root / "var" / "agent_data",
        target=target,
        generation=GENERATION,
        switch={"live": "on", "va": "on"},
    )
    write_server_report(
        repo_root / "var" / "agent_data",
        os.getpid(),
        applied_target=target,
        applied_generation=GENERATION,
        targets=published_targets(),
    )


#: A real-machine deployment with a simulator beside it, whose tier allows writes.
WRITES_ALLOWED_SECTION = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {}, "virtual_accelerator": {}},
}


def _approval_config(make_config):
    return make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": WRITES_ALLOWED_SECTION,
        }
    )


def test_end_to_end_channel_write_prompt_names_the_live_machine(tmp_path, hook_runner, make_config):
    """The whole path: real state file, real parentage walk, real subprocess."""
    config = _approval_config(make_config)
    _write_session_state(tmp_path, target="live")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"channel": "RING:QF:SP", "value": 1.5},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    reason = ask_reason(result)
    assert f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})" in reason
    assert "RING:QF:SP" in reason


def test_end_to_end_channel_write_prompt_names_the_simulation(tmp_path, hook_runner, make_config):
    """The same path, pointed at the virtual accelerator."""
    config = _approval_config(make_config)
    _write_session_state(tmp_path, target="va")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"channel": "RING:QF:SP", "value": 1.5},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    assert f"Target: {SIM_WORDS}" in ask_reason(result)


def test_end_to_end_a_non_write_tool_carries_the_baseline_line(tmp_path, hook_runner, make_config):
    """A tool that moves no hardware gets the line too, and on a stateless repo.

    Both halves matter: the approver of a patch has to know where the session
    points, and a deployment with no switch capability must still say so out
    loud rather than leave the line off.
    """
    config = _approval_config(make_config)

    result = hook_runner(
        "osprey_approval.py",
        "mcp__workspace__setup_patch",
        {"path": "config.yml"},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    reason = ask_reason(result)
    assert BASELINE_LINE in reason
    assert "Tool: setup_patch" in reason


@pytest.mark.parametrize(
    "approval_config",
    [
        {"enabled": True, "default_policy": "always", "tools": {"channel_write": "skip"}},
        {"enabled": False, "default_policy": "always"},
    ],
    ids=["policy-skip", "approval-disabled"],
)
def test_end_to_end_an_allowed_call_carries_no_target_line(
    tmp_path, hook_runner, make_config, approval_config
):
    """The identity line belongs to prompts, and only to prompts.

    Nobody is being asked anything on the allow path, so there is no approver to
    inform — and an `allow` envelope carrying a machine identity would be a
    safety claim made to a decision that was never a decision.
    """
    config = make_config({"approval": approval_config, "control_system": WRITES_ALLOWED_SECTION})
    _write_session_state(tmp_path, target="live")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"channel": "RING:QF:SP", "value": 1.5},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "Target:" not in json.dumps(result)
    assert REAL_WORDS not in json.dumps(result)


def test_end_to_end_switch_prompt_previews_the_destination(tmp_path, hook_runner, make_config):
    """`control_target_set` reaches the describer once it flows through the hook."""
    config = _approval_config(make_config)
    _write_session_state(tmp_path, target="va")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__control_target_set",
        {"target": "live"},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    reason = ask_reason(result)
    assert f"Target: {SIM_WORDS}" in reason
    assert f"Destination: {REAL_WORDS} ({LIVE_ENDPOINT})" in reason
    assert "Destination probe channel: RING:BEAM:CURRENT" in reason


# ---------------------------------------------------------------------------
# DRA: the rehearsal asks exactly when the real machine would
# ---------------------------------------------------------------------------

#: A facility deployment with its live stand-in beside the real machine, whose
#: tier allows writes.
STANDIN_SECTION = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {}, "live_standin": {"prefix": "S:"}},
}


def _write_standin_state(repo_root, target, switch, standin_kind="rehearsal"):
    """The record (on *target*, with *switch*) and a live report naming both machines."""
    write_control_context(
        repo_root / "var" / "agent_data",
        target=target,
        generation=GENERATION,
        switch=switch,
    )
    write_server_report(
        repo_root / "var" / "agent_data",
        os.getpid(),
        applied_target=target,
        applied_generation=GENERATION,
        targets=published_targets(standin=dict(STANDIN_META, machine_kind=standin_kind)),
    )


def _channel_write_result(tmp_path, hook_runner, make_config):
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": STANDIN_SECTION,
        }
    )
    return hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"channel": "RING:QF:SP", "value": 1.5},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )


def _decision(result):
    if result is None:
        return None
    return result["hookSpecificOutput"]["permissionDecision"]


@pytest.mark.parametrize("switch_state", ["on", "off"], ids=["switched-on", "switched-off"])
def test_the_rehearsal_asks_exactly_when_the_real_machine_would(
    tmp_path, hook_runner, make_config, switch_state
):
    """The ask decision is the same for the rehearsal and the real machine.

    One write, one config, one switch state; only the machine the deployment is
    on differs. The rehearsal moves nothing, but it is where an operator
    practises the real machine's procedure, so it must ask (and defer) exactly
    where the real machine does — a rehearsal that skipped the prompt would
    rehearse a different procedure. The words differ; the decision does not.
    """
    decisions = {}
    reasons = {}
    for target in ("standin", "live"):
        # One repo root, rewritten between the two calls: the hook derives the
        # record's root from the rendered config beside it.
        _write_standin_state(tmp_path, target, {target: switch_state})
        result = _channel_write_result(tmp_path, hook_runner, make_config)
        decisions[target] = _decision(result)
        if decisions[target] == "ask":
            reasons[target] = result["hookSpecificOutput"]["permissionDecisionReason"]

    assert decisions["standin"] == decisions["live"]
    if switch_state == "on":
        assert decisions["live"] == "ask"
        assert f"Target: {REHEARSAL_WORDS} ({STANDIN_ENDPOINT})" in reasons["standin"]
        assert f"Target: {REAL_WORDS} ({LIVE_ENDPOINT})" in reasons["live"]
    else:
        assert decisions["live"] != "ask"


def test_end_to_end_a_standin_pointing_elsewhere_prompts_as_the_real_machine(
    tmp_path, hook_runner, make_config
):
    """A ``standin`` the server publishes as ``real`` never says "nothing moves"."""
    _write_standin_state(tmp_path, "standin", {"standin": "on"}, standin_kind="real")

    reason = ask_reason(_channel_write_result(tmp_path, hook_runner, make_config))

    assert f"Target: {REAL_WORDS} ({STANDIN_ENDPOINT})" in reason
    assert "nothing moves" not in reason


# ---------------------------------------------------------------------------
# which targets a deployment has: the stdlib restatement of the framework's resolver
# ---------------------------------------------------------------------------
# `osprey_connectors.types` is the authority; the reader mirrors its target-to-type
# mapping and its reachable-target rule for hooks, which cannot import osprey.
#
# The pre-switch posture predicates are retired: a write is decided by `write_verdict`, twin of the package's, and
# the shared case table drives it. Each removed case and the row that covers it:
#
# * the per-shape `writes_enabled` answers, the literal-`True` rules, a dotted
#   custom type, a non-mapping section -> `tier-absent-unarmed`,
#   `tier-absent-old-armed-flat`/`-connector`, `tier-garbage-true`,
#   `tier-section-not-object`,
#   `tier-section-null`
# * an unknown target, an underivable `live` -> `target-bogus`, `target-null`,
#   `target-empty`, `target-number`, `target-live-without-a-real-machine`
# * the AND over reachable targets for an unidentified call -> `target-null`
#   (an unidentified target now refuses `target_unknown` outright)

#: Baseline is the machine, beside a simulator block.
DISARM_LIVE_SECTION = {
    "type": "epics",
    "connector": {
        "epics": {"address_list": "10.0.0.1"},
        "virtual_accelerator": {},
    },
}

#: Two non-simulated blocks under a simulated baseline: `live` names no single type.
UNDERIVABLE_LIVE_SECTION = {
    "type": "virtual_accelerator",
    "connector": {
        "virtual_accelerator": {"serving": "in_process"},
        "epics": {"address_list": "10.0.0.1"},
        "doocs": {"address": "x"},
    },
}

#: A simulator deployment with only the connector type it builds.
VA_ONLY_SECTION = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"serving": "in_process"}},
}


def test_a_simulator_only_deployment_reaches_one_target(reader):
    """No switch rendered means one target a session can be pointed at, not two."""
    assert reader.session_types(VA_ONLY_SECTION) == {"va": "virtual_accelerator"}


def test_the_simulator_in_process_carrying_one_live_block_is_two_targets(reader):
    """`live` resolves to the block and the baseline is `va`, so both are reachable."""
    section = {
        "type": "virtual_accelerator",
        "connector": {
            "virtual_accelerator": {"serving": "in_process"},
            "epics": {"address_list": "10.0.0.1"},
        },
    }

    assert reader.session_types(section) == {"live": "epics", "va": "virtual_accelerator"}


def test_both_targets_are_reachable_only_on_a_switch_capable_render(reader):
    """Both types configured with a block."""
    section = {
        "type": "epics",
        "connector": {
            "epics": {"address_list": "10.0.0.1"},
            "virtual_accelerator": {"port": 5074},
        },
    }

    assert reader.session_types(section) == {"live": "epics", "va": "virtual_accelerator"}


def test_a_standin_beside_a_simulator_reaches_both_without_a_live_block(reader):
    """Two configured targets are the switching world, whichever two they are.

    Mirrors ``osprey_connectors.types.switch_capable``: a deployment rehearsing
    on its stand-in beside the simulator has two machines a session can be
    pointed at, and ``live`` — underivable here — is simply not among them.
    """
    section = {
        "type": "live_standin",
        "connector": {
            "virtual_accelerator": {"port": 5075},
            "live_standin": {"port": 5074},
        },
    }

    assert reader.session_types(section) == {
        "standin": "live_standin",
        "va": "virtual_accelerator",
    }


def test_the_target_to_type_mapping_is_public(reader):
    """`osprey_writes_check` spells its refusal keys from this mapping.

    Public on purpose: a refusal has to name the block its own answer was read
    from, and a hook re-deriving the mapping privately is a hook whose message
    can drift away from its decision without either looking wrong.
    """
    assert "target_type" in reader.__all__
    assert reader.target_type(DISARM_LIVE_SECTION, "live") == "epics"
    assert reader.target_type(DISARM_LIVE_SECTION, "va") == "virtual_accelerator"
    assert reader.target_type(UNDERIVABLE_LIVE_SECTION, "live") is None


def test_a_dotted_custom_type_is_one_key_and_not_a_path(reader):
    """A custom connector's module path names a single block, dots and all."""
    section = {"type": "mypackage.MoatConnector"}

    assert reader.target_type(section, "live") == "mypackage.MoatConnector"
