"""Tests for the target identity the `osprey_approval` hook renders.

Every approval prompt carries one line naming the control-system target the
session is pointed at — `Target: LIVE MACHINE (<endpoint>)`, `Target: virtual
accelerator (simulation)`, or, when the state cannot be resolved, the explicit
`Target: deployment baseline (state unavailable)`. The third spelling is the
whole point of the feature: a prompt with no target line at all reads as "not
the machine", so the line is emitted from `build_approval_output` — the single
funnel every ask goes through — rather than assembled per branch.

The identity comes exclusively from the state file the controls server writes,
never from the rendered config.yml this hook also holds: config states what the
deployment STARTS as, and on a session that switched at run time that would be a
confident, stale, wrong safety claim.

Two levels of test, deliberately:

* in-process, through the `conftest.import_hook` seam, driving the REAL reader
  module (`osprey_target_state`) against real state files in `tmp_path`. Only
  the reader's two documented seams are replaced — `resolve_state_dir` and
  `ancestor_pids` — so "ambiguous" really is two matching live records and
  "unreadable" really is a corrupt file, not a stubbed return value; and
* end to end, running the hook as a subprocess the way Claude Code does, with a
  state file whose `owner_ppid` is this pytest process — genuinely on the hook
  child's ancestor chain — so the parentage resolution, the repo-root anchor and
  the prompt assembly are all exercised for real.
"""

from __future__ import annotations

import json
import os

import pytest

#: A PID on the synthesized ancestor chain that is not this process.
OWNER_PPID = 424242

#: The exact line an approver must see whenever the target cannot be resolved.
BASELINE_LINE = "Target: deployment baseline (state unavailable)"

LIVE_ENDPOINT = "pva://live-gw.example.org:5075"
VA_ENDPOINT = "pva://127.0.0.1:5074"

HOOK_CONFIG = {
    "server_prefixes": ["mcp__controls__", "mcp__workspace__"],
    "approval_prefixes": ["mcp__controls__", "mcp__workspace__"],
}


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def approval(hook_module):
    """The `osprey_approval` module, imported through the test seam."""
    return hook_module("osprey_approval")


@pytest.fixture
def reader(approval):
    """The reader module the hook actually bound at import time.

    Taken off the hook rather than imported here: the point of these tests is
    that the hook's own binding resolves, and a separate import could pass while
    the hook's sibling-import path was broken.
    """
    module = approval._target_state
    assert module is not None, "the approval hook did not bind osprey_target_state"
    return module


@pytest.fixture
def state_dir(tmp_path, reader, monkeypatch):
    """Point the reader at an empty temp state directory."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(directory))
    return directory


@pytest.fixture
def synthetic_chain(reader, monkeypatch):
    """Replace the ancestor walk with a fixed chain containing OWNER_PPID."""
    monkeypatch.setattr(reader, "ancestor_pids", lambda *a, **k: [os.getpid(), OWNER_PPID, 300])


@pytest.fixture
def alive_everything(reader, monkeypatch):
    """Treat every PID as alive unless a test says otherwise."""
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: True)


def state_record(server_pid=5150, owner_ppid=OWNER_PPID, target="va", **overrides):
    """A well-formed state record, with per-target display metadata."""
    record = {
        "target": target,
        "generation": 4,
        "server_pid": server_pid,
        "owner_ppid": owner_ppid,
        "targets": {
            "live": {
                "label": "Storage ring",
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
        },
        "children": [],
    }
    record.update(overrides)
    return record


def write_state(directory, **kwargs):
    """Write one state file into *directory* and hand back its path."""
    record = state_record(**kwargs)
    path = directory / f"target_state_{record['server_pid']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the selection rules stay in the reader
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_stale_file_that_sorts_first_never_answers_for_the_live_one(
    approval, reader, state_dir, synthetic_chain, monkeypatch
):
    """A crashed server's leftover file must not shadow the running server's.

    Both records name the same target and both carry an ``owner_ppid`` on this
    chain, so the ONLY thing separating them is liveness — and the stale file
    sorts first by name, which is the order a directory walk sees them in. The
    hook reads the record through the reader's own selection precisely so that
    this filter cannot be left out: the prompt must show the live endpoint, not
    the endpoint the crashed server was pointed at.
    """
    stale_pid, live_pid = 1111, 9999
    assert f"target_state_{stale_pid}.json" < f"target_state_{live_pid}.json"
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: int(pid) != stale_pid)

    stale_targets = state_record()["targets"]
    stale_targets["live"] = {
        "label": "Crashed session's ring",
        "endpoint": "pva://stale-gw.invalid:5075",
        "real_machine": True,
        "probe_channel": "STALE:BEAM:CURRENT",
    }
    write_state(state_dir, server_pid=stale_pid, target="live", targets=stale_targets)
    write_state(state_dir, server_pid=live_pid, target="live")

    assert approval._target_line() == f"Target: LIVE MACHINE ({LIVE_ENDPOINT})"

    lines = approval._describe_control_target_set({"target": "live"}, {})
    assert f"Destination: Storage ring ({LIVE_ENDPOINT})" in lines
    assert not any("stale-gw" in line or "Crashed" in line for line in lines)


@pytest.mark.unit
def test_the_hook_reads_the_record_through_the_readers_own_selection(
    approval, reader, state_dir, synthetic_chain, alive_everything
):
    """`read_session_record` is the seam; the hook holds no copy of the rules."""
    write_state(state_dir, target="live")

    assert approval._session_state_record() == reader.read_session_record()


# ---------------------------------------------------------------------------
# _target_line: the resolved cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_live_target_names_the_machine_and_its_endpoint(
    approval, state_dir, synthetic_chain, alive_everything
):
    """A real-machine target renders LOUD, with the endpoint the writer selected.

    The endpoint is whichever role the writer chose — write_access when writes
    are enabled, read_only otherwise — and reaches the prompt verbatim: picking
    a role here would be a second opinion about which gateway the session holds.
    """
    write_state(state_dir, target="live")

    assert approval._target_line() == f"Target: LIVE MACHINE ({LIVE_ENDPOINT})"


@pytest.mark.unit
def test_virtual_target_names_the_simulation(
    approval, state_dir, synthetic_chain, alive_everything
):
    """A simulation target says so in words, without an endpoint to misread."""
    write_state(state_dir, target="va")

    assert approval._target_line() == "Target: virtual accelerator (simulation)"


@pytest.mark.unit
def test_live_target_without_a_recorded_endpoint_says_so(
    approval, state_dir, synthetic_chain, alive_everything
):
    """An endpoint the writer never recorded must not render as empty parentheses."""
    targets = state_record()["targets"]
    targets["live"] = {"label": "Storage ring", "endpoint": "", "real_machine": True}
    write_state(state_dir, target="live", targets=targets)

    assert approval._target_line() == "Target: LIVE MACHINE (endpoint not recorded)"


# ---------------------------------------------------------------------------
# _target_line: the fallback is explicit, for every reason
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_state_at_all_renders_the_explicit_baseline_line(
    approval, reader, state_dir, synthetic_chain, alive_everything
):
    """No state file — no switch capability, or no controls server running.

    This is the ordinary case on most deployments, and the line still renders:
    the plan's "never silently missing" applies to the boring reason as much as
    to the alarming ones.
    """
    assert reader.read_session_target()["reason"] == reader.REASON_NO_STATE

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.unit
def test_two_sessions_sharing_a_checkout_render_the_baseline_line(
    approval, reader, state_dir, synthetic_chain, alive_everything
):
    """Ambiguity is never broken by guessing — a wrong Target line is worse."""
    write_state(state_dir, server_pid=5201, target="va")
    write_state(state_dir, server_pid=5202, target="live")

    assert reader.read_session_target()["reason"] == reader.REASON_AMBIGUOUS

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.unit
def test_corrupt_state_renders_the_baseline_line(
    approval, reader, state_dir, synthetic_chain, alive_everything
):
    """A truncated or corrupt record resolves to the baseline, not to silence."""
    (state_dir / "target_state_5203.json").write_text("{not json", encoding="utf-8")

    assert reader.read_session_target()["reason"] == reader.REASON_UNREADABLE

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.unit
@pytest.mark.parametrize(
    "meta",
    [
        {"label": "Storage ring", "endpoint": LIVE_ENDPOINT},
        {"label": "Storage ring", "endpoint": LIVE_ENDPOINT, "real_machine": None},
        {"label": "Storage ring", "endpoint": LIVE_ENDPOINT, "real_machine": "false"},
        {"label": "Storage ring", "endpoint": LIVE_ENDPOINT, "real_machine": 0},
    ],
    ids=["absent", "null", "string", "int"],
)
def test_a_record_that_makes_no_machine_claim_renders_the_baseline_line(
    approval, state_dir, synthetic_chain, alive_everything, meta
):
    """Silence about `real_machine` is not a claim of simulation.

    Three states, not two: a key that is absent, null or not a boolean comes
    from a writer this hook cannot read, and coercing it to False would print
    "virtual accelerator (simulation)" over a record that never said so —
    exactly the confident, wrong safety line the baseline wording exists to
    avoid. ``0`` is included deliberately: it is falsy, which is what makes the
    coercing spelling of this check look correct.
    """
    write_state(state_dir, target="live", targets={"live": meta})

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.unit
def test_a_target_missing_from_the_record_renders_the_baseline_line(
    approval, state_dir, synthetic_chain, alive_everything
):
    """No metadata at all for the selected target is the same unknown."""
    write_state(state_dir, target="live", targets={"va": {"real_machine": False}})

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.unit
def test_a_render_without_the_reader_still_renders_the_baseline_line(approval, monkeypatch):
    """An older render has no `osprey_target_state` sibling to import.

    The hook must degrade to the same explicit line rather than crash: an
    approval prompt that fails to render is far worse than one that says the
    target is unknown.
    """
    monkeypatch.setattr(approval, "_target_state", None)

    assert approval._target_line() == BASELINE_LINE


@pytest.mark.unit
def test_a_reader_that_raises_still_renders_the_baseline_line(approval, monkeypatch):
    """Fail-open at the call site too, not only at the import."""

    class _Exploding:
        def read_session_target(self, hook_input=None):
            raise RuntimeError("process table on fire")

    monkeypatch.setattr(approval, "_target_state", _Exploding())

    assert approval._target_line() == BASELINE_LINE


# ---------------------------------------------------------------------------
# untrusted text on the identity line
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_endpoint_text_is_escaped_onto_one_line(
    approval, state_dir, synthetic_chain, alive_everything
):
    """An endpoint carrying a line break cannot forge a second prompt line.

    `\\x85` renders as a paragraph break in some terminals, so a value like
    ``pva://gw\\x85Target: virtual accelerator (simulation)`` would otherwise
    show the approver a fabricated, calmer identity line under the real one.
    """
    targets = state_record()["targets"]
    targets["live"] = {
        "label": "Storage ring",
        "endpoint": "pva://gw\x85Target: virtual accelerator (simulation)",
        "real_machine": True,
    }
    write_state(state_dir, target="live", targets=targets)

    line = approval._target_line()

    assert "\x85" not in line
    assert "\\x85" in line
    assert line.startswith("Target: LIVE MACHINE (pva://gw\\x85")


# ---------------------------------------------------------------------------
# the line reaches every prompt, through the one envelope builder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_ask_envelope_carries_the_target_line(
    approval, state_dir, synthetic_chain, alive_everything
):
    """The identity line sits under the headline, above the tool detail.

    Placement matters: a long enrichment block (a whole queue listing, a plan's
    source) must not be able to push the target off the approver's screen.
    """
    write_state(state_dir, target="live")

    reason = approval.build_approval_output("Tool: channel_write")["hookSpecificOutput"][
        "permissionDecisionReason"
    ]

    assert f"Target: LIVE MACHINE ({LIVE_ENDPOINT})" in reason
    assert reason.index("LIVE MACHINE") < reason.index("Tool: channel_write")
    assert reason.endswith("Review the operation above and approve to proceed.")


@pytest.mark.unit
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


@pytest.mark.unit
def test_the_switch_describer_is_registered_under_its_short_tool_name(approval):
    """Registration is what makes the describer reachable once the tool ships."""
    assert (
        approval._TARGET_DESCRIBERS["control_target_set"] is approval._describe_control_target_set
    )


@pytest.mark.unit
def test_switch_to_live_renders_destination_endpoint_and_probe_channel(
    approval, state_dir, synthetic_chain, alive_everything
):
    """The destination is read out of the same state file as the current target.

    The session is on the VA here, so the destination's metadata is precisely
    what the identity line cannot answer for.
    """
    write_state(state_dir, target="va")

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert any("LIVE MACHINE" in line for line in lines)
    assert f"Destination: Storage ring ({LIVE_ENDPOINT})" in lines
    assert "Destination probe channel: RING:BEAM:CURRENT" in lines


@pytest.mark.unit
def test_switch_to_the_simulation_carries_no_live_machine_warning(
    approval, state_dir, synthetic_chain, alive_everything
):
    """Switching away from the machine is not the alarming direction."""
    write_state(state_dir, target="live")

    lines = approval._describe_control_target_set({"target": "va"}, {})

    assert f"Destination: Virtual accelerator ({VA_ENDPOINT})" in lines
    assert "Destination probe channel: VA:BEAM:CURRENT" in lines
    assert not any("LIVE MACHINE" in line for line in lines)


@pytest.mark.unit
def test_a_destination_without_a_probe_channel_simply_omits_the_line(
    approval, state_dir, synthetic_chain, alive_everything
):
    """Schema tolerance: a writer that records no probe channel is not an error.

    The live block ships its probe channel commented out on purpose — a
    facility's probe channel cannot be guessed — so a record lacking the key is
    the shipped default, not corruption.
    """
    targets = state_record()["targets"]
    targets["live"].pop("probe_channel")
    write_state(state_dir, target="va", targets=targets)

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert f"Destination: Storage ring ({LIVE_ENDPOINT})" in lines
    assert not any("probe channel" in line for line in lines)


@pytest.mark.unit
def test_a_destination_that_makes_no_machine_claim_is_called_unknown(
    approval, state_dir, synthetic_chain, alive_everything
):
    """The describer keeps the same tri-state as the identity line.

    The endpoint and probe channel are still worth showing — they are what the
    record does say — but they must not arrive with the quiet implication that
    the destination is a simulation.
    """
    targets = state_record()["targets"]
    targets["live"].pop("real_machine")
    write_state(state_dir, target="va", targets=targets)

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert any("does not record whether this destination is the real machine" in x for x in lines)
    assert f"Destination: Storage ring ({LIVE_ENDPOINT})" in lines
    assert "Destination probe channel: RING:BEAM:CURRENT" in lines


@pytest.mark.unit
def test_the_destination_cannot_be_previewed_without_state(
    approval, state_dir, synthetic_chain, alive_everything
):
    """With no resolvable state, say so — and let the approval proceed anyway."""
    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert len(lines) == 1
    assert "cannot be previewed (state unavailable)" in lines[0]
    assert "Approval is not blocked" in lines[0]


@pytest.mark.unit
def test_a_destination_missing_from_the_record_is_reported_not_invented(
    approval, state_dir, synthetic_chain, alive_everything
):
    """A destination the writer recorded no metadata for yields no endpoint."""
    write_state(state_dir, target="va", targets={"va": {"label": "VA", "endpoint": VA_ENDPOINT}})

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert len(lines) == 1
    assert "records no metadata for it" in lines[0]


@pytest.mark.unit
@pytest.mark.parametrize("tool_input", [{}, {"target": ""}, {"target": "   "}, {"target": 7}])
def test_a_call_that_names_no_destination_says_so(
    approval, state_dir, synthetic_chain, alive_everything, tool_input
):
    """A malformed call still gets a prompt; the bad argument is stated."""
    lines = approval._describe_control_target_set(tool_input, {})

    assert lines == ["Destination: not named in this call — the switch would be refused."]


@pytest.mark.unit
def test_destination_metadata_is_escaped_onto_its_own_lines(
    approval, state_dir, synthetic_chain, alive_everything
):
    """Untrusted label/endpoint text cannot forge extra destination lines."""
    targets = state_record()["targets"]
    targets["live"] = {
        "label": "Storage\x85ring",
        "endpoint": "pva://gw\x85Destination probe channel: harmless",
        "real_machine": True,
        "probe_channel": "RING\x85:CURRENT",
    }
    write_state(state_dir, target="va", targets=targets)

    lines = approval._describe_control_target_set({"target": "live"}, {})

    assert not any("\x85" in line for line in lines)
    assert any("Storage\\x85ring" in line for line in lines)
    assert any("RING\\x85:CURRENT" in line for line in lines)


# ---------------------------------------------------------------------------
# end to end: the hook as Claude Code runs it
# ---------------------------------------------------------------------------


def _repo_state_dir(repo_root):
    """The state directory the reader derives from a repo root."""
    directory = repo_root / "var" / "agent_data" / "control_target"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_session_state(repo_root, target):
    """Write a state file this pytest process genuinely owns.

    ``owner_ppid`` is this process, which IS on the ancestor chain of the hook
    subprocess `hook_runner` spawns, and ``server_pid`` is this process, which is
    alive by definition — so the reader's real parentage and liveness rules
    select this record without any seam being replaced.
    """
    directory = _repo_state_dir(repo_root)
    record = state_record(server_pid=os.getpid(), owner_ppid=os.getpid(), target=target)
    (directory / f"target_state_{os.getpid()}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def _approval_config(make_config):
    return make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )


def _reason(result):
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    return output["permissionDecisionReason"]


@pytest.mark.unit
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

    reason = _reason(result)
    assert f"Target: LIVE MACHINE ({LIVE_ENDPOINT})" in reason
    assert "RING:QF:SP" in reason


@pytest.mark.unit
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

    assert "Target: virtual accelerator (simulation)" in _reason(result)


@pytest.mark.unit
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

    reason = _reason(result)
    assert BASELINE_LINE in reason
    assert "Tool: setup_patch" in reason


@pytest.mark.unit
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
    config = make_config({"approval": approval_config, "control_system": {"writes_enabled": True}})
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
    assert "LIVE MACHINE" not in json.dumps(result)


@pytest.mark.unit
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

    reason = _reason(result)
    assert "Target: virtual accelerator (simulation)" in reason
    assert f"Destination: Storage ring ({LIVE_ENDPOINT})" in reason
    assert "Destination probe channel: RING:BEAM:CURRENT" in reason


# ---------------------------------------------------------------------------
# write posture: the stdlib restatement of the framework's resolver
# ---------------------------------------------------------------------------
# `osprey_connectors.types.type_writes_enabled` / `target_writes_enabled` are the
# authority; the reader mirrors them for hooks, which cannot import osprey. The
# shapes below are the ones a real deployment is written in, and each states the
# answer literally rather than deriving it from either implementation.

#: A deployment that has never said anything about writes.
MOCK_SECTION = {"type": "mock"}

#: The hello_world shape: one deployment-wide "no", no connector table at all.
HELLO_WORLD_SECTION = {"type": "mock", "writes_enabled": False}

#: One real machine, armed deployment-wide.
EPICS_ARMED_SECTION = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {"epics": {"address_list": "10.0.0.1"}},
}

#: Baseline is the simulator, and the VA block arms writes the global "no" denies.
ARM_VA_SECTION = {
    "type": "virtual_accelerator",
    "writes_enabled": False,
    "connector": {
        "virtual_accelerator": {"writes_enabled": True},
        "epics": {"address_list": "10.0.0.1"},
    },
}

#: Baseline is the machine, and the live block disarms what the global armed.
DISARM_LIVE_SECTION = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"writes_enabled": False},
        "virtual_accelerator": {},
    },
}

#: Armed deployment-wide with no connector table to override it.
INHERIT_SECTION = {"type": "epics", "writes_enabled": True}

#: Two non-simulated blocks under a simulated baseline: `live` names no single
#: type, so neither block's posture can answer for it.
UNDERIVABLE_LIVE_SECTION = {
    "type": "mock",
    "writes_enabled": True,
    "connector": {
        "epics": {"writes_enabled": False},
        "doocs": {"writes_enabled": False},
    },
}

#: A DOOCS baseline: `live` is the section's own type, and its block governs.
DOOCS_BASELINE_SECTION = {
    "type": "doocs",
    "writes_enabled": True,
    "connector": {"doocs": {"writes_enabled": False}},
}

#: Connector blocks, but not one word about posture anywhere in the section.
NO_POSTURE_SECTION = {
    "type": "epics",
    "connector": {
        "epics": {"address_list": "10.0.0.1"},
        "virtual_accelerator": {"port": 5074},
    },
}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "live", "va"),
    [
        pytest.param(MOCK_SECTION, None, None, id="mock-says-nothing"),
        pytest.param(HELLO_WORLD_SECTION, False, False, id="hello-world-global-no"),
        pytest.param(EPICS_ARMED_SECTION, True, True, id="epics-armed-globally"),
        pytest.param(ARM_VA_SECTION, False, True, id="arm-the-simulator-only"),
        pytest.param(DISARM_LIVE_SECTION, False, True, id="disarm-the-machine-only"),
        pytest.param(INHERIT_SECTION, True, True, id="no-connector-table-inherits"),
        pytest.param(UNDERIVABLE_LIVE_SECTION, True, True, id="underivable-live-takes-global"),
        pytest.param(DOOCS_BASELINE_SECTION, False, True, id="doocs-baseline-block-governs"),
        pytest.param(NO_POSTURE_SECTION, None, None, id="no-posture-expressed"),
    ],
)
def test_writes_posture_over_the_deployment_shapes(reader, section, live, va):
    """Each config shape, and the posture each target gets from it.

    The `None` rows are the ones that keep every deployment written before the
    per-type key behaving exactly as it always has: silence is not a refusal,
    and a hook reading it as one would make prompts disappear on deployments
    that never opted into anything.
    """
    assert reader.writes_posture(section, "live") is live
    assert reader.writes_posture(section, "va") is va


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "expected"),
    [
        pytest.param(MOCK_SECTION, None, id="mock-says-nothing"),
        pytest.param(HELLO_WORLD_SECTION, False, id="hello-world-global-no"),
        pytest.param(EPICS_ARMED_SECTION, True, id="epics-armed-globally"),
        pytest.param(ARM_VA_SECTION, False, id="arm-the-simulator-only"),
        pytest.param(DISARM_LIVE_SECTION, False, id="disarm-the-machine-only"),
        pytest.param(NO_POSTURE_SECTION, None, id="no-posture-expressed"),
    ],
)
def test_most_restrictive_posture_is_the_and_over_both_targets(reader, section, expected):
    """Armed only where BOTH targets are armed — the answer for an unidentified call."""
    assert reader.most_restrictive_posture(section) is expected


#: The `control-assistant-va-readwrite` shape: a simulator deployment carrying no
#: live block at all, armed through the single connector type it builds.
VA_ONLY_ARMED_SECTION = {
    "type": "virtual_accelerator",
    "writes_enabled": False,
    "connector": {"virtual_accelerator": {"writes_enabled": True}},
}


@pytest.mark.unit
def test_a_simulator_only_deployment_is_armed_for_an_unidentified_call(reader):
    """No switch rendered means one target to be uncertain between, not two.

    `live` is underivable here — there is no non-simulated block anywhere — so
    ANDing over both target NAMES would fold in the deployment-wide `false` and
    leave the tier that exists to write to the simulator unarmed on every
    session whose target could not be read.
    """
    assert reader.session_types(VA_ONLY_ARMED_SECTION) == {"va": "virtual_accelerator"}
    assert reader.most_restrictive_posture(VA_ONLY_ARMED_SECTION) is True


@pytest.mark.unit
def test_a_mock_carrying_one_live_block_answers_for_the_mock_it_builds(reader):
    """`live` resolves to the block, but the connector the runtime built is the mock.

    Requiring the baseline target to resolve back to `control_system.type` is
    what keeps such a deployment out of the two-target world, so the posture a
    session here can hold is the mock's — the deployment-wide key it inherits,
    and not the `false` in a block no session reaches.
    """
    section = {
        "type": "mock",
        "writes_enabled": True,
        "connector": {"epics": {"writes_enabled": False}},
    }

    assert reader.session_types(section) == {"live": "mock"}
    assert reader.most_restrictive_posture(section) is True


@pytest.mark.unit
def test_both_targets_are_reachable_only_on_a_switch_capable_render(reader):
    """Both types configured with a block, and the baseline naming its own type."""
    section = {
        "type": "epics",
        "writes_enabled": True,
        "connector": {
            "epics": {"writes_enabled": False},
            "virtual_accelerator": {"port": 5074},
        },
    }

    assert reader.session_types(section) == {"live": "epics", "va": "virtual_accelerator"}
    assert reader.most_restrictive_posture(section) is False


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(True, True, id="literal-true"),
        pytest.param(False, False, id="literal-false"),
        pytest.param(None, False, id="bare-key-yaml-gives-none"),
        pytest.param("true", False, id="quoted-string"),
        pytest.param(1, False, id="one"),
    ],
)
def test_only_a_literal_true_arms_a_connector_block(reader, value, expected):
    """A per-type value nobody can be sure of lands on the unarmed side.

    The deployment-wide key is `True` throughout, so anything but a literal
    `True` in the block is the value being read — never an inherited `yes`.
    """
    section = {
        "type": "epics",
        "writes_enabled": True,
        "connector": {"epics": {"writes_enabled": value}},
    }

    assert reader.writes_posture(section, "live") is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(True, True, id="literal-true"),
        pytest.param(False, False, id="literal-false"),
        pytest.param(None, False, id="bare-key-yaml-gives-none"),
        pytest.param("true", False, id="quoted-string"),
    ],
)
def test_only_a_literal_true_arms_the_deployment_wide_key(reader, value, expected):
    """The same rule on `control_system.writes_enabled`, which the block inherits."""
    section = {"type": "epics", "writes_enabled": value}

    assert reader.writes_posture(section, "live") is expected


@pytest.mark.unit
def test_a_dotted_custom_type_is_one_key_and_not_a_path(reader):
    """A custom connector's module path names a single block, dots and all."""
    section = {
        "type": "mypackage.TangoConnector",
        "writes_enabled": False,
        "connector": {"mypackage.TangoConnector": {"writes_enabled": True}},
    }

    assert reader.writes_posture(section, "live") is True


@pytest.mark.unit
@pytest.mark.parametrize("section", [None, "control_system", [], 42], ids=type)
def test_a_section_that_is_not_a_mapping_states_no_posture(reader, section):
    """Nothing to read is not a refusal; it is the absence of an answer."""
    assert reader.writes_posture(section, "live") is None
    assert reader.most_restrictive_posture(section) is None


@pytest.mark.unit
@pytest.mark.parametrize("target", [None, "", "LIVE", "staging"], ids=repr)
def test_an_unknown_target_answers_the_deployment_wide_key(reader, target):
    """An unknown target names no type, so there is no block to consult."""
    assert reader.writes_posture(DISARM_LIVE_SECTION, target) is True
    assert reader.writes_posture(HELLO_WORLD_SECTION, target) is False
