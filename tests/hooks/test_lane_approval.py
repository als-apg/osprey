"""Tests for the plan-lane awareness of the `osprey_approval` hook.

A deployment that opted into a second Bluesky plan lane runs one whole plan
stack per control-system target, and then two questions the old prompt never had
to answer become safety questions: WHICH lane would this plan be queued on, and
does the lane a start names still serve the target this session is on?

What these tests hold the hook to:

* **a single-lane deployment renders exactly what it always did** — no lane
  line, the same first line, the same bridge. Nearly every deployment is
  single-lane, and a prompt that grew a line there would be a change to the
  safety surface of every project that never asked for lanes. Pinned literally,
  and pinned against a config that has no ``services`` block at all;
* **a two-lane deployment names the lane and its target in the identity voice
  the ``Target:`` line uses** — "virtual accelerator (simulation)" and "LIVE
  MACHINE (<endpoint>)" — because two different phrasings for one machine on one
  prompt is the ambiguity the identity line exists to remove;
* **the detail comes from the addressed lane's own bridge.** Each lane holds its
  own draft and its own queue, so lane 1's answers would describe a different
  machine than the one being approved. Where no lane can be named, nothing is
  listed and the prompt says why;
* **a mismatch is rendered, never approved silently.** A session that switched
  targets between the enqueue and the start would otherwise be shown a queue
  listing with no hint that the start it is approving drives the machine the
  session has just left — and that the deployment will refuse it outright;
* **every gap is explicit.** No state file, no lane serving the session's
  target, a lane whose target the state file records nothing about, a second
  lane with no published port: each has its own line, and none of them is a
  guess or a missing line.

The lane map is read from the rendered config's ``services.<lane>`` blocks —
render-time truth, the same key the host reads — while the session's target
comes only from the state file, driven here through the REAL reader module
against real state files in ``tmp_path``.
"""

from __future__ import annotations

import json
import os

import pytest

#: A PID on the synthesized ancestor chain that is not this process.
OWNER_PPID = 515151

LIVE_ENDPOINT = "pva://live-gw.example.org:5075"
VA_ENDPOINT = "pva://127.0.0.1:5074"

#: How the two targets are spoken of, everywhere on the prompt.
LIVE_PHRASE = f"LIVE MACHINE ({LIVE_ENDPOINT})"
VA_PHRASE = "virtual accelerator (simulation)"

#: Lane 1's bridge, as the rendered config names it, and lane 2's, as the port
#: the build published for it implies.
LANE_ONE_URL = "http://bridge-one.test:8090"
LANE_TWO_PORT = 8190
LANE_TWO_URL = f"http://127.0.0.1:{LANE_TWO_PORT}"

#: The start prompt's opening line, which a single-lane deployment must keep
#: byte for byte.
START_HEADLINE = (
    "Starting the queue runs EVERY pending item below, in order — "
    "not only the most recently added one."
)

HOOK_CONFIG = {
    "server_prefixes": ["mcp__bluesky__"],
    "approval_prefixes": ["mcp__bluesky__"],
}

QUEUE_ROUTE = {
    "status": {"manager_state": "idle", "items_in_queue": 0},
    "items": [],
    "running_item": None,
}

DRAFT_ROUTE = {"draft": {"plan_name": "orbit_scan", "plan_args": {}}, "revision": 3}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def approval(hook_module, monkeypatch):
    """The `osprey_approval` module, imported through the test seam."""
    module = hook_module("osprey_approval")
    # Lane 1's URL must come from the config under test, not from a variable a
    # developer happens to export.
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
    monkeypatch.delenv("BLUESKY_VA_BRIDGE_URL", raising=False)
    monkeypatch.delenv("BLUESKY_LIVE_BRIDGE_URL", raising=False)
    return module


@pytest.fixture
def reader(approval):
    """The reader module the hook actually bound at import time."""
    module = approval._target_state
    assert module is not None, "the approval hook did not bind osprey_target_state"
    return module


@pytest.fixture
def state_dir(tmp_path, reader, monkeypatch):
    """Point the real reader at an empty temp state directory."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(directory))
    monkeypatch.setattr(reader, "ancestor_pids", lambda *a, **k: [os.getpid(), OWNER_PPID, 300])
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: True)
    return directory


def state_record(target="va", targets=None):
    """A well-formed state record, with per-target display metadata."""
    return {
        "target": target,
        "generation": 2,
        "server_pid": 6060,
        "owner_ppid": OWNER_PPID,
        "targets": (
            {
                "live": {
                    "label": "Storage ring",
                    "endpoint": LIVE_ENDPOINT,
                    "real_machine": True,
                },
                "va": {
                    "label": "Virtual accelerator",
                    "endpoint": VA_ENDPOINT,
                    "real_machine": False,
                },
            }
            if targets is None
            else targets
        ),
    }


def write_state(directory, target="va", targets=None):
    """Write the session's state file into *directory*."""
    record = state_record(target=target, targets=targets)
    (directory / f"target_state_{record['server_pid']}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


@pytest.fixture
def bridge_calls():
    """Recorder for the (base_url, path) pairs the describers ask for."""
    return []


@pytest.fixture
def fake_bridge(approval, bridge_calls, monkeypatch):
    """Replace both bridge verbs with a routing table that records its callers.

    Recording the base URL is the point: on a two-lane deployment the prompt is
    only honest if the queue it lists came from the lane it named, and the URL
    is the only evidence of that.
    """

    def _install(routes: dict[str, object]):
        def _get(base_url, path, timeout=3.0):
            bridge_calls.append((base_url, path))
            return routes.get(path)

        def _post(base_url, path, body, timeout):
            bridge_calls.append((base_url, path))
            return routes.get(path)

        monkeypatch.setattr(approval, "_bridge_get_json", _get)
        monkeypatch.setattr(approval, "_bridge_post_json", _post)

    return _install


def single_lane_config(*, with_services=True):
    """A config for the deployment shape nearly every project has."""
    config = {"bluesky": {"bridge_url": LANE_ONE_URL}}
    if with_services:
        config["services"] = {"bluesky": {"port": 8090}}
    return config


def two_lane_config(*, lane_one_target="live", second="bluesky_va", second_target="va", port=None):
    """A config for a deployment that opted into a second plan lane."""
    second_block: dict = {"target": second_target}
    if port is not False:
        second_block["port"] = port or LANE_TWO_PORT
    return {
        "bluesky": {"bridge_url": LANE_ONE_URL},
        "services": {
            "bluesky": {"port": 8090, "target": lane_one_target},
            second: second_block,
        },
    }


def text(lines) -> str:
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# single lane: nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_lane_queue_start_renders_exactly_what_it_always_did(
    approval, fake_bridge, bridge_calls, state_dir
):
    """The pin. A deployment with one lane has nothing to address, so its start
    prompt keeps its opening line byte for byte, says nothing about lanes, and
    asks the one bridge it has always asked."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_start({}, single_lane_config(), None)

    assert lines[0] == START_HEADLINE
    assert "lane" not in text(lines).lower()
    assert {url for url, _ in bridge_calls} == {LANE_ONE_URL}


@pytest.mark.unit
def test_single_lane_start_is_identical_with_and_without_a_services_block(
    approval, fake_bridge, state_dir
):
    """A config this hook cannot read a ``services`` block out of must not be
    able to make the default lane disappear — or appear as a second one."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    with_block = approval._describe_queue_start({}, single_lane_config(), None)
    without_block = approval._describe_queue_start(
        {}, single_lane_config(with_services=False), None
    )

    assert with_block == without_block


@pytest.mark.unit
def test_single_lane_queue_add_renders_no_lane_line_even_for_a_switched_session(
    approval, fake_bridge, bridge_calls, state_dir
):
    """A single-lane deployment refuses a switched session at the tool, not with
    a lane line: there is no other lane to name, and inventing one would change
    a prompt no second lane exists to justify."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_add({"draft_revision": 3}, single_lane_config(), None)

    assert "lane" not in text(lines).lower()
    assert "Plan: orbit_scan" in lines
    assert {url for url, _ in bridge_calls} == {LANE_ONE_URL}


@pytest.mark.unit
def test_a_lane_argument_on_a_single_lane_deployment_changes_nothing(
    approval, fake_bridge, state_dir
):
    """An agent that names a lane on a deployment with one lane is describing
    the only lane there is; the prompt has nothing extra to say about it."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    named = approval._describe_queue_start({"lane": "bluesky"}, single_lane_config(), None)
    unnamed = approval._describe_queue_start({}, single_lane_config(), None)

    assert named == unnamed


@pytest.mark.unit
def test_a_single_lane_describer_never_reads_the_state_file(approval, fake_bridge, monkeypatch):
    """Not merely "renders no lane line" — does not look. A deployment with one
    lane has no lane to resolve, and the prompt's one read of the state file
    belongs to the `Target:` line, which happens later and only once."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE, "/plans": []})

    def refuse(hook_input=None):
        raise AssertionError("a single-lane describer read the target state")

    monkeypatch.setattr(approval, "_read_record_once", refuse)

    approval._describe_queue_start({"lane": "bluesky"}, single_lane_config(), None)
    approval._describe_queue_add({"draft_revision": 3}, single_lane_config(), None)


@pytest.mark.unit
def test_one_prompt_describes_one_read_of_the_state_file(
    approval, fake_bridge, state_dir, monkeypatch
):
    """The lane lines and the `Target:` line come from a single read.

    Two reads are two claims: a switch landing between them would produce a
    prompt whose lane block and identity line describe different machines, and
    an approver has no way to see that they disagree."""
    reads = []
    real = approval._session_state_record

    def counting(hook_input=None):
        reads.append(hook_input)
        return real(hook_input)

    monkeypatch.setattr(approval, "_session_state_record", counting)
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    read_record = approval._record_reader(None)
    lines = approval._describe_queue_start(
        {"lane": "bluesky_va"}, two_lane_config(), None, read_record
    )
    output = approval.build_approval_output(text(lines), None, read_record)

    assert len(reads) == 1
    reason = output["hookSpecificOutput"]["permissionDecisionReason"]
    assert f"Target: {VA_PHRASE}" in reason
    assert f"Bluesky PLAN lane: bluesky_va (target: {VA_PHRASE})" in reason


# ---------------------------------------------------------------------------
# two lanes: queue_add binds to the active lane
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_two_lane_queue_add_names_the_simulation_lane_and_asks_its_bridge(
    approval, fake_bridge, bridge_calls, state_dir
):
    """The session is on the VA, so the plan binds to the VA lane — and the
    draft and queue shown below the line come from THAT lane's bridge."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_add({"draft_revision": 3}, two_lane_config(), None)

    assert lines[0] == (
        f"Bluesky PLAN lane: bluesky_va (target: {VA_PHRASE}) — the lane serving this "
        f"session's target, which is where this plan binds."
    )
    assert {url for url, _ in bridge_calls} == {LANE_TWO_URL}
    assert "Plan: orbit_scan" in lines


@pytest.mark.unit
def test_two_lane_queue_add_names_the_live_lane_with_its_endpoint(
    approval, fake_bridge, bridge_calls, state_dir
):
    """Pointed at the machine, the same line names the machine and the gateway
    it holds — the identity voice the `Target:` line uses, not a bare 'live'."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE, "/plans": []})
    write_state(state_dir, target="live")

    lines = approval._describe_queue_add({"draft_revision": 3}, two_lane_config(), None)

    assert lines[0].startswith(f"Bluesky PLAN lane: bluesky (target: {LIVE_PHRASE})")
    assert {url for url, _ in bridge_calls} == {LANE_ONE_URL}


@pytest.mark.unit
def test_two_lane_queue_add_without_state_says_the_lane_is_unresolved(
    approval, fake_bridge, bridge_calls, state_dir
):
    """With no readable state there is no active lane, so there is no honest
    queue to show: the prompt says which lanes exist and that it cannot tell
    which one the plan would land in."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE})

    lines = approval._describe_queue_add({"draft_revision": 3}, two_lane_config(), None)
    rendered = text(lines)

    assert "Bluesky PLAN lane: unresolved" in rendered
    assert "the session's target state could not be read" in rendered
    assert "'bluesky' (live), 'bluesky_va' (va)" in rendered
    assert "NOT previewed" in rendered
    assert bridge_calls == []


@pytest.mark.unit
def test_two_lane_queue_add_says_so_when_no_lane_serves_the_session_target(
    approval, fake_bridge, bridge_calls, state_dir
):
    """A misrendered lane pair — two lanes, neither serving where the session
    is pointed — is a refusal, and the prompt states it as one."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE})
    write_state(state_dir, target="live")

    lines = approval._describe_queue_add(
        {"draft_revision": 3},
        two_lane_config(lane_one_target="va", second_target="va"),
        None,
    )
    rendered = text(lines)

    assert "NO ACTIVE BLUESKY PLAN LANE" in rendered
    assert "this session is on target live" in rendered
    assert "REFUSED" in rendered
    assert bridge_calls == []


@pytest.mark.unit
def test_a_second_lane_with_no_published_port_shows_no_queue_rather_than_lane_ones(
    approval, fake_bridge, bridge_calls, state_dir
):
    """A lane whose bridge cannot be addressed loses its listing, never borrows
    the other lane's: a queue from the wrong lane describes another machine."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_add({"draft_revision": 3}, two_lane_config(port=False), None)
    rendered = text(lines)

    assert "Bluesky PLAN lane: bluesky_va" in rendered
    assert "publishes no port for the 'bluesky_va' lane" in rendered
    assert bridge_calls == []


@pytest.mark.unit
def test_a_lane_target_the_state_file_says_nothing_about_is_called_unrecorded(
    approval, fake_bridge, state_dir
):
    """Silence in the state file must not read as "simulation". A lane whose
    target carries no `real_machine` claim is named with its target string and
    an explicit statement that the identity is not recorded."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE})
    write_state(state_dir, target="va", targets={"va": {"label": "Virtual accelerator"}})

    lines = approval._describe_queue_add({"draft_revision": 3}, two_lane_config(), None)

    assert lines[0].startswith(
        "Bluesky PLAN lane: bluesky_va (target: va (identity not recorded in the state file))"
    )
    assert VA_PHRASE not in lines[0]


# ---------------------------------------------------------------------------
# two lanes: queue_start is an address
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_start_renders_the_bound_lane_and_lists_its_queue(
    approval, fake_bridge, bridge_calls, state_dir
):
    """The lane the start names, the target it serves, and the queue that would
    actually drain — all three from the lane that was named."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_start({"lane": "bluesky_va"}, two_lane_config(), None)

    assert lines[0] == (
        f"Bluesky PLAN lane: bluesky_va (target: {VA_PHRASE}) — the lane this session's "
        f"target is on."
    )
    assert lines[1] == START_HEADLINE
    assert "MISMATCH" not in text(lines)
    assert {url for url, _ in bridge_calls} == {LANE_TWO_URL}


@pytest.mark.unit
def test_queue_start_on_a_lane_the_session_left_renders_the_mismatch(
    approval, fake_bridge, bridge_calls, state_dir
):
    """The case the whole axis exists for: the item was queued on the live lane,
    the session has since switched to the simulation, and starting now would
    drive the machine the session left. The deployment refuses it — so the
    prompt says which lane serves what, which target the session is on, and that
    approving achieves nothing."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")
    config = two_lane_config(lane_one_target="va", second="bluesky_live", second_target="live")

    lines = approval._describe_queue_start({"lane": "bluesky_live"}, config, None)
    rendered = text(lines)

    assert lines[0] == (
        f"⚠️  LANE MISMATCH — the 'bluesky_live' lane serves {LIVE_PHRASE}; this session's "
        f"target is {VA_PHRASE}. Starting will be REFUSED."
    )
    assert "The lane serving this session's target is 'bluesky'" in rendered
    # The listing still belongs to the lane that was named, not to the active
    # one — and the line directly above it says so.
    assert (
        f"Bluesky PLAN lane: bluesky_live (target: {LIVE_PHRASE}) — the queue listed "
        f"below is this lane's."
    ) in rendered
    assert {url for url, _ in bridge_calls} == {LANE_TWO_URL}


@pytest.mark.unit
def test_queue_start_naming_no_lane_on_a_two_lane_deployment_shows_no_queue(
    approval, fake_bridge, bridge_calls, state_dir
):
    """Two lanes and no address is an ambiguous instruction about hardware
    motion. Nothing is listed — a listing would have to pick a lane, which is
    the guess this refuses to make — and the active lane is named so the
    operator can see what the missing argument should have been."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_start({}, two_lane_config(), None)
    rendered = text(lines)

    assert "NO LANE NAMED" in rendered
    assert "REFUSED" in rendered
    assert f"The lane serving this session's target is 'bluesky_va' (target: {VA_PHRASE})" in (
        rendered
    )
    assert "Queue contents: not shown" in rendered
    assert bridge_calls == []


@pytest.mark.unit
def test_queue_start_naming_a_lane_this_deployment_does_not_render(
    approval, fake_bridge, bridge_calls, state_dir
):
    """An unrendered lane is not served from lane 1: answering about a different
    machine than the one asked about is the confusion lanes exist to remove."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_start({"lane": "bluesky_live"}, two_lane_config(), None)
    rendered = text(lines)

    assert "UNKNOWN LANE" in rendered
    assert "'bluesky' (live), 'bluesky_va' (va)" in rendered
    assert "REFUSED" in rendered
    assert bridge_calls == []


@pytest.mark.unit
def test_queue_start_without_state_names_the_lane_but_claims_no_mismatch(
    approval, fake_bridge, bridge_calls, state_dir
):
    """Unknown is not the same claim as mismatched. With no readable state the
    lane is still real and its queue is still its own, so both are shown — what
    the prompt withholds is the comparison it cannot make."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})

    lines = approval._describe_queue_start({"lane": "bluesky_va"}, two_lane_config(), None)
    rendered = text(lines)

    assert "Bluesky PLAN lane: bluesky_va" in rendered
    assert "cannot say whether that is the lane this session is on" in rendered
    assert "MISMATCH" not in rendered
    assert {url for url, _ in bridge_calls} == {LANE_TWO_URL}


@pytest.mark.unit
def test_the_per_lane_bridge_url_override_wins_over_the_published_port(
    approval, fake_bridge, bridge_calls, state_dir, monkeypatch
):
    """The framework sets one URL per bridge instance; a lane's listing has to
    follow the bridge that is actually running, not the port the build wrote."""
    monkeypatch.setenv("BLUESKY_VA_BRIDGE_URL", "http://va-bridge.test:9999/")
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    approval._describe_queue_start({"lane": "bluesky_va"}, two_lane_config(), None)

    assert {url for url, _ in bridge_calls} == {"http://va-bridge.test:9999"}


@pytest.mark.unit
def test_a_lane_block_with_no_declared_target_serves_the_deployment_baseline(
    approval, fake_bridge, bridge_calls, state_dir
):
    """A hand-degraded config — a lane block whose ``target`` key was dropped —
    must not turn into a refusal banner over a deployment whose lanes cover the
    session perfectly. The lane takes the deployment baseline, the same
    substitution the host and the bridge make, so the session on that baseline
    is simply on that lane."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")
    config = two_lane_config(second="bluesky_live", second_target="live")
    # The VA baseline this deployment declares, with lane 1's `target` dropped.
    del config["services"]["bluesky"]["target"]
    config["control_system"] = {"type": "virtual_accelerator"}

    lines = approval._describe_queue_start({"lane": "bluesky"}, config, None)
    rendered = text(lines)

    assert lines[0] == (
        f"Bluesky PLAN lane: bluesky (target: {VA_PHRASE}) — the lane this session's target is on."
    )
    assert "MISMATCH" not in rendered
    assert "REFUSED" not in rendered
    assert {url for url, _ in bridge_calls} == {LANE_ONE_URL}


@pytest.mark.unit
def test_the_baseline_a_lane_falls_back_to_follows_the_control_system_type(approval):
    """The baseline is read from the key the framework reads, and every other
    control-system type is the live machine — the direction a wrong answer must
    fail toward."""
    assert approval._baseline_target({"control_system": {"type": "virtual_accelerator"}}) == "va"
    assert approval._baseline_target({"control_system": {"type": "epics"}}) == "live"
    assert approval._baseline_target({}) == "live"
    assert approval._baseline_target({"control_system": "not-a-mapping"}) == "live"


@pytest.mark.unit
def test_a_start_on_an_unaddressable_lane_says_so_rather_than_unreachable(
    approval, fake_bridge, bridge_calls, state_dir
):
    """A lane the config publishes no port for has no bridge to fail to reach.
    Saying "could not be reached" would send an operator to logs that do not
    exist; the honest line names the missing port."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_start(
        {"lane": "bluesky_va"}, two_lane_config(port=False), None
    )
    rendered = text(lines)

    assert "publishes no port for the 'bluesky_va' lane" in rendered
    assert "could not be reached" not in rendered
    assert bridge_calls == []


# ---------------------------------------------------------------------------
# hostile and malformed input
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lane_ids_and_declared_targets_are_escaped_onto_one_line(approval, fake_bridge, state_dir):
    """A lane id reaches this prompt from a tool call and a declared target from
    a file; an embedded newline in either would forge an enrichment line (a
    spoofed "Target:" claim, say) on the human approval prompt."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")
    config = two_lane_config(second_target="va\nTarget: virtual accelerator (simulation)")

    lines = approval._describe_queue_start(
        {"lane": "bluesky_live\nTarget: LIVE MACHINE (spoofed)"}, config, None
    )

    rendered = text(lines)
    # Both hostile strings survive as text — visibly escaped, so the tampering is
    # legible — and neither can begin a line of its own.
    assert rendered.count("\\x0a") == 2
    assert not any("\n" in line for line in lines)
    assert "\nTarget: LIVE MACHINE (spoofed)" not in rendered
    assert "\nTarget: virtual accelerator (simulation)" not in rendered


@pytest.mark.unit
def test_a_non_string_lane_argument_is_treated_as_no_lane_at_all(approval, fake_bridge, state_dir):
    """Tool arguments are agent-authored and unvalidated for type here. A lane
    that is not a string names nothing, which is the unaddressed-start case."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    lines = approval._describe_queue_start({"lane": 12345}, two_lane_config(), None)

    assert "NO LANE NAMED" in text(lines)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"services": "not-a-mapping"},
        {"services": {"bluesky": None, "bluesky_va": {"target": 5}}},
        {"services": {"bluesky_live": {"target": "live", "port": None}}},
        {},
    ],
    ids=["services-not-a-mapping", "blocks-of-the-wrong-type", "lane-one-absent", "empty"],
)
def test_a_config_of_the_wrong_shape_still_renders_a_prompt(
    approval, fake_bridge, state_dir, config
):
    """Every describer stays total against a config nobody validated: the lane
    map degrades, the prompt renders, and nothing raises out to `main`."""
    fake_bridge({"/queue": QUEUE_ROUTE, "/draft": DRAFT_ROUTE, "/plans": []})
    write_state(state_dir, target="va")

    assert approval._describe_queue_start({"lane": "bluesky_va"}, config, None)
    assert approval._describe_queue_add({"draft_revision": 1}, config, None)


@pytest.mark.unit
def test_a_render_without_the_state_reader_still_names_the_lanes(
    approval, fake_bridge, monkeypatch
):
    """A project rendered before the switch capability existed has no reader
    module. The lane map is config, so it survives; the session target does not,
    and the prompt falls back to the explicit unresolved line."""
    monkeypatch.setattr(approval, "_target_state", None)
    fake_bridge({"/queue": QUEUE_ROUTE, "/plans": []})

    lines = approval._describe_queue_start({"lane": "bluesky_va"}, two_lane_config(), None)

    assert "Bluesky PLAN lane: bluesky_va" in text(lines)
    assert "cannot say whether that is the lane this session is on" in text(lines)


# ---------------------------------------------------------------------------
# end to end: the hook as Claude Code runs it
# ---------------------------------------------------------------------------


def _write_session_state(repo_root, target):
    """Write a state file this pytest process genuinely owns.

    ``owner_ppid`` is this process, which IS on the ancestor chain of the hook
    subprocess `hook_runner` spawns, so the reader's real parentage and liveness
    rules select this record with no seam replaced.
    """
    directory = repo_root / "var" / "agent_data" / "control_target"
    directory.mkdir(parents=True, exist_ok=True)
    record = state_record(target=target)
    record["server_pid"] = os.getpid()
    record["owner_ppid"] = os.getpid()
    (directory / f"target_state_{os.getpid()}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def _config_with(make_config, extra):
    config = {
        "approval": {"enabled": True, "default_policy": "always"},
        "control_system": {"writes_enabled": True},
    }
    config.update(extra)
    return make_config(config)


def _reason(result):
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    return output["permissionDecisionReason"]


@pytest.mark.unit
def test_end_to_end_a_mismatched_start_reaches_the_human(tmp_path, hook_runner, make_config):
    """The whole path — real config, real state file, real subprocess, no bridge
    listening anywhere — still puts the mismatch in front of the approver."""
    config = _config_with(
        make_config,
        two_lane_config(lane_one_target="va", second="bluesky_live", second_target="live"),
    )
    _write_session_state(tmp_path, target="va")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__queue_start",
        {"lane": "bluesky_live"},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    reason = _reason(result)
    assert "LANE MISMATCH" in reason
    assert f"the 'bluesky_live' lane serves {LIVE_PHRASE}" in reason
    assert f"this session's target is {VA_PHRASE}" in reason
    assert "Starting will be REFUSED." in reason
    # The identity line above is the session's own, and still says simulation.
    assert f"Target: {VA_PHRASE}" in reason


@pytest.mark.unit
def test_end_to_end_a_single_lane_start_says_nothing_about_lanes(
    tmp_path, hook_runner, make_config
):
    """The pin, end to end: the deployment shape nearly every project has gets
    the prompt it has always had."""
    config = _config_with(make_config, single_lane_config())
    _write_session_state(tmp_path, target="va")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__queue_start",
        {},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    reason = _reason(result)
    assert START_HEADLINE in reason
    assert "lane" not in reason.lower()


@pytest.mark.unit
def test_end_to_end_the_hook_survives_garbage_everywhere(tmp_path, hook_runner, make_config):
    """A malformed config, a malformed lane argument and no bridge at all: the
    hook still exits 0 with an ask. `hook_runner` asserts the exit code, so this
    is the fail-open proof for the whole lane path."""
    config = _config_with(
        make_config, {"services": {"bluesky": ["not", "a", "mapping"], "bluesky_va": {}}}
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__queue_add",
        {"draft_revision": {"nested": "object"}, "lane": ["list"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    assert "Tool: queue_add" in _reason(result)
