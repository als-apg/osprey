"""Operator surfacing for the control-system target switch (FR-7).

Two surfaces are pinned here, both of which answer the same operator question —
*which machine is this touching?* — in the two places an operator actually
looks:

* the **web-terminal activity strip**, where a control-system activity event
  (``channel_write``, ``execute``, ``execute_file``) now names the target the
  session is on, and a target switch itself is reported as its own event on
  success **and** on failure;
* the **build-frozen safety rule** ``control-system-safety.md.j2``, which has to
  describe a switchable deployment as standing truth: which targets exist, that
  every approval prompt names the active one, that the roster is the authority
  on availability, that ``control_target_set`` is the only way to change
  targets, and that the deny list does not move when the target does.

The template half lives in this file rather than beside the other rules-render
tests because it is the same contract as the activity half: one task's answer to
"the operator must be able to tell which machine is in play". The pre-existing
render tests in ``tests/templates/test_control_system_safety_rule.py`` continue
to own the unswitched shape of that rule.

Honest spelling, and why it matters
-----------------------------------
The target is stamped on an activity event **only when a controls server this
session owns has published one**. There is no "baseline" fallback: the
deployment baseline of a ``mock`` project resolves to ``live``, and stamping
that on an activity event would tell the operator a mock write touched the real
machine. An absent record therefore means an absent key — "unknown" and "live"
are different claims.
"""

import json
import os

import pytest

from osprey.mcp_server import http
from osprey.mcp_server.control_system import target_state

#: A parent PID belonging to nobody — stands in for another session's server.
_FOREIGN_PPID = 999_999

#: The route bounds ``detail`` at 1024 characters (``agent_activity`` router).
_MAX_DETAIL_LEN = 1024


# ── fixtures / helpers ──────────────────────────────────────────────────────
@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Redirect the target-state directory into ``tmp_path``.

    Rebinding the one name ``target_state`` resolves the shared data root with
    keeps a real deployment's ``var/agent_data`` invisible in both directions.
    """
    root = tmp_path / "agent_data"
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    return root / target_state.STATE_DIR_NAME


@pytest.fixture
def posted(monkeypatch):
    """Capture every activity POST instead of talking to a web terminal."""
    calls: list[tuple[str, dict]] = []

    def _capture(url, payload, *, timeout=3):
        calls.append((url, payload))

    monkeypatch.setattr(http, "post_json", _capture)
    monkeypatch.setattr(http, "web_terminal_url", lambda: "http://127.0.0.1:10100")
    return calls


def write_state(state_dir, *, target, owner_ppid=None, server_pid=None):
    """Write one published target-state record, the way the server writes it."""
    state_dir.mkdir(parents=True, exist_ok=True)
    server_pid = os.getpid() if server_pid is None else server_pid
    owner_ppid = os.getppid() if owner_ppid is None else owner_ppid
    path = state_dir / f"{target_state.STATE_FILE_PREFIX}{server_pid}.json"
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 1,
                "server_pid": server_pid,
                "owner_ppid": owner_ppid,
                "targets": {
                    "live": {"label": "live machine", "endpoint": "gw:5064"},
                    "va": {"label": "virtual accelerator", "endpoint": "localhost:5074"},
                },
                "children": [],
            }
        )
    )
    return path


def only_target(calls):
    """The ``target`` sub-dict of the single captured activity POST."""
    assert len(calls) == 1, f"expected exactly one activity POST, got {len(calls)}"
    return calls[0][1]["target"]


# ── the session target on control-system activity ───────────────────────────
def test_channel_activity_names_the_published_target(state_root, posted):
    """A write reported while the session is on ``va`` says so, in the one
    field the activity route carries through to the browser."""
    write_state(state_root, target="va")

    http.notify_agent_activity("channel_write", "channel", detail="SR:MAG:QF:01:CURRENT:SP")

    assert only_target(posted)["detail"] == "[va] SR:MAG:QF:01:CURRENT:SP"


def test_execute_activity_names_the_published_target(state_root, posted):
    """The executor's write report is stamped by the same seam — the emitting
    tool passes nothing extra, so the two surfaces cannot describe the same
    session differently."""
    write_state(state_root, target="live")

    http.notify_agent_activity(
        "execute", "channel", detail="ran a script with control-system writes"
    )

    assert only_target(posted)["detail"] == "[live] ran a script with control-system writes"


def test_no_published_state_leaves_the_detail_alone(state_root, posted):
    """No server has published a target: the key is absent rather than guessed.
    A deployment baseline is not a claim about what this write touched."""
    http.notify_agent_activity("channel_write", "channel", detail="SR:MAG:QF:01:CURRENT:SP")

    assert only_target(posted)["detail"] == "SR:MAG:QF:01:CURRENT:SP"
    assert http.resolve_activity_target() is None


def test_another_sessions_record_is_not_ours(state_root, posted):
    """A live server owned by a different Claude Code process says nothing
    about this session, so nothing is stamped."""
    write_state(state_root, target="va", owner_ppid=_FOREIGN_PPID)

    http.notify_agent_activity("channel_write", "channel", detail="SR:MAG:QF:01:CURRENT:SP")

    assert only_target(posted)["detail"] == "SR:MAG:QF:01:CURRENT:SP"


def test_unknown_target_name_is_not_stamped(state_root, posted):
    """A record naming a target this framework does not know is no answer."""
    write_state(state_root, target="banana")

    http.notify_agent_activity("channel_write", "channel", detail="SR:MAG:QF:01:CURRENT:SP")

    assert only_target(posted)["detail"] == "SR:MAG:QF:01:CURRENT:SP"


def test_non_control_activity_is_never_stamped(state_root, posted):
    """Only control-system activity is about a machine. A panel highlight is
    not, and prefixing it would put a target on an event that has none."""
    write_state(state_root, target="va")

    http.notify_agent_activity("entry_create", "panel", panel="ariel", detail="entry-7")

    assert only_target(posted)["detail"] == "entry-7"


def test_detail_less_activity_still_names_the_target(state_root, posted):
    """A control-system event with nothing to say about *what* it touched still
    says *where* it touched it."""
    write_state(state_root, target="va")

    http.notify_agent_activity("channel_write", "channel")

    assert only_target(posted)["detail"] == "[va]"


def test_stamp_survives_the_routes_detail_bound(state_root, posted):
    """The prefix goes on before truncation and sits at the front, so a bulk
    write long enough to be cut still names its target."""
    write_state(state_root, target="va")

    http.notify_agent_activity("channel_write", "channel", detail="X" * 4000)

    detail = only_target(posted)["detail"]
    assert detail.startswith("[va] ")
    assert len(detail) <= _MAX_DETAIL_LEN


def test_an_unreadable_state_directory_never_breaks_an_emit(state_root, monkeypatch, posted):
    """Resolution is best-effort: a failure to answer must degrade to an
    unstamped event, never to an exception in a fire-and-forget notify."""

    def _boom():
        raise RuntimeError("state dir exploded")

    monkeypatch.setattr(target_state, "state_dir", _boom)

    http.notify_agent_activity("channel_write", "channel", detail="SR:MAG:QF:01:CURRENT:SP")

    assert only_target(posted)["detail"] == "SR:MAG:QF:01:CURRENT:SP"
    assert http.resolve_activity_target() is None


# ── the switch event ────────────────────────────────────────────────────────
def test_switch_success_event_shape(posted):
    """A completed switch reports both targets, the outcome and the generation
    the session is now on."""
    http.notify_target_switch(
        from_target="live",
        to_target="va",
        outcome=http.SWITCH_OUTCOME_SUCCESS,
        generation=3,
    )

    url, payload = posted[0]
    assert url.endswith("/api/agent-activity")
    assert payload["tool"] == http.TARGET_SWITCH_TOOL
    assert payload["target"]["kind"] == http.TARGET_SWITCH_KIND
    assert payload["target"]["detail"] == "live → va · success (generation 3)"


def test_switch_failure_event_names_the_reason(posted):
    """A refused or aborted switch is reported too — an operator who saw the
    approval prompt has to see what came of it, and the session is still on the
    target it started from."""
    http.notify_target_switch(
        from_target="live",
        to_target="va",
        outcome=http.SWITCH_OUTCOME_FAILURE,
        reason="probe channel never connected",
    )

    detail = only_target(posted)["detail"]
    assert detail == "live → va · failure: probe channel never connected"


def test_switch_failure_without_a_reason_still_reports_the_outcome(posted):
    """The reason is optional; the outcome never is."""
    http.notify_target_switch(
        from_target="va", to_target="live", outcome=http.SWITCH_OUTCOME_FAILURE
    )

    assert only_target(posted)["detail"] == "va → live · failure"


def test_switch_outcomes_are_two_distinct_spellings():
    """The switch tool takes its outcome from these constants rather than
    spelling it itself, so the two events cannot drift apart."""
    assert http.SWITCH_OUTCOME_SUCCESS != http.SWITCH_OUTCOME_FAILURE
    assert {http.SWITCH_OUTCOME_SUCCESS, http.SWITCH_OUTCOME_FAILURE} == {"success", "failure"}


def test_switch_event_is_not_target_stamped(state_root, posted):
    """The event names both targets itself; a session-target prefix on top of
    that would be a third opinion about where the session is."""
    write_state(state_root, target="live")

    http.notify_target_switch(
        from_target="live", to_target="va", outcome=http.SWITCH_OUTCOME_SUCCESS, generation=1
    )

    assert only_target(posted)["detail"] == "live → va · success (generation 1)"


def test_switch_emit_never_raises(monkeypatch):
    """Fire-and-forget, like every other emit here: a switch must not fail
    because the web terminal is not running."""

    def _boom():
        raise RuntimeError("no web terminal")

    monkeypatch.setattr(http, "web_terminal_url", _boom)

    http.notify_target_switch(
        from_target="live", to_target="va", outcome=http.SWITCH_OUTCOME_SUCCESS, generation=1
    )


async def test_switch_async_emits_the_same_body(posted):
    """The awaitable form is the one the switch tool calls; it must produce the
    same frame the sync form does, off the event loop."""
    await http.notify_target_switch_async(
        from_target="live",
        to_target="va",
        outcome=http.SWITCH_OUTCOME_SUCCESS,
        generation=2,
    )

    assert only_target(posted)["detail"] == "live → va · success (generation 2)"


# ── the build-frozen safety rule ────────────────────────────────────────────
#: Everything the switch-aware rule has to state, as standing truth.
SWITCH_RULE_MARKERS = (
    "## Which Machine You Are Talking To",
    "the real accelerator",
    "virtual accelerator",
    "the live stand-in",
    "every approval prompt names the active target",
    "`control_target` reports the roster",
    "`control_target_set` is the only way to change targets",
    "stay refused on every target",
)

#: The three machines the rule has to name, and the one sentence about each
#: that an operator's safety depends on. The stand-in is the one that reads
#: wrong if it is described as a mode of ``live``: it is a machine of its own,
#: carrying a real machine's posture.
#:
#: The label clause is pinned as CONDITIONAL on purpose. ``_label`` drops the
#: ``(stand-in)`` parenthesis when the target's endpoint is not this
#: deployment's stand-in container, and the roster still carries the row (the
#: switch refuses it as ``standin_not_deployed``). A rule promising the
#: parenthesis unconditionally would have the agent treat a plain LIVE MACHINE
#: label as proof it is somewhere it is not.
THREE_MACHINE_MARKERS = (
    "**live** — the real accelerator",
    "**va** — the virtual accelerator",
    "**standin** — the live stand-in",
    "a soft IOC this deployment runs for itself",
    "marked **(stand-in)** only while this deployment has actually stood the soft IOC up",
)


def _render_rule_raw(**context) -> str:
    """Render the safety rule straight from Jinja with an explicit context."""
    from osprey.cli.templates.manager import TemplateManager

    template = TemplateManager().jinja_env.get_template(
        "claude_code/claude/rules/control-system-safety.md.j2"
    )
    return template.render(**context)


def _render_rule(**context) -> str:
    """The rendered rule, whitespace-normalised.

    The rule is hard-wrapped prose, so a sentence-level assertion against the
    raw text would pin the line breaks rather than the wording.
    """
    return " ".join(_render_rule_raw(**context).split())


def test_switch_aware_rule_states_the_switchable_reality():
    """With the switch rendered, the rule describes it: which targets exist,
    that the prompt names the active one, that the roster is the authority,
    that one approval-gated tool moves it, and that the deny list does not."""
    content = _render_rule(
        control_system_type="epics",
        enabled_servers={"controls"},
        target_switch_enabled=True,
    )

    for marker in SWITCH_RULE_MARKERS:
        assert marker in content, f"switch-aware rule missing: {marker!r}"


def test_switch_aware_rule_names_all_three_machines():
    """A target is a machine, and OSPREY has three of them.

    The rule is frozen at build time and the render is not told which targets
    this deployment configures, so it describes the vocabulary and points the
    agent at ``control_target`` for what is actually here. What it must not do
    is describe two machines: an agent told the choice is live-or-simulation
    reads ``standin`` — when the roster offers it — as a spelling of one of
    those, and the stand-in is neither. It carries a real machine's posture
    without being the facility's machine, which is exactly the distinction an
    operator approving a write is relying on.
    """
    content = _render_rule(
        control_system_type="epics",
        enabled_servers={"controls"},
        target_switch_enabled=True,
    )

    for marker in THREE_MACHINE_MARKERS:
        assert marker in content, f"three-machine rule missing: {marker!r}"
    assert "either of two machines" not in content, "rule still describes only two machines"


def test_switch_aware_rule_is_absent_without_the_switch():
    """A deployment with one target must not be told it has two. The gate is
    the rendered switch capability, and it is off by default."""
    for context in (
        {"control_system_type": "epics", "enabled_servers": {"controls"}},
        {
            "control_system_type": "epics",
            "enabled_servers": {"controls"},
            "target_switch_enabled": False,
        },
        {"control_system_type": "mock", "enabled_servers": {"controls"}},
    ):
        content = _render_rule(**context)
        for marker in SWITCH_RULE_MARKERS:
            assert marker not in content, f"{context}: switch text leaked: {marker!r}"
        assert "control_target" not in content, f"{context}: switch tool named without the switch"


def test_switch_section_does_not_disturb_the_existing_rule_shape():
    """The rule's other sections are a discovery contract; the new section is
    additive and sits outside the tool-routing section, which several tests
    read as a protocol-neutral slice."""
    content = _render_rule_raw(
        control_system_type="epics",
        enabled_servers={"controls"},
        target_switch_enabled=True,
    )

    for heading in (
        "### Allowed",
        "### Prohibited",
        "### Why This Matters",
        "## Write Operations",
        "## Choosing the Right Tool",
    ):
        assert heading in content, f"missing section heading: {heading}"

    start = content.index("## Choosing the Right Tool")
    next_heading = content.find("\n## ", start + 1)
    routing = content[start : next_heading if next_heading != -1 else len(content)]
    assert "control_target" not in routing, "switch text leaked into the routing section"


# ── the gate: which deployments render the switch ───────────────────────────
#
# The build-time gate must not be its own opinion about what "switchable" means.
# It delegates to osprey_connectors.types.switch_capable, the predicate the
# controls server uses at run time to decide whether its tools are served by a
# connector-host child — a frozen rule promising a switch the runtime refuses to
# perform is worse than a rule that never mentions one. The cases below are
# chosen for the two ways a locally-restated predicate gets it wrong: a doocs
# deployment (switchable, but named by no epics literal) and a mock deployment
# carrying a live block (NOT switchable — its baseline never resolves back to
# its own declared type, so a session on "live" would reach a real machine the
# config never selected).
_EPICS_BLOCK = {"gateways": {"read_only": {"address": "gw.example.org"}}}
_DOOCS_BLOCK = {"facility": "XFEL"}
_VA_BLOCK = {"gateways": {"read_only": {"address": "localhost"}}}
_MOCK_BLOCK = {"noise_level": 0.0}

#: (label, control_system section, switch-capable?) — shared by the gate test,
#: the end-to-end render test and the runtime-agreement test, so the three
#: cannot be pinned against three different pictures of a deployment.
GATE_CASES = [
    (
        "epics and va",
        {"type": "epics", "connector": {"epics": _EPICS_BLOCK, "virtual_accelerator": _VA_BLOCK}},
        True,
    ),
    (
        "doocs and va",
        {"type": "doocs", "connector": {"doocs": _DOOCS_BLOCK, "virtual_accelerator": _VA_BLOCK}},
        True,
    ),
    (
        "va baseline with a live block",
        {
            "type": "virtual_accelerator",
            "connector": {"epics": _EPICS_BLOCK, "virtual_accelerator": _VA_BLOCK},
        },
        True,
    ),
    ("epics only", {"type": "epics", "connector": {"epics": _EPICS_BLOCK}}, False),
    (
        "va only",
        {"type": "virtual_accelerator", "connector": {"virtual_accelerator": _VA_BLOCK}},
        False,
    ),
    ("mock only", {"type": "mock", "connector": {"mock": _MOCK_BLOCK}}, False),
    (
        "mock baseline carrying a live block",
        {
            "type": "mock",
            "connector": {
                "mock": _MOCK_BLOCK,
                "epics": _EPICS_BLOCK,
                "virtual_accelerator": _VA_BLOCK,
            },
        },
        False,
    ),
    (
        # The baseline is a configured target whether or not its block carries
        # keys — a session sits on it either way — so an epics deployment with
        # an empty epics block still has two machines once the VA is beside it.
        "empty live block beside a va",
        {"type": "epics", "connector": {"epics": {}, "virtual_accelerator": _VA_BLOCK}},
        True,
    ),
    ("no connector section", {"type": "epics"}, False),
    (
        "tuning keys but one machine",
        {
            "type": "epics",
            "connector": {"epics": _EPICS_BLOCK},
            "target_switch": {"drain_timeout_s": 5},
        },
        False,
    ),
]


def _derived(control_system, tmp_path):
    """The config-derived template context for a ``control_system`` section."""
    from osprey.cli.templates import claude_code

    return claude_code.config_derived_context({"control_system": control_system}, tmp_path)


@pytest.mark.parametrize(("label", "control_system", "expected"), GATE_CASES)
def test_switch_gate_follows_the_runtime_capability(tmp_path, label, control_system, expected):
    """The rendered flag is the runtime's own capability question, not a
    connector-block spelling: doocs switches, and the tuning keys do not make a
    one-machine deployment switchable."""
    assert _derived(control_system, tmp_path)["target_switch_enabled"] is expected, label


@pytest.mark.parametrize(("label", "control_system", "expected"), GATE_CASES)
def test_build_time_gate_agrees_with_the_connector_host_manager(label, control_system, expected):
    """The anti-drift pin: the build and the server must answer identically, or
    the frozen rule describes a deployment the server does not run."""
    from osprey.mcp_server.control_system.connector_host_manager import switch_capable as runtime

    assert runtime({"control_system": control_system}) is expected, label


@pytest.mark.parametrize(("label", "control_system", "expected"), GATE_CASES)
def test_rule_renders_the_switch_section_for_switchable_deployments_only(
    tmp_path, label, control_system, expected
):
    """End to end through the real context builder: the section the agent reads
    appears exactly where a session can actually be pointed at two machines."""
    context = _derived(control_system, tmp_path)
    content = _render_rule(enabled_servers={"controls"}, **context)

    for marker in SWITCH_RULE_MARKERS:
        assert (marker in content) is expected, f"{label}: {marker!r}"
