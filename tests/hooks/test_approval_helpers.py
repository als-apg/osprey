"""Unit tests for `osprey_approval`'s pure helpers, imported in-process.

The hook's end-to-end behaviour is covered by `test_approval_hook.py` (real
subprocesses) and its queue enrichment by `test_approval_queue_enrichment.py`
(a real HTTP server standing in for the bridge). What neither reaches cheaply
is the *shape* of the individual helpers: the escape token `_sanitize_label`
emits for each hostile code point, the exact wording `_revision_match_line`
picks per revision pairing, the precedence `_resolve_bridge_url` applies, and
the per-branch line list `_describe_plan_provenance` builds from a bridge
response. Those are pure functions of their arguments, so this file calls them
directly through the `hook_module` seam (see `conftest.import_hook`).

No test here touches the network. `_bridge_get_json` is the hook's single
egress point for the launch enrichment, and every test that exercises a caller
of it replaces it with a routing table, which also lets the tests assert *which*
bridge URL and paths the callers ask for.
"""

from __future__ import annotations

import pytest

from osprey.port_layout import default_port


@pytest.fixture
def approval(hook_module):
    """The `osprey_approval` module, imported through the test seam.

    Imported in a fixture rather than at module scope: the hook prepends its own
    directory to `sys.path` on import, and `import_hook` is what undoes that.
    """
    return hook_module("osprey_approval")


@pytest.fixture
def bridge_calls():
    """Recorder for the (base_url, path) pairs a helper asks the bridge for."""
    return []


@pytest.fixture
def fake_bridge(approval, bridge_calls, monkeypatch):
    """Factory replacing `_bridge_get_json` with a routing table.

    Takes ``{path: body}``; an unmapped path yields `None`, which is exactly what
    the real helper returns for a 404, an unreachable bridge or a malformed
    body — so a route left out of the table is a genuine failure case, not a
    test-harness gap.
    """

    def _install(routes: dict[str, object]):
        def _get(base_url, path, timeout=3.0):
            bridge_calls.append((base_url, path))
            return routes.get(path)

        monkeypatch.setattr(approval, "_bridge_get_json", _get)

    return _install


# --------------------------------------------------------------------------
# _sanitize_label
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("\x85", "\\x85"),
        ("\u2028", "\\x2028"),
        ("\u2029", "\\x2029"),
        ("\n", "\\x0a"),
        ("\r", "\\x0d"),
        ("\x7f", "\\x7f"),
        ("\x00", "\\x00"),
    ],
    ids=["nel", "line-sep", "para-sep", "newline", "carriage-return", "del", "nul"],
)
def test_sanitize_label_escapes_line_breaking_code_points(approval, raw, escaped):
    """Every code point that could break the prompt onto a new line is escaped.

    The label lands in a human approval prompt built by line concatenation, so a
    raw break in an agent-supplied plan name would let the plan forge an
    enrichment line of its own (a spoofed "Validation status: PASSED", say).
    The C0 range and DEL are the obvious carriers; NEL, LINE SEPARATOR and
    PARAGRAPH SEPARATOR are the ones a terminal may also honour.
    """
    result = approval._sanitize_label(f"before{raw}after")

    assert raw not in result
    assert result == f"before{escaped}after"


@pytest.mark.unit
def test_sanitize_label_passes_ordinary_text_through(approval):
    """Printable text — including non-ASCII — is returned byte for byte.

    The escaping must be invisible in the normal case; a label mangled into
    ``\\xNN`` soup would train approvers to ignore the escapes that matter.
    """
    label = "orbit_correction_v2 (µm, 90° phase) — ARM/BPM:1"

    assert approval._sanitize_label(label) == label


@pytest.mark.unit
def test_sanitize_label_stringifies_non_string_input(approval):
    """Non-string metadata is coerced rather than raising.

    ``PLAN_METADATA`` is agent-authored and unvalidated for type, so a numeric
    or `None` field must render, not blow up the approval prompt.
    """
    assert approval._sanitize_label(42) == "42"
    assert approval._sanitize_label(None) == "None"


# --------------------------------------------------------------------------
# _revision_match_line
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_revision_match_line_states_a_match_plainly(approval):
    """Equal pinned and current revisions get the quiet wording."""
    line = approval._revision_match_line(7, 7)

    assert line == "Draft revision 7 — matches pinned revision 7."
    assert "⚠️" not in line


@pytest.mark.unit
@pytest.mark.parametrize(
    ("pinned", "current"),
    [(7, 8), (None, 8), (8, None)],
    ids=["moved-on", "no-pin", "no-current"],
)
def test_revision_match_line_is_loud_on_any_mismatch(approval, pinned, current):
    """Anything short of an exact match warns, and shows both revisions.

    A missing pin is treated as a mismatch on purpose: the approver cannot tell
    from a silent line whether the draft is the one the agent staged.
    """
    line = approval._revision_match_line(pinned, current)

    assert "DRAFT CHANGED" in line
    assert str(pinned) in line
    assert str(current) in line
    assert "CURRENT draft" in line


# --------------------------------------------------------------------------
# build_approval_output / build_allow_output
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_build_approval_output_emits_an_ask_envelope(approval):
    """The ask envelope carries the event name, the decision, and the detail."""
    output = approval.build_approval_output("Tool: execute\nPolicy: always")

    assert set(output) == {"hookSpecificOutput"}
    specific = output["hookSpecificOutput"]
    assert specific["hookEventName"] == "PreToolUse"
    assert specific["permissionDecision"] == "ask"
    reason = specific["permissionDecisionReason"]
    assert "OSPREY APPROVAL REQUIRED" in reason
    assert "Tool: execute\nPolicy: always" in reason
    assert reason.endswith("Review the operation above and approve to proceed.")


@pytest.mark.unit
def test_build_allow_output_is_silent_in_a_dispatch_run(approval, monkeypatch):
    """Under `OSPREY_DISPATCH_RUN=1` the hook emits no decision at all.

    Hook aggregation in the CLI is not deny-dominates, so an explicit allow here
    would override the dispatch worker's per-trigger allowlist and widen a
    sandboxed run's tool surface. Emitting nothing hands the call back to the
    worker's own callback.
    """
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")

    assert approval.build_allow_output() == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "dispatch_run",
    [None, "", "0", "true", "2"],
    ids=["unset", "empty", "zero", "true", "two"],
)
def test_build_allow_output_is_explicit_outside_a_dispatch_run(approval, monkeypatch, dispatch_run):
    """Only the exact string "1" suppresses the allow; everything else allows.

    The suppression is a narrow carve-out for the dispatch worker, which sets
    the variable itself. A stray or unrelated value must not silently disable
    the explicit allow that overrides static permission lists.
    """
    if dispatch_run is None:
        monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    else:
        monkeypatch.setenv("OSPREY_DISPATCH_RUN", dispatch_run)

    assert approval.build_allow_output() == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }


# --------------------------------------------------------------------------
# _resolve_bridge_url
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_bridge_url_prefers_the_environment(approval, monkeypatch):
    """`BLUESKY_BRIDGE_URL` wins outright, even with config.yml set."""
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", "http://bridge.env:9000/")

    url = approval._resolve_bridge_url({"bluesky": {"bridge_url": "http://bridge.config:8000"}})

    assert url == "http://bridge.env:9000"


@pytest.mark.unit
def test_resolve_bridge_url_falls_back_to_config(approval, monkeypatch):
    """With no env override, `bluesky.bridge_url` from config.yml is used."""
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)

    url = approval._resolve_bridge_url({"bluesky": {"bridge_url": "http://bridge.config:8000/"}})

    assert url == "http://bridge.config:8000"


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [{}, {"bluesky": {}}, {"services": {}}, {"services": {"bluesky": {}}}],
    ids=["no-bluesky-section", "no-bridge-url-key", "no-services", "no-published-port"],
)
def test_resolve_bridge_url_falls_back_to_the_layout_slot(approval, monkeypatch, config):
    """Neither env nor config leaves the bridge's slot in the port block.

    None of these configs sets ``deployment.port_base``, so the base is the
    layout's default one — but the number is derived, never written out here.
    """
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)

    assert approval._resolve_bridge_url(config) == f"http://127.0.0.1:{default_port('bluesky')}"


@pytest.mark.unit
def test_resolve_bridge_url_follows_a_moved_port_base(approval, monkeypatch):
    """A deployment that moved its block moved its bridge with it.

    The regression this pins: a frozen default would keep dialing the layout's
    default base, which on a host running two deployments is the other one's
    bridge.
    """
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)

    url = approval._resolve_bridge_url({"deployment": {"port_base": 20000}})

    assert url == f"http://127.0.0.1:{default_port('bluesky', base=20000)}"


@pytest.mark.unit
def test_resolve_bridge_url_dials_the_port_the_deployment_publishes(approval, monkeypatch):
    """With no env and no `bluesky.bridge_url`, `services.bluesky.port` — the port
    the build wrote for the bridge it deploys, or projected into an attached
    render — is dialed on loopback. The same rule as
    `osprey.bluesky_bridge_connection.bridge_url_from_config`."""
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)

    url = approval._resolve_bridge_url({"services": {"bluesky": {"port": 18090}}})

    assert url == "http://127.0.0.1:18090"


@pytest.mark.unit
def test_resolve_bridge_url_config_url_beats_the_published_port(approval, monkeypatch):
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)

    url = approval._resolve_bridge_url(
        {
            "bluesky": {"bridge_url": "http://bridge.config:8000"},
            "services": {"bluesky": {"port": 1}},
        }
    )

    assert url == "http://bridge.config:8000"


@pytest.mark.unit
def test_resolve_bridge_url_ignores_an_empty_environment_value(approval, monkeypatch):
    """An empty `BLUESKY_BRIDGE_URL` is no override, not an empty base URL.

    Exporting the variable unset is a common shell accident; taking it literally
    would point every bridge call at a relative path.
    """
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", "")

    url = approval._resolve_bridge_url({"bluesky": {"bridge_url": "http://bridge.config:8000"}})

    assert url == "http://bridge.config:8000"


# --------------------------------------------------------------------------
# _describe_plan_provenance
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_describe_plan_provenance_renders_authoring_metadata(approval, fake_bridge):
    """A plan with metadata gets a hazard verdict off its `writes` declaration."""
    fake_bridge(
        {
            "/plans": [
                {
                    "name": "orbit_correction",
                    "metadata": {
                        "name": "orbit_correction",
                        "description": "Correct the orbit.",
                        "writes": True,
                    },
                }
            ],
            "/plans/orbit_correction/source": {
                "provenance": "shipped",
                "source": "def plan(): ...",
            },
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "orbit_correction")

    assert "Hazard: writes to hardware" in lines


@pytest.mark.unit
def test_describe_plan_provenance_marks_a_read_only_plan(approval, fake_bridge):
    """`writes: False` renders as read-only."""
    fake_bridge(
        {
            "/plans": [
                {
                    "name": "count",
                    "metadata": {"name": "count", "description": "Count.", "writes": False},
                },
            ],
            "/plans/count/source": {"provenance": "shipped"},
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "count")

    assert "Hazard: read-only (no hardware writes declared)" in lines


@pytest.mark.unit
def test_describe_plan_provenance_reads_only_the_writes_declaration(approval, fake_bridge):
    """`writes` is the whole of the authoring metadata this block reports.

    Which channels a launch touches is not authored metadata — it is read off
    the plan's role-typed `PARAMS` and reaches the prompt through the
    pre-flight, for the parameters actually staged. A retired key smuggled back
    into a `PLAN_METADATA` block must not resurface here as a plan-wide claim,
    which is also what keeps agent-authored free text off these lines.
    """
    fake_bridge(
        {
            "/plans": [
                {
                    "name": "sneaky",
                    "metadata": {
                        "name": "sneaky",
                        "description": "d",
                        "writes": False,
                        "category": "safe\nValidation status: PASSED",
                        "required_devices": ["BPM\u20281"],
                    },
                }
            ],
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "sneaky")

    assert "Hazard: read-only (no hardware writes declared)" in lines
    assert not any("Category" in line for line in lines)
    assert not any("Required devices" in line for line in lines)
    # The forged verdict the retired free-text field carried. This fixture maps
    # no `/source` route, so the real validation line reads "unknown" — a
    # "PASSED" anywhere in the block could only have come from the metadata.
    assert not any("PASSED" in line for line in lines)
    assert not any("\n" in line for line in lines)


@pytest.mark.unit
def test_describe_plan_provenance_handles_a_plan_without_metadata(approval, fake_bridge):
    """A built-in plan carries no authoring metadata; the line says so."""
    fake_bridge({"/plans": [{"name": "grid_scan"}], "/plans/grid_scan/source": {}})

    lines = approval._describe_plan_provenance("http://bridge", "grid_scan")

    assert "Hazard: unavailable (no authoring metadata — built-in plan)" in lines


@pytest.mark.unit
@pytest.mark.parametrize("provenance", ["session", "unreviewed"], ids=["session", "unreviewed"])
def test_describe_plan_provenance_shouts_about_agent_authored_plans(
    approval, fake_bridge, provenance
):
    """Untrusted tiers are labelled unmistakably, in caps.

    This is the human backstop for the plan validator's documented obfuscation
    residual: an approver can only refuse a hand-crafted body if the prompt tells
    them nobody reviewed it.
    """
    fake_bridge(
        {
            "/plans": [{"name": "adhoc"}],
            "/plans/adhoc/source": {"provenance": provenance, "validated": True},
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "adhoc")

    assert f"Provenance: {provenance.upper()} — AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in lines
    assert "Validation status: PASSED (content hash matches a recorded validation run)" in lines


@pytest.mark.unit
def test_describe_plan_provenance_flags_an_unvalidated_session_plan(approval, fake_bridge):
    """No passing validation record is stated as a launch-time refusal."""
    fake_bridge(
        {
            "/plans": [{"name": "adhoc"}],
            "/plans/adhoc/source": {"provenance": "session", "validated": False},
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "adhoc")

    assert "Validation status: NO PASSING RECORD — would be refused at enqueue" in lines


@pytest.mark.unit
def test_describe_plan_provenance_reports_an_operator_supplied_plan(approval, fake_bridge):
    """A reviewed tier is named plainly, and validation does not apply to it."""
    fake_bridge(
        {
            "/plans": [{"name": "orm"}],
            "/plans/orm/source": {"provenance": "facility"},
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "orm")

    assert "Provenance: facility (operator-supplied)" in lines
    assert "Validation status: not applicable (operator-supplied plan)" in lines


@pytest.mark.unit
def test_describe_plan_provenance_falls_back_to_the_registry_provenance(approval, fake_bridge):
    """When `/source` is silent on provenance, the `/plans` entry supplies it."""
    fake_bridge(
        {
            "/plans": [{"name": "adhoc", "provenance": "session"}],
            "/plans/adhoc/source": {"validated": False},
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "adhoc")

    assert "Provenance: SESSION — AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in lines


@pytest.mark.unit
def test_describe_plan_provenance_degrades_when_the_bridge_is_silent(approval, fake_bridge):
    """Both endpoints failing yields short lines, not an exception.

    Every bridge call is fail-open by design — an unreachable bridge must never
    block the approval prompt from rendering.
    """
    fake_bridge({})

    lines = approval._describe_plan_provenance("http://bridge", "orm")

    assert "Hazard: unavailable (no authoring metadata — built-in plan)" in lines
    assert "Provenance: unknown" in lines
    assert "Validation status: unknown (could not reach the plan-source endpoint)" in lines
    assert not any(line.startswith("\nPlan source") for line in lines)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("truncated", "note"), [(True, " (truncated)"), (False, "")], ids=["truncated", "complete"]
)
def test_describe_plan_provenance_appends_the_plan_source_verbatim(
    approval, fake_bridge, truncated, note
):
    """The source block is rendered raw — unlike the labels — and flagged if cut.

    Escaping it would defeat its purpose: the approver is meant to read the real,
    multi-line plan body.
    """
    fake_bridge(
        {
            "/plans": [{"name": "adhoc"}],
            "/plans/adhoc/source": {
                "provenance": "session",
                "validated": True,
                "source": "def plan():\n    yield from count([det])",
                "truncated": truncated,
            },
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "adhoc")

    assert lines[-1] == f"\nPlan source{note}:\ndef plan():\n    yield from count([det])"


# --------------------------------------------------------------------------
# _describe_queue_add / _describe_queue_start / _describe_queue_stop
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "snapshot",
    [None, [], "not-json-object"],
    ids=["bridge-unreachable", "wrong-shape-list", "wrong-shape-string"],
)
def test_describe_queue_add_degrades_to_queue_state_only_for_an_unusable_draft(
    approval, monkeypatch, snapshot
):
    """A missing or misshaped `GET /draft` degrades to the queue-state line only.

    The plain tool/policy reason still asks for approval; only the draft detail
    is lost. The queue-state line survives because it comes from a different
    fetch — and it is the line that says whether approving this enqueue is
    approving an execution, so it must not be collateral damage.
    """
    monkeypatch.setattr(approval, "_bridge_get_json", lambda *args, **kwargs: snapshot)

    lines = approval._describe_queue_add({"draft_revision": 3}, {})

    assert lines == ["Queue state: unavailable (the bridge could not be reached)."]
    assert not any(line.startswith("Draft revision") for line in lines)


@pytest.mark.unit
def test_describe_queue_add_reports_an_empty_draft(approval, fake_bridge):
    """Nothing staged means nothing to queue, and the prompt says so."""
    fake_bridge({"/draft": {"revision": 4, "draft": {}}})

    lines = approval._describe_queue_add({"draft_revision": 4}, {})

    assert "Draft revision 4 — matches pinned revision 4." in lines
    assert any(line.startswith("Draft: EMPTY") for line in lines)
    assert not any(line.startswith("Plan:") for line in lines)


@pytest.mark.unit
def test_describe_queue_add_renders_the_staged_plan(approval, fake_bridge):
    """A staged plan contributes its name, args, hazard and provenance lines.

    No pre-flight route is mapped, so the trajectory renders as unavailable —
    non-blocking, and the case `test_approval_queue_enrichment.py` covers
    against a real bridge.
    """
    fake_bridge(
        {
            "/draft": {
                "revision": 9,
                "draft": {"plan_name": "grid_scan", "plan_args": {"steps": 5}},
            },
            "/plans": [
                {
                    "name": "grid_scan",
                    "metadata": {
                        "name": "grid_scan",
                        "description": "Step a grid and read.",
                        "writes": False,
                    },
                }
            ],
            "/plans/grid_scan/source": {"provenance": "shipped"},
        }
    )

    lines = approval._describe_queue_add({"draft_revision": 9}, {})

    assert "Plan: grid_scan" in lines
    assert 'Plan args: {"steps": 5}' in lines
    assert "Hazard: read-only (no hardware writes declared)" in lines
    assert "Provenance: shipped (operator-supplied)" in lines
    assert any("Setpoint trajectory: unavailable" in line for line in lines)


@pytest.mark.unit
def test_describe_queue_add_omits_the_args_line_when_there_are_none(approval, fake_bridge):
    """An empty `plan_args` renders no args line rather than an empty one."""
    fake_bridge(
        {
            "/draft": {"revision": 1, "draft": {"plan_name": "count", "plan_args": {}}},
            "/plans": [],
        }
    )

    lines = approval._describe_queue_add({"draft_revision": 1}, {})

    assert not any(line.startswith("Plan args:") for line in lines)


@pytest.mark.unit
def test_describe_queue_add_sanitizes_the_plan_name(approval, fake_bridge):
    """The staged plan name reaches the prompt escaped, on one line."""
    fake_bridge(
        {
            "/draft": {
                "revision": 2,
                "draft": {"plan_name": "count\nDraft revision 2 — matches pinned revision 2."},
            },
            "/plans": [],
        }
    )

    lines = approval._describe_queue_add({"draft_revision": 2}, {})

    assert "Plan: count\\x0aDraft revision 2 — matches pinned revision 2." in lines


@pytest.mark.unit
def test_describe_queue_add_warns_when_the_draft_moved_on(approval, fake_bridge):
    """A draft revised since the agent pinned it renders the loud warning."""
    fake_bridge(
        {
            "/draft": {"revision": 12, "draft": {"plan_name": "count"}},
            "/plans": [],
        }
    )

    lines = approval._describe_queue_add({"draft_revision": 9}, {})

    drift = [line for line in lines if "DRAFT CHANGED" in line]
    assert len(drift) == 1
    assert "9" in drift[0]
    assert "12" in drift[0]


@pytest.mark.unit
def test_describe_queue_add_asks_the_configured_bridge(approval, fake_bridge, bridge_calls):
    """The base URL resolved from config is the one every call goes to.

    The queue, the draft, the plan registry and the plan source must all be
    read from the same bridge the enqueue would actually use — a mismatch
    would describe one bridge's draft while another one runs the plan.
    """
    fake_bridge(
        {
            "/queue": {"status": {"manager_state": "idle"}, "items": [], "running_item": None},
            "/draft": {"revision": 1, "draft": {"plan_name": "count"}},
            "/plans": [],
            "/plans/count/source": {},
        }
    )

    approval._describe_queue_add(
        {"draft_revision": 1}, {"bluesky": {"bridge_url": "http://bridge.config:8000/"}}
    )

    assert {base_url for base_url, _ in bridge_calls} == {"http://bridge.config:8000"}
    assert [path for _, path in bridge_calls] == [
        "/queue",
        "/draft",
        "/plans",
        "/plans/count/source",
    ]


@pytest.mark.unit
def test_queue_activity_lines_classify_from_what_was_observed(approval):
    """A running item and an autostart flag each earn their own loud headline.

    Classified from the observation, not from a copy of the manager's
    active-state vocabulary — this hook cannot import OSPREY, and a replica of
    that list here would be one more thing to drift.
    """
    running = approval._queue_activity_lines(
        {
            "status": {"manager_state": "executing_queue", "items_in_queue": 2},
            "running_item": {"item_uid": "u0", "name": "orm"},
        }
    )
    assert "A PLAN IS ALREADY RUNNING" in running[0]
    assert "executing_queue" in running[0]
    assert "Items already queued: 2" in running

    autostart = approval._queue_activity_lines(
        {"status": {"manager_state": "idle", "queue_autostart_enabled": True}, "running_item": None}
    )
    assert "AUTOSTART IS ENABLED" in autostart[0]

    idle = approval._queue_activity_lines(
        {"status": {"manager_state": "idle"}, "running_item": None}
    )
    assert idle[0] == "No plan is currently running (manager state: idle)."


@pytest.mark.unit
def test_queue_activity_lines_report_a_pending_stop(approval):
    """A pending stop changes what a start or a withdrawal actually means."""
    lines = approval._queue_activity_lines(
        {"status": {"manager_state": "paused", "queue_stop_pending": True}, "running_item": None}
    )
    assert any("A stop is PENDING" in line for line in lines)


@pytest.mark.unit
def test_describe_queue_start_lists_the_whole_queue_and_flags_untrusted_plans(
    approval, fake_bridge
):
    """A start drains everything queued, so the prompt names everything queued."""
    fake_bridge(
        {
            "/queue": {
                "status": {"manager_state": "idle", "items_in_queue": 2},
                "items": [
                    {"item_uid": "u1", "name": "orm", "kwargs": {"num_points": 5}},
                    {"item_uid": "u2", "name": "sneaky_plan"},
                ],
                "running_item": None,
            },
            "/plans": [
                {"name": "orm", "provenance": "shipped"},
                {"name": "sneaky_plan", "provenance": "unreviewed"},
            ],
        }
    )

    lines = approval._describe_queue_start({}, {})

    assert any("EVERY pending item" in line for line in lines)
    assert any(line.startswith("  1. orm") and "num_points" in line for line in lines)
    assert any(line.startswith("  2. sneaky_plan") for line in lines)
    assert any("AGENT-AUTHORED" in line and "sneaky_plan" in line for line in lines)


@pytest.mark.unit
def test_describe_queue_start_does_not_flag_a_fully_trusted_queue(approval, fake_bridge):
    """Negative control: no untrusted plan queued means no untrusted warning."""
    fake_bridge(
        {
            "/queue": {
                "status": {"manager_state": "idle"},
                "items": [{"item_uid": "u1", "name": "orm"}],
                "running_item": None,
            },
            "/plans": [{"name": "orm", "provenance": "shipped"}],
        }
    )

    lines = approval._describe_queue_start({}, {})

    assert not any("AGENT-AUTHORED" in line for line in lines)


@pytest.mark.unit
def test_describe_queue_start_caps_the_listed_items(approval, fake_bridge):
    """A long queue is summarised so the warning lines stay on screen."""
    count = approval._MAX_LISTED_QUEUE_ITEMS + 3
    fake_bridge(
        {
            "/queue": {
                "status": {"manager_state": "idle"},
                "items": [{"item_uid": f"u{i}", "name": "orm"} for i in range(count)],
                "running_item": None,
            },
            "/plans": [],
        }
    )

    lines = approval._describe_queue_start({}, {})

    numbered = [line for line in lines if line.startswith("  ") and ". orm" in line]
    assert len(numbered) == approval._MAX_LISTED_QUEUE_ITEMS
    assert any("and 3 more" in line for line in lines)


@pytest.mark.unit
def test_describe_queue_start_says_when_the_queue_cannot_be_read(approval, monkeypatch):
    """Fail-open, but never silently: an unseen queue is itself the warning."""
    monkeypatch.setattr(approval, "_bridge_get_json", lambda *args, **kwargs: None)

    lines = approval._describe_queue_start({}, {})

    assert any("nobody here can see" in line for line in lines)


@pytest.mark.unit
def test_describe_queue_stop_distinguishes_halting_from_un_halting(approval, fake_bridge):
    """The two directions of one tool must never read alike."""
    fake_bridge({"/queue": {"status": {"manager_state": "executing_queue"}, "running_item": None}})

    halt = approval._describe_queue_stop({}, {})
    assert "Requests a stop" in halt[0]
    assert not any("WITHDRAWS" in line for line in halt)

    withdraw = approval._describe_queue_stop({"cancel": True}, {})
    assert "WITHDRAWS A PENDING STOP" in withdraw[0]
    assert "does not halt anything" in withdraw[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "state",
    ["starting_queue", "executing_queue", "executing_task", "paused", "some_future_state"],
)
def test_queue_activity_middle_tier_warns_on_any_non_idle_state(approval, state):
    """The negative test: anything that is not "idle" earns a warning.

    `starting_queue` is the case that motivated this tier — a start is already
    in flight but no item is running yet, so the observed-only tests showed the
    CALM headline on the one line whose job is telling a human that an enqueue
    is really an execution. `some_future_state` is the point of testing for the
    single idle token rather than replicating the manager's active-state list:
    a state this hook has never heard of still warns, because unknown means
    not-idle. A replica would have classified it as safe.
    """
    lines = approval._queue_activity_lines(
        {"status": {"manager_state": state}, "running_item": None}
    )
    assert "THE QUEUE IS NOT IDLE" in lines[0]
    assert state in lines[0]


@pytest.mark.unit
def test_queue_activity_missing_manager_state_is_treated_as_not_idle(approval):
    """An unknown state must never render inside the confident calm sentence."""
    lines = approval._queue_activity_lines({"status": {}, "running_item": None})

    assert "THE QUEUE IS NOT IDLE" in lines[0]
    assert "No plan is currently running" not in lines[0]


@pytest.mark.unit
def test_queue_activity_calm_headline_requires_the_idle_token(approval):
    """Negative control for the tier above: only "idle" earns the calm line.

    Without this, a middle tier that fired unconditionally would pass every
    warning test while making the prompt cry wolf on every enqueue.
    """
    lines = approval._queue_activity_lines(
        {"status": {"manager_state": "idle"}, "running_item": None}
    )
    assert lines[0] == "No plan is currently running (manager state: idle)."


@pytest.mark.unit
def test_queue_activity_precise_headlines_win_over_the_middle_tier(approval):
    """Tier order: a running item or autostart keeps its specific sentence.

    The middle tier is a catch-all, so it must not swallow the two cases that
    can say something exact — "may already be draining" is a downgrade from
    "a plan IS running".
    """
    running = approval._queue_activity_lines(
        {
            "status": {"manager_state": "executing_queue"},
            "running_item": {"item_uid": "u0", "name": "orm"},
        }
    )
    assert "A PLAN IS ALREADY RUNNING" in running[0]

    autostart = approval._queue_activity_lines(
        {
            "status": {"manager_state": "some_future_state", "queue_autostart_enabled": True},
            "running_item": None,
        }
    )
    assert "AUTOSTART IS ENABLED" in autostart[0]


@pytest.mark.unit
def test_stop_describers_state_the_limit_and_name_the_tool_that_has_none(approval, fake_bridge):
    """The stop prompt is read by someone deciding whether a queue-halt is
    enough, at the moment delay costs most.

    It must state the real limit — a plain `queue_stop` does NOT touch the item
    already in motion — and name the tool that does. Pinning the opposite ("no
    tool here can") would be dangerous: a real abort exists, and a prompt that
    denies it costs the operator the fastest way to stop the machine.
    """
    fake_bridge({"/queue": {"status": {"manager_state": "idle"}, "running_item": None}})

    halt = approval._describe_queue_stop({}, {})
    withdraw = approval._describe_queue_stop({"cancel": True}, {})

    assert "does NOT abort the item already in motion" in halt[0]
    assert "stop_run" in halt[0], "the halt prompt must name the tool that DOES abort"
    assert "no tool here can" not in halt[0], (
        "pre-abort wording: it tells an operator no halt exists for a moving plan"
    )
    # The withdrawal is the opposite direction and must not offer an abort as
    # if it were part of the same action.
    assert not any("stop_run" in line for line in withdraw)


@pytest.mark.unit
def test_stop_run_describer_states_what_an_abort_costs(approval, fake_bridge):
    """The abort's own approval prompt. It has to be honest in both
    directions: not a routine stop (the plan's remainder is discarded and the
    machine stays where it stopped), and not something to hesitate over when a
    machine needs stopping — which is why it also names what is running."""
    fake_bridge(
        {
            "/queue": {
                "status": {"manager_state": "executing_queue", "items_in_queue": 2},
                "running_item": {"item_uid": "u1", "name": "orm"},
            }
        }
    )

    lines = approval._describe_stop_run({}, {})

    assert "ABORTS THE PLAN THAT IS RUNNING NOW" in lines[0]
    assert "left wherever the plan moved it" in lines[0]
    assert "data" in lines[0], "what survives an abort matters as much as what does not"
    # The shared activity lines still run, so the approver sees what is running.
    assert any("A PLAN IS ALREADY RUNNING" in line for line in lines)


@pytest.mark.unit
def test_stop_run_is_wired_into_the_describer_table(approval):
    """A describer nobody dispatches to is a prompt that never renders. The
    abort is the most consequential Bluesky approval there is, so its entry is
    pinned rather than assumed."""
    assert approval._QUEUE_DESCRIBERS["stop_run"] is approval._describe_stop_run
