"""Unit tests for `osprey_approval`'s pure helpers, imported in-process.

The hook's end-to-end behaviour is covered by `test_approval_hook.py` (real
subprocesses) and its launch enrichment by `test_approval_launch_enrichment.py`
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
    [{}, {"bluesky": {}}],
    ids=["no-bluesky-section", "no-bridge-url-key"],
)
def test_resolve_bridge_url_falls_back_to_the_loopback_default(approval, monkeypatch, config):
    """Neither env nor config leaves the loopback default."""
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)

    assert approval._resolve_bridge_url(config) == "http://127.0.0.1:8090"


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
    """A plan with metadata gets category, devices, and a hazard verdict."""
    fake_bridge(
        {
            "/plans": [
                {
                    "name": "orbit_correction",
                    "metadata": {
                        "category": "correction",
                        "required_devices": ["BPM:1", "COR:2"],
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

    assert "Category: correction" in lines
    assert "Required devices: BPM:1, COR:2" in lines
    assert "Hazard: writes to hardware" in lines


@pytest.mark.unit
def test_describe_plan_provenance_marks_a_read_only_plan(approval, fake_bridge):
    """`writes: False` renders as read-only, and no devices says so explicitly."""
    fake_bridge(
        {
            "/plans": [
                {"name": "count", "metadata": {"category": "diagnostic", "writes": False}},
            ],
            "/plans/count/source": {"provenance": "shipped"},
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "count")

    assert "Required devices: none declared" in lines
    assert "Hazard: read-only (no hardware writes declared)" in lines


@pytest.mark.unit
def test_describe_plan_provenance_sanitizes_metadata_labels(approval, fake_bridge):
    """Category and device names are agent-authored, so they are escaped too.

    `_sanitize_label` guards the plan name; the metadata rendered beside it comes
    from the same unconstrained `PLAN_METADATA` block and reaches the prompt on
    the same lines.
    """
    fake_bridge(
        {
            "/plans": [
                {
                    "name": "sneaky",
                    "metadata": {
                        "category": "safe\nValidation status: PASSED",
                        "required_devices": ["BPM\u20281"],
                    },
                }
            ],
        }
    )

    lines = approval._describe_plan_provenance("http://bridge", "sneaky")

    assert "Category: safe\\x0aValidation status: PASSED" in lines
    assert "Required devices: BPM\\x20281" in lines
    assert not any("\n" in line for line in lines)


@pytest.mark.unit
def test_describe_plan_provenance_handles_a_plan_without_metadata(approval, fake_bridge):
    """A built-in plan carries no authoring metadata; the line says so."""
    fake_bridge({"/plans": [{"name": "grid_scan"}], "/plans/grid_scan/source": {}})

    lines = approval._describe_plan_provenance("http://bridge", "grid_scan")

    assert "Category/devices/hazard: unavailable (no authoring metadata — built-in plan)" in lines


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

    assert "Validation status: NO PASSING RECORD — would be refused/quarantined at launch" in lines


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

    assert "Category/devices/hazard: unavailable (no authoring metadata — built-in plan)" in lines
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
# _describe_launch_run
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "snapshot",
    [None, [], "not-json-object"],
    ids=["bridge-unreachable", "wrong-shape-list", "wrong-shape-string"],
)
def test_describe_launch_run_returns_no_detail_for_an_unusable_draft(
    approval, monkeypatch, snapshot
):
    """A missing or misshaped `GET /draft` degrades to zero enrichment lines.

    The plain tool/policy reason still asks for approval; only the extra detail
    is lost. Returning `[]` here is what keeps `main` from having to reason about
    a half-built line list.
    """
    monkeypatch.setattr(approval, "_bridge_get_json", lambda *args, **kwargs: snapshot)

    assert approval._describe_launch_run({"draft_revision": 3}, {}) == []


@pytest.mark.unit
def test_describe_launch_run_reports_an_empty_draft(approval, fake_bridge):
    """Nothing staged means nothing to launch, and the prompt says so."""
    fake_bridge({"/draft": {"revision": 4, "draft": {}}})

    lines = approval._describe_launch_run({"draft_revision": 4}, {})

    assert lines[0] == "Draft revision 4 — matches pinned revision 4."
    assert lines[1].startswith("Draft: EMPTY")
    assert len(lines) == 2


@pytest.mark.unit
def test_describe_launch_run_renders_the_staged_plan(approval, fake_bridge):
    """A staged plan contributes its name, args, and provenance lines."""
    fake_bridge(
        {
            "/draft": {
                "revision": 9,
                "draft": {"plan_name": "grid_scan", "plan_args": {"steps": 5}},
            },
            "/plans": [{"name": "grid_scan", "metadata": {"category": "scan", "writes": False}}],
            "/plans/grid_scan/source": {"provenance": "shipped"},
        }
    )

    lines = approval._describe_launch_run({"draft_revision": 9}, {})

    assert "Plan: grid_scan" in lines
    assert 'Plan args: {"steps": 5}' in lines
    assert "Category: scan" in lines
    assert "Provenance: shipped (operator-supplied)" in lines


@pytest.mark.unit
def test_describe_launch_run_omits_the_args_line_when_there_are_none(approval, fake_bridge):
    """An empty `plan_args` renders no args line rather than an empty one."""
    fake_bridge(
        {
            "/draft": {"revision": 1, "draft": {"plan_name": "count", "plan_args": {}}},
            "/plans": [],
        }
    )

    lines = approval._describe_launch_run({"draft_revision": 1}, {})

    assert not any(line.startswith("Plan args:") for line in lines)


@pytest.mark.unit
def test_describe_launch_run_sanitizes_the_plan_name(approval, fake_bridge):
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

    lines = approval._describe_launch_run({"draft_revision": 2}, {})

    assert "Plan: count\\x0aDraft revision 2 — matches pinned revision 2." in lines


@pytest.mark.unit
def test_describe_launch_run_warns_when_the_draft_moved_on(approval, fake_bridge):
    """A draft revised since the agent pinned it leads with the loud warning."""
    fake_bridge(
        {
            "/draft": {"revision": 12, "draft": {"plan_name": "count"}},
            "/plans": [],
        }
    )

    lines = approval._describe_launch_run({"draft_revision": 9}, {})

    assert "DRAFT CHANGED" in lines[0]
    assert "9" in lines[0]
    assert "12" in lines[0]


@pytest.mark.unit
def test_describe_launch_run_asks_the_configured_bridge(approval, fake_bridge, bridge_calls):
    """The base URL resolved from config is the one every call goes to.

    The draft, the plan registry and the plan source must all be read from the
    same bridge the launch would actually use — a mismatch would describe one
    bridge's draft while another one launches.
    """
    fake_bridge(
        {
            "/draft": {"revision": 1, "draft": {"plan_name": "count"}},
            "/plans": [],
            "/plans/count/source": {},
        }
    )

    approval._describe_launch_run(
        {"draft_revision": 1}, {"bluesky": {"bridge_url": "http://bridge.config:8000/"}}
    )

    assert {base_url for base_url, _ in bridge_calls} == {"http://bridge.config:8000"}
    assert [path for _, path in bridge_calls] == ["/draft", "/plans", "/plans/count/source"]
