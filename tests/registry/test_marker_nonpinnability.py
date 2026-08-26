"""The audit-critical markers are not pinnable from a server spec.

An audit record answers three questions no facility may answer for itself:
*who* acted (the whole identity ladder — ``OSPREY_TERMINAL_USER`` and
``OSPREY_AUDIT_IDENTITY`` — plus ``OSPREY_AUDIT_WRITER``), *which posture* the
process is under (``OSPREY_EXECUTION_MODE``, the value the middleware clamp
and the executor gate both read) and *where the posture claim came from*
(``OSPREY_POSTURE_SOURCE`` and the exported session key
``OSPREY_POSTURE_SESSION``). Each is assigned by a real site outside the MCP
spec path — compose ``environment:`` for the identities, the three
web-terminal spawn sites for the posture and its provenance pair, the root
maintenance heredoc's per-command prefix for the writer. A spec's ``env:`` is
therefore never a legitimate source for any of them, and a spec that could set
one could file its records under another service's name, present a writes
posture inside a sandboxed session, or claim a posture provenance it was never
granted.

The set is spelled off its sources, not off a hand-picked list: two of the
containment tests below exist because the first cut listed four markers and
missed both the ladder's WINNING rung (a spec pinning ``OSPREY_TERMINAL_USER``
misrouted its whole ledger into an unmounted subdirectory) and the posture
value itself (a clone pinning ``OSPREY_EXECUTION_MODE=write_access`` ran write
tools through a sandboxed session and filed ``posture=writes`` for it).

The anti-spoof is post-merge REMOVAL at ``_server_to_dict``'s env loop — the
one site every launch path funnels through. That location is the whole point:
spec env WINS the merge everywhere upstream (an extends clone merges it over
the template, a custom server copies it verbatim), so a clone-path
``merged_env`` pop would leave custom servers wide open. These tests pin the
removal on all three launch paths, the operator-facing lint that says so, and
the constants' agreement with the modules that own the spellings.

The mirror property — ``OSPREY_MCP_TOOL_PREFIX`` is ASSIGNED at the same site,
not removed — lives in ``test_tool_prefix.py``; it is re-asserted here only
where the two settlements sit close enough to be swapped by accident.
"""

from __future__ import annotations

import pytest

from osprey.audit import posture
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV as SPAWN_POSTURE_SESSION_ENV,
)
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SOURCE_ENV as SPAWN_POSTURE_SOURCE_ENV,
)
from osprey.registry.mcp import (
    _FRAMEWORK_OWNED_SPEC_ENV,
    AUDIT_IDENTITY_ENV,
    AUDIT_WRITER_ENV,
    FRAMEWORK_SERVERS,
    NON_PINNABLE_AUDIT_MARKERS,
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    TOOL_PREFIX_ENV,
    resolve_servers,
)
from osprey.utils.identity import AUDIT_IDENTITY_ENV as IDENTITY_MODULE_AUDIT_IDENTITY_ENV
from osprey.utils.identity import IDENTITY_ENV_LADDER, TERMINAL_USER_ENV
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR


def _base_ctx(**overrides):
    """Minimal render context, matching ``tests/registry/test_mcp.py``."""
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "agent_data_root": DEFAULT_AGENT_DATA_BASE_DIR,
    }
    ctx.update(overrides)
    return ctx


def _resolve_one(cfg, name, ctx=None):
    servers = resolve_servers(cfg, ctx or _base_ctx())
    matches = [s for s in servers if s["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} server, got {len(matches)}"
    return matches[0]


#: A spoof value distinct from anything the framework would ever assign, so a
#: survivor is unmistakably the spec's and not a coincidence.
SPOOF = "somebody-else"


def _spec_for_path(path: str, marker: str) -> dict:
    """A spec pinning ``marker``, shaped for one of the three launch paths."""
    if path == "framework-override":
        return {"enabled": True, "env": {marker: SPOOF, "KEEP_ME": "kept"}}
    if path == "extends-clone":
        return {"extends": "phoebus", "env": {marker: SPOOF, "KEEP_ME": "kept"}}
    if path == "custom-spec":
        return {
            "command": "node",
            "args": ["site.js"],
            "env": {marker: SPOOF, "KEEP_ME": "kept"},
        }
    raise AssertionError(f"unknown launch path {path!r}")


#: The server name each launch path is exercised under. ``controls`` is an
#: existing framework server; the other two names the registry has never heard
#: of, which is what makes them the clone and custom paths.
_PATH_SERVER = {
    "framework-override": "controls",
    "extends-clone": "phoebus2",
    "custom-spec": "site-tools",
}


class TestMarkerSpellings:
    """The names are the wire contract — pinned against their owners."""

    def test_the_documented_spellings(self):
        assert NON_PINNABLE_AUDIT_MARKERS == (
            "OSPREY_TERMINAL_USER",
            "OSPREY_AUDIT_IDENTITY",
            "OSPREY_EXECUTION_MODE",
            "OSPREY_AUDIT_WRITER",
            "OSPREY_POSTURE_SOURCE",
            "OSPREY_POSTURE_SESSION",
        )

    def test_identity_marker_is_the_identity_ladder_rung(self):
        """Not a copy: the registry imports the stdlib-only leaf module's name."""
        assert AUDIT_IDENTITY_ENV is IDENTITY_MODULE_AUDIT_IDENTITY_ENV

    def test_every_identity_ladder_rung_is_non_pinnable(self):
        """The whole ladder, not one rung. ``acting_identity`` consults the rungs
        in order and the FIRST usable one wins, so stripping only the lower rung
        leaves a spec free to pin the winner and route every record its server
        emits under a name nothing mounts. A rung added to the ladder is
        stripped here without this tuple being touched — and if it ever is not,
        this is the test that says so.
        """
        assert set(IDENTITY_ENV_LADDER) <= set(NON_PINNABLE_AUDIT_MARKERS)
        assert TERMINAL_USER_ENV in NON_PINNABLE_AUDIT_MARKERS
        assert IDENTITY_ENV_LADDER[0] == TERMINAL_USER_ENV, "the winning rung must be first"

    def test_the_posture_value_is_non_pinnable(self):
        """The VALUE the clamp reads, not only its provenance. Stripping the
        source/session pair while leaving ``OSPREY_EXECUTION_MODE`` pinnable let
        a clone present ``writes`` inside a sandboxed session: the middleware
        clamp passed the write tool through, and the ledger filed
        ``posture=writes source=live`` for a session that was sandboxed.
        Pinned by identity against the posture module, which owns the name.
        """
        assert posture.POSTURE_ENV_VAR == "OSPREY_EXECUTION_MODE"
        assert posture.POSTURE_ENV_VAR in NON_PINNABLE_AUDIT_MARKERS

    @pytest.mark.parametrize(
        ("registry_name", "spawn_name"),
        [
            (POSTURE_SOURCE_ENV, SPAWN_POSTURE_SOURCE_ENV),
            (POSTURE_SESSION_ENV, SPAWN_POSTURE_SESSION_ENV),
        ],
        ids=["posture-source", "posture-session"],
    )
    def test_posture_spellings_match_the_spawn_sites(self, registry_name, spawn_name):
        """Copied, not imported (an interfaces import from the registry risks a
        cycle) — so the copy is pinned against the original here. A rename at
        the spawn site fails this test rather than silently un-stripping the
        marker it no longer matches.
        """
        assert registry_name == spawn_name

    def test_writer_marker_is_reserved_before_its_assigner_exists(self):
        """The writer marker is refused by the spec path already; the site that
        legitimately assigns it (the root maintenance heredoc's per-command
        prefix) lands later. Reserving early is the point: a marker that only
        becomes non-pinnable once something assigns it is spoofable in the gap.
        """
        assert AUDIT_WRITER_ENV == "OSPREY_AUDIT_WRITER"
        assert AUDIT_WRITER_ENV in NON_PINNABLE_AUDIT_MARKERS


class TestRemovedOnEveryLaunchPath:
    """A pinned marker never reaches the rendered env — on any path."""

    @pytest.mark.parametrize("marker", NON_PINNABLE_AUDIT_MARKERS)
    @pytest.mark.parametrize("path", ["framework-override", "extends-clone", "custom-spec"])
    def test_pinned_marker_is_absent_from_the_resolved_env(self, marker, path):
        name = _PATH_SERVER[path]
        srv = _resolve_one({"servers": {name: _spec_for_path(path, marker)}}, name)
        assert marker not in srv["env"], (
            f"{marker} survived the {path} path with value {srv['env'].get(marker)!r}"
        )
        if path != "framework-override":
            # The override contract is intact — only framework-owned keys move.
            # (A spec against a framework NAME contributes no env at all; see
            # test_framework_override_spec_env_never_merges below.)
            assert srv["env"]["KEEP_ME"] == "kept"

    @pytest.mark.parametrize("path", ["framework-override", "extends-clone", "custom-spec"])
    def test_all_markers_pinned_at_once_are_all_removed(self, path):
        """A spec pinning the whole set loses the whole set, not just the first."""
        name = _PATH_SERVER[path]
        spec = _spec_for_path(path, NON_PINNABLE_AUDIT_MARKERS[0])
        spec["env"].update(dict.fromkeys(NON_PINNABLE_AUDIT_MARKERS, SPOOF))
        srv = _resolve_one({"servers": {name: spec}}, name)
        assert not [m for m in NON_PINNABLE_AUDIT_MARKERS if m in srv["env"]]

    @pytest.mark.parametrize(
        ("marker", "spoof"),
        [
            ("OSPREY_TERMINAL_USER", "somebody-else"),
            ("OSPREY_EXECUTION_MODE", "write_access"),
        ],
        ids=["winning-identity-rung", "posture-value"],
    )
    @pytest.mark.parametrize("path", ["framework-override", "extends-clone", "custom-spec"])
    def test_the_two_once_missed_markers_are_stripped_by_literal(self, marker, spoof, path):
        """Spelled as LITERALS on purpose. The parametrised tests above draw
        their marker list from the module under test, so a marker missing from
        the tuple is a marker they never exercise — which is exactly how these
        two shipped unstripped the first time. This test names them itself: a
        clone pinning ``OSPREY_TERMINAL_USER`` misrouted its whole ledger, and
        one pinning ``OSPREY_EXECUTION_MODE=write_access`` ran write tools
        through a sandboxed session.
        """
        name = _PATH_SERVER[path]
        spec = _spec_for_path(path, marker)
        spec["env"][marker] = spoof
        srv = _resolve_one({"servers": {name: spec}}, name)
        assert marker not in srv["env"], (
            f"{marker} survived the {path} path with value {srv['env'].get(marker)!r}"
        )

    def test_framework_override_spec_env_never_merges(self):
        """Why that path is belt-and-braces: a spec keyed on a FRAMEWORK name
        contributes only ``enabled`` — its ``env:`` is dropped wholesale, so a
        marker pinned there could not have survived anyway. The removal covers
        it regardless, because "which paths merge spec env" is exactly the kind
        of fact a later refactor changes, and the anti-spoof must not depend on
        it. The lint still warns, which is the operator-visible half.
        """
        spec = _spec_for_path("framework-override", AUDIT_IDENTITY_ENV)
        controls = _resolve_one({"servers": {"controls": spec}}, "controls")
        assert "KEEP_ME" not in controls["env"]
        assert AUDIT_IDENTITY_ENV not in controls["env"]

    @pytest.mark.parametrize("marker", NON_PINNABLE_AUDIT_MARKERS)
    def test_placeholder_pin_is_not_expanded_into_the_env(self, marker):
        """A ``${...}`` pin must not survive as a runtime-expanded value either."""
        spec = {"extends": "phoebus", "env": {marker: "${EVIL:-somebody-else}"}}
        p2 = _resolve_one({"servers": {"phoebus2": spec}}, "phoebus2")
        assert marker not in p2["env"]

    def test_removal_does_not_disturb_the_assigned_marker(self):
        """Removal and assignment share one block — swapping them would show here."""
        spec = _spec_for_path("extends-clone", AUDIT_IDENTITY_ENV)
        spec["env"][TOOL_PREFIX_ENV] = SPOOF
        p2 = _resolve_one({"servers": {"phoebus2": spec}}, "phoebus2")
        assert AUDIT_IDENTITY_ENV not in p2["env"]
        assert p2["env"][TOOL_PREFIX_ENV] == "phoebus2"

    def test_unpinned_servers_never_grow_the_markers(self):
        """No marker is invented for a server that never asked for one: the
        registry removes, it does not assign — the values belong to sites the
        render cannot see.
        """
        for srv in resolve_servers({}, _base_ctx()):
            for marker in NON_PINNABLE_AUDIT_MARKERS:
                assert marker not in srv["env"], f"{srv['name']} grew {marker}"


class TestFrameworkDefinitionsAreClean:
    """The registry's own catalog must not declare the markers either."""

    def test_no_framework_server_declares_an_audit_marker(self):
        offenders = {
            name: [m for m in NON_PINNABLE_AUDIT_MARKERS if m in (sdef.env or {})]
            for name, sdef in FRAMEWORK_SERVERS.items()
        }
        offenders = {name: found for name, found in offenders.items() if found}
        assert not offenders, (
            f"framework server definitions declare audit-critical markers: {offenders}"
        )


class TestPinLint:
    """The operator hears about the pin instead of watching it vanish."""

    @pytest.mark.parametrize("marker", NON_PINNABLE_AUDIT_MARKERS)
    @pytest.mark.parametrize("path", ["framework-override", "extends-clone", "custom-spec"])
    def test_pin_is_flagged_on_every_path(self, caplog, marker, path):
        name = _PATH_SERVER[path]
        with caplog.at_level("WARNING"):
            resolve_servers({"servers": {name: _spec_for_path(path, marker)}}, _base_ctx())
        assert marker in caplog.text
        assert name in caplog.text

    @pytest.mark.parametrize("marker", NON_PINNABLE_AUDIT_MARKERS)
    def test_warning_says_removed_not_assigned(self, caplog, marker):
        """The two marker classes settle differently, and the operator's next
        move differs with them: an assigned marker comes back with the
        framework's value, a removed one is simply gone from the rendered env.
        """
        spec = _spec_for_path("custom-spec", marker)
        with caplog.at_level("WARNING"):
            resolve_servers({"servers": {"site-tools": spec}}, _base_ctx())
        assert "removed after the spec env merge" in caplog.text
        assert "assigned after the spec env merge" not in caplog.text

    def test_tool_prefix_still_says_assigned(self, caplog):
        """The mirror wording, so the variant cannot collapse to one message."""
        spec = _spec_for_path("custom-spec", TOOL_PREFIX_ENV)
        with caplog.at_level("WARNING"):
            resolve_servers({"servers": {"site-tools": spec}}, _base_ctx())
        assert "assigned after the spec env merge" in caplog.text
        assert "removed after the spec env merge" not in caplog.text

    def test_ordinary_spec_is_silent(self, caplog):
        """No false positives: an honest spec triggers none of this."""
        cfg = {
            "servers": {
                "site-tools": {
                    "command": "node",
                    "args": ["site.js"],
                    "env": {"SITE_TOKEN": "x", "OSPREY_SERVER_NAME": "panel"},
                }
            }
        }
        with caplog.at_level("WARNING"):
            resolve_servers(cfg, _base_ctx())
        for marker in NON_PINNABLE_AUDIT_MARKERS:
            assert marker not in caplog.text


class TestDriftSeam:
    """The membership a gate-wiring drift check imports rather than re-encodes."""

    def test_every_audit_marker_is_framework_owned(self):
        """Both halves of the settlement are driven by one tuple: the lint reads
        ``_FRAMEWORK_OWNED_SPEC_ENV``, the removal reads
        ``NON_PINNABLE_AUDIT_MARKERS``. A marker added to the removal list but
        not to the owned list would strip silently, with no warning — this
        assertion is what makes that a test failure instead.
        """
        assert set(NON_PINNABLE_AUDIT_MARKERS) <= set(_FRAMEWORK_OWNED_SPEC_ENV)

    def test_the_assigned_marker_is_owned_too_and_is_not_removed(self):
        assert TOOL_PREFIX_ENV in _FRAMEWORK_OWNED_SPEC_ENV
        assert TOOL_PREFIX_ENV not in NON_PINNABLE_AUDIT_MARKERS

    def test_owned_set_is_exactly_the_two_classes(self):
        """A fifth framework-owned key must join one class or the other — the
        lint's wording is keyed on that membership, so an unclassified key
        would be reported as "assigned" while behaving however it behaves.
        """
        assert set(_FRAMEWORK_OWNED_SPEC_ENV) == {TOOL_PREFIX_ENV, *NON_PINNABLE_AUDIT_MARKERS}
