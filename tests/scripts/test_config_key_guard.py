"""Suite wrapper for the config-key resurrection guard.

Two halves, and the second is the load-bearing one.

The first asserts the guard is green on this tree. The second injects one
synthetic violation per failure mode and asserts the guard goes RED. A guard
nobody has seen fail is indistinguishable from a guard wired to nothing — which
is the exact defect the manifest's own back-test exists to catch, and which four
of its first twenty-two orphan regexes turned out to be. Every check the guard
performs therefore has a negative control here.
"""

from __future__ import annotations

import copy
import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "check_config_keys.py"


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("check_config_keys", GUARD_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # import-time required because scripts/ is not a package: check_config_keys.py
    # is loaded by path and registered in sys.modules before exec so @dataclass can
    # resolve annotations through cls.__module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard_module = _load_guard_module()
ConfigKeyGuard = guard_module.ConfigKeyGuard
MANIFEST = guard_module.load_manifest(guard_module.DEFAULT_MANIFEST)

Mutation = Callable[[dict[str, Any]], None]


def make_guard(mutate: Mutation | None = None, root: Path = REPO_ROOT) -> Any:
    """A guard over *root* with an optionally doctored copy of the manifest."""
    manifest = copy.deepcopy(MANIFEST)
    if mutate is not None:
        mutate(manifest)
    return ConfigKeyGuard(root, manifest)


def modes(guard: Any) -> list[str]:
    return [failure.mode for failure in guard.result.failures]


def details(guard: Any) -> str:
    return "\n".join(str(failure) for failure in guard.result.failures)


# ── the tree is clean ────────────────────────────────────────────────────


def test_guard_is_green_on_this_tree():
    guard = make_guard()
    guard.run()
    assert guard.result.ok, "config-key guard failed:\n" + details(guard)


def test_manifest_carries_every_required_section():
    for section in guard_module.REQUIRED_SECTIONS:
        assert section in MANIFEST


def test_render_matrix_still_covers_every_conditional_branch():
    """The matrix must exercise each Jinja branch the audited keys live in.

    If it stops, the union shrinks and every check built on it weakens without
    anything going red — so the discriminating keys are asserted directly.

    The guard enforces both arms: the key is in the union, and the framework
    template is its ONLY author. Mere presence is not enough, because a key a
    preset also spells stays in the union after the matrix stops rendering its
    branch, and the check would then pass on the preset's copy.
    """
    guard = make_guard()
    guard.check_branch_self_test()
    assert guard.result.ok, details(guard)


def test_approval_matcher_parser_extracts_tool_names():
    """An empty governed set means a broken parser, not a permissive config.

    Matchers are single tokens and ``_`` is a word character, so the obvious
    identifier findall silently returns the whole ``mcp__server__tool`` matcher
    and every governed set comes back empty — and equal to nothing.
    """
    guard = make_guard()
    assert guard.governed_tools("ariel-standalone")


def test_line_anchored_regexes_are_matched_with_re_m():
    """Every manifest regex is written against line starts, grep-style.

    Without ``re.M`` a ``^`` anchors at the start of the whole file, so each of
    these patterns silently reports zero matches — and a check that reports zero
    matches is exactly what an orphan-site check reports on success. The whole
    manifest would go green while testing nothing.
    """
    anchored = [
        site["regex"]
        for sites in MANIFEST["orphan_sites"].values()
        for site in sites
        if site["regex"].startswith("^")
    ]
    assert anchored, "expected the manifest to carry line-anchored orphan regexes"

    # Orphan regexes match zero on the branch by construction, so the engine is
    # proved on a line-anchored pattern that DOES match.
    guard = make_guard()
    text = guard.joined("src/osprey/templates")
    probe = r"^project_name:"
    assert re.findall(probe, text) == [], "^ without re.M anchors at the start of the file"
    assert re.findall(probe, text, re.M), "re.M is what makes ^ mean start-of-line"
    assert guard.count_matches(probe, text) > 0
    assert guard.has_match(probe, text)


def test_path_absence_is_measured_by_git_not_the_filesystem(tmp_path):
    """A deleted package leaves ``__pycache__`` behind in any tree that ran it.

    ``Path.exists()`` therefore reports the module as still present on a
    developer worktree while CI, which never imported it, stays green.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    deleted = tmp_path / "src/osprey/services/machine_state/__pycache__"
    deleted.mkdir(parents=True)
    (deleted / "reader.cpython-313.pyc").write_bytes(b"\x00machine_state stale bytecode")

    guard = make_guard(root=tmp_path)
    assert deleted.exists()
    assert guard.tracked_files("src/osprey/services/machine_state") == []


def test_cli_exits_nonzero_when_the_manifest_is_violated(tmp_path):
    doctored = copy.deepcopy(MANIFEST)
    doctored["deleted"].append("system.timezone")
    manifest_path = tmp_path / "config_key_manifest.yml"
    manifest_path.write_text(yaml.safe_dump(doctored))

    assert guard_module.main(["--manifest", str(manifest_path)]) == 1
    assert guard_module.main([]) == 0


# ── one negative control per failure mode ────────────────────────────────


def test_mode_1_unmapped_rendered_key_goes_red():
    def drop_a_mapped_key(manifest):
        del manifest["keys"]["web.theme"]

    guard = make_guard(drop_a_mapped_key)
    guard.check_unmapped_keys()
    assert "unmapped-key" in modes(guard)
    assert "web.theme" in details(guard)


def test_the_union_covers_both_sources():
    """A key from each side, so neither half can quietly stop contributing.

    The framework template and the presets are the two authors of a rendered
    config, and every check here is built on their union. If one dropped out,
    the union would shrink and the whole manifest would go green against half
    the configuration.
    """
    union = make_guard().union()

    assert union["claude_code.provider"] == {guard_module.FRAMEWORK_SOURCE}, (
        "a derived key comes from the framework template alone"
    )
    assert guard_module.FRAMEWORK_SOURCE not in union["approval.enabled"], (
        "an operator-stated key comes from the presets alone"
    )
    assert union["approval.enabled"] == {
        guard_module.preset_id(rel) for rel in MANIFEST["render_contexts"]["presets"]
    }


def test_leaves_below_a_data_map_are_not_demanded():
    """Provider, persona and layout NAMES are the deployment's, not schema.

    Without the skip every persona a facility declares would be an unmapped
    key, so this asserts both halves: the names below a data map pass, and the
    data map's own entry is still required.
    """
    guard = make_guard()
    prefixes = guard.data_map_prefixes()
    assert "api.providers" in prefixes
    assert "modules.web_terminals.personas" in prefixes

    below = [
        path for path in guard.union() if any(path.startswith(f"{prefix}.") for prefix in prefixes)
    ]
    assert below, "the shipped presets are supposed to populate at least one data map"
    assert not (set(below) & set(MANIFEST["keys"])), (
        "a data map's leaves must not be enumerated in the manifest"
    )

    def stop_calling_it_a_data_map(manifest):
        del manifest["keys"]["modules.web_terminals.personas"]["data-map"]

    demanding = make_guard(stop_calling_it_a_data_map)
    demanding.check_unmapped_keys()
    assert "unmapped-key" in modes(demanding)
    assert "modules.web_terminals.personas." in details(demanding)


def test_mode_2_unmatched_evidence_regex_goes_red():
    def break_the_evidence(manifest):
        manifest["keys"]["facility.name"]["evidence"] = "no_reader_spells_this_anywhere"

    guard = make_guard(break_the_evidence)
    guard.check_evidence()
    assert "evidence" in modes(guard)
    assert "facility.name" in details(guard)


def test_mode_3_deleted_key_back_in_the_rendered_union_goes_red():
    def resurrect_in_the_union(manifest):
        manifest["deleted"].append("system.timezone")

    guard = make_guard(resurrect_in_the_union)
    guard.check_deleted()
    assert "resurrection" in modes(guard)
    assert "rendered again" in details(guard)


def test_mode_3_deleted_key_in_a_preset_config_override_goes_red():
    """The preset surface reports the FILE, which the union does not.

    Since the presets became half the union, a preset key trips both arms — so
    what this pins is the second one's message: it names the preset file the
    key came back in, which is what an operator has to open.
    """

    def resurrect_in_a_preset(manifest):
        manifest["deleted"].append("modules.web_terminals.enabled")

    guard = make_guard(resurrect_in_a_preset)
    guard.check_deleted()
    assert "resurrection" in modes(guard)
    assert "preset config override" in details(guard)
    assert "control-assistant.yml" in details(guard)


def test_mode_3_deleted_key_in_the_loader_defaults_goes_red():
    """And a third: a key nothing ships but the loader synthesizes."""

    def resurrect_in_the_loader(manifest):
        manifest["deleted"].append("facility_timezone")

    guard = make_guard(resurrect_in_the_loader)
    guard.check_deleted()
    assert "resurrection" in modes(guard)
    assert "synthesizes" in details(guard)


def test_an_exempt_commented_example_does_not_go_red():
    """Two deleted keys are documented commented on purpose; see the manifest.

    Both left what OSPREY SHIPS while staying live in their readers, so the
    commented line beside that prose documents an override that works. The
    exemption is narrow, which the second half asserts: the same key spelled
    LIVE in a preset is still a resurrection.
    """
    guard = make_guard()
    exempt = MANIFEST["deleted_commented_examples"]
    assert exempt, "the manifest is supposed to record the deliberate exemptions"

    for key in exempt:
        assert key in MANIFEST["deleted"]
        assert guard.commented_preset_overrides(key) == []

    guard.check_deleted()
    guard.check_deleted_commented_examples()
    assert guard.result.ok, details(guard)


def test_an_exemption_for_a_key_that_was_never_deleted_goes_red():
    def exempt_a_live_key(manifest):
        manifest["deleted_commented_examples"]["cli.theme"] = "not deleted at all"

    guard = make_guard(exempt_a_live_key)
    guard.check_deleted_commented_examples()
    assert "resurrection" in modes(guard)
    assert "not on the deleted list" in details(guard)


def test_an_exemption_with_no_reason_goes_red():
    """An exemption is only as good as the reason recorded beside it.

    A blank one reads as "someone waved this through" and is the shape the next
    retired key gets added under, so the empty arm has its own control. Both
    spellings of blank are exercised, because the check strips before testing
    and a whitespace-only reason is the one that would otherwise pass.
    """

    for blank in ("", "   "):

        def blank_the_reason(manifest, blank=blank):
            manifest["deleted_commented_examples"]["ariel.database.uri"] = blank

        guard = make_guard(blank_the_reason)
        guard.check_deleted_commented_examples()
        assert "resurrection" in modes(guard), f"a reason of {blank!r} must not pass"
        assert "gives no reason" in details(guard)


def test_an_exemption_no_preset_uses_any_more_goes_red():
    """An unused exemption is a hole waiting for the next retired key."""

    def exempt_a_key_no_preset_documents(manifest):
        manifest["deleted_commented_examples"]["facility_name"] = "nothing documents this"

    guard = make_guard(exempt_a_key_no_preset_documents)
    guard.check_deleted_commented_examples()
    assert "resurrection" in modes(guard)
    assert "is unused" in details(guard)


def test_commented_preset_override_of_a_deleted_key_is_detected(tmp_path):
    """A retired key put back as a commented example is still put back.

    That is precisely what ``system.facility_name`` was before this branch
    removed it, so the detector has to see comments — without mistaking prose
    that merely names a key for an override.
    """
    presets = tmp_path / guard_module.PRESET_DIR
    presets.mkdir(parents=True)
    (presets / "probe.yml").write_text(
        "config:\n"
        "  # control_system.connector.timeout: 5.0\n"
        "  # Note that control_system.writes_enabled: gates every write.\n"
    )
    guard = make_guard(root=tmp_path)
    assert guard.commented_preset_overrides("control_system.connector.timeout") == ["probe.yml"]
    assert guard.commented_preset_overrides("control_system.writes_enabled") == []


#: A key whose ``evidence`` regex the manifest's own prose spells UNESCAPED, so
#: the pattern matches the ledger as well as the reader. Picking one that cannot
#: self-match (``cli.theme``, whose regex survives only in its escaped form)
#: makes the silent half of the exclusion test vacuous: it asserts the ledger is
#: excluded using a pattern that would have found nothing there either way.
SELF_MATCHING_EVIDENCE_KEY = "services.qmd.path"


def test_the_manifest_is_not_scanned_as_source(monkeypatch):
    """The ledger lives inside the tree it greps, and names every deleted key.

    Both text checks would answer themselves without the exclusion, so both are
    asserted by NEUTRALISING it and watching them break — an implementation that
    skipped the file only for orphan sites would pass on the loud half alone.

    The loud half is the orphan scan: regexes matched their own ``deleted:``
    entries the moment the file moved into the package. The silent half is the
    one that matters — an ``evidence`` regex the manifest itself spells matches
    the entry quoting it, so a key whose reader was deleted would stay green
    forever, and nothing would go red to say so.
    """
    guard = make_guard()
    scanned = {path for path, _text in guard.texts("src/osprey")}
    assert scanned, "the evidence root is supposed to hold files"
    assert not any(Path(path).name == guard_module.MANIFEST_FILENAME for path in scanned), (
        "the manifest must not be scanned as source"
    )

    text = "\n".join(guard.joined(root) for root in guard_module.EVIDENCE_ROOTS)
    a_deleted_key = "channel_finder.explicit_validation_mode"
    assert a_deleted_key in MANIFEST["deleted"]
    assert a_deleted_key not in text, (
        "a deleted key's name reaches the scanned text only from the ledger itself"
    )

    an_evidence_regex = MANIFEST["keys"][SELF_MATCHING_EVIDENCE_KEY]["evidence"]
    assert guard.has_match(an_evidence_regex, text), "the real reader must still match"
    assert not guard.has_match(
        an_evidence_regex, "\n".join(t for _p, t in guard.texts("src/osprey/profiles"))
    ), "the only match under profiles/ would have been the manifest entry quoting it"

    # Neutralise the exclusion. The text cache is keyed on (root, rel_root) and
    # not on the filename rule, so it is cleared on both sides of the mutation.
    guard_module._TEXT_CACHE.clear()
    try:
        monkeypatch.setattr(guard_module, "MANIFEST_FILENAME", "not-the-ledger.yml")
        unguarded = make_guard()
        assert any(
            Path(path).name == "config_key_manifest.yml"
            for path, _text in unguarded.texts("src/osprey")
        ), "neutralising the rule is supposed to let the ledger into the scan"

        unguarded.check_orphan_sites()
        assert "orphan-site" in modes(unguarded), (
            "the loud half: orphan regexes match their own `deleted:` entries"
        )

        profiles_only = "\n".join(t for _p, t in unguarded.texts("src/osprey/profiles"))
        assert unguarded.has_match(an_evidence_regex, profiles_only), (
            "the silent half: the ledger answers the evidence check on its own, so a "
            "key whose reader was deleted would never go red"
        )
    finally:
        guard_module._TEXT_CACHE.clear()


def test_mode_4_orphan_site_regex_matching_again_goes_red():
    def resurrect_a_code_site(manifest):
        manifest["orphan_sites"]["synthetic"] = [
            {"root": "src/osprey/templates", "regex": "osprey", "why": "negative control"}
        ]

    guard = make_guard(resurrect_a_code_site)
    guard.check_orphan_sites()
    assert "orphan-site" in modes(guard)
    assert "synthetic" in details(guard)


def test_mode_4_scans_every_root_of_a_multi_root_site():
    """A site's ``root`` may be a list; resurrection under ANY root goes red.

    The applications site grew a second root when the config loader moved to
    the osprey-connectors workspace member (src/osprey/utils/config.py is now a
    shim its regex can never match) — a site scanning only the shim tree would
    be a green light wired to nothing.
    """

    def resurrect_under_the_second_root(manifest):
        manifest["orphan_sites"]["synthetic"] = [
            {
                "root": [
                    "src/osprey/templates",
                    "packages/osprey-connectors/src/osprey_connectors",
                ],
                "regex": "class MockArchiverConnector",
                "why": "negative control: matches only under the connectors root",
            }
        ]

    guard = make_guard(resurrect_under_the_second_root)
    guard.check_orphan_sites()
    assert "orphan-site" in modes(guard)
    assert "packages/osprey-connectors" in details(guard)


def test_mode_5_all_presets_parity_miss_goes_red():
    def demand_parity_where_none_exists(manifest):
        # web.theme is live in control-assistant, commented in hello-world and
        # absent from the two standalones — deliberately per-preset.
        manifest["keys"]["web.theme"]["all-templates"] = True

    guard = make_guard(demand_parity_where_none_exists)
    guard.check_parity()
    assert "parity" in modes(guard)
    assert "web.theme" in details(guard)


CONTROL_ASSISTANT = "src/osprey/profiles/presets/control-assistant.yml"
ARIEL_STANDALONE = "src/osprey/profiles/presets/ariel-standalone.yml"
HELLO_WORLD = "src/osprey/profiles/presets/hello-world.yml"


def copy_presets(tmp_path: Path) -> Path:
    """A temp root holding just the four presets the markers live in.

    The stanzas moved out of the app templates and into the presets when the
    templates were deleted: the preset is the file ``osprey init`` copies out as
    ``profile.yml``, so it is the document an operator can actually find a panel
    port in.
    """
    for rel in MANIFEST["render_contexts"]["presets"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, target)

    clean = make_guard(root=tmp_path)
    clean.check_panel_port_markers()
    assert clean.result.ok, "the copied presets should start clean:\n" + details(clean)
    return tmp_path


def test_panel_port_stanzas_are_reconciled_with_the_live_registry():
    """The default argument must be the real registry, not a copy of it.

    Every control below injects a registry_keys set, so without this the checks
    could be reconciling against nothing at all in the real run.
    """
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    guard = make_guard()
    markers = guard.panel_port_markers()
    assert set().union(*markers.values()) == set(FRAMEWORK_WEB_SERVERS)
    guard.check_panel_port_markers()
    assert guard.result.ok, details(guard)


def test_mode_6_marker_naming_a_panel_that_does_not_exist_goes_red(tmp_path):
    """A count-based check cannot see this: renaming preserves the count."""
    root = copy_presets(tmp_path)
    victim = root / CONTROL_ASSISTANT
    before = victim.read_text()
    victim.write_text(before.replace("osprey:panel-port okf", "osprey:panel-port okf_panel"))
    assert victim.read_text().count(guard_module.PANEL_PORT_MARKER) == before.count(
        guard_module.PANEL_PORT_MARKER
    ), "the injected fault must leave the marker COUNT untouched"

    guard = make_guard(root=root)
    guard.check_panel_port_markers()
    assert "panel-port" in modes(guard)
    assert "okf_panel" in details(guard)
    assert "not FRAMEWORK_WEB_SERVERS entries" in details(guard)


def test_mode_6_registered_server_with_no_stanza_anywhere_goes_red(tmp_path):
    """A newly registered web server that no preset documents."""
    root = copy_presets(tmp_path)
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    guard = make_guard(root=root)
    guard.check_panel_port_markers(registry_keys=set(FRAMEWORK_WEB_SERVERS) | {"timing_panel"})
    assert "panel-port" in modes(guard)
    assert "timing_panel" in details(guard)


def test_mode_6_one_preset_losing_a_stanza_goes_red(tmp_path):
    """The union claims are blind to this, which is why the per-preset map exists.

    ariel-standalone drops its own ``ariel`` stanza while control-assistant
    still carries one. Every union-based claim therefore still holds — no
    invented name, coverage complete, reference preset complete — so the
    per-preset check is provably the only thing that can fire.
    """
    root = copy_presets(tmp_path)
    victim = root / ARIEL_STANDALONE
    victim.write_text(victim.read_text().replace("  # osprey:panel-port ariel\n", ""))

    guard = make_guard(root=root)
    markers = guard.panel_port_markers()
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    registry = set(FRAMEWORK_WEB_SERVERS)
    assert markers[ARIEL_STANDALONE] == {"artifact"}
    assert set().union(*markers.values()) == registry, (
        "coverage must stay complete, or the union claim would fire instead"
    )
    assert any(names >= registry for names in markers.values()), (
        "the reference preset must stay complete, or claim (c) would fire instead"
    )

    guard.check_panel_port_markers()
    assert "panel-port" in modes(guard)
    assert "no longer documents ['ariel']" in details(guard)


def test_mode_6_full_set_documented_nowhere_goes_red(tmp_path):
    """Coverage can hold while no single preset shows an operator all of them.

    Moving one stanza out of the reference preset into a minimal one keeps the
    union complete, so the coverage claim cannot be what fires. The per-preset
    map also objects to both edits, which is correct — each preset really has
    drifted — so this asserts the third claim's own message specifically.
    """
    root = copy_presets(tmp_path)
    reference = root / CONTROL_ASSISTANT
    reference.write_text(
        reference.read_text().replace("# osprey:panel-port lattice_dashboard", "#")
    )
    minimal = root / HELLO_WORLD
    minimal.write_text(minimal.read_text() + "\n# osprey:panel-port lattice_dashboard\n")

    guard = make_guard(root=root)
    markers = guard.panel_port_markers()
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    assert set().union(*markers.values()) == set(FRAMEWORK_WEB_SERVERS), (
        "the union must stay complete, or this would fire on coverage instead"
    )
    guard.check_panel_port_markers()
    assert "panel-port" in modes(guard)
    assert "no single preset documents the full panel-port set" in details(guard)


def test_kept_reader_names_are_present_in_src_but_never_grepped():
    """Deleted keys with deliberate surviving readers must not turn the guard red.

    ``resolve_facility_name`` still honours the retired ``facility_name``
    spelling, and ``generate_tree_preview`` is a kept utility.
    Those names are therefore in ``src/`` BY DESIGN, and a guard that grepped
    deleted key names across the tree would go permanently red against intended
    code — which is why the deleted list is enforced against the rendered union,
    the presets and the loader, but never against source text.

    The first assertion is what makes this discriminating rather than
    decorative: it proves the trap is ARMED, i.e. a naive deleted-name grep
    really would hit every one of these. Without it the test would still pass on
    an implementation that had quietly stopped looking at anything at all.
    """
    guard = make_guard()
    src = "\n".join(guard.joined(root) for root in guard_module.EVIDENCE_ROOTS)

    covered = {name for entry in MANIFEST["kept_readers"].values() for name in entry["covers"]}
    assert covered, "the manifest is supposed to record deliberate surviving readers"
    leaves = {name.rsplit(".", 1)[-1] for name in covered}

    unarmed = sorted(leaf for leaf in leaves if leaf not in src)
    assert not unarmed, (
        f"these kept-reader names are absent from {' and '.join(guard_module.EVIDENCE_ROOTS)}, so this "
        f"test could not detect a deleted-name grep being added: {unarmed}"
    )

    # The two checks that consume the `deleted` list must stay silent anyway.
    guard.check_deleted()
    guard.check_orphan_sites()
    assert guard.result.ok, "the guard fired on a deliberate surviving reader:\n" + details(guard)


# ── negative controls for the manifest's other sections ──────────────────


def test_phantom_manifest_key_goes_red():
    def add_a_key_nothing_renders(manifest):
        manifest["keys"]["invented.key"] = {"evidence": "osprey"}

    guard = make_guard(add_a_key_nothing_renders)
    guard.check_phantom_keys()
    assert "phantom-key" in modes(guard)


def test_a_rendered_key_marked_unrendered_goes_red():
    """``rendered: false`` is read in BOTH directions, or it is a one-way escape.

    Skipped only when the key is absent, the flag rots silently in exactly the
    direction the phantom check exists to catch: a key that starts being shipped
    keeps a marking saying nothing ships it, and the prose beside it goes on
    describing a commented example while a preset writes the key live. Four
    entries had drifted that way — ``facility.prefix`` and the three
    virtual-accelerator ``limits_checking`` paths, all of them live in
    ``control-assistant``.

    The failure has to name the source, because the fix is to say what ships it.
    """

    def mark_a_shipped_key_unrendered(manifest):
        manifest["keys"]["web.theme"]["rendered"] = False

    guard = make_guard(mark_a_shipped_key_unrendered)
    assert "web.theme" in guard.union(), "the control needs a key something really renders"

    guard.check_phantom_keys()
    assert "phantom-key" in modes(guard)
    assert "web.theme" in details(guard)
    assert "control-assistant" in details(guard), "the failure must name what renders it"


def test_manifest_defaults_column_is_complete():
    """The whole default column, checked against the live manifest.

    Deliberately not folded into ``test_guard_is_green_on_this_tree``: that one
    runs the full guard, which renders the framework template and resolves
    every preset, and the column is a property of the MANIFEST alone. Keeping
    it separate is
    what lets the column stay pinned while the render side is being repointed —
    a rendering failure must not be able to take this assertion down with it,
    because a column nobody checks is a column that rots.
    """
    guard = make_guard()
    guard.check_defaults()
    assert guard.result.ok, details(guard)


def test_manifest_self_consistency_holds_without_rendering():
    """Every manifest-only check, run together against the live manifest.

    These four ask nothing of the render: they read the manifest and the
    source tree. Running them as one test says plainly which properties survive
    independently of the render matrix.
    """
    guard = make_guard()
    guard.check_defaults()
    guard.check_covered_by_chains()
    guard.check_evidence()
    guard.check_evidence_vacuity()
    assert guard.result.ok, details(guard)


def test_missing_default_goes_red():
    """A key with no `default:` is a hole, not a key with no default."""

    def drop_the_default(manifest):
        del manifest["keys"]["cli.theme"]["default"]

    guard = make_guard(drop_the_default)
    guard.check_defaults()
    assert "default" in modes(guard)
    assert "cli.theme" in details(guard)


def test_required_on_a_non_posture_key_goes_red():
    """`required` documents a refusal, so it may not be spelled where none happens."""

    def demand_a_key_the_build_does_not(manifest):
        manifest["keys"]["cli.theme"]["default"] = "required"

    guard = make_guard(demand_a_key_the_build_does_not)
    guard.check_defaults()
    assert "default" in modes(guard)
    assert "cli.theme" in details(guard)


def test_posture_floor_key_losing_required_goes_red():
    """The other direction: a floor key must not acquire an invented fallback."""

    def invent_a_fallback(manifest):
        manifest["keys"]["hooks.debug"]["default"] = False

    guard = make_guard(invent_a_fallback)
    guard.check_defaults()
    assert "default" in modes(guard)
    assert "hooks.debug" in details(guard)


def test_unexplained_derived_default_goes_red():
    """`derived` with no note is the escape hatch the note rule exists to close."""

    def strip_the_note(manifest):
        manifest["keys"]["services.qmd.port"].pop("default_note", None)

    guard = make_guard(strip_the_note)
    guard.check_defaults()
    assert "default" in modes(guard)
    assert "services.qmd.port" in details(guard)


def test_unexplained_no_fallback_default_goes_red():
    """The note rule binds both sentinels, not just `derived`."""

    def strip_the_note(manifest):
        manifest["keys"]["services.qmd.path"].pop("default_note", None)

    guard = make_guard(strip_the_note)
    guard.check_defaults()
    assert "default" in modes(guard)
    assert "services.qmd.path" in details(guard)


@pytest.mark.parametrize("misspelling", ["no_fallback", "Derived", "N/A", " required "])
def test_near_miss_sentinel_goes_red(misspelling):
    """A sentinel spelled wrong is read as a literal by every other check.

    That is the silent failure: `no_fallback` skips the note rule and would be
    rendered by `osprey config --defaults` as if it were a real default value.
    """

    def mistype_the_sentinel(manifest):
        manifest["keys"]["cli.theme"]["default"] = misspelling

    guard = make_guard(mistype_the_sentinel)
    guard.check_defaults()
    assert "default" in modes(guard)
    assert "cli.theme" in details(guard)


def test_note_on_a_required_default_stays_green():
    """`required` says no fallback exists; the note says which values are legal.

    A key whose admissible values are a closed set has nowhere else to name
    them for a reader holding only `osprey config --defaults`, so the note is
    the answer rather than an excuse for a missing one.
    """

    def name_the_legal_values(manifest):
        manifest["keys"]["hooks.debug"]["default_note"] = "true or false, nothing else"

    guard = make_guard(name_the_legal_values)
    guard.check_defaults()
    assert modes(guard) == []


def test_note_on_a_literal_default_goes_red():
    """A literal is the whole answer; a note beside one means the wrong sentinel."""

    def explain_a_literal(manifest):
        manifest["keys"]["cli.theme"]["default_note"] = "something about the theme"

    guard = make_guard(explain_a_literal)
    guard.check_defaults()
    assert "default" in modes(guard)
    assert "cli.theme" in details(guard)


def test_dangling_covered_by_chain_goes_red():
    def point_at_a_missing_parent(manifest):
        manifest["keys"]["approval.tools.execute"] = {"covered-by": "not.a.manifest.key"}

    guard = make_guard(point_at_a_missing_parent)
    guard.check_covered_by_chains()
    assert "evidence" in modes(guard)
    assert "not.a.manifest.key" in details(guard)


def test_vacuous_bare_word_evidence_goes_red():
    """The threshold that retired 62 first-draft regexes has to still bite."""

    def swap_in_a_bare_word(manifest):
        manifest["keys"]["facility.name"] = {"evidence": "port"}

    guard = make_guard(swap_in_a_bare_word)
    guard.check_evidence_vacuity()
    assert "evidence" in modes(guard)
    assert "vacuous" in details(guard)


def test_wrong_governed_set_claim_goes_red():
    def overstate_the_claim(manifest):
        manifest["governed_sets"]["claims"]["channel-finder-standalone"]["tools"] = [
            "setup_patch",
            "channel_write",
        ]

    guard = make_guard(overstate_the_claim)
    guard.check_governed_sets()
    assert "governed-set" in modes(guard)


def test_missing_keeps_asset_goes_red():
    def point_at_a_deleted_asset(manifest):
        manifest["keeps"] = [{"path": "src/osprey/templates/apps/control_assistant/gone.json.j2"}]

    guard = make_guard(point_at_a_deleted_asset)
    guard.check_keeps()
    assert "keeps" in modes(guard)


def test_resurrected_absent_path_goes_red():
    def point_at_a_path_that_still_exists(manifest):
        manifest["absent_paths"] = [{"path": "src/osprey/utils", "assert": "absent"}]

    guard = make_guard(point_at_a_path_that_still_exists)
    guard.check_absent_paths()
    assert "absent-path" in modes(guard)


def test_uncovered_conditional_branch_goes_red():
    def name_a_key_the_matrix_never_renders(manifest):
        manifest["render_contexts"]["self_test_keys"]["enable_hierarchical"] = "never.rendered"

    guard = make_guard(name_a_key_the_matrix_never_renders)
    guard.check_branch_self_test()
    assert "branch-self-test" in modes(guard)


def test_a_self_test_key_a_preset_also_spells_goes_red():
    """A shared key cannot detect a shrunk matrix, so the check refuses one.

    It would stay in the union on the preset's copy after the framework stopped
    rendering its branch, and the self-test would go on passing — silence in the
    one place the manifest built a tripwire. The mutation walks one path up from
    the real ``ariel_server_on`` key to its parent block, which the framework
    template and two presets all render, so it is the realistic way this rots:
    an edit that keeps the key plausible while dropping what made it decisive.
    """

    def point_it_at_a_key_the_presets_share(manifest):
        manifest["render_contexts"]["self_test_keys"]["ariel_server_on"] = (
            "ariel.enhancement_modules.semantic_processor.model"
        )

    guard = make_guard(point_it_at_a_key_the_presets_share)
    shared = guard.union()["ariel.enhancement_modules.semantic_processor.model"]
    assert guard_module.FRAMEWORK_SOURCE in shared and len(shared) > 1, (
        "the control needs a key the framework and a preset both render"
    )

    guard.check_branch_self_test()
    assert "branch-self-test" in modes(guard)
    assert "must come from the framework template" in details(guard)


def test_incomplete_provider_tier_map_goes_red():
    def demand_a_tier_no_provider_maps(manifest):
        manifest["keys"]["api.providers"]["key-shape"]["models-tiers"] = ["fable"]

    guard = make_guard(demand_a_tier_no_provider_maps)
    guard.check_provider_shape()
    assert "provider-shape" in modes(guard)


# ── back-test: developer-time, behind a flag ─────────────────────────────


def _baseline_commit() -> str:
    return MANIFEST["scan_rules"]["back_test_baseline"]["commit"]


def _baseline_reachable() -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{_baseline_commit()}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


needs_baseline = pytest.mark.skipif(
    not _baseline_reachable(),
    reason="merge-base commit unavailable (shallow clone); back-test is developer-time",
)


@pytest.mark.slow
@needs_baseline
def test_back_test_confirms_every_orphan_regex_is_falsifiable():
    guard = make_guard()
    guard.back_test(_baseline_commit())
    assert guard.result.ok, details(guard)


@pytest.mark.slow
@needs_baseline
def test_back_test_catches_a_regex_that_cannot_fail():
    """Zero matches on both sides is a green light wired to nothing."""

    def wire_it_to_nothing(manifest):
        manifest["orphan_sites"]["facility_name"] = [
            {"root": "src/osprey/templates", "regex": "zzz_never_existed_anywhere"}
        ]

    guard = make_guard(wire_it_to_nothing)
    guard.back_test(_baseline_commit())
    assert "back-test" in modes(guard)
    assert "cannot fail" in details(guard)


@pytest.mark.slow
@needs_baseline
def test_back_test_sums_hits_across_roots_newer_than_the_baseline():
    """A root absent at the baseline reads as zero base hits, not an error —
    and summing across roots must not mask a regex that cannot fail anywhere.

    The falsifiability of the real multi-root site (applications) is proved by
    ``test_back_test_confirms_every_orphan_regex_is_falsifiable`` over the
    unmodified manifest.
    """

    def cannot_fail_under_either_root(manifest):
        manifest["orphan_sites"]["synthetic"] = [
            {
                "root": [
                    "src/osprey/templates",
                    "packages/osprey-connectors/src/osprey_connectors",
                ],
                "regex": "zzz_never_existed_anywhere",
                "why": "negative control",
            }
        ]

    guard = make_guard(cannot_fail_under_either_root)
    guard.back_test(_baseline_commit())
    assert "back-test" in modes(guard)
    assert "cannot fail" in details(guard)
