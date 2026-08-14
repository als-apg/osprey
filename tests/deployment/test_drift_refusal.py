"""The drift verdict the lifecycle start verbs refuse on.

``osprey build`` renders a deployment repo's ``build/`` and stamps the profile
fingerprint into the build manifest; ``osprey up``/``restart`` recompute that
fingerprint and refuse when it has moved. This file pins the properties that
decide whether the refusal is trustworthy:

- a comment-only edit to ``profile.yml`` is NOT drift — a check that fires on
  cosmetics trains operators to reach for ``--as-built`` reflexively, which
  costs the refusal all of its value;
- an edit anywhere in the ``data/`` tree IS drift — the profile YAML is
  untouched in exactly the case a facility changes its channel database, and a
  fingerprint blind to that would let ``up`` deploy last week's data while
  reporting itself in sync (SC-3);
- "cannot compare" is a refusal, not silence: an unparseable profile or a build
  with no fingerprint is ``UNRESOLVABLE``, with ``--as-built`` as the escape;
- version skew is a warning in every state, never a refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.deployment import staleness
from osprey.deployment.staleness import DriftState


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


PROFILE_YAML = """\
# The facility's build profile.
name: Exemplar
app_template: hello_world
data: data
"""


def _write_manifest(repo: Path, **creation: object) -> Path:
    """Stamp a build manifest carrying the CURRENT profile fingerprint."""
    from osprey.cli.templates.manifest import get_framework_release_version

    fingerprint = staleness.profile_fingerprint(repo / "profile.yml")
    assert fingerprint is not None, "fixture profile must resolve"
    block: dict[str, object] = {
        "osprey_version": get_framework_release_version(),
        "template": "hello-world",
        "preset_hash": fingerprint.profile_hash,
        staleness.KEY_DIGESTS_MANIFEST_KEY: fingerprint.key_digests,
    }
    block.update(creation)
    manifest = staleness.build_manifest_path(repo)
    _write(
        manifest,
        json.dumps({"schema_version": "1.2.0", "creation": block, "build_args": {}}),
    )
    return manifest


@pytest.fixture
def repo(tmp_path):
    """A minimal three-zone repo: profile at root, data beside it, build/ rendered."""
    root = tmp_path / "exemplar"
    _write(root / "profile.yml", PROFILE_YAML)
    _write(root / "data" / "channel_databases" / "als.json", '{"channels": []}\n')
    _write(root / "build" / "config.yml", "project: exemplar\n")
    _write_manifest(root)
    return root


# --- the two non-negotiable properties -------------------------------------


def test_comment_only_profile_edit_is_clean(repo):
    """Cosmetics must not refuse a start.

    The fingerprint hashes the RESOLVED document, not the file bytes, so
    comments and key reordering are invisible to it by construction. Pinned
    here because a regression to byte-hashing would still pass every
    "detects a change" test while making the refusal useless in practice.
    """
    (repo / "profile.yml").write_text(
        "# The facility's build profile.\n"
        "# Reviewed 2026-08-10 — no functional change.\n"
        "name: Exemplar\n"
        "app_template: hello_world   # the bundle\n"
        "data: data\n",
        encoding="utf-8",
    )

    report = staleness.check_drift(repo)

    assert report.state is DriftState.CLEAN
    assert not report.refuses
    assert report.changed_keys == ()


def test_data_tree_edit_is_drift(repo):
    """SC-3: editing facility data the profile points at must refuse a start.

    The profile YAML is byte-identical here. Only the tree it names moved —
    which is what a facility editing its channel database actually does.
    """
    _write(repo / "data" / "channel_databases" / "als.json", '{"channels": ["SR:BPM1"]}\n')

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.refuses
    assert report.changed_keys == ("data/",)
    assert "--as-built" in report.remedy


# --- drift detail ----------------------------------------------------------


def test_profile_key_edit_names_the_changed_key(repo):
    """Named with the spelling the file uses.

    ``app_template:`` is stored canonically as ``data_bundle``; a refusal that
    printed the canonical name would send the operator looking for a key their
    profile does not contain.
    """
    _write(repo / "profile.yml", PROFILE_YAML.replace("hello_world", "control_assistant"))

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.changed_keys == ("app_template",)
    assert "app_template" in report.message


def test_alias_spelling_change_alone_is_clean(repo):
    """Rewriting a key in its other sanctioned spelling changes nothing real."""
    _write(repo / "profile.yml", PROFILE_YAML.replace("app_template:", "data_bundle:"))

    assert staleness.check_drift(repo).state is DriftState.CLEAN


def test_added_profile_key_is_named(repo):
    _write(repo / "profile.yml", PROFILE_YAML + "tier: 2\n")

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert "tier" in report.changed_keys


def test_persona_delta_edit_names_the_delta(repo):
    """The refusal must name the file the operator edited.

    Every file input a profile points at once folded into a single digest
    labelled ``data/``, so editing ``personas/readonly.yml`` refused with
    "Changed: data/" — a directory the operator had not touched, and one this
    repo need not even have.
    """
    _write(repo / "personas" / "readonly.yml", "name: readonly\n")
    _write_manifest(repo)

    _write(repo / "personas" / "readonly.yml", "name: readonly\nmodel: haiku\n")

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.changed_keys == ("personas/readonly.yml",)
    assert "personas/readonly.yml" in report.message
    assert "data/" not in report.message


def test_convention_directory_edit_names_that_directory(repo):
    """Each convention directory is its own input, so each is its own name."""
    _write(repo / "agents" / "beamline.md", "# beamline\n")
    _write_manifest(repo)

    _write(repo / "agents" / "beamline.md", "# beamline ops\n")

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.changed_keys == ("agents/",)


def test_material_change_is_named_once(repo):
    """The combined digest moves with every material edit; it is not a second name.

    Reporting it beside the input that explains it would say the same edit
    twice, once vaguely — which is how the old message read.
    """
    _write(repo / "data" / "channel_databases" / "als.json", '{"channels": ["SR:BPM1"]}\n')

    assert staleness.MATERIAL_DISPLAY not in staleness.check_drift(repo).changed_keys


def test_unenumerated_material_is_still_reported(repo):
    """The combined digest is the backstop for material no key names.

    The per-input split mirrors the build's material fold. Should the fold grow
    an input this module does not enumerate, a change to it moves only the
    combined digest — and must still be reported. Vaguely is acceptable;
    silently is the failure this whole check exists to prevent.
    """
    stamped = {
        "name": "unchanged",
        staleness.MATERIAL_KEY: "before",
        f"{staleness.MATERIAL_PREFIX}data/": "same",
    }
    current = dict(stamped, **{staleness.MATERIAL_KEY: "after"})

    assert staleness._changed_keys(current, stamped) == (staleness.MATERIAL_DISPLAY,)


def test_an_empty_stamp_names_nothing_rather_than_everything(repo):
    """A stamp with no digests in it supports no comparison at all.

    Diffing against ``{}`` would find every key "changed" and report a total
    rewrite of a profile the operator edited one line of. Unknowable has to read
    as unknowable, which the caller then states plainly.
    """
    _write_manifest(repo, **{staleness.KEY_DIGESTS_MANIFEST_KEY: {}})
    _write(repo / "profile.yml", PROFILE_YAML.replace("hello_world", "control_assistant"))

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.changed_keys == ()
    assert "recorded no detail" in report.message


def test_drift_without_stamped_digests_still_refuses(repo):
    """The verdict comes from the hash; the key list is only commentary.

    The digests are optional in the manifest, so a build can carry none —
    hand-edited, truncated, or written by a renderer that skipped them. Losing
    the detail must not lose the refusal: degrading to "something changed" is
    correct, degrading to CLEAN would be a silent deploy of edited source.
    """
    _write_manifest(repo, **{staleness.KEY_DIGESTS_MANIFEST_KEY: None})
    _write(repo / "profile.yml", PROFILE_YAML.replace("hello_world", "control_assistant"))

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.changed_keys == ()
    assert "profile.yml" in report.message


def test_clean_repo_reports_clean(repo):
    report = staleness.check_drift(repo)

    assert report.state is DriftState.CLEAN
    assert not report.refuses
    assert report.current_hash == report.stamped_hash


# --- the two ways of having no verdict -------------------------------------


def test_unparseable_profile_is_unresolvable(repo):
    _write(repo / "profile.yml", "name: [unterminated\n")

    report = staleness.check_drift(repo)

    assert report.state is DriftState.UNRESOLVABLE
    assert report.refuses
    assert staleness.UNVERIFIABLE_MESSAGE in report.message
    assert "--as-built" in report.remedy


def test_missing_profile_is_unresolvable(repo):
    (repo / "profile.yml").unlink()

    report = staleness.check_drift(repo)

    assert report.state is DriftState.UNRESOLVABLE
    assert staleness.UNVERIFIABLE_MESSAGE in report.message


def test_build_without_fingerprint_is_unresolvable(repo):
    """A manifest with no ``preset_hash`` vouches for nothing.

    The legacy advisory fails open here. The refusal must not: an unverifiable
    build is exactly the case ``--as-built`` exists to make deliberate.
    """
    _write_manifest(repo, preset_hash=None)

    report = staleness.check_drift(repo)

    assert report.state is DriftState.UNRESOLVABLE
    assert report.refuses
    assert "--as-built" in report.remedy


def test_missing_build_is_not_drift(repo):
    """No build is its own state — ``--as-built`` cannot start what was never rendered."""
    staleness.build_manifest_path(repo).unlink()

    report = staleness.check_drift(repo)

    assert report.state is DriftState.NO_BUILD
    assert not report.refuses
    assert "osprey build" in report.remedy
    assert "--as-built" not in report.remedy


def test_corrupt_manifest_reads_as_no_build(tmp_path):
    root = tmp_path / "exemplar"
    _write(root / "profile.yml", PROFILE_YAML)
    _write(staleness.build_manifest_path(root), "{not json")

    assert staleness.check_drift(root).state is DriftState.NO_BUILD


def test_empty_repo_reads_as_no_build(tmp_path):
    assert staleness.check_drift(tmp_path).state is DriftState.NO_BUILD


# --- version skew ----------------------------------------------------------


def test_version_skew_warns_but_never_refuses(repo):
    """A patch bump must not stop an operator starting the stack they have."""
    _write_manifest(repo, osprey_version="1999.1.1")

    report = staleness.check_drift(repo)

    assert report.state is DriftState.CLEAN
    assert not report.refuses
    assert report.version_skew is not None
    assert report.version_skew.stamped == "1999.1.1"
    assert "osprey build" in report.version_skew.message


def test_version_skew_reported_alongside_drift(repo):
    """The two signals are independent; drift must not swallow the skew line."""
    _write_manifest(repo, osprey_version="1999.1.1")
    _write(repo / "profile.yml", PROFILE_YAML.replace("hello_world", "control_assistant"))

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.version_skew is not None


def test_matching_version_reports_no_skew(repo):
    assert staleness.check_drift(repo).version_skew is None


def test_unknown_stamped_version_is_not_skew(repo):
    """An unknown version means "can't compare", which must never read as skew."""
    _write_manifest(repo, osprey_version="unknown")

    assert staleness.check_drift(repo).version_skew is None


# --- consumer surface ------------------------------------------------------


def test_status_line_distinguishes_every_state(repo):
    clean = staleness.check_drift(repo).status_line
    assert "in sync" in clean

    _write(repo / "profile.yml", PROFILE_YAML.replace("hello_world", "control_assistant"))
    drifted = staleness.check_drift(repo)
    assert "OUT OF DATE" in drifted.status_line
    assert "app_template" in drifted.status_line

    staleness.build_manifest_path(repo).unlink()
    assert "run `osprey build`" in staleness.check_drift(repo).status_line


def test_manifest_lives_at_the_build_root(tmp_path):
    """One place, so a stray nested render can never answer for the repo."""
    assert staleness.build_manifest_path(tmp_path) == tmp_path / "build" / ".osprey-manifest.json"


def test_fingerprint_reuses_the_build_hash(repo):
    """IA-1: no second hash mechanism — the verdict input IS ``compute_profile_hash``."""
    from osprey.cli.build_profile import compute_profile_hash

    fingerprint = staleness.profile_fingerprint(repo / "profile.yml")

    assert fingerprint is not None
    assert fingerprint.profile_hash == compute_profile_hash(repo / "profile.yml")


def test_fingerprint_of_unresolvable_profile_is_none(tmp_path):
    _write(tmp_path / "profile.yml", "name: [unterminated\n")

    assert staleness.profile_fingerprint(tmp_path / "profile.yml") is None


def test_check_drift_never_raises_on_a_hostile_repo(tmp_path):
    """Every lifecycle verb calls this before doing anything; it must not throw."""
    root = tmp_path / "hostile"
    _write(root / "profile.yml", PROFILE_YAML)
    _write(staleness.build_manifest_path(root), json.dumps({"creation": "not-a-dict"}))

    assert staleness.check_drift(root).refuses
