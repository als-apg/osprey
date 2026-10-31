"""``check_drift`` refuses a render made before the tiers/apps vocabulary.

A profile spelled with the retired roster keys hashes identically to its
``tier`` rewrite, so the fingerprint alone would call a render from an older
release clean while its ``config.yml`` still says the old words. The stamped
``creation.osprey_version`` is what tells the two apart: a render from a
release older than the vocabulary release (the profile-schema floor) is
refused with a sentence naming that release and the remedy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from packaging.version import Version

from osprey.cli.build_profile_load import _PROFILE_SCHEMA_MIN_OSPREY
from osprey.deployment import staleness
from osprey.deployment.staleness import DriftState

PROFILE_YAML = """\
name: Exemplar
provider: anthropic
data: data
"""

PRE_VOCABULARY = "2026.6.2"
POST_VOCABULARY = "2099.1.1"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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
    root = tmp_path / "exemplar"
    _write(root / "profile.yml", PROFILE_YAML)
    _write(root / "data" / "channel_databases" / "als.json", '{"channels": []}\n')
    _write(root / "build" / "config.yml", "project: exemplar\n")
    _write_manifest(root)
    return root


def test_fixture_stamps_bracket_the_vocabulary_release():
    floor = Version(_PROFILE_SCHEMA_MIN_OSPREY)
    assert Version(PRE_VOCABULARY) < floor <= Version(POST_VOCABULARY)


def test_render_before_the_vocabulary_release_is_refused(repo):
    """Same fingerprint, older release: refused, not CLEAN."""
    _write_manifest(repo, osprey_version=PRE_VOCABULARY)

    report = staleness.check_drift(repo)

    assert report.refuses
    assert report.state is DriftState.UNRESOLVABLE
    assert report.message == (
        f"rendered by osprey {PRE_VOCABULARY} before tiers/apps; run osprey build"
    )
    assert "--as-built" in report.remedy


def test_refusal_wins_over_a_matching_fingerprint(repo):
    """The old-spelled profile hashes like its rewrite — the hash must not decide."""
    _write_manifest(repo, osprey_version=PRE_VOCABULARY)

    report = staleness.check_drift(repo)

    assert report.stamped_hash == staleness.profile_fingerprint(repo / "profile.yml").profile_hash
    assert report.state is not DriftState.CLEAN


def test_refusal_also_covers_a_drifted_profile(repo):
    _write_manifest(repo, osprey_version=PRE_VOCABULARY)
    _write(repo / "profile.yml", PROFILE_YAML.replace("anthropic", "cborg"))

    report = staleness.check_drift(repo)

    assert report.refuses
    assert "before tiers/apps" in report.message


def test_the_floor_itself_is_not_refused(repo):
    _write_manifest(repo, osprey_version=_PROFILE_SCHEMA_MIN_OSPREY)

    report = staleness.check_drift(repo)

    assert report.state is DriftState.CLEAN


def test_a_later_release_is_clean_with_a_skew_warning(repo):
    _write_manifest(repo, osprey_version=POST_VOCABULARY)

    report = staleness.check_drift(repo)

    assert report.state is DriftState.CLEAN
    assert report.version_skew is not None


def test_the_installed_lineage_counts_as_the_floor(repo, monkeypatch):
    """Between releases the lineage sits behind the floor it already honours.

    A render made by this very checkout stamps its lineage; refusing it would
    make every fresh build unstartable until the release is tagged.
    """
    monkeypatch.setattr(staleness, "_installed_version", lambda: PRE_VOCABULARY)
    _write_manifest(repo, osprey_version=PRE_VOCABULARY)

    report = staleness.check_drift(repo)

    assert report.state is DriftState.CLEAN
    assert report.version_skew is None


def test_a_fresh_build_from_this_checkout_is_clean(repo):
    assert staleness.check_drift(repo).state is DriftState.CLEAN


@pytest.mark.parametrize("stamp", ["unknown", "", "not-a-version", None, 7])
def test_an_uncomparable_stamp_is_not_judged_too_old(repo, stamp):
    """ "Cannot compare" must never read as "too old"; check_drift never raises."""
    _write_manifest(repo, osprey_version=stamp)

    report = staleness.check_drift(repo)

    assert "before tiers/apps" not in report.message
    assert report.state is DriftState.CLEAN


def test_no_build_still_reads_as_no_build(tmp_path):
    assert staleness.check_drift(tmp_path).state is DriftState.NO_BUILD


# ── Per-input digests ────────────────────────────────────────────────


def test_every_folded_input_has_its_own_digest(tmp_path):
    """The drift digests name exactly the inputs the enumeration lists."""
    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.build_profile_merge import (
        ProfileInputRole,
        profile_inputs,
        resolve_profile_document,
    )
    from osprey.cli.variant_selection import VARIANT_SETTING_FILENAME, VARIANT_SETTING_KEY

    repo = tmp_path / "facility"
    _write(repo / "data" / "channels.json", "{}\n")
    _write(repo / "rules" / "safety.md", "rule\n")
    _write(repo / "triggers.yml", "triggers: []\n")
    _write(repo / "tiers" / "readonly.yml", "name: readonly\n")
    _write(repo / "apps" / "logbook.yml", "name: logbook\n")
    _write(repo / "profiles" / "teststand.yml", "# teststand\n")
    _write(repo / VARIANT_SETTING_FILENAME, f"{VARIANT_SETTING_KEY}=teststand\n")
    profile = _write(repo / "profile.yml", PROFILE_YAML + "dispatch:\n  triggers: triggers.yml\n")

    fingerprint = staleness.profile_fingerprint(profile)
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)

    assert fingerprint is not None
    named = {
        key
        for key in fingerprint.key_digests
        if key.startswith(staleness.MATERIAL_PREFIX) and key != staleness.MATERIAL_KEY
    }
    assert named == {
        staleness.MATERIAL_PREFIX + entry.key
        for entry in profile_inputs(document)
        if entry.role is not ProfileInputRole.DOCUMENT
    }
