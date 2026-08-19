"""The channel-database snapshot stays current because the build-drift gate says so.

``osprey build`` snapshots the Channel Finder database into a ``channels.json``
artifact under ``build/``. That artifact goes stale the moment a facility edits
its channel database, and the DECISION documented here is how staleness is
caught: the existing build-drift gate, not a regeneration hook at ``osprey up``
time.

A channel database lives under the profile's ``data:`` tree, which
``_fold_profile_material`` folds into the profile fingerprint. Editing one moves
``compute_profile_hash`` while ``profile.yml`` itself is byte-identical, so the
fingerprint stamped into the build manifest no longer matches and ``osprey up``
refuses with the rebuild remedy. No second mechanism is needed, and a
regeneration hook would add an up-time code path that can disagree with what
``build`` produced.

The limitation this buys is deliberate: a database pointed at from OUTSIDE the
profile's ``data:`` tree is not folded, so its edits are invisible to the gate.
That is recorded in the proposal's Risks and is not asserted here — asserting it
would pin behaviour we have chosen not to provide.

The fold itself is pinned in ``test_profile_material_hash.py`` and
``test_profile_hash_folding.py``; the refusal is pinned in
``tests/deployment/test_drift_refusal.py``. This file speaks only for the
snapshot's source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.cli.build_profile import compute_profile_hash

PROFILE_YAML = "name: Facility\napp_template: hello_world\ndata: data\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def profile(tmp_path):
    """A profile whose ``data:`` tree holds the database the snapshot is taken from."""
    root = tmp_path / "facility"
    _write(root / "data" / "channel_databases" / "als.json", '{"channels": []}\n')
    return _write(root / "profile.yml", PROFILE_YAML)


def test_editing_a_channel_database_under_data_moves_the_fingerprint(profile):
    """The whole decision rests on this: the snapshot's source is folded material.

    ``profile.yml`` is untouched — only the database the snapshot was taken from
    changed, which is exactly what a facility adding a channel does.
    """
    before = compute_profile_hash(profile)

    _write(
        profile.parent / "data" / "channel_databases" / "als.json",
        '{"channels": ["SR:BPM1"]}\n',
    )

    assert compute_profile_hash(profile) != before


def test_adding_a_channel_database_moves_the_fingerprint(profile):
    """A new database file is new snapshot input, so it has to count as a change."""
    before = compute_profile_hash(profile)

    _write(profile.parent / "data" / "channel_databases" / "booster.json", '{"channels": []}\n')

    assert compute_profile_hash(profile) != before


def test_an_untouched_channel_database_leaves_the_fingerprint_alone(profile):
    """Without this, every start would refuse and operators would learn to bypass."""
    assert compute_profile_hash(profile) == compute_profile_hash(profile)


def test_a_channel_database_edit_makes_a_built_repo_refuse_to_start(profile):
    """End of the chain: the moved fingerprint is what ``osprey up`` acts on.

    The repo below is built — a manifest stamped with the fingerprint as it was
    when ``channels.json`` was snapshotted. Editing the database afterwards is
    the stale-snapshot situation, and the gate has to call it drift and point at
    a rebuild rather than start the stale build in silence.
    """
    from osprey.deployment import staleness
    from osprey.deployment.staleness import DriftState

    repo = profile.parent
    _write(repo / "build" / "config.yml", "project: facility\n")
    fingerprint = staleness.profile_fingerprint(profile)
    assert fingerprint is not None, "fixture profile must resolve"
    _write(
        staleness.build_manifest_path(repo),
        json.dumps(
            {
                "schema_version": "1.2.0",
                "creation": {
                    "template": "hello-world",
                    "preset_hash": fingerprint.profile_hash,
                    staleness.KEY_DIGESTS_MANIFEST_KEY: fingerprint.key_digests,
                },
                "build_args": {},
            }
        ),
    )
    assert staleness.check_drift(repo).state is DriftState.CLEAN

    _write(
        repo / "data" / "channel_databases" / "als.json",
        '{"channels": ["SR:BPM1"]}\n',
    )

    report = staleness.check_drift(repo)

    assert report.state is DriftState.DRIFT
    assert report.refuses
    assert "data/" in report.changed_keys
    assert "osprey build" in report.remedy
