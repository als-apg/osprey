"""Pin: every bundled preset's and login's content hash.

``preset_hash`` is stamped into ``.osprey-manifest.json`` at build time, and
each login's digest into the provenance header of the file ``osprey
init`` materializes from it; the deploy-side staleness advisory compares both,
so a hash that moves reports drift on every already-deployed project. The
digests below are deliberately hardcoded rather than recomputed — a test that
derives its expectation from the code under test would pin nothing.

Any change to these values means a preset's *resolved content*, or a login's
document, changed. That is a legitimate thing to do, but it is a
deploy-visible event: update the digest here in the same commit, knowingly.
"""

from __future__ import annotations

from pathlib import Path

from osprey.cli.build_profile import list_presets
from osprey.cli.build_profile_drift import app_tree_digest
from osprey.cli.build_profile_merge import _hash_resolved_profile, compute_preset_hash
from osprey.cli.build_profile_presets import list_tiers
from osprey.profiles import presets

PRESET_APPS_DIR = Path(presets.__file__).parent / "apps"


def _bundled_names() -> list[str]:
    """Every bundled root preset, then every login as ``tiers/<stem>``."""
    return list_presets() + [f"tiers/{name}" for name in list_tiers()]


# preset or login name -> content hash. A root preset hashes its resolved
# profile; a login hashes its own document, since that document is what
# ``osprey init`` copies into a deployment.
PINNED_PRESET_HASHES: dict[str, str] = {
    # A digest moves whenever its preset's resolved content does, and the commit
    # that moves it says why. A comment is not resolved content, and neither is
    # a key the hash pops before digesting, so editing either moves nothing.
    "control-assistant": (
        "sha256:0d7a5850e1bdc7d3b5bb124f1f6f80e2a7b9b4b22521f403fb0fd7a08a5e9c63"
    ),
    "hello-world": ("sha256:93890612a70173a6dde7f0a912b929738732a61b157186262377da32d869f9cf"),
    "tiers/admin": ("sha256:dc41dc870db1340a707c9bf6e475ed0998b37e0db3f635a77c80115106d1f106"),
    "tiers/readonly": ("sha256:9cfce627f4df87aac7f8d212cce4819d487471e2ce4b386c4b6da3f127f41605"),
    "tiers/readwrite": ("sha256:787e5efd355a84b81c1d9c8cd9d6fa6252dda5e0d7d096ea417a0e8805ceb3bd"),
    "tiers/channel-finder": (
        "sha256:fae0acda4aeea5d60ce3815317c71cdeb521c94b8fe386da5b6dac9146dccb5f"
    ),
    "tiers/knowledge": ("sha256:aad58175d81be08290f70d7112b4660805de3965f6d14c1db6ef56f201d76ba7"),
    "tiers/logbook": ("sha256:a52457ba046a8a47ae66000d46ced8f07641e10037958518579427d16a1fe5ea"),
}


# application name -> the digest ``osprey init`` records for its copy in the
# application's ``.provenance``. An application is deploy-visible source as a
# preset is: moving its digest reports every copy in a deployment as drifted.
PINNED_APP_TREE_DIGESTS: dict[str, str] = {
    "knob-example": ("sha256:87174cbbcc46166ea276b67b8929924fc5c61a2e81f0de3c051a6d25625e790d"),
}


def test_bundled_preset_set_is_pinned():
    """A new preset or login must be classified here before it ships.

    Without this the digest comparison below would silently skip an unpinned
    one, and the pin would degrade as presets and logins are added.
    """
    assert sorted(_bundled_names()) == sorted(PINNED_PRESET_HASHES)


def test_every_bundled_preset_hash_is_unchanged():
    """Every preset and login hashes to its pinned digest.

    This is what gives the values above their meaning: without it the dict is
    dead data, and a preset's resolved content could move — a deploy-visible
    event every already-deployed project reads as stale — with nothing saying
    so.
    """
    actual = {name: compute_preset_hash(name) for name in _bundled_names()}
    assert actual == PINNED_PRESET_HASHES


def test_every_bundled_application_tree_is_pinned_and_unchanged():
    """Every application the presets ship is classified here, at its pinned digest."""
    actual = {path.name: app_tree_digest(path) for path in PRESET_APPS_DIR.iterdir()}
    assert actual == PINNED_APP_TREE_DIGESTS


def test_hashing_does_not_mutate_the_callers_dict(tmp_path):
    """The caller's dict survives hashing unchanged.

    ``_hash_resolved_profile`` resolves ``extends`` and folds in file material
    to build what it digests, and every caller keeps using the dict it passed
    in afterwards.
    """
    raw = {"name": "Demo", "provider": "anthropic", "model": "claude-haiku-4-5"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "provider": "anthropic", "model": "claude-haiku-4-5"}
