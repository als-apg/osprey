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

from osprey.cli.build_profile import list_presets
from osprey.cli.build_profile_merge import _hash_resolved_profile, compute_preset_hash
from osprey.cli.build_profile_presets import list_tiers


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
        "sha256:6ba7255b585db3ba909beffd64d7d32e5c834e5a32d465ff3eb9e6a76bec0acc"
    ),
    "hello-world": ("sha256:64e800d50fa5250991e06c374a10282f0728f3daa77dfb6db748bf8306dc576f"),
    "tiers/admin": ("sha256:dc41dc870db1340a707c9bf6e475ed0998b37e0db3f635a77c80115106d1f106"),
    "tiers/readonly": ("sha256:9cfce627f4df87aac7f8d212cce4819d487471e2ce4b386c4b6da3f127f41605"),
    "tiers/readwrite": ("sha256:787e5efd355a84b81c1d9c8cd9d6fa6252dda5e0d7d096ea417a0e8805ceb3bd"),
    "tiers/channel-finder": (
        "sha256:7d5489e8dcf7f77a7576a3f320697b5b10651b46300dd687f33cf4d0af10f028"
    ),
    "tiers/knowledge": ("sha256:fcf1418d1119679cd91daf65533ad8e8093f3e5e1548893b19c02127ac608355"),
    "tiers/logbook": ("sha256:27cbc67c42676bc424a9abb899ad3948dfd43236585608ccaaa252c9c07fa7f0"),
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


def test_hashing_does_not_mutate_the_callers_dict(tmp_path):
    """The caller's dict survives hashing unchanged.

    ``_hash_resolved_profile`` resolves ``extends`` and folds in file material
    to build what it digests, and every caller keeps using the dict it passed
    in afterwards.
    """
    raw = {"name": "Demo", "provider": "anthropic", "model": "claude-haiku-4-5"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "provider": "anthropic", "model": "claude-haiku-4-5"}
