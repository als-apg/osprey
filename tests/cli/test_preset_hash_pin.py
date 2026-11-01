"""Pin: every bundled preset's, tier's and app's content hash.

``preset_hash`` is stamped into ``.osprey-manifest.json`` at build time, and
each tier's and app's digest into the provenance header of the file ``osprey
init`` materializes from it; the deploy-side staleness advisory compares both,
so a hash that moves reports drift on every already-deployed project. The
digests below are deliberately hardcoded rather than recomputed — a test that
derives its expectation from the code under test would pin nothing.

Any change to these values means a preset's *resolved content*, or a tier's or
app's document, changed. That is a legitimate thing to do, but it is a
deploy-visible event: update the digest here in the same commit, knowingly.
"""

from __future__ import annotations

from osprey.cli.build_profile import list_presets
from osprey.cli.build_profile_merge import _hash_resolved_profile, compute_preset_hash
from osprey.cli.build_profile_presets import list_apps, list_tiers


def _bundled_names() -> list[str]:
    """Every bundled root preset, then every tier and app as ``<kind>/<stem>``."""
    return (
        list_presets()
        + [f"tiers/{name}" for name in list_tiers()]
        + [f"apps/{name}" for name in list_apps()]
    )


# preset, tier or app name -> content hash. A root preset hashes its resolved
# profile; a tier or app hashes its own document, since that document is what
# ``osprey init`` copies into a deployment.
PINNED_PRESET_HASHES: dict[str, str] = {
    # A digest moves whenever its preset's resolved content does, and the commit
    # that moves it says why. A comment is not resolved content, and neither is
    # a key the hash pops before digesting, so editing either moves nothing.
    "control-assistant": (
        "sha256:7d90bd6a35a692ae1a8baca9a4d7fc2b1a77a83f11b653697c9a9059773dc64b"
    ),
    "hello-world": ("sha256:64e800d50fa5250991e06c374a10282f0728f3daa77dfb6db748bf8306dc576f"),
    "tiers/admin": ("sha256:dc41dc870db1340a707c9bf6e475ed0998b37e0db3f635a77c80115106d1f106"),
    "tiers/readonly": ("sha256:9cfce627f4df87aac7f8d212cce4819d487471e2ce4b386c4b6da3f127f41605"),
    "tiers/readwrite": ("sha256:787e5efd355a84b81c1d9c8cd9d6fa6252dda5e0d7d096ea417a0e8805ceb3bd"),
    "apps/channel-finder": (
        "sha256:ba2a4a1ecf0e0909186e356e6c7048914c04f85de986fdb021b8b5760a1f0ccf"
    ),
    "apps/knowledge": ("sha256:2935d762f22f240e7b5c70ba4277dd657daa7cfb3012b7b2f78c87502c7e2e7e"),
    "apps/logbook": ("sha256:940270aee05d7b67254e91c61f0bb459bb8433282d183736c60e39993a7d7290"),
}


def test_bundled_preset_set_is_pinned():
    """A new preset, tier or app must be classified here before it ships.

    Without this the digest comparison below would silently skip an unpinned
    one, and the pin would degrade as presets, tiers and apps are added.
    """
    assert sorted(_bundled_names()) == sorted(PINNED_PRESET_HASHES)


def test_every_bundled_preset_hash_is_unchanged():
    """Every preset, tier and app hashes to its pinned digest.

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
