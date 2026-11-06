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
    "demo": ("sha256:16ab8b870a5ac08054be2cdb0a8fa1856af345b03b499140974f424588b190ef"),
    "hello-world": ("sha256:64e800d50fa5250991e06c374a10282f0728f3daa77dfb6db748bf8306dc576f"),
    "tiers/admin": ("sha256:197ef2a65251887b0af8a2a03dac1494b366d2bb3fdabe42c5f8227fdc729736"),
    "tiers/readonly": ("sha256:8d24dbff3b01876824e3fc10ac61595d9e841feddf632064894ceef7e5031a81"),
    "tiers/readwrite": ("sha256:b9d4b4bbc15d569649062636055ae0a9997948679b7ec84228991e287b7df2a3"),
    "apps/channel-finder": (
        "sha256:2e35415afcd9927741408391f7f40a2fe2635f3c39c193463e4c2998d9ab0094"
    ),
    "apps/knowledge": ("sha256:ff403d3df3b38db9d6cc27c34d5d8b4180120d63d5f70c690da1e569ac3a47b5"),
    "apps/logbook": ("sha256:f6820d02f5bb2b2afbaee291a243718b7aa8218bbfc9a0b810438e435e1e4932"),
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
