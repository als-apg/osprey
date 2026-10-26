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
        "sha256:467acb560630af8ad73cf6438dfa6da16f920d0bc701a71bc1a76b07145afb61"
    ),
    "hello-world": ("sha256:029244f425e473d9e22635202920f1bb67474520cb5cffa97318a33c80c45240"),
    "tiers/admin": ("sha256:1a50a4f75750dd6d006e4ad3238eaeddeba1ff44182bd689a84efcde5585932b"),
    "tiers/readonly": ("sha256:bec2c97daa2bd8809bd3eb95b8509b3fba3425471a3b93c562a9429b1acde898"),
    "tiers/readwrite": ("sha256:04c999b7f9da029263f3103065a0d224889a3147359237aa6b815c1fe4dd0bd7"),
    "apps/channel-finder": (
        "sha256:e2b8213367141ab27e1d71a6fba42143c6a94ade7b7efe48f36b9ca0c0a13df6"
    ),
    "apps/knowledge": ("sha256:3968102d73b9fd2ee13e8f0eb62f9e1e17700a725673222604b3cb833ec29a04"),
    "apps/logbook": ("sha256:10fdb6c17c6725fc314745f767772c242afbc5ebd67ff6f61f38858e5c8755f6"),
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
