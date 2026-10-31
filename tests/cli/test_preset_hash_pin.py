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
        "sha256:9b8ee13e44b0a54b28cd071fac21fea2056fac71cc3da18cb714ad331b3f3555"
    ),
    "hello-world": ("sha256:3634eced947779b5da33c87273b7db9fe8e92d62449efa86ed329df354ca6ac4"),
    "tiers/admin": ("sha256:b3c44a1abcfd7f1e0a2172b1bc3ef91017d581c04441ce37b8ca50bea4322fec"),
    "tiers/readonly": ("sha256:2de7e2480b79ffafde8a2fd34b37c4195c31f12fdce5ff376e92c048c44af47f"),
    "tiers/readwrite": ("sha256:2691f6cdd043bc06eba77aefe140bc86140009465677757c3f0fd5f95e3d5c36"),
    "tiers/channel-finder": (
        "sha256:54cf3c8e918e7c8edb22001f8fa8bfd4510ff402d1dafe7bd71526e573d7436c"
    ),
    "tiers/knowledge": ("sha256:21850356b24cb23bbd3af536750c266a51b5792a86f3ac14b68f9b4854c18c0f"),
    "tiers/logbook": ("sha256:20bdfa74ef9ffac62b269f6dcee1389e63a2b2c211bb284a90f3ba039702a35d"),
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
