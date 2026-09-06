"""Pin: every bundled preset's resolved-content hash.

``preset_hash`` is stamped into ``.osprey-manifest.json`` at build time and
compared by the deploy-side staleness advisory, so a hash that moves reports
drift on every already-deployed project. The digests below are deliberately
hardcoded rather than recomputed — a test that derives its expectation from the
code under test would pin nothing.

Any change to these values means a preset's *resolved content* changed. That is
a legitimate thing to do, but it is a deploy-visible event: update the digest
here in the same commit, knowingly.
"""

from __future__ import annotations

from osprey.cli.build_profile import list_presets
from osprey.cli.build_profile_merge import _hash_resolved_profile

# preset name -> resolved-content hash, pre-rename.
PINNED_PRESET_HASHES: dict[str, str] = {
    # Every digest moved together when the app templates were converted into the
    # presets. A preset now carries the whole declarative config a deployment
    # renders — the control system, the services, the approval table, the panel
    # selection — where before that content lived in `apps/<name>/config.yml.j2`
    # and was rendered underneath the profile. `app_template:` is also popped
    # before hashing now, so it contributes nothing to a digest. Every already
    # deployed project therefore reads stale exactly once, which is the correct
    # signal: its rebuilt config really is written from a different source.
    "ariel-standalone": ("sha256:878eda90042c8b8f7cf510eb8ce36fdd34c5643dab99b0654b6488d78e4be4f3"),
    "channel-finder-standalone": (
        "sha256:3ad01aea92de1be203ccd6252c58892b32e8c8ad3643ffdca96f8bf6b9ac624f"
    ),
    "control-assistant": (
        "sha256:ce44817a8f321f3ba6e860c01939844177df8bc2ec9b303de53b331c51e04787"
    ),
    "control-assistant-admin": (
        "sha256:e6faada8065b009f7bdf42086b22401d1034c605085e4dd12e29d8c2283443a6"
    ),
    "control-assistant-knowledge": (
        "sha256:12d1f17757121b22ef14d9d0c7d0cddab07ca74ef9c80b6a5e6ac38a8332c3dc"
    ),
    "control-assistant-logbook": (
        "sha256:dbba58b25a2ea53414a1dea4a834168d06cb69dafe002a4a9888036edc74a1c2"
    ),
    "control-assistant-readonly": (
        "sha256:3ba73e914a0df9681d83f9a2d0480baace62cbe7e6ef501f5dcd289bc109e015"
    ),
    "control-assistant-readwrite": (
        "sha256:f4c9e8ead30574fedabba09fb4db636fd277cd3502f5f2f2ba9dc97e0a90af9d"
    ),
    "control-assistant-va-readwrite": (
        "sha256:fa502a4de7d556148c76d4cbf902346096069aa26ce47431d68e36a5a1fd9359"
    ),
    "hello-world": ("sha256:6ed67c73be15b1a24a4b1a05a7e095cd2769b4332fb5697bdcfe34a07ed3eaad"),
}


def test_bundled_preset_set_is_pinned():
    """A new preset must be classified here before it ships.

    Without this the per-preset loop below would silently skip an unpinned
    preset, and the pin would degrade as presets are added.
    """
    assert list_presets() == sorted(PINNED_PRESET_HASHES)


def test_hashing_does_not_mutate_the_callers_dict(tmp_path):
    """The caller's dict survives hashing unchanged.

    ``_hash_resolved_profile`` resolves ``extends`` and folds in file material
    to build what it digests, and every caller keeps using the dict it passed
    in afterwards.
    """
    raw = {"name": "Demo", "provider": "anthropic", "model": "haiku"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "provider": "anthropic", "model": "haiku"}
