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
from osprey.cli.build_profile_merge import _hash_resolved_profile, compute_preset_hash

# preset name -> resolved-content hash.
PINNED_PRESET_HASHES: dict[str, str] = {
    # Every digest moved together when the app templates were converted into the
    # presets. A preset now carries the whole declarative config a deployment
    # renders — the control system, the services, the approval table, the panel
    # selection — where before that content lived in `apps/<name>/config.yml.j2`
    # and was rendered underneath the profile. `app_template:` is also popped
    # before hashing now, so it contributes nothing to a digest. Every already
    # deployed project therefore reads stale exactly once, which is the correct
    # signal: its rebuilt config really is written from a different source.
    #
    # Merging main moved them all a second time, and for a separate reason:
    # `turn-state` joined every preset's `hooks:`, shipping the reporter that
    # tells the web terminal when a turn starts and ends. A rebuilt project
    # gains the hook file in `.claude/hooks/` and four settings.json
    # registrations (UserPromptSubmit, Stop, StopFailure, and a SessionStart
    # entry matched to `startup|resume|clear`). Deploy-visible, so the
    # staleness advisory firing on already-deployed projects is correct.
    "ariel-standalone": ("sha256:eede725ef001ae4a81569d2dc11bf0cda189de2bf748854191f50c90701be4b6"),
    "channel-finder-standalone": (
        "sha256:7485126a38c7d8282d12c952930969a728d7b68d0228abbb3788f15d5754490e"
    ),
    "control-assistant": (
        "sha256:64ce952375e4899364cdecb3b1f0718428476f205b8ee5fc1c7199d5222b58bd"
    ),
    "control-assistant-admin": (
        "sha256:fb939cd02c7618020be32b5c9918031867d1cb0c3b11c506b51ec1ac4781046c"
    ),
    "control-assistant-knowledge": (
        "sha256:552538d9966725f18e63a17d065b08e2ceaf4319508870d87eba0306807adb3c"
    ),
    "control-assistant-logbook": (
        "sha256:cd88a1a9f6a7d0e68a2d46a76924dd256eebb845a6b6d302d50d4f8dbb0b4822"
    ),
    "control-assistant-readonly": (
        "sha256:1089443113d8b249703936635059a302c41946ed4be5b793d5234fb7ed009336"
    ),
    "control-assistant-readwrite": (
        "sha256:4d63a4075534e9284bf20fb6cd72ce370f2a3bcf3b2ebf42233d6063d8418875"
    ),
    "control-assistant-va-readwrite": (
        "sha256:53da90b4a2a90d1c6926c0b0a7dc59867f9ea98c22c49b7977d527bd0132e852"
    ),
    "hello-world": ("sha256:d5a177599044c7f1ba4c8c67ead8d6e8f684a87c95d2f03d703f6f98c3ae233e"),
}


def test_bundled_preset_set_is_pinned():
    """A new preset must be classified here before it ships.

    Without this the digest comparison below would silently skip an unpinned
    preset, and the pin would degrade as presets are added.
    """
    assert list_presets() == sorted(PINNED_PRESET_HASHES)


def test_every_bundled_preset_hash_is_unchanged():
    """Every preset resolves to its pinned digest.

    This is what gives the values above their meaning: without it the dict is
    dead data, and a preset's resolved content could move — a deploy-visible
    event every already-deployed project reads as stale — with nothing saying
    so.
    """
    actual = {name: compute_preset_hash(name) for name in list_presets()}
    assert actual == PINNED_PRESET_HASHES


def test_hashing_does_not_mutate_the_callers_dict(tmp_path):
    """The caller's dict survives hashing unchanged.

    ``_hash_resolved_profile`` resolves ``extends`` and folds in file material
    to build what it digests, and every caller keeps using the dict it passed
    in afterwards.
    """
    raw = {"name": "Demo", "provider": "anthropic", "model": "haiku"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "provider": "anthropic", "model": "haiku"}
