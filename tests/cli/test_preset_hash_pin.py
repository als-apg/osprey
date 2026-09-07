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
    # The third move, one commit: `control-context` joined every preset's
    # `hooks:`, putting the active control target and each target's write state
    # in front of the agent at session start and whenever it changed since its
    # last turn. A rebuilt project gains the hook file in `.claude/hooks/` and
    # two settings.json registrations (SessionStart, UserPromptSubmit).
    # The fourth move, and the only one that is not every preset: every preset
    # that spells an ARIEL approval policy gained
    # `approval.tools.entry_publish: always` beside its existing `entry_create`
    # line. Publishing is the half of a logbook write that reaches the
    # facility, and it was gated nowhere. channel-finder-standalone runs no
    # ARIEL server and has no approval table for it, so it alone stands still.
    # A rebuilt project prompts before a publish, so the staleness advisory
    # firing on already-deployed projects is the correct signal.
    "ariel-standalone": ("sha256:2522f525c8850daf2915e59b898a13269d1c924d59d0d0f868b219a2b8e72c6d"),
    "channel-finder-standalone": (
        "sha256:5c5d670fb6d048e854dfa5ecff9b02d6b9bf504a4cd574d6a5f66b34e18e3fa1"
    ),
    "control-assistant": (
        "sha256:e1cd276ba9e3062452b5afe86daed341be5c8cfa12b6a992d7b4ea0e33c2ff05"
    ),
    "control-assistant-admin": (
        "sha256:c29f07d93cee491afda6276603d5a86a9cf0e919960ba598f78c8906e419eb3c"
    ),
    "control-assistant-knowledge": (
        "sha256:84fb320d701ef1d14d18155c0c59dfff4b2bcc2d5441175733f7fb6d2aad90f0"
    ),
    "control-assistant-logbook": (
        "sha256:a63cee6c479a8bee82c1f6e261b921ea96708226cb9e0d331d52d9e800b195e4"
    ),
    "control-assistant-readonly": (
        "sha256:99a75fd3714a9114fd527724b77de6b7bd36aac571b821882b8903be6c626c3c"
    ),
    "control-assistant-readwrite": (
        "sha256:3a8c88791900e5c0d7bff064db59f65a1244dd07425a90e1e17301311aa63a19"
    ),
    "control-assistant-va-readwrite": (
        "sha256:93c34788a885b8013162c85bb9f39fd506b6950c210fc8f07ae1579e13f42f20"
    ),
    "hello-world": ("sha256:5f23b2a31f03c885b20dae82781aa87efda81db3960d6865c3bc93e54e4ca837"),
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
