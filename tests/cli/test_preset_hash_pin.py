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
    # The fifth move, and again not every preset: the three root presets stopped
    # shipping a feedback recipient. `web.feedback.email` carried a maintainer's
    # own mailbox, and the prefilled draft it receives can carry a session's
    # scrollback, so an unconfigured deployment now offers no Email channel at
    # all and the facility names the recipient. The five control-assistant
    # personas inherit the root preset and move with it; hello-world names no
    # `web:` block and stands still.
    "ariel-standalone": ("sha256:388c3db8e75bbf31519c1b9fa1a30483c6a42221a18758bc7f5ff4a6f40500f0"),
    "channel-finder-standalone": (
        "sha256:e9191d66287745d731fd238b3893677d8a16426a4a17c495b9e2086f42b6e77d"
    ),
    "control-assistant": (
        "sha256:6feee637a96e871ee57f437375eead18cc5fd5652345cb918d9ede9a91e9c68c"
    ),
    "control-assistant-admin": (
        "sha256:85556bc278e3fc27e323a5c4699a1ccb9119d503ed535dbe7bc3a1d31db1ac02"
    ),
    "control-assistant-knowledge": (
        "sha256:3a2664feecf9f4a22cd8a0e10d90d8ca39c99aa647863f2fadcd1c2c4584f7b1"
    ),
    "control-assistant-logbook": (
        "sha256:e52cb8666629459d370f47264ee6737805dc6d00b113ba3108646d09e37a05e9"
    ),
    "control-assistant-readonly": (
        "sha256:19fc54cb889725e742a1b7d44a8a8e9ae4b74506c9e47ed22c5c991c366ac6f3"
    ),
    "control-assistant-readwrite": (
        "sha256:3ea78f64f7e5dbffacdca537465ebdc9530917ffb938760b83a36ad754f49691"
    ),
    "control-assistant-va-readwrite": (
        "sha256:71c9c770a0cfc611e06d38228b8cc797225a408f55ea02ecd31af7bcaf8bff05"
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
