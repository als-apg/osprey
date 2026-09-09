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
    # The fifth move, and again nine of ten: the panel-rail verbs
    # (`add_panel_to_rail`, `remove_panel_from_rail`, `register_panel`) became
    # approval-governed, and the three presets that ship an approval table —
    # ariel-standalone, hello-world, control-assistant, whose six `extends`
    # children inherit the lines — each gained an `always` entry for them.
    # channel-finder-standalone ships no approval table at all: its rail verbs
    # are governed too but fall to `default_policy`, so nothing in its resolved
    # content moved and its digest alone stands still. Rewriting its approval
    # comment did not move it either — the hash is over resolved content, not
    # over the file's text. A rebuilt project prompts before the rail changes,
    # so the staleness advisory firing on already-deployed projects is correct.
    # The sixth move, and again not every preset: the three root presets stopped
    # shipping a feedback recipient. `web.feedback.email` carried a maintainer's
    # own mailbox, and the prefilled draft it receives can carry a session's
    # scrollback, so an unconfigured deployment now offers no Email channel at
    # all and the facility names the recipient. The five control-assistant
    # personas inherit the root preset and move with it; hello-world names no
    # `web:` block and stands still.
    "ariel-standalone": ("sha256:2da76441f9b23fbb5f6be2a3ffb0ece92f81a5e490261bc42afa0b1220139dd4"),
    "channel-finder-standalone": (
        "sha256:e9191d66287745d731fd238b3893677d8a16426a4a17c495b9e2086f42b6e77d"
    ),
    "control-assistant": (
        "sha256:d3d49b8f4a4ccbde304b6018bec3629bcd28b8b067ede712f59a4ba5f440f0e1"
    ),
    "control-assistant-admin": (
        "sha256:d9f13eed69481c0f9c0c8c3c4fd7da566d40e76d2daa8bcefd4127dce4962bd1"
    ),
    "control-assistant-knowledge": (
        "sha256:5c6bfa6e800cf870d1c312a8d68b877065a32fca5e313ff9560850b4c0bb5134"
    ),
    "control-assistant-logbook": (
        "sha256:ed131ffa232d813321f485524f5404fc3242eb888fd6a4a8884760fb37c291f3"
    ),
    "control-assistant-readonly": (
        "sha256:e4011bd738871d17ad5b344108758c577458915117ac77cd09210dccb4c1b921"
    ),
    "control-assistant-readwrite": (
        "sha256:ca7d10ac9f4adca9e2c71601c441d154ee7084934c0f3da73a90944f979a5fd9"
    ),
    "control-assistant-va-readwrite": (
        "sha256:27976f8102e39e45e26f006604f95448acd6bdbbda71014aa83277cdb6587f9e"
    ),
    "hello-world": ("sha256:10ce4bc73c7a244debbf355189dfbcd13feb7d31007c79e83c93aec978831b65"),
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
