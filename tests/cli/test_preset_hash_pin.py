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
    # The seventh move, and again not every preset: the three presets carrying a
    # `web:` block stopped rendering `web.docs_url`, `web.feedback.email` and
    # `web.feedback.github_repo` as live keys — each documents them as a
    # commented example instead, so a deployment's own profile.yml no longer
    # carries the OSPREY project's documentation site and tracker as if the
    # facility had chosen them. The five control-assistant personas extend the
    # root preset and move with it; hello-world names no `web:` block and
    # stands still. The code defaults still apply — the docs and tracker
    # defaults are unchanged, and the mail default already ships blank — so a
    # rebuilt project behaves exactly as before; the advisory firing is the
    # correct signal that its rendered config really is three leaves shorter.
    # The eighth move, and the narrowest yet: the two presets that carry a
    # `channel_finder` block — control-assistant and channel-finder-standalone —
    # gained `channel_finder.query_max_rows: 500`, the cap on what the
    # middle-layer `run_sql` tool hands back. It was a number fixed in the tool,
    # so a facility could not decide how much of its channel table was worth a
    # turn of the agent's context. The value is the one the tool already
    # applied, so a rebuilt project behaves identically — the digest moves
    # because the preset now STATES it. control-assistant's six `extends`
    # children inherit the root preset and move with it; ariel-standalone and
    # hello-world carry no channel finder and stand still. Every other key this
    # change added is shipped commented, and a comment is not resolved content.
    "ariel-standalone": ("sha256:389fad6bd826efc4b53ea263800110585867aab31c92d9931206897c75548643"),
    "channel-finder-standalone": (
        "sha256:2dfc06f64433fcb1d8393931dccf76550e75ac76dc12f5011029010e02aa9448"
    ),
    "control-assistant": (
        "sha256:ab96027bc359a067f44aa04028ebbac34358e675fca503a1d1cc4bcd790c18aa"
    ),
    "control-assistant-admin": (
        "sha256:cef4c0b2de1d152ef4b88d8b8cd1e31f900f4719d99c171ebecc08de3347d769"
    ),
    "control-assistant-knowledge": (
        "sha256:98cdd6a4930cf92ceaa9bc6c3e741ac51e90ca56cf86d40551e84c00e9179c20"
    ),
    "control-assistant-logbook": (
        "sha256:38483b2115bc153440dfce862c6d337b5ad6beb2addf3c1e4dc6c57f1efdf2ce"
    ),
    "control-assistant-readonly": (
        "sha256:c89b22d754632805e7cdb8b3026bdaf7f7d4401628904083480f4681b4a71a0f"
    ),
    "control-assistant-readwrite": (
        "sha256:a4a695bbc044975ba2258a9cd58c46e283b832e2d4b9efd1720b9e1860502ba8"
    ),
    "control-assistant-va-readwrite": (
        "sha256:862dd6a5c5df3843c066ebdc26806c9283f75c9a658145fba1856b47d6e14ce1"
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
