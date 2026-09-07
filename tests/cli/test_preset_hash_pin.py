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
    "ariel-standalone": ("sha256:389fad6bd826efc4b53ea263800110585867aab31c92d9931206897c75548643"),
    "channel-finder-standalone": (
        "sha256:6c42cc563c637073dd76803ebfb0a371f29437747ee3b6ea701662de44bc4ca9"
    ),
    "control-assistant": (
        "sha256:54780712d416102f266fa303bc96fccc06db262bd70d18c9e2ef511468e8c252"
    ),
    "control-assistant-admin": (
        "sha256:d5bcb632db87fbd7be5339a56be6cdca4e6c5d121a148b73b8b9d369543d2de2"
    ),
    "control-assistant-knowledge": (
        "sha256:11248666b3009d9d8fd3f81460dad2cfeb62e555f19fe2482dbe4876de9ef599"
    ),
    "control-assistant-logbook": (
        "sha256:67d66dbb1d476115efb91b3f444fa7b0f512e446363c4179213d1bf2fb4bf350"
    ),
    "control-assistant-readonly": (
        "sha256:4b9b2fa8c40c1fa9e0eed6eb27f9e6ef181cd0b03c15b93cb99c0db895e5d6ec"
    ),
    "control-assistant-readwrite": (
        "sha256:70111bf6029217c2999f7c1e98c7c43a23cd071a357962bff3530412767a2243"
    ),
    "control-assistant-va-readwrite": (
        "sha256:b2a14f6358a6c896c84fa465bec0c472b0c0538b6f207bc06dea83c4ed7ce679"
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
