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
    # The ninth move, and hello-world alone: it dropped the two ARIEL approval
    # rows — `approval.tools.entry_create` and `approval.tools.entry_publish` —
    # for a server it never runs, so its resolved content is two leaves shorter.
    # Both were fail-closed either way (the tools do not exist there), so a
    # rebuilt project behaves identically. Every other preset stands still:
    # control-assistant's own additions in this change are commented examples,
    # which the hash does not see.
    # The tenth move, and control-assistant's family alone: the root preset
    # gained `dispatch.network: host`. Its dispatched jobs reach the control
    # system and the plan queue at this machine's own loopback, which a worker
    # on the compose network reads as its own container, so the pair now runs
    # in the host's network namespace beside the web tier that was already
    # there. A rebuilt project deploys its dispatcher and workers differently,
    # so the staleness advisory firing on already-deployed projects is the
    # correct signal. The six `extends` children inherit it; every preset that
    # declares no `dispatch:` block stands still.
    # The eleventh move, and two presets: the write-capable tiers,
    # control-assistant-readwrite and control-assistant-admin, went from one
    # posture key to three — the flat key false, the epics block pinned false
    # by name, the virtual_accelerator block armed. A rebuilt project refuses a
    # write on both hardware-shaped targets, so the staleness advisory firing on
    # already-deployed projects is the correct signal. Every other preset
    # stands still: the edits to the root and read-only presets are comments,
    # which the hash does not see.
    # The twelfth move, and control-assistant's family alone: the root preset's
    # session baseline moved from the live stand-in to the sandbox simulator
    # (`control_system.type: virtual_accelerator`). A rebuilt project opens on a
    # different machine and gains the EPICS-family agent rules, so the
    # staleness advisory firing on already-deployed projects is the correct
    # signal. The six `extends` children inherit it; ariel-standalone,
    # channel-finder-standalone and hello-world stand still.
    # The thirteenth move, and every preset: none sets `model:`, so the
    # provider's default_model answers, and none names a composition tier. A
    # rebuilt project may run a different main model, so the staleness advisory
    # firing on already-deployed projects is the correct signal.
    # The fourteenth move, and control-assistant's family alone: the root preset
    # stopped pinning three helper agents to Claude ids, so every agent runs the
    # deployment's main model. A rebuilt project renders a different `model:`
    # line in those agents' frontmatter, so the staleness advisory firing on
    # already-deployed projects is the correct signal. The five `extends`
    # children inherit it.
    # The fifteenth move, and control-assistant's family alone: the root
    # preset keeps transcripts ten years (`claude_code.transcripts.retention_days:
    # 3650`) and deploys the record archive (`services.archive`). The five
    # `extends` children inherit both; they build no services, so only the
    # retention reaches them. ariel-standalone, channel-finder-standalone and
    # hello-world stand still.
    # The sixteenth move, and control-assistant's family alone: the root
    # preset turned on the tool-content gate and set the content limit, so a
    # rebuilt project's agent exports built-in tool output as span events. The
    # five `extends` children inherit it; ariel-standalone,
    # channel-finder-standalone and hello-world stand still.
    # The seventeenth move, and control-assistant's family alone: the root
    # preset turns on the full tool-call record (`audit.tool_call.*`), which
    # the five `extends` children inherit; the other three stand still.
    "ariel-standalone": ("sha256:e430af35441251fbc5fb24ddd87175b18341919a5bae8ceb5788a96a86faeece"),
    "channel-finder-standalone": (
        "sha256:b96693984048dec0897c6bab4a3a16867b1e277037c0647930f40457965b1cdc"
    ),
    "control-assistant": (
        "sha256:f053c6de8fd9497d272a068a03f8174b9ed9c1d9c21c6fb800ef60b641a2f3ce"
    ),
    "control-assistant-admin": (
        "sha256:b27fb8d6fe3d79ba8ff5febdb28ccb70cbc117e3d8717b825002daa608694c57"
    ),
    "control-assistant-knowledge": (
        "sha256:53265269ecd689a2355c8c5bd892e68d022e0de03186ab4338104a805c54a6d0"
    ),
    "control-assistant-logbook": (
        "sha256:cffa27c611273350222d9a3e064002435c36f670f7db8d383d94639002726437"
    ),
    "control-assistant-readonly": (
        "sha256:251d61c019a064f0b5d3386d7744c2a68c35ca929f61caa9bd1cd1418b4f532a"
    ),
    "control-assistant-readwrite": (
        "sha256:88d62d2fdd672b4bf16d30ae9c72294e4b64f63a0819cdc0bc0e0be71a5746b6"
    ),
    "hello-world": ("sha256:ac89cdddebf7f249c0aab55057fce9b6872ff5d0de9679b12221814628e4c2e6"),
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
    raw = {"name": "Demo", "provider": "anthropic", "model": "claude-haiku-4-5"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "provider": "anthropic", "model": "claude-haiku-4-5"}
