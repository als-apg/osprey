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
    # The eighteenth move, and the two presets that carry a keyword block:
    # ariel-standalone and control-assistant gained
    # `ariel.search_modules.keyword.settings.fuzzy_threshold: 0.3`, the
    # fuzzy-fallback similarity floor that was a literal in the keyword module.
    # The value is the one the module already applied, so a rebuilt project
    # behaves identically; the digest moves because the preset now states it.
    # The five `extends` children inherit it; channel-finder-standalone and
    # hello-world stand still.
    # The nineteenth move, and control-assistant's family alone: the root
    # preset states `control_system.target_switch.probe_timeout_s: 5` beside
    # the drain timeout, and the five `extends` children inherit it; the other
    # three stand still. 5 is the reader's default, so a rebuilt project
    # behaves as before.
    # The twentieth move, and the two presets that carry a text-embedding block:
    # ariel-standalone and control-assistant gained `max_input_tokens: 2048` on
    # the `nomic-embed-text` entry under
    # `ariel.enhancement_modules.text_embedding.models`, the input window the
    # embedding server applies. A rebuilt project cuts a longer entry to that
    # window where it used the 512-token default before, so the staleness
    # advisory firing on already-deployed projects is the correct signal. The
    # five `extends` children inherit it; channel-finder-standalone and
    # hello-world stand still.
    # The twenty-first move, and every preset: each states
    # `simulation.models: null`, the served-model list, whose null serves every
    # model in the facility file as the absent key does, so a rebuilt project
    # behaves identically; the digest moves because the preset now states it.
    # The twenty-second move, and control-assistant's family and hello-world:
    # the limits block names its mode. hello-world states
    # `control_system.limits_checking.mode: optional` in place of the boolean
    # it carried for the same posture; control-assistant states the same leaf
    # deployment-wide and drops its per-type `virtual_accelerator` limits
    # block, so one pair now covers every target, and the five `extends`
    # children inherit it. A rebuilt control-assistant project writes a channel
    # the limits file does not list on every target instead of refusing it on
    # the hardware-shaped ones, so the staleness advisory firing on
    # already-deployed projects is the correct signal. ariel-standalone and
    # channel-finder-standalone carry no limits block and stand still.
    # The twenty-third move, and the same seven: control-assistant and
    # hello-world no longer state `control_system.limits_checking.database_path`.
    # The build writes it into the render, naming the limits database it
    # renders from `data/facility/limits.yaml`, so a rebuilt project carries
    # the same path and reads its limits from the records of that file.
    # The twenty-fourth move, and control-assistant's family alone: the root
    # preset authors its knowledge pages inside the facility tree, so
    # `facility_knowledge.bundle_path` reads `data/facility/knowledge` where it
    # read `data/facility_knowledge`. A rebuilt project mounts and serves the
    # bundle from the new directory, so the staleness advisory firing on
    # already-deployed projects is the correct signal. The five `extends`
    # children inherit it; ariel-standalone, channel-finder-standalone and
    # hello-world carry no bundle and stand still.
    # The twenty-fifth move, and every preset but hello-world: the presets stop
    # stating the retired facility leaves. ariel-standalone and
    # channel-finder-standalone drop `facility.name`, whose display name is now
    # the facility identity's; channel-finder-standalone and control-assistant
    # drop `facility.ontology`, whose terminology tables now render from the
    # build's facts. Neither leaf had a reader, so a rebuilt project behaves as
    # before. The five `extends` children inherit control-assistant's change;
    # hello-world stated neither leaf and stands still.
    # The twenty-sixth move, and control-assistant's family alone: the root
    # preset spells the EPICS and virtual-accelerator call bound `timeout_s`,
    # the one key every control-system connector reads, and the five `extends`
    # children inherit it; the other three stand still. The value is unchanged.
    # The twenty-seventh move, and the four presets that reach no machine:
    # control-assistant-logbook and control-assistant-knowledge drop the
    # JUPYTER panel, whose kernels reach the control target, and they,
    # ariel-standalone and channel-finder-standalone state
    # `web.control_target_picker: false`; the logbook persona also pins the
    # epics and virtual_accelerator write keys off, as the knowledge persona
    # already did. A rebuilt project of any of the four has no picker, so the
    # advisory firing is correct. control-assistant and hello-world gained a
    # comment only, which moves no digest; the other five stand still.
    # The twenty-eighth move, and the two presets that run a graph store:
    # control-assistant and ariel-standalone stop spelling
    # `services.graphdb.ttl_path`, which the build now fills with the graph view
    # it writes from the facility file. A rebuilt project seeds its store from
    # that view, so the advisory firing is correct. The five `extends` children
    # inherit control-assistant's change; the other two stand still.
    # The twenty-ninth move, and control-assistant's family alone: the root
    # preset stops stating `facility.prefix`, whose last reader is gone —
    # container names and persona projects come from the project name. The
    # five `extends` children inherit the change; ariel-standalone and
    # channel-finder-standalone lose only a commented example and stand still.
    "ariel-standalone": ("sha256:8a35722846d7faa6b5f14512d89f383c33146acb481cb22c82c6f8d0f37c8d37"),
    "channel-finder-standalone": (
        "sha256:91e29784cba676d47fa479e63bd3ce03cf469d165b8cfd27aad32762e8c7bc87"
    ),
    "control-assistant": (
        "sha256:0b720503973b1a38053c1e11f55608e93ba6f03fac518c24ddba3e92d0cb9608"
    ),
    "control-assistant-admin": (
        "sha256:31b377c70c444106d0289b17276474cb0d6f307f168129c034827a7aa07832ad"
    ),
    "control-assistant-knowledge": (
        "sha256:d1d221bb514dc9aa8254d831fb9a35536e37360c19829bfdb42b3bc1df6f514a"
    ),
    "control-assistant-logbook": (
        "sha256:3ddfa8d668be3a06cfda5ccb2892b65503f062a80fa7f196b5012783183a4221"
    ),
    "control-assistant-readonly": (
        "sha256:e8fc9bef01920b6741174f07c86d0d7c55d8b285778c13374cc0bfcce7421d45"
    ),
    "control-assistant-readwrite": (
        "sha256:0be7b86d24dc5b9d4b06b3326e82535245f6b2d5b2e74e9c5be85d6a256811a0"
    ),
    "hello-world": ("sha256:79ec0599628ed246440d6796191262c6ea844ce92ad2bb2e006b356f3a9a9ee7"),
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
