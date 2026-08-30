"""Pin: the YAML-surface rename must not move any bundled preset's hash.

``preset_hash`` is stamped into ``.osprey-manifest.json`` at build time and
compared by the deploy-side staleness advisory, so a hash that moves for a
purely cosmetic key rename would report drift on every already-deployed
project. The digests below were recorded from the bundled presets *before* the
``data_bundle`` → ``app_template`` YAML rename; they are the durable evidence
that the rename stayed hash-neutral, and they are deliberately hardcoded rather
than recomputed — a test that derives its expectation from the code under test
would pin nothing.

Any change to these values means a preset's *resolved content* changed. That is
a legitimate thing to do, but it is a deploy-visible event: update the digest
here in the same commit, knowingly.
"""

from __future__ import annotations

from osprey.cli.build_profile import compute_preset_hash, compute_profile_hash, list_presets
from osprey.cli.build_profile_merge import _hash_resolved_profile

# preset name -> resolved-content hash, pre-rename.
PINNED_PRESET_HASHES: dict[str, str] = {
    # Moved when the graph tools left the main agent for the new
    # facility-knowledge-graph subagent: this preset's `agents:` went from the
    # empty list to naming that one agent. NOT behavior-neutral — a rebuilt
    # ARIEL deployment answers structural questions through delegation instead
    # of direct mcp__graph__* calls — so the staleness advisory firing on
    # already-deployed projects is the correct signal.
    "ariel-standalone": "sha256:ae72c2982cf20738d0d6783531e3619fa4fcccce13199b25056c4db4ce4e51f9",
    "channel-finder-standalone": (
        "sha256:71c5399c9ff3f181c1998f2e35d2cdb65a29efc499dcbae73f0d4a0982544f3a"
    ),
    # A digest here is the resolved content of the preset AND of every preset
    # that extends it, so a change in the base moves every control-assistant
    # entry with it. Last moved when the bluesky-web sidecar shed its
    # panels-era name, a deploy-visible change (the panel's backing service
    # and its config keys are renamed). Comment rewrites cannot move a digest
    # (`_hash_resolved_profile` hashes resolved canonical JSON and says so);
    # only a key or value change can. Some such changes are behavior-neutral
    # and some are not — a rebuilt project can gain or lose a tab, a hook, a
    # health check or a skill directory — and in either case the deploy-side
    # staleness advisory firing on already-deployed projects is the correct
    # signal, not noise.
    # Moved when the control-assistant preset turned password login on for its
    # web terminals (auth stanza, ariel's `login: false`, demo passwords under
    # `env.defaults`). Every control-assistant entry moved together, as the
    # note above predicts.
    # Moved again when pymongo became a core OSPREY dependency and the preset
    # dropped its `dependencies: [pymongo>=4.0]` line. Behavior-neutral for a
    # rebuilt project — pymongo still lands in its venv, now from the base
    # install rather than the profile — but the generated pyproject.toml no
    # longer names it, so the advisory firing is correct.
    # Moved twice in one release window. The control-assistant tier gained the
    # `bluesky-plans` skill, so a rebuilt project grows a
    # `.claude/skills/bluesky-plans/` directory; and it gained
    # `landing.notices` and `landing.footer`, so its landing page grows a
    # collapsible "working safely" section and a footer line. Both are
    # deploy-visible, so the staleness advisory firing on already-deployed
    # projects is correct. `control-assistant-ariel` excludes the skill by
    # name, so only the notices moved its digest.
    # Moved again — and this time EVERY bundled preset moved, which is the
    # signature of a change to the shared hook list rather than to one tier.
    # `target-state` joined every preset's `hooks:`, shipping the stdlib
    # control-target state reader into `.claude/hooks/`. It is a library the
    # approval hook imports, not an event hook: selection is what copies a hook
    # file and docstring frontmatter is what wires one, and this module has no
    # frontmatter, so a rebuilt project gains the file and gains no wiring. The
    # advisory firing on already-deployed projects is correct — the rendered
    # `.claude/hooks/` really does grow a file.
    # Moved (with every tier extending it) when the base gained the
    # facility-knowledge-graph agent in its `agents:` list — the subagent that
    # now owns the graph tools. A rebuilt project grows
    # `.claude/agents/facility-knowledge-graph.md` and its CLAUDE.md roster
    # entry; deploy-visible, so the advisory firing is correct.
    # Re-recorded where the target-switch branch met main: the `target-state`
    # hook and the facility-knowledge-graph agent are both in the resolved
    # content now, so every digest below (bar channel-finder-standalone, on
    # whose resolved content both lines already agreed) is the merged value,
    # not either branch's own.
    #
    # Moved again — and the family grew a fifth member — when the base gained
    # its TIER FLOOR and the admin tier shipped. Two deploy-visible deltas,
    # both landing on the base and therefore on all four presets that extend
    # it:
    #
    #   * The floor itself. The base now denies
    #     `mcp__osprey_workspace__setup_patch` and pins
    #     `web.config_panel.enabled: false` and
    #     `web.scaffold_gallery.write_enabled: false`, so a rebuilt project's
    #     settings.json grows a deny entry and its config.yml flips two keys
    #     from the app template's permissive defaults to false:
    #     the agent loses its deployment-editing tool and the browser loses the
    #     Config panel and the gallery's editors.
    #   * The roster. A third login (`carol`, persona `admin`) and its
    #     `OSPREY_AUTH_PW_CAROL` default join the base's web-terminals block,
    #     alongside the `admin` persona catalog entry — so a rebuilt hosting
    #     project grows a landing card, a terminal container and a per-user
    #     port in every family.
    #
    # `control-assistant-admin` is the new entry, not a moved one: it is the
    # single tier that lifts the floor back off (`remove_deny` for the tool,
    # both web keys back to true) and adds the `setup-mode` skill.
    # Re-recorded again where the render-zone/posture branch met main: the
    # tier floor, the admin tier, the `target-state` hook and the
    # facility-knowledge-graph agent all sit in the resolved content now, so
    # the five control-assistant digests below are the merged value.
    # Re-recorded once more where the Reach Contract met the tiers: the admin
    # persona's `services.graphdb.port_host` pin left with its two siblings'
    # (the build projects every service address into an attached render), so
    # the four attached digests are what the merged presets hash to. The base
    # is NOT among them — it pinned no address to lose — and its digest below
    # is main's, unmoved, which is the check that the contract touched only
    # the attached tiers.
    # Moved once more, and this time ALL SIX control-assistant entries at once
    # — base and every tier — when the base preset gained
    # `virtual_accelerator.live_standin: 5074`. A rebuilt project now deploys a
    # second copy of the simulator on that port as its `live` target, so
    # `control_target_set live` rehearses the real go-live ritual instead of
    # refusing for want of a live machine. Deploy-visible in the plainest way
    # there is — the rebuilt deployment runs one more container and answers on
    # one more port — so the staleness advisory firing on already-deployed
    # control-assistant projects is the correct signal.
    # Moved when the default host-port layout was unified (per-user families to
    # round-hundred bases 9100/9200/../9700). The web-terminal base-port keys
    # in the base preset changed value, so every control-assistant entry moved
    # together, as the note above predicts. Deploy-visible: a rebuilt project's
    # terminals and companion panels bind the new numbers.
    # Moved again, and again ALL SIX at once, when the base preset gained
    # `modules.web_terminals.auth.session_lifetime: 43200`. That key is the one
    # thing `osprey web` reads out of this block, and it is also what every
    # multi-user terminal sets its session cookie from, so a signed-in browser
    # expires the same way in both shapes. The value is the default the render
    # already substituted for the absent key, so a rebuilt deployment's sessions
    # expire exactly as before — but the resolved content now names the
    # lifetime, so the staleness advisory firing on already-deployed
    # control-assistant projects is the correct signal.
    # Moved once more, ALL SIX at once again, when the live stand-in became a
    # control target of its own. The base preset's `control_system.type` is now
    # `live_standin`: a session opens on the stand-in soft IOC rather than on
    # the sandbox simulator, and `live` goes on meaning the machine a facility
    # authors under `epics:`. The base also states the strict limits pair
    # (`control_system.limits_checking.enabled: true` and
    # `...allow_unlisted_channels: false`) that switching onto either
    # hardware-shaped target requires. Deploy-visible twice over — a rebuilt
    # project's sessions start on a different machine, and a write to a channel
    # `data/channel_limits.json` does not list is refused rather than waved
    # through — so the staleness advisory firing on already-deployed
    # control-assistant projects is the correct signal.
    # Moved with the base above: `live_standin` baseline + strict limits pair.
    #
    # Moved once more — all six control-assistant entries together — when host
    # ports became `deployment.port_base` plus a fixed offset. The base
    # preset's `bluesky.port`/`tiled_port`, `bluesky_web.port` and the
    # web-terminals' `nginx_port`/`web_base_port`/`artifact_base_port`/
    # `ariel_base_port`/`lattice_base_port`/`channel_finder_base_port`
    # literals all dropped — the build now derives them from the port block —
    # and `virtual_accelerator.live_standin` went from a hardcoded
    # second-simulator port to `true`, letting the loader place it in the same
    # block. Deploy-visible: a rebuilt project's landing page, terminals,
    # companion panels, bluesky sidecar and live-standin simulator all bind to
    # the port-block addresses instead of the retired literals.
    #
    # The two changes above met in one merge, so the digests below are neither
    # branch's recorded value but the one the merged preset actually hashes to:
    # a `live_standin` baseline with the strict limits pair, placed on the port
    # block. Deploy-visible for both reasons at once.
    #
    # Moved once more, and all six control-assistant entries together again,
    # when the base preset gave the sandbox simulator a limits posture of its
    # own: `control_system.connector.virtual_accelerator.limits_checking.
    # enabled: true` and `...allow_unlisted_channels: true`. A per-type block
    # replaces the deployment-wide pair for that connector type as a whole, so
    # both leaves are stated. The strict pair itself is unchanged and still
    # governs the two hardware-shaped targets, and `live_standin` deliberately
    # gained no block of its own — a permissive one there would make
    # `control_target_set standin` refuse the switch this preset exists to
    # rehearse. Deploy-visible on the simulator: a write to a channel
    # `data/channel_limits.json` does not list is now allowed on the `va`
    # target where it was refused, so the staleness advisory firing on
    # already-deployed control-assistant projects is the correct signal.
    "control-assistant": "sha256:c8d4f280789d363bc418002b27e229c7fc80cedf6bac469cae42ace2318d2098",
    # Moved with the base above: `live_standin` baseline + strict limits pair.
    # Moved again with the base: permissive `virtual_accelerator` limits block.
    "control-assistant-admin": (
        "sha256:3f111428abad719585dfa50fc86f5f93fc1dd776ff648f1d99e16cdb8a10b84e"
    ),
    # Moved with the base above: `live_standin` baseline + strict limits pair.
    # Moved again with the base: permissive `virtual_accelerator` limits block.
    "control-assistant-ariel": (
        "sha256:25df29b2f388287941c9ce4a1a05531f2fffca06ea8e4229f66b40c9c820a85d"
    ),
    # The two operator tiers below moved together, and alone, when each gained
    # the single dotted key `services.graphdb.port_host: 7687` in its `config:`
    # block — the attached-render personas scaffold no services of their own, so
    # without it their terminals would dial the shipped default port rather than
    # the port the hosting deployment publishes its graph store on. The base
    # `control-assistant` and the `control-assistant-ariel` tier are untouched
    # (the change is in these two leaves, not in the base they extend), which is
    # why only two of the four digests above move here. NOT behavior-neutral: a
    # rebuilt operator terminal gains the `graph` MCP server and its tools, so
    # the deploy-side staleness advisory firing on already-deployed
    # operator-tier projects is the correct signal.
    # Moved once more where the graph work met main: these two leaves already
    # carried the `services.graphdb.port_host` key above, and main independently
    # re-recorded them for the landing notices. Both edits are in the resolved
    # content, so the digest here is neither branch's recorded value but the one
    # the merged preset actually hashes to. Deploy-visible for both reasons at
    # once, which is what the advisory should say.
    # Moved again — with `control-assistant-ariel` this time, all three leaves
    # and not the base — when each gained the single dotted key
    # `services.qmd.port: 8180`: the same attached-render reasoning as the graph
    # port above, for the qmd sidecar that hybrid logbook search dials. NOT
    # behavior-neutral: a rebuilt persona terminal's logbook search stops
    # failing with "no qmd sidecar is configured", so the deploy-side staleness
    # advisory firing on already-deployed persona projects is the correct
    # signal.
    # Moved again — all three leaves, not the base — when the Reach Contract
    # landed and every service-address pin left these presets: the
    # `services.graphdb.port_host` and `services.qmd.port` keys recorded
    # above, and (readwrite) the `web.panels.events.*`/`web.panels.bluesky.*`
    # URL pins. The build now copies each of those facts — and more: the
    # Postgres, the telemetry store, the bridge, the VA port — from the
    # hosting deployment's render into every attached persona, so the presets
    # state nothing an operator's port move could strand. Deploy-visible: a
    # rebuilt persona's rendered config gains the projected blocks, so the
    # staleness advisory firing on already-deployed persona projects is the
    # correct signal.
    # Moved once more, and this time READONLY ALONE, when write posture became
    # per connector type. The flat `control_system.writes_enabled` is only what
    # a type inherits when its own block says nothing, so a read-only tier that
    # pinned nothing else could be armed over by a per-type key added anywhere
    # in the chain; readonly now pins `control_system.connector.epics.
    # writes_enabled` and `…virtual_accelerator.writes_enabled` false beside it.
    # Deploy-visible in two ways, and the second is the one worth knowing:
    # a rebuilt read-only terminal's config.yml grows the two keys, AND its
    # `connector.epics` block goes from absent to present — which is what
    # `resolve_target` reads to derive the live machine, so the tier becomes
    # switch-capable in shape (still not switchable TO live, which needs a
    # probe channel the preset cannot guess). The advisory firing on
    # already-deployed read-only projects is correct.
    #
    # `control-assistant-readwrite` and `control-assistant-admin` did NOT move:
    # they write no per-type key, so every type still reads their flat `true`
    # and their resolved content is unchanged. Their comment rewrites in the
    # same commit contributed nothing, which is the digest being comment-blind
    # exactly as the note at the top of this table says.
    # Moved with the base above: `live_standin` baseline + strict limits pair.
    # Moved again with the base: permissive `virtual_accelerator` limits block.
    "control-assistant-readonly": (
        "sha256:d6680f9c8c1e37d249cd3852f293ea9b81ae2ddaa56d5779c76fb350358221ef"
    ),
    # Moved with the base above: `live_standin` baseline + strict limits pair.
    # Moved again with the base: permissive `virtual_accelerator` limits block.
    "control-assistant-readwrite": (
        "sha256:7c2c8c4ed559f160a225ad7c42078585592b7fece43927454526ac8c9782ef97"
    ),
    # New with per-target write posture, not a moved entry: the rung between
    # the two flat tiers, armed on the virtual accelerator alone. It pins the
    # flat key `false` — the posture a live machine's block inherits — and
    # `control_system.connector.virtual_accelerator.writes_enabled: true` over
    # it, so which machine the session is pointed at decides whether its writes
    # land.
    # Moved with the base above: `live_standin` baseline + strict limits pair.
    # Moved again with the base: permissive `virtual_accelerator` limits block.
    "control-assistant-va-readwrite": (
        "sha256:30aac62fd5b81dfaaaf3ac75c3358278aadd33e35de1cad74d3ecd37262d2704"
    ),
    # Moved when the onboarding rewrite dropped the `facility` rule. The
    # wholesale comment rewrite that shipped alongside it contributed nothing:
    # the digest is comment-blind, so the rule drop is the entire delta.
    # Moved again when the preset gained the `memory-guard` hook entry, so a
    # rebuilt project's PreToolUse chain now also gates Write/MultiEdit to
    # Claude memory files and NotebookEdit to the agent-data artifacts tree.
    # The comment-only fixes that shipped alongside it (correcting the
    # mislabelled memory-guard/writes-check comments in the other presets)
    # contributed nothing to any digest, including this one.
    # Moved again when the preset gained a live `mcp_servers.example_server`
    # block, so a rebuilt project launches the seeded example MCP server and
    # its `example_status` tool appears in the session.
    "hello-world": "sha256:fdd41e470ce46d49f206640e558eaab9e909ce034b6bcc456ff50e3edb1e0436",
}


def test_bundled_preset_set_is_pinned():
    """A new preset must be classified here before it ships.

    Without this the per-preset loop below would silently skip an unpinned
    preset, and the pin would degrade as presets are added.
    """
    assert list_presets() == sorted(PINNED_PRESET_HASHES)


def test_every_bundled_preset_hash_is_unchanged():
    """Every preset resolves to its pre-rename digest — the rename is hash-neutral."""
    actual = {name: compute_preset_hash(name) for name in list_presets()}
    assert actual == PINNED_PRESET_HASHES


def test_hash_is_neutral_to_the_yaml_surface_spelling(tmp_path):
    """Both spellings of the same profile hash identically.

    Exercises ``_hash_resolved_profile`` with dicts that did NOT pass through
    ``_read_profile_document``, which is the only way the YAML-surface spelling
    can still reach it — and the case the canonicalization exists for.
    """
    profile_path = tmp_path / "profile.yml"
    yaml_spelling = {"name": "Demo", "app_template": "hello_world"}
    field_spelling = {"name": "Demo", "data_bundle": "hello_world"}

    assert _hash_resolved_profile(yaml_spelling, profile_path) == _hash_resolved_profile(
        field_spelling, profile_path
    )


def test_hash_still_tracks_the_bundle_value(tmp_path):
    """Canonicalizing spellings must not collapse *different* bundles to one hash."""
    profile_path = tmp_path / "profile.yml"
    assert _hash_resolved_profile(
        {"name": "Demo", "app_template": "hello_world"}, profile_path
    ) != _hash_resolved_profile({"name": "Demo", "app_template": "ariel_standalone"}, profile_path)


def test_hashing_does_not_mutate_the_callers_dict(tmp_path):
    """The caller's dict survives hashing with its own spelling intact."""
    raw = {"name": "Demo", "app_template": "hello_world"}
    _hash_resolved_profile(raw, tmp_path / "profile.yml")
    assert raw == {"name": "Demo", "app_template": "hello_world"}


def test_normalization_happens_before_extends_resolution(tmp_path):
    """Pins the *placement* of the canonicalization, not just its presence.

    A mixed-spelling chain — parent on disk in the canonical spelling, caller
    dict in the YAML spelling with a *different* value — only merges child-wins
    if ``raw`` is canonicalized before ``_resolve_extends``. Normalize after the
    merge instead and the resolved dict carries both keys with conflicting
    values, which raises; both public entry points swallow that to ``None``,
    so the build would stamp no ``preset_hash`` and deploy-side staleness would
    silently stop comparing rather than fail loudly.

    The plain neutrality test above passes under either placement, so this is
    the one that holds the line.
    """
    (tmp_path / "parent.yml").write_text(
        "name: Parent\ndata_bundle: ariel_standalone\n", encoding="utf-8"
    )
    child_path = tmp_path / "child.yml"

    mixed = {"extends": "parent.yml", "name": "Child", "app_template": "hello_world"}
    canonical = {"extends": "parent.yml", "name": "Child", "data_bundle": "hello_world"}
    assert _hash_resolved_profile(mixed, child_path) == _hash_resolved_profile(
        canonical, child_path
    )

    # The same chain through the public entry point must produce a hash, never
    # the "cannot compare" None that a raise would collapse to.
    child_path.write_text(
        "extends: parent.yml\nname: Child\napp_template: hello_world\n", encoding="utf-8"
    )
    assert compute_profile_hash(child_path) is not None
