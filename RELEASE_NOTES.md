# Osprey Framework - Latest Release (v2026.11.0)

**From a single-user assistant to a control-room deployment: sign-in and
tiers, an agent record, watchdogs and code review around every write, and
connectors on their own release stream.**

First stable release since 2026.6.2. These notes are written for an upgrade
from 2026.6.2; a site on a 2026.9.0 beta reads `CHANGELOG.md`, section
2026.11.0. Install with

    uv tool install osprey-framework

It requires `osprey-connectors>=2026.11.11`, released on its own version
stream; a deployment that pins versions pins both. CI runs Python 3.11, 3.12
and 3.14.

## What changed since 2026.6.2

- **A deployment is a profile.** `profile.yml`, `tiers/`, `apps/` and
  `data/facility/` are rendered by `osprey build` and run by `osprey up` as
  named containers behind a login service. The application presets and the
  `mock` control system are gone: the one stand-in is the virtual accelerator,
  served from its container or in process.
- **One write switch.** Hardware writes follow a three-position switch per
  machine (`off`, `asked`, `allowed`) under a tier ceiling. It is off on every
  machine until someone turns it on. The agent's tools, Python runs, notebooks,
  Bluesky lanes, dispatch and apps share one write verdict, and every refusal
  names its reason.
- **The machine is described once.** `data/facility/` drives the limits, the
  channel-finder index, the simulator, the knowledge graph and the pyAML
  configuration; hand-written copies of those views are refused.
  `osprey facility import mml` brings a Middle Layer export in.
- **A multi-user web terminal.** Sign-in by password, OIDC or token; each card
  on the landing page is a login file in `tiers/` that says what it may do and
  who may open it; a labelled sidebar of dockable panels; an operator chat; and
  an agent record that keeps every session, write and audit event, with an
  Activity page over it.
- **Apps.** A facility adds its own tools as `apps/<name>/` with one manifest:
  a backend container, a panel in the sidebar, tools and skills for the agent.
  Their writes pass the same tier, switch and approval as every other write.
  `osprey app new` and `osprey app check` make and check one.
- **Supervisors on the write path.** Safety watchdogs judge each operator
  prompt and can hold a session's writes; a second agent can review the agent's
  Python before the operator is asked to approve it. Both are off by default.
- **More routes in and out.** Event dispatch with triggers, Bluesky plans
  through a queue server, chat bridges for Slack, Google Chat, Nextcloud Talk
  and Microsoft Teams, and the optimizer as its own package,
  `osprey-optimizer`, whose runs write under one bounded authorization a
  person approves.
- **Connectors are their own package.** `osprey-connectors` carries the EPICS,
  DOOCS, virtual accelerator and archiver adapters and is released separately.

## Upgrading from 2026.6.2

Run `/osprey:setup` (the OSPREY plugin) to migrate a deployment, or start from
the `demo` or `hello-world` preset. `osprey build` stops on most of the items
below and names the fix. The ones a 6.2 deployment meets first:

1. `osprey deploy`, `osprey claude` and `osprey config` are gone: use
   `osprey init`, `set`, `build`, `up`, `down` and `web`. `osprey ariel web`,
   `artifacts web` and `channel-finder web` are `osprey web`.
2. `app_template:` and the application presets are gone, and `project_name:`
   is required in the profile.
3. `control_system.type: mock` is retired: use `virtual_accelerator` with
   `serving: in_process`.
4. Writes are off after the upgrade. Set `control_system.writes` on the tiers
   that should write, then turn the switch on per machine. Raw client puts in
   Python (`caput`, `PV.put`, caproto, DOOCS, PyTango) are refused: use
   `osprey.runtime.write_channel` or `write_channels`.
5. API keys are no longer taken from the shell: write them into the project
   `.env` (`.env.example` is the template).
6. Choose `modules.web_terminals.auth.method`. `none` means open, and
   `osprey up` refuses an open deployment that exposes a shell, a launch token
   or the notebook panel.
7. Ports move to `deployment.port_base` (10000) and containers are named
   `<project>-nginx`, `-auth` and `-web-<user>`: update bookmarks, firewall
   rules and systemd units.
8. Connector imports move from `osprey.*` to `osprey_connectors`. `timeout`
   is `timeout_s`, and aioca replaces pyepics.
9. `facility-config.yml`, `data/mml/`, hand-written channel-finder databases
   and `data/channel_limits.json` are replaced by `data/facility/`.
10. Agent tools were renamed (`artifact_*`, `ask_channels`, `run_sql`), and the
    write result vocabulary changed (`confirm` replaces `verification_level`):
    update custom prompts, rules, allow-lists and anything that parses results.
11. ARIEL: run `osprey ariel migrate`, create the read-only database role and
    set `ARIEL_DB_READONLY_PASSWORD`.
12. The Tuning, Health and Lattice panels are removed.
13. `claude_code.models` is `claude_code.aliases`, and `claude_code.cli_version`
    must be one exact version.
14. After upgrading, rebuild rendered pyAML views with `osprey build`; a view
    written before fails to load.
15. A registry mirror must carry `library/node:22-bookworm-slim`
    (`images.base_registry`): the images take Node 22 from it.
16. Run `osprey vendor fetch` to refresh the vendored browser libraries of an
    existing checkout (DOMPurify and KaTeX security updates).

The full list, each with its migration, is under **Breaking changes** in
`CHANGELOG.md`, sections 2026.9.0b1 to 2026.11.0.

## Known limits

- The write switch does not yet govern PVAccess puts (p4p `Context.put`, pvaPy
  `Channel.put`) from the Python executor and notebooks: they keep their
  approval ask and limits check.
- External-worker Bluesky lanes, and plans that reach a queue other than
  through OSPREY, are not governed by the write switch: they run at the lane's
  ceiling.
- Scheduled and triggered dispatch jobs cannot write.

## Not tested for this release

- The MATLAB Middle Layer export parity tests (no licensed MATLAB on a test
  host).
- The vLLM route of the OpenAI-compatible provider (no vLLM server).
- One agent-record request answered 401 in a single local test run; it did not
  recur and is not explained.

See `CHANGELOG.md`: section 2026.11.0 for what changed since 2026.9.0b3.
