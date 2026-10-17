# Persona preset measurements

Tree `d026bb7f7`; produced by `measure.py` through the real resolver.

## Every bundled preset

| preset | extends | deploy_services | bundle | brief | default panel | exclude entries | servers off | MCP servers (render) | MCP servers (profile card) |
|---|---|---|---|---|---|---|---|---|---|
| `ariel-standalone` | — | True | ariel_standalone | CLAUDE.ariel.md.j2 | ariel | 0 | controls, python | osprey_workspace, ariel, osprey_facility_knowledge, graph | osprey_workspace, ariel, osprey_facility_knowledge |
| `channel-finder-standalone` | — | True | channel_finder_standalone | CLAUDE.channel-finder.md.j2 | channel-finder | 0 | ariel, controls, osprey_facility_knowledge, python | osprey_workspace, channel-finder | osprey_workspace, channel-finder |
| `control-assistant` | — | True | control_assistant | CLAUDE.md.j2 | — | 0 | — | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder |
| `control-assistant-admin` | control-assistant | False | control_assistant | CLAUDE.md.j2 | — | 0 | — | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph, event_dispatcher | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder |
| `control-assistant-knowledge` | control-assistant | False | control_assistant | CLAUDE.knowledge.md.j2 | okf | 23 | ariel, bluesky, controls, health, python | osprey_workspace, osprey_facility_knowledge, channel-finder, graph | osprey_workspace, osprey_facility_knowledge, channel-finder |
| `control-assistant-logbook` | control-assistant | False | control_assistant | CLAUDE.ariel.md.j2 | ariel | 25 | bluesky, channel-finder, controls, graph, health, osprey_facility_knowledge, python | osprey_workspace, ariel | osprey_workspace, ariel |
| `control-assistant-readonly` | control-assistant | False | control_assistant | CLAUDE.md.j2 | — | 0 | — | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder |
| `control-assistant-readwrite` | control-assistant | False | control_assistant | CLAUDE.md.j2 | — | 0 | — | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph, event_dispatcher | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder |
| `hello-world` | — | True | hello_world | CLAUDE.md.j2 | — | 0 | ariel, channel-finder | controls, python, osprey_workspace, osprey_facility_knowledge, example_server | controls, python, osprey_workspace, osprey_facility_knowledge, example_server |

## Pair 1 — the logbook agent: `ariel-standalone` vs `control-assistant-logbook`

| Item | `ariel-standalone` | `control-assistant-logbook` | same? |
|---|---|---|---|
| deploy_services | True | False | **no** |
| app_template | ariel_standalone | control_assistant | **no** |
| facility | example | example | yes |
| claude_md_template | CLAUDE.ariel.md.j2 | CLAUDE.ariel.md.j2 | yes |
| default_panel | ariel | ariel | yes |
| channel_finder_mode | None | graph | **no** |
| renders_graphdb | True | True | yes |
| exclude_entries | 0 | 25 | **no** |
| hooks (13) | hook-log, target-state, hook-config, approval, error-guidance, notebook-update, memory-guard, config-drift, focus-validate, panels-context, workspace-delta, control-context, turn-state | same | yes |
| rules | error-handling, artifacts, facility, workflows, timezone | safety, error-handling, artifacts, facility, workflows, timezone | **no** — only ariel-standalone: —; only control-assistant-logbook: safety |
| skills (2) | session-report, logbook-deep-research | same | yes |
| agents (1) | facility-knowledge-graph | same | yes |
| output_styles (1) | control-operator | same | yes |
| web_panels (1) | ariel | same | yes |
| agents_rendered | facility-knowledge-graph | — | **no** — only ariel-standalone: facility-knowledge-graph; only control-assistant-logbook: — |
| mcp_servers | osprey_workspace, ariel, osprey_facility_knowledge, graph | osprey_workspace, ariel | **no** — only ariel-standalone: osprey_facility_knowledge, graph; only control-assistant-logbook: — |
| services | — | archive | **no** — only ariel-standalone: —; only control-assistant-logbook: archive |
| deployed_services (5) | postgresql, telemetry_store, telemetry_gateway, qmd, graphdb | same | yes |
| servers_switched_off | controls, python | bluesky, channel-finder, controls, graph, health, osprey_facility_knowledge, python | **no** — only ariel-standalone: —; only control-assistant-logbook: bluesky, channel-finder, graph, health, osprey_facility_knowledge |
| config leaves | 79 | 145 | 72 differ |

### Config leaves that differ (deployment-shape blocks folded)

| key | `ariel-standalone` | `control-assistant-logbook` |
|---|---|---|
| `ariel.demo_narrative` | "all" | "<absent>" |
| `ariel.vocabulary.enabled` | false | true |
| `ariel.vocabulary.path` | "<absent>" | "data/ariel/vocabulary.yml" |
| `artifact_server.example_artifact` | "<absent>" | false |
| `audit.tool_call.enabled` | "<absent>" | true |
| `audit.tool_call.max_inline_bytes` | "<absent>" | 262144 |
| `claude_code.permissions.deny` | "<absent>" | ["mcp__osprey_workspace__setup_patch"] |
| `claude_code.servers.bluesky.enabled` | "<absent>" | false |
| `claude_code.servers.channel-finder.enabled` | "<absent>" | false |
| `claude_code.servers.graph.enabled` | "<absent>" | false |
| `claude_code.servers.health.enabled` | "<absent>" | false |
| `claude_code.servers.osprey_facility_knowledge.enabled` | "<absent>" | false |
| `claude_code.transcripts.retention_days` | "<absent>" | 3650 |
| `cli.theme` | "<absent>" | "default" |
| `deploy.fqdn` | "<absent>" | "127.0.0.1" |
| `web.channel_suggestions.enabled` | "<absent>" | true |
| `web.channel_suggestions.max_channels` | "<absent>" | 50000 |
| `web.config_panel.enabled` | true | false |
| `web.scaffold_gallery.write_enabled` | true | false |
| `web.theme` | "<absent>" | "light" |
| `web.ui_mode` | "<absent>" | "expert" |
| `approval.tools.*` (5 leaves) | folded | folded |
| `archiver.*` (2 leaves) | folded | folded |
| `bluesky*` (1 leaves) | folded | folded |
| `channel_finder*` (2 leaves) | folded | folded |
| `claude_code.telemetry*` (2 leaves) | folded | folded |
| `control_system.*` (18 leaves) | folded | folded |
| `facility_knowledge*` (1 leaves) | folded | folded |
| `modules.web_terminals*` (17 leaves) | folded | folded |
| `services.*` (1 leaves) | folded | folded |
| `simulation*` (1 leaves) | folded | folded |
| `web.panels*` (1 leaves) | folded | folded |

## Pair 2 — `channel-finder-standalone` vs `control-assistant-knowledge`

| Item | `channel-finder-standalone` | `control-assistant-knowledge` | same? |
|---|---|---|---|
| deploy_services | True | False | **no** |
| app_template | channel_finder_standalone | control_assistant | **no** |
| facility | example | example | yes |
| claude_md_template | CLAUDE.channel-finder.md.j2 | CLAUDE.knowledge.md.j2 | **no** |
| default_panel | channel-finder | okf | **no** |
| channel_finder_mode | hierarchical | graph | **no** |
| renders_graphdb | False | True | **no** |
| exclude_entries | 0 | 23 | **no** |
| hooks (14) | hook-log, target-state, hook-config, approval, error-guidance, notebook-update, memory-guard, config-drift, cf-feedback-capture, focus-validate, panels-context, workspace-delta, control-context, turn-state | same | yes |
| rules | error-handling, artifacts, facility, workflows, timezone | safety, error-handling, artifacts, facility, timezone | **no** — only channel-finder-standalone: workflows; only control-assistant-knowledge: safety |
| skills | session-report | — | **no** — only channel-finder-standalone: session-report; only control-assistant-knowledge: — |
| agents | channel-finder | channel-finder, facility-knowledge, facility-knowledge-graph | **no** — only channel-finder-standalone: —; only control-assistant-knowledge: facility-knowledge, facility-knowledge-graph |
| output_styles (1) | control-operator | same | yes |
| web_panels | channel-finder | channel-finder, okf | **no** — only channel-finder-standalone: —; only control-assistant-knowledge: okf |
| agents_rendered | channel-finder | channel-finder, facility-knowledge, facility-knowledge-graph | **no** — only channel-finder-standalone: —; only control-assistant-knowledge: facility-knowledge, facility-knowledge-graph |
| mcp_servers | osprey_workspace, channel-finder | osprey_workspace, osprey_facility_knowledge, channel-finder, graph | **no** — only channel-finder-standalone: —; only control-assistant-knowledge: osprey_facility_knowledge, graph |
| services | — | archive | **no** — only channel-finder-standalone: —; only control-assistant-knowledge: archive |
| deployed_services | — | postgresql, telemetry_store, telemetry_gateway, qmd, graphdb | **no** — only channel-finder-standalone: —; only control-assistant-knowledge: postgresql, telemetry_store, telemetry_gateway, qmd, graphdb |
| servers_switched_off | ariel, controls, osprey_facility_knowledge, python | ariel, bluesky, controls, health, python | **no** — only channel-finder-standalone: osprey_facility_knowledge; only control-assistant-knowledge: bluesky, health |
| config leaves | 21 | 143 | 128 differ |

## The access-level axis — `control-assistant-readonly` / `-readwrite` / `-admin`

| Item | readonly | readwrite | admin |
|---|---|---|---|
| hooks | hook-log, target-state, hook-config, approval, writes-check, limits, error-guidance, memory-guard, notebook-update, cf-feedback-capture, config-drift, focus-validate, panels-context, workspace-delta, control-context, turn-state | hook-log, target-state, hook-config, approval, writes-check, limits, error-guidance, memory-guard, notebook-update, cf-feedback-capture, config-drift, focus-validate, panels-context, workspace-delta, control-context, turn-state | hook-log, target-state, hook-config, approval, writes-check, limits, error-guidance, memory-guard, notebook-update, cf-feedback-capture, config-drift, focus-validate, panels-context, workspace-delta, control-context, turn-state |
| rules | safety, error-handling, artifacts, facility, workflows, timezone, python-execution, data-visualization, control-system-safety, test-ioc-safety | safety, error-handling, artifacts, facility, workflows, timezone, python-execution, data-visualization, control-system-safety, test-ioc-safety | safety, error-handling, artifacts, facility, workflows, timezone, python-execution, data-visualization, control-system-safety, test-ioc-safety |
| skills | diagnose, session-report, demo-gallery, demo-ui, writing-bluesky-plans, operating-bluesky-plans, bluesky-plans, pyaml-tools | diagnose, session-report, demo-gallery, demo-ui, writing-bluesky-plans, operating-bluesky-plans, bluesky-plans, pyaml-tools | diagnose, session-report, demo-gallery, demo-ui, writing-bluesky-plans, operating-bluesky-plans, bluesky-plans, pyaml-tools, setup-mode |
| agents | channel-finder, data-visualizer, logbook-search, logbook-deep-research, facility-knowledge, facility-knowledge-graph, pyat-specialist | channel-finder, data-visualizer, logbook-search, logbook-deep-research, facility-knowledge, facility-knowledge-graph, pyat-specialist | channel-finder, data-visualizer, logbook-search, logbook-deep-research, facility-knowledge, facility-knowledge-graph, pyat-specialist |
| output_styles | control-operator | control-operator | control-operator |
| web_panels | ariel, channel-finder, lattice, okf, system-health, jupyter | ariel, channel-finder, lattice, okf, system-health, jupyter, events, bluesky | ariel, channel-finder, lattice, okf, system-health, jupyter, events, bluesky |
| mcp_servers | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph, event_dispatcher | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph, event_dispatcher |
| claude_md_template | None | None | None |
| default_panel | None | None | None |
| exclude_entries | 0 | 0 | 0 |

5 config leaves differ across the three:

| key | readonly | readwrite | admin |
|---|---|---|---|
| `claude_code.permissions.remove_deny` | "<absent>" | "<absent>" | ["mcp__osprey_workspace__setup_patch"] |
| `control_system.connector.virtual_accelerator.writes_enabled` | false | true | true |
| `web.config_panel.enabled` | false | false | true |
| `web.scaffold_gallery.write_enabled` | false | false | true |
| `web.ui_mode` | "simple" | "expert" | "expert" |

## P1.3 — what `osprey profile card` drops

| preset | render says | card says | missing on the card |
|---|---|---|---|
| `ariel-standalone` | osprey_workspace, ariel, osprey_facility_knowledge, graph | osprey_workspace, ariel, osprey_facility_knowledge | graph |
| `channel-finder-standalone` | osprey_workspace, channel-finder | osprey_workspace, channel-finder | — |
| `control-assistant` | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder | graph |
| `control-assistant-admin` | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph, event_dispatcher | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder | graph, event_dispatcher |
| `control-assistant-knowledge` | osprey_workspace, osprey_facility_knowledge, channel-finder, graph | osprey_workspace, osprey_facility_knowledge, channel-finder | graph |
| `control-assistant-logbook` | osprey_workspace, ariel | osprey_workspace, ariel | — |
| `control-assistant-readonly` | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder | graph |
| `control-assistant-readwrite` | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder, graph, event_dispatcher | controls, python, osprey_workspace, ariel, osprey_facility_knowledge, bluesky, health, channel-finder | graph, event_dispatcher |
| `hello-world` | controls, python, osprey_workspace, osprey_facility_knowledge, example_server | controls, python, osprey_workspace, osprey_facility_knowledge, example_server | — |
