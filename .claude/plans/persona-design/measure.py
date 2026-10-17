#!/usr/bin/env python
"""Measure the persona presets the way the build reads them.

Resolves every bundled preset through the real resolver
(``resolve_build_profile(None, <preset>)``, the path ``osprey build --preset``
takes) and the MCP server registry (``resolve_servers`` / ``resolve_agents``),
then diffs the pairs the persona design is about:

* the logbook pair      — ariel-standalone  vs  control-assistant-logbook
* the knowledge "pair"  — channel-finder-standalone  vs  control-assistant-knowledge
* the access-level axis — control-assistant-readonly / -readwrite / -admin

Nothing is read off the YAML directly. Run from the worktree root with the
worktree's sources on the path::

    PYTHONPATH=src:packages/osprey-connectors/src ~/.venv/bin/python \
        .claude/plans/persona-design/measure.py > .claude/plans/persona-design/MEASURE.md

The output is Markdown. ``--json`` prints the raw per-preset facts instead.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.build_profile_emit import effective_config_subtree
from osprey.cli.build_profile_presets import preset_data_bundle, preset_facility
from osprey.registry.mcp import resolve_agents, resolve_servers

LISTS = ("hooks", "rules", "skills", "agents", "output_styles", "web_panels")

# The render's derived context, as far as a resolved profile alone can tell:
# the channel-finder pipeline is keyed the way the render keys it; the graph
# server's gate (``graphdb_configured``) is whether the render will carry a
# ``services.graphdb`` block — the profile's own, or (attached) the one the build
# projects from the host; the dispatcher's gate is the EVENTS panel.
# ``served_decks`` / ``pyaml_view_present`` depend on the facility file and are
# left unset, so pyat-specialist reads as "off" for every preset here.


def _renders_graphdb(profile: Any) -> bool:
    return bool(profile._renders_graphdb_block())  # same predicate the validator uses


def _ctx(profile: Any, *, graphdb_configured: bool | None = None) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    if profile.channel_finder_mode and "channel-finder" in profile.agents:
        ctx["channel_finder_pipeline"] = profile.channel_finder_mode
    if graphdb_configured is None:
        graphdb_configured = _renders_graphdb(profile)
    ctx["graphdb_configured"] = graphdb_configured
    ctx["event_dispatcher_wired"] = "events" in profile.web_panels
    return ctx


def _servers(profile: Any, ctx: dict[str, Any]) -> list[str]:
    claude_code = effective_config_subtree(profile.config, ("claude_code",))
    resolved = resolve_servers(claude_code, ctx)
    names = [s["name"] for s in resolved if s["enabled"]]
    for name in profile.mcp_servers:
        if name not in names:
            names.append(name)
    return names


def _rendered_agents(profile: Any, ctx: dict[str, Any]) -> list[str]:
    """The subagents the render keeps: selected by the profile AND not dropped
    by the registry (a server dependency switched off, an unmet condition)."""
    claude_code = effective_config_subtree(profile.config, ("claude_code",))
    resolved_servers = resolve_servers(claude_code, ctx)
    enabled = {
        a["name"] for a in resolve_agents(claude_code, ctx, None, resolved_servers) if a["enabled"]
    }
    return [a for a in profile.agents if a in enabled]


def _config_leaves(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten the resolved ``config:`` bag to dotted leaves (lists stay whole)."""
    out: dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict) and node:
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        else:
            out[prefix] = node

    for key, value in config.items():
        walk(value, key)
    return out


def measure(name: str) -> dict[str, Any]:
    profile, _ = resolve_build_profile(None, preset=name)
    ctx = _ctx(profile)
    card_ctx = {
        k: v for k, v in ctx.items() if k == "channel_finder_pipeline"
    }  # what profile_card passes
    facts: dict[str, Any] = {
        "name": profile.name,
        "extends": None,
        "deploy_services": profile.deploy_services,
        "app_template": preset_data_bundle(name),
        "facility": preset_facility(name),
        "claude_md_template": profile.claude_md_template,
        "default_panel": profile.default_panel,
        "channel_finder_mode": profile.channel_finder_mode,
        "services": sorted(profile.services),
        "deployed_services": list(profile.config.get("deployed_services") or []),
        "renders_graphdb": _renders_graphdb(profile),
        "mcp_servers": _servers(profile, ctx),
        "mcp_servers_as_profile_card_sees_them": _servers(profile, card_ctx),
        "agents_rendered": _rendered_agents(profile, ctx),
        "config_leaves": _config_leaves(profile.config),
    }
    for field in LISTS:
        facts[field] = list(getattr(profile, field))
    import yaml

    from osprey.cli.build_profile_presets import _presets_dir

    raw = yaml.safe_load((_presets_dir() / f"{name}.yml").read_text(encoding="utf-8"))
    facts["extends"] = raw.get("extends")
    facts["exclude_entries"] = sum(
        len(v) for v in (raw.get("exclude") or {}).values() if isinstance(v, list)
    )
    facts["servers_switched_off"] = sorted(
        k.removeprefix("claude_code.servers.").removesuffix(".enabled")
        for k, v in (raw.get("config") or {}).items()
        if k.startswith("claude_code.servers.") and k.endswith(".enabled") and v is False
    )
    return facts


def _md_list(items: list[str]) -> str:
    return ", ".join(items) if items else "—"


def _pair_table(a: str, fa: dict[str, Any], b: str, fb: dict[str, Any]) -> list[str]:
    rows = [f"| Item | `{a}` | `{b}` | same? |", "|---|---|---|---|"]
    scalar = (
        "deploy_services",
        "app_template",
        "facility",
        "claude_md_template",
        "default_panel",
        "channel_finder_mode",
        "renders_graphdb",
        "exclude_entries",
    )
    for key in scalar:
        same = "yes" if fa[key] == fb[key] else "**no**"
        rows.append(f"| {key} | {fa[key]} | {fb[key]} | {same} |")
    for key in LISTS + (
        "agents_rendered",
        "mcp_servers",
        "services",
        "deployed_services",
        "servers_switched_off",
    ):
        sa, sb = fa[key], fb[key]
        if set(sa) == set(sb):
            rows.append(f"| {key} ({len(sa)}) | {_md_list(sa)} | same | yes |")
        else:
            only_a = [x for x in sa if x not in sb]
            only_b = [x for x in sb if x not in sa]
            rows.append(
                f"| {key} | {_md_list(sa)} | {_md_list(sb)} | **no** — only {a}: {_md_list(only_a)}; only {b}: {_md_list(only_b)} |"
            )
    la, lb = fa["config_leaves"], fb["config_leaves"]
    differing = sorted(
        k for k in set(la) | set(lb) if la.get(k, "<absent>") != lb.get(k, "<absent>")
    )
    rows.append(f"| config leaves | {len(la)} | {len(lb)} | {len(differing)} differ |")
    return rows


def _config_diff(
    a: str, fa: dict[str, Any], b: str, fb: dict[str, Any], *, prefixes_to_fold: tuple[str, ...]
) -> list[str]:
    la, lb = fa["config_leaves"], fb["config_leaves"]
    keys = sorted(k for k in set(la) | set(lb) if la.get(k, "<absent>") != lb.get(k, "<absent>"))
    folded: dict[str, int] = {}
    shown: list[str] = []
    for k in keys:
        head = next((p for p in prefixes_to_fold if k.startswith(p)), None)
        if head:
            folded[head] = folded.get(head, 0) + 1
        else:
            shown.append(k)
    out = [f"| key | `{a}` | `{b}` |", "|---|---|---|"]
    for k in shown:
        va = json.dumps(la.get(k, "<absent>"), default=str)
        vb = json.dumps(lb.get(k, "<absent>"), default=str)
        out.append(f"| `{k}` | {va[:60]} | {vb[:60]} |")
    for head, n in sorted(folded.items()):
        out.append(f"| `{head}*` ({n} leaves) | folded | folded |")
    return out


def main() -> None:
    names = list_presets()
    facts = {n: measure(n) for n in names}
    if "--json" in sys.argv:
        print(json.dumps(facts, indent=1, default=str))
        return
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(
        f"# Persona preset measurements\n\nTree `{sha}`; produced by `measure.py` through the real resolver.\n"
    )

    print("## Every bundled preset\n")
    print(
        "| preset | extends | deploy_services | bundle | brief | default panel | exclude entries | servers off | MCP servers (render) | MCP servers (profile card) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        f = facts[n]
        print(
            f"| `{n}` | {f['extends'] or '—'} | {f['deploy_services']} | {f['app_template']} | {f['claude_md_template'] or 'CLAUDE.md.j2'} | {f['default_panel'] or '—'} | {f['exclude_entries']} | {_md_list(f['servers_switched_off'])} | {_md_list(f['mcp_servers'])} | {_md_list(f['mcp_servers_as_profile_card_sees_them'])} |"
        )
    print()

    print("## Pair 1 — the logbook agent: `ariel-standalone` vs `control-assistant-logbook`\n")
    print(
        "\n".join(
            _pair_table(
                "ariel-standalone",
                facts["ariel-standalone"],
                "control-assistant-logbook",
                facts["control-assistant-logbook"],
            )
        )
    )
    print("\n### Config leaves that differ (deployment-shape blocks folded)\n")
    print(
        "\n".join(
            _config_diff(
                "ariel-standalone",
                facts["ariel-standalone"],
                "control-assistant-logbook",
                facts["control-assistant-logbook"],
                prefixes_to_fold=(
                    "services.",
                    "control_system.",
                    "approval.tools.",
                    "archiver.",
                    "ariel.ingestion",
                    "modules.web_terminals",
                    "bluesky",
                    "virtual_accelerator",
                    "dispatch",
                    "web.panels",
                    "claude_code.telemetry",
                    "ariel.search_modules",
                    "ariel.enhancement_modules",
                    "ariel.attachments",
                    "facility_knowledge",
                    "channel_finder",
                    "simulation",
                    "python_executor",
                    "limits",
                ),
            )
        )
    )
    print()

    print("## Pair 2 — `channel-finder-standalone` vs `control-assistant-knowledge`\n")
    print(
        "\n".join(
            _pair_table(
                "channel-finder-standalone",
                facts["channel-finder-standalone"],
                "control-assistant-knowledge",
                facts["control-assistant-knowledge"],
            )
        )
    )
    print()

    print("## The access-level axis — `control-assistant-readonly` / `-readwrite` / `-admin`\n")
    ro, rw, ad = (
        facts["control-assistant-readonly"],
        facts["control-assistant-readwrite"],
        facts["control-assistant-admin"],
    )
    print("| Item | readonly | readwrite | admin |")
    print("|---|---|---|---|")
    for key in LISTS + ("mcp_servers", "claude_md_template", "default_panel", "exclude_entries"):
        print(
            f"| {key} | {_md_list(ro[key]) if isinstance(ro[key], list) else ro[key]} | {_md_list(rw[key]) if isinstance(rw[key], list) else rw[key]} | {_md_list(ad[key]) if isinstance(ad[key], list) else ad[key]} |"
        )
    lro, lrw, lad = ro["config_leaves"], rw["config_leaves"], ad["config_leaves"]
    keys = sorted(
        k
        for k in set(lro) | set(lrw) | set(lad)
        if len({json.dumps(d.get(k, "<absent>"), default=str) for d in (lro, lrw, lad)}) > 1
    )
    print(f"\n{len(keys)} config leaves differ across the three:\n")
    print("| key | readonly | readwrite | admin |")
    print("|---|---|---|---|")
    for k in keys:
        print(
            f"| `{k}` | {json.dumps(lro.get(k, '<absent>'), default=str)[:50]} | {json.dumps(lrw.get(k, '<absent>'), default=str)[:50]} | {json.dumps(lad.get(k, '<absent>'), default=str)[:50]} |"
        )
    print()

    print("## P1.3 — what `osprey profile card` drops\n")
    print("| preset | render says | card says | missing on the card |")
    print("|---|---|---|---|")
    for n in names:
        f = facts[n]
        missing = [
            s for s in f["mcp_servers"] if s not in f["mcp_servers_as_profile_card_sees_them"]
        ]
        print(
            f"| `{n}` | {_md_list(f['mcp_servers'])} | {_md_list(f['mcp_servers_as_profile_card_sees_them'])} | {_md_list(missing)} |"
        )


if __name__ == "__main__":
    main()
