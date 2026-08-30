#!/usr/bin/env python3
"""Config-key resurrection guard — executes ``scripts/config_key_manifest.yml``.

The manifest records, for every dotted key the five shipped ``config.yml.j2``
templates render, either the code fragment that reads it or the structural
reason it has no independent reader. It also records the keys this branch
deleted, the code sites that went with them, and the assets whose continued
existence is load-bearing.

This tool re-derives all of that against the working tree on every run.

Failure modes
-------------
1. ``unmapped-key``      a rendered key with no manifest disposition
2. ``evidence``          an ``evidence`` regex that no longer matches src/osprey
3. ``resurrection``      a ``deleted`` path back in the rendered union, in a
                         preset ``config:`` dotted override, or in the loader's
                         synthesized defaults
4. ``orphan-site``       an ``orphan_sites`` regex that matches again
5. ``parity``            an ``all-templates`` key missing from a template
6. ``panel-port``        a ``# osprey:panel-port`` marker count that drifted

plus the manifest's own consistency checks (covered-by chains, evidence
vacuity, phantom entries, governed sets, keeps, absent paths) and a self-test
that every Jinja conditional branch is still covered by the render matrix.

Regex engine
------------
``re.M`` is applied to EVERY pattern, not only ``multiline: true`` ones. The
manifest's regexes were authored and verified with grep, which is line-based,
so ``^`` means "start of line" throughout. Without ``re.M`` those patterns
anchor at the start of the whole file and silently report zero matches —
manufacturing exactly the cannot-fail check the back-test exists to detect.
``multiline: true`` means something else: the pattern SPANS newlines, so it
needs whole-file text rather than a line-at-a-time reader. Reading whole files
with ``re.M`` serves both.

Usage
-----
    python scripts/check_config_keys.py
    python scripts/check_config_keys.py --back-test          # dev-time only
    python scripts/check_config_keys.py --back-test <commit>

The back-test proves every orphan regex is falsifiable — it must match at the
recorded merge-base and not on the branch. It shells out to git and extracts
the base tree, so it is a developer-time check, not part of the CI run.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "config_key_manifest.yml"

REQUIRED_SECTIONS = ("keys", "deleted", "orphan_sites", "render_contexts")

# Reader evidence may live in the framework tree or in the extracted
# osprey-connectors workspace member (config/logger/connectors moved there;
# the src/osprey paths are compatibility shims with no reader bodies).
EVIDENCE_ROOTS = ("src/osprey", "packages/osprey-connectors/src/osprey_connectors")
PRESET_DIR = "src/osprey/profiles/presets"

#: Ledger 6.1 wired one ``# osprey:panel-port <name>`` stanza per panel-port
#: entry, and the rule the proposal states is a NAME correspondence with the
#: web-server registry — not an arithmetic one. Counting markers pins the wrong
#: property: a renamed marker keeps the count identical, and a newly registered
#: server that no template documents is invisible to a count. The names are
#: parsed and reconciled against FRAMEWORK_WEB_SERVERS, so the registry is the
#: source for WHICH panels exist.
PANEL_PORT_MARKER = "osprey:panel-port"
PANEL_PORT_MARKER_RE = re.compile(rf"{re.escape(PANEL_PORT_MARKER)}\s+([A-Za-z0-9_-]+)")

#: Which panels each bundle is supposed to document, from ledger 6.1's mapping.
#: The registry reconciliation below cannot replace this, because it reasons
#: about the UNION across templates: a non-reference bundle can drop a stanza
#: the reference bundle still carries, leaving coverage complete and every
#: registry claim satisfied while that bundle silently stops documenting a port
#: it serves. Measured, not assumed — ariel_standalone dropping ``ariel`` is
#: green under the union checks alone.
#:
#: Keyed on NAMES rather than the counts this replaced, so it also catches a
#: per-template rename. It is hand-maintained, but not unchecked: claims (a)
#: and (b) reconcile every name here against the live registry, so an entry
#: that drifts from FRAMEWORK_WEB_SERVERS fails rather than rotting quietly.
EXPECTED_PANEL_PORT_MARKERS: dict[str, set[str]] = {
    "src/osprey/templates/project/config.yml.j2": {"artifact"},
    "src/osprey/templates/apps/control_assistant/config.yml.j2": {
        "artifact",
        "ariel",
        "channel_finder",
        "lattice_dashboard",
        "okf",
        "system_health",
    },
    "src/osprey/templates/apps/hello_world/config.yml.j2": {"artifact"},
    "src/osprey/templates/apps/ariel_standalone/config.yml.j2": {"artifact", "ariel"},
    "src/osprey/templates/apps/channel_finder_standalone/config.yml.j2": {
        "artifact",
        "channel_finder",
    },
}

#: A bare-word ``evidence`` regex appearing in more than this many files proves
#: nothing — it would keep matching after its reader was deleted. See the
#: manifest header for the 62 first-draft regexes this threshold retired.
EVIDENCE_VACUITY_MAX_FILES = 30  # raised from 25: the posture and protected-set
# test suites (posture clamp/hook/connector, write gates) legitimately mention
# channel_write; the cap is a vacuity heuristic, not a budget

#: ``api.providers`` is a data-map: provider NAMES are user-extensible data,
#: so leaves below it are shape-checked, never enumerated.
PROVIDER_LEAF_RE = re.compile(r"^api\.providers\.[^.]+")

#: File separator for the concatenated per-root text. A line holding a single
#: NUL cannot appear inside a source file, so no pattern — including the one
#: ``multiline: true`` entry — can match across a file boundary.
_SEP = "\n\x00\n"

#: Read source text once per (tree, root) for the whole process. The evidence
#: scan asks ~120 questions of the same 22 MB and the test suite instantiates
#: the guard a dozen times over one tree; keys are absolute, so a temporary
#: tree can never collide with the repository.
_TEXT_CACHE: dict[tuple[str, str], list[tuple[str, str]]] = {}


@dataclass
class Failure:
    """One guard failure, tagged with the mode that produced it."""

    mode: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.mode}] {self.detail}"


@dataclass
class Result:
    """Outcome of a guard run."""

    failures: list[Failure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def walk_paths(node: Any, prefix: str = "") -> Iterator[str]:
    """Yield every dotted path in a nested mapping, parents included."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from walk_paths(value, path)


def prefixes(dotted: str) -> list[str]:
    """``a.b.c`` -> ``[a, a.b, a.b.c]``."""
    parts = dotted.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def template_id(rel_path: str) -> str:
    """``.../apps/hello_world/config.yml.j2`` -> ``hello_world``."""
    return Path(rel_path).parent.name


class ConfigKeyGuard:
    """Executes one manifest against one tree."""

    def __init__(self, repo_root: Path, manifest: dict[str, Any]) -> None:
        self.root = Path(repo_root)
        self.manifest = manifest
        self.result = Result()
        self._joined: dict[str, str] = {}
        self._union: dict[str, set[str]] | None = None
        self._rendered: dict[str, list[dict[str, Any]]] | None = None

    # ── plumbing ────────────────────────────────────────────────────────

    def fail(self, mode: str, detail: str) -> None:
        self.result.failures.append(Failure(mode, detail))

    def note(self, text: str) -> None:
        self.result.notes.append(text)

    @property
    def exclude_dirs(self) -> set[str]:
        rules = self.manifest.get("scan_rules") or {}
        return set(rules.get("exclude_dirs") or ["__pycache__"])

    def texts(self, rel_root: str) -> list[tuple[str, str]]:
        """(relative path, contents) for every readable file under *rel_root*.

        Cached per root: the evidence scan alone asks ~120 questions of the
        same 22 MB of source, and re-reading it per question is the difference
        between a two-second run and a two-minute one.
        """
        cache_key = (str(self.root), rel_root)
        if cache_key in _TEXT_CACHE:
            return _TEXT_CACHE[cache_key]
        base = self.root / rel_root
        out: list[tuple[str, str]] = []
        if base.is_dir():
            excluded = self.exclude_dirs
            for path in sorted(base.rglob("*")):
                if not path.is_file() or any(part in excluded for part in path.parts):
                    continue
                try:
                    out.append((str(path.relative_to(self.root)), path.read_text(errors="ignore")))
                except OSError:
                    continue
        elif base.is_file():
            out.append((rel_root, base.read_text(errors="ignore")))
        _TEXT_CACHE[cache_key] = out
        return out

    def joined(self, rel_root: str) -> str:
        if rel_root not in self._joined:
            self._joined[rel_root] = _SEP.join(text for _, text in self.texts(rel_root))
        return self._joined[rel_root]

    @staticmethod
    def site_roots(site: dict[str, Any]) -> list[str]:
        """An orphan site's scan roots — ``root`` is one path or a list.

        A site grows a second root when the code it guards moves to another
        tree (the connectors extraction moved the config loader out of
        src/osprey/utils, leaving a shim the regex can never match). Every
        root is scanned on the branch; the back-test SUMS hits across roots,
        because a root newer than the baseline commit had nothing to match.
        """
        root = site["root"]
        return list(root) if isinstance(root, list) else [root]

    @staticmethod
    def count_matches(pattern: str, text: str) -> int:
        return len(re.findall(pattern, text, re.M))

    @staticmethod
    def has_match(pattern: str, text: str) -> bool:
        return re.search(pattern, text, re.M) is not None

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=False
        )

    # ── rendering ───────────────────────────────────────────────────────

    def contexts(self) -> list[dict[str, Any]]:
        """The full render matrix: base context × every matrix cell.

        ``osprey_ports`` is computed rather than spelled in the manifest: the
        five templates read their default host ports from it, and the layout is
        the one place those numbers live. The default base is the right one
        here because these renders carry no ``deployment.port_base`` — a real
        build derives the mapping from the base it resolved.
        """
        from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

        rc = self.manifest["render_contexts"]
        base = dict(rc.get("base") or {})
        base.setdefault("osprey_ports", layout_ports(DEFAULT_PORT_BASE))
        matrix = rc.get("matrix") or {}
        out = []
        for cell in matrix.get("channel_finder_mode", [{}]):
            for deploy in matrix.get("deploy_services", [True]):
                for panels in matrix.get("selected_web_panels", [[]]):
                    ctx = dict(base)
                    if cell:
                        ctx["channel_finder_mode"] = cell["mode"]
                        ctx["default_pipeline"] = cell["mode"]
                        ctx.update(cell.get("flags") or {})
                    ctx["deploy_services"] = deploy
                    ctx["selected_web_panels"] = list(panels)
                    out.append(ctx)
        return out

    def _jinja_env(self):
        import jinja2

        rc = self.manifest["render_contexts"]
        undefined = getattr(jinja2, rc.get("undefined") or "ChainableUndefined")
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                [str(self.root), str(self.root / "src/osprey/templates")]
            ),
            undefined=undefined,
            keep_trailing_newline=True,
        )

    def rendered(self) -> dict[str, list[dict[str, Any]]]:
        """template id -> parsed config for every matrix cell."""
        if self._rendered is not None:
            return self._rendered
        env = self._jinja_env()
        contexts = self.contexts()
        out: dict[str, list[dict[str, Any]]] = {}
        for rel in self.manifest["render_contexts"]["templates"]:
            template = env.get_template(rel)
            out[template_id(rel)] = [
                yaml.safe_load(template.render(**ctx)) or {} for ctx in contexts
            ]
        self._rendered = out
        return out

    def union(self) -> dict[str, set[str]]:
        """Every rendered dotted path -> the templates that render it."""
        if self._union is not None:
            return self._union
        merged: dict[str, set[str]] = {}
        for name, configs in self.rendered().items():
            for cfg in configs:
                for path in walk_paths(cfg):
                    merged.setdefault(path, set()).add(name)
        self._union = merged
        return merged

    def render_one(self, name: str) -> dict[str, Any]:
        """Render a single template under the base context only."""
        env = self._jinja_env()
        for rel in self.manifest["render_contexts"]["templates"]:
            if template_id(rel) == name:
                ctx = self.contexts()[0]
                return yaml.safe_load(env.get_template(rel).render(**ctx)) or {}
        raise KeyError(f"no template named {name!r} in the manifest")

    # ── failure mode 1: unmapped rendered key ───────────────────────────

    def check_unmapped_keys(self) -> None:
        mapped = set(self.manifest["keys"])
        for path in sorted(self.union()):
            if path in mapped or PROVIDER_LEAF_RE.match(path):
                continue
            self.fail(
                "unmapped-key",
                f"{path} is rendered by {sorted(self.union()[path])} but has no manifest entry",
            )

    def check_phantom_keys(self) -> None:
        """A manifest entry no template renders any more — manifest rot."""
        union = self.union()
        for key, spec in self.manifest["keys"].items():
            if key in union:
                continue
            if isinstance(spec, dict) and spec.get("rendered") is False:
                continue
            self.fail("phantom-key", f"{key} is in the manifest but no template renders it")

    # ── failure mode 2: evidence ────────────────────────────────────────

    def check_evidence(self) -> None:
        text = "\n".join(self.joined(root) for root in EVIDENCE_ROOTS)
        for key, spec in self.manifest["keys"].items():
            if not isinstance(spec, dict):
                continue
            pattern = spec.get("evidence")
            if pattern is None:
                continue
            if not self.has_match(pattern, text):
                self.fail(
                    "evidence",
                    f"evidence for {key} no longer matches under {' or '.join(EVIDENCE_ROOTS)}: {pattern}",
                )

    def check_covered_by_chains(self) -> None:
        """Every ``covered-by`` must terminate at a real disposition."""
        keys = self.manifest["keys"]

        def terminus(key: str, seen: set[str]) -> str | None:
            if key in seen:
                return f"covered-by cycle at {key}"
            seen.add(key)
            spec = keys.get(key)
            if spec is None:
                return f"covered-by names {key}, which is not in the manifest"
            if not isinstance(spec, dict):
                return f"{key} has a malformed entry"
            if spec.get("evidence") or spec.get("data-map") or spec.get("forward-declared"):
                return None
            parent = spec.get("covered-by")
            if not parent:
                return f"{key} carries no disposition"
            return terminus(parent, seen)

        for key in keys:
            problem = terminus(key, set())
            if problem:
                self.fail("evidence", f"chain for {key}: {problem}")

    def check_evidence_vacuity(self) -> None:
        """A bare-word regex matching too many files proves nothing.

        Only literal patterns are measured: anything carrying regex syntax is
        an anchored fragment of a real reader by construction.
        """
        literals = {
            key: spec["evidence"]
            for key, spec in self.manifest["keys"].items()
            if isinstance(spec, dict)
            and isinstance(spec.get("evidence"), str)
            and not re.search(r"[\\()\[\]{}|^$*+?]", spec["evidence"])
        }
        if not literals:
            return
        counts = dict.fromkeys(literals, 0)
        for _path, text in (pair for root in EVIDENCE_ROOTS for pair in self.texts(root)):
            for key, word in literals.items():
                if word in text:
                    counts[key] += 1
        for key, word in literals.items():
            if counts[key] > EVIDENCE_VACUITY_MAX_FILES:
                self.fail(
                    "evidence",
                    f"vacuous evidence for {key}: {word!r} appears in {counts[key]} files "
                    f"(limit {EVIDENCE_VACUITY_MAX_FILES}) — it would stay green without its reader",
                )

    # ── failure mode 3: resurrection ────────────────────────────────────

    def preset_texts(self) -> list[tuple[str, str]]:
        return [
            (Path(rel).name, text) for rel, text in self.texts(PRESET_DIR) if rel.endswith(".yml")
        ]

    def preset_paths(self) -> dict[str, set[str]]:
        """Dotted paths reachable through any preset's ``config:`` block."""
        out: dict[str, set[str]] = {}
        for name, text in self.preset_texts():
            data = yaml.safe_load(text) or {}
            config = data.get("config") or {}
            for dotted, value in config.items():
                for prefix in prefixes(str(dotted)):
                    out.setdefault(prefix, set()).add(name)
                for nested in walk_paths(value, str(dotted)):
                    out.setdefault(nested, set()).add(name)
        return out

    def commented_preset_overrides(self, key: str) -> list[str]:
        """Preset files carrying *key* as a commented dotted override.

        A retired key documented as a commented example is still a retired key
        put back in front of the operator, which is what ``system.facility_name``
        was before this branch removed it.
        """
        pattern = rf"^\s*#\s*{re.escape(key)}\s*:"
        return [name for name, text in self.preset_texts() if self.has_match(pattern, text)]

    def loader_default_paths(self) -> set[str]:
        """Paths the config loader synthesizes when the file omits them."""
        from osprey.utils.config import ConfigBuilder

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "config.yml"
            probe.write_text("project_name: config-key-guard-probe\n")
            builder = ConfigBuilder(str(probe), load_env=False)
        return set(walk_paths(builder.configurable))

    def check_deleted(self) -> None:
        union = self.union()
        presets = self.preset_paths()
        loader = self.loader_default_paths()
        for key in self.manifest["deleted"]:
            if key in union:
                self.fail(
                    "resurrection",
                    f"{key} is deleted but rendered again by {sorted(union[key])}",
                )
            if key in presets:
                self.fail(
                    "resurrection",
                    f"{key} is deleted but present as a preset config override in "
                    f"{sorted(presets[key])}",
                )
            commented = self.commented_preset_overrides(key)
            if commented:
                self.fail(
                    "resurrection",
                    f"{key} is deleted but documented as a commented preset override in "
                    f"{commented}",
                )
            if key in loader:
                self.fail(
                    "resurrection",
                    f"{key} is deleted but the config loader still synthesizes it as a default",
                )

    # ── failure mode 4: orphan sites ────────────────────────────────────

    def check_orphan_sites(self) -> None:
        for key, sites in self.manifest["orphan_sites"].items():
            for site in sites:
                pattern = site["regex"]
                for root in self.site_roots(site):
                    if not (self.root / root).exists():
                        self.fail("orphan-site", f"root for {key} does not exist: {root}")
                        continue
                    hits = self.count_matches(pattern, self.joined(root))
                    if hits:
                        self.fail(
                            "orphan-site",
                            f"removed site for {key} matches {hits}x under {root}/: {pattern}",
                        )

    # ── failure mode 5: all-templates parity ────────────────────────────

    def check_parity(self) -> None:
        """Presence in all five, never value equality.

        ``deployed_services`` is present everywhere with deliberately different
        values; ``container_runtime`` is present everywhere with the same one.
        Presence and value-equality are separate properties and neither may be
        inferred from the other, so only presence is asserted here.
        """
        union = self.union()
        expected = {template_id(rel) for rel in self.manifest["render_contexts"]["templates"]}
        for key, spec in self.manifest["keys"].items():
            if not (isinstance(spec, dict) and spec.get("all-templates")):
                continue
            present = union.get(key, set())
            missing = expected - present
            if missing:
                self.fail(
                    "parity",
                    f"{key} is marked all-templates but is absent from {sorted(missing)}",
                )

    # ── failure mode 6: panel-port markers ──────────────────────────────

    def panel_port_markers(self) -> dict[str, set[str]]:
        """Template -> the panel-port NAMES its stanzas declare."""
        found: dict[str, set[str]] = {}
        for rel in self.manifest["render_contexts"]["templates"]:
            path = self.root / rel
            if not path.is_file():
                self.fail("panel-port", f"template missing: {rel}")
                continue
            found[rel] = set(PANEL_PORT_MARKER_RE.findall(path.read_text()))
        return found

    def check_panel_port_markers(self, registry_keys: set[str] | None = None) -> None:
        """Reconcile the stanza names against the web-server registry.

        Four claims, because a marker can drift in four directions: a stanza
        can name a panel that does not exist, a registered panel can go
        undocumented everywhere, the reference bundle can stop being the one
        place that documents all of them, and an individual bundle can lose (or
        rename) a stanza while the other three claims still hold — the first
        three all reason about the union, so none of them can see that.
        """
        if registry_keys is None:
            from osprey.registry.web import FRAMEWORK_WEB_SERVERS

            registry_keys = set(FRAMEWORK_WEB_SERVERS)
        markers = self.panel_port_markers()
        if not markers:
            return

        for rel, names in markers.items():
            invented = sorted(names - registry_keys)
            if invented:
                self.fail(
                    "panel-port",
                    f"{rel} declares panel-port stanzas for {invented}, which are not "
                    f"FRAMEWORK_WEB_SERVERS entries",
                )

        for rel, names in markers.items():
            expected = EXPECTED_PANEL_PORT_MARKERS.get(rel)
            if expected is None or names == expected:
                continue
            drift = []
            if expected - names:
                drift.append(f"no longer documents {sorted(expected - names)}")
            if names - expected:
                drift.append(f"has gained {sorted(names - expected)}")
            self.fail(
                "panel-port",
                f"{rel} {' and '.join(drift)}; the bundle is supposed to carry a stanza "
                f"for exactly {sorted(expected)}",
            )

        documented: set[str] = set().union(*markers.values())
        undocumented = sorted(registry_keys - documented)
        if undocumented:
            self.fail(
                "panel-port",
                f"no template carries a panel-port stanza for the registered "
                f"server(s) {undocumented}",
            )

        if not any(names >= registry_keys for names in markers.values()):
            self.fail(
                "panel-port",
                "no single template documents the full panel-port set "
                f"{sorted(registry_keys)}; the reference bundle is supposed to be the "
                f"one place an operator can see every panel port",
            )

    # ── manifest self-consistency ───────────────────────────────────────

    def check_branch_self_test(self) -> None:
        """Each Jinja conditional branch must still be covered by the matrix.

        Every key below lives inside exactly one branch. If the matrix ever
        stops exercising a branch, the union quietly shrinks and every check
        built on it weakens without going red — so the discriminating keys are
        asserted directly.
        """
        union = self.union()
        for branch, key in (self.manifest["render_contexts"].get("self_test_keys") or {}).items():
            if key not in union:
                self.fail(
                    "branch-self-test",
                    f"the render matrix no longer covers the {branch} branch: "
                    f"{key} is absent from the union",
                )

    def check_union_size(self) -> None:
        expected = self.manifest["render_contexts"].get("expected_union_size")
        actual = len(self.union())
        if expected is None or expected == actual:
            self.note(f"rendered union: {actual} paths")
            return
        self.note(
            f"rendered union: {actual} paths (manifest records {expected}, "
            f"delta {actual - expected:+d}) — informational, not a failure"
        )

    def check_provider_shape(self) -> None:
        """``api.providers`` is shape-checked per provider, never enumerated."""
        spec = self.manifest["keys"].get("api.providers") or {}
        shape = spec.get("key-shape") or {}
        required = shape.get("required") or []
        tiers = shape.get("models-tiers") or []
        for name, configs in self.rendered().items():
            for cfg in configs:
                providers = ((cfg.get("api") or {}).get("providers")) or {}
                for provider, block in providers.items():
                    block = block or {}
                    for field_name in required:
                        if field_name not in block:
                            self.fail(
                                "provider-shape",
                                f"{name}: provider {provider} is missing {field_name}",
                            )
                    models = block.get("models") or {}
                    missing = [tier for tier in tiers if tier not in models]
                    if missing:
                        self.fail(
                            "provider-shape",
                            f"{name}: provider {provider} maps no model for {missing}",
                        )

    def governed_tools(self, name: str) -> set[str]:
        """Tools whose PreToolUse hook runs the approval script, for one app."""
        from osprey.registry.mcp import resolve_servers

        section = self.manifest["governed_sets"]["method"]
        marker = section["approval_hook_marker"]
        cfg = self.render_one(name)
        ctx: dict[str, Any] = {}
        if cfg.get("channel_finder"):
            ctx["channel_finder_pipeline"] = cfg["channel_finder"].get("pipeline_mode")
        tools: set[str] = set()
        for server in resolve_servers(cfg.get("claude_code") or {}, ctx):
            if not server.get("enabled"):
                continue
            for rule in server.get("hooks_pre") or []:
                commands = [hook.get("command", "") for hook in (rule.get("hooks") or [])]
                if not any(marker in command for command in commands):
                    continue
                # Matchers are single tokens and `_` is a word character, so a
                # naive identifier findall returns the whole matcher. Anchor on
                # the mcp__<server>__<tool> shape instead.
                tools.update(
                    re.findall(r"mcp__[A-Za-z0-9_-]+?__([A-Za-z0-9_]+)", rule.get("matcher", ""))
                )
        return tools

    def check_governed_sets(self) -> None:
        """Two shipped comments make exhaustive claims; re-derive them."""
        section = self.manifest.get("governed_sets") or {}
        for name, claim in (section.get("claims") or {}).items():
            actual = self.governed_tools(name)
            if not actual:
                self.fail(
                    "governed-set",
                    f"{name}: extracted an EMPTY governed set — the matcher parser is broken, "
                    f"not the config",
                )
                continue
            expected = set(claim["tools"])
            if actual != expected:
                self.fail(
                    "governed-set",
                    f"{name}: the shipped comment claims {sorted(expected)} but the resolved "
                    f"servers govern {sorted(actual)}",
                )
        for name, tools in (section.get("reference_counts") or {}).items():
            if not isinstance(tools, list):
                continue
            actual = self.governed_tools(name)
            if actual != set(tools):
                self.fail(
                    "governed-set",
                    f"{name}: the approval-governed tool set moved from {sorted(tools)} "
                    f"to {sorted(actual)}",
                )

    def check_keeps(self) -> None:
        """Assets whose continued existence no grep gate protects."""
        for entry in self.manifest.get("keeps") or []:
            if not self.tracked_files(entry["path"]):
                self.fail(
                    "keeps",
                    f"load-bearing asset is gone: {entry['path']}",
                )

    def check_absent_paths(self) -> None:
        """Deletions only a path absence can assert.

        Absence is measured with ``git ls-files``, never ``Path.exists()``: a
        deleted package leaves a populated ``__pycache__/`` behind in any tree
        that ever imported it, so a filesystem check goes red on developer
        worktrees while CI stays green.
        """
        for entry in self.manifest.get("absent_paths") or []:
            tracked = self.tracked_files(entry["path"])
            if tracked:
                self.fail(
                    "absent-path",
                    f"{entry['path']} should be gone but still tracks {len(tracked)} files",
                )

    def tracked_files(self, rel_path: str) -> list[str]:
        proc = self.git("ls-files", "--", rel_path)
        return [line for line in proc.stdout.splitlines() if line.strip()]

    # ── back-test (developer-time) ──────────────────────────────────────

    def back_test(self, commit: str) -> None:
        """Prove every orphan regex is falsifiable against the merge-base.

        A regex matching zero times BOTH at the base and on the branch is a
        green light wired to nothing. Four of the manifest's first 22 were
        exactly that. Both sides are measured with this module's own matcher so
        the comparison cannot drift on engine semantics.
        """
        for key, sites in self.manifest["orphan_sites"].items():
            for site in sites:
                pattern = site["regex"]
                roots = self.site_roots(site)
                base_hits = sum(
                    self.count_matches(pattern, self._base_text(commit, root)) for root in roots
                )
                branch_hits = sum(self.count_matches(pattern, self.joined(root)) for root in roots)
                if base_hits < 1 or branch_hits != 0:
                    self.fail(
                        "back-test",
                        f"orphan regex for {key} matched {base_hits}x at {commit} and "
                        f"{branch_hits}x on the branch (needs >=1 and 0) — it cannot fail: "
                        f"{pattern}",
                    )
        for entry in self.manifest.get("absent_paths") or []:
            base = self._base_tracked(commit, entry["path"])
            if not base:
                self.fail(
                    "back-test",
                    f"absent_path {entry['path']} tracked no files at {commit} either — "
                    f"the assertion cannot fail",
                )

    def _base_text(self, commit: str, rel_root: str) -> str:
        """Joined text of *rel_root* at *commit*.

        Extracted with ``git archive`` into a temp tree and read back through
        the same reader the branch side uses, so both sides go through one
        regex engine. Shelling out to ``git grep`` instead would compare the
        manifest's Python regexes against git's matcher.
        """
        cache_key = f"\x00base\x00{commit}\x00{rel_root}"
        if cache_key in self._joined:
            return self._joined[cache_key]
        # A root that did not exist at *commit* contributes no base text: a
        # multi-root site may name a tree newer than the back-test baseline,
        # and its falsifiability is proved by the root(s) that DID exist.
        if self.git("cat-file", "-e", f"{commit}:{rel_root}").returncode != 0:
            self._joined[cache_key] = ""
            return ""
        with tempfile.TemporaryDirectory() as tmp:
            archive = subprocess.run(
                ["git", "archive", commit, "--", rel_root],
                cwd=self.root,
                capture_output=True,
                check=False,
            )
            if archive.returncode != 0:
                raise RuntimeError(
                    f"git archive {commit} -- {rel_root} failed: "
                    f"{archive.stderr.decode(errors='ignore').strip()}"
                )
            subprocess.run(
                ["tar", "-x", "-C", tmp], input=archive.stdout, check=True, capture_output=True
            )
            sub = ConfigKeyGuard(Path(tmp), self.manifest)
            text = sub.joined(rel_root)
        self._joined[cache_key] = text
        return text

    def _base_tracked(self, commit: str, rel_path: str) -> list[str]:
        proc = self.git("ls-tree", "-r", "--name-only", commit, "--", rel_path)
        return [line for line in proc.stdout.splitlines() if line.strip()]

    # ── driver ──────────────────────────────────────────────────────────

    def run(self, back_test: str | None = None) -> Result:
        self.check_unmapped_keys()
        self.check_phantom_keys()
        self.check_evidence()
        self.check_covered_by_chains()
        self.check_evidence_vacuity()
        self.check_deleted()
        self.check_orphan_sites()
        self.check_parity()
        self.check_panel_port_markers()
        self.check_branch_self_test()
        self.check_provider_shape()
        self.check_governed_sets()
        self.check_keeps()
        self.check_absent_paths()
        self.check_union_size()
        if back_test:
            self.back_test(back_test)
        return self.result


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not a mapping: {path}")
    missing = [section for section in REQUIRED_SECTIONS if section not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required sections: {missing}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--back-test",
        nargs="?",
        const="",
        default=None,
        metavar="COMMIT",
        help="also prove every orphan regex fires at the merge-base (developer-time; "
        "defaults to the commit recorded in the manifest)",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    commit = args.back_test
    if commit == "":
        commit = manifest["scan_rules"]["back_test_baseline"]["commit"]

    guard = ConfigKeyGuard(args.repo_root, manifest)
    result = guard.run(back_test=commit)

    for note in result.notes:
        print(note)
    if result.failures:
        print(f"\n{len(result.failures)} failure(s):")
        for failure in result.failures:
            print(f"  {failure}")
        return 1
    counts = (
        len(manifest["keys"]),
        len(manifest["deleted"]),
        sum(len(sites) for sites in manifest["orphan_sites"].values()),
    )
    print(
        f"config-key guard: OK ({counts[0]} keys, {counts[1]} deleted paths, "
        f"{counts[2]} orphan sites" + (f", back-tested against {commit}" if commit else "") + ")"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
