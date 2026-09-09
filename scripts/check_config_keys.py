#!/usr/bin/env python3
"""Config-key resurrection guard — executes ``osprey/profiles/config_key_manifest.yml``.

The manifest records, for every dotted key a deployment's rendered config
carries, either the code fragment that reads it or the structural reason it has
no independent reader. A rendered config now comes from two places, and the
guard reads both: the framework template
(``templates/project/config.yml.j2``), which writes only what the BUILD derives,
and the ``config:`` block of each shipped preset, which is the operator's own
document and holds everything else. The manifest also records the keys this
branch deleted, the code sites that went with them, and the assets whose
continued existence is load-bearing.

This tool re-derives all of that against the working tree on every run.

Failure modes
-------------
1. ``unmapped-key``      a rendered key with no manifest disposition
2. ``phantom-key``       the mirror of 1: a manifest entry nothing renders any
                         more, or one flagged ``rendered: false`` that a
                         source has started rendering again
3. ``evidence``          an ``evidence`` regex that no longer matches src/osprey
4. ``resurrection``      a ``deleted`` path back in the rendered union, in a
                         preset ``config:`` dotted override, or in the loader's
                         synthesized defaults
5. ``orphan-site``       an ``orphan_sites`` regex that matches again
6. ``parity``            an ``all-templates`` key missing from a preset
7. ``panel-port``        a ``# osprey:panel-port`` marker set that drifted
8. ``default``           a key with no ``default:``, a ``default: required``
                         that is not a posture-floor key (or a posture-floor
                         key without one), or a ``derived`` / ``no-fallback``
                         entry with no ``default_note``

plus the manifest's own consistency checks (covered-by chains, evidence
vacuity, governed sets, keeps, absent paths) and a self-test
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

#: The manifest ships INSIDE the package, beside ``providers.yml``, because
#: ``osprey config --defaults`` renders its ``default:`` column at run time and a
#: wheel carries only ``src/osprey``. This script is developer-time and runs from
#: a checkout, so it addresses the file by repo-relative path; the runtime reader
#: resolves the same file through ``importlib.resources`` (see
#: ``osprey.profiles.providers.packaged_catalog_path`` for the shape).
DEFAULT_MANIFEST = REPO_ROOT / "src" / "osprey" / "profiles" / "config_key_manifest.yml"

#: The manifest's own filename, excluded from every source scan below.
#:
#: Moving the file into the package put the ledger INSIDE the tree the guard
#: greps, and the ledger names every deleted key and quotes every reader
#: fragment. Unexcluded it makes both text checks answer themselves: eleven
#: orphan regexes matched their own `deleted:` entries, and — the silent half —
#: every `evidence` regex would have matched the entry that spells it, so a key
#: whose reader was deleted would stay green forever. Excluded by NAME rather
#: than by path so a guard pointed at a temporary tree behaves the same way.
MANIFEST_FILENAME = "config_key_manifest.yml"

REQUIRED_SECTIONS = ("keys", "deleted", "orphan_sites", "render_contexts")

# Reader evidence may live in the framework tree or in the extracted
# osprey-connectors workspace member (config/logger/connectors moved there;
# the src/osprey paths are compatibility shims with no reader bodies).
EVIDENCE_ROOTS = ("src/osprey", "packages/osprey-connectors/src/osprey_connectors")
PRESET_DIR = "src/osprey/profiles/presets"

#: The union's non-preset half: the framework template's renders, under one
#: name so a failure can say which side of the config a key came from.
FRAMEWORK_SOURCE = "framework"

#: Ledger 6.1 wired one ``# osprey:panel-port <name>`` stanza per panel-port
#: entry, and the rule the proposal states is a NAME correspondence with the
#: web-server registry — not an arithmetic one. Counting markers pins the wrong
#: property: a renamed marker keeps the count identical, and a newly registered
#: server that no preset documents is invisible to a count. The names are
#: parsed and reconciled against FRAMEWORK_WEB_SERVERS, so the registry is the
#: source for WHICH panels exist.
PANEL_PORT_MARKER = "osprey:panel-port"
PANEL_PORT_MARKER_RE = re.compile(rf"{re.escape(PANEL_PORT_MARKER)}\s+([A-Za-z0-9_-]+)")

#: Which panels each preset is supposed to document, carried over from ledger
#: 6.1's mapping when the app templates it annotated became these presets.
#:
#: The stanzas live in the PRESETS and not in the framework template because the
#: preset is the document an operator edits: ``osprey init`` copies it out as
#: ``profile.yml``, comments and all, while the framework template's output is
#: regenerated by every build and never hand-edited. A panel port is exactly the
#: kind of fact an operator has to be able to find in the file in front of them.
#:
#: The registry reconciliation below cannot replace this, because it reasons
#: about the UNION across presets: a non-reference preset can drop a stanza the
#: reference preset still carries, leaving coverage complete and every registry
#: claim satisfied while that preset silently stops documenting a port it
#: serves. Measured, not assumed — ariel-standalone dropping ``ariel`` is green
#: under the union checks alone.
#:
#: Keyed on NAMES rather than the counts this replaced, so it also catches a
#: per-preset rename. It is hand-maintained, but not unchecked: claims (a) and
#: (b) reconcile every name here against the live registry, so an entry that
#: drifts from FRAMEWORK_WEB_SERVERS fails rather than rotting quietly.
EXPECTED_PANEL_PORT_MARKERS: dict[str, set[str]] = {
    f"{PRESET_DIR}/control-assistant.yml": {
        "artifact",
        "ariel",
        "channel_finder",
        "lattice_dashboard",
        "okf",
        "system_health",
    },
    f"{PRESET_DIR}/hello-world.yml": {"artifact"},
    f"{PRESET_DIR}/ariel-standalone.yml": {"artifact", "ariel"},
    f"{PRESET_DIR}/channel-finder-standalone.yml": {"artifact", "channel_finder"},
}

#: A bare-word ``evidence`` regex appearing in more than this many files proves
#: nothing — it would keep matching after its reader was deleted. See the
#: manifest header for the 62 first-draft regexes this threshold retired.
EVIDENCE_VACUITY_MAX_FILES = 30  # raised from 25: the posture and protected-set
# test suites (posture clamp/hook/connector, write gates) legitimately mention
# channel_write; the cap is a vacuity heuristic, not a budget

#: ``api.providers`` was the first data-map: provider NAMES are user-extensible
#: data, so leaves below it are shape-checked, never enumerated. The rule is now
#: read off the manifest's own ``data-map: true`` entries rather than hardcoded
#: here — the preset surface brought three more (the web-terminal persona
#: catalog, the user roster's own subtree, and the web workspace's named panel
#: layouts), and a second hardcoded regex per data map is how the two
#: definitions drift.

#: The keys a build refuses to render without, checked over the rendered config
#: by ``_missing_posture_errors``. Every one of them decides a posture that has
#: no safe silent answer — which connector is spoken to, whether writes are
#: gated, whether the run is observable — so the manifest may not offer a
#: fallback for them, and ``default: required`` is the only reading allowed.
#: Spelled here rather than imported so the guard stays a standalone script.
POSTURE_FLOOR_KEYS = frozenset(
    {
        "control_system.type",
        "archiver.type",
        "approval.enabled",
        "approval.default_policy",
        "claude_code.telemetry.enabled",
        "hooks.debug",
    }
)

#: ``facility_knowledge.bundle_path`` joins them in the ``default:`` column
#: without being a posture key: the OKF panel is gated on the SECTION being
#: present, and a section that names no bundle has nothing to serve, so there
#: is no fallback to record either.
REQUIRED_DEFAULT_KEYS = POSTURE_FLOOR_KEYS | {"facility_knowledge.bundle_path"}

#: ``default:`` values that are not a literal, and what each one claims.
#:
#: ``required``     no fallback exists and none may be invented — see above.
#: ``n/a``          the leaf has no independent reader; the parent block's own
#:                  default covers it (the ``covered-by`` / ``data-map`` case).
#: ``derived``      the fallback is computed at read time — a port from the
#:                  layout, an image from the registry/tag axes, or another
#:                  config key this one falls through to.
#: ``no-fallback``  the reader supplies none: it raises, or the surface goes
#:                  off, or nothing reads the key at all yet.
#:
#: ``derived`` and ``no-fallback`` say nothing on their own, so each one must
#: carry a ``default_note:`` naming the derivation or the consequence. Without
#: that rule the two would be an escape hatch from reading the code, which is
#: the whole point of the column.
#:
#: ``required`` may carry one and need not: "no fallback exists" is a complete
#: answer, but a key whose admissible values are a closed set has nowhere else
#: to name them for a reader holding only the ledger. A literal default may
#: not: the literal is the whole answer, so prose beside one is an excuse for
#: it or a sign the sentinel is wrong.
DEFAULT_SENTINELS = frozenset({"required", "n/a", "derived", "no-fallback"})
DEFAULT_SENTINELS_NEEDING_NOTE = frozenset({"derived", "no-fallback"})
DEFAULT_SENTINELS_ALLOWING_NOTE = DEFAULT_SENTINELS_NEEDING_NOTE | {"required"}


def sentinel_lookalike(value: object) -> str | None:
    """The sentinel *value* was meant to be, when it is not exactly one.

    ``no_fallback``, ``Derived``, ``N/A`` are strings a reader takes for
    sentinels and every check here takes for literals — the worst of both, and
    silent: such a key would sail past the note rule and then be rendered by
    ``osprey config --defaults`` as though ``no_fallback`` were a real default
    value. Normalising the way a person would mistype tells them apart from a
    genuine literal, which is never one word from this vocabulary.
    """
    if not isinstance(value, str) or value in DEFAULT_SENTINELS:
        return None
    normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
    return normalized if normalized in DEFAULT_SENTINELS else None


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


def preset_id(rel_path: str) -> str:
    """``.../presets/hello-world.yml`` -> ``hello-world``."""
    return Path(rel_path).stem


def deep_merge(base: dict[str, Any], over: Any) -> dict[str, Any]:
    """*over* laid on *base*, mappings merged and everything else replaced."""
    merged = dict(base)
    if not isinstance(over, dict):
        return merged
    for key, value in over.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


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
                if path.name == MANIFEST_FILENAME:
                    continue  # the ledger is not source; see MANIFEST_FILENAME
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

    @property
    def framework_template(self) -> str:
        return self.manifest["render_contexts"]["framework_template"]

    @property
    def preset_rels(self) -> list[str]:
        return list(self.manifest["render_contexts"]["presets"])

    def framework_base(self) -> dict[str, Any]:
        """The context every framework render starts from.

        Three values are computed rather than spelled in the manifest, because
        each has exactly one home in the tree and a copy here would be the
        silent one when they diverge: ``osprey_ports`` comes from the layout
        (the default base is the right one — these renders carry no
        ``deployment.port_base``, and a real build derives the mapping from the
        base it resolved), ``builtin_panels`` from the panel registry, and
        ``provider_catalog`` from ``providers.yml``, which is the template's
        only source for ``api.providers``.
        """
        from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports
        from osprey.profiles.providers import load_provider_catalog
        from osprey.profiles.web_panels import BUILTIN_PANELS

        base = dict(self.manifest["render_contexts"].get("base") or {})
        base.setdefault("osprey_ports", layout_ports(DEFAULT_PORT_BASE))
        base.setdefault("builtin_panels", sorted(BUILTIN_PANELS))
        base.setdefault("provider_catalog", load_provider_catalog(self.root).entries)
        return base

    def contexts(self) -> list[dict[str, Any]]:
        """The framework render matrix: base context × every matrix cell.

        The cells are the template's own remaining branches — the three
        channel-finder pipeline modes, the ARIEL gate, and the web block's
        three switches. ``_enable_flags`` is imported rather than spelled so the
        mode → flag mapping cannot drift from the one the build applies.
        """
        from osprey.cli.templates.manager import _enable_flags

        base = self.framework_base()
        matrix = self.manifest["render_contexts"].get("matrix") or {}
        out = []
        for mode in matrix.get("channel_finder_mode", [None]):
            for ariel_on in matrix.get("ariel_server_on", [True]):
                for selection in matrix.get("web_selection", [{}]):
                    ctx = dict(base)
                    if mode:
                        ctx["channel_finder_mode"] = mode
                        ctx["default_pipeline"] = mode
                        ctx.update(_enable_flags(mode))
                    ctx["ariel_server_on"] = ariel_on
                    ctx["selected_web_panels"] = self._selected_panels(base, selection)
                    ctx["default_panel"] = selection.get("default_panel")
                    ctx["panel_presets"] = selection.get("panel_presets")
                    out.append(ctx)
        return out

    @staticmethod
    def _selected_panels(base: dict[str, Any], selection: dict[str, Any]) -> list[str]:
        """A ``web_selection`` cell's panel list; ``all`` means the registry."""
        panels = selection.get("panels")
        if panels == "all":
            return list(base["builtin_panels"])
        return list(panels or [])

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

    def framework_renders(self) -> list[dict[str, Any]]:
        """The framework template parsed once per matrix cell."""
        env = self._jinja_env()
        template = env.get_template(self.framework_template)
        return [yaml.safe_load(template.render(**ctx)) or {} for ctx in self.contexts()]

    def resolved_presets(self) -> dict[str, tuple[Any, dict[str, Any]]]:
        """preset id -> (resolved profile, its ``config:`` as a nested mapping).

        Resolved through the real resolver, so a persona-bearing preset is read
        exactly as ``osprey build`` reads it, and the ``config:`` block arrives
        in whatever mix of dotted and nested spellings the file uses.

        :func:`layout_port_fill` is applied for the same reason the build
        applies it: a preset deliberately spells no host port, so without the
        fill the union would be missing every ``services.*`` port key that a
        real render carries — and the manifest entries for them would read as
        rot.
        """
        from osprey.cli.build_profile_archiver import _expand_dotted
        from osprey.cli.build_profile_ports import layout_port_fill
        from osprey.cli.build_profile_resolve import resolve_build_profile
        from osprey.port_layout import DEFAULT_PORT_BASE

        out: dict[str, tuple[Any, dict[str, Any]]] = {}
        for rel in self.preset_rels:
            name = preset_id(rel)
            profile, _preset_dir = resolve_build_profile(None, name)
            overlay = dict(profile.config)
            for key, port in layout_port_fill(overlay, DEFAULT_PORT_BASE).items():
                overlay.setdefault(key, port)
            out[name] = (profile, _expand_dotted(overlay))
        return out

    def rendered(self) -> dict[str, list[dict[str, Any]]]:
        """source -> the configs it contributes to the union.

        Two kinds of source. ``framework`` is the framework template once per
        matrix cell. Each preset is its resolved ``config:`` block — the half of
        a rendered config the operator owns, which no template renders any more.
        """
        if self._rendered is not None:
            return self._rendered
        out: dict[str, list[dict[str, Any]]] = {FRAMEWORK_SOURCE: self.framework_renders()}
        for name, (_profile, config) in self.resolved_presets().items():
            out[name] = [config]
        self._rendered = out
        return out

    def union(self) -> dict[str, set[str]]:
        """Every rendered dotted path -> the sources that carry it."""
        if self._union is not None:
            return self._union
        merged: dict[str, set[str]] = {}
        for name, configs in self.rendered().items():
            for cfg in configs:
                for path in walk_paths(cfg):
                    merged.setdefault(path, set()).add(name)
        self._union = merged
        return merged

    def _preset_context(self, profile: Any) -> dict[str, Any]:
        """The framework-render context a build of *profile* would use."""
        from osprey.cli.build_cmd import _ariel_server_enabled
        from osprey.cli.templates.manager import _enable_flags

        ctx = self.framework_base()
        ctx["default_provider"] = profile.provider
        ctx["default_model"] = profile.model
        mode = profile.channel_finder_mode
        if mode:
            ctx["channel_finder_mode"] = mode
            ctx["default_pipeline"] = mode
            ctx.update(_enable_flags(mode))
        ctx["default_panel"] = profile.default_panel
        ctx["panel_presets"] = profile.panel_presets
        ctx["selected_web_panels"] = list(profile.web_panels or [])
        ctx["ariel_server_on"] = _ariel_server_enabled(profile)
        return ctx

    def render_one(self, name: str) -> dict[str, Any]:
        """One source's config, in the shape a build of it writes.

        For a preset that is the framework render made from ITS fields with its
        own ``config:`` over the top — deliberately more than the preset's
        contribution to the union, because the caller
        (:meth:`governed_tools`) hands the result to ``resolve_servers``, which
        reads ``channel_finder.pipeline_mode``: a derived key that lives on the
        framework side. Reading the ``config:`` block alone would resolve every
        preset's servers against a pipeline mode of ``None``.
        """
        if name == FRAMEWORK_SOURCE:
            env = self._jinja_env()
            ctx = self.contexts()[0]
            return yaml.safe_load(env.get_template(self.framework_template).render(**ctx)) or {}
        resolved = self.resolved_presets()
        if name not in resolved:
            raise KeyError(f"no preset named {name!r} in the manifest")
        profile, config = resolved[name]
        env = self._jinja_env()
        rendered = (
            yaml.safe_load(
                env.get_template(self.framework_template).render(**self._preset_context(profile))
            )
            or {}
        )
        return deep_merge(rendered, config)

    # ── failure mode 1: unmapped rendered key ───────────────────────────

    def data_map_prefixes(self) -> set[str]:
        """Keys whose leaves are user-extensible data, not schema.

        A path below one of these is deliberately unenumerated: provider names,
        persona names, per-user subtrees and named panel layouts are all things
        a deployment invents, so demanding a manifest entry for each would
        demand one for data the shipped presets happen to carry.
        """
        return {
            key
            for key, spec in self.manifest["keys"].items()
            if isinstance(spec, dict) and spec.get("data-map")
        }

    def check_unmapped_keys(self) -> None:
        mapped = set(self.manifest["keys"])
        data_maps = self.data_map_prefixes()
        for path in sorted(self.union()):
            if path in mapped or any(path.startswith(f"{prefix}.") for prefix in data_maps):
                continue
            self.fail(
                "unmapped-key",
                f"{path} is rendered by {sorted(self.union()[path])} but has no manifest entry",
            )

    # ── failure mode 2: phantom manifest key ────────────────────────────

    def check_phantom_keys(self) -> None:
        """A manifest entry no source carries any more — manifest rot.

        ``rendered: false`` is the escape, and it says something specific: the
        key has a real reader and a real spelling, but nothing in the shipped
        configuration writes it — it is documented as a commented example in a
        preset, injected by the build from another declaration, or authored per
        facility. It is not a way to keep a dead key in the manifest.

        The flag is read in BOTH directions, and the mirror arm is the one that
        matters. Skipping the flag only when the key is absent makes it a
        one-way escape: a key that later starts being rendered keeps a marking
        that says nothing renders it, and the prose beside it goes on describing
        a commented example while a preset ships the key live. That is manifest
        rot in exactly the direction this check exists to catch, and it is the
        silent direction — nothing else in the guard reads the flag.
        """
        union = self.union()
        for key, spec in self.manifest["keys"].items():
            flagged = isinstance(spec, dict) and spec.get("rendered") is False
            if key in union:
                if flagged:
                    self.fail(
                        "phantom-key",
                        f"{key} is marked rendered: false but {sorted(union[key])} renders it: "
                        f"drop the flag and say what ships it",
                    )
                continue
            if flagged:
                continue
            self.fail("phantom-key", f"{key} is in the manifest but nothing renders it")

    # ── failure mode 3: evidence ────────────────────────────────────────

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

    def check_defaults(self) -> None:
        """Every key states what a reader sees when the profile omits it.

        The presets are the operator-facing document now, so a key's *value* is
        visible but the answer to "what happens if I delete this line" is not.
        That answer lives in the reader, and this column is where it is written
        down — ``osprey config --defaults`` renders it, so the literals have to
        be YAML values that round-trip, not prose about them.

        Three claims:

        1. every ``keys`` entry carries ``default:`` — a column with holes in
           it is one that gets read as "this key has no default" instead of
           "nobody looked";
        2. ``default: required`` is spelled on exactly the posture floor plus
           ``facility_knowledge.bundle_path``. Both directions matter. A key
           that drops out of the floor while keeping ``required`` documents a
           refusal that no longer happens, and a floor key that loses
           ``required`` invites a fallback to be invented for something the
           build refuses to guess at;
        3. ``derived`` and ``no-fallback`` carry a ``default_note:``,
           ``required`` may carry one, a literal carries none, and a near-miss
           spelling of a sentinel (``no_fallback``, ``Derived``) is refused
           rather than waved through as the literal string it technically is.
        """
        keys = self.manifest["keys"]
        for key, spec in keys.items():
            if not isinstance(spec, dict):
                self.fail("default", f"{key} has a malformed entry and so no default")
                continue
            if "default" not in spec:
                self.fail("default", f"{key} states no default: what does a reader see without it?")
                continue
            value = spec["default"]
            meant = sentinel_lookalike(value)
            if meant is not None:
                self.fail(
                    "default",
                    f"{key} spells its default {value!r}, which is read as a literal but "
                    f"means {meant!r} — spell the sentinel exactly",
                )
                continue
            needs_note = isinstance(value, str) and value in DEFAULT_SENTINELS_NEEDING_NOTE
            may_note = isinstance(value, str) and value in DEFAULT_SENTINELS_ALLOWING_NOTE
            has_note = bool(str(spec.get("default_note") or "").strip())
            if needs_note and not has_note:
                self.fail(
                    "default",
                    f"{key} is {value!r} but carries no default_note naming the "
                    f"derivation or the consequence",
                )
            elif has_note and not may_note:
                # A note on a literal reads as an excuse for it. The literal is
                # the whole answer, or it is the wrong sentinel.
                self.fail(
                    "default",
                    f"{key} carries a default_note but its default is the literal "
                    f"{value!r}; a note belongs only on "
                    f"{sorted(DEFAULT_SENTINELS_ALLOWING_NOTE)}",
                )

        declared = {
            key
            for key, spec in keys.items()
            if isinstance(spec, dict) and spec.get("default") == "required"
        }
        for key in sorted(declared - REQUIRED_DEFAULT_KEYS):
            self.fail(
                "default",
                f"{key} is marked 'default: required' but is not a posture-floor key — "
                f"record the reader's fallback instead",
            )
        for key in sorted(REQUIRED_DEFAULT_KEYS - declared):
            spec = keys.get(key)
            if spec is None:
                self.fail("default", f"posture-floor key {key} has no manifest entry at all")
                continue
            self.fail(
                "default",
                f"{key} is a posture-floor key but its default is "
                f"{spec.get('default')!r}, not 'required'",
            )

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

    # ── failure mode 4: resurrection ────────────────────────────────────

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

        Two keys are exempt, and the manifest says which and why
        (``deleted_commented_examples``). Both were removed from what OSPREY
        SHIPS while staying live in their readers — an explicitly set
        ``ariel.database.uri`` still wins over the derived DSN — and the
        commented line beside that prose is the documentation of an override
        that works, not a retired knob put back. The exemption is narrow: those
        two keys are still checked against the rendered union, against a LIVE
        preset override and against the loader's defaults, which is where an
        actual resurrection would show.
        """
        if key in (self.manifest.get("deleted_commented_examples") or {}):
            return []
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

    def check_deleted_commented_examples(self) -> None:
        """The exemptions above must stay tied to something real.

        Three ways one rots: it can name a key that was never deleted, it can
        carry no reason, and it can outlive the preset line it was written for —
        the last being the one that matters, because an unused exemption is a
        hole waiting for the next retired key that happens to share the name.
        """
        deleted = set(self.manifest["deleted"])
        for key, why in (self.manifest.get("deleted_commented_examples") or {}).items():
            if key not in deleted:
                self.fail(
                    "resurrection",
                    f"deleted_commented_examples exempts {key}, which is not on the deleted list",
                )
                continue
            if not str(why or "").strip():
                self.fail(
                    "resurrection", f"the commented-example exemption for {key} gives no reason"
                )
            pattern = rf"^\s*#\s*{re.escape(key)}\s*:"
            if not any(self.has_match(pattern, text) for _name, text in self.preset_texts()):
                self.fail(
                    "resurrection",
                    f"the commented-example exemption for {key} is unused — no preset documents "
                    f"it any more, so the exemption is a hole with nothing behind it",
                )

    # ── failure mode 5: orphan sites ────────────────────────────────────

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

    # ── failure mode 6: all-templates parity ────────────────────────────

    def check_parity(self) -> None:
        """Presence in all four PRESETS, never value equality.

        Parity is a claim about the operator's document, so the framework
        template is deliberately not in the expected set: it renders the derived
        keys and nothing else, and requiring a preset key to appear there too
        would ask for the second home this branch removed. Conversely a derived
        key cannot be marked ``all-templates`` at all — no preset spells one.

        ``deployed_services`` is present in all four with deliberately different
        values; ``container_runtime`` is present in all four with the same one.
        Presence and value-equality are separate properties and neither may be
        inferred from the other, so only presence is asserted here.
        """
        union = self.union()
        expected = {preset_id(rel) for rel in self.preset_rels}
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

    # ── failure mode 7: panel-port markers ──────────────────────────────

    def panel_port_markers(self) -> dict[str, set[str]]:
        """Preset -> the panel-port NAMES its stanzas declare."""
        found: dict[str, set[str]] = {}
        for rel in self.preset_rels:
            path = self.root / rel
            if not path.is_file():
                self.fail("panel-port", f"preset missing: {rel}")
                continue
            found[rel] = set(PANEL_PORT_MARKER_RE.findall(path.read_text()))
        return found

    def check_panel_port_markers(self, registry_keys: set[str] | None = None) -> None:
        """Reconcile the stanza names against the web-server registry.

        Four claims, because a marker can drift in four directions: a stanza
        can name a panel that does not exist, a registered panel can go
        undocumented everywhere, the reference preset can stop being the one
        place that documents all of them, and an individual preset can lose (or
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
                f"{rel} {' and '.join(drift)}; the preset is supposed to carry a stanza "
                f"for exactly {sorted(expected)}",
            )

        documented: set[str] = set().union(*markers.values())
        undocumented = sorted(registry_keys - documented)
        if undocumented:
            self.fail(
                "panel-port",
                f"no preset carries a panel-port stanza for the registered "
                f"server(s) {undocumented}",
            )

        if not any(names >= registry_keys for names in markers.values()):
            self.fail(
                "panel-port",
                "no single preset documents the full panel-port set "
                f"{sorted(registry_keys)}; the reference preset is supposed to be the "
                f"one place an operator can see every panel port",
            )

    # ── manifest self-consistency ───────────────────────────────────────

    def check_branch_self_test(self) -> None:
        """Each Jinja conditional branch must still be covered by the matrix.

        Every key below lives inside exactly one branch. If the matrix ever
        stops exercising a branch, the union quietly shrinks and every check
        built on it weakens without going red — so the discriminating keys are
        asserted directly.

        Presence is not enough, and the second arm is what makes the first one
        mean anything. A self-test key the PRESETS also spell stays in the union
        after the matrix stops rendering its branch, so the check would pass on
        the preset's copy and the shrunk matrix would go unnoticed — the exact
        silence this check exists to break. The framework template must
        therefore be the key's only author. Being under a ``DERIVED_KEYS``
        prefix is the usual reason a preset cannot supply one, but it is a
        property of today's presets rather than a rule, so the source set is
        checked instead of the prefix: ``web.panels.<id>.enabled`` is
        preset-spellable and framework-only only because no preset spells it.
        """
        union = self.union()
        for branch, key in (self.manifest["render_contexts"].get("self_test_keys") or {}).items():
            sources = union.get(key)
            if not sources:
                self.fail(
                    "branch-self-test",
                    f"the render matrix no longer covers the {branch} branch: "
                    f"{key} is absent from the union",
                )
            elif sources != {FRAMEWORK_SOURCE}:
                self.fail(
                    "branch-self-test",
                    f"the {branch} self-test key {key} must come from the framework template "
                    f"alone, or a preset spelling it would mask a branch the matrix stopped "
                    f"covering; the union has it from {sorted(sources)}",
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
        self.check_defaults()
        self.check_evidence_vacuity()
        self.check_deleted()
        self.check_deleted_commented_examples()
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
