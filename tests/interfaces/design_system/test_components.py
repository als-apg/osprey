"""The shared component sheet (``static/css/components.css``) and its contract.

The panel patterns every surface draws the same way — buttons, field rows,
switch, segmented control, status chip, disclosure, card, tabs, status strip,
banner, browser layout, section label, step list, edge tab, machine map and
device rows — live in one hand-written stylesheet beside ``base.css``. A panel
opts in with one ``<link>`` after ``base.css``.

These tests pin what a panel author relies on:

- the control-height scale comes from the token generator (``core.json``),
  never from a hand edit of ``tokens.css``, and the sheet sizes its controls
  from it;
- every component class the design handoff (``DESIGN.md``) promises is
  defined in the sheet, documented in ``DESIGN.md``, and drawn in the
  reference panel authors copy from;
- the sheet reads only tokens the generator emits (or a local knob with a
  fallback), and names no app.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import osprey.interfaces.design_system as design_system_pkg
from osprey.interfaces.design_system.generator.emit_css import emit_css
from osprey.interfaces.design_system.generator.model import load_token_tree

_DS_ROOT = Path(design_system_pkg.__file__).parent
_STATIC = _DS_ROOT / "static"
_COMPONENTS_CSS = _STATIC / "css" / "components.css"
_TOKENS_CSS = _STATIC / "css" / "tokens.css"
_DESIGN_MD = _DS_ROOT / "DESIGN.md"
_REFERENCE_PANEL = _DS_ROOT / "panels" / "reference" / "index.html"
_TOKENS_DIR = _DS_ROOT / "tokens"

#: Every class a panel author may rely on, grouped by component. The first
#: class of each group is the component's root; DESIGN.md documents each
#: group and the reference panel draws every root.
COMPONENTS: dict[str, tuple[str, ...]] = {
    "button": ("btn", "btn-quiet", "btn-ghost", "btn-danger", "btn-sm", "btn-block", "btn-icon"),
    "field": (
        "param-row",
        "form-grid",
        "field-label",
        "field-name",
        "field-unit",
        "field-help",
        "field-input",
        "field-inline",
    ),
    "switch": ("switch", "switch-input", "switch-track", "switch-thumb", "switch-text"),
    "segmented": ("seg", "seg-count"),
    "chip": ("chip", "no-dot"),
    "disclosure": ("disclosure-btn", "folder", "folder-summary", "chev"),
    "card": ("card", "card-head", "card-title", "card-sub", "card-body", "card-foot"),
    "tabs": ("tabs", "tab", "tab-badge"),
    "status-strip": ("status-strip", "strip-kicker", "fact", "fact-value"),
    "banner": ("banner", "banner-info", "banner-ok", "banner-warn", "banner-err", "banner-text"),
    "browser": (
        "browser",
        "sidebar",
        "sidebar-head",
        "sidebar-foot",
        "search",
        "tree",
        "detail",
        "detail-head",
        "detail-title",
        "detail-body",
        "detail-foot",
    ),
    "section-label": ("section-label",),
    "step-list": (
        "step-list",
        "step",
        "step-row",
        "step-index",
        "step-name",
        "step-options",
    ),
    "edge-tab": ("edge-tab", "edge-label", "edge-dot"),
    "machine-map": ("machine-map", "sector-rows", "sector-row", "cell", "map-key"),
    "device-rows": ("dev-table", "dev-row", "dev-name", "dev-pos"),
}

#: The six states a status chip draws.
CHIP_STATES = ("done", "running", "pending", "failed", "skipped", "stale")


def _css() -> str:
    return _COMPONENTS_CSS.read_text(encoding="utf-8")


def _defined_classes(css: str) -> set[str]:
    """Every class name a selector in ``css`` mentions (comments stripped)."""
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    selectors = "".join(re.findall(r"([^{}]*)\{", body))
    return set(re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]*)", selectors))


# --- the control-height scale -----------------------------------------------------


def test_control_heights_are_emitted_by_the_generator() -> None:
    css = emit_css(load_token_tree(_TOKENS_DIR))
    assert "--control-h: 28px;" in css
    assert "--control-h-sm: 24px;" in css


def test_control_heights_are_in_the_committed_tokens_css() -> None:
    tokens = _TOKENS_CSS.read_text(encoding="utf-8")
    assert "--control-h: 28px;" in tokens
    assert "--control-h-sm: 24px;" in tokens


def test_controls_size_from_the_control_height_tokens() -> None:
    css = _css()
    for selector in (".btn", ".field-input", ".seg", ".search"):
        rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert rule, f"{selector} has no rule"
        assert "height: var(--control-h)" in rule.group(1), selector
    for selector in (".btn-sm", ".seg.sm", ".field-input.sm"):
        rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert rule, f"{selector} has no rule"
        assert "height: var(--control-h-sm)" in rule.group(1), selector


# --- the sheet itself -------------------------------------------------------------


@pytest.mark.parametrize("component", sorted(COMPONENTS))
def test_every_component_class_is_defined(component: str) -> None:
    missing = set(COMPONENTS[component]) - _defined_classes(_css())
    assert not missing, f"{component}: no rule for {sorted(missing)}"


@pytest.mark.parametrize("state", CHIP_STATES)
def test_every_chip_state_is_drawn(state: str) -> None:
    assert f'.chip[data-state="{state}"]' in _css()


def test_button_hover_mixes_the_accent_toward_the_text_colour() -> None:
    """Hover must show even in a theme where accent == accent-light."""
    assert "color-mix(in srgb, var(--color-accent) 86%, var(--text-primary) 14%)" in _css()


def test_sheet_reads_only_emitted_tokens_or_local_knobs_with_a_fallback() -> None:
    css = _css()
    emitted = set(re.findall(r"--([a-zA-Z0-9-]+)\s*:", _TOKENS_CSS.read_text(encoding="utf-8")))
    unresolved = sorted(
        name
        for name, rest in re.findall(r"var\(--([a-zA-Z0-9-]+)(\s*,)?", css)
        if name not in emitted and not rest
    )
    assert not unresolved, f"var() with no emitted token and no fallback: {unresolved}"


def test_sheet_names_no_app() -> None:
    """A shared sheet carries no selector scoped to one interface or panel."""
    body = re.sub(r"/\*.*?\*/", "", _css(), flags=re.S)
    for word in ("bluesky", "trajectory", "traj", "web-terminal", "ariel", "#"):
        selectors = "".join(re.findall(r"([^{}]*)\{", body))
        assert word not in selectors.lower(), word


# --- documentation and the reference panel ----------------------------------------


@pytest.mark.parametrize("component", sorted(COMPONENTS))
def test_design_md_documents_every_component(component: str) -> None:
    design = _DESIGN_MD.read_text(encoding="utf-8")
    section = design.split("## 9. Components", 1)
    assert len(section) == 2, "DESIGN.md has no '## 9. Components' section"
    for cls in COMPONENTS[component]:
        named = re.escape(cls) + r"(?![\w-])"
        as_selector = re.search(r"\." + named, section[1])
        in_markup = re.search(r'class="(?:[^"]*\s)?' + named, section[1])
        assert as_selector or in_markup, f"DESIGN.md §9 does not mention .{cls}"


def test_design_md_lists_the_control_height_lever() -> None:
    design = _DESIGN_MD.read_text(encoding="utf-8")
    assert "--control-h" in design.split("## 4.", 1)[0]


def test_reference_panel_links_the_component_sheet_after_base() -> None:
    html = _REFERENCE_PANEL.read_text(encoding="utf-8")
    base = html.index("/design-system/css/base.css")
    components = html.index("/design-system/css/components.css")
    assert components > base


@pytest.mark.parametrize("component", sorted(COMPONENTS))
def test_reference_panel_draws_every_component(component: str) -> None:
    html = _REFERENCE_PANEL.read_text(encoding="utf-8")
    root = COMPONENTS[component][0]
    if component == "machine-map":
        assert "<osprey-machine-map" in html
        return
    assert re.search(r'class="(?:[^"]*\s)?' + re.escape(root) + r'(?:\s[^"]*)?"', html), (
        f"reference panel draws no .{root}"
    )
