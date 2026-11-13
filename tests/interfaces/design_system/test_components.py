"""The panel components in ``static/css/primitives.css`` and their contract.

The panel patterns every surface draws the same way — buttons, field rows,
switch, segmented control, status chip, disclosure, card, status strip,
banner, section label, master-detail, step list, edge tab, machine map and
device rows — are one ``osprey-`` family in ``primitives.css``, beside the
panel chrome of DESIGN.md §12. There is no second sheet and no bare-named
copy of a control.

These tests pin what a panel author relies on:

- the control-height scale comes from the token generator (``core.json``),
  never from a hand edit of ``tokens.css``, and the controls size from it;
- every component selector is defined in the sheet, documented in
  DESIGN.md §13, and drawn in the reference panel authors copy from;
- every class root the sheet defines is an ``osprey-`` name;
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
_CSS_DIR = _DS_ROOT / "static" / "css"
_PRIMITIVES_CSS = _CSS_DIR / "primitives.css"
_TOKENS_CSS = _CSS_DIR / "tokens.css"
_DESIGN_MD = _DS_ROOT / "DESIGN.md"
_REFERENCE_PANEL = _DS_ROOT / "panels" / "reference" / "index.html"
_TOKENS_DIR = _DS_ROOT / "tokens"

#: Every selector a panel author may rely on, grouped by component. The first
#: entry of each group is the component's root; DESIGN.md §13 documents each
#: group and the reference panel draws every root.
COMPONENTS: dict[str, tuple[str, ...]] = {
    "button": (
        ".osprey-btn",
        ".osprey-btn.primary",
        ".osprey-btn.ghost",
        ".osprey-btn.danger",
        ".osprey-btn.sm",
        ".osprey-btn.block",
    ),
    "field": (
        ".osprey-form-grid",
        ".osprey-field",
        ".osprey-field.wide",
        ".osprey-field-label",
        ".osprey-field-name",
        ".osprey-field-unit",
        ".osprey-field-required",
        ".osprey-field-input",
        ".osprey-field-input.sm",
        ".osprey-field-inline",
    ),
    "switch": (".osprey-switch-label", ".osprey-switch-input", ".osprey-switch"),
    "segmented": (".osprey-seg", ".osprey-seg.block", ".osprey-seg.sm", ".osprey-seg-count"),
    "chip": (".osprey-chip", ".osprey-chip--plain"),
    "disclosure": (
        ".osprey-disclosure",
        ".osprey-disclosure.icon",
        ".osprey-folder",
        ".osprey-chev",
    ),
    "card": (
        ".osprey-card",
        ".osprey-card-head",
        ".osprey-card-title",
        ".osprey-card-sub",
        ".osprey-card-body",
        ".osprey-card-foot",
    ),
    "status-strip": (
        ".osprey-status-strip",
        ".osprey-fact",
        ".osprey-fact-value",
        ".osprey-kicker",
    ),
    "banner": (
        ".osprey-banner",
        ".osprey-banner--info",
        ".osprey-banner--ok",
        ".osprey-banner--warn",
        ".osprey-banner--err",
        ".osprey-banner-text",
    ),
    "section-label": (".osprey-section-label", ".osprey-aside"),
    "master-detail": (
        ".osprey-detail",
        ".osprey-panel-sidebar-head",
        ".osprey-panel-sidebar-foot",
        ".osprey-detail-head",
        ".osprey-detail-title",
        ".osprey-detail-desc",
        ".osprey-detail-body",
        ".osprey-detail-section",
        ".osprey-detail-foot",
        ".osprey-pending-line",
    ),
    "step-list": (
        ".osprey-step-list",
        ".osprey-step-list.no-switch",
        ".osprey-step",
        ".osprey-step-row",
        ".osprey-step-index",
        ".osprey-step-name",
        ".osprey-step-title",
        ".osprey-step-meta",
        ".osprey-step-options",
    ),
    "edge-tab": (".osprey-edge-tab", ".osprey-edge-label", ".osprey-edge-dot"),
    "machine-map": (
        ".osprey-map",
        ".osprey-map-key",
        ".osprey-map-rows",
        ".osprey-map-row",
        ".osprey-map-cell",
    ),
    "device-rows": (".osprey-dev-table", ".osprey-dev-row", ".osprey-dev-name", ".osprey-dev-pos"),
}

#: The run states a status chip draws, beside the connection states of §12.
CHIP_RUN_STATES = ("done", "running", "pending", "stale", "failed", "skipped")


def _css() -> str:
    return _PRIMITIVES_CSS.read_text(encoding="utf-8")


def _selectors(css: str) -> list[str]:
    """Every selector of every rule in ``css`` (comments and at-rule preludes out)."""
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    preludes = re.findall(r"([^{}]*)\{", body)
    out: list[str] = []
    for prelude in preludes:
        prelude = prelude.strip()
        if not prelude or prelude.startswith("@"):
            continue
        out.extend(part.strip() for part in prelude.split(","))
    return [s for s in out if s and not re.fullmatch(r"(from|to|\d+%)", s)]


def _defines(selector: str) -> bool:
    """True when some rule's selector names ``selector`` as a whole compound."""
    pattern = re.escape(selector) + r"(?![\w-])"
    return any(re.search(pattern, s) for s in _selectors(_css()))


def _rule(selector: str) -> str:
    match = re.search(r"(?:^|[},\s])" + re.escape(selector) + r"\s*\{([^}]*)\}", _css())
    assert match, f"{selector} has no rule of its own"
    return match.group(1)


def _design_section() -> str:
    parts = _DESIGN_MD.read_text(encoding="utf-8").split("## 13. Panel components", 1)
    assert len(parts) == 2, "DESIGN.md has no '## 13. Panel components' section"
    return parts[1]


# --- the control-height scale -----------------------------------------------------


def test_control_heights_are_emitted_by_the_generator() -> None:
    css = emit_css(load_token_tree(_TOKENS_DIR))
    assert "--control-h: 28px;" in css
    assert "--control-h-sm: 24px;" in css


def test_control_heights_are_in_the_committed_tokens_css() -> None:
    tokens = _TOKENS_CSS.read_text(encoding="utf-8")
    assert "--control-h: 28px;" in tokens
    assert "--control-h-sm: 24px;" in tokens


@pytest.mark.parametrize(
    "selector", [".osprey-btn", ".osprey-seg", ".osprey-search-field", ".osprey-field-input"]
)
def test_controls_size_from_the_control_height(selector: str) -> None:
    assert "height: var(--control-h);" in _rule(selector)


@pytest.mark.parametrize("selector", [".osprey-btn.sm", ".osprey-seg.sm", ".osprey-field-input.sm"])
def test_dense_controls_size_from_the_dense_control_height(selector: str) -> None:
    assert "height: var(--control-h-sm);" in _rule(selector)


# --- one family -------------------------------------------------------------------


def test_there_is_no_second_component_sheet() -> None:
    assert not (_CSS_DIR / "components.css").exists()


def test_every_class_root_is_an_osprey_name() -> None:
    """Each compound that names a class names an ``osprey-`` one: no bare-named family."""
    bare = sorted(
        {
            compound
            for selector in _selectors(_css())
            for compound in re.split(r"\s*[>+~\s]\s*", selector)
            if "." in compound and not re.search(r"\.osprey-", compound)
        }
    )
    assert not bare, f"class roots outside the osprey- family: {bare}"


@pytest.mark.parametrize("component", sorted(COMPONENTS))
def test_every_component_selector_is_defined(component: str) -> None:
    missing = [s for s in COMPONENTS[component] if not _defines(s)]
    assert not missing, f"{component}: no rule for {missing}"


@pytest.mark.parametrize("state", CHIP_RUN_STATES)
def test_every_chip_run_state_is_drawn(state: str) -> None:
    assert _defines(f".osprey-chip--{state}")


def test_sheet_reads_only_emitted_tokens_or_local_knobs_with_a_fallback() -> None:
    emitted = set(re.findall(r"--([a-zA-Z0-9-]+)\s*:", _TOKENS_CSS.read_text(encoding="utf-8")))
    declared = set(re.findall(r"--([a-zA-Z0-9-]+)\s*:", _css()))
    unresolved = sorted(
        name
        for name, rest in re.findall(r"var\(--([a-zA-Z0-9-]+)(\s*,)?", _css())
        if name not in emitted and name not in declared and not rest
    )
    assert not unresolved, f"var() with no emitted token and no fallback: {unresolved}"


def test_sheet_names_no_app() -> None:
    """A shared sheet carries no selector scoped to one interface or panel."""
    selectors = " ".join(_selectors(_css())).lower()
    for word in ("bluesky", "trajectory", "traj", "web-terminal", "ariel", "#"):
        assert word not in selectors, word


# --- documentation and the reference panel ----------------------------------------


@pytest.mark.parametrize("component", sorted(COMPONENTS))
def test_design_md_documents_every_component(component: str) -> None:
    section = _design_section()
    for selector in COMPONENTS[component]:
        classes = selector.lstrip(".").split(".")
        named = all(re.search(re.escape(c) + r"(?![\w-])", section) for c in classes)
        assert named, f"DESIGN.md §13 does not mention {selector}"


def test_design_md_lists_the_control_height_lever() -> None:
    design = _DESIGN_MD.read_text(encoding="utf-8")
    assert "--control-h" in design.split("## 4.", 1)[0]


def test_reference_panel_links_primitives_after_base() -> None:
    html = _REFERENCE_PANEL.read_text(encoding="utf-8")
    assert html.index("/design-system/css/primitives.css") > html.index(
        "/design-system/css/base.css"
    )


@pytest.mark.parametrize("component", sorted(COMPONENTS))
def test_reference_panel_draws_every_component(component: str) -> None:
    html = _REFERENCE_PANEL.read_text(encoding="utf-8")
    if component == "machine-map":
        assert "<osprey-machine-map" in html
        return
    root = COMPONENTS[component][0].lstrip(".")
    assert re.search(r'class="(?:[^"]*\s)?' + re.escape(root) + r'(?:\s[^"]*)?"', html), (
        f"reference panel draws no .{root}"
    )
