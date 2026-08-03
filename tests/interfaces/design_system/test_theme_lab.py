"""Browser smoke test for the Theme Lab page.

Proves the one thing no unit test can reach: that the lab's DOM layer actually
boots in a real browser and re-skins both scoped previews live. The math, the
gate scoring and the export markdown are unit-tested in
``tests/interfaces/design_system/js/theme-lab-*.test.mjs``; what is checked here
is the wiring between them and the page.

Specifically:

* the page boots with no console errors (a throwing ``initLab`` would otherwise
  fail silently, leaving a page that merely *looks* right),
* the two ``[data-theme]`` preview scopes really do resolve different token
  sets from ``tokens.css`` — the scoped-theme trick the whole page rests on,
* typing an accent updates each scope's computed ``--color-accent`` *without*
  rebuilding the mock,
* the export block carries the accent, the slug and the full change set, and
* a deliberately low-contrast accent raises a FAIL badge and a warning section.

Follows the harness pattern of ``test_behavioral.py``: a real uvicorn server on
a free port in a background thread, a real chromium browser, and a clean skip
when the chromium binary is not installed.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_theme_lab.py -v
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

pytestmark = pytest.mark.slow

LAB_PATH = "/design-system/theme-lab.html"

#: A vivid violet, far from the shipped teal so a re-skin is unambiguous.
ACCENT_HEX = "#8b5cf6"

#: Near-black: on the dark preview's near-black background this cannot clear
#: the 3:1 accent-vs-background gate, so it must raise a FAIL badge.
LOW_CONTRAST_HEX = "#0b1220"

#: Every file the exported change set has to name for it to be actionable.
CHANGE_SET_FILES = (
    "core.json",
    "ariel.json",
    "artifacts.json",
    "channel_finder.json",
    "lattice_dashboard.json",
    "web_terminal.json",
)


def _hex_channels(value: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` into 8-bit channels."""
    match = re.fullmatch(r"#([0-9a-fA-F]{6})", value.strip())
    assert match is not None, f"expected a 6-digit hex color, got {value!r}"
    digits = match.group(1)
    return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _scope_var(page: Page, mode: str, name: str) -> str:
    """Read a custom property off one preview scope's themed container."""
    return page.evaluate(
        """([mode, name]) => {
            const mount = document.getElementById(`preview-${mode}`);
            const panel = mount.closest('[data-theme]');
            return getComputedStyle(panel).getPropertyValue(name).trim();
        }""",
        [mode, name],
    )


def _set_accent(page: Page, value: str) -> None:
    """Type an accent into the hex field and let the lab settle."""
    page.fill("#hex-input", value)
    page.wait_for_timeout(100)


@pytest.fixture
def lab_page(chromium_browser: Browser) -> Iterator[Page]:
    """Serve the lab through its real CLI app factory and open it."""
    from osprey.cli.theme_lab_cmd import create_app

    errors: list[str] = []
    with _run_app_server(create_app()) as base_url:
        page = chromium_browser.new_page()
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(f"{base_url}{LAB_PATH}")
        page.wait_for_selector("#preview-dark .ms-shell")
        # Surfaced before any assertion below, so a boot failure is reported as
        # itself rather than as a confusing downstream selector timeout.
        assert errors == [], f"console errors on load: {errors}"
        yield page
        page.close()


def test_both_scopes_resolve_distinct_theme_tokens(lab_page: Page) -> None:
    """Each scoped container picks up its own mode's tokens, with no JS."""
    dark_bg = _scope_var(lab_page, "dark", "--bg-primary")
    light_bg = _scope_var(lab_page, "light", "--bg-primary")

    assert dark_bg != ""
    assert light_bg != ""
    assert dark_bg != light_bg, (
        "both preview scopes resolved the same --bg-primary; the scoped "
        "[data-theme] containers are not receiving separate token sets"
    )
    # The dark scope must genuinely be the darker one -- a swap would mean the
    # previews are mislabelled.
    assert sum(_hex_channels(dark_bg)) < sum(_hex_channels(light_bg))


def test_mini_shell_is_instantiated_into_both_scopes(lab_page: Page) -> None:
    """The template is cloned once per scope, so every accent surface exists."""
    for mode in ("dark", "light"):
        shell = lab_page.locator(f"#preview-{mode} .ms-shell")
        assert shell.count() == 1, f"expected exactly one mini-shell in {mode}"
        for selector in (".ms-btn-primary", ".ms-tab.is-active", ".ms-rail-item.is-active"):
            assert lab_page.locator(f"#preview-{mode} {selector}").count() >= 1, (
                f"{mode} preview is missing an accent-consuming surface: {selector}"
            )


def test_accent_change_reskins_both_scopes_without_rebuilding(lab_page: Page) -> None:
    """A color change is a custom-property write, not a DOM rebuild."""
    before_dark = _scope_var(lab_page, "dark", "--color-accent")
    # Captured too, and asserted below. Checking only "light is non-empty and
    # differs from dark" would NOT catch a light scope that never gets written
    # at all: it would still read the theme's own shipped accent out of
    # tokens.css, which is non-empty and differs from whatever was typed.
    before_light = _scope_var(lab_page, "light", "--color-accent")

    # Tag the existing node; if the mock were re-created the tag would vanish.
    lab_page.evaluate("document.querySelector('#preview-dark .ms-shell').dataset.sentinel = 'kept'")

    _set_accent(lab_page, ACCENT_HEX)

    after_dark = _scope_var(lab_page, "dark", "--color-accent")
    after_light = _scope_var(lab_page, "light", "--color-accent")

    assert after_dark != before_dark, "the dark preview did not re-skin"
    assert after_light != before_light, "the light preview did not re-skin"
    assert after_dark != after_light, (
        "both modes derived the same accent; the per-mode lightness offset is not being applied"
    )

    # The dark scope is the mode the hex field speaks for, so it should land on
    # (very close to) exactly what was typed -- allowing only hex->HSL->hex
    # rounding.
    typed = _hex_channels(ACCENT_HEX)
    landed = _hex_channels(after_dark)
    assert all(abs(a - b) <= 2 for a, b in zip(typed, landed, strict=True)), (
        f"dark accent {after_dark} is not the accent that was typed ({ACCENT_HEX})"
    )

    sentinel = lab_page.evaluate(
        "document.querySelector('#preview-dark .ms-shell').dataset.sentinel"
    )
    assert sentinel == "kept", "the mini-shell was rebuilt instead of re-skinned"


def test_export_carries_values_slug_and_change_set(lab_page: Page) -> None:
    """The export is implementable on its own, without the lab."""
    _set_accent(lab_page, ACCENT_HEX)
    lab_page.fill("#theme-name", "Harbour Dusk")
    lab_page.wait_for_timeout(100)

    export = lab_page.locator("#export-text").inner_text()
    dark_accent = _scope_var(lab_page, "dark", "--color-accent")

    assert "harbour-dusk" in export, "the slug is missing from the export"
    assert "harbour-dusk-dark" in export
    assert "harbour-dusk-light" in export
    assert dark_accent in export, "the chosen accent value is missing from the export"

    for filename in CHANGE_SET_FILES:
        assert filename in export, f"the change set does not name {filename}"
    assert "python -m osprey.interfaces.design_system.generator.build" in export, (
        "the change set does not say how to regenerate the artifacts"
    )
    # Token dot-paths, not just CSS variable names -- the developer edits JSON.
    for path in ("accent.base", "accent.light", "accent.on", "border.accent", "tint.accent.04"):
        assert path in export, f"the export does not map to the token path {path}"


def test_low_contrast_accent_fails_a_gate_and_warns_in_the_export(lab_page: Page) -> None:
    """Gates warn, never block -- but they must actually fire."""
    _set_accent(lab_page, ACCENT_HEX)
    assert "FAIL" not in lab_page.locator("#gates-dark").inner_text(), (
        "the control accent should clear the dark-mode gates"
    )

    _set_accent(lab_page, LOW_CONTRAST_HEX)

    badges = lab_page.locator("#gates-dark").inner_text()
    assert "FAIL" in badges, f"a near-background accent did not fail a gate: {badges!r}"

    export = lab_page.locator("#export-text").inner_text()
    assert "Warnings" in export, "a failing gate produced no warning section in the export"
    assert "below the required" in export


def test_colliding_theme_name_is_flagged(lab_page: Page) -> None:
    """A name that would shadow a shipped family is called out."""
    lab_page.fill("#theme-name", "apex")
    lab_page.wait_for_timeout(100)

    warning = lab_page.locator("#slug-warning").inner_text()
    assert "apex" in warning
    assert "already" in warning.lower(), f"collision not reported: {warning!r}"

    export = lab_page.locator("#export-text").inner_text()
    assert "Warnings" in export


# ---------------------------------------------------------------------------
# Validator parity — no browser required, so this runs in every lane
# ---------------------------------------------------------------------------
#
# The lab tells a colleague whether their accent passes. The generator decides
# whether it actually ships. If those two ever disagree, the lab lies -- someone
# is told their theme is fine and then CI rejects it, or worse, the reverse. The
# two implementations are in different languages and cannot share code, so what
# is asserted here is that they still agree on the numbers.


def _theme_lab_source() -> str:
    from pathlib import Path

    import osprey.interfaces.design_system as design_system_pkg

    return (Path(design_system_pkg.__file__).parent / "static" / "js" / "theme-lab.js").read_text()


def test_lab_wcag_constants_match_the_generator() -> None:
    """Both luminance implementations use the same WCAG 2.x constants."""
    import inspect

    from osprey.interfaces.design_system.generator import validate

    python_source = inspect.getsource(validate.relative_luminance) + inspect.getsource(
        validate.contrast_ratio
    )
    js_source = _theme_lab_source()

    # Every constant WCAG 2.x fixes. A typo in any one of them changes a ratio
    # without changing anything a name-based guard would notice.
    for constant in ("0.2126", "0.7152", "0.0722", "0.03928", "12.92", "0.055", "1.055", "2.4"):
        assert constant in python_source, f"{constant} vanished from the generator's luminance"
        assert constant in js_source, (
            f"the theme lab's luminance no longer uses {constant}; it has drifted from "
            "generator/validate.py and its contrast badges can no longer be trusted"
        )
    assert "0.05" in python_source and "0.05" in js_source, (
        "the contrast-ratio offset differs between the lab and the generator"
    )


def test_lab_gates_match_the_generators_accent_gates() -> None:
    """The lab scores exactly the accent gates the generator enforces."""
    from osprey.interfaces.design_system.generator.validate import WCAG_GATES

    expected = {
        f"{gate.foreground} vs {gate.background}": gate.minimum
        for gate in WCAG_GATES
        if gate.foreground.startswith("accent")
    }
    assert expected, "the generator no longer defines any accent gates"

    found = {
        name: float(threshold)
        for name, threshold in re.findall(
            r"\{\s*name:\s*'([^']+)',\s*threshold:\s*([0-9.]+)\s*\}", _theme_lab_source()
        )
    }

    assert found == expected, (
        "the theme lab's gates have drifted from generator/validate.py's WCAG_GATES.\n"
        f"  generator: {expected}\n"
        f"  theme lab: {found}\n"
        "A colleague would be told their theme passes a check the build does not run, "
        "or fails one it does not enforce."
    )
