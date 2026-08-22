"""Real-browser tests for the plan form's channel-suggestion combobox.

Runs the composed bluesky-web sidecar for real (``bluesky_live_server``, bridge
stubbed) and drives headless Chromium against ``/bluesky/``, where the plans
view auto-selects the stub plan and renders its form: a channel-tagged string
field (``Target``, ``buildString``), a channel-list field (``Monitors``,
``buildChannelList``), and an untagged number (``Setpoint``). The vitest suites
pin the combobox and schema-form contracts unit-by-unit; this suite pins what
only a real browser exercises end to end — genuine focus/hover/keyboard event
ordering across the combobox listener and the commit-list listener on the same
input, and the real ``/channels`` fetch feeding the module-global catalog:

- arrow keys arm and wrap, Escape closes, focus never leaves the input
- a *hovered* popup item never arms: with the pointer parked on the open popup,
  a single unarmed Enter commits the typed text exactly (pinned by typing a
  lowercase exact address and asserting the hovered uppercase catalog entry
  does not replace it)
- prefix + click commits only the clicked address — mousedown-preventDefault
  keeps focus in the add-input, so no blur ever commits the half-typed prefix
- an arrow-armed Enter flows through both buildChannelList (accept + commit as
  one keystroke) and a table cell's channel-tagged input
- a value outside the catalog is never blocked: no popup, and it submits/
  commits exactly as typed
- the no-catalog deployment (``/channels`` 404) renders channel-tagged fields
  identical in shape to untagged ones, with zero console errors

The stub plan has no array-of-objects field, so the table-cell case cannot be
reached through the served plan form; that one test bootstraps a second form on
the same page via the panel's own ``renderSchemaForm`` export (same module
instance, so the catalog installed by the panel's boot ``/channels`` fetch is
live in it).

Run:
    .venv/bin/pytest tests/interfaces/bluesky_web/test_channel_combobox_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from tests.interfaces.bluesky_web.conftest import DEFAULT_CHANNEL_CATALOG

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Locator, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# Addresses the tests type against, pinned to the fixture catalog here so a
# future catalog edit fails with a clear message instead of a popup timeout.
HCM_ADDRESS = "BR:MAG:HCM:01:CURRENT:SP"
QF_ADDRESS = "SR:MAG:QF:03:CURRENT:SP"
assert HCM_ADDRESS in DEFAULT_CHANNEL_CATALOG
assert QF_ADDRESS in DEFAULT_CHANNEL_CATALOG

# Comfortably longer than the combobox's DEFAULT_DEBOUNCE_MS (150 ms). Positive
# popup assertions poll via expect(); only the negative "the popup never
# opened" assertions need to outwait the debounced query like this.
_DEBOUNCE_SETTLE_MS = 400

_FORM_TIMEOUT_MS = 15_000

TARGET = '#param-form input[aria-label="Target"]'
CHANNEL_ADD = "#param-form .channel-list input.channel-add"
CHANNEL_ROWS = "#param-form .channel-items li"

# Bootstrap for the table-cell case (see module docstring): render an
# array-of-flat-objects field — schema-form's buildTable — whose string column
# is channel-tagged, through the same schema-form module the panel already
# loaded (absolute URL == panel.js's resolved './schema-form.js', so the
# dynamic import returns the instance holding the booted channel catalog).
_TABLE_BOOTSTRAP = """
async () => {
  const mod = await import('/bluesky/schema-form.js');
  const form = document.createElement('form');
  form.id = 'table-probe';
  form.addEventListener('submit', (event) => event.preventDefault());
  document.body.appendChild(form);
  mod.renderSchemaForm(form, {
    title: 'PARAMS',
    type: 'object',
    properties: {
      taps: {
        title: 'Taps',
        type: 'array',
        items: {
          title: 'Tap',
          type: 'object',
          properties: {
            channel: { title: 'Channel', type: 'string', 'x-channel-role': 'readable' },
            gain: { title: 'Gain', type: 'number' },
          },
        },
      },
    },
  });
}
"""


@contextmanager
def _panel_page(browser: Browser, base_url: str) -> Iterator[Page]:
    """Open ``/bluesky/`` in a fresh page; assert no uncaught JS errors on exit.

    The pageerror listener attaches before navigation so boot-time exceptions
    are never missed. The trailing assertion only runs when the test body
    succeeded — a failing body propagates its own, more specific failure.
    """
    page = browser.new_page()
    pageerrors: list[str] = []
    page.on("pageerror", lambda err: pageerrors.append(str(err)))
    try:
        page.goto(f"{base_url}/bluesky/", wait_until="load")
        yield page
    finally:
        page.close()
    assert pageerrors == []


def _plan_form_target(page: Page) -> Locator:
    """Wait for the auto-selected stub plan's form and return the Target input."""
    target = page.locator(TARGET)
    expect(target).to_be_visible(timeout=_FORM_TIMEOUT_MS)
    return target


def _listbox_for(page: Page, input_locator: Locator) -> Locator:
    """Resolve an input's own popup listbox via its aria-controls id.

    The combobox stamps ``aria-controls`` when it attaches to an input, which
    for a field the caller has not already waited on can land after the first
    read. Poll for the attribute rather than sampling it once: it is removed
    only on ``destroy()``, so an input that is genuinely never enhanced still
    fails here, just on the timeout instead of on a race.
    """
    expect(input_locator).to_have_attribute(
        "aria-controls", re.compile(r".+"), timeout=_FORM_TIMEOUT_MS
    )
    list_id = input_locator.get_attribute("aria-controls")
    assert list_id, "input carries no aria-controls — combobox not attached"
    return page.locator(f"#{list_id}")


# ---------------------------------------------------------------------------
# 1. Keyboard navigation
# ---------------------------------------------------------------------------


def test_arrow_keys_wrap_escape_closes_and_focus_stays(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        target = _plan_form_target(page)
        target.fill("BPM")

        listbox = _listbox_for(page, target)
        expect(listbox).to_be_visible()
        expect(target).to_have_attribute("aria-expanded", "true")

        items = listbox.locator(".channel-combobox-item")
        count = items.count()
        assert count >= 2, "need at least two suggestions for wrap-around to mean anything"
        for i in range(count):
            assert "BPM" in (items.nth(i).text_content() or "")

        # No default highlight: nothing is armed until an arrow key.
        assert target.get_attribute("aria-activedescendant") is None

        first_id = items.nth(0).get_attribute("id")
        last_id = items.nth(count - 1).get_attribute("id")
        assert first_id and last_id

        target.press("ArrowDown")
        expect(target).to_have_attribute("aria-activedescendant", first_id)
        expect(items.nth(0)).to_have_attribute("aria-selected", "true")

        target.press("ArrowUp")  # wraps: first -> last
        expect(target).to_have_attribute("aria-activedescendant", last_id)
        expect(items.nth(0)).to_have_attribute("aria-selected", "false")
        expect(items.nth(count - 1)).to_have_attribute("aria-selected", "true")

        target.press("ArrowDown")  # wraps: last -> first
        expect(target).to_have_attribute("aria-activedescendant", first_id)

        expect(target).to_be_focused()

        target.press("Escape")
        expect(listbox).to_be_hidden()
        expect(target).to_have_attribute("aria-expanded", "false")
        assert page.locator(".channel-combobox.open").count() == 0
        expect(target).to_be_focused()
        expect(target).to_have_value("BPM")  # Escape dismisses the popup, not the text


# ---------------------------------------------------------------------------
# 2. Hover never arms: unarmed Enter commits the typed text exactly
# ---------------------------------------------------------------------------


def test_hovered_item_never_steals_the_unarmed_enter(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """Pointer parked on the open popup + single Enter = the typed value, unmodified.

    Typed lowercase, the case-insensitive matcher still offers the uppercase
    catalog entry — so if hovering armed the item (or Enter honored the
    hovered row), the committed value would flip to uppercase. It must not.
    """
    typed = HCM_ADDRESS.lower()
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        target = _plan_form_target(page)
        target.fill(typed)

        listbox = _listbox_for(page, target)
        expect(listbox).to_be_visible()
        item = listbox.locator(".channel-combobox-item", has_text=HCM_ADDRESS)
        expect(item).to_be_visible()

        item.hover()  # park the pointer on the suggestion and leave it there
        assert target.get_attribute("aria-activedescendant") is None
        expect(item).to_have_attribute("aria-selected", "false")

        target.press("Enter")  # single, unarmed Enter

        expect(listbox).to_be_hidden()
        expect(target).to_have_value(typed)  # NOT the hovered item's casing
        expect(target).to_be_focused()


# ---------------------------------------------------------------------------
# 3. Prefix + click commits only the clicked address
# ---------------------------------------------------------------------------


def test_prefix_plus_click_commits_only_the_clicked_address(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        _plan_form_target(page)

        # Channel-list add-input: the click must not blur-commit the prefix.
        add = page.locator(CHANNEL_ADD)
        add.fill("BPM")
        listbox = _listbox_for(page, add)
        expect(listbox).to_be_visible()
        clicked = listbox.locator(".channel-combobox-item").nth(0).text_content()
        assert clicked in DEFAULT_CHANNEL_CATALOG

        listbox.locator(".channel-combobox-item").nth(0).click()

        rows = page.locator(CHANNEL_ROWS)
        expect(rows).to_have_count(0)  # no partial-text "BPM" row snuck in
        expect(add).to_have_value(clicked)  # half-typed prefix fully replaced
        expect(add).to_be_focused()  # mousedown preventDefault kept focus here
        expect(listbox).to_be_hidden()

        add.press("Enter")
        expect(rows).to_have_count(1)
        expect(rows.nth(0).locator(".channel-name")).to_have_text(clicked)
        expect(add).to_have_value("")

        # Same shape on the single-channel buildString field.
        target = page.locator(TARGET)
        target.fill("QF")
        target_listbox = _listbox_for(page, target)
        expect(target_listbox).to_be_visible()
        option = target_listbox.locator(".channel-combobox-item").nth(0)
        expect(option).to_have_text(QF_ADDRESS)
        option.click()
        expect(target).to_have_value(QF_ADDRESS)
        expect(target_listbox).to_be_hidden()


# ---------------------------------------------------------------------------
# 4. Armed-Enter flows: channel list and table cell
# ---------------------------------------------------------------------------


def test_armed_enter_commits_the_suggestion_in_the_channel_list(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """One keystroke: the combobox accepts the armed value, the list commits it."""
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        _plan_form_target(page)
        add = page.locator(CHANNEL_ADD)
        add.fill("BPM")
        listbox = _listbox_for(page, add)
        expect(listbox).to_be_visible()

        add.press("ArrowDown")
        armed = listbox.locator(".channel-combobox-item").nth(0)
        expect(armed).to_have_attribute("aria-selected", "true")
        armed_value = armed.text_content()
        assert armed_value in DEFAULT_CHANNEL_CATALOG

        add.press("Enter")

        rows = page.locator(CHANNEL_ROWS)
        expect(rows).to_have_count(1)
        expect(rows.nth(0).locator(".channel-name")).to_have_text(armed_value)
        expect(add).to_have_value("")
        expect(listbox).to_be_hidden()
        expect(page.locator("#param-form .channel-count")).to_have_text("1 channel")


def test_armed_enter_commits_the_suggestion_in_a_table_cell(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        _plan_form_target(page)  # panel booted: /channels catalog is installed
        page.evaluate(_TABLE_BOOTSTRAP)

        page.locator("#table-probe .table-add").click()
        cell = page.locator('#table-probe td input[aria-label="Channel"]')
        expect(cell).to_be_visible()

        cell.fill("BPM")
        listbox = _listbox_for(page, cell)
        expect(listbox).to_be_visible()

        cell.press("ArrowDown")
        armed = listbox.locator(".channel-combobox-item").nth(0)
        expect(armed).to_have_attribute("aria-selected", "true")
        armed_value = armed.text_content()
        assert armed_value in DEFAULT_CHANNEL_CATALOG

        cell.press("Enter")

        expect(cell).to_have_value(armed_value)
        expect(listbox).to_be_hidden()
        expect(cell).to_be_focused()
        assert page.url.endswith("/bluesky/")  # armed Enter never submitted the form


# ---------------------------------------------------------------------------
# 5. Free-form non-catalog values are never blocked
# ---------------------------------------------------------------------------


def test_freeform_noncatalog_value_flows_as_typed(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    freeform = "ZZ:NOT:IN:CATALOG"  # no catalog entry contains a Z: zero matches
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        target = _plan_form_target(page)

        target.fill(freeform)
        page.wait_for_timeout(_DEBOUNCE_SETTLE_MS)
        assert page.locator(".channel-combobox.open").count() == 0
        expect(target).to_have_attribute("aria-expanded", "false")
        target.press("Enter")
        expect(target).to_have_value(freeform)

        add = page.locator(CHANNEL_ADD)
        add.fill(freeform)
        page.wait_for_timeout(_DEBOUNCE_SETTLE_MS)
        assert page.locator(".channel-combobox.open").count() == 0
        add.press("Enter")

        rows = page.locator(CHANNEL_ROWS)
        expect(rows).to_have_count(1)
        expect(rows.nth(0).locator(".channel-name")).to_have_text(freeform)
        expect(add).to_have_value("")


# ---------------------------------------------------------------------------
# 6. No-catalog deployment: channel fields render as plain inputs
# ---------------------------------------------------------------------------


def test_no_catalog_variant_renders_baseline_dom_with_no_console_errors(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """With /channels 404ing, channel-tagged fields match the untagged baseline.

    "Identical to baseline" is asserted structurally: zero combobox artifacts
    anywhere, none of the combobox ARIA attributes on the tagged inputs, and
    the tagged inputs parented exactly like the untagged Setpoint input.
    Console gating allowlists only Chromium's own resource-error line for the
    expected /channels 404 (matched on the message's source URL).
    """
    with bluesky_live_server(catalog=None) as base_url:
        page = chromium_browser.new_page()
        pageerrors: list[str] = []
        console_errors: list[str] = []

        def _on_console(msg) -> None:  # noqa: ANN001 - Playwright ConsoleMessage
            if msg.type != "error":
                return
            url = (msg.location or {}).get("url") or ""
            if url.rstrip("/").endswith("/channels"):
                return
            console_errors.append(msg.text)

        page.on("pageerror", lambda err: pageerrors.append(str(err)))
        page.on("console", _on_console)
        try:
            page.goto(f"{base_url}/bluesky/", wait_until="load")
            target = _plan_form_target(page)
            add = page.locator(CHANNEL_ADD)

            assert page.locator(".channel-combobox").count() == 0
            assert page.locator(".channel-combobox-list").count() == 0
            combobox_attrs = (
                "role",
                "aria-expanded",
                "aria-autocomplete",
                "aria-controls",
                "aria-activedescendant",
            )
            for attr in combobox_attrs:
                assert target.get_attribute(attr) is None, f"Target carries {attr}"
                assert add.get_attribute(attr) is None, f"channel-add carries {attr}"

            # Parent shape matches the untagged baseline: the tagged string
            # input sits in its .field-control exactly like the untagged
            # number input, and the add-input hangs directly off .channel-list.
            shapes = page.evaluate(
                """() => {
                  const parentShape = (el) => ({
                    tag: el.parentElement.tagName,
                    cls: el.parentElement.className,
                  });
                  return {
                    target: parentShape(
                      document.querySelector('#param-form input[aria-label="Target"]')
                    ),
                    setpoint: parentShape(
                      document.querySelector('#param-form input[aria-label="Setpoint"]')
                    ),
                    add: parentShape(document.querySelector('#param-form .channel-add')),
                  };
                }"""
            )
            assert shapes["target"] == shapes["setpoint"]
            assert shapes["add"] == {"tag": "DIV", "cls": "channel-list"}

            # Interaction stays bare-input: typing conjures no popup material.
            target.fill("BPM")
            add.fill("BPM")
            page.wait_for_timeout(_DEBOUNCE_SETTLE_MS)
            assert page.locator(".channel-combobox").count() == 0
            expect(target).to_have_value("BPM")

            add.press("Enter")
            expect(page.locator(CHANNEL_ROWS)).to_have_count(1)
        finally:
            page.close()

        assert pageerrors == []
        assert console_errors == []


# ---------------------------------------------------------------------------
# 7. The popup is actually on screen, not clipped away by an ancestor
# ---------------------------------------------------------------------------

# Whether the popup's options are really painted where the user would click.
#
# Every DOM-level assertion the suite already makes -- aria-expanded="true",
# to_be_visible(), a full complement of .channel-combobox-item children -- is
# equally true of a popup an ancestor's `overflow: clip`/`hidden` has cropped
# to a two-pixel sliver: clipping changes neither an element's bounding box nor
# its computed visibility, which is all Playwright's visibility check reads.
#
# So ask the browser instead of reasoning about it. document.elementFromPoint
# is the same hit test a real click goes through, and it already accounts for
# clipping, stacking order, and the top layer. Walking the ancestor chain to
# intersect rectangles in JS would be actively wrong here: the fix puts the
# popup in the top layer via `popover`, which leaves it exactly where it is in
# the DOM -- its clipping ancestors are still its parents, they simply no
# longer clip it -- so a hand-rolled intersection would report a false clip.
_HIT_TEST_OPTIONS = """
(listId) => {
  const list = document.getElementById(listId);
  const items = [...list.querySelectorAll('.channel-combobox-item')];
  return items.slice(0, 3).map((item) => {
    const box = item.getBoundingClientRect();
    const x = box.left + box.width / 2;
    const y = box.top + box.height / 2;
    const hit = document.elementFromPoint(x, y);
    return {
      address: (item.textContent || '').trim(),
      // null == the point is outside the viewport entirely.
      hit: hit ? `${hit.tagName.toLowerCase()}.${hit.className}` : null,
      reachable: !!hit && (hit === item || item.contains(hit) || list.contains(hit)),
    };
  });
}
"""


def _assert_options_are_on_screen(page: Page, listbox: Locator, where: str) -> None:
    """Fail unless the popup's first options answer their own hit test.

    Asserts on the first three rather than only the first: a popup cropped to a
    sliver can still surface a few pixels of row one, and a suggestion list you
    cannot pick the second entry from is not a working suggestion list.
    """
    list_id = listbox.get_attribute("id")
    assert list_id, "listbox carries no id"
    probes = page.evaluate(_HIT_TEST_OPTIONS, list_id)
    assert len(probes) >= 2, f"{where}: need at least two suggestions to probe"

    blocked = [p for p in probes if not p["reachable"]]
    assert not blocked, (
        f"{where}: the popup is not on screen — "
        f"{len(blocked)} of {len(probes)} options fail their own hit test. "
        f"An ancestor's overflow has clipped the popup away: {blocked}"
    )


def test_popup_is_on_screen_inside_the_channel_list(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """``.channel-list`` sets ``overflow: hidden`` for its rounded corners."""
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        _plan_form_target(page)
        add = page.locator(CHANNEL_ADD)
        add.fill("BPM")

        listbox = _listbox_for(page, add)
        expect(listbox).to_be_visible()
        expect(listbox.locator(".channel-combobox-item").nth(1)).to_be_visible()

        _assert_options_are_on_screen(page, listbox, "channel-list add-input")


def test_popup_is_on_screen_inside_a_table_cell(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """``.obj-table`` sets ``overflow: clip`` for its rounded corners."""
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        _plan_form_target(page)
        page.evaluate(_TABLE_BOOTSTRAP)

        page.locator("#table-probe .table-add").click()
        cell = page.locator('#table-probe td input[aria-label="Channel"]')
        expect(cell).to_be_visible()
        cell.fill("BPM")

        listbox = _listbox_for(page, cell)
        expect(listbox).to_be_visible()
        expect(listbox.locator(".channel-combobox-item").nth(1)).to_be_visible()

        _assert_options_are_on_screen(page, listbox, "obj-table channel cell")


# ---------------------------------------------------------------------------
# 8. The armed row is visibly distinguishable from its neighbours
# ---------------------------------------------------------------------------

# Keyboard selection is only usable if the user can SEE which row Enter will
# take. aria-selected="true" and a non-empty style.background both hold for a
# row painted the exact colour of the popup underneath it, which is what the
# `light` theme did: it defines --bg-secondary and --bg-elevated to the same
# #ecedef, so the old surface-token highlight was literally invisible there.
# Resolve the painted colours instead and require them to actually differ.
_ROW_COLOURS = """
(listId) => {
  const list = document.getElementById(listId);
  const rows = [...list.querySelectorAll('.channel-combobox-item')];
  const armed = rows.find((r) => r.getAttribute('aria-selected') === 'true');
  const plain = rows.find((r) => r.getAttribute('aria-selected') !== 'true');
  const paint = (node) => getComputedStyle(node).backgroundColor;
  return { armed: armed ? paint(armed) : null, plain: plain ? paint(plain) : null,
           surface: paint(list) };
}
"""


def test_armed_row_is_visually_distinct_from_the_popup_surface(
    chromium_browser, bluesky_live_server
) -> None:
    """An armed row the user cannot see is not a usable keyboard cursor."""
    with bluesky_live_server() as base_url, _panel_page(chromium_browser, base_url) as page:
        target = _plan_form_target(page)
        target.fill("BPM")

        listbox = _listbox_for(page, target)
        expect(listbox).to_be_visible()
        expect(listbox.locator(".channel-combobox-item").nth(1)).to_be_visible()

        target.press("ArrowDown")
        expect(listbox.locator(".channel-combobox-item").nth(0)).to_have_attribute(
            "aria-selected", "true"
        )

        colours = page.evaluate(_ROW_COLOURS, listbox.get_attribute("id"))
        assert colours["armed"], "no row reports itself armed"
        assert colours["armed"] != colours["surface"], (
            "the armed row is painted the same colour as the popup it sits on "
            f"({colours['armed']}) — the keyboard cursor is invisible"
        )
        assert colours["armed"] != colours["plain"], (
            "the armed row is painted the same colour as an unarmed row "
            f"({colours['armed']}) — the keyboard cursor is indistinguishable"
        )
