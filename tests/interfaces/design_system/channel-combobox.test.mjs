// @ts-check
/**
 * Unit tests for channel-combobox.js — the anchored suggestion popover behind
 * plan-form channel fields. Pins the selection semantics contract:
 *
 *   - NO default highlight: aria-activedescendant appears only after
 *     ArrowUp/ArrowDown, which wrap; hover is visual-only + click-select
 *   - option mousedown is preventDefault()ed so the input keeps focus
 *   - Enter accepts only an arrow-armed item (then stops propagation);
 *     unarmed Enter closes the popup and still propagates to the form
 *   - Escape closes; outside mousedown closes via a capture-phase document
 *     listener installed on open and removed on close
 *   - input re-queries the matcher behind a trailing-edge debounce, and a
 *     user-intent close cancels a still-pending query
 *   - highlights are built from DOM nodes, so catalog values containing
 *     markup render as literal text (no HTML injection)
 *   - free-form entry is never blocked: values outside the catalog type fine
 *
 * Run with:
 *   npx vitest run tests/interfaces/design_system/channel-combobox.test.mjs
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

import {
  attachChannelCombobox,
  DEFAULT_DEBOUNCE_MS,
} from '/design-system/js/channel-combobox.js';
import { RENDER_CAP } from '/design-system/js/channel-matcher.js';

/**
 * Three entries match "BPM" (as substring and hence subsequence); the HCM
 * entries contain no "b"/"p" so they cannot even fuzzy-match it. That makes
 * the popup contents for query "BPM" exactly three rows, which the arrow
 * wrapping tests rely on.
 */
const CATALOG = ['SR01:BPM:X', 'SR02:BPM:X', 'SR03:BPM:X', 'SR01:HCM:X', 'SR02:HCM:X'];

/**
 * Mount a fresh input inside a holder div and attach the combobox.
 * @param {string[]} [catalog]
 * @param {import('/design-system/js/channel-combobox.js').ChannelComboboxOptions} [opts]
 * @returns {{ input: HTMLInputElement, holder: HTMLElement, handle: ReturnType<typeof attachChannelCombobox> }}
 */
function setup(catalog = CATALOG, opts = {}) {
  const holder = document.createElement('div');
  document.body.appendChild(holder);
  const input = document.createElement('input');
  holder.appendChild(input);
  const handle = attachChannelCombobox(input, catalog, opts);
  return { input, holder, handle };
}

/**
 * Simulate typing: set the value, fire `input`, and run out the debounce.
 * @param {HTMLInputElement} input
 * @param {string} text
 * @returns {void}
 */
function type(input, text) {
  input.value = text;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  vi.advanceTimersByTime(DEFAULT_DEBOUNCE_MS);
}

/**
 * Dispatch a bubbling, cancelable keydown and return it for assertions.
 * @param {HTMLInputElement} input
 * @param {string} key
 * @returns {KeyboardEvent}
 */
function press(input, key) {
  const e = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
  input.dispatchEvent(e);
  return e;
}

/**
 * The wrapper the combobox inserted around the input.
 * @param {HTMLInputElement} input
 * @returns {HTMLElement}
 */
function wrapperOf(input) {
  const wrapper = input.parentElement;
  if (!wrapper) throw new Error('input has no parent');
  return wrapper;
}

/**
 * All rendered option rows.
 * @param {HTMLInputElement} input
 * @returns {HTMLElement[]}
 */
function optionsOf(input) {
  return Array.from(wrapperOf(input).querySelectorAll('[role="option"]'));
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('attachChannelCombobox', () => {
  it('ATTACH: wraps the input in a position:relative container with the ARIA combobox contract', () => {
    const { input, holder } = setup();
    const wrapper = wrapperOf(input);

    expect(wrapper.classList.contains('channel-combobox')).toBe(true);
    expect(wrapper.style.position).toBe('relative');
    expect(wrapper.parentElement).toBe(holder);

    expect(input.getAttribute('role')).toBe('combobox');
    expect(input.getAttribute('aria-expanded')).toBe('false');
    expect(input.getAttribute('aria-autocomplete')).toBe('list');

    const listId = input.getAttribute('aria-controls');
    expect(listId).toBeTruthy();
    const listbox = wrapper.querySelector(`#${listId}`);
    if (!listbox) throw new Error('listbox not found');
    expect(listbox.getAttribute('role')).toBe('listbox');
  });

  it('OPEN: typing opens the popup (.open class, aria-expanded, option rows) after the debounce', () => {
    const { input, handle } = setup();
    input.value = 'BPM';
    input.dispatchEvent(new Event('input', { bubbles: true }));

    // Trailing-edge debounce: nothing happens until the delay elapses.
    expect(handle.isOpen()).toBe(false);
    expect(wrapperOf(input).classList.contains('open')).toBe(false);

    vi.advanceTimersByTime(DEFAULT_DEBOUNCE_MS);

    expect(handle.isOpen()).toBe(true);
    expect(wrapperOf(input).classList.contains('open')).toBe(true);
    expect(input.getAttribute('aria-expanded')).toBe('true');
    expect(optionsOf(input).length).toBe(3);
  });

  it('NO DEFAULT HIGHLIGHT: opening arms nothing — no aria-activedescendant, no aria-selected row', () => {
    const { input } = setup();
    type(input, 'BPM');

    expect(input.hasAttribute('aria-activedescendant')).toBe(false);
    for (const row of optionsOf(input)) {
      expect(row.getAttribute('aria-selected')).toBe('false');
    }
  });

  it('ARROWS: ArrowDown/ArrowUp move aria-activedescendant and wrap at both ends', () => {
    const { input } = setup();
    type(input, 'BPM');
    const rows = optionsOf(input);

    const down = press(input, 'ArrowDown');
    expect(down.defaultPrevented).toBe(true);
    expect(input.getAttribute('aria-activedescendant')).toBe(rows[0].id);
    expect(rows[0].getAttribute('aria-selected')).toBe('true');

    press(input, 'ArrowDown');
    press(input, 'ArrowDown');
    expect(input.getAttribute('aria-activedescendant')).toBe(rows[2].id);

    // Wrap bottom -> top.
    press(input, 'ArrowDown');
    expect(input.getAttribute('aria-activedescendant')).toBe(rows[0].id);
    expect(rows[2].getAttribute('aria-selected')).toBe('false');

    // Wrap top -> bottom.
    press(input, 'ArrowUp');
    expect(input.getAttribute('aria-activedescendant')).toBe(rows[2].id);
  });

  it('ARROWS: ArrowUp from the unarmed state arms the LAST option', () => {
    const { input } = setup();
    type(input, 'BPM');
    const rows = optionsOf(input);

    press(input, 'ArrowUp');
    expect(input.getAttribute('aria-activedescendant')).toBe(rows[rows.length - 1].id);
  });

  it('ENTER unarmed: closes the popup, does not consume the event, and propagates to the parent', () => {
    const { input, holder, handle } = setup();
    /** @type {KeyboardEvent[]} */
    const seen = [];
    holder.addEventListener('keydown', (e) => seen.push(e));
    type(input, 'BPM');

    const e = press(input, 'Enter');

    expect(handle.isOpen()).toBe(false);
    expect(e.defaultPrevented).toBe(false);
    expect(seen.length).toBe(1);
    expect(input.value).toBe('BPM');
  });

  it('ENTER armed: selects the armed value, stops propagation, and calls onSelect', () => {
    const onSelect = vi.fn();
    const { input, holder, handle } = setup(CATALOG, { onSelect });
    /** @type {KeyboardEvent[]} */
    const seen = [];
    holder.addEventListener('keydown', (e) => seen.push(e));
    type(input, 'BPM');
    const armedValue = optionsOf(input)[0].textContent;

    press(input, 'ArrowDown');
    const e = press(input, 'Enter');

    expect(input.value).toBe(armedValue);
    expect(onSelect).toHaveBeenCalledWith(armedValue);
    expect(handle.isOpen()).toBe(false);
    expect(e.defaultPrevented).toBe(true);
    // stopPropagation: the parent's listener never saw the armed Enter
    // (the earlier ArrowDown was not stopped, so filter to Enter).
    expect(seen.filter((k) => k.key === 'Enter').length).toBe(0);
  });

  it('SELECT: accepting re-dispatches a bubbling input event without re-opening the popup', () => {
    const { input, holder, handle } = setup();
    /** @type {string[]} */
    const observed = [];
    holder.addEventListener('input', () => observed.push(input.value));
    type(input, 'BPM');
    observed.length = 0;

    press(input, 'ArrowDown');
    press(input, 'Enter');

    // Form-level listeners saw the programmatic value change...
    expect(observed).toEqual([input.value]);
    // ...and the synthetic event did not schedule a re-query that reopens.
    vi.advanceTimersByTime(DEFAULT_DEBOUNCE_MS * 2);
    expect(handle.isOpen()).toBe(false);
  });

  it('PENDING QUERY: a select cancels an in-flight debounced query so the popup stays closed', () => {
    const { input, handle } = setup();
    type(input, 'BPM');
    // New keystroke schedules a query, but the user selects before it fires.
    input.value = 'BPM:';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    press(input, 'ArrowDown');
    press(input, 'Enter');
    expect(handle.isOpen()).toBe(false);

    vi.advanceTimersByTime(DEFAULT_DEBOUNCE_MS * 2);
    expect(handle.isOpen()).toBe(false);
  });

  it('ESCAPE: closes the popup and consumes the event', () => {
    const { input, handle } = setup();
    type(input, 'BPM');
    expect(handle.isOpen()).toBe(true);

    const e = press(input, 'Escape');
    expect(handle.isOpen()).toBe(false);
    expect(input.getAttribute('aria-expanded')).toBe('false');
    expect(e.defaultPrevented).toBe(true);
  });

  it('MOUSEDOWN on an option is preventDefault()ed (input keeps focus) and does not close the popup', () => {
    const { input, handle } = setup();
    type(input, 'BPM');
    const row = optionsOf(input)[0];

    const e = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
    row.dispatchEvent(e);

    expect(e.defaultPrevented).toBe(true);
    expect(handle.isOpen()).toBe(true);
  });

  it('CLICK on an option selects its value and closes', () => {
    const onSelect = vi.fn();
    const { input, handle } = setup(CATALOG, { onSelect });
    type(input, 'BPM');
    const row = optionsOf(input)[1];
    const value = row.textContent;

    row.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(input.value).toBe(value);
    expect(onSelect).toHaveBeenCalledWith(value);
    expect(handle.isOpen()).toBe(false);
  });

  it('HOVER is visual-only: mouseenter never moves aria-activedescendant or aria-selected', () => {
    const { input } = setup();
    type(input, 'BPM');
    const row = optionsOf(input)[1];

    row.dispatchEvent(new MouseEvent('mouseenter', { bubbles: false }));

    expect(input.hasAttribute('aria-activedescendant')).toBe(false);
    expect(row.getAttribute('aria-selected')).toBe('false');
  });

  it('OUTSIDE MOUSEDOWN: a capture-phase document listener closes the popup', () => {
    const { input, handle } = setup();
    type(input, 'BPM');
    expect(handle.isOpen()).toBe(true);

    document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(handle.isOpen()).toBe(false);
  });

  it('LISTENER LIFECYCLE: the document mousedown listener is added (capture) on open and removed on close', () => {
    const addSpy = vi.spyOn(document, 'addEventListener');
    const removeSpy = vi.spyOn(document, 'removeEventListener');
    const { input } = setup();

    type(input, 'BPM');
    const added = addSpy.mock.calls.filter(([name]) => name === 'mousedown');
    expect(added.length).toBe(1);
    expect(added[0][2]).toBe(true);

    press(input, 'Escape');
    const removed = removeSpy.mock.calls.filter(([name]) => name === 'mousedown');
    expect(removed.length).toBe(1);
    expect(removed[0][1]).toBe(added[0][1]);
    expect(removed[0][2]).toBe(true);
  });

  it('BLUR: closes the popup', () => {
    const { input, handle } = setup();
    type(input, 'BPM');
    expect(handle.isOpen()).toBe(true);

    input.dispatchEvent(new Event('blur'));
    expect(handle.isOpen()).toBe(false);
  });

  it('HIGHLIGHT SAFETY: a catalog value containing markup renders as literal text with match spans', () => {
    const hostile = 'SR<b>01:BPM:X';
    const { input } = setup([hostile, ...CATALOG]);
    type(input, 'BPM');

    const row = optionsOf(input).find((r) => r.textContent === hostile);
    if (!row) throw new Error('hostile catalog value not rendered');
    // No element was created from the markup...
    expect(row.querySelector('b')).toBeNull();
    // ...but real highlight spans wrap the matched slices. Case-insensitive:
    // the fuzzy matcher may bind the query's "B" to the "b" inside "<b>".
    const marks = Array.from(row.querySelectorAll('.channel-combobox-match'));
    expect(marks.length).toBeGreaterThan(0);
    expect(marks.map((m) => m.textContent).join('').toUpperCase()).toContain('BPM');
  });

  it('FREE-FORM: a value outside the catalog keeps the popup closed and the input untouched', () => {
    const { input, holder, handle } = setup();
    /** @type {KeyboardEvent[]} */
    const seen = [];
    holder.addEventListener('keydown', (e) => seen.push(e));

    type(input, 'MY:CUSTOM:CHANNEL');

    expect(handle.isOpen()).toBe(false);
    expect(input.value).toBe('MY:CUSTOM:CHANNEL');
    // With the popup closed the combobox never touches keys: Enter reaches
    // the form unconsumed.
    const e = press(input, 'Enter');
    expect(e.defaultPrevented).toBe(false);
    expect(seen.filter((k) => k.key === 'Enter').length).toBe(1);
  });

  it('EMPTY: clearing the field closes the popup instead of browsing the whole catalog', () => {
    const { input, handle } = setup();
    type(input, 'BPM');
    expect(handle.isOpen()).toBe(true);

    type(input, '');
    expect(handle.isOpen()).toBe(false);
  });

  it('TRUNCATION: an over-cap pool renders RENDER_CAP options plus a non-option footer that arrows skip', () => {
    const big = Array.from(
      { length: RENDER_CAP + 10 },
      (_, i) => `SR${String(i).padStart(3, '0')}:BPM:X`
    );
    const { input } = setup(big);
    type(input, 'BPM');

    const rows = optionsOf(input);
    expect(rows.length).toBe(RENDER_CAP);
    const footer = wrapperOf(input).querySelector('.channel-combobox-more');
    if (!footer) throw new Error('truncation footer missing');
    expect(footer.getAttribute('role')).toBeNull();
    expect(footer.textContent).toContain('10 more');

    // ArrowUp from unarmed lands on the last real option, not the footer.
    press(input, 'ArrowUp');
    expect(input.getAttribute('aria-activedescendant')).toBe(rows[rows.length - 1].id);
  });

  it('DESTROY: unwraps the input, restores its attributes, and detaches all listeners', () => {
    const { input, holder, handle } = setup();
    type(input, 'BPM');

    handle.destroy();

    expect(input.parentElement).toBe(holder);
    expect(holder.querySelector('.channel-combobox')).toBeNull();
    expect(input.hasAttribute('role')).toBe(false);
    expect(input.hasAttribute('aria-expanded')).toBe(false);
    expect(input.hasAttribute('aria-controls')).toBe(false);

    // Typing after destroy is inert: no popup DOM exists to open.
    type(input, 'BPM');
    expect(handle.isOpen()).toBe(false);
    expect(document.querySelector('[role="listbox"]')).toBeNull();
  });
});
