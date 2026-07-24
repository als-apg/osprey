/**
 * Unit tests for the vertical panel rail DOM renderer.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-rail.test.mjs
 *
 * panel-rail.js is a pure DOM module: it builds the 74px rail markup and exposes
 * imperative mutators (active / health / enabled / visible / non-destructive
 * append) that panel-manager's state machine drives. These tests pin that DOM
 * contract — selectors, data-* attributes, class hooks, callback wiring — the
 * same contract the Playwright browser suite selects on.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply. The environment is happy-dom
 * (vitest.config.js), so `document` is a global.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import {
  createRail,
  addEntry,
  getEntry,
  setActive,
  setHealth,
  setEntryEnabled,
  setEntryVisible,
  setEntryAttention,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js';

const PANELS = [
  { id: 'artifacts', label: 'WORKSPACE' },
  { id: 'ariel', label: 'ARIEL' },
  { id: 'channel-finder', label: 'CHANNELS' },
];

/** @returns {HTMLElement} */
function freshRail() {
  document.body.innerHTML = '';
  const el = document.createElement('nav');
  document.body.appendChild(el);
  return el;
}

/** @param {HTMLElement} rail @returns {(string | null)[]} */
function entryIds(rail) {
  return [...rail.querySelectorAll('.panel-rail-button')].map((b) =>
    b.getAttribute('data-panel-id')
  );
}

describe('createRail', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  test('renders one entry per panel, in order, with data-panel-id', () => {
    createRail(rail, PANELS);
    expect(entryIds(rail)).toEqual(['artifacts', 'ariel', 'channel-finder']);
  });

  test('marks the container as a tablist', () => {
    createRail(rail, PANELS);
    expect(rail.classList.contains('panel-rail')).toBe(true);
    expect(rail.getAttribute('role')).toBe('tablist');
  });

  test('each entry carries LED, icon, and label sub-nodes', () => {
    createRail(rail, PANELS);
    const first = getEntry(rail, 'artifacts');
    expect(first?.querySelector('.panel-rail-led')?.className).toBe('panel-rail-led offline');
    expect(first?.querySelector('.panel-rail-icon')?.getAttribute('data-icon')).toBe('artifacts');
    expect(first?.querySelector('.panel-rail-label')?.textContent).toBe('WORKSPACE');
  });

  test('entries start disabled and unselected', () => {
    createRail(rail, PANELS);
    const first = getEntry(rail, 'artifacts');
    expect(first?.classList.contains('disabled')).toBe(true);
    expect(first?.getAttribute('aria-selected')).toBe('false');
    expect(first?.getAttribute('title')).toBe('WORKSPACE');
  });

  test('label is set via textContent (no HTML injection)', () => {
    createRail(rail, [{ id: 'x', label: '<img src=x onerror=alert(1)>' }]);
    const entry = getEntry(rail, 'x');
    expect(entry?.querySelector('.panel-rail-label')?.textContent).toBe(
      '<img src=x onerror=alert(1)>'
    );
    expect(entry?.querySelector('img')).toBeNull();
  });

  test('full render replaces prior content', () => {
    createRail(rail, PANELS);
    createRail(rail, [{ id: 'okf', label: 'KNOWLEDGE' }]);
    expect(entryIds(rail)).toEqual(['okf']);
  });

  test('renders the ＋ add button only when onAdd is given', () => {
    createRail(rail, PANELS);
    expect(rail.querySelector('.panel-rail-add')).toBeNull();

    createRail(rail, PANELS, { onAdd: () => {} });
    const add = rail.querySelector('.panel-rail-add');
    expect(add?.textContent).toBe('＋');
    expect(add?.getAttribute('aria-label')).toBe('Add panel');
  });

  test('onAdd fires when the ＋ button is clicked', () => {
    let clicked = 0;
    createRail(rail, PANELS, { onAdd: () => { clicked += 1; } });
    /** @type {HTMLButtonElement} */ (rail.querySelector('.panel-rail-add')).click();
    expect(clicked).toBe(1);
  });

  test('the add button stays last, after every entry', () => {
    createRail(rail, PANELS, { onAdd: () => {} });
    const last = rail.children[rail.children.length - 1];
    expect(last.classList.contains('panel-rail-add')).toBe(true);
  });

  test('hides entries absent from the visible set', () => {
    createRail(rail, PANELS, { visible: new Set(['artifacts', 'channel-finder']) });
    expect(getEntry(rail, 'artifacts')?.classList.contains('panel-rail-hidden')).toBe(false);
    expect(getEntry(rail, 'ariel')?.classList.contains('panel-rail-hidden')).toBe(true);
    expect(getEntry(rail, 'channel-finder')?.classList.contains('panel-rail-hidden')).toBe(false);
  });
});

describe('entry interactions', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  test('clicking an entry invokes onActivate with its id', () => {
    /** @type {string[]} */
    const activated = [];
    createRail(rail, PANELS, { onActivate: (id) => activated.push(id) });
    /** @type {HTMLButtonElement} */ (getEntry(rail, 'ariel')).click();
    expect(activated).toEqual(['ariel']);
  });

  test('close affordance renders only when onClose is provided', () => {
    createRail(rail, PANELS);
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-close')).toBeNull();

    rail = freshRail();
    createRail(rail, PANELS, { onClose: () => {} });
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-close')?.textContent).toBe('×');
  });

  test('clicking close invokes onClose without activating the entry', () => {
    /** @type {string[]} */
    const activated = [];
    /** @type {string[]} */
    const closed = [];
    createRail(rail, PANELS, {
      onActivate: (id) => activated.push(id),
      onClose: (id) => closed.push(id),
    });
    /** @type {HTMLElement} */ (
      /** @type {HTMLElement} */ (getEntry(rail, 'ariel')).querySelector('.panel-rail-close')
    ).click();
    expect(closed).toEqual(['ariel']);
    expect(activated).toEqual([]);
  });
});

describe('addEntry (non-destructive)', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  test('appends a new entry, preserving existing ones', () => {
    createRail(rail, PANELS);
    addEntry(rail, { id: 'lattice', label: 'LATTICE' });
    expect(entryIds(rail)).toEqual(['artifacts', 'ariel', 'channel-finder', 'lattice']);
  });

  test('preserves live state on existing entries (no rebuild)', () => {
    createRail(rail, PANELS);
    setActive(rail, 'artifacts');
    setHealth(rail, 'artifacts', true);
    setEntryEnabled(rail, 'artifacts', true);

    addEntry(rail, { id: 'lattice', label: 'LATTICE' });

    const artifacts = getEntry(rail, 'artifacts');
    expect(artifacts?.classList.contains('active')).toBe(true);
    expect(artifacts?.classList.contains('disabled')).toBe(false);
    expect(artifacts?.querySelector('.panel-rail-led')?.className).toBe('panel-rail-led healthy');
  });

  test('inserts before the ＋ button so add stays last', () => {
    createRail(rail, PANELS, { onAdd: () => {} });
    addEntry(rail, { id: 'lattice', label: 'LATTICE' });
    const last = rail.children[rail.children.length - 1];
    expect(last.classList.contains('panel-rail-add')).toBe(true);
    expect(entryIds(rail)).toEqual(['artifacts', 'ariel', 'channel-finder', 'lattice']);
  });

  test('is idempotent by id — no duplicate node', () => {
    createRail(rail, PANELS);
    const first = addEntry(rail, { id: 'ariel', label: 'ARIEL' });
    expect(first).toBe(getEntry(rail, 'ariel'));
    expect(entryIds(rail).filter((id) => id === 'ariel')).toEqual(['ariel']);
  });

  test('wires onActivate on the appended entry', () => {
    /** @type {string[]} */
    const activated = [];
    createRail(rail, PANELS);
    addEntry(rail, { id: 'lattice', label: 'LATTICE' }, { onActivate: (id) => activated.push(id) });
    /** @type {HTMLButtonElement} */ (getEntry(rail, 'lattice')).click();
    expect(activated).toEqual(['lattice']);
  });

  test('honors the visible set for the appended entry', () => {
    createRail(rail, PANELS);
    addEntry(rail, { id: 'lattice', label: 'LATTICE' }, { visible: new Set(['artifacts']) });
    expect(getEntry(rail, 'lattice')?.classList.contains('panel-rail-hidden')).toBe(true);
  });
});

describe('setActive', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('marks exactly one entry active and updates aria-selected', () => {
    setActive(rail, 'ariel');
    expect(getEntry(rail, 'ariel')?.classList.contains('active')).toBe(true);
    expect(getEntry(rail, 'ariel')?.getAttribute('aria-selected')).toBe('true');
    expect(getEntry(rail, 'artifacts')?.classList.contains('active')).toBe(false);
    expect(getEntry(rail, 'artifacts')?.getAttribute('aria-selected')).toBe('false');
  });

  test('switching active clears the previous one', () => {
    setActive(rail, 'artifacts');
    setActive(rail, 'channel-finder');
    expect(getEntry(rail, 'artifacts')?.classList.contains('active')).toBe(false);
    expect(getEntry(rail, 'channel-finder')?.classList.contains('active')).toBe(true);
  });

  test('a null / unknown id clears active on every entry', () => {
    setActive(rail, 'artifacts');
    setActive(rail, null);
    expect(rail.querySelectorAll('.panel-rail-button.active').length).toBe(0);
  });
});

describe('setHealth', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('flips the LED between healthy and offline', () => {
    setHealth(rail, 'artifacts', true);
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-led')?.className).toBe(
      'panel-rail-led healthy'
    );
    setHealth(rail, 'artifacts', false);
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-led')?.className).toBe(
      'panel-rail-led offline'
    );
  });

  test('is a no-op for an unknown id', () => {
    expect(() => setHealth(rail, 'nope', true)).not.toThrow();
  });
});

describe('setEntryEnabled / setEntryVisible', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('setEntryEnabled toggles the disabled class', () => {
    setEntryEnabled(rail, 'artifacts', true);
    expect(getEntry(rail, 'artifacts')?.classList.contains('disabled')).toBe(false);
    setEntryEnabled(rail, 'artifacts', false);
    expect(getEntry(rail, 'artifacts')?.classList.contains('disabled')).toBe(true);
  });

  test('setEntryVisible toggles the panel-rail-hidden class', () => {
    setEntryVisible(rail, 'ariel', false);
    expect(getEntry(rail, 'ariel')?.classList.contains('panel-rail-hidden')).toBe(true);
    setEntryVisible(rail, 'ariel', true);
    expect(getEntry(rail, 'ariel')?.classList.contains('panel-rail-hidden')).toBe(false);
  });

  test('both are no-ops for an unknown id', () => {
    expect(() => setEntryEnabled(rail, 'nope', true)).not.toThrow();
    expect(() => setEntryVisible(rail, 'nope', false)).not.toThrow();
  });
});

describe('setEntryAttention', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('on sets the persistent badge class and fires the transient flash', () => {
    expect(setEntryAttention(rail, 'ariel', true)).toBe(true);
    const entry = getEntry(rail, 'ariel');
    expect(entry?.classList.contains('agent-attention')).toBe(true);
    expect(entry?.classList.contains('agent-flash')).toBe(true);
  });

  test('off removes the badge without force-stopping an in-flight flash', () => {
    setEntryAttention(rail, 'ariel', true);
    expect(setEntryAttention(rail, 'ariel', false)).toBe(true);
    const entry = getEntry(rail, 'ariel');
    expect(entry?.classList.contains('agent-attention')).toBe(false);
    // No animationend has fired in happy-dom, so the flash class must survive.
    expect(entry?.classList.contains('agent-flash')).toBe(true);
  });

  test('returns false and does not throw for an unknown id', () => {
    expect(() => setEntryAttention(rail, 'nope', true)).not.toThrow();
    expect(setEntryAttention(rail, 'nope', true)).toBe(false);
  });

  test('badge persists across setEntryVisible hide/show toggles', () => {
    setEntryAttention(rail, 'ariel', true);
    setEntryVisible(rail, 'ariel', false);
    expect(getEntry(rail, 'ariel')?.classList.contains('agent-attention')).toBe(true);
    setEntryVisible(rail, 'ariel', true);
    expect(getEntry(rail, 'ariel')?.classList.contains('agent-attention')).toBe(true);
  });

  test('a hidden entry accepts the badge', () => {
    setEntryVisible(rail, 'ariel', false);
    expect(setEntryAttention(rail, 'ariel', true)).toBe(true);
    const entry = getEntry(rail, 'ariel');
    expect(entry?.classList.contains('panel-rail-hidden')).toBe(true);
    expect(entry?.classList.contains('agent-attention')).toBe(true);
  });

  test('is class-only: no child nodes or attributes added or removed', () => {
    const entry = /** @type {HTMLElement} */ (getEntry(rail, 'ariel'));
    const childrenBefore = [...entry.children];
    const attrsBefore = entry.getAttributeNames().sort();

    setEntryAttention(rail, 'ariel', true);
    setEntryAttention(rail, 'ariel', false);

    expect([...entry.children]).toEqual(childrenBefore);
    expect(entry.getAttributeNames().sort()).toEqual(attrsBefore);
  });
});

describe('getEntry', () => {
  test('returns the button for a known id and null otherwise', () => {
    const rail = freshRail();
    createRail(rail, PANELS);
    expect(getEntry(rail, 'artifacts')?.getAttribute('data-panel-id')).toBe('artifacts');
    expect(getEntry(rail, 'ghost')).toBeNull();
  });
});
