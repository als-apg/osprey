/**
 * Unit tests for the vertical panel rail DOM renderer.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-rail.test.mjs
 *
 * panel-rail.js is a pure DOM module: it builds the 74px rail markup and exposes
 * imperative mutators (active / enabled / attention / non-destructive
 * append & remove) that panel-manager's state machine drives. These tests pin
 * that DOM contract — selectors, data-* attributes, class hooks, callback
 * wiring — the same contract the Playwright browser suite selects on.
 *
 * The rail is a MEMBERSHIP list: an entry exists iff the panel is in the rail.
 * There is no dimmed/closed state — removal takes the node out of the DOM.
 *
 * The rail carries NO per-panel health readout: backend liveness is reported by
 * the SYSTEM panel's `web_panels` health category. `.disabled` is the only
 * health-derived state the rail renders, and the absence of an LED node is
 * asserted below so it cannot creep back in.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply. The environment is happy-dom
 * (vitest.config.js), so `document` is a global.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import {
  createRail,
  addEntry,
  removeEntry,
  getEntry,
  setActive,
  setEntryEnabled,
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

  test('each entry carries icon and label sub-nodes', () => {
    createRail(rail, PANELS);
    const first = getEntry(rail, 'artifacts');
    expect(first?.querySelector('.panel-rail-icon')?.getAttribute('data-icon')).toBe('artifacts');
    expect(first?.querySelector('.panel-rail-label')?.textContent).toBe('WORKSPACE');
  });

  test('entries render no health LED — liveness lives in the SYSTEM panel', () => {
    createRail(rail, PANELS);
    expect(rail.querySelector('.panel-rail-led')).toBeNull();
  });

  test('entries start disabled and unselected', () => {
    createRail(rail, PANELS);
    const first = getEntry(rail, 'artifacts');
    expect(first?.classList.contains('disabled')).toBe(true);
    expect(first?.getAttribute('aria-selected')).toBe('false');
    expect(first?.getAttribute('title')).toBe('WORKSPACE');
  });

  test('entries never render the retired closed/dimmed state', () => {
    createRail(rail, PANELS);
    expect(rail.querySelector('.panel-rail-closed')).toBeNull();
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

  test('never renders a ＋ inside the rail — the add control is the template\'s sibling #panel-add-btn', () => {
    createRail(rail, PANELS);
    expect(rail.querySelector('.panel-rail-add')).toBeNull();
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

  test('popout affordance renders only when onPopout is provided', () => {
    createRail(rail, PANELS);
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-popout')).toBeNull();

    rail = freshRail();
    createRail(rail, PANELS, { onPopout: () => {} });
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-popout')?.textContent).toBe('↗');
  });

  test('clicking popout invokes onPopout without activating the entry', () => {
    /** @type {string[]} */
    const activated = [];
    /** @type {string[]} */
    const popped = [];
    createRail(rail, PANELS, {
      onActivate: (id) => activated.push(id),
      onPopout: (id) => popped.push(id),
    });
    /** @type {HTMLElement} */ (
      /** @type {HTMLElement} */ (getEntry(rail, 'ariel')).querySelector('.panel-rail-popout')
    ).click();
    expect(popped).toEqual(['ariel']);
    expect(activated).toEqual([]);
  });

  test('close and popout occupy opposite corners of the same entry', () => {
    createRail(rail, PANELS, { onClose: () => {}, onPopout: () => {} });
    const entry = /** @type {HTMLElement} */ (getEntry(rail, 'ariel'));
    expect(entry.querySelector('.panel-rail-close')).toBeTruthy();
    expect(entry.querySelector('.panel-rail-popout')).toBeTruthy();
  });

  test('open-beside affordance renders only when onOpenBeside is provided', () => {
    createRail(rail, PANELS);
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-beside')).toBeNull();

    rail = freshRail();
    createRail(rail, PANELS, { onOpenBeside: () => {} });
    expect(getEntry(rail, 'artifacts')?.querySelector('.panel-rail-beside')?.textContent).toBe('⊞');
  });

  test('clicking open-beside invokes onOpenBeside without activating the entry', () => {
    /** @type {string[]} */
    const activated = [];
    /** @type {string[]} */
    const beside = [];
    createRail(rail, PANELS, {
      onActivate: (id) => activated.push(id),
      onOpenBeside: (id) => beside.push(id),
    });
    /** @type {HTMLElement} */ (
      /** @type {HTMLElement} */ (getEntry(rail, 'ariel')).querySelector('.panel-rail-beside')
    ).click();
    expect(beside).toEqual(['ariel']);
    expect(activated).toEqual([]);
  });

});

describe('drag from the rail (onDragStart / onDragEnd)', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  /** Build a dragstart-shaped event happy-dom can dispatch (no DragEvent there).
   *  @param {string} type */
  function dragEvent(type) {
    const ev = new Event(type, { bubbles: true, cancelable: true });
    /** @type {any} */ (ev).dataTransfer = {
      data: /** @type {Record<string, string>} */ ({}),
      effectAllowed: '',
      setData(/** @type {string} */ k, /** @type {string} */ v) { this.data[k] = v; },
    };
    return ev;
  }

  test('entries are draggable only when onDragStart is provided', () => {
    createRail(rail, PANELS);
    expect(getEntry(rail, 'ariel')?.getAttribute('draggable')).toBeNull();

    rail = freshRail();
    createRail(rail, PANELS, { onDragStart: () => true });
    expect(getEntry(rail, 'ariel')?.getAttribute('draggable')).toBe('true');
  });

  test('dragstart hands (id, dataTransfer) to onDragStart and marks the source entry', () => {
    /** @type {any[]} */
    const calls = [];
    createRail(rail, PANELS, {
      onDragStart: (id, dt) => { calls.push([id, dt]); return true; },
    });
    const entry = /** @type {HTMLElement} */ (getEntry(rail, 'ariel'));
    const ev = dragEvent('dragstart');
    entry.dispatchEvent(ev);

    expect(calls).toEqual([['ariel', /** @type {any} */ (ev).dataTransfer]]);
    expect(entry.classList.contains('dragging')).toBe(true);
    expect(ev.defaultPrevented).toBe(false);
  });

  test('onDragStart returning false cancels the drag (no dragging state, default prevented)', () => {
    createRail(rail, PANELS, { onDragStart: () => false });
    const entry = /** @type {HTMLElement} */ (getEntry(rail, 'ariel'));
    const ev = dragEvent('dragstart');
    entry.dispatchEvent(ev);

    expect(entry.classList.contains('dragging')).toBe(false);
    expect(ev.defaultPrevented).toBe(true);
  });

  test('dragend clears the dragging state and calls onDragEnd', () => {
    /** @type {string[]} */
    const ended = [];
    createRail(rail, PANELS, {
      onDragStart: () => true,
      onDragEnd: (id) => ended.push(id),
    });
    const entry = /** @type {HTMLElement} */ (getEntry(rail, 'ariel'));
    entry.dispatchEvent(dragEvent('dragstart'));
    expect(entry.classList.contains('dragging')).toBe(true);

    entry.dispatchEvent(dragEvent('dragend'));
    expect(entry.classList.contains('dragging')).toBe(false);
    expect(ended).toEqual(['ariel']);
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
    setEntryEnabled(rail, 'artifacts', true);

    addEntry(rail, { id: 'lattice', label: 'LATTICE' });

    const artifacts = getEntry(rail, 'artifacts');
    expect(artifacts?.classList.contains('active')).toBe(true);
    expect(artifacts?.classList.contains('disabled')).toBe(false);
  });

  test('appends at the end of the rail', () => {
    createRail(rail, PANELS);
    addEntry(rail, { id: 'lattice', label: 'LATTICE' });
    const last = rail.children[rail.children.length - 1];
    expect(last.getAttribute('data-panel-id')).toBe('lattice');
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
});

describe('removeEntry', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('removes the entry node, preserving the others', () => {
    removeEntry(rail, 'ariel');
    expect(entryIds(rail)).toEqual(['artifacts', 'channel-finder']);
    expect(getEntry(rail, 'ariel')).toBeNull();
  });

  test('a removed entry can be re-added (remove ≠ forget)', () => {
    removeEntry(rail, 'ariel');
    addEntry(rail, { id: 'ariel', label: 'ARIEL' });
    expect(entryIds(rail)).toEqual(['artifacts', 'channel-finder', 'ariel']);
  });

  test('is a no-op for an unknown id', () => {
    expect(() => removeEntry(rail, 'nope')).not.toThrow();
    expect(entryIds(rail)).toEqual(['artifacts', 'ariel', 'channel-finder']);
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

describe('setEntryEnabled', () => {
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

  test('is a no-op for an unknown id', () => {
    expect(() => setEntryEnabled(rail, 'nope', true)).not.toThrow();
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
