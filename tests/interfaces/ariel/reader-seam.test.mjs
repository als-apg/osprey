// @ts-check
/**
 * The ARIEL reader seam's registry and helpers:
 *   npx vitest run tests/interfaces/ariel/reader-seam.test.mjs
 *
 * Sections, tabs and actions register by slug id; registering an id again
 * replaces it in place, and a bad id throws. Keys register by
 * `KeyboardEvent.key`. `place` renders into a slot only while the entry (and
 * an extra check) is current, logs a thrown render and renders nothing. The
 * thread section is seeded first.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  seam,
  registerReaderSection,
  registerReaderTab,
  registerReaderAction,
  registerReaderKey,
  place,
  renderActionButtons,
  renderTabStrip,
} from '../../../src/osprey/interfaces/ariel/static/js/reader-seam.js';
import { threadSection } from '../../../src/osprey/interfaces/ariel/static/js/reader-thread.js';

/** @type {{sections: any[], tabs: any[], actions: any[], keys: Map<string, any>}} */
let saved;

beforeEach(() => {
  saved = {
    sections: [...seam.sections],
    tabs: [...seam.tabs],
    actions: [...seam.actions],
    keys: new Map(seam.keys),
  };
});

afterEach(() => {
  seam.sections.splice(0, seam.sections.length, ...saved.sections);
  seam.tabs.splice(0, seam.tabs.length, ...saved.tabs);
  seam.actions.splice(0, seam.actions.length, ...saved.actions);
  seam.keys.clear();
  for (const [k, v] of saved.keys) seam.keys.set(k, v);
  vi.restoreAllMocks();
});

/**
 * A context whose `isCurrent` reads a flag the test flips.
 * @param {{current: boolean}} [state]
 * @returns {any}
 */
function ctx(state = { current: true }) {
  return {
    container: document.createElement('div'),
    entry: { entry_id: 'e1' },
    entryId: 'e1',
    callbacks: {},
    isCurrent: () => state.current,
  };
}

/** @param {string} html */
function dom(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  return div;
}

const noop = () => '';

describe('registry', () => {
  test('seeds the thread section first, and no tabs, actions or keys', () => {
    expect(saved.sections[0]).toBe(threadSection);
    expect(saved.tabs).toEqual([]);
    expect(saved.actions).toEqual([]);
    expect(saved.keys.size).toBe(0);
  });

  test('appends sections in registration order', () => {
    const a = { id: 'a', render: noop };
    const b = { id: 'b-2', render: noop };
    registerReaderSection(a);
    registerReaderSection(b);
    expect(seam.sections).toEqual([threadSection, a, b]);
  });

  test('replaces a section registered again under the same id, in place', () => {
    const a = { id: 'a', render: noop };
    const b = { id: 'b', render: noop };
    const a2 = { id: 'a', render: () => 'again' };
    registerReaderSection(a);
    registerReaderSection(b);
    registerReaderSection(a2);
    expect(seam.sections).toEqual([threadSection, a2, b]);
  });

  test('can replace the seeded thread section', () => {
    const replacement = { id: 'thread', render: noop };
    registerReaderSection(replacement);
    expect(seam.sections).toEqual([replacement]);
  });

  test('registers and replaces tabs and actions the same way', () => {
    const t1 = { id: 'one', label: 'One', render: noop };
    const t1b = { id: 'one', label: 'One again', render: noop };
    registerReaderTab(t1);
    registerReaderTab(t1b);
    expect(seam.tabs).toEqual([t1b]);

    const run = vi.fn();
    const act = { id: 'go', label: 'Go', run };
    const act2 = { id: 'stop', label: 'Stop', run };
    registerReaderAction(act);
    registerReaderAction(act2);
    expect(seam.actions).toEqual([act, act2]);
  });

  test.each(['', 'Upper', '1lead', 'has space', 'a_b', 'a"b', '-a'])(
    'rejects the bad id %j',
    (id) => {
      expect(() => registerReaderSection({ id, render: noop })).toThrow(/Bad reader feature id/);
      expect(() => registerReaderTab({ id, label: 'x', render: noop })).toThrow(
        /Bad reader feature id/
      );
      expect(() => registerReaderAction({ id, label: 'x', run: () => {} })).toThrow(
        /Bad reader feature id/
      );
      expect(seam.sections).toEqual(saved.sections);
      expect(seam.tabs).toEqual([]);
      expect(seam.actions).toEqual([]);
    }
  );

  test('registers a key, and a second registration of the key replaces it', () => {
    const first = vi.fn();
    const second = vi.fn();
    registerReaderKey('x', first);
    expect(seam.keys.get('x')).toBe(first);
    registerReaderKey('x', second);
    expect(seam.keys.get('x')).toBe(second);
    expect(seam.keys.size).toBe(1);
  });
});

describe('place', () => {
  test('does nothing without a slot, and does not render', async () => {
    const render = vi.fn(() => 'x');
    await place(ctx(), render, null);
    expect(render).not.toHaveBeenCalled();
  });

  test('sets a string result as the slot HTML, passing the context', async () => {
    const slot = document.createElement('div');
    const c = ctx();
    const render = vi.fn(async () => '<p class="out">hi</p>');
    await place(c, render, slot);
    expect(render).toHaveBeenCalledWith(c);
    expect(slot.querySelector('p.out')?.textContent).toBe('hi');
  });

  test('places a node result, keeping the node itself', async () => {
    const slot = document.createElement('div');
    slot.innerHTML = '<span>old</span>';
    const node = document.createElement('section');
    await place(ctx(), () => node, slot);
    expect(slot.children).toHaveLength(1);
    expect(slot.firstElementChild).toBe(node);
  });

  test.each([null, undefined])('empties the slot for a %s result', async (out) => {
    const slot = document.createElement('div');
    slot.innerHTML = '<span>old</span>';
    await place(ctx(), () => out, slot);
    expect(slot.childNodes).toHaveLength(0);
  });

  test('drops the result once the entry is no longer current', async () => {
    const slot = document.createElement('div');
    slot.innerHTML = '<span>old</span>';
    const state = { current: true };
    /** @type {(v: string) => void} */
    let resolve = () => {};
    const pending = place(ctx(state), () => new Promise((r) => (resolve = r)), slot);
    state.current = false;
    resolve('<p>stale</p>');
    await pending;
    expect(slot.innerHTML).toBe('<span>old</span>');
  });

  test('drops the result when the extra check fails', async () => {
    const slot = document.createElement('div');
    slot.innerHTML = '<span>old</span>';
    await place(
      ctx(),
      () => '<p>new</p>',
      slot,
      () => false
    );
    expect(slot.innerHTML).toBe('<span>old</span>');
  });

  test('logs a thrown render and empties the slot', async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    const slot = document.createElement('div');
    slot.innerHTML = '<span>old</span>';
    const boom = new Error('boom');
    await place(ctx(), () => Promise.reject(boom), slot);
    expect(error).toHaveBeenCalledWith('Reader feature failed:', boom);
    expect(slot.childNodes).toHaveLength(0);
  });

  test('a thrown render for a stale entry leaves the slot alone', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const slot = document.createElement('div');
    slot.innerHTML = '<span>old</span>';
    await place(
      ctx({ current: false }),
      () => {
        throw new Error('boom');
      },
      slot
    );
    expect(slot.innerHTML).toBe('<span>old</span>');
  });
});

describe('renderActionButtons', () => {
  test('is empty without actions', () => {
    expect(renderActionButtons()).toBe('');
  });

  test('renders one button per action, in order, with escaped label and optional title', () => {
    registerReaderAction({ id: 'first', label: 'First <b>', run: () => {} });
    registerReaderAction({ id: 'second', label: 'Second', title: 'Do "it" <now>', run: () => {} });
    const div = dom(renderActionButtons());
    const buttons = [...div.querySelectorAll('button')];
    expect(buttons.map((b) => b.dataset.readerAction)).toEqual(['first', 'second']);
    expect(buttons[0].textContent).toBe('First <b>');
    expect(buttons[0].querySelector('b')).toBeNull();
    expect(buttons[0].hasAttribute('title')).toBe(false);
    expect(buttons[1].getAttribute('title')).toBe('Do "it" <now>');
    expect(buttons.every((b) => b.type === 'button')).toBe(true);
  });
});

describe('renderTabStrip', () => {
  test('is empty without registered tabs', () => {
    expect(renderTabStrip()).toBe('');
  });

  test('leads with the selected "Entry" tab, then each registered tab unselected', () => {
    registerReaderTab({ id: 'enhancements', label: 'Enhancements', render: noop });
    registerReaderTab({ id: 'raw', label: '<Raw>', render: noop });
    const div = dom(renderTabStrip());
    const list = div.querySelector('.entry-detail-tabs[role="tablist"]');
    expect(list).not.toBeNull();
    const tabs = [...(list?.querySelectorAll('[role="tab"]') ?? [])];
    expect(tabs.map((t) => t.getAttribute('data-reader-tab'))).toEqual(['', 'enhancements', 'raw']);
    expect(tabs.map((t) => t.textContent)).toEqual(['Entry', 'Enhancements', '<Raw>']);
    expect(tabs.map((t) => t.getAttribute('aria-selected'))).toEqual(['true', 'false', 'false']);
  });
});
