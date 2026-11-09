// @ts-check
/**
 * The record of panes in windows of their own, on its own and against the real
 * dock engine, with a placement the test injects:
 *   npx vitest run tests/interfaces/web_terminal/dock-detached.test.mjs
 *
 * dock-apply.test.mjs covers the same module through the holder, with OSPREY's
 * placement. Here the placement is a recording stand-in, so each test can name
 * what the module hands it and what it does with the answer.
 */

import { describe, expect, test, vi } from 'vitest';

import { tileOf } from '../../../src/osprey/interfaces/web_terminal/static/vendor/dock-engine.mjs';
import {
  DETACHED,
  createDetachedPanes,
  panelRecordOf,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-detached.js';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/dock-detached.js').Placement} Placement */

const CTX = /** @type {any} */ ({});

/** @returns {any} alpha above beta, beside the terminal */
const stacked = () => ({
  split: 'row',
  items: [
    {
      split: 'column',
      size: 0.6,
      items: [
        { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.5 },
        { tile: 't3', tabs: ['beta'], showing: 'beta', size: 0.5 },
      ],
    },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
  ],
});

/**
 * A placement rule that answers `layout` as the accepted result.
 * @param {any} layout
 */
const landsAs = (layout) => vi.fn(() => ({ layout, changed: true }));

/** A placement that has no place for any pane. */
const nowhere = () => ({ reattach: vi.fn(() => null), persist: vi.fn(() => null) });

describe('leave', () => {
  test('closes the tile and keeps what brings the pane back', () => {
    const panes = createDetachedPanes(nowhere());
    const result = panes.leave(stacked(), 'beta', { title: 'Beta' }, '/panel/beta/page');

    expect(tileOf(result?.layout ?? null, 'beta')).toBeNull();
    expect(panes.queries.isDetached('beta')).toBe(true);
    expect(panes.queries.detachedKeys()).toEqual(['beta']);
    expect(panes.queries.detachedEntry('beta')).toMatchObject({
      neighbourKey: 'alpha',
      panelRecord: { title: 'Beta' },
      page: '/panel/beta/page',
      seq: 1,
    });
  });

  test('a pane with no tile leaves as it is, with no record to come back by', () => {
    const panes = createDetachedPanes(nowhere());
    const layout = stacked();

    expect(panes.leave(layout, 'gamma', undefined, '/panel/gamma/')).toEqual({
      layout,
      changed: false,
    });
    expect(panes.queries.detachedEntry('gamma')).toMatchObject({
      record: null,
      neighbourKey: null,
      page: '/panel/gamma/',
    });
  });

  test('stores nothing for a pane already detached', () => {
    const panes = createDetachedPanes(nowhere());
    panes.leave(stacked(), 'beta', undefined, null);

    expect(panes.leave(stacked(), 'beta', undefined, null)).toBeNull();
    expect(panes.queries.detachedKeys()).toEqual(['beta']);
  });
});

describe('comeBack', () => {
  test('finds the neighbour again by key, and drops the entry when the pane lands', () => {
    const after = stacked();
    const placement = { reattach: landsAs(after), persist: vi.fn(() => null) };
    const panes = createDetachedPanes(placement);
    const left = panes.leave(stacked(), 'beta', undefined, null);
    expect(left).not.toBeNull();
    const remaining = left?.layout ?? null;

    const result = panes.comeBack(remaining, 'beta', CTX);

    expect(result?.layout).toBe(after);
    expect(panes.queries.isDetached('beta')).toBe(false);
    const handed = /** @type {any[]} */ (placement.reattach.mock.calls[0]);
    expect(handed[2].record.neighbour).toBe(tileOf(remaining, 'alpha')?.tile);
  });

  test('keeps the entry when the placement finds no place', () => {
    const panes = createDetachedPanes(nowhere());
    const left = panes.leave(stacked(), 'beta', undefined, null);

    expect(panes.comeBack(left?.layout ?? null, 'beta', CTX)).toBeNull();
    expect(panes.queries.isDetached('beta')).toBe(true);
  });

  test('keeps the entry when the landing would crowd a tile', () => {
    const crowded = {
      tile: 't1',
      tabs: ['alpha', 'beta', 'terminal'],
      showing: 'alpha',
    };
    const placement = { reattach: landsAs(crowded), persist: vi.fn(() => null) };
    const panes = createDetachedPanes(placement);
    const left = panes.leave(stacked(), 'beta', undefined, null);

    expect(panes.comeBack(left?.layout ?? null, 'beta', CTX)).toBeNull();
    expect(panes.queries.isDetached('beta')).toBe(true);
  });

  test('without a placement a pane can leave but not come back', () => {
    const panes = createDetachedPanes(null);
    const left = panes.leave(stacked(), 'beta', undefined, null);

    expect(left).not.toBeNull();
    expect(panes.comeBack(left?.layout ?? null, 'beta', CTX)).toBeNull();
    expect(panes.queries.isDetached('beta')).toBe(true);
  });

  test('refuses a pane that is not detached', () => {
    const placement = nowhere();
    const panes = createDetachedPanes(placement);

    expect(panes.comeBack(stacked(), 'beta', CTX)).toBeNull();
    expect(placement.reattach).not.toHaveBeenCalled();
  });
});

describe('forget', () => {
  test('drops the entry and announces it, once', () => {
    const panes = createDetachedPanes(nowhere());
    panes.leave(stacked(), 'beta', undefined, null);
    const heard = vi.fn();
    const unsubscribe = panes.queries.onDetachedChange(heard);

    expect(panes.forget('beta')).toBe(true);
    expect(panes.forget('beta')).toBe(false);
    expect(heard).toHaveBeenCalledTimes(1);

    unsubscribe();
    panes.changed();
    expect(heard).toHaveBeenCalledTimes(1);
  });

  test('a listener that throws does not stop the next one', () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    const panes = createDetachedPanes(nowhere());
    const heard = vi.fn();
    panes.queries.onDetachedChange(() => {
      throw new Error('boom');
    });
    panes.queries.onDetachedChange(heard);

    panes.changed();

    expect(heard).toHaveBeenCalledTimes(1);
    expect(logged).toHaveBeenCalledOnce();
    logged.mockRestore();
  });
});

describe('docks and prune', () => {
  test('docks names a layout that holds a detached pane', () => {
    const panes = createDetachedPanes(nowhere());
    expect(panes.docks(stacked())).toBe(false);

    const left = panes.leave(stacked(), 'beta', undefined, null);

    expect(panes.docks(stacked())).toBe(true);
    expect(panes.docks(left?.layout ?? null)).toBe(false);
  });

  test('prune closes the pane in an incoming layout and takes that close as its way back', () => {
    const panes = createDetachedPanes(nowhere());
    panes.leave(stacked(), 'beta', undefined, null);

    const pruned = panes.prune(stacked());

    expect(tileOf(pruned, 'beta')).toBeNull();
    expect(tileOf(pruned, 'alpha')).not.toBeNull();
    expect(panes.queries.detachedEntry('beta')?.neighbourKey).toBe('alpha');
  });

  test('prune forgets the neighbour when the incoming layout says nothing about the pane', () => {
    const panes = createDetachedPanes(nowhere());
    const left = panes.leave(stacked(), 'beta', undefined, null);

    const pruned = panes.prune(left?.layout ?? null);

    expect(pruned).toBe(left?.layout);
    expect(panes.queries.detachedEntry('beta')?.neighbourKey).toBeNull();
  });
});

describe('persisted', () => {
  test('docks each pane on a copy, newest first, and leaves the layout alone', () => {
    const seen = /** @type {string[]} */ ([]);
    const placement = {
      reattach: vi.fn(() => null),
      persist: vi.fn((layout, key) => {
        seen.push(key);
        return { layout, changed: true };
      }),
    };
    const panes = createDetachedPanes(placement);
    const first = panes.leave(stacked(), 'beta', undefined, null);
    const second = panes.leave(first?.layout ?? null, 'alpha', undefined, null);
    const live = second?.layout ?? null;

    const copy = panes.persisted(live, CTX);

    expect(seen).toEqual(['alpha', 'beta']);
    expect(copy).toEqual(live);
    expect(copy).not.toBe(live);
  });

  test('a pane with no place stays out and is named in a warning', () => {
    const warned = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const panes = createDetachedPanes(nowhere());
    const left = panes.leave(stacked(), 'beta', undefined, null);

    const copy = panes.persisted(left?.layout ?? null, CTX);

    expect(tileOf(copy, 'beta')).toBeNull();
    expect(warned.mock.calls[0][0]).toContain('beta');
    warned.mockRestore();
  });
});

describe('panelRecordOf', () => {
  test('takes the title and params of a panel entry', () => {
    expect(panelRecordOf({ title: 'Beta', params: { a: 1 }, id: 'x' })).toEqual({
      title: 'Beta',
      params: { a: 1 },
    });
  });

  test('leaves out what the entry does not have', () => {
    expect(panelRecordOf({ title: 3, params: null })).toEqual({});
    expect(panelRecordOf(undefined)).toBeUndefined();
  });
});

test('DETACHED is the refusal name a holder reports', () => {
  expect(DETACHED).toBe('detached');
});
