// @ts-check
/**
 * The params that belong to each dock tile (dock-params.js): read and written
 * where the tile is (on its live dockview panel, in its detached pane's record,
 * or pending between a restore or a reattach and the apply that builds it),
 * laid over the serializer's entries while pending, and spent by the apply.
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-params.test.mjs
 */

import { describe, expect, test, vi } from 'vitest';

import { createDetachedPanes } from '../../../src/osprey/interfaces/web_terminal/static/js/dock-detached.js';
import { createPanelParams } from '../../../src/osprey/interfaces/web_terminal/static/js/dock-params.js';

/** @typedef {Record<string, any>} Params */

/** @param {string} key @returns {string} the dockview id of a key */
const idOf = (key) => 'dv-' + key;

/** @returns {any} alpha and beta in one tile, the terminal beside them */
const layoutOf = () => ({
  split: 'row',
  items: [
    { tile: 't1', tabs: ['alpha', 'beta'], showing: 'alpha', size: 0.6 },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
  ],
});

/**
 * A fake dockview API holding panels by dockview id; a panel's
 * `updateParameters` merges as dockview does (undefined deletes the field).
 * @param {Record<string, Params>} byId  the params of each live panel
 */
function fakeApi(byId) {
  /** @type {Record<string, {params: Params, api: {updateParameters: (patch: Params) => void}}>} */
  const panels = {};
  for (const [id, params] of Object.entries(byId)) {
    const panel = {
      params: { ...params },
      api: {
        /** @param {Params} patch */
        updateParameters(patch) {
          panel.params = { ...panel.params, ...patch };
          for (const [field, value] of Object.entries(patch)) {
            if (value === undefined) delete panel.params[field];
          }
        },
      },
    };
    panels[id] = panel;
  }
  return { panels, getPanel: (/** @type {string} */ id) => panels[id] };
}

/**
 * The params of one holder, over a fake dock and the real detached panes.
 * @param {Record<string, Params>} [live]  live panels' params, by KEY
 * @param {{ useApi?: boolean }} [options]
 */
function setup(live = {}, { useApi = true } = {}) {
  const byId = Object.fromEntries(Object.entries(live).map(([key, p]) => [idOf(key), p]));
  const api = fakeApi(byId);
  const panes = createDetachedPanes(null);
  const params = createPanelParams({
    api: useApi ? api : null,
    idOf,
    panes,
  });
  return { api, panes, params };
}

/**
 * Detach `key` (with no tile) carrying `panelRecord`.
 * @param {ReturnType<typeof createDetachedPanes>} panes
 * @param {string} key
 * @param {any} panelRecord
 */
function detach(panes, key, panelRecord) {
  panes.leave(layoutOf(), key, panelRecord, '/panel/' + key + '/');
}

describe('read', () => {
  test('a docked panel reads its live params, a panel with none reads undefined', () => {
    const { params } = setup({ alpha: { zoom: 2 }, beta: {} });

    expect(params.read('alpha')).toEqual({ zoom: 2 });
    expect(params.read('beta')).toBeUndefined();
  });

  test('a key with no panel, or no dock at all, reads undefined', () => {
    const { params } = setup({ alpha: { zoom: 2 } });
    expect(params.read('gamma')).toBeUndefined();

    const { params: noDock } = setup({ alpha: { zoom: 2 } }, { useApi: false });
    expect(noDock.read('alpha')).toBeUndefined();
  });

  test("a detached pane reads its record's params, not the dock's", () => {
    const { params, panes } = setup({ alpha: { zoom: 2 } });
    detach(panes, 'alpha', { title: 'Alpha', params: { zoom: 5 } });

    expect(params.read('alpha')).toEqual({ zoom: 5 });
  });
});

describe('write', () => {
  test('a docked panel takes the patch live, merged, and a field set to undefined goes', () => {
    const { params, api } = setup({ alpha: { zoom: 2, mode: 'a' } });

    expect(params.write('alpha', { zoom: 3, extra: true })).toBe(true);
    expect(api.panels['dv-alpha'].params).toEqual({ zoom: 3, mode: 'a', extra: true });

    expect(params.write('alpha', { mode: undefined })).toBe(true);
    expect(params.read('alpha')).toEqual({ zoom: 3, extra: true });
  });

  test("a detached pane's record takes the patch, merged, and the detached set announces it", () => {
    const { params, panes, api } = setup({ alpha: { zoom: 2 } });
    detach(panes, 'alpha', { title: 'Alpha', params: { zoom: 5, mode: 'a' } });
    const heard = vi.fn();
    panes.queries.onDetachedChange(heard);

    expect(params.write('alpha', { zoom: 6, mode: undefined, extra: 1 })).toBe(true);

    expect(panes.queries.detachedEntry('alpha')?.panelRecord).toEqual({
      title: 'Alpha',
      params: { zoom: 6, extra: 1 },
    });
    expect(heard).toHaveBeenCalledTimes(1);
    expect(api.panels['dv-alpha'].params).toEqual({ zoom: 2 });
  });

  test('a key with no panel anywhere returns false and changes nothing', () => {
    const { params, api } = setup({ alpha: { zoom: 2 } });

    expect(params.write('gamma', { zoom: 9 })).toBe(false);
    expect(params.read('gamma')).toBeUndefined();
    expect(api.panels['dv-alpha'].params).toEqual({ zoom: 2 });
    expect(setup({}, { useApi: false }).params.write('alpha', { zoom: 1 })).toBe(false);
  });
});

describe('restored', () => {
  test('each docked key gets the params its record stores, whatever its live panel holds', () => {
    const { params } = setup({ alpha: { zoom: 2 }, beta: { zoom: 7 } });

    const layout = layoutOf();
    const back = params.restored(layout, {
      alpha: { params: { zoom: 4 } },
      beta: { title: 'Beta' },
      terminal: { params: { font: 12 } },
    });

    expect(back).toBe(layout);
    expect(params.read('alpha')).toEqual({ zoom: 4 });
    expect(params.read('beta')).toBeUndefined();
    expect(params.read('terminal')).toEqual({ font: 12 });
  });

  test('the pending params are laid over the stored entries by dockview id and over the new-panel records', () => {
    const { params } = setup();
    params.restored(layoutOf(), { alpha: { params: { zoom: 4 } }, beta: { title: 'Beta' } });

    const stored = {
      'dv-alpha': { contentComponent: 'x', params: { zoom: 1 } },
      'dv-beta': { contentComponent: 'x', params: { zoom: 9 } },
    };
    const newPanel = vi.fn((/** @type {string} */ key) => ({ title: 'T-' + key }));
    const over = params.laidOver(stored, newPanel);

    expect(over.panels['dv-alpha']).toEqual({ contentComponent: 'x', params: { zoom: 4 } });
    expect(over.panels['dv-beta']).toEqual({ contentComponent: 'x' });
    expect(over.newPanel('alpha')).toEqual({ title: 'T-alpha', params: { zoom: 4 } });
    expect(over.newPanel('beta')).toEqual({ title: 'T-beta' });
  });

  test('without records nothing is pending: live params show and laidOver returns its arguments', () => {
    const { params } = setup({ alpha: { zoom: 2 } });
    const layout = layoutOf();

    expect(params.restored(layout)).toBe(layout);
    expect(params.read('alpha')).toEqual({ zoom: 2 });

    const stored = { 'dv-alpha': { params: { zoom: 2 } } };
    const newPanel = () => undefined;
    const over = params.laidOver(stored, newPanel);
    expect(over.panels).toBe(stored);
    expect(over.newPanel).toBe(newPanel);
  });

  test("a detached key's params go to its record, not to pending", () => {
    const { params, panes, api } = setup({ beta: { zoom: 7 } });
    detach(panes, 'beta', { title: 'Beta', params: { zoom: 5 } });

    params.restored(layoutOf(), { beta: { params: { zoom: 8 } } });

    expect(panes.queries.detachedEntry('beta')?.panelRecord).toEqual({
      title: 'Beta',
      params: { zoom: 8 },
    });
    expect(params.laidOver({ 'dv-beta': {} }, () => undefined).panels['dv-beta']).toEqual({});
    expect(api.panels['dv-beta'].params).toEqual({ zoom: 7 });
  });
});

describe('spent', () => {
  test('clears the pending params: the live panel reads again and laidOver passes through', () => {
    const { params } = setup({ alpha: { zoom: 2 } });
    params.restored(layoutOf(), { alpha: { params: { zoom: 4 } } });
    expect(params.read('alpha')).toEqual({ zoom: 4 });

    params.spent();

    expect(params.read('alpha')).toEqual({ zoom: 2 });
    const stored = { 'dv-alpha': { params: { zoom: 2 } } };
    expect(params.laidOver(stored, () => undefined).panels).toBe(stored);
  });
});

describe('write while pending', () => {
  test('merges into the pending value and succeeds before any live panel exists', () => {
    const { params } = setup();
    params.restored(layoutOf(), { alpha: { params: { zoom: 4, mode: 'a' } } });

    expect(params.write('alpha', { zoom: 5, mode: undefined })).toBe(true);

    expect(params.read('alpha')).toEqual({ zoom: 5 });
    const over = params.laidOver({ 'dv-alpha': {} }, () => undefined);
    expect(over.panels['dv-alpha']).toEqual({ params: { zoom: 5 } });
  });
});

describe('returning', () => {
  test('hands back the record without its params and keeps those pending until spent', () => {
    const { params } = setup();
    const record = { title: 'Alpha', params: { zoom: 5 } };

    expect(params.returning('alpha', record)).toEqual({ title: 'Alpha' });
    expect(record).toEqual({ title: 'Alpha', params: { zoom: 5 } });
    expect(params.read('alpha')).toEqual({ zoom: 5 });
    const over = params.laidOver({ 'dv-alpha': { title: 'Alpha' } }, () => ({ title: 'Alpha' }));
    expect(over.panels['dv-alpha']).toEqual({ title: 'Alpha', params: { zoom: 5 } });
    expect(over.newPanel('alpha')).toEqual({ title: 'Alpha', params: { zoom: 5 } });

    params.spent();
    expect(params.read('alpha')).toBeUndefined();
  });
});

describe('laidOver', () => {
  test('leaves entries and records of keys that are not pending untouched, and the stored object as given', () => {
    const { params } = setup();
    params.restored(layoutOf(), { alpha: { params: { zoom: 4 } } });

    const stored = {
      'dv-alpha': { title: 'Alpha', params: { zoom: 1 } },
      'dv-gamma': { title: 'Gamma', params: { zoom: 3 } },
    };
    const before = structuredClone(stored);
    const over = params.laidOver(stored, (key) => ({ title: 'T-' + key }));

    expect(over.panels['dv-gamma']).toBe(stored['dv-gamma']);
    expect(over.newPanel('gamma')).toEqual({ title: 'T-gamma' });
    expect(over.panels['dv-alpha']).toEqual({ title: 'Alpha', params: { zoom: 4 } });
    expect(stored).toEqual(before);
  });
});
