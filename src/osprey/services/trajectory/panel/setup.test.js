// @ts-check
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { placeRows } from './figures.js';
import { settingsFor } from './settings.js';
import {
  CLICK_PLOTS,
  NO_SESSION,
  OPEN_BPM_CONTROL,
  applyFind,
  bpmsBody,
  clickTab,
  cmsBody,
  emptyPending,
  familyCount,
  findSuggestions,
  mount,
  resolveToken,
  startFromExcluded,
  startFromGroups,
  tabEntries,
  toggleNames,
  withExcluded,
} from './setup.js';

/**
 * A catalog entry in a place chain of a ring and one cell of it.
 * @param {string} name
 * @param {string} label
 * @param {string | null} place
 * @param {number | null} s
 * @param {string | null} [plane]
 */
const entry = (name, label, place, s, plane = null) => ({
  name,
  device: `dev-${name}`,
  label,
  places: place ? [{ id: 'RING', level: 'ring' }, { id: place, level: 'cell' }] : [],
  s,
  plane,
  groups: [],
});

const CATALOG = {
  bpms: [
    entry('B1', 'Pick 1', 'CELL-A', 1),
    entry('B2', 'Pick 2', 'CELL-A', 2),
    entry('B3', 'Pick 3', 'CELL-B', 3),
    entry('B4', 'Pick 4', 'CELL-B', 4),
    entry('B5', 'Pick 5', 'CELL-C', 5),
    entry('B6', 'Pick 6', null, null),
  ],
  hcms: [entry('H1', 'Kick H1', 'CELL-A', 1.5, 'x'), entry('H2', 'Kick H2', 'CELL-B', 3.5, 'x')],
  vcms: [entry('V1', 'Kick V1', 'CELL-A', 1.6, 'y'), entry('V2', 'Kick V2', 'CELL-B', 3.6, 'y')],
  groups: [
    { id: 'all-picks', label: 'All picks', members: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'] },
    { id: 'arc-picks', label: 'Arc picks', members: ['B1', 'B2', 'B3'] },
    { id: 'same-as-all', label: 'Same as all', members: ['B6', 'B5', 'B4', 'B3', 'B2', 'B1'] },
    { id: 'kicks', label: 'Kicks', members: ['H1', 'H2', 'V1', 'V2'] },
  ],
};

/**
 * @param {string[]} [bpms]
 * @param {string[]} [cms]
 * @param {any} [catalog]
 */
function snapshotWith(bpms = [], cms = [], catalog = CATALOG) {
  return {
    model: 'ring-model',
    catalog,
    setup: null,
    solve: { config: { excluded_bpms: bpms, excluded_cms: cms, alpha: 0, gain: 1, keep_live: 1, n_discard: 0, loss_cutoff: 0, overridden: [] } },
  };
}

const READBACKS = {
  settings: [
    { id: 'pilot_tone', scope: 'global', value: 'Off', values: {} },
    { id: 'turns_tracked', scope: 'global', value: 260, values: {} },
  ],
};

/** @param {any} body */
const ok = (body) => ({ kind: 'ok', status: 200, body });
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** @param {{holding?: boolean, snapshot?: any}} [options] */
function harness({ holding = true, snapshot = snapshotWith(['B2']) } = {}) {
  /** @type {Set<(s: any) => void>} */
  const snapshotFns = new Set();
  /** @type {Map<string, (device: string) => void>} */
  const clicks = new Map();
  let current = snapshot;
  const api = {
    setupReadbacks: vi.fn(async () => ok(READBACKS)),
    setup: vi.fn(async () => ok({ moved: [], unsettled: [], setup: null })),
    solve: vi.fn(async () => ok({})),
  };
  const notify = vi.fn();
  const ctx = /** @type {any} */ ({
    api,
    notify,
    holding: () => holding,
    snapshot: () => current,
    onSnapshot(/** @type {(s: any) => void} */ fn) {
      snapshotFns.add(fn);
      return () => void snapshotFns.delete(fn);
    },
    adapter: {
      onPointClick: vi.fn((/** @type {string} */ id, /** @type {(d: string) => void} */ fn) => {
        clicks.set(id, fn);
        return () => {};
      }),
    },
  });
  const host = document.createElement('div');
  document.body.replaceChildren(host);
  mount(host, ctx);
  /** @param {string} name */
  const el = (name) => /** @type {any} */ (host.querySelector(`[data-control="${name}"]`));
  /** @param {string} tab @param {string} device */
  const square = (tab, device) => /** @type {HTMLButtonElement} */ (host.querySelector(`[data-map="${tab}"] [data-device="${device}"]`));
  /** @param {string} tab @param {string} place */
  const place = (tab, place) => /** @type {HTMLButtonElement} */ (host.querySelector(`[data-map="${tab}"] [data-place-toggle="${place}"]`));
  /** @param {string} tab @param {string} device */
  const state = (tab, device) => [square(tab, device).dataset.state, square(tab, device).dataset.pending ?? null];
  /** @param {string} name @param {string} text */
  const find = (name, text) => {
    const input = el(name);
    input.value = text;
    input.dispatchEvent(new Event('input'));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
  };
  return {
    host,
    ctx,
    api,
    notify,
    clicks,
    el,
    square,
    place,
    state,
    find,
    /** @param {any} next */
    push(next) {
      current = next;
      for (const fn of [...snapshotFns]) fn(next);
    },
  };
}

beforeEach(() => {
  document.body.replaceChildren();
});

describe('pure helpers', () => {
  it('rows the map by leaf place in ring order, placeless devices on the model row', () => {
    const rows = placeRows(tabEntries(snapshotWith(), 'bpms'), 'ring-model');
    expect(rows.map((row) => [row.label, row.entries.map((e) => e.name)])).toEqual([
      ['CELL-A', ['B1', 'B2']],
      ['CELL-B', ['B3', 'B4']],
      ['CELL-C', ['B5']],
      ['ring-model', ['B6']],
    ]);
  });

  it('toggles a set as one: all in goes out, otherwise all come back', () => {
    expect(toggleNames([], ['B1', 'B2']).sort()).toEqual(['B1', 'B2']);
    expect(toggleNames(['B1'], ['B1', 'B2'])).toEqual([]);
    expect(toggleNames(['B1', 'B2'], ['B1', 'B2'])).toEqual([]);
  });

  it('holds nothing pending once the staged set is the one in force', () => {
    const snap = snapshotWith(['B2']);
    expect(withExcluded(emptyPending(), snap, 'bpms', ['B2'])).toEqual(emptyPending());
    expect(withExcluded(emptyPending(), snap, 'bpms', ['B3', 'B2'])).toEqual({ bpms: ['B2', 'B3'], cms: null });
  });

  it('resolves a Find token by label, name, device id or place, ignoring case', () => {
    const snap = snapshotWith();
    expect(resolveToken(snap, 'bpms', 'pick 3')).toEqual({ names: ['B3'] });
    expect(resolveToken(snap, 'bpms', 'B4')).toEqual({ names: ['B4'] });
    expect(resolveToken(snap, 'bpms', 'dev-B5')).toEqual({ names: ['B5'] });
    expect(resolveToken(snap, 'bpms', 'cell-b')).toEqual({ names: ['B3', 'B4'] });
    expect(resolveToken(snap, 'cms', 'CELL-A')).toEqual({ names: ['H1', 'V1'] });
  });

  it('refuses an unknown token and a typed position list by name', () => {
    const snap = snapshotWith();
    expect(resolveToken(snap, 'bpms', 'Kick H1')).toEqual({ error: '"Kick H1" names no BPM and no place' });
    expect(applyFind([], snap, 'bpms', '[1 2 8 17]')).toEqual({
      error: '"[1 2 8 17]" is a position, not a device: find by label, name or place',
    });
    expect(applyFind([], snap, 'bpms', '3:5')).toEqual({
      error: '"3:5" is a position, not a device: find by label, name or place',
    });
    expect(applyFind([], snap, 'bpms', 'Pick 1, nowhere')).toEqual({ error: '"nowhere" names no BPM and no place' });
  });

  it('applies every token of a Find entry', () => {
    expect(applyFind(['B1'], snapshotWith(), 'bpms', 'Pick 1, CELL-C')).toEqual({ excluded: ['B5'] });
  });

  it('suggests labels and place ids, no facility word of its own', () => {
    expect(findSuggestions(snapshotWith(), 'bpms')).toEqual(
      ['Pick 1', 'Pick 2', 'Pick 3', 'Pick 4', 'Pick 5', 'Pick 6', 'RING', 'CELL-A', 'CELL-B', 'CELL-C']);
  });

  it('offers Start from only for two or more distinct non-empty sets of the tab', () => {
    const snap = snapshotWith();
    expect(startFromGroups(snap, 'bpms').map((g) => g.id)).toEqual(['all-picks', 'arc-picks']);
    expect(startFromGroups(snap, 'cms')).toEqual([]);
    expect(startFromExcluded(snap, 'bpms', 'arc-picks')).toEqual(['B4', 'B5', 'B6']);
    expect(startFromExcluded(snap, 'bpms', 'kicks')).toBeNull();
    const single = { ...CATALOG, groups: [CATALOG.groups[0]] };
    expect(startFromGroups(snapshotWith([], [], single), 'bpms')).toEqual([]);
  });

  it('counts each family in use', () => {
    expect(familyCount(snapshotWith(), 'bpms', ['B2', 'B5'])).toBe('4 of 6 BPMs in use');
    expect(familyCount(null, 'hcms', [])).toBe('0 of 0 HCMs in use');
  });

  it('builds the bodies from the pending set, refusing an empty plane', () => {
    const snap = snapshotWith(['B2']);
    expect(bpmsBody(emptyPending(), snap, { pilot_tone: 'On' })).toEqual({
      body: { settings: { pilot_tone: 'On' }, excluded_bpms: ['B2'] },
    });
    const all = { bpms: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'], cms: null };
    expect(bpmsBody(all, snap, {})).toEqual({ error: 'Keep at least one BPM in use' });
    expect(cmsBody({ bpms: null, cms: ['H1', 'V2'] }, snap)).toEqual({ body: { excluded_cms: ['H1', 'V2'] } });
    expect(cmsBody({ bpms: null, cms: ['V1', 'V2'] }, snap)).toEqual({ error: 'Keep at least one VCM in use' });
  });

  it('resolves a click to a tab only on the plots that exclude', () => {
    const snap = snapshotWith();
    expect(clickTab(snap, 'plot-traj-x', 'B3')).toBe('bpms');
    expect(clickTab(snap, 'plot-diff-y', 'B3')).toBe('bpms');
    expect(clickTab(snap, 'plot-cm-step', 'V1')).toBe('cms');
    expect(clickTab(snap, 'plot-cm-step', 'B3')).toBeNull();
    expect(clickTab(snap, 'plot-raw-x', 'B3')).toBeNull();
  });
});

describe('the setup area', () => {
  it('draws the map from the catalog with the counts in use', () => {
    const { el, state, host, square } = harness();
    expect(el('bpm-family').textContent).toBe('5 of 6 BPMs in use');
    expect(el('cm-family').textContent).toBe('H 2 of 2 · V 2 of 2 in use');
    expect(state('bpms', 'B2')).toEqual(['out', null]);
    expect(state('bpms', 'B1')).toEqual(['in', null]);
    expect(square('bpms', 'B3').title).toBe('Pick 3 · CELL-B · s = 3.0 m');
    const cmRow = host.querySelector('[data-map="cms"] [data-place="CELL-A"]');
    expect([...(cmRow?.querySelectorAll('[data-plane]') ?? [])].map((p) => p.textContent)).toEqual(['H', 'V']);
    expect(el('bpm-exclude')).not.toBeNull();
    expect(el('cm-exclude')).not.toBeNull();
  });

  it('a device click marks one pending change that Revert clears', () => {
    const { el, state, square, host } = harness();
    square('bpms', 'B4').click();
    expect(state('bpms', 'B4')).toEqual(['out', 'true']);
    expect(el('bpm-family').textContent).toBe('4 of 6 BPMs in use');
    expect(host.querySelector('[data-pending-change="bpms"]')?.textContent).toBe('pending: −Pick 4');
    el('setup-bpms-revert').click();
    expect(state('bpms', 'B4')).toEqual(['in', null]);
    expect(host.querySelector('[data-pending-change="bpms"]')).toBeNull();
  });

  it('a place click toggles every device in the place', () => {
    const { place, state } = harness();
    place('bpms', 'CELL-B').click();
    expect(state('bpms', 'B3')).toEqual(['out', 'true']);
    expect(state('bpms', 'B4')).toEqual(['out', 'true']);
    place('bpms', 'CELL-B').click();
    expect(state('bpms', 'B3')).toEqual(['in', null]);
    place('bpms', 'CELL-A').click();
    expect(state('bpms', 'B1')).toEqual(['in', null]);
    expect(state('bpms', 'B2')).toEqual(['in', 'true']);
  });

  it('Find toggles a device by label and a place by id, then empties itself', () => {
    const { find, state, el } = harness();
    find('bpm-exclude', 'Pick 5');
    expect(state('bpms', 'B5')).toEqual(['out', 'true']);
    expect(el('bpm-exclude').value).toBe('');
    find('bpm-exclude', 'CELL-B');
    expect(state('bpms', 'B3')).toEqual(['out', 'true']);
    expect(state('bpms', 'B4')).toEqual(['out', 'true']);
  });

  it('Find refuses an unknown token and a position list by name, changing nothing', () => {
    const { find, state, host, el } = harness();
    find('bpm-exclude', 'nowhere');
    expect(host.querySelector('[data-error="bpm-exclude"]')?.textContent).toBe('"nowhere" names no BPM and no place');
    expect(el('bpm-exclude').value).toBe('nowhere');
    find('bpm-exclude', '[1 2 8 17]');
    expect(host.querySelector('[data-error="bpm-exclude"]')?.textContent)
      .toBe('"[1 2 8 17]" is a position, not a device: find by label, name or place');
    for (const device of ['B1', 'B3', 'B4', 'B5', 'B6']) expect(state('bpms', device)).toEqual(['in', null]);
    expect(el('setup-bpms-revert').disabled).toBe(true);
  });

  it('the Find box is the design system typeahead', () => {
    const { el } = harness();
    expect(el('bpm-exclude').getAttribute('role')).toBe('combobox');
    expect(el('bpm-exclude').closest('.channel-combobox')).not.toBeNull();
  });

  it('holds a pending change across snapshots; Setup BPMs commits it by name', async () => {
    const { square, state, el, api, push } = harness();
    await flush();
    square('bpms', 'B4').click();
    push(snapshotWith(['B2']));
    expect(state('bpms', 'B4')).toEqual(['out', 'true']);
    el('setup-bpms').click();
    await flush();
    expect(api.setup).toHaveBeenCalledWith({ settings: {}, excluded_bpms: ['B2', 'B4'] });
    push(snapshotWith(['B2', 'B4']));
    expect(state('bpms', 'B4')).toEqual(['out', null]);
    push(snapshotWith(['B1']));
    expect(state('bpms', 'B4')).toEqual(['in', null]);
    expect(state('bpms', 'B1')).toEqual(['out', null]);
  });

  it('counts a pending setting as an edit of the BPM tab', async () => {
    const { el, ctx } = harness();
    await flush();
    expect(el('setup-bpms-revert').disabled).toBe(true);
    settingsFor(ctx).model.set('pilot_tone', 'On');
    expect(el('setup-bpms-revert').disabled).toBe(false);
  });

  it('Setup CMs posts the corrector names to the solve route', async () => {
    const { el, api, square, place } = harness();
    el('setup-tab-cms').click();
    place('cms', 'CELL-A').click();
    square('cms', 'H2').click();
    expect(el('setup-cms').getAttribute('data-reason')).toBe('Keep at least one HCM in use');
    square('cms', 'H2').click();
    el('setup-cms').click();
    await flush();
    expect(api.solve).toHaveBeenCalledWith({ excluded_cms: ['H1', 'V1'] });
    expect(api.setup).not.toHaveBeenCalled();
  });

  it('shows Start from only when the catalog offers several sets, and stages its complement', () => {
    const { el, state } = harness();
    expect(el('cm-start-from')).toBeNull();
    const select = el('bpm-start-from');
    select.value = 'arc-picks';
    select.dispatchEvent(new Event('change'));
    expect(state('bpms', 'B2')).toEqual(['in', 'true']);
    for (const device of ['B4', 'B5', 'B6']) expect(state('bpms', device)).toEqual(['out', 'true']);
  });

  it('hides Start from with a single set', () => {
    const { el } = harness({ snapshot: snapshotWith([], [], { ...CATALOG, groups: [CATALOG.groups[0]] }) });
    expect(el('bpm-start-from')).toBeNull();
  });

  it('subscribes clicks on the trajectory, difference and CM-step plots only', () => {
    const { ctx, clicks } = harness();
    const ids = ctx.adapter.onPointClick.mock.calls.map((/** @type {any[]} */ call) => call[0]);
    expect(ids.sort()).toEqual([...CLICK_PLOTS.bpms, ...CLICK_PLOTS.cms].sort());
    expect([...clicks.keys()].some((id) => id.startsWith('plot-raw'))).toBe(false);
  });

  it('a plot click marks a pending change only', async () => {
    const { clicks, state, api } = harness();
    clicks.get('plot-traj-x')?.('B4');
    expect(state('bpms', 'B4')).toEqual(['out', 'true']);
    clicks.get('plot-diff-y')?.('B2');
    expect(state('bpms', 'B2')).toEqual(['in', 'true']);
    await flush();
    expect(api.setup).not.toHaveBeenCalled();
  });

  it('a corrector-bar click toggles the CM map and shows its tab', () => {
    const { clicks, state, host } = harness();
    clicks.get('plot-cm-step')?.('V2');
    expect(state('cms', 'V2')).toEqual(['out', 'true']);
    expect(/** @type {HTMLElement} */ (host.querySelector('[data-tab="cms"]')).hidden).toBe(false);
  });

  it('renders the shared settings model, so a change elsewhere shows here', async () => {
    const { el, ctx } = harness();
    await flush();
    expect(el('setting-turns-tracked').value).toBe('260');
    settingsFor(ctx).model.set('turns_tracked', 512);
    expect(el('setting-turns-tracked').value).toBe('512');
  });

  it('disables every edit without a session', async () => {
    const { el, square, place } = harness({ holding: false });
    await flush();
    for (const name of ['setup-bpms', 'setup-cms', 'bpm-exclude', 'cm-exclude', 'setting-pilot-tone', 'bpm-start-from']) {
      expect(el(name).disabled).toBe(true);
    }
    expect(square('bpms', 'B1').disabled).toBe(true);
    expect(place('bpms', 'CELL-A').disabled).toBe(true);
    expect(el('setup-bpms').getAttribute('data-reason')).toBe(NO_SESSION);
  });

  it('Open BPM control asks the drawer to open', () => {
    const { el, host } = harness();
    const opened = vi.fn();
    host.addEventListener(OPEN_BPM_CONTROL, opened);
    el('open-bpm-control').click();
    expect(opened).toHaveBeenCalledTimes(1);
  });
});
