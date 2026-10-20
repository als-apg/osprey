// @ts-check
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { settingsFor } from './settings.js';
import {
  CLICK_PLOTS,
  NO_SESSION,
  OPEN_BPM_CONTROL,
  bpmsBody,
  clickFamily,
  cmsBody,
  familyCount,
  mount,
  seededTexts,
} from './setup.js';

const FAMILIES = {
  bpms: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'],
  hcms: ['H1', 'H2', 'H3'],
  vcms: ['V1', 'V2'],
};

/**
 * @param {string[]} [bpms]
 * @param {string[]} [cms]
 */
function snapshotWith(bpms = [], cms = []) {
  return {
    families: FAMILIES,
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
  return {
    host,
    ctx,
    api,
    notify,
    clicks,
    el,
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
  it('counts each family and its exclusions', () => {
    expect(familyCount(snapshotWith(['B2', 'B5']), 'bpms')).toBe('6 BPMs, 2 excluded');
    expect(familyCount(snapshotWith([], ['V2']), 'vcms')).toBe('2 VCMs, 1 excluded');
    expect(familyCount(null, 'hcms')).toBe('0 HCMs');
  });

  it('seeds the fields from the SolveConfig, by index', () => {
    expect(seededTexts(snapshotWith(['B4', 'B2', 'B3'], ['H3', 'V1']))).toEqual({ bpms: '2:4', hcms: '3', vcms: '1' });
  });

  it('builds the setup body with names and the pending settings only', () => {
    const texts = { bpms: '[1, 5:6]', hcms: '', vcms: '' };
    expect(bpmsBody(texts, snapshotWith(), { pilot_tone: 'On' })).toEqual({
      body: { settings: { pilot_tone: 'On' }, excluded_bpms: ['B1', 'B5', 'B6'] },
    });
    expect(bpmsBody({ ...texts, bpms: '7' }, snapshotWith(), {})).toEqual({
      error: 'BPM 7 is not in the BPM family (1-6)',
    });
  });

  it('builds the solve body from both corrector families', () => {
    expect(cmsBody({ bpms: '', hcms: '1 3', vcms: '2' }, snapshotWith())).toEqual({
      body: { excluded_cms: ['H1', 'H3', 'V2'] },
    });
    expect(cmsBody({ bpms: '', hcms: '', vcms: '3' }, snapshotWith())).toEqual({
      error: 'VCM 3 is not in the VCM family (1-2)',
    });
  });

  it('resolves a click to a family only on the plots that exclude', () => {
    const snap = snapshotWith();
    expect(clickFamily(snap, 'plot-traj-x', 'B3')).toBe('bpms');
    expect(clickFamily(snap, 'plot-diff-y', 'B3')).toBe('bpms');
    expect(clickFamily(snap, 'plot-cm-step', 'V1')).toBe('vcms');
    expect(clickFamily(snap, 'plot-cm-step', 'B3')).toBeNull();
    expect(clickFamily(snap, 'plot-raw-x', 'B3')).toBeNull();
  });
});

describe('the setup area', () => {
  it('shows the family counts and the exclusions in force', () => {
    const { el } = harness();
    expect(el('bpm-family').textContent).toBe('6 BPMs, 1 excluded');
    expect(el('cm-family').textContent).toBe('3 HCMs; 2 VCMs');
    expect(el('bpm-exclude').value).toBe('2');
    expect(el('cm-exclude')).not.toBeNull();
  });

  it('re-seeds the exclude list from every snapshot while it is not edited', () => {
    const { el, push } = harness();
    push(snapshotWith(['B1', 'B6'], ['H2']));
    expect(el('bpm-exclude').value).toBe('1 6');
    expect(el('cm-exclude-hcm').value).toBe('2');
  });

  it('holds a pending edit across snapshots until Revert clears it', () => {
    const { el, push } = harness();
    harnessType(el, 'bpm-exclude', '3:5');
    push(snapshotWith(['B1']));
    expect(el('bpm-exclude').value).toBe('3:5');
    expect(el('bpm-exclude').getAttribute('data-pending')).toBe('true');
    el('setup-bpms-revert').click();
    expect(el('bpm-exclude').value).toBe('1');
    push(snapshotWith(['B6']));
    expect(el('bpm-exclude').value).toBe('6');
  });

  it('counts a pending setting as an edit of the BPM tab', async () => {
    const { el, push, ctx } = harness();
    await flush();
    settingsFor(ctx).model.set('pilot_tone', 'On');
    push(snapshotWith(['B4']));
    expect(el('bpm-exclude').value).toBe('2');
    el('setup-bpms-revert').click();
    expect(el('bpm-exclude').value).toBe('4');
  });

  it('Setup BPMs posts the pending settings and the BPM names, then re-seeds', async () => {
    const { el, api, push } = harness();
    await flush();
    harnessType(el, 'bpm-exclude', '[1, 5:6]');
    harnessType(el, 'setting-pilot-tone', 'On');
    el('setup-bpms').click();
    await flush();
    expect(api.setup).toHaveBeenCalledWith({ settings: { pilot_tone: 'On' }, excluded_bpms: ['B1', 'B5', 'B6'] });
    expect(api.setupReadbacks).toHaveBeenCalledTimes(2);
    push(snapshotWith(['B1', 'B5', 'B6']));
    expect(el('bpm-exclude').value).toBe('1 5 6');
    expect(el('bpm-exclude').getAttribute('data-pending')).toBeNull();
    push(snapshotWith(['B3']));
    expect(el('bpm-exclude').value).toBe('3');
  });

  it('Setup CMs posts the corrector names to the solve route', async () => {
    const { el, api } = harness();
    el('setup-tab-cms').click();
    harnessType(el, 'cm-exclude-hcm', '1,3');
    harnessType(el, 'cm-exclude-vcm', '2');
    el('setup-cms').click();
    await flush();
    expect(api.solve).toHaveBeenCalledWith({ excluded_cms: ['H1', 'H3', 'V2'] });
    expect(api.setup).not.toHaveBeenCalled();
  });

  it('refuses an out-of-family index by name and disables the commit', () => {
    const { el, host } = harness();
    harnessType(el, 'bpm-exclude', '2 9');
    expect(host.querySelector('[data-error="bpm-exclude"]').textContent).toBe('BPM 9 is not in the BPM family (1-6)');
    expect(el('setup-bpms').disabled).toBe(true);
    expect(el('setup-bpms').getAttribute('data-reason')).toBe('BPM 9 is not in the BPM family (1-6)');
  });

  it('subscribes clicks on the trajectory, difference and CM-step plots only', () => {
    const { ctx, clicks } = harness();
    const ids = ctx.adapter.onPointClick.mock.calls.map((/** @type {any[]} */ call) => call[0]);
    expect(ids.sort()).toEqual([...CLICK_PLOTS.bpms, ...CLICK_PLOTS.cms].sort());
    expect([...clicks.keys()].some((id) => id.startsWith('plot-raw'))).toBe(false);
  });

  it('a click toggles a pending exclusion that Setup BPMs then commits', async () => {
    const { el, clicks, api, push } = harness();
    clicks.get('plot-traj-x')?.('B4');
    expect(el('bpm-exclude').value).toBe('2 4');
    push(snapshotWith(['B1']));
    expect(el('bpm-exclude').value).toBe('2 4');
    clicks.get('plot-diff-y')?.('B2');
    expect(el('bpm-exclude').value).toBe('4');
    el('setup-bpms').click();
    await flush();
    expect(api.setup).toHaveBeenCalledWith({ settings: {}, excluded_bpms: ['B4'] });
  });

  it('a corrector-bar click toggles the CM list and shows its tab', () => {
    const { el, clicks, host } = harness();
    clicks.get('plot-cm-step')?.('V2');
    expect(el('cm-exclude-vcm').value).toBe('2');
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
    const { el } = harness({ holding: false });
    await flush();
    for (const name of ['setup-bpms', 'setup-cms', 'bpm-exclude', 'setting-pilot-tone']) {
      expect(el(name).disabled).toBe(true);
    }
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

/**
 * Type into a field and commit the edit, as the operator does.
 * @param {(name: string) => any} el
 * @param {string} name
 * @param {string} value
 */
function harnessType(el, name, value) {
  const input = el(name);
  input.value = value;
  input.dispatchEvent(new Event('input'));
  input.dispatchEvent(new Event('change'));
}
