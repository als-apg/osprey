// @ts-check
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  DRAWER_ID,
  NOTHING_PENDING,
  drawerBody,
  mount as mountDrawer,
  readbackById,
  storedCurrentText,
} from './drawer.js';
import { NO_SESSION, mount as mountSetup } from './setup.js';

const FAMILIES = { bpms: ['B1', 'B2', 'B3'], hcms: ['H1'], vcms: ['V1'] };

const READBACKS = {
  read_at: 1,
  setup: null,
  settings: [
    { id: 'pilot_tone', scope: 'device', value: null, values: { B1: 'Off', B2: 'Off', B3: 'Off' } },
    { id: 'bpm_mode', scope: 'device', value: null, values: { B1: 'Low Current', B2: 'Low Current', B3: 'Low Current' } },
    { id: 'attenuation', scope: 'device', value: null, values: { B1: 10, B2: 10, B3: 12 } },
    { id: 'turns_tracked', scope: 'global', value: 40, values: {} },
  ],
};

/** @param {number} [current] */
function snapshotWith(current = 12.5) {
  return {
    families: FAMILIES,
    machine: { rf_on: 1, stored_current: current },
    setup: { settings_sha: 'a' },
    solve: { config: { excluded_bpms: [], excluded_cms: [], alpha: 0, gain: 1, keep_live: 1, n_discard: 0, loss_cutoff: 0, overridden: [] } },
  };
}

/** @param {any} body */
const ok = (body) => ({ kind: 'ok', status: 200, body });
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** The page's drawer markup, as index.html carries it. */
const PAGE = `
  <div id="panel-root"><div data-area="setup"></div><div data-area="drawer"></div></div>
  <div class="drawer-backdrop" id="drawer-backdrop"></div>
  <osprey-drawer class="drawer" id="${DRAWER_ID}">
    <div class="drawer-header"><span class="drawer-title">BPM Control</span>
      <button type="button" class="drawer-close-btn">x</button></div>
    <div class="drawer-body"></div>
  </osprey-drawer>`;

/** @param {{holding?: boolean, withSetup?: boolean}} [options] */
async function harness({ holding = true, withSetup = true } = {}) {
  document.body.innerHTML = PAGE;
  /** @type {Set<(s: any) => void>} */
  const snapshotFns = new Set();
  let current = snapshotWith();
  const api = {
    setupReadbacks: vi.fn(async () => ok(READBACKS)),
    setup: vi.fn(async () => ok({ moved: [], unsettled: [], setup: null })),
    solve: vi.fn(async () => ok({})),
  };
  const adapter = { render: vi.fn(() => true), onPointClick: vi.fn() };
  const ctx = /** @type {any} */ ({
    api,
    adapter,
    notify: vi.fn(),
    holding: () => holding,
    snapshot: () => current,
    onSnapshot: (/** @type {(s: any) => void} */ fn) => {
      snapshotFns.add(fn);
      return () => snapshotFns.delete(fn);
    },
  });
  const setupHost = /** @type {HTMLElement} */ (document.querySelector('[data-area=setup]'));
  const drawerHost = /** @type {HTMLElement} */ (document.querySelector('[data-area=drawer]'));
  if (withSetup) mountSetup(setupHost, ctx);
  mountDrawer(drawerHost, ctx);
  await flush();
  const drawer = /** @type {HTMLElement} */ (document.getElementById(DRAWER_ID));
  /** @param {string} name */
  const inDrawer = (name) => /** @type {any} */ (drawer.querySelector(`[data-control=${name}]`));
  /** @param {string} name */
  const inTab = (name) => /** @type {any} */ (setupHost.querySelector(`[data-control=${name}]`));
  /** @param {any} next */
  const emit = (next) => {
    current = next;
    for (const fn of [...snapshotFns]) fn(next);
  };
  return { api, adapter, ctx, drawer, drawerHost, setupHost, inDrawer, inTab, emit };
}

/**
 * Edit an input and commit the edit, as the operator does.
 * @param {any} input
 * @param {string} value
 */
function change(input, value) {
  input.value = value;
  input.dispatchEvent(new Event('input'));
  input.dispatchEvent(new Event('change'));
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('pure helpers', () => {
  it('formats the stored current like the header, and a dash when unread', () => {
    expect(storedCurrentText(snapshotWith(12.5))).toBe('12.50 mA');
    expect(storedCurrentText({ machine: { stored_current: null } })).toBe('—');
    expect(storedCurrentText(null)).toBe('—');
  });

  it('builds a body of settings only', () => {
    expect(drawerBody({ pilot_tone: 'On' })).toEqual({ settings: { pilot_tone: 'On' } });
  });

  it('finds a readback by id', () => {
    expect(readbackById(READBACKS, 'attenuation').values.B3).toBe(12);
    expect(readbackById(READBACKS, 'missing')).toBeNull();
    expect(readbackById(null, 'attenuation')).toBeNull();
  });
});

describe('drawer', () => {
  it('fills the page drawer, not the area host', async () => {
    const { drawer, drawerHost, inDrawer } = await harness();
    expect(drawerHost.children.length).toBe(0);
    expect(drawer.querySelector('.drawer-body [data-control=setting-pilot-tone]')).not.toBeNull();
    for (const name of ['setting-pilot-tone', 'setting-bpm-mode', 'setting-attenuation', 'setting-turns-tracked',
      'stored-current-readback', 'plot-bpm-atten', 'plot-bpm-pt-state']) {
      expect(inDrawer(name), name).not.toBeNull();
    }
  });

  it('opens on Open BPM control', async () => {
    const { drawer, inTab } = await harness();
    expect(drawer.hasAttribute('open')).toBe(false);
    inTab('open-bpm-control').click();
    expect(drawer.hasAttribute('open')).toBe(true);
    expect(document.getElementById('drawer-backdrop')?.hasAttribute('open')).toBe(true);
  });

  it('shows the readback values the setup route serves', async () => {
    const { inDrawer } = await harness();
    expect(inDrawer('setting-pilot-tone').value).toBe('Off');
    expect(inDrawer('setting-bpm-mode').value).toBe('Low Current');
    expect(inDrawer('setting-turns-tracked').value).toBe('40');
    // Attenuation disagrees across BPMs: unset, shown as mixed.
    expect(inDrawer('setting-attenuation').value).toBe('');
  });

  it('shows the stored current from the snapshot and follows it', async () => {
    const { inDrawer, emit } = await harness();
    expect(inDrawer('stored-current-readback').textContent).toBe('12.50 mA');
    emit(snapshotWith(3.25));
    expect(inDrawer('stored-current-readback').textContent).toBe('3.250 mA');
  });

  it('plots attenuation and pilot tone from the readbacks through the adapter', async () => {
    const { adapter } = await harness();
    const calls = /** @type {any[]} */ (adapter.render.mock.calls);
    const ids = calls.map(([id]) => id);
    expect(ids).toContain('plot-bpm-atten');
    expect(ids).toContain('plot-bpm-pt-state');
    const atten = calls.filter(([id]) => id === 'plot-bpm-atten').at(-1)[1]({});
    expect(atten.layout.uirevision).toBe('drawer-attenuation');
    expect(atten.data[0].y).toEqual([10, 10, 12]);
    const pt = calls.filter(([id]) => id === 'plot-bpm-pt-state').at(-1)[1]({});
    expect(pt.layout.uirevision).toBe('drawer-pilot-tone');
    expect(pt.data[0].y).toEqual(['Off', 'Off', 'Off']);
  });

  it('does not redraw the plots on a snapshot that changes neither readbacks nor families', async () => {
    const { adapter, emit } = await harness();
    const before = adapter.render.mock.calls.length;
    emit({ ...snapshotWith(5), families: FAMILIES, setup: { settings_sha: 'a' } });
    expect(adapter.render.mock.calls.length).toBe(before);
  });

  it('a drawer commit carries no exclusion key, even with an exclusion pending in the tab', async () => {
    const { api, inDrawer, inTab } = await harness();
    change(inTab('bpm-exclude'), '2');
    change(inDrawer('setting-pilot-tone'), 'On');
    inDrawer('bpm-control-set').click();
    await flush();
    expect(api.setup).toHaveBeenCalledTimes(1);
    const body = /** @type {any} */ (api.setup.mock.calls[0])[0];
    expect(body).toEqual({ settings: { pilot_tone: 'On' } });
    expect(Object.keys(body)).not.toContain('excluded_bpms');
    // The tab's exclusion draft is untouched.
    expect(inTab('bpm-exclude').value).toBe('2');
  });

  it('renders a change made in the tab', async () => {
    const { inDrawer, inTab } = await harness();
    change(inTab('setting-bpm-mode'), 'High Current');
    expect(inDrawer('setting-bpm-mode').value).toBe('High Current');
    expect(inDrawer('bpm-control-set').disabled).toBe(false);
  });

  it('a change made in the drawer shows in the tab', async () => {
    const { inDrawer, inTab } = await harness();
    change(inDrawer('setting-pilot-tone'), 'On');
    expect(inTab('setting-pilot-tone').value).toBe('On');
    expect(inDrawer('setting-pilot-tone').value).toBe('On');
  });

  it('Revert drops the pending values in both views', async () => {
    const { inDrawer, inTab } = await harness();
    change(inDrawer('setting-pilot-tone'), 'On');
    inDrawer('bpm-control-revert').click();
    expect(inDrawer('setting-pilot-tone').value).toBe('Off');
    expect(inTab('setting-pilot-tone').value).toBe('Off');
  });

  it('Set is disabled with a reason when nothing is pending', async () => {
    const { inDrawer } = await harness();
    expect(inDrawer('bpm-control-set').disabled).toBe(true);
    expect(inDrawer('bpm-control-set').getAttribute('data-reason')).toBe(NOTHING_PENDING);
  });

  it('without a session every setting and Set is disabled with the session reason', async () => {
    const { inDrawer } = await harness({ holding: false });
    expect(inDrawer('setting-pilot-tone').disabled).toBe(true);
    expect(inDrawer('bpm-control-set').getAttribute('data-reason')).toBe(NO_SESSION);
  });

  it('fills the area host when the page has no drawer element', async () => {
    document.body.innerHTML = '<div data-area="drawer"></div>';
    const host = /** @type {HTMLElement} */ (document.querySelector('[data-area=drawer]'));
    const ctx = /** @type {any} */ ({
      api: { setupReadbacks: vi.fn(async () => ok(READBACKS)), setup: vi.fn() },
      adapter: { render: vi.fn() },
      notify: vi.fn(),
      holding: () => true,
      snapshot: () => snapshotWith(),
      onSnapshot: () => () => {},
    });
    mountDrawer(host, ctx);
    await flush();
    expect(host.querySelector('[data-control=stored-current-readback]')).not.toBeNull();
  });
});

describe('drawer opening', () => {
  it('re-reads the readbacks and redraws with the current session state', async () => {
    let holding = false;
    const h = await harness({ holding: false });
    h.ctx.holding = () => holding;
    expect(h.inDrawer('setting-pilot-tone').disabled).toBe(true);
    const reads = h.api.setupReadbacks.mock.calls.length;
    holding = true;
    h.inTab('open-bpm-control').click();
    await flush();
    expect(h.inDrawer('setting-pilot-tone').disabled).toBe(false);
    expect(h.api.setupReadbacks.mock.calls.length).toBe(reads + 1);
  });
});
