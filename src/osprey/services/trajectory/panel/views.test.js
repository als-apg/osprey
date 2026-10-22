// @ts-check
import { describe, expect, it, vi } from 'vitest';

import { PENDING_MESSAGE } from './api.js';

const contributed = vi.hoisted(() => vi.fn());
vi.mock('/design-system/js/header-contrib.js', async (importOriginal) => ({
  .../** @type {object} */ (await importOriginal()),
  contributeHeader: contributed,
}));

import { AREA_MODULES, HEARTBEAT_MS, startPanel } from './panel.js';
import { sessionControls } from './session.js';
import {
  AREAS,
  CENTRE_TABS_ITEM,
  END_REASONS,
  PLOT_HOSTS,
  VIEW_DEFAULTS,
  control,
  createView,
  describeEnd,
  headerItems,
  headerView,
  noticeFromEvent,
  noticeFromResult,
  noticeView,
  sessionButtons,
  shellView,
  shouldHeartbeat,
  showTab,
  statusView,
} from './views.js';

const MOVED = [
  { name: 'HCM1', before: 1.5, outcome: 'verified' },
  { name: 'HCM2', before: -2, outcome: 'written' },
];

/**
 * A full snapshot; *over* replaces fields.
 * @param {Record<string, unknown>} [over]
 * @returns {any}
 */
function snapshot(over = {}) {
  return {
    target: 'va',
    caller: 'alice',
    writes_enabled: true,
    posture: 'readwrite',
    limits: 'optional',
    writing_available: true,
    model: 'SR',
    model_kind: 'single_pass',
    holder: null,
    taken_over: null,
    mode: 'idle',
    action: null,
    queue_wait_s: 5,
    action_timeout_s: 60,
    action_budget_s: 65,
    stop_budget_s: 10,
    last_shot: null,
    step: null,
    reference: null,
    last_action: null,
    continuous: null,
    chain: [],
    catalog: null,
    machine: { rf_on: 'on', stored_current: 12.5 },
    setup: null,
    solve: null,
    retention: 20,
    n_shots_max: 5,
    notices: [],
    ...over,
  };
}

const NOOP = () => {};

const ALICE = { owner: 'alice', kind: 'panel' };

/** @param {any} s @param {boolean} holding @param {boolean} [pending] */
function buttonsFor(s, holding, pending = false) {
  return Object.fromEntries(sessionButtons(s, sessionControls(s, { holding, pending }))
    .map((b) => [b.name, b.reason]));
}

describe('the shell', () => {
  it('has one host per area, each named by data-area', () => {
    const { root, areas } = shellView(NOOP);
    for (const name of AREAS) {
      expect(root.querySelectorAll(`[data-area="${name}"]`)).toHaveLength(1);
      expect(areas[name]?.getAttribute('data-area')).toBe(name);
    }
  });

  it('names every plot host by id and data-control, the correction results under dy', () => {
    const { root } = shellView(NOOP);
    for (const id of [...PLOT_HOSTS.raw, ...PLOT_HOSTS.correction, 'plot-sv']) {
      const host = root.querySelector(`#${id}`);
      expect(host?.getAttribute('data-control')).toBe(id);
    }
    const centre = /** @type {HTMLElement} */ (root.querySelector('[data-column="centre"]'));
    const order = [...centre.querySelectorAll('[id], [data-area="correction-results"]')]
      .map((n) => n.id || n.getAttribute('data-area'));
    expect(order.slice(order.indexOf('plot-diff-y'))).toEqual(['plot-diff-y', 'correction-results']);
  });

  it('puts the setup tabs and the chain left, the plots centre, the per-tab controls and SV right', () => {
    const { root } = shellView(NOOP);
    /** @param {string} column */
    const inColumn = (column) =>
      [...root.querySelectorAll(`[data-column="${column}"] [data-area], [data-column="${column}"] [id]`)]
        .map((n) => n.getAttribute('data-area') ?? n.id);
    expect(inColumn('left')).toEqual(['setup', 'chain']);
    expect(inColumn('centre')).toEqual(expect.arrayContaining(['plot-raw-sum', 'plot-traj-x', 'correction-results']));
    expect(inColumn('right')).toEqual(['plots', 'records', 'references', 'correction', 'plot-sv']);
  });

  it('shows one centre tab at a time and announces what became visible', () => {
    /** @type {string[]} */
    const tabs = [];
    const shell = shellView((tab) => tabs.push(tab));
    /** @type {string[]} */
    const shownPanels = [];
    shell.root.addEventListener('tab:show', (e) =>
      shownPanels.push(/** @type {HTMLElement} */ (e.target).getAttribute('data-tab-panel') ?? ''));
    showTab(shell, 'correction');
    expect(shell.tabPanels.raw.every((p) => p.hidden)).toBe(true);
    expect(shell.tabPanels.correction.every((p) => !p.hidden)).toBe(true);
    expect(shownPanels).toEqual(['correction', 'correction']);
    const selected = shell.tabStrip.querySelector('[aria-selected="true"]');
    expect(selected?.getAttribute('data-control')).toBe('tab-correction');
    /** @type {HTMLButtonElement} */ (shell.tabStrip.querySelector('[data-control="tab-raw"]')).click();
    expect(tabs).toEqual(['raw']);
  });
});

describe('the view store', () => {
  it('starts from the defaults and tells a key\'s listeners of a change only', () => {
    const view = createView();
    expect(view.get('plotRaw')).toBe(VIEW_DEFAULTS.plotRaw);
    /** @type {boolean[]} */
    const seen = [];
    const off = view.on('plotRaw', (v) => seen.push(v));
    view.on('plotCmStep', () => seen.push(true));
    view.set('plotRaw', false);
    view.set('plotRaw', false);
    off();
    view.set('plotRaw', true);
    expect(seen).toEqual([false]);
    expect(view.get('plotRaw')).toBe(true);
  });

  it('a redraw request bumps the counter every time', () => {
    const view = createView();
    /** @type {number[]} */
    const seen = [];
    view.on('redraw', (n) => seen.push(n));
    view.requestRedraw();
    view.requestRedraw();
    expect(seen).toEqual([1, 2]);
  });
});

describe('disabled controls', () => {
  it('a control with a reason is disabled and shows the reason', () => {
    const off = control('x', 'X', NOOP, 'Setup BPMs first');
    expect(off.disabled).toBe(true);
    expect(off.title).toBe('Setup BPMs first');
    expect(off.getAttribute('data-reason')).toBe('Setup BPMs first');
    const on = control('x', 'X', NOOP);
    expect(on.disabled).toBe(false);
    expect(on.hasAttribute('data-reason')).toBe(false);
  });

  it('the session controls are always all three, each disabled with its reason', () => {
    expect(buttonsFor(snapshot(), false)).toEqual({
      'session-open': null,
      'session-takeover': 'Nobody holds the session; open it.',
      'session-end': 'This tab holds no session.',
    });
    const held = snapshot({ holder: { owner: 'bob', kind: 'agent' } });
    expect(buttonsFor(held, false)).toMatchObject({
      'session-open': 'Held by agent of bob; take over instead.', 'session-takeover': null });
    const running = snapshot({ holder: { owner: 'bob', kind: 'agent' }, mode: 'action',
      action: { name: 'shot', stopping: false } });
    expect(buttonsFor(running, false)['session-takeover']).toBe('Wait for shot to end, or Stop it.');
    expect(buttonsFor(snapshot({ holder: ALICE }), true)).toEqual({
      'session-open': 'This tab holds the session.',
      'session-takeover': 'This tab holds the session.',
      'session-end': null,
    });
    expect(buttonsFor(snapshot(), false, true)['session-open']).toBe('A session request is in flight.');
    expect(buttonsFor(null, false)['session-open']).toBe('Waiting for the service.');
  });

  it('Stop says why when nothing runs and is offered while something does', () => {
    const idle = statusView(snapshot(), 'connected', sessionControls(snapshot(), { holding: false }), NOOP);
    expect(idle.querySelector('[data-control="stop"]')?.getAttribute('data-reason')).toBe('Nothing is running.');
    const s = snapshot({ holder: { owner: 'bob', kind: 'agent' }, mode: 'action',
      action: { name: 'apply', stopping: true } });
    const busy = statusView(s, 'connected', sessionControls(s, { holding: false }), NOOP);
    expect(busy.querySelector('[data-running]')?.textContent).toBe('Running: apply (agent of bob) — stopping');
    expect(/** @type {HTMLButtonElement} */ (busy.querySelector('[data-control="stop"]')).disabled).toBe(false);
  });
});

describe('the hub tile bar', () => {
  it('hands over the session controls, then the centre tabs last', () => {
    const items = headerItems(sessionButtons(snapshot(), sessionControls(snapshot(), { holding: false })),
      'correction');
    expect(items.map((i) => `${i.kind}:${i.id}`)).toEqual([
      'button:session-open', 'button:session-takeover', 'button:session-end', `nav:${CENTRE_TABS_ITEM}`,
    ]);
    expect(items[1]).toMatchObject({ disabled: true, title: 'Nobody holds the session; open it.' });
    expect(items[0]).toMatchObject({ disabled: false });
    expect(/** @type {any} */ (items[3]).items).toEqual([
      { id: 'tab-raw', label: 'Raw BPM Signal', active: false },
      { id: 'tab-correction', label: 'Trajectory Correction', active: true },
    ]);
  });
});

describe('the header', () => {
  it('shows target, holder, writes, RF and the stored current', () => {
    const s = snapshot({ holder: ALICE, writes_enabled: false,
      taken_over: { previous: { owner: 'bob', kind: 'agent' }, by: ALICE } });
    const node = headerView(s, true);
    expect(node.querySelector('[data-control="stored-current"]')?.textContent).toBe('12.50 mA');
    for (const part of ['va', 'panel of alice (this tab)', 'readwrite (writes disabled)', 'on',
      'Taken over by panel of alice from agent of bob']) {
      expect(node.textContent).toContain(part);
    }
  });

  it('shows a dash for an unread stored current', () => {
    const node = headerView(snapshot({ machine: null }), false);
    expect(node.querySelector('[data-control="stored-current"]')?.textContent).toBe('—');
  });

  it('writes service strings as text, never as markup', () => {
    const node = headerView(snapshot({ target: '<img src=x onerror=alert(1)>' }), false);
    expect(node.querySelector('img')).toBeNull();
    expect(node.textContent).toContain('<img');
  });
});

describe('shouldHeartbeat', () => {
  it('beats only while a run is in progress and this tab holds', () => {
    const running = snapshot({ continuous: { shots: 1, lease_expires_in_s: 12 } });
    expect(shouldHeartbeat(running, true)).toBe(true);
    expect(shouldHeartbeat(running, false)).toBe(false);
    expect(shouldHeartbeat(snapshot(), true)).toBe(false);
    expect(shouldHeartbeat(null, true)).toBe(false);
  });
});

describe('notices', () => {
  it('shows a refusal with its code, its extras and a 502 moved list', () => {
    const notice = noticeFromResult('apply', {
      kind: 'refused', status: 502, code: 'apply_failed', detail: 'readback missed',
      moved: MOVED, extras: { commands_fired: ['inject'] },
    });
    expect(notice).toMatchObject({ kind: 'refused', code: 'apply_failed', moved: MOVED });
    expect(notice?.detail).toContain('commands_fired: inject');
    const node = noticeView(/** @type {any} */ (notice));
    expect(node.querySelector('[data-code]')?.textContent).toBe('apply_failed');
    const cells = [...node.querySelectorAll('[data-view="moved"] td')].map((td) => td.textContent);
    expect(cells).toEqual(['HCM1', '1.5', 'verified', 'HCM2', '-2', 'written']);
  });

  it('reports an unknown outcome as pending, never as a refusal', () => {
    const notice = noticeFromResult('shot', { kind: 'pending', status: null, message: PENDING_MESSAGE });
    expect(notice).toMatchObject({ kind: 'pending', detail: PENDING_MESSAGE });
    expect(notice?.code).toBeUndefined();
  });

  it('needs no notice for a success', () => {
    expect(noticeFromResult('shot', { kind: 'ok', status: 200, body: {} })).toBeNull();
  });

  it('shows a stopped action with its stopper and before-values', () => {
    const notice = noticeFromEvent('stopped', { type: 'stopped', action: 'setup', stopper: 'bob', moved: MOVED });
    expect(notice).toMatchObject({ kind: 'stopped', title: 'setup stopped by bob', moved: MOVED });
  });

  it('tells a step apply from a put-back', () => {
    expect(noticeFromEvent('apply', { step_digest: 'd-1', shot_id: 's', moved: MOVED })?.title).toBe('step applied');
    expect(noticeFromEvent('apply', { step_digest: null, shot_id: null, moved: [] })?.title).toBe('put back applied');
  });

  it('explains every continuous_ended reason', () => {
    for (const end of Object.keys(END_REASONS)) {
      const notice = noticeFromEvent('continuous_ended', { end, shots: 3 });
      expect(notice?.kind).toBe('ended');
      expect(notice?.detail).toContain(/** @type {any} */ (END_REASONS)[end]);
    }
    expect(describeEnd({ end: 'lease_expired', shots: 2 }).detail).toMatch(/stopped heartbeating/);
    expect(describeEnd({ end: 'zero_step', shots: 4, step_reason: 'ON_REFERENCE', step_status: 'done' }).detail)
      .toContain('Last step: ON_REFERENCE — done.');
  });
});

/** A fake EventSource the test drives. */
class FakeSource {
  /** @type {FakeSource[]} */
  static instances = [];
  /** @param {string} url */
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    /** @type {((ev: Event) => void) | null} */
    this.onopen = null;
    /** @type {((ev: Event) => void) | null} */
    this.onerror = null;
    /** @type {Map<string, (ev: MessageEvent) => void>} */
    this.listeners = new Map();
    FakeSource.instances.push(this);
  }
  /** @param {string} type @param {(ev: MessageEvent) => void} fn */
  addEventListener(type, fn) {
    this.listeners.set(type, fn);
  }
  close() {}
  /** @param {string} type @param {unknown} payload */
  emit(type, payload) {
    this.listeners.get(type)?.(/** @type {MessageEvent} */ ({ data: JSON.stringify(payload) }));
  }
}

/** Let pending promise callbacks run. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** @param {HTMLElement} root @param {string} name */
function click(root, name) {
  const button = /** @type {HTMLButtonElement | null} */ (root.querySelector(`[data-control="${name}"]`));
  if (!button) throw new Error(`no ${name} control`);
  button.click();
}

/**
 * Mounts that record what they were handed, one per area module.
 * @returns {{ modules: Record<string, import('./panel.js').Mount>, mounted: Record<string, [HTMLElement, any]> }}
 */
function recordingModules() {
  /** @type {Record<string, [HTMLElement, any]>} */
  const mounted = {};
  /** @type {Record<string, import('./panel.js').Mount>} */
  const modules = {};
  for (const name of Object.keys(AREA_MODULES)) {
    modules[name] = (host, ctx) => {
      mounted[name] = [host, ctx];
    };
  }
  return { modules, mounted };
}

/**
 * Start a panel against a fake service answering by route.
 * @param {Record<string, (init: RequestInit) => [number, unknown]>} routes
 */
function harness(routes = {}) {
  FakeSource.instances = [];
  /** @type {{ url: string, init: RequestInit }[]} */
  const calls = [];
  /** @type {(() => void)[]} */
  const timers = [];
  /** @param {string} url @param {RequestInit} init */
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    const key = `${init.method} ${url}`;
    const [status, body] = routes[key] ? routes[key](init) : [200, {}];
    return /** @type {Response} */ ({ ok: status < 300, status, json: async () => body });
  };
  const root = document.createElement('div');
  const { modules, mounted } = recordingModules();
  const panel = startPanel({
    root,
    EventSourceCtor: /** @type {any} */ (FakeSource),
    fetchImpl,
    setIntervalImpl: (fn, ms) => {
      expect(ms).toBe(HEARTBEAT_MS);
      timers.push(fn);
      return timers.length;
    },
    clearIntervalImpl: () => timers.splice(0),
    adapter: { stub: true },
    modules,
  });
  const source = FakeSource.instances[0];
  /** @param {Record<string, unknown>} [over] */
  const emitSnapshot = (over) => source.emit('snapshot', { type: 'snapshot', snapshot: snapshot(over) });
  return { panel, root, calls, timers, source, mounted, emitSnapshot };
}

/** Open the session as alice and see it held. */
async function opened(/** @type {ReturnType<typeof harness>} */ h) {
  h.emitSnapshot();
  click(h.root, 'session-open');
  await flush();
  h.emitSnapshot({ holder: ALICE });
}

const GRANT = { 'POST api/session': () => /** @type {[number, unknown]} */ ([200, { client_id: 'c-1', owner: 'alice', target: 'va' }]) };


describe('startPanel', () => {
  it('mounts every area module once, into its own host, with the whole context', () => {
    const { panel, mounted } = harness();
    expect(Object.keys(mounted).sort()).toEqual(Object.keys(AREA_MODULES).sort());
    for (const [name, [host, ctx]] of Object.entries(mounted)) {
      expect(host.getAttribute('data-area')).toBe(name);
      for (const key of ['api', 'snapshot', 'onSnapshot', 'onEvent', 'adapter', 'view', 'notify',
        'startContinuous', 'stopContinuous']) {
        expect(ctx).toHaveProperty(key);
      }
      expect(ctx.adapter).toEqual({ stub: true });
      expect(ctx.area('correction-results')?.getAttribute('data-area')).toBe('correction-results');
    }
    expect(mounted.chain[1]).toBe(mounted.plots[1]);
    panel.close();
  });

  it('hands the areas each snapshot and each event of the type they asked for', () => {
    const { panel, mounted, source, emitSnapshot } = harness();
    const ctx = mounted.plots[1];
    /** @type {any[]} */
    const snaps = [];
    /** @type {any[]} */
    const shots = [];
    const off = ctx.onSnapshot((/** @type {any} */ s) => snaps.push(s.mode));
    ctx.onEvent('shot', (/** @type {any} */ p) => shots.push(p.shot.shot_id));
    emitSnapshot();
    source.emit('shot', { type: 'shot', shot: { shot_id: 's-1' } });
    source.emit('step', { type: 'step', step: null });
    off();
    emitSnapshot({ mode: 'action' });
    expect(snaps).toEqual(['idle']);
    expect(shots).toEqual(['s-1']);
    expect(ctx.snapshot().mode).toBe('action');
    panel.close();
  });

  it('opens and ends the session from the header strip, with the client header', async () => {
    const h = harness(GRANT);
    await opened(h);
    expect(h.root.textContent).toContain('panel of alice (this tab)');
    expect(h.mounted.chain[1].holding()).toBe(true);
    click(h.root, 'session-end');
    await flush();
    const release = h.calls.find((c) => c.init.method === 'DELETE');
    expect(/** @type {Record<string, string>} */ (release?.init.headers ?? {})['X-Trajectory-Client']).toBe('c-1');
    expect(h.calls.every((c) => !c.url.startsWith('/'))).toBe(true);
    h.panel.close();
  });

  it('drops the session on a lost-session refusal of an area call', async () => {
    const h = harness({ ...GRANT,
      'POST api/shot': () => [409, { detail: { code: 'session_taken_over', detail: 'taken over' } }] });
    await opened(h);
    const ctx = h.mounted.chain[1];
    expect(ctx.holding()).toBe(true);
    expect((await ctx.api.shot({ steps: ['readout'] })).kind).toBe('refused');
    expect(ctx.holding()).toBe(false);
    const takeOver = /** @type {HTMLButtonElement} */ (h.root.querySelector('[data-control="session-takeover"]'));
    expect(takeOver.disabled).toBe(false);
    h.panel.close();
  });

  it('a lost-session refusal of a context call drops the session', async () => {
    const h = harness({ ...GRANT,
      'POST api/stop': () => [409, { detail: { code: 'not_session_holder', detail: 'not yours' } }] });
    await opened(h);
    await h.mounted.chain[1].stopContinuous();
    expect(h.mounted.chain[1].holding()).toBe(false);
    expect(h.root.querySelector('[data-code]')?.textContent).toBe('not_session_holder');
    h.panel.close();
  });

  it('heartbeats while its continuous run lasts and shows lease_expired prominently', async () => {
    const h = harness({ ...GRANT,
      'POST api/continuous': () => [200, { shots: 0, lease_expires_in_s: 15 }] });
    await opened(h);
    const result = await h.mounted.chain[1].startContinuous({ steps: ['arm', 'inject', 'readout'] });
    expect(result.kind).toBe('ok');
    expect(h.panel.heartbeating()).toBe(true);
    const start = h.calls.find((c) => c.url === 'api/continuous');
    expect(JSON.parse(String(start?.init.body))).toEqual({ steps: ['arm', 'inject', 'readout'] });
    h.emitSnapshot({ holder: ALICE, mode: 'continuous', continuous: { shots: 1, lease_expires_in_s: 14 } });
    h.timers[0]();
    await flush();
    expect(h.calls.some((c) => c.url === 'api/continuous/heartbeat')).toBe(true);
    h.source.emit('continuous_ended', { type: 'continuous_ended', end: 'lease_expired', shots: 1 });
    h.emitSnapshot({ holder: ALICE });
    expect(h.panel.heartbeating()).toBe(false);
    const banner = h.root.querySelector('[data-view="ended"]');
    expect(banner?.textContent).toContain('lease_expired');
    expect(banner?.querySelector('[role="alert"]')).not.toBeNull();
    h.panel.close();
  });

  it('a refused continuous start sends no heartbeat', async () => {
    const h = harness({ ...GRANT,
      'POST api/continuous': () => [409, { detail: { code: 'setup_required', detail: 'Setup BPMs first' } }] });
    await opened(h);
    await h.mounted.chain[1].startContinuous({ steps: ['readout'] });
    expect(h.panel.heartbeating()).toBe(false);
    expect(h.root.querySelector('[data-code]')?.textContent).toBe('setup_required');
    h.panel.close();
  });

  it('shows the notices an area raises and the ones the service serves', () => {
    const { panel, root, mounted, emitSnapshot } = harness();
    emitSnapshot({ notices: ['setup voided: BPM mode changed'] });
    mounted.records[1].notify({ kind: 'info', title: 'record s-3 evicted' });
    const text = root.querySelector('[data-area="notices"]')?.textContent;
    expect(text).toContain('setup voided: BPM mode changed');
    expect(text).toContain('record s-3 evicted');
    panel.close();
  });

  it('Stop posts stop from the status strip', async () => {
    const h = harness();
    h.emitSnapshot({ holder: { owner: 'bob', kind: 'agent' }, mode: 'action',
      action: { name: 'shot', stopping: false } });
    click(h.root, 'stop');
    await flush();
    expect(h.calls.map((c) => c.url)).toContain('api/stop');
    h.panel.close();
  });

  it('switches the centre tab from the strip and from the tile bar', () => {
    const { panel, root } = harness();
    expect(panel.tab()).toBe('raw');
    click(root, 'tab-correction');
    expect(panel.tab()).toBe('correction');
    expect(/** @type {HTMLElement} */ (root.querySelector('[data-area="raw-plots"]')).hidden).toBe(true);
    panel.headerAction(CENTRE_TABS_ITEM, 'tab-raw');
    expect(panel.tab()).toBe('raw');
    panel.close();
  });

  it('a tile-bar session button calls the same function as the in-body one', async () => {
    const h = harness(GRANT);
    h.emitSnapshot();
    h.panel.headerAction('session-open');
    await flush();
    expect(h.calls.map((c) => `${c.init.method} ${c.url}`)).toContain('POST api/session');
    h.panel.close();
  });

  it('hands the tile bar the session controls and the centre tabs on every change', async () => {
    contributed.mockClear();
    const h = harness(GRANT);
    h.emitSnapshot();
    h.panel.showTab('correction');
    let items = contributed.mock.calls.at(-1)?.[0];
    expect(items.map((/** @type {any} */ i) => `${i.kind}:${i.id}`)).toEqual([
      'button:session-open', 'button:session-takeover', 'button:session-end', 'nav:centre-tabs']);
    expect(items[3].items.find((/** @type {any} */ e) => e.active).id).toBe('tab-correction');
    await opened(h);
    items = contributed.mock.calls.at(-1)?.[0];
    expect(items.find((/** @type {any} */ i) => i.id === 'session-end').disabled).toBe(false);
    h.panel.close();
  });
});

/**
 * Every rule of `sheet` (media blocks included) whose selector matches `el`,
 * flattened to the property names it sets.
 * @param {CSSStyleSheet} sheet
 * @param {Element} el
 */
function styledProps(sheet, el) {
  /** @type {Set<string>} */
  const props = new Set();
  /** @param {CSSRuleList} rules */
  const walk = (rules) => {
    for (const rule of Array.from(rules)) {
      const nested = /** @type {any} */ (rule).cssRules;
      if (nested && !(/** @type {any} */ (rule).selectorText)) { walk(nested); continue; }
      const selector = /** @type {any} */ (rule).selectorText;
      if (!selector) continue;
      let hit = false;
      try { hit = el.matches(selector); } catch { hit = false; }
      if (!hit) continue;
      const style = /** @type {CSSStyleRule} */ (rule).style;
      for (let i = 0; i < style.length; i += 1) props.add(style[i]);
    }
  };
  walk(sheet.cssRules);
  return [...props];
}

describe('panel.css styles every control the panel renders', () => {
  it('no button, field, checkbox or disclosure the real areas mount is left to browser defaults', async () => {
    // The stylesheet exactly as index.html links it (vitest runs from the repo root).
    const { readFileSync } = await import('node:fs');
    const { cwd } = await import('node:process');
    const css = readFileSync(`${cwd()}/src/osprey/services/trajectory/panel/panel.css`, 'utf8');
    const style = document.createElement('style');
    style.textContent = css;
    document.head.append(style);
    const sheet = /** @type {CSSStyleSheet} */ (style.sheet);

    // The real panel: every area module mounted, the drawer included, so a
    // control a later area adds is checked without this test naming it.
    FakeSource.instances = [];
    const root = document.createElement('div');
    document.body.append(root);
    const panel = startPanel({
      root,
      EventSourceCtor: /** @type {any} */ (FakeSource),
      fetchImpl: async () => /** @type {Response} */ ({ ok: true, status: 200, json: async () => ({}) }),
      setIntervalImpl: () => 1,
      clearIntervalImpl: () => {},
      // Plotly is not loaded here; every adapter call is a no-op.
      adapter: /** @type {any} */ (new Proxy({}, { get: () => () => true })),
    });
    FakeSource.instances[0].emit('snapshot', { type: 'snapshot', snapshot: snapshot() });
    await flush();

    /** @type {string[]} */
    const bare = [];
    /** @param {Element} el @param {string[]} needs */
    const check = (el, needs) => {
      const props = styledProps(sheet, el);
      const missing = needs.filter((n) => !props.some((p) => p.startsWith(n)));
      if (missing.length) {
        const name = el.getAttribute('data-control') ?? el.getAttribute('aria-label') ?? el.textContent?.trim();
        bare.push(`${el.tagName.toLowerCase()}${el.getAttribute('type') ? `[${el.getAttribute('type')}]` : ''} "${name}" lacks ${missing.join(', ')}`);
      }
    };
    const controls = Array.from(root.querySelectorAll('button, input, select, summary'))
      .filter((el) => !(el instanceof HTMLInputElement && el.type === 'file'));
    expect(controls.length).toBeGreaterThan(10);
    for (const el of controls) {
      const type = el.getAttribute('type');
      if (el.tagName === 'INPUT' && (type === 'checkbox' || type === 'radio')) check(el, ['accent-color']);
      else if (el.tagName === 'SUMMARY') check(el, ['cursor', 'font']);
      else if (el.tagName === 'BUTTON') check(el, ['font', 'border', 'cursor']);
      else check(el, ['font', 'border', 'background']);
    }
    expect(bare).toEqual([]);

    panel.close();
    root.remove();
    style.remove();
  });
});
