// @ts-check
import { describe, expect, it } from 'vitest';

import { PENDING_MESSAGE } from './api.js';
import { HEARTBEAT_MS, startPanel } from './panel.js';
import { sessionControls } from './session.js';
import {
  END_REASONS,
  actionView,
  describeEnd,
  headerView,
  loopArgs,
  loopControls,
  loopForm,
  noticeFromEvent,
  noticeFromResult,
  noticeView,
  referencesView,
  setLoopEnabled,
  shouldHeartbeat,
  statusView,
  stepView,
} from './views.js';

const STEP = {
  digest: 'd-1',
  deltas: { HCM2: -0.25, HCM1: 0.5 },
  factor: 0.5,
  reason: 'OK',
  status: 'solved with rank 2',
  all_zero: false,
  zeroed_h: ['HCM9'],
  zeroed_v: [],
  no_response_names: ['VCM4'],
  unsolved: ['VCM7', 'VCM8'],
  rank_discarded: 1,
  singular_values: [1, 0.1],
  inverse_weights: [1, 0],
  spectrum_kind: ['LIVE', 'RANK_DISCARDED'],
  n_discard: 1,
  shot_id: 's-1',
  reference: null,
};

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
    ...over,
  };
}

const NOOP = () => {};

describe('loopControls', () => {
  it('shows Stop to every admitted owner, holding or not, running or not', () => {
    expect(loopControls(snapshot(), { holding: false }).stop).toBe(true);
    expect(loopControls(snapshot({ mode: 'continuous' }), { holding: true }).stop).toBe(true);
  });

  it('shows no Stop without an admitted operator or a snapshot', () => {
    expect(loopControls(snapshot({ caller: null }), { holding: true }).stop).toBe(false);
    expect(loopControls(null, { holding: true }).stop).toBe(false);
  });

  it('enables the writing controls only to the holder while idle', () => {
    const s = snapshot({ step: STEP, last_action: { name: 'apply', stopped: false, moved: MOVED } });
    expect(loopControls(s, { holding: true })).toMatchObject({
      setup: true, shot: true, apply: true, putBack: true, continuous: true, selectReference: true,
    });
    expect(loopControls(s, { holding: false })).toMatchObject({
      setup: false, shot: false, apply: false, putBack: false, continuous: false,
    });
    expect(loopControls({ ...s, mode: 'action' }, { holding: true })).toMatchObject({
      setup: false, shot: false, apply: false, putBack: false, continuous: false,
    });
  });

  it('offers no apply without a step or on an all-zero step', () => {
    expect(loopControls(snapshot(), { holding: true }).apply).toBe(false);
    expect(loopControls(snapshot({ step: { ...STEP, all_zero: true } }), { holding: true }).apply).toBe(false);
  });

  it('offers no Put back after a put-back or a shot', () => {
    for (const name of ['put_back', 'shot']) {
      const s = snapshot({ last_action: { name, stopped: false, moved: MOVED } });
      expect(loopControls(s, { holding: true }).putBack).toBe(false);
    }
    const stoppedSetup = snapshot({ last_action: { name: 'setup', stopped: true, moved: MOVED } });
    expect(loopControls(stoppedSetup, { holding: true }).putBack).toBe(true);
  });

  it('offers no shot or continuous without a trajectory model', () => {
    const c = loopControls(snapshot({ model: null, model_kind: 'none' }), { holding: true });
    expect(c).toMatchObject({ shot: false, continuous: false, saveReference: false, setup: true });
  });
});

describe('shouldHeartbeat', () => {
  it('beats only while a run is in progress and this tab holds', () => {
    const running = snapshot({ continuous: { shots: 1, max_shots: 10, lease_expires_in_s: 12 } });
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

describe('views', () => {
  it('renders the header facts, the holder and the take-over notice', () => {
    const s = snapshot({
      holder: { owner: 'alice', kind: 'panel' },
      taken_over: { previous: { owner: 'bob', kind: 'agent' }, by: { owner: 'alice', kind: 'panel' } },
      writing_available: false,
    });
    const text = headerView(s, 'connected', true).textContent;
    for (const part of ['va', 'readwrite', 'optional', 'unavailable', 'SR (single_pass)',
      'panel of alice (this tab)', 'Taken over by panel of alice from agent of bob', 'Stream: connected']) {
      expect(text).toContain(part);
    }
  });

  it('names the running action, its stopping state, the budgets and Stop', () => {
    const s = snapshot({ holder: { owner: 'bob', kind: 'agent' }, mode: 'action',
      action: { name: 'apply', stopping: true, armed_stragglers: [] } });
    const node = actionView(s, sessionControls(s, { holding: false }), loopControls(s, { holding: false }),
      { onOpen: NOOP, onTakeOver: NOOP, onRelease: NOOP, onStop: NOOP });
    expect(node.querySelector('[data-running]')?.textContent).toBe('Running: apply (agent of bob) — stopping');
    expect(node.querySelector('[data-control="stop"]')).not.toBeNull();
    expect(node.querySelector('[data-control="takeover"]')).toBeNull();
    expect(node.textContent).toContain('65 s');
  });

  it('renders the step with deltas, rank and zeroed info, and the unsolved correctors named', () => {
    const node = stepView(STEP);
    expect(node.querySelector('[data-view="unsolved"]')?.textContent).toBe('Not solved: VCM7, VCM8');
    const rows = [...node.querySelectorAll('.deltas tbody tr')].map((tr) => tr.textContent);
    expect(rows).toEqual(['HCM10.5', 'HCM2-0.25']);
    for (const part of ['d-1', '0.5', 'solved with rank 2', '1 (n_discard 1)', 'HCM9', 'VCM4']) {
      expect(node.textContent).toContain(part);
    }
    expect(stepView(null).textContent).toContain('No step pending');
  });

  it('shows the last action moved list and the continuous lease', () => {
    const s = snapshot({ last_action: { name: 'setup', stopped: true, moved: MOVED },
      continuous: { shots: 2, max_shots: 50, lease_expires_in_s: 11.6 } });
    const node = statusView(s);
    expect(node.querySelector('[data-view="last-action"]')?.textContent).toContain('setup (stopped)');
    expect(node.querySelectorAll('[data-view="moved"] tbody tr')).toHaveLength(2);
    expect(node.querySelector('[data-view="continuous"]')?.textContent).toBe(
      'Running: 2 of 50 shot(s); lease expires in 12 s');
  });

  it('reads the loop form and disables what the controls forbid', () => {
    const form = loopForm({ onSetup: NOOP, onShot: NOOP, onApply: NOOP, onPutBack: NOOP, onContinuous: NOOP });
    form.nShots.value = '4';
    form.average.value = 'nanmean';
    expect(loopArgs(form)).toEqual({ n_shots: 4, average: 'nanmean', max_shots: null });
    form.maxShots.value = '20';
    expect(loopArgs(form).max_shots).toBe(20);
    setLoopEnabled(form, loopControls(snapshot(), { holding: false }));
    expect(form.buttons.shot.disabled).toBe(true);
  });

  it('lists references and shots, and saves under a typed name', () => {
    /** @type {[string, string | undefined][]} */
    const saved = [];
    const shots = [{ shot_id: 's-2', shots_included: 3, shots_failed: [], last_bpm: 'BPM9', lost: [] }];
    const node = referencesView({ names: ['golden', 'other'], current: 'golden' }, shots, snapshot(),
      loopControls(snapshot(), { holding: true }),
      { onSave: (n, s) => saved.push([n, s]), onSelect: NOOP, onRefresh: NOOP });
    const pick = /** @type {HTMLSelectElement} */ (node.querySelector('select[name="reference"]'));
    expect(pick.value).toBe('golden');
    expect(node.querySelectorAll('[data-view="shots"] tbody tr')).toHaveLength(1);
    /** @type {HTMLInputElement} */ (node.querySelector('input[name="reference_name"]')).value = ' new ';
    /** @type {HTMLButtonElement} */ (node.querySelector('[data-control="save-reference"]')).click();
    expect(saved).toEqual([['new', undefined]]);
  });

  it('writes service strings as text, never as markup', () => {
    const node = headerView(snapshot({ target: '<img src=x onerror=alert(1)>' }), 'connected', false);
    expect(node.querySelector('img')).toBeNull();
    expect(node.textContent).toContain('<img');
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

/**
 * Start a panel against a fake service answering by route.
 * @param {Record<string, (init: RequestInit) => [number, unknown]>} routes
 */
function harness(routes) {
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
  });
  return { panel, root, calls, timers, source: FakeSource.instances[0] };
}

/** @param {HTMLElement} root @param {string} name */
function click(root, name) {
  const button = /** @type {HTMLButtonElement | null} */ (root.querySelector(`[data-control="${name}"]`));
  if (!button) throw new Error(`no ${name} control`);
  button.click();
}

describe('startPanel', () => {
  it('threads open session, shot and apply with relative URLs and the client header', async () => {
    const { panel, root, calls, source } = harness({
      'POST api/session': () => [200, { client_id: 'c-1', owner: 'alice', target: 'va' }],
    });
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot() });
    click(root, 'open');
    await flush();
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot({ holder: { owner: 'alice', kind: 'panel' } }) });
    expect(root.textContent).toContain('panel of alice (this tab)');
    click(root, 'shot');
    await flush();
    source.emit('snapshot', { type: 'snapshot',
      snapshot: snapshot({ holder: { owner: 'alice', kind: 'panel' }, step: STEP }) });
    click(root, 'apply');
    await flush();
    const apply = calls.find((c) => c.url === 'api/apply');
    expect(apply?.init.body).toBe(JSON.stringify({ step_digest: 'd-1' }));
    const headers = /** @type {Record<string, string>} */ (apply?.init.headers ?? {});
    expect(headers['X-Trajectory-Client']).toBe('c-1');
    expect(calls.every((c) => !c.url.startsWith('/'))).toBe(true);
    panel.close();
  });

  it('drops the session on a lost-session refusal and shows the code', async () => {
    const { panel, root, source } = harness({
      'POST api/session': () => [200, { client_id: 'c-1', owner: 'alice', target: 'va' }],
      'POST api/shot': () => [409, { detail: { code: 'session_taken_over', detail: 'taken over' } }],
    });
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot() });
    click(root, 'open');
    await flush();
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot({ holder: { owner: 'alice', kind: 'panel' } }) });
    click(root, 'shot');
    await flush();
    expect(root.querySelector('[data-code]')?.textContent).toBe('session_taken_over');
    expect(root.querySelector('[data-control="takeover"]')).not.toBeNull();
    panel.close();
  });

  it('heartbeats while its continuous run lasts and shows lease_expired prominently', async () => {
    const { panel, root, calls, timers, source } = harness({
      'POST api/session': () => [200, { client_id: 'c-1', owner: 'alice', target: 'va' }],
      'POST api/continuous': () => [200, { shots: 0, max_shots: 10, lease_expires_in_s: 15 }],
    });
    const held = { holder: { owner: 'alice', kind: 'panel' } };
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot() });
    click(root, 'open');
    await flush();
    click(root, 'continuous');
    await flush();
    expect(panel.heartbeating()).toBe(true);
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot({ ...held, mode: 'continuous',
      continuous: { shots: 1, max_shots: 10, lease_expires_in_s: 14 } }) });
    timers[0]();
    await flush();
    expect(calls.some((c) => c.url === 'api/continuous/heartbeat')).toBe(true);
    source.emit('continuous_ended', { type: 'continuous_ended', end: 'lease_expired', shots: 1 });
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot(held) });
    expect(panel.heartbeating()).toBe(false);
    const banner = root.querySelector('[data-view="ended"]');
    expect(banner?.textContent).toContain('lease_expired');
    expect(banner?.querySelector('[role="alert"]')).not.toBeNull();
    panel.close();
  });

  it('puts back the earliest before-value of what the last apply moved', async () => {
    const { panel, root, calls, source } = harness({
      'POST api/session': () => [200, { client_id: 'c-1', owner: 'alice', target: 'va' }],
    });
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot() });
    click(root, 'open');
    await flush();
    const moved = [...MOVED, { name: 'HCM1', before: 9, outcome: 'verified' }];
    source.emit('snapshot', { type: 'snapshot', snapshot: snapshot({ holder: { owner: 'alice', kind: 'panel' },
      last_action: { name: 'apply', stopped: false, moved } }) });
    click(root, 'put-back');
    await flush();
    const putBack = calls.find((c) => c.url === 'api/apply');
    expect(JSON.parse(String(putBack?.init.body))).toEqual({
      model: 'SR', values: [{ name: 'HCM1', before: 1.5 }, { name: 'HCM2', before: -2 }],
    });
    panel.close();
  });
});
