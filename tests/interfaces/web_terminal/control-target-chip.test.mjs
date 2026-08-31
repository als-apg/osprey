// @ts-check
/**
 * Unit tests for the control-target header chip (control-target-chip.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-chip.test.mjs
 *
 * The chip is the header's one-line answer to "if the agent writes now, where
 * does it land, and will it be refused?". It reads ONE truth —
 * `GET /api/terminal/posture?session_id=` — and derives nothing from what this
 * page last did, so a posture narrowed in another tab and a target the agent
 * switched mid-turn both reach the operator.
 *
 * What these tests pin down, in the order a reviewer would ask about it:
 *
 * - the chip mounts once, into `.header-actions`, ahead of the palette trigger,
 *   and a second init re-renders rather than mounting a second chip;
 * - the full state matrix reaches the DOM as data attributes and words: four
 *   kinds (live / stand-in / virtual accelerator / simulated) × three states
 *   (writes / sandbox / read-only). Not one colour is decided here — the
 *   stylesheet owns the map, so the attributes ARE the contract;
 * - `sandbox` and `read-only` stay separate words: one is the operator's own
 *   narrowing and one click undoes it, the other is the deployment's ceiling;
 * - an `agent_activity` frame naming `control_target_set` triggers a REFETCH
 *   and nothing else — the frame's prose is the agent's narration of a switch
 *   that may belong to any client on the stream, and the chip never reads it;
 * - the 500 ms poll is armed only while a switch request this browser wrote is
 *   outstanding, and is torn down the moment the route publishes that
 *   `request_id`'s outcome;
 * - a request nothing ever answers (a dead controls server publishes no
 *   outcome at all) expires locally after REQUEST_TTL_S rather than showing
 *   `switching…` forever;
 * - a slow read cannot repaint over a newer one (`readSeq`);
 * - every request goes through api.js's `withPrefix`, so the multi-user
 *   per-user mount is covered.
 *
 * Seams: terminal.js is mocked (it owns the session id); `fetch` is stubbed the
 * way the other suites here stub it; the SSE factory is injected the way
 * session.js's `wireActivityStrip` injects it, since happy-dom has no
 * EventSource. Module-private state (the mounted chip, the last payload) has no
 * reset API, so each test gets a fresh module instance via vi.resetModules() +
 * dynamic import — same pattern as posture-badge.test.mjs.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SESSION = 'aaaaaaaa-1111-2222-3333-444444444444';
const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js';

/** Mutable stand-in for terminal.js, reachable from the hoisted vi.mock factory. */
const term = vi.hoisted(() => ({
  /** @type {string|null} */
  sessionId: /** @type {string|null} */ (null),
  /** @type {(() => void)[]} */
  listeners: [],
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => term.sessionId,
  /** @param {() => void} fn */
  onSessionChange: (fn) => term.listeners.push(fn),
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js')} */
let chipModule;

/* ---- roster fixtures ---------------------------------------------------- */

/**
 * The four machine kinds, in the shape the route publishes them. `kind` is the
 * PLAIN-LANGUAGE word the route mints from `real_machine` and the label's
 * shape; the chip maps it onto the CSS value, which is what these tests read.
 */
const KINDS = {
  live: {
    target: 'live',
    label: 'LIVE MACHINE',
    short_label: 'LIVE',
    kind: 'live machine',
    endpoint: 'als-gw.lbl.gov:5064',
    real_machine: true,
  },
  standin: {
    target: 'standin',
    label: 'LIVE MACHINE (stand-in)',
    short_label: 'STAND-IN',
    kind: 'stand-in',
    endpoint: '127.0.0.1:10090',
    real_machine: true,
  },
  va: {
    target: 'va',
    label: 'virtual accelerator (simulation)',
    short_label: 'VIRTUAL',
    kind: 'virtual accelerator',
    endpoint: '127.0.0.1:10064',
    real_machine: false,
  },
  simulated: {
    target: 'live',
    label: 'live machine (simulated)',
    short_label: 'SIMULATED',
    kind: 'simulated',
    endpoint: 'mock',
    real_machine: false,
  },
};

/**
 * The three effective states as the route's three columns express them.
 * `read-only` and `sandbox` differ only in `posture`: the effective answer is
 * the same "no", and which "no" it is decides whether a toggle can undo it.
 */
const STATES = {
  writes: { effective: true, posture: 'writes' },
  sandbox: { effective: false, posture: 'sandbox' },
  'read-only': { effective: false, posture: 'writes' },
};

/** @param {object} [o] */
const rowOf = (o = {}) => ({
  ...KINDS.standin,
  ...STATES.writes,
  active: false,
  is_baseline: false,
  available_now: true,
  reason: null,
  ceiling_writes: true,
  reachability: {
    state: 'reached',
    role: 'write_access',
    probed_at: '2026-08-30T12:00:00+00:00',
    age_s: 3,
    role_detail: { write_access: 'reached' },
  },
  ...o,
});

/** @param {object} [o] */
const viewOf = (o = {}) => ({
  session_id: SESSION,
  session_target: 'standin',
  store_available: true,
  enforceable: true,
  enforceable_reason: null,
  execution_in_flight: false,
  last_switch: null,
  last_posture_realign: null,
  targets: [rowOf({ ...KINDS.standin, active: true, is_baseline: true })],
  ...o,
});

/* ---- harness ------------------------------------------------------------ */

/** What GET /api/terminal/posture answers next. */
/** @type {any} */
let served;

/** @type {{url: string, method: string}[]} */
let fetchCalls = [];

/** GET count, the thing every refresh assertion is really about. */
const getCount = () => fetchCalls.filter((c) => c.method === 'GET').length;

/**
 * When true, the NEXT GET is parked instead of resolving. Its payload is
 * snapshotted at call time — that is the whole point: a held read carries the
 * answer the server gave BEFORE whatever happened next.
 */
let holdNextGet = false;
/** @type {(() => void)[]} Release functions for parked GETs, in order. */
let heldGets = [];

function stubFetch() {
  fetchCalls = [];
  holdNextGet = false;
  heldGets = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
      const method = init?.method ?? 'GET';
      fetchCalls.push({ url: String(url), method });
      const snapshot = JSON.parse(JSON.stringify(served));
      const ok = { ok: true, status: 200, statusText: 'OK', json: async () => snapshot };
      if (method === 'GET' && holdNextGet) {
        holdNextGet = false;
        return new Promise((resolve) => heldGets.push(() => resolve(ok)));
      }
      return ok;
    })
  );
}

/** The global header the chip mounts into (index.html's shape). */
function mountFixture() {
  document.body.innerHTML = `
    <header class="header">
      <div class="header-left"></div>
      <div class="header-right">
        <div class="header-actions">
          <button id="command-palette-btn" class="command-palette-trigger" type="button"></button>
          <osprey-display-menu id="display-menu"></osprey-display-menu>
        </div>
      </div>
    </header>`;
}

/** The injected SSE factory: records what was subscribed, drives it by hand. */
/** @type {{url: string|null, onMessage: ((data: any) => void)|null, stopped: number}} */
let stream;

function fakeEventSourceFactory() {
  stream = { url: null, onMessage: null, stopped: 0 };
  return /** @type {any} */ ((/** @type {string} */ url, /** @type {any} */ handlers) => {
    stream.url = url;
    stream.onMessage = handlers?.onMessage ?? null;
    return { stop: () => { stream.stopped += 1; } };
  });
}

/** Drain the microtask/timer queue the async handlers chain through. */
async function flush() {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Boot the chip against a given roster.
 * @param {any} [payload]
 * @param {{sessionId?: string|null}} [opts]
 */
async function boot(payload, opts = {}) {
  served = payload ?? viewOf();
  term.sessionId = opts.sessionId === undefined ? SESSION : opts.sessionId;
  chipModule.initControlTargetChip({ eventSourceFactory: fakeEventSourceFactory() });
  await flush();
}

const chipEl = () =>
  /** @type {HTMLButtonElement|null} */ (document.querySelector('.control-target-chip'));
const anchorEl = () => /** @type {HTMLElement|null} */ (document.querySelector('.ctc-anchor'));
const shortText = () => document.querySelector('.ctc-short')?.textContent ?? '';
const stateText = () => document.querySelector('.ctc-state')?.textContent ?? '';

/** Fire one agent-activity frame at the chip's subscription. */
function pushFrame(/** @type {any} */ frame) {
  stream.onMessage?.(frame);
}

beforeEach(async () => {
  vi.resetModules();
  term.sessionId = SESSION;
  term.listeners = [];
  served = viewOf();
  stubFetch();
  mountFixture();
  delete (/** @type {any} */ (window)).__OSPREY_PREFIX__;
  chipModule = await import(MODULE);
});

afterEach(() => {
  chipModule.teardownControlTargetChip();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/* ---- mount -------------------------------------------------------------- */

describe('mount', () => {
  test('mounts into .header-actions immediately before the palette trigger', async () => {
    await boot();
    const actions = /** @type {HTMLElement} */ (document.querySelector('.header-actions'));
    const anchor = /** @type {HTMLElement} */ (document.querySelector('.ctc-anchor'));
    const chip = chipEl();
    expect(chip).not.toBeNull();
    // The chip lives inside its own positioning context (the popover is
    // absolute under it), and that context is what sits in the action cluster.
    expect(chip?.parentElement).toBe(anchor);
    expect(anchor.parentElement).toBe(actions);
    expect(actions.firstElementChild).toBe(anchor);
    expect(anchor.nextElementSibling?.id).toBe('command-palette-btn');
    expect(chipModule.getAnchorElement()).toBe(anchor);
  });

  test('is idempotent — a second init re-renders rather than mounting twice', async () => {
    await boot();
    chipModule.initControlTargetChip({ eventSourceFactory: fakeEventSourceFactory() });
    await flush();
    expect(document.querySelectorAll('.control-target-chip')).toHaveLength(1);
    expect(document.querySelectorAll('.ctc-anchor')).toHaveLength(1);
  });

  test('carries the popover trigger ARIA and the four spans the stylesheet keys on', async () => {
    await boot();
    const chip = /** @type {HTMLButtonElement} */ (chipEl());
    expect(chip.type).toBe('button');
    expect(chip.getAttribute('aria-haspopup')).toBe('true');
    expect(chip.getAttribute('aria-expanded')).toBe('false');
    for (const cls of ['.ctc-dot', '.ctc-short', '.ctc-sep', '.ctc-state', '.ctc-caret']) {
      expect(chip.querySelector(cls)).not.toBeNull();
    }
    expect(chip.querySelector('.ctc-sep')?.textContent).toBe('·');
    expect(chip.querySelector('.ctc-caret')?.textContent).toBe('▾');
  });

  test('stays hidden, and reads nothing, until the terminal reports a session', async () => {
    await boot(viewOf(), { sessionId: null });
    expect(anchorEl()?.hidden).toBe(true);
    expect(chipEl()?.hidden).toBe(true);
    expect(getCount()).toBe(0);
  });

  test('does nothing at all on a page with no header', async () => {
    document.body.innerHTML = '<div id="elsewhere"></div>';
    chipModule.initControlTargetChip({ eventSourceFactory: fakeEventSourceFactory() });
    await flush();
    expect(chipEl()).toBeNull();
    expect(anchorEl()).toBeNull();
    expect(getCount()).toBe(0);
  });
});

/* ---- the state matrix --------------------------------------------------- */

describe('state matrix', () => {
  /** @type {Record<string, string>} */
  const KIND_ATTR = { live: 'live', standin: 'standin', va: 'va', simulated: 'simulated' };
  /** The consequence word each kind renders when no display_name is configured. */
  /** @type {Record<string, string>} */
  const WORDS = { live: 'Real machine', standin: 'Rehearsal', va: 'Simulator', simulated: 'Demo' };
  /** @type {Record<string, string>} */
  const PHRASES = { writes: 'writes on', sandbox: 'writes off', 'read-only': 'writes locked' };

  for (const [kindName, kind] of Object.entries(KINDS)) {
    for (const [stateName, state] of Object.entries(STATES)) {
      test(`${kindName} × ${stateName} → data attributes and words`, async () => {
        await boot(
          viewOf({
            session_target: kind.target,
            targets: [rowOf({ ...kind, ...state, active: true, is_baseline: true })],
          })
        );
        const chip = /** @type {HTMLButtonElement} */ (chipEl());
        expect(chip.hidden).toBe(false);
        expect(chip.dataset.targetKind).toBe(KIND_ATTR[kindName]);
        expect(chip.dataset.state).toBe(stateName);
        expect(shortText()).toBe(WORDS[kindName]);
        expect(stateText()).toBe(PHRASES[stateName]);
        // The tooltip is the FULL label; the chip's own word is the short one.
        expect(chip.title).toBe(kind.label);
      });
    }
  }

  test('an operator narrowing and the deployment ceiling are different words', async () => {
    await boot(
      viewOf({ targets: [rowOf({ ...KINDS.va, ...STATES.sandbox, active: true })] })
    );
    expect(stateText()).toBe('writes off');

    served = viewOf({
      targets: [rowOf({ ...KINDS.va, ...STATES['read-only'], ceiling_writes: false, active: true })],
    });
    await chipModule.refetch();
    expect(stateText()).toBe('writes locked');
  });

  test('kind falls back to real_machine + the label shape when the route sends none', async () => {
    await boot(
      viewOf({
        targets: [
          rowOf({
            ...KINDS.standin,
            kind: undefined,
            active: true,
          }),
        ],
      })
    );
    expect(chipEl()?.dataset.targetKind).toBe('standin');
  });

  test('an unrecognised real machine falls back to the loud answer', async () => {
    await boot(
      viewOf({
        targets: [
          rowOf({
            target: 'other',
            label: 'some machine nobody named',
            short_label: 'LIVE',
            kind: undefined,
            real_machine: true,
            active: true,
          }),
        ],
      })
    );
    expect(chipEl()?.dataset.targetKind).toBe('live');
  });

  test("the deployment's configured display_name wins over the kind default", async () => {
    await boot(
      viewOf({
        targets: [
          rowOf({ ...KINDS.live, display_name: 'ALS storage ring', active: true }),
        ],
      })
    );
    expect(shortText()).toBe('ALS storage ring');
    expect(chipEl()?.dataset.targetKind).toBe('live');
  });

  test('data-enforceable folds in the store, and a dimmed chip still renders', async () => {
    await boot();
    expect(chipEl()?.dataset.enforceable).toBe('true');

    served = viewOf({ store_available: false });
    await chipModule.refetch();
    expect(chipEl()?.dataset.enforceable).toBe('false');
    expect(anchorEl()?.hidden).toBe(false);

    served = viewOf({ enforceable: false, enforceable_reason: 'no session-owned record' });
    await chipModule.refetch();
    expect(chipEl()?.dataset.enforceable).toBe('false');
  });
});

/* ---- which row the chip speaks for -------------------------------------- */

describe('active row', () => {
  test('speaks for the active row, not the first one', async () => {
    await boot(
      viewOf({
        session_target: 'va',
        targets: [
          rowOf({ ...KINDS.live, is_baseline: true }),
          rowOf({ ...KINDS.va, active: true }),
        ],
      })
    );
    expect(shortText()).toBe('Simulator');
    expect(chipEl()?.dataset.targetKind).toBe('va');
  });

  test('falls back to the baseline row when no row is marked active', async () => {
    await boot(
      viewOf({
        session_target: 'nothing-configured',
        targets: [rowOf(KINDS.live), rowOf({ ...KINDS.standin, is_baseline: true })],
      })
    );
    expect(shortText()).toBe('Rehearsal');
  });

  test('hides itself rather than guessing when the roster is empty', async () => {
    await boot(viewOf({ targets: [] }));
    expect(anchorEl()?.hidden).toBe(true);
    expect(chipEl()?.hidden).toBe(true);
  });
});

/* ---- the refetch hint --------------------------------------------------- */

describe('agent-activity hint', () => {
  test('subscribes to the shared panel event stream', async () => {
    await boot();
    expect(stream.url).toBe('/api/files/events');
  });

  test('a control_target_set frame triggers a re-read', async () => {
    await boot();
    const before = getCount();
    pushFrame({ type: 'agent_activity', tool: 'control_target_set', target: { kind: 'config' } });
    await flush();
    expect(getCount()).toBe(before + 1);
  });

  test('renders the ROUTE, never the frame text', async () => {
    await boot(
      viewOf({ targets: [rowOf({ ...KINDS.standin, active: true, is_baseline: true })] })
    );
    expect(shortText()).toBe('Rehearsal');
    // The frame narrates a switch to the facility's own machine, and the route
    // — the only truth — still says stand-in. A chip that read the prose would
    // now claim LIVE on a session that never moved.
    pushFrame({
      type: 'agent_activity',
      tool: 'control_target_set',
      target: { kind: 'config', detail: 'standin → live · success' },
      ts: Date.now(),
    });
    await flush();
    expect(shortText()).toBe('Rehearsal');
    expect(chipEl()?.dataset.targetKind).toBe('standin');
  });

  test('ignores other tools, other frame types and unparsed frames', async () => {
    await boot();
    const before = getCount();
    pushFrame({ type: 'agent_activity', tool: 'channel_write', target: { kind: 'channel' } });
    pushFrame({ type: 'panel_focus', panel: 'gallery' });
    pushFrame('not json at all');
    pushFrame(null);
    await flush();
    expect(getCount()).toBe(before);
  });
});

/* ---- polling ------------------------------------------------------------ */

describe('polling', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] });
  });

  test('re-reads on the idle cadence', async () => {
    await boot();
    const before = getCount();
    vi.advanceTimersByTime(chipModule.IDLE_POLL_MS);
    await flush();
    expect(getCount()).toBe(before + 1);
  });

  test('markPending arms the fast poll and puts the chip in switching…', async () => {
    await boot();
    chipModule.markPending('r-7f21', 'live');
    expect(chipEl()?.dataset.pending).toBe('true');
    expect(stateText()).toBe('switching…');
    // `data-state` still describes the machine the session is ON — the switch
    // has not landed, and the dot must not go quiet before it does.
    expect(chipEl()?.dataset.state).toBe('writes');

    const before = getCount();
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS * 3);
    await flush();
    expect(getCount()).toBe(before + 3);
  });

  test('the fast poll is torn down when the route publishes that request_id', async () => {
    await boot();
    chipModule.markPending('r-7f21', 'live');
    served = viewOf({
      session_target: 'live',
      last_switch: { request_id: 'r-7f21', status: 'success', reason: null, age_s: 0 },
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();

    expect(chipModule.isPending()).toBe(false);
    expect(chipEl()?.dataset.pending).toBeUndefined();
    expect(stateText()).toBe('writes on');
    expect(shortText()).toBe('Real machine');

    const settled = getCount();
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS * 4);
    await flush();
    expect(getCount()).toBe(settled);
  });

  test('another tab’s outcome does not land this browser’s request', async () => {
    await boot();
    chipModule.markPending('r-mine', 'live');
    served = viewOf({
      last_switch: { request_id: 'r-someone-else', status: 'success', age_s: 1 },
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();
    expect(chipModule.isPending()).toBe(true);
    expect(stateText()).toBe('switching…');
  });

  test('synthesises request_expired when nothing ever answers', async () => {
    await boot();
    chipModule.markPending('r-dead', 'live');
    vi.advanceTimersByTime(chipModule.REQUEST_TTL_S * 1000);
    await flush();

    expect(chipModule.isPending()).toBe(false);
    expect(chipEl()?.dataset.pending).toBeUndefined();
    const last = chipModule.getState()?.last_switch;
    expect(last).toMatchObject({
      request_id: 'r-dead',
      target: 'live',
      status: 'expired',
      reason: 'request_expired',
      synthesized: true,
    });
    // And the fast poll is gone with it.
    const settled = getCount();
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS * 4);
    await flush();
    expect(getCount()).toBe(settled);
  });

  test('a late real outcome still wins over the synthesised expiry', async () => {
    await boot();
    chipModule.markPending('r-slow', 'live');
    vi.advanceTimersByTime(chipModule.REQUEST_TTL_S * 1000);
    await flush();
    expect(chipModule.getState()?.last_switch?.synthesized).toBe(true);

    served = viewOf({
      last_switch: { request_id: 'r-slow', status: 'refused', reason: 'unreachable', age_s: 2 },
    });
    await chipModule.refetch();
    expect(chipModule.getState()?.last_switch).toMatchObject({
      request_id: 'r-slow',
      status: 'refused',
      reason: 'unreachable',
    });
    expect(chipModule.getState()?.last_switch?.synthesized).toBeUndefined();
  });

  test('teardown releases both timers and the stream', async () => {
    await boot();
    chipModule.markPending('r-1');
    chipModule.teardownControlTargetChip();
    expect(stream.stopped).toBe(1);
    expect(chipEl()).toBeNull();
    expect(anchorEl()).toBeNull();

    const settled = getCount();
    vi.advanceTimersByTime(chipModule.IDLE_POLL_MS * 2);
    await flush();
    expect(getCount()).toBe(settled);
  });
});

/* ---- read ordering ------------------------------------------------------ */

describe('read ordering', () => {
  test('a slow read cannot repaint over a newer one', async () => {
    await boot();
    expect(shortText()).toBe('Rehearsal');

    // Park a read carrying the CURRENT (stand-in) roster.
    holdNextGet = true;
    const slow = chipModule.refetch();
    await flush();

    // The session moves; a newer read lands with the new roster.
    served = viewOf({
      session_target: 'live',
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    await chipModule.refetch();
    expect(shortText()).toBe('Real machine');

    // The parked read finally resolves, carrying history.
    heldGets.shift()?.();
    await slow;
    await flush();
    expect(shortText()).toBe('Real machine');
    expect(chipEl()?.dataset.targetKind).toBe('live');
  });

  test('a failed read says nothing rather than naming a machine nobody confirmed', async () => {
    await boot();
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => ({ detail: { error: 'store_unavailable', message: 'no store' } }),
      }))
    );
    await chipModule.refetch();
    expect(anchorEl()?.hidden).toBe(true);
    expect(chipModule.getState()).toBeNull();
    spy.mockRestore();
  });
});

/* ---- the multi-user prefix ---------------------------------------------- */

describe('withPrefix', () => {
  test('every read goes through the per-user mount', async () => {
    (/** @type {any} */ (window)).__OSPREY_PREFIX__ = '/u/alice';
    await boot();
    const gets = fetchCalls.filter((c) => c.method === 'GET');
    expect(gets.length).toBeGreaterThan(0);
    for (const call of gets) {
      expect(call.url.startsWith('/u/alice/api/terminal/posture?session_id=')).toBe(true);
    }
  });

  test('the session id is passed as the query the route reads', async () => {
    await boot();
    expect(fetchCalls[0].url).toBe(`/api/terminal/posture?session_id=${SESSION}`);
  });
});

/* ---- the API the popover consumes --------------------------------------- */

describe('popover API', () => {
  test('getState answers the payload the route sent', async () => {
    await boot();
    const state = chipModule.getState();
    expect(state?.session_target).toBe('standin');
    expect(state?.targets).toHaveLength(1);
  });

  test('subscribers are called after each render and can unsubscribe', async () => {
    await boot();
    const seen = /** @type {any[]} */ ([]);
    const off = chipModule.subscribe((s) => seen.push(s));
    await chipModule.refetch();
    expect(seen).toHaveLength(1);
    expect(seen[0]?.session_target).toBe('standin');

    off();
    await chipModule.refetch();
    expect(seen).toHaveLength(1);
  });

  test('a throwing subscriber does not stop the others', async () => {
    await boot();
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const seen = /** @type {any[]} */ ([]);
    chipModule.subscribe(() => { throw new Error('boom'); });
    chipModule.subscribe((s) => seen.push(s));
    await chipModule.refetch();
    expect(seen).toHaveLength(1);
    spy.mockRestore();
  });

  test('the chip click only toggles aria-expanded and announces it', async () => {
    await boot();
    const chip = /** @type {HTMLButtonElement} */ (chipEl());
    const seen = /** @type {boolean[]} */ ([]);
    chip.addEventListener(chipModule.CHIP_TOGGLE_EVENT, (e) => {
      seen.push(/** @type {any} */ (e).detail.expanded);
    });

    const before = getCount();
    chip.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(chip.getAttribute('aria-expanded')).toBe('true');
    expect(chipModule.isExpanded()).toBe(true);
    chip.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(chip.getAttribute('aria-expanded')).toBe('false');
    expect(seen).toEqual([true, false]);
    // Status and action are separate gestures: opening changes nothing.
    await flush();
    expect(fetchCalls.filter((c) => c.method !== 'GET')).toHaveLength(0);
    expect(getCount()).toBe(before);
  });

  test('setExpanded mirrors a dismissal the popover drove on its own', async () => {
    await boot();
    chipModule.setExpanded(true);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('true');
    chipModule.setExpanded(false);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('false');
  });

  test('getChipElement hands the popover its anchor', async () => {
    await boot();
    expect(chipModule.getChipElement()).toBe(chipEl());
  });

  test('refusalMessage unwraps all three refusal body shapes', async () => {
    expect(chipModule.refusalMessage({ detail: { error: 'x', message: 'dict detail' } }, 409)).toBe(
      'dict detail'
    );
    expect(chipModule.refusalMessage({ detail: 'string detail' }, 400)).toBe('string detail');
    expect(chipModule.refusalMessage(null, 503)).toContain('503');
  });
});

/* ---- session changes ---------------------------------------------------- */

describe('session changes', () => {
  test('re-reads when the terminal settles on a session', async () => {
    await boot(viewOf(), { sessionId: null });
    expect(getCount()).toBe(0);

    term.sessionId = SESSION;
    for (const fn of term.listeners) fn();
    await flush();
    expect(getCount()).toBe(1);
    expect(anchorEl()?.hidden).toBe(false);
  });

  test('a torn-down chip is not revived by a late session change', async () => {
    await boot();
    chipModule.teardownControlTargetChip();
    const settled = getCount();
    for (const fn of term.listeners) fn();
    await flush();
    expect(getCount()).toBe(settled);
  });
});
