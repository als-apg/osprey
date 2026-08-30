// @ts-check
/**
 * Unit tests for the control-target popover (control-target-popover.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-popover.test.mjs
 *
 * The popover is the panel behind the header chip: one row per configured
 * control target, and every gesture that CHANGES where this session writes. It
 * fetches nothing of its own — it subscribes to the chip and renders
 * `getState()` — so these tests drive it the way the page does: stub the
 * roster route, boot the chip, click the chip.
 *
 * What they pin down, in the order a reviewer would ask about it:
 *
 * - the toggle LOCKS for each of the five reasons, in the documented order,
 *   and the reason reaches both `.ctc-lock` and the `title` (simple mode drops
 *   the line and keeps the tooltip);
 * - arming (read-only → writes) raises a confirm and narrowing does NOT: only
 *   a gesture that can end with a write landing somewhere new asks;
 * - a confirmed change leaves the popover OPEN and re-renders the row in
 *   place, so an operator changing several rows never reopens it;
 * - Escape dismisses an open confirm BEFORE it closes the popover, and a click
 *   inside the confirm is not an outside click;
 * - Switch POSTs, shows `switching…` on that row, and renders the outcome the
 *   route publishes for the `request_id` — success, refusal, and the expiry a
 *   dead controls server never answers;
 * - a chat session's rows offer no Switch and live toggles;
 * - `Sandbox everything` POSTs `all` and renders every target the store
 *   `skipped`;
 * - FR5b: ONE DOM for both `ui_mode`s. The same markup is produced with
 *   `html[data-ui-mode]` set to `simple` and to `expert` — the density lives
 *   entirely in terminal.css, and a JS branch on it would be a second place
 *   for the rule to drift.
 *
 * Seams: terminal.js is mocked (it owns the session id); `fetch` is stubbed
 * the way the other suites here stub it; the chip's SSE factory is injected,
 * since happy-dom has no EventSource. Both modules hold module-private state
 * with no reset API beyond their teardowns, so each test gets fresh instances
 * via vi.resetModules() + dynamic import — same pattern as
 * control-target-chip.test.mjs.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SESSION = 'aaaaaaaa-1111-2222-3333-444444444444';
const CHIP = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js';
const POPOVER = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-popover.js';

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
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-popover.js')} */
let popoverModule;

/* ---- roster fixtures ---------------------------------------------------- */

/** The three machines a switch-capable deployment configures, as the route publishes them. */
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
};

/**
 * The three effective states in the route's columns. `read-only` and `sandbox`
 * are the same "no"; which one it is decides whether a toggle can undo it.
 */
const STATES = {
  writes: { effective: true, posture: 'writes' },
  sandbox: { effective: false, posture: 'sandbox' },
  'read-only': { effective: false, posture: 'writes', ceiling_writes: false },
};

/** @param {object} kind @param {object} [o] */
const rowOf = (kind, o = {}) => ({
  ...kind,
  ...STATES.writes,
  active: false,
  is_baseline: false,
  available_now: true,
  reason: null,
  ceiling_writes: true,
  narrowing_refusal: null,
  reachability: {
    state: 'reached',
    role: 'write_access',
    probed_at: '2026-08-30T12:00:00+00:00',
    age_s: 3,
    role_detail: {},
  },
  ...o,
});

/** The roster a plain switch-capable deployment answers with. */
const viewOf = (o = {}) => ({
  session_id: SESSION,
  session_target: 'standin',
  store_available: true,
  enforceable: true,
  enforceable_reason: null,
  execution_in_flight: false,
  last_switch: null,
  last_posture_realign: null,
  targets: [
    rowOf(KINDS.live, { ...STATES.sandbox }),
    rowOf(KINDS.standin, { active: true, is_baseline: true, available_now: false, reason: 'already_active' }),
    rowOf(KINDS.va),
  ],
  ...o,
});

/* ---- harness ------------------------------------------------------------ */

/** What the roster route answers next. */
/** @type {any} */
let served;

/** Queued answers for the next POSTs, in order: `{ok, body}`. */
/** @type {{ok: boolean, status?: number, body: any}[]} */
let postAnswers = [];

/** @type {{url: string, method: string, body: any}[]} */
let fetchCalls = [];

const posts = () => fetchCalls.filter((c) => c.method === 'POST');

function stubFetch() {
  fetchCalls = [];
  postAnswers = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
      const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) : null;
      fetchCalls.push({ url: String(url), method, body });
      if (method === 'GET') {
        const snapshot = JSON.parse(JSON.stringify(served));
        return { ok: true, status: 200, json: async () => snapshot };
      }
      const answer = postAnswers.shift() ?? { ok: true, body: {} };
      return {
        ok: answer.ok,
        status: answer.status ?? (answer.ok ? 200 : 409),
        json: async () => answer.body,
      };
    })
  );
}

/** The global header the chip mounts into (index.html's shape). */
function mountFixture() {
  document.body.innerHTML = `
    <header class="header">
      <div class="header-right">
        <div class="header-actions">
          <button id="command-palette-btn" type="button"></button>
        </div>
      </div>
    </header>
    <div id="outside"></div>`;
}

/** The injected SSE factory: happy-dom has no EventSource. */
function fakeEventSourceFactory() {
  return /** @type {any} */ (() => ({ stop: () => {} }));
}

/** Drain the microtask/timer queue the async handlers chain through. */
async function flush() {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Boot the chip and its popover against a roster.
 * @param {any} [payload]
 */
async function boot(payload) {
  served = payload ?? viewOf();
  term.sessionId = SESSION;
  chipModule.initControlTargetChip({ eventSourceFactory: fakeEventSourceFactory() });
  popoverModule.initControlTargetPopover();
  await flush();
}

/** Boot, then click the chip open. @param {any} [payload] */
async function bootOpen(payload) {
  await boot(payload);
  chipEl()?.click();
  await flush();
}

const chipEl = () =>
  /** @type {HTMLButtonElement|null} */ (document.querySelector('.control-target-chip'));
const popEl = () => /** @type {HTMLElement|null} */ (document.querySelector('.ctc-popover'));
const isOpen = () => Boolean(popEl()?.classList.contains('open'));
/** @param {string} target */
const rowEl = (target) =>
  /** @type {HTMLElement|null} */ (document.querySelector(`.ctc-row[data-target="${target}"]`));
/** @param {string} target @param {string} seg */
const segEl = (target, seg) =>
  /** @type {HTMLButtonElement|null} */ (
    rowEl(target)?.querySelector(`.ctc-seg[data-seg="${seg}"]`) ?? null
  );
/** @param {string} target */
const switchEl = (target) =>
  /** @type {HTMLButtonElement|null} */ (rowEl(target)?.querySelector('.ctc-switch') ?? null);
/** @param {string} target */
const outcomes = (target) =>
  [...(rowEl(target)?.querySelectorAll('.ctc-outcome') ?? [])].map((n) => n.textContent);
/** A confirm that is up — one on its way out carries `data-closing`. */
const confirmEl = () =>
  /** @type {HTMLElement|null} */ (
    document.querySelector('.posture-modal-overlay:not([data-closing])')
  );
const confirmTitle = () => confirmEl()?.querySelector('.posture-modal-title')?.textContent ?? '';
/** One node inside the confirm that is up. Throws rather than silently no-op. */
const inConfirm = (/** @type {string} */ sel) => {
  const node = confirmEl()?.querySelector(sel);
  if (!(node instanceof HTMLElement)) throw new Error(`no ${sel} in the confirm`);
  return node;
};
const confirmBtn = () =>
  /** @type {HTMLButtonElement|null} */ (
    confirmEl()?.querySelector('.posture-modal-confirm') ?? null
  );

const pressEscape = () =>
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

beforeEach(async () => {
  vi.resetModules();
  term.sessionId = SESSION;
  term.listeners = [];
  served = viewOf();
  stubFetch();
  mountFixture();
  document.documentElement.removeAttribute('data-ui-mode');
  delete (/** @type {any} */ (window)).__OSPREY_PREFIX__;
  chipModule = await import(CHIP);
  popoverModule = await import(POPOVER);
});

afterEach(() => {
  popoverModule.teardownControlTargetPopover();
  chipModule.teardownControlTargetChip();
  for (const stale of document.querySelectorAll('.posture-modal-overlay')) stale.remove();
  vi.unstubAllGlobals();
});

/* ---- mount and open/close ----------------------------------------------- */

describe('mounting', () => {
  test('mounts inside the chip anchor and starts closed', async () => {
    await boot();
    const anchor = document.querySelector('.ctc-anchor');
    expect(anchor?.contains(/** @type {Node} */ (popEl()))).toBe(true);
    expect(isOpen()).toBe(false);
    expect(chipEl()?.getAttribute('aria-controls')).toBe('control-target-popover');
  });

  test('a second init reuses the same popover', async () => {
    await boot();
    popoverModule.initControlTargetPopover();
    expect(document.querySelectorAll('.ctc-popover')).toHaveLength(1);
  });

  test('init no-ops on a page with no chip', () => {
    document.body.innerHTML = '<header></header>';
    expect(popoverModule.initControlTargetPopover()).toBeNull();
  });
});

describe('open and close', () => {
  test('the chip click opens it and clicking again closes it', async () => {
    await bootOpen();
    expect(isOpen()).toBe(true);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('true');

    chipEl()?.click();
    await flush();
    expect(isOpen()).toBe(false);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('false');
  });

  test('an outside click closes it and mirrors the state onto the chip', async () => {
    await bootOpen();
    /** @type {HTMLElement} */ (document.getElementById('outside')).click();
    expect(isOpen()).toBe(false);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('false');
  });

  test('a click inside the popover does not close it', async () => {
    await bootOpen();
    /** @type {HTMLElement} */ (document.querySelector('.ctc-head-title')).click();
    expect(isOpen()).toBe(true);
  });

  test('Escape closes it and returns focus to the chip', async () => {
    await bootOpen();
    pressEscape();
    expect(isOpen()).toBe(false);
    expect(document.activeElement).toBe(chipEl());
  });
});

/* ---- rows --------------------------------------------------------------- */

describe('rows', () => {
  test('one row per configured target, keyed on kind, state and realness', async () => {
    await bootOpen();
    expect(document.querySelectorAll('.ctc-row')).toHaveLength(3);

    const live = /** @type {HTMLElement} */ (rowEl('live'));
    expect(live.dataset.targetKind).toBe('live');
    expect(live.dataset.state).toBe('sandbox');
    expect(live.dataset.real).toBe('true');
    expect(live.dataset.active).toBe('false');

    const standin = /** @type {HTMLElement} */ (rowEl('standin'));
    expect(standin.dataset.targetKind).toBe('standin');
    expect(standin.dataset.active).toBe('true');
    expect(standin.querySelector('.ctc-tag-current')?.textContent).toBe('current');

    const va = /** @type {HTMLElement} */ (rowEl('va'));
    expect(va.dataset.targetKind).toBe('va');
    expect(va.dataset.real).toBe('false');
    expect(va.dataset.state).toBe('writes');
  });

  test('a row names the machine, where it points and what reached it', async () => {
    await bootOpen();
    const va = /** @type {HTMLElement} */ (rowEl('va'));
    expect(va.querySelector('.ctc-label')?.textContent).toBe('virtual accelerator (simulation)');
    expect(va.querySelector('.ctc-kind')?.textContent).toBe('virtual accelerator');
    expect(va.querySelector('.ctc-endpoint')?.textContent).toBe('127.0.0.1:10064');
    expect(va.querySelector('.ctc-role')?.textContent).toBe('write_access');

    const reach = /** @type {HTMLElement} */ (va.querySelector('.ctc-reach'));
    expect(reach.dataset.state).toBe('reached');
    expect(reach.querySelector('.ctc-reach-text')?.textContent).toBe('connected · 3 s');
    // The measured word and the role stay on the title, which is the whole of
    // what simple mode's LED-only row carries.
    expect(reach.title).toContain('reached · 3 s');
    expect(reach.title).toContain('write_access endpoint');
  });

  test('the reach LED is a direct child of .ctc-ident, never inside .ctc-meta', async () => {
    // FR5b hangs on this. Simple mode hides `.ctc-meta` WHOLESALE (terminal.css
    // "simple-mode gating") and keeps `.ctc-reach` as the row's LED — nested
    // inside the meta line the LED would vanish with it, silently, and simple
    // mode would lose the one signal that says whether the machine is there.
    // The word and age live in `.ctc-reach-text`, which is the supported shape:
    // a bare text node would only collapse through the `font-size: 0` rule.
    await bootOpen();
    for (const row of document.querySelectorAll('.ctc-row')) {
      const ident = /** @type {HTMLElement} */ (row.querySelector('.ctc-ident'));
      const reach = /** @type {HTMLElement} */ (row.querySelector('.ctc-reach'));
      expect(reach.parentElement).toBe(ident);
      expect(row.querySelector('.ctc-meta')?.contains(reach)).toBe(false);
      expect(reach.querySelector('.ctc-reach-dot')).not.toBeNull();
      expect(reach.querySelector('.ctc-reach-text')).not.toBeNull();
    }
  });

  test('no row ever claims the kind that never renders', async () => {
    // `_label()` can produce "live machine (not configured)", but
    // `configured_targets()` never lists such a target and no session can sit
    // on one, so `unconfigured` is a stylesheet fallback with no data behind it.
    await bootOpen();
    expect(document.querySelectorAll('[data-target-kind="unconfigured"]')).toHaveLength(0);
    for (const row of document.querySelectorAll('.ctc-row')) {
      expect(['live', 'standin', 'va', 'simulated']).toContain(
        /** @type {HTMLElement} */ (row).dataset.targetKind
      );
    }
  });

  test('down and stale collapse to the plain words, keeping the measured state', async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, { reachability: { state: 'down', role: 'read_only', age_s: 12 } }),
          rowOf(KINDS.va, { reachability: { state: 'stale', role: null, age_s: 90 } }),
        ],
      })
    );
    const live = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-reach'));
    expect(live.dataset.state).toBe('down');
    expect(live.querySelector('.ctc-reach-text')?.textContent).toBe('unreachable · 12 s');

    const va = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-reach'));
    expect(va.dataset.state).toBe('stale');
    expect(va.querySelector('.ctc-reach-text')?.textContent).toBe('unknown · 90 s');
    expect(va.title).toContain('last probe older than the prober interval');
  });

  test('the baseline row is tagged on the reach line', async () => {
    await bootOpen();
    expect(rowEl('standin')?.querySelector('.ctc-baseline')?.textContent).toBe('baseline');
    expect(rowEl('va')?.querySelector('.ctc-baseline')).toBeNull();
  });

  test('the head note is empty in the plain case', async () => {
    await bootOpen();
    expect(document.querySelector('.ctc-head-title')?.textContent).toBe(
      'Control target · this session'
    );
    const note = /** @type {HTMLElement} */ (document.querySelector('.ctc-head-note'));
    expect(note.textContent).toBe('');
    expect(note.dataset.tone).toBeUndefined();
  });

  test('the foot carries the one-gesture narrowing and the config sentence', async () => {
    await bootOpen();
    expect(document.querySelector('.ctc-sandbox-all')?.textContent).toBe('Sandbox everything');
    expect(document.querySelector('.ctc-foot-note')?.textContent).toBe(
      "Nothing here changes the deployment's config."
    );
  });
});

/* ---- locks -------------------------------------------------------------- */

describe('the toggle locks, one reason at a time', () => {
  /** @param {any} view @param {string} target */
  async function lockOn(view, target) {
    await bootOpen(view);
    const toggle = /** @type {HTMLElement} */ (rowEl(target)?.querySelector('.ctc-toggle'));
    return {
      toggle,
      lock: rowEl(target)?.querySelector('.ctc-lock')?.textContent ?? null,
    };
  }

  test('store unavailable outranks everything else', async () => {
    const { toggle, lock } = await lockOn(
      viewOf({ store_available: false, enforceable: false }),
      'va'
    );
    expect(lock).toBe('store unavailable');
    expect(toggle.dataset.locked).toBe('true');
    expect(toggle.title).toBe('store unavailable');
    expect(segEl('va', 'writes')?.disabled).toBe(true);
    expect(segEl('va', 'read-only')?.disabled).toBe(true);
    // The reason reaches the title too: simple mode drops the line and keeps
    // the tooltip.
    expect(segEl('va', 'writes')?.title).toBe('store unavailable');
  });

  test('not enforceable, when the store is there but nothing reads it', async () => {
    const { lock } = await lockOn(
      viewOf({ enforceable: false, enforceable_reason: 'no_session_record' }),
      'va'
    );
    expect(lock).toBe('not enforceable');
    expect(document.querySelector('.ctc-head-note')?.textContent).toBe(
      'not enforceable · no_session_record'
    );
  });

  test('a readonly run: the ceiling is up, nothing is narrowed, writes are still off', async () => {
    const readonly = { ceiling_writes: true, posture: 'writes', effective: false };
    const { lock } = await lockOn(
      viewOf({
        targets: [rowOf(KINDS.live, readonly), rowOf(KINDS.va, readonly)],
      }),
      'va'
    );
    expect(lock).toBe('readonly run');
    expect(document.querySelector('.ctc-head-note')?.textContent).toBe(
      'readonly run · deployment-wide'
    );
  });

  test('a persona ceiling that never armed the target', async () => {
    const { lock } = await lockOn(
      viewOf({ targets: [rowOf(KINDS.va, { ...STATES['read-only'] })] }),
      'va'
    );
    expect(lock).toBe('persona ceiling');
  });

  test('no read-only endpoint: narrowing would select a role nothing configures', async () => {
    const { lock } = await lockOn(
      viewOf({ targets: [rowOf(KINDS.va, { narrowing_refusal: 'selected_role_missing' })] }),
      'va'
    );
    expect(lock).toBe('no read-only endpoint');
  });

  test('an unlocked toggle carries no lock line and stays live', async () => {
    await bootOpen();
    const toggle = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-toggle'));
    expect(toggle.dataset.locked).toBe('false');
    expect(rowEl('va')?.querySelector('.ctc-lock')).toBeNull();
    expect(segEl('va', 'writes')?.getAttribute('aria-pressed')).toBe('true');
    expect(segEl('va', 'read-only')?.getAttribute('aria-pressed')).toBe('false');
  });
});

/* ---- posture gestures --------------------------------------------------- */

describe('narrowing and arming', () => {
  test('narrowing applies on click, with no confirm', async () => {
    await bootOpen();
    postAnswers.push({ ok: true, body: { entry: { va: 'sandbox' }, skipped: [] } });
    segEl('va', 'read-only')?.click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(1);
    expect(posts()[0].url).toContain('/api/terminal/posture');
    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'va', posture: 'sandbox' });
  });

  test('arming raises a confirm and POSTs nothing until it is confirmed', async () => {
    await bootOpen();
    segEl('live', 'writes')?.click();
    await flush();

    expect(confirmTitle()).toBe('Allow writes on LIVE MACHINE?');
    expect(posts()).toHaveLength(0);
    // The facility's own machine, and only it, carries the hardware notice.
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Real machine. A confirmed write moves hardware.'
    );
    expect(confirmBtn()?.dataset.live).toBe('true');
    expect(confirmBtn()?.textContent).toBe('Allow writes');
  });

  test('the simulator arms without the hardware notice', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.sandbox })] }));
    segEl('va', 'writes')?.click();
    await flush();
    expect(confirmEl()?.querySelector('.posture-modal-live')).toBeNull();
  });

  test('cancelling the confirm changes nothing and leaves the popover open', async () => {
    await bootOpen();
    segEl('live', 'writes')?.click();
    await flush();
    inConfirm('.posture-modal-cancel').click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(0);
    expect(isOpen()).toBe(true);
  });

  test('confirming POSTs, keeps the popover open, and re-renders the row in place', async () => {
    await bootOpen();
    segEl('live', 'writes')?.click();
    await flush();

    postAnswers.push({ ok: true, body: { entry: {}, skipped: [] } });
    // The next read is the truth the row repaints from — never what this click
    // intended.
    served = viewOf({
      targets: [
        rowOf(KINDS.live),
        rowOf(KINDS.standin, { active: true, is_baseline: true, available_now: false, reason: 'already_active' }),
        rowOf(KINDS.va),
      ],
    });
    confirmBtn()?.click();
    await flush();

    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'live', posture: 'writes' });
    expect(confirmEl()).toBeNull();
    expect(isOpen()).toBe(true);
    expect(rowEl('live')?.dataset.state).toBe('writes');
    expect(segEl('live', 'writes')?.getAttribute('aria-pressed')).toBe('true');
  });

  test('a refused narrowing keeps the server sentence on the row', async () => {
    await bootOpen();
    postAnswers.push({
      ok: false,
      status: 409,
      body: { detail: { error: 'execution_in_flight', message: 'An execution is running.' } },
    });
    segEl('va', 'read-only')?.click();
    await flush();

    expect(outcomes('va')).toContain('✗ An execution is running.');
  });

  test('a refused arming keeps the dialog up carrying the reason', async () => {
    await bootOpen();
    segEl('live', 'writes')?.click();
    await flush();
    postAnswers.push({
      ok: false,
      status: 403,
      body: { detail: { error: 'ceiling', message: 'This render arms no writes there.' } },
    });
    confirmBtn()?.click();
    await flush();

    // The dialog is where the operator is looking, and nothing was applied.
    expect(confirmEl()).not.toBeNull();
    expect(confirmEl()?.querySelector('.posture-modal-error')?.textContent).toBe(
      'This render arms no writes there.'
    );
    expect(confirmBtn()?.disabled).toBe(false);
    expect(isOpen()).toBe(true);
  });
});

describe('Sandbox everything', () => {
  test('POSTs the all-targets narrowing and renders what the store skipped', async () => {
    await bootOpen();
    postAnswers.push({
      ok: true,
      body: { entry: {}, skipped: [{ target: 'va', reason: 'selected_role_missing' }] },
    });
    /** @type {HTMLButtonElement} */ (document.querySelector('.ctc-sandbox-all')).click();
    await flush();

    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'all', posture: 'sandbox' });
    expect(outcomes('va')).toContain('✗ selected_role_missing');
  });

  test('it is disabled when there is nothing left to lift', async () => {
    await bootOpen(
      viewOf({
        targets: [rowOf(KINDS.live, { ...STATES.sandbox }), rowOf(KINDS.va, { ...STATES.sandbox })],
      })
    );
    expect(
      /** @type {HTMLButtonElement} */ (document.querySelector('.ctc-sandbox-all')).disabled
    ).toBe(true);
  });
});

/* ---- switching ---------------------------------------------------------- */

describe('switching', () => {
  test('Switch confirms, POSTs, and shows switching… on that row', async () => {
    await bootOpen();
    switchEl('va')?.click();
    await flush();

    expect(confirmTitle()).toBe('Switch to virtual accelerator (simulation)?');
    expect(confirmBtn()?.textContent).toBe('Switch');

    postAnswers.push({ ok: true, body: { request_id: 'req-1', target: 'va' } });
    confirmBtn()?.click();
    await flush();

    expect(posts()[0].url).toContain('/api/terminal/target');
    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'va' });
    expect(chipModule.isPending()).toBe(true);
    expect(outcomes('va')).toContain('switching…');
    // One outstanding gesture at a time: no row offers a second Switch.
    expect(document.querySelectorAll('.ctc-switch')).toHaveLength(0);
    expect(isOpen()).toBe(true);
  });

  test('the outcome the route publishes for that request lands on the row', async () => {
    await bootOpen();
    switchEl('va')?.click();
    await flush();
    postAnswers.push({ ok: true, body: { request_id: 'req-1', target: 'va' } });
    confirmBtn()?.click();
    await flush();

    served = viewOf({
      session_target: 'va',
      last_switch: {
        request_id: 'req-1',
        target: 'va',
        status: 'success',
        reason: null,
        age_s: 4,
      },
      targets: [
        rowOf(KINDS.live, { ...STATES.sandbox }),
        rowOf(KINDS.standin, { is_baseline: true }),
        rowOf(KINDS.va, { active: true, available_now: false, reason: 'already_active' }),
      ],
    });
    await chipModule.refetch();
    await flush();

    expect(chipModule.isPending()).toBe(false);
    expect(outcomes('va')).toContain('✓ switched · 4 s ago');
  });

  test('a refusal renders the gate word, not the status', async () => {
    await bootOpen(
      viewOf({
        last_switch: {
          request_id: 'req-9',
          target: 'va',
          status: 'refused',
          reason: 'unreachable',
          age_s: 2,
        },
      })
    );
    expect(outcomes('va')).toContain('✗ unreachable');
    const line = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-outcome'));
    expect(line.dataset.status).toBe('refused');
  });

  test('an outcome that names no target renders on no row', async () => {
    // The other half of the contract. The reconciler publishes `target` with
    // every terminus, and a block that arrives without one is not spread over
    // the roster as a guess about which machine it meant.
    await bootOpen(
      viewOf({
        last_switch: { request_id: 'req-9', status: 'success', reason: null, age_s: 1 },
      })
    );
    expect(outcomes('live')).toHaveLength(0);
    expect(outcomes('standin')).toHaveLength(0);
    expect(outcomes('va')).toHaveLength(0);
  });

  test('a request nothing answered renders as expired', async () => {
    await bootOpen(
      viewOf({
        last_switch: {
          request_id: 'req-9',
          target: 'va',
          status: 'expired',
          reason: 'request_expired',
          age_s: 0,
          synthesized: true,
        },
      })
    );
    expect(outcomes('va')).toContain('✗ request_expired');
    const line = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-outcome'));
    expect(line.dataset.status).toBe('expired');
  });

  test('an outcome older than the freshness window is history, not news', async () => {
    await bootOpen(
      viewOf({
        last_switch: {
          request_id: 'req-9',
          target: 'va',
          status: 'success',
          reason: null,
          age_s: popoverModule.OUTCOME_MAX_AGE_S + 1,
        },
      })
    );
    expect(outcomes('va')).toHaveLength(0);
  });

  test('no Switch where the route offers none; the refusal word takes its place', async () => {
    await bootOpen();
    // The active row says `current` and offers nothing.
    expect(switchEl('standin')).toBeNull();
    expect(rowEl('standin')?.querySelector('.ctc-reason')).toBeNull();

    await bootOpen(
      viewOf({ targets: [rowOf(KINDS.live, { available_now: false, reason: 'unreachable' })] })
    );
    expect(switchEl('live')).toBeNull();
    expect(rowEl('live')?.querySelector('.ctc-reason')?.textContent).toBe('unreachable');
  });

  test('a store that cannot be resolved explains the missing Switch', async () => {
    await bootOpen(
      viewOf({ store_available: false, targets: [rowOf(KINDS.live, { reason: null })] })
    );
    expect(rowEl('live')?.querySelector('.ctc-reason')?.textContent).toBe('store_unavailable');
  });

  test('switching onto the facility machine says what a write would do there', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.live)] }));
    switchEl('live')?.click();
    await flush();
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Real machine, writes armed. The next write the agent makes lands on hardware.'
    );

    await bootOpen(viewOf({ targets: [rowOf(KINDS.live, { ...STATES.sandbox })] }));
    switchEl('live')?.click();
    await flush();
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Real machine. Writes are sandboxed on it for this session.'
    );
  });

  test('a refused switch keeps the dialog up carrying the server sentence', async () => {
    await bootOpen();
    switchEl('va')?.click();
    await flush();
    postAnswers.push({
      ok: false,
      status: 409,
      body: {
        detail: {
          error: 'session_not_started',
          message: 'This session has no running control-system server yet.',
        },
      },
    });
    confirmBtn()?.click();
    await flush();

    expect(confirmEl()).not.toBeNull();
    expect(confirmEl()?.querySelector('.posture-modal-error')?.textContent).toBe(
      'This session has no running control-system server yet.'
    );
    expect(chipModule.isPending()).toBe(false);
  });
});

/* ---- the confirm sits ABOVE the popover, which stays open ---------------- */

describe('a confirm and the popover beneath it', () => {
  test('a click inside the confirm is not an outside click', async () => {
    await bootOpen();
    segEl('live', 'writes')?.click();
    await flush();

    inConfirm('.posture-modal-body').click();
    expect(isOpen()).toBe(true);
    expect(confirmEl()).not.toBeNull();
  });

  test('Escape dismisses the confirm first and the popover only after', async () => {
    await bootOpen();
    segEl('live', 'writes')?.click();
    await flush();

    pressEscape();
    expect(confirmEl()).toBeNull();
    expect(isOpen()).toBe(true);

    pressEscape();
    expect(isOpen()).toBe(false);
  });

  test('closing the popover takes any confirm with it', async () => {
    await bootOpen();
    switchEl('va')?.click();
    await flush();
    chipEl()?.click();
    await flush();

    expect(isOpen()).toBe(false);
    expect(confirmEl()).toBeNull();
  });
});

/* ---- chat sessions ------------------------------------------------------ */

describe('a chat session', () => {
  const chatView = () =>
    viewOf({
      targets: [
        rowOf(KINDS.live, { available_now: false, reason: 'chat_session' }),
        rowOf(KINDS.va, { available_now: false, reason: 'chat_session', active: true }),
      ],
    });

  test('offers no Switch and no refusal word in its place', async () => {
    await bootOpen(chatView());
    expect(document.querySelectorAll('.ctc-switch')).toHaveLength(0);
    expect(document.querySelectorAll('.ctc-reason')).toHaveLength(0);
    expect(document.querySelector('.ctc-head-note')?.textContent).toBe('chat session');
  });

  test('keeps its toggles live — posture is keyed on the session, not the topology', async () => {
    await bootOpen(chatView());
    expect(segEl('live', 'read-only')?.disabled).toBe(false);

    postAnswers.push({ ok: true, body: { entry: { live: 'sandbox' }, skipped: [] } });
    segEl('live', 'read-only')?.click();
    await flush();
    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'live', posture: 'sandbox' });
  });
});

/* ---- realign and the execution in flight --------------------------------- */

describe('a narrowing that has not reached the agent yet', () => {
  test('the active row says the read-only applies after the run', async () => {
    await bootOpen(
      viewOf({ execution_in_flight: true, last_posture_realign: { state: 'pending' } })
    );
    expect(outcomes('standin')).toContain(
      'read-only applies after the running execution finishes'
    );
    expect(outcomes('va')).toHaveLength(0);
    expect(document.querySelector('.ctc-head-note')?.textContent).toBe('execution running');
  });
});

/* ---- FR5b: one DOM for both densities ------------------------------------ */

describe('simple and expert are one DOM', () => {
  /**
   * Every popover render, as markup, for a given ui-mode.
   * @param {string} mode @param {any} view
   */
  async function markupUnder(mode, view) {
    document.documentElement.setAttribute('data-ui-mode', mode);
    await bootOpen(view);
    const html = popEl()?.innerHTML ?? '';
    popoverModule.teardownControlTargetPopover();
    chipModule.teardownControlTargetChip();
    return html;
  }

  test('the plain roster produces identical markup in either mode', async () => {
    const simple = await markupUnder('simple', viewOf());
    const expert = await markupUnder('expert', viewOf());
    expect(simple).toBe(expert);
    // And every expert-only piece is present in BOTH: terminal.css hides them.
    expect(simple).toContain('ctc-meta');
    expect(simple).toContain('ctc-kind');
    expect(simple).toContain('ctc-reach-text');
    expect(simple).toContain('ctc-baseline');
    expect(simple).toContain('ctc-foot-note');
  });

  test('a locked, unreachable, not-enforceable roster is identical too', async () => {
    const view = () =>
      viewOf({
        enforceable: false,
        enforceable_reason: 'no_session_record',
        targets: [
          rowOf(KINDS.live, {
            ...STATES.sandbox,
            reachability: { state: 'down', role: 'read_only', age_s: 8 },
          }),
          rowOf(KINDS.va, { is_baseline: true, active: true }),
        ],
      });
    const simple = await markupUnder('simple', view());
    const expert = await markupUnder('expert', view());
    expect(simple).toBe(expert);
    expect(simple).toContain('ctc-lock');
    expect(simple).toContain('ctc-head-note');
  });
});

/* ---- request plumbing ---------------------------------------------------- */

describe('request plumbing', () => {
  test('every request goes through the per-user mount prefix', async () => {
    /** @type {any} */ (window).__OSPREY_PREFIX__ = '/u/alice';
    await bootOpen();
    postAnswers.push({ ok: true, body: { entry: {}, skipped: [] } });
    segEl('va', 'read-only')?.click();
    await flush();

    expect(fetchCalls.length).toBeGreaterThan(0);
    for (const call of fetchCalls) expect(call.url.startsWith('/u/alice/')).toBe(true);
  });
});
