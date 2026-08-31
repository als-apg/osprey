// @ts-check
/**
 * Unit tests for the control-target popover (control-target-popover.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-popover.test.mjs
 *
 * The popover is the panel behind the header chip: a card for the machine the
 * agent stands on, a row per other configured target, and every gesture that
 * CHANGES where this session writes. It fetches nothing of its own — it
 * subscribes to the chip and renders `getState()` — so these tests drive it
 * the way the page does: stub the roster route, boot the chip, click the chip.
 *
 * What they pin down, in the order a reviewer would ask about it:
 *
 * - machines are named by consequence (`Real machine`, `Rehearsal`,
 *   `Simulator`), the deployment's configured `display_name` wins, and the
 *   server's implementation label survives only on the tooltip;
 * - NO PROCESS CLAIMS: nothing rendered — rows, card, either confirm — states
 *   whether a write asks for approval or what limits apply, because that is
 *   deployment configuration the browser cannot see;
 * - the verb locks for each of the five reasons, in the documented order, and
 *   the reason reaches the `title` in operator words;
 * - turning writes on raises a confirm and turning them off does NOT: only a
 *   gesture that can end with a write landing somewhere new asks;
 * - a confirmed change leaves the popover OPEN and re-renders in place, so an
 *   operator changing several machines never reopens it;
 * - Escape dismisses an open confirm BEFORE it closes the popover, and a click
 *   inside the confirm is not an outside click;
 * - Switch to confirms, POSTs, shows `switching…` on that row, and renders the
 *   outcome the route publishes for the `request_id` — success, refusal, and
 *   the expiry a dead controls server never answers;
 * - a refusal code renders as its operator phrase, never raw; a code the map
 *   does not know renders verbatim;
 * - a chat session's rows offer no Switch and live verbs;
 * - `Turn all writes off` POSTs `all` and renders every target the store
 *   `skipped`;
 * - ONE DOM for both `ui_mode`s: the same markup under `simple` and `expert`.
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
    display_name: '',
    endpoint: 'als-gw.lbl.gov:5064',
    real_machine: true,
  },
  standin: {
    target: 'standin',
    label: 'LIVE MACHINE (stand-in)',
    short_label: 'STAND-IN',
    kind: 'stand-in',
    display_name: '',
    endpoint: '127.0.0.1:10090',
    real_machine: true,
  },
  va: {
    target: 'va',
    label: 'virtual accelerator (simulation)',
    short_label: 'VIRTUAL',
    kind: 'virtual accelerator',
    display_name: '',
    endpoint: '127.0.0.1:10064',
    real_machine: false,
  },
};

/**
 * The three effective states in the route's columns. `read-only` and `sandbox`
 * are the same "no"; which one it is decides whether a verb can undo it.
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
  reason_detail: null,
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
const cardEl = () => /** @type {HTMLElement|null} */ (document.querySelector('.ctc-card'));
const cardVerb = () =>
  /** @type {HTMLButtonElement|null} */ (document.querySelector('.ctc-card-verb'));
const bannerEl = () => /** @type {HTMLElement|null} */ (document.querySelector('.ctc-banner'));
/** @param {string} target */
const rowEl = (target) =>
  /** @type {HTMLElement|null} */ (document.querySelector(`.ctc-row[data-target="${target}"]`));
/** The card or the row that carries this target. @param {string} target */
const surfaceEl = (target) => {
  const card = cardEl();
  if (card?.dataset.target === target) return card;
  return rowEl(target);
};
/** @param {string} target */
const verbEl = (target) => {
  const surface = surfaceEl(target);
  return /** @type {HTMLButtonElement|null} */ (
    surface?.querySelector('.ctc-verb, .ctc-card-verb') ?? null
  );
};
/** @param {string} target */
const pillEl = (target) =>
  /** @type {HTMLElement|null} */ (rowEl(target)?.querySelector('.ctc-pill') ?? null);
/** @param {string} target */
const switchEl = (target) =>
  /** @type {HTMLButtonElement|null} */ (rowEl(target)?.querySelector('.ctc-switch') ?? null);
/** @param {string} target */
const outcomes = (target) =>
  [...(surfaceEl(target)?.querySelectorAll('.ctc-outcome') ?? [])].map((n) => n.textContent);
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
    /** @type {HTMLElement} */ (document.querySelector('.ctc-list-title')).click();
    expect(isOpen()).toBe(true);
  });

  test('Escape closes it and returns focus to the chip', async () => {
    await bootOpen();
    pressEscape();
    expect(isOpen()).toBe(false);
    expect(document.activeElement).toBe(chipEl());
  });
});

/* ---- the card ------------------------------------------------------------ */

describe('the card', () => {
  test('the machine the agent stands on is the card, not a row', async () => {
    await bootOpen();
    const card = /** @type {HTMLElement} */ (cardEl());
    expect(card.dataset.target).toBe('standin');
    expect(card.dataset.targetKind).toBe('standin');
    expect(card.dataset.state).toBe('writes');
    expect(card.dataset.real).toBe('true');
    expect(rowEl('standin')).toBeNull();
    expect(card.querySelector('.ctc-card-eyebrow')?.textContent).toBe('The agent is on');
  });

  test('the card names the machine by consequence and keeps the label on hover', async () => {
    await bootOpen();
    const title = /** @type {HTMLElement} */ (cardEl()?.querySelector('.ctc-card-title'));
    expect(title.querySelector('.ctc-name')?.textContent).toBe('Rehearsal');
    // The implementation vocabulary lives on the tooltip, never at rest.
    expect(title.title).toContain('LIVE MACHINE (stand-in)');
    expect(title.title).toContain('127.0.0.1:10090');
    expect(cardEl()?.querySelector('.ctc-desc')?.textContent).toBe(
      "Copy of the real machine's controls · nothing moves"
    );
    expect(cardEl()?.querySelector('.ctc-state-phrase')?.textContent).toBe('writes on');
  });

  test('writes off on the card says whose doing it was', async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, { ...STATES.sandbox }),
          rowOf(KINDS.standin, {
            ...STATES.sandbox,
            active: true,
            available_now: false,
            reason: 'already_active',
          }),
          rowOf(KINDS.va),
        ],
      })
    );
    expect(cardEl()?.querySelector('.ctc-state-phrase')?.textContent).toBe('writes off');
    expect(cardEl()?.querySelector('.ctc-state-note')?.textContent).toBe('— you turned them off');
    expect(cardVerb()?.textContent).toBe('Turn writes on');
  });
});

/* ---- names and descriptors ----------------------------------------------- */

describe('names', () => {
  test('every machine is named by what it is, not how it is wired', async () => {
    await bootOpen();
    expect(rowEl('live')?.querySelector('.ctc-name')?.textContent).toBe('Real machine');
    expect(rowEl('va')?.querySelector('.ctc-name')?.textContent).toBe('Simulator');
    // No raw server label reaches a resting surface.
    expect(popEl()?.textContent).not.toContain('LIVE MACHINE');
    expect(popEl()?.textContent).not.toContain('virtual accelerator');
  });

  test("the deployment's configured display_name wins over the default", async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, { ...STATES.sandbox, display_name: 'ALS storage ring' }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
          rowOf(KINDS.va),
        ],
      })
    );
    expect(rowEl('live')?.querySelector('.ctc-name')?.textContent).toBe('ALS storage ring');
  });

  test('the live machine carries the one hazard descriptor', async () => {
    await bootOpen();
    const desc = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-desc'));
    expect(desc.textContent).toBe('Writes move hardware');
    expect(desc.dataset.tone).toBe('hazard');
    // Nothing-moves machines never carry the tone.
    const va = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-desc'));
    expect(va.textContent).toBe('Physics model · nothing moves');
    expect(va.dataset.tone).toBeUndefined();
  });

  test('a live machine nothing has authored reads not set up, without the hazard tone', async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, {
            ...STATES['read-only'],
            available_now: false,
            reason: 'gateways_missing',
          }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
        ],
      })
    );
    const desc = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-desc'));
    expect(desc.textContent).toBe('Not set up yet');
    expect(desc.dataset.tone).toBeUndefined();
  });
});

/* ---- reachability -------------------------------------------------------- */

describe('reachability', () => {
  test('a machine that answers says nothing about it', async () => {
    await bootOpen();
    expect(document.querySelectorAll('.ctc-reach-word')).toHaveLength(0);
  });

  test('down renders as not answering, with the measured detail on the title', async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, {
            ...STATES.sandbox,
            reachability: { state: 'down', role: 'read_only', age_s: 12 },
          }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
          rowOf(KINDS.va, { reachability: { state: 'stale', role: null, age_s: 90 } }),
        ],
      })
    );
    const down = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-reach-word'));
    expect(down.textContent).toBe('not answering');
    expect(down.dataset.state).toBe('down');
    expect(down.title).toContain('down · 12 s');
    expect(down.title).toContain('read_only endpoint');

    const stale = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-reach-word'));
    expect(stale.textContent).toBe('may be stale');
    expect(stale.title).toContain('last probe older than the prober interval');
  });
});

/* ---- the banner ---------------------------------------------------------- */

describe('the banner', () => {
  test('absent in the plain case — absence is the good news', async () => {
    await bootOpen();
    expect(bannerEl()).toBeNull();
  });

  test('an unavailable store speaks first, in the error tone', async () => {
    await bootOpen(viewOf({ store_available: false, enforceable: false }));
    expect(bannerEl()?.textContent).toContain('cannot be recorded');
    expect(bannerEl()?.dataset.tone).toBe('error');
  });

  test('a session nothing enforces warns that changes will not reach the agent', async () => {
    await bootOpen(viewOf({ enforceable: false, enforceable_reason: 'no_session_record' }));
    expect(bannerEl()?.textContent).toBe('Changes here will not reach the agent yet.');
    expect(bannerEl()?.dataset.tone).toBe('warn');
  });

  test('a readonly run is named deployment-wide', async () => {
    const readonly = { ceiling_writes: true, posture: 'writes', effective: false };
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, readonly),
          rowOf(KINDS.standin, {
            ...readonly,
            active: true,
            available_now: false,
            reason: 'already_active',
          }),
          rowOf(KINDS.va, readonly),
        ],
      })
    );
    expect(bannerEl()?.textContent).toBe('The whole deployment is running read-only.');
    expect(bannerEl()?.dataset.tone).toBe('warn');
  });
});

/* ---- locks --------------------------------------------------------------- */

describe('the verb locks, one reason at a time', () => {
  /** @param {any} view @param {string} target */
  async function lockOn(view, target) {
    await bootOpen(view);
    const verb = verbEl(target);
    return { verb, reason: verb?.title ?? '' };
  }

  test('an unrecordable store outranks everything else', async () => {
    const { verb, reason } = await lockOn(
      viewOf({ store_available: false, enforceable: false }),
      'va'
    );
    expect(verb?.disabled).toBe(true);
    expect(reason).toBe('changes cannot be recorded right now');
  });

  test('not enforceable, when the store is there but nothing reads it', async () => {
    const { reason } = await lockOn(
      viewOf({ enforceable: false, enforceable_reason: 'no_session_record' }),
      'va'
    );
    expect(reason).toBe('changes here would not reach the agent');
  });

  test('a readonly run: the ceiling is up, nothing is narrowed, writes are still off', async () => {
    const readonly = { ceiling_writes: true, posture: 'writes', effective: false };
    const { reason } = await lockOn(
      viewOf({ targets: [rowOf(KINDS.live, readonly), rowOf(KINDS.va, readonly)] }),
      'va'
    );
    expect(reason).toBe('the whole deployment is running read-only');
  });

  test('a ceiling that never armed the target reads as the deployment holding it', async () => {
    const { verb, reason } = await lockOn(
      viewOf({ targets: [rowOf(KINDS.va, { ...STATES['read-only'] })] }),
      'va'
    );
    expect(reason).toBe('kept read-only by the deployment');
    expect(verb?.disabled).toBe(true);
    // The pill says locked, not off: nothing this session did holds it.
    expect(pillEl('va')?.textContent).toBe('locked');
    expect(pillEl('va')?.title).toBe('kept read-only by the deployment');
  });

  test('no read-only endpoint: narrowing would select a role nothing configures', async () => {
    const { reason } = await lockOn(
      viewOf({ targets: [rowOf(KINDS.va, { narrowing_refusal: 'selected_role_missing' })] }),
      'va'
    );
    expect(reason).toBe('no read-only endpoint configured');
  });

  test('an unlocked verb stays live and names its outcome', async () => {
    await bootOpen();
    const verb = /** @type {HTMLButtonElement} */ (verbEl('va'));
    expect(verb.disabled).toBe(false);
    expect(verb.textContent).toBe('Turn writes off');
    expect(verb.dataset.direction).toBe('off');
    expect(pillEl('va')?.textContent).toBe('writes on');
  });
});

/* ---- write-state gestures ------------------------------------------------ */

describe('turning writes off and on', () => {
  test('turning writes off applies on click, with no confirm', async () => {
    await bootOpen();
    postAnswers.push({ ok: true, body: { entry: { va: 'sandbox' }, skipped: [] } });
    verbEl('va')?.click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(1);
    expect(posts()[0].url).toContain('/api/terminal/posture');
    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'va', posture: 'sandbox' });
  });

  test('turning writes on raises a confirm and POSTs nothing until it is confirmed', async () => {
    await bootOpen();
    verbEl('live')?.click();
    await flush();

    expect(confirmTitle()).toBe('Turn writes on for Real machine?');
    expect(posts()).toHaveLength(0);
    // The facility's own machine, and only it, carries the hardware notice.
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Real machine — writes move hardware.'
    );
    expect(confirmBtn()?.dataset.live).toBe('true');
    expect(confirmBtn()?.textContent).toBe('Turn writes on');
  });

  test('the confirm body states scope and endpoint, and claims nothing about process', async () => {
    // Whether a write prompts for approval, and what limits apply, are
    // deployment configuration the browser cannot see — the dialog must not
    // state either as fact.
    await bootOpen();
    verbEl('live')?.click();
    await flush();

    const body = inConfirm('.posture-modal-body').textContent ?? '';
    expect(body).toContain('For your session · als-gw.lbl.gov:5064.');
    expect(body).toContain('Takes effect at the next write — nothing restarts.');
    expect(body).not.toMatch(/approval|asks you|limits/i);
    // The title names the machine; the body's own sentences do not repeat it.
    // (The hardware notice below them is the one deliberate exception.)
    const paragraphs = [...(confirmEl()?.querySelectorAll('.posture-modal-body p') ?? [])]
      .map((n) => n.textContent)
      .join(' ');
    expect(paragraphs).not.toContain('Real machine');
  });

  test('the simulator turns on without the hardware notice', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.sandbox })] }));
    verbEl('va')?.click();
    await flush();
    expect(confirmTitle()).toBe('Turn writes on for Simulator?');
    expect(confirmEl()?.querySelector('.posture-modal-live')).toBeNull();
  });

  test('cancelling the confirm changes nothing and leaves the popover open', async () => {
    await bootOpen();
    verbEl('live')?.click();
    await flush();
    inConfirm('.posture-modal-cancel').click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(0);
    expect(isOpen()).toBe(true);
  });

  test('confirming POSTs, keeps the popover open, and re-renders in place', async () => {
    await bootOpen();
    verbEl('live')?.click();
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
    expect(pillEl('live')?.textContent).toBe('writes on');
  });

  test('a refused narrowing keeps the server sentence on the row', async () => {
    await bootOpen();
    postAnswers.push({
      ok: false,
      status: 409,
      body: { detail: { error: 'execution_in_flight', message: 'An execution is running.' } },
    });
    verbEl('va')?.click();
    await flush();

    expect(outcomes('va')).toContain('✗ An execution is running.');
  });

  test('a refused turn-on keeps the dialog up carrying the reason', async () => {
    await bootOpen();
    verbEl('live')?.click();
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

describe('Turn all writes off', () => {
  test('POSTs the all-targets narrowing and renders what the store skipped', async () => {
    await bootOpen();
    postAnswers.push({
      ok: true,
      body: { entry: {}, skipped: [{ target: 'va', reason: 'selected_role_missing' }] },
    });
    /** @type {HTMLButtonElement} */ (document.querySelector('.ctc-all-off')).click();
    await flush();

    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'all', posture: 'sandbox' });
    // The skip reason is the store's machine code, rendered as its phrase.
    expect(outcomes('va')).toContain('✗ no endpoint for role');
  });

  test('it is disabled when there is nothing left to turn off', async () => {
    await bootOpen(
      viewOf({
        targets: [rowOf(KINDS.live, { ...STATES.sandbox }), rowOf(KINDS.va, { ...STATES.sandbox })],
      })
    );
    expect(
      /** @type {HTMLButtonElement} */ (document.querySelector('.ctc-all-off')).disabled
    ).toBe(true);
  });

  test('the foot bounds the popover to the session, once', async () => {
    await bootOpen();
    expect(document.querySelector('.ctc-all-off')?.textContent).toBe('Turn all writes off');
    expect(document.querySelector('.ctc-foot-note')?.textContent).toBe('Your session only');
  });
});

/* ---- switching ----------------------------------------------------------- */

describe('switching', () => {
  test('Switch to confirms, POSTs, and shows switching… on that row', async () => {
    await bootOpen();
    switchEl('va')?.click();
    await flush();

    expect(confirmTitle()).toBe('Switch to Simulator?');
    expect(confirmBtn()?.textContent).toBe('Switch');

    // The first line states the consequence; the second names the write state
    // the session ARRIVES in — it is per machine and does not travel.
    const body = inConfirm('.posture-modal-body').textContent ?? '';
    expect(body).toContain('All control reads and writes go to 127.0.0.1:10064.');
    expect(body).toContain('Writes are on there for your session.');
    expect(body).not.toMatch(/approval|asks you/i);

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

  test('arriving with writes off is said as the nothing-moves sentence', async () => {
    await bootOpen();
    switchEl('live')?.click();
    await flush();
    const body = inConfirm('.posture-modal-body').textContent ?? '';
    expect(body).toContain(
      'Writes are off there for your session — nothing moves until you turn them on.'
    );
    // Off means off: no hardware notice until writes are actually on there.
    expect(confirmEl()?.querySelector('.posture-modal-live')).toBeNull();
  });

  test('switching onto the live machine with writes on carries the hardware notice', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.live)] }));
    switchEl('live')?.click();
    await flush();
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Real machine — writes move hardware.'
    );
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

  test('an outcome that names no target renders on no surface', async () => {
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

  test('a code the phrase map does not know renders verbatim', async () => {
    // Failing informative: the raw code stands in for its own tooltip.
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, { ...STATES.sandbox, available_now: false, reason: 'unreachable' }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
        ],
      })
    );
    expect(switchEl('live')).toBeNull();
    const reason = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-reason'));
    expect(reason.textContent).toBe('unreachable');
    expect(reason.title).toBe('unreachable');
  });

  test('an unauthored target reads as a quiet phrase, with the sentence on the title', async () => {
    // The three not-set-up codes are one phrase: on a stock deployment the
    // live target is deliberately unauthored (authoring it is the go-live
    // edit), and a raw `gateways_missing` would read as a fault.
    for (const code of ['connector_block_missing', 'gateways_missing', 'probe_channel_missing']) {
      await bootOpen(
        viewOf({
          targets: [
            rowOf(KINDS.live, {
              ...STATES.sandbox,
              available_now: false,
              reason: code,
              reason_detail: 'The epics block configures no gateways.',
            }),
            rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
          ],
        })
      );
      const reason = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-reason'));
      expect(reason.textContent).toBe('not set up');
      expect(reason.title).toBe('The epics block configures no gateways.');
      popoverModule.teardownControlTargetPopover();
      chipModule.teardownControlTargetChip();
    }
  });

  test('every published refusal code has an operator phrase', async () => {
    const phrases = {
      target_unresolvable: 'unavailable',
      limits_posture: 'needs strict limits',
      operator_ack_missing: 'needs gateway ack',
      archive_belongs_to_standin: 'archive conflict',
      invented_history: 'no archive',
      standin_not_deployed: 'stand-in not deployed',
      selected_role_missing: 'no endpoint for role',
    };
    for (const [code, phrase] of Object.entries(phrases)) {
      await bootOpen(
        viewOf({
          targets: [
            rowOf(KINDS.live, { ...STATES.sandbox, available_now: false, reason: code }),
            rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
          ],
        })
      );
      expect(rowEl('live')?.querySelector('.ctc-reason')?.textContent).toBe(phrase);
      popoverModule.teardownControlTargetPopover();
      chipModule.teardownControlTargetChip();
    }
  });

  test('a store that cannot be resolved explains the missing Switch', async () => {
    await bootOpen(
      viewOf({
        store_available: false,
        targets: [
          rowOf(KINDS.live, { ...STATES.sandbox, reason: null }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
        ],
      })
    );
    expect(rowEl('live')?.querySelector('.ctc-reason')?.textContent).toBe('store unavailable');
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
    verbEl('live')?.click();
    await flush();

    inConfirm('.posture-modal-body').click();
    expect(isOpen()).toBe(true);
    expect(confirmEl()).not.toBeNull();
  });

  test('Escape dismisses the confirm first and the popover only after', async () => {
    await bootOpen();
    verbEl('live')?.click();
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

/* ---- chat sessions ------------------------------------------------------- */

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
    expect(bannerEl()).toBeNull();
  });

  test('keeps its verbs live — the write state is keyed on the session, not the topology', async () => {
    await bootOpen(chatView());
    expect(verbEl('live')?.disabled).toBe(false);

    postAnswers.push({ ok: true, body: { entry: { live: 'sandbox' }, skipped: [] } });
    verbEl('live')?.click();
    await flush();
    expect(posts()[0].body).toEqual({ session_id: SESSION, target: 'live', posture: 'sandbox' });
  });
});

/* ---- realign and the execution in flight ---------------------------------- */

describe('a narrowing that has not reached the agent yet', () => {
  test('the card says it takes effect after the run', async () => {
    await bootOpen(
      viewOf({ execution_in_flight: true, last_posture_realign: { state: 'pending' } })
    );
    expect(outcomes('standin')).toContain('takes effect when the running execution finishes');
    expect(outcomes('va')).toHaveLength(0);
  });
});

/* ---- no process claims, anywhere ------------------------------------------ */

describe('no process claims', () => {
  test('no resting surface and neither confirm states approval behavior', async () => {
    // Whether a write prompts for approval is deployment configuration the
    // browser cannot see; a blanket sentence about it is false for some
    // configurations. The popover states consequence only.
    await bootOpen();
    const forbidden = /approval|asks you|ask for permission|still apply/i;
    expect(popEl()?.textContent ?? '').not.toMatch(forbidden);

    verbEl('live')?.click();
    await flush();
    expect(confirmEl()?.textContent ?? '').not.toMatch(forbidden);
    pressEscape();

    switchEl('va')?.click();
    await flush();
    expect(confirmEl()?.textContent ?? '').not.toMatch(forbidden);
  });
});

/* ---- one DOM for both densities ------------------------------------------- */

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
  });
});

/* ---- request plumbing ----------------------------------------------------- */

describe('request plumbing', () => {
  test('every request goes through the per-user mount prefix', async () => {
    /** @type {any} */ (window).__OSPREY_PREFIX__ = '/u/alice';
    await bootOpen();
    postAnswers.push({ ok: true, body: { entry: {}, skipped: [] } });
    verbEl('va')?.click();
    await flush();

    expect(fetchCalls.length).toBeGreaterThan(0);
    for (const call of fetchCalls) expect(call.url.startsWith('/u/alice/')).toBe(true);
  });
});
