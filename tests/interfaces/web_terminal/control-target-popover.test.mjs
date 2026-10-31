// @ts-check
/**
 * Unit tests for the control-target popover (control-target-popover.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-popover.test.mjs
 *
 * The popover is the panel behind the sidebar's chip: a card for the machine the
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
 * - the writes switch is this login's switch (C-GET): a switch that is off
 *   and cannot be turned on is disabled with the reason on its `title`; under
 *   every lock a switch that is on keeps its off control and shows the
 *   verdict's `write_phrase`;
 * - turning writes on raises a confirm and turning them off does NOT: only a
 *   gesture that can end with a write landing somewhere new asks;
 * - every switch gesture posts C-POST `{session_id, target, state, epoch}`
 *   carrying the `switch_epoch` the chip read; a stamped 409 is retried once
 *   with its epoch, any other 409 re-reads and wants a new click, and a 503 on
 *   a stop says the stop did not take effect while the machine still reads on;
 * - a confirmed change leaves the popover OPEN and re-renders in place, so an
 *   operator changing several machines never reopens it;
 * - the popover is a body-level design-system popover (popover.js): beside
 *   the sidebar it opens 8 px right of the sidebar's edge, under a chip on a
 *   page bar it hangs below the chip, and a press outside, Escape, a resize
 *   or a change of sidebar width closes it;
 * - Escape dismisses an open confirm BEFORE it closes the popover, and a
 *   press inside the confirm is not a press outside;
 * - Switch to confirms, POSTs, shows `switching…` on that row, and renders the
 *   outcome the route publishes for the `request_id` — success, refusal, and
 *   the expiry a dead controls server never answers;
 * - a refusal code renders as its operator phrase, never raw; a code the map
 *   does not know renders verbatim;
 * - rows the route refuses offer no Switch and say why in its place, while
 *   their write switches stay live — writing is a separate question;
 * - `Turn all writes off` POSTs `all`/`off`, enabled exactly when `any_on`;
 * - a waived turn-on shows the turn-on sentence once in the status line;
 * - ONE DOM for both `ui_mode`s: the same markup under `simple` and `expert`.
 *
 * Seams: `fetch` is stubbed the way the other suites here stub it, and
 * `EventSource` is stubbed as a class, since happy-dom has none. Both modules hold module-private state
 * with no reset API beyond their teardowns, so each test gets fresh instances
 * via vi.resetModules() + dynamic import — same pattern as
 * control-target-chip.test.mjs.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { stubEventSource } from '../_support/event-source.mjs';

const SESSION = 'aaaaaaaa-1111-2222-3333-444444444444';
const CHIP = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js';
const POPOVER = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-popover.js';

const FACTS = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-facts.js';
const { STOP_NOT_TAKEN_SENTENCE: STOP_NOT_TAKEN, TURN_ON_SENTENCE: TURN_ON } = await import(FACTS);
/** The outcome freshness window lives with the rows that show the outcome. */
const { OUTCOME_MAX_AGE_S } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-rows.js');
/** What a turn-on a stop overtook reads on its row. */
const STOPPED_SINCE =
  'Writes were turned off after this turn-on began — nothing was turned on. Click again to turn them on.';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js')} */
let chipModule;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-popover.js')} */
let popoverModule;

/* ---- roster fixtures ---------------------------------------------------- */

/** The three machines a switch-capable deployment configures, as the route publishes them. */
const KINDS = {
  live: {
    target: 'live',
    label: 'Real machine — writes move hardware',
    short_label: 'LIVE',
    machine_kind: 'real',
    display_name: '',
    endpoint: 'gw.example.org:5064',
    real_machine: true,
  },
  standin: {
    target: 'standin',
    label: 'Rehearsal — nothing moves',
    short_label: 'STAND-IN',
    machine_kind: 'rehearsal',
    display_name: '',
    endpoint: '127.0.0.1:10090',
    real_machine: true,
  },
  va: {
    target: 'va',
    label: 'Simulator — nothing moves',
    short_label: 'VIRTUAL',
    machine_kind: 'simulator',
    display_name: '',
    endpoint: '127.0.0.1:10064',
    real_machine: false,
  },
};

/** The verdict's sentence for a run started read-only (C-PHRASE). */
const READONLY_PHRASE =
  'Writes are refused because the deployment is running read-only; the chip cannot lift this — ask the deployment admin.';
/** The verdict's sentence for a login whose tier never writes. */
const TIER_NEVER_PHRASE =
  "Writes are refused because this login's write tier does not allow them; ask the deployment admin.";

/**
 * The states as C-GET's columns express them. `off` and `locked` are both
 * "no", and which "no" it is decides what the switch offers. Each carries a
 * legacy `posture` that DISAGREES with it, so a derivation that still read
 * `posture` would fail here.
 */
const STATES = {
  on: {
    switch: 'on',
    write_allowed: true,
    write_reason: null,
    write_phrase: null,
    posture: 'sandbox',
  },
  off: {
    switch: 'off',
    write_allowed: false,
    write_reason: 'switch_off',
    write_phrase: 'Writes are off; turn it on from the chip in the web terminal.',
    posture: 'writes',
  },
  /** On, and still refused: the run was started read-only. */
  locked: {
    switch: 'on',
    write_allowed: false,
    write_reason: 'readonly_run',
    write_phrase: READONLY_PHRASE,
    posture: 'writes',
  },
  /** Off, and a turn-on could not lift it: the login's tier never writes. */
  never: {
    switch: 'off',
    write_allowed: false,
    write_reason: 'tier_never',
    write_phrase: TIER_NEVER_PHRASE,
    posture: 'writes',
  },
  /** A record nobody can read. */
  unreadable: {
    switch: null,
    write_allowed: false,
    write_reason: 'record_unavailable',
    write_phrase:
      'Writes are refused because the switch record cannot be read; ask the deployment admin.',
    posture: 'writes',
  },
};

/** @param {object} kind @param {object} [o] */
const rowOf = (kind, o = {}) => ({
  ...kind,
  ...STATES.on,
  active: false,
  is_baseline: false,
  available_now: true,
  reason: null,
  reason_detail: null,
  reachability: {
    state: 'reached',
    role: 'write_access',
    probed_at: '2026-08-30T12:00:00+00:00',
    age_s: 3,
    role_detail: {},
  },
  ...o,
});

/** The epoch the default roster's record carries. */
// A decimal string, as the route sends it: nanoseconds with low digits that a
// JSON number would round away.
const EPOCH = '1793181592449148123';

/** The roster a plain switch-capable deployment answers with. */
const viewOf = (o = {}) => ({
  // The chip's read carries no session id, so the route echoes none back.
  session_id: null,
  control_target: 'standin',
  generation: 3,
  // This terminal owns the control context, which is what makes its target
  // switches writable at all; `self: false` is a terminal that FOLLOWS another.
  owner: { kind: 'web_terminal', pid: 1000, port: 8080, self: true },
  servers: [],
  store_available: true,
  readonly_run: false,
  switch_epoch: EPOCH,
  any_on: true,
  locks: { readonly_run: false, tier: 'readwrite' },
  execution_in_flight: [],
  last_switch: null,
  last_posture_realign: null,
  targets: [
    rowOf(KINDS.live, { ...STATES.off }),
    rowOf(KINDS.standin, {
      active: true,
      is_baseline: true,
      available_now: false,
      reason: 'already_active',
    }),
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

/**
 * The sidebar's control-system block the chip mounts into (index.html's
 * shape), and somewhere outside it to click.
 */
function mountFixture() {
  document.body.innerHTML = `
    <nav class="sidebar" id="sidebar" aria-label="Panels">
      <div class="sidebar-target" id="sidebar-target"></div>
      <button class="osprey-search" id="command-palette-btn" type="button"></button>
    </nav>
    <div id="outside"></div>`;
}

/** The sidebar's control-system block, the host app.js hands the chip. */
const targetHost = () => /** @type {HTMLElement} */ (document.getElementById('sidebar-target'));

/** Drain the microtask/timer queue the async handlers chain through. */
async function flush() {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Boot the chip and its popover against a roster.
 * @param {any} [payload]
 * @param {HTMLElement} [host] where the chip mounts; the sidebar's block by default
 */
async function boot(payload, host = targetHost()) {
  served = payload ?? viewOf();
  chipModule.initControlTargetChip({ host });
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
const popEl = () =>
  /** @type {HTMLElement|null} */ (document.getElementById('control-target-popover'));
const isOpen = () => Boolean(popEl() && !popEl()?.hidden);
/** A pointer press, the event popover.js dismisses on. @param {Element} target */
const pointerDown = (target) =>
  target.dispatchEvent(new Event('pointerdown', { bubbles: true, cancelable: true }));
const cardEl = () => /** @type {HTMLElement|null} */ (document.querySelector('.ctc-card'));
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
/** The surface's writes switch — `aria-checked` is the readout. @param {string} target */
const toggleEl = (target) => {
  const surface = surfaceEl(target);
  return /** @type {HTMLButtonElement|null} */ (surface?.querySelector('.ctc-toggle') ?? null);
};
/** The word beside a surface's switch: on / off / locked. @param {string} target */
const switchWord = (target) =>
  /** @type {HTMLElement|null} */ (surfaceEl(target)?.querySelector('.ctc-switch-state') ?? null);
/** @param {string} target */
const switchEl = (target) =>
  /** @type {HTMLButtonElement|null} */ (rowEl(target)?.querySelector('.ctc-switch') ?? null);
/** The refusal word standing where a row's Switch would be. @param {string} target */
const reasonEl = (target) =>
  /** @type {HTMLElement|null} */ (rowEl(target)?.querySelector('.ctc-reason') ?? null);
/** @param {string} target */
const outcomes = (target) =>
  [...(surfaceEl(target)?.querySelectorAll('.ctc-outcome') ?? [])].map((n) => n.textContent);
/** The verdict sentence a surface shows at rest, or null. @param {string} target */
const phraseEl = (target) =>
  /** @type {HTMLElement|null} */ (surfaceEl(target)?.querySelector('.ctc-write-phrase') ?? null);
/** The popover's status line, or null. */
const statusEl = () => /** @type {HTMLElement|null} */ (document.querySelector('.ctc-status'));
const allOffEl = () => /** @type {HTMLButtonElement} */ (document.querySelector('.ctc-all-off'));
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

const pressEscape = () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

beforeEach(async () => {
  vi.resetModules();
  served = viewOf();
  stubFetch();
  stubEventSource({ open: true });
  mountFixture();
  // The don't-ask-again waivers persist in localStorage; no test inherits
  // another's.
  localStorage.clear();
  document.documentElement.removeAttribute('data-ui-mode');
  delete (/** @type {any} */ (window).__OSPREY_PREFIX__);
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
  test('mounts at <body> level as a design-system popover dialog, closed', async () => {
    await boot();
    const pop = /** @type {HTMLElement} */ (popEl());
    expect(pop.parentElement).toBe(document.body);
    expect(pop.classList.contains('osprey-popover')).toBe(true);
    expect(pop.getAttribute('role')).toBe('dialog');
    expect(pop.getAttribute('aria-label')).toBe('Control target');
    expect(isOpen()).toBe(false);
    expect(chipEl()?.getAttribute('aria-controls')).toBe('control-target-popover');
  });

  test('a second init reuses the same popover', async () => {
    await boot();
    popoverModule.initControlTargetPopover();
    expect(document.querySelectorAll('#control-target-popover')).toHaveLength(1);
  });

  test('init no-ops on a page with no chip', () => {
    document.body.innerHTML = '<div id="elsewhere"></div>';
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

  test('a press outside closes it and mirrors the state onto the chip', async () => {
    await bootOpen();
    pointerDown(/** @type {HTMLElement} */ (document.getElementById('outside')));
    expect(isOpen()).toBe(false);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('false');
  });

  test('a press inside the popover does not close it', async () => {
    await bootOpen();
    pointerDown(/** @type {HTMLElement} */ (popEl()?.querySelector('.osprey-popover-title')));
    expect(isOpen()).toBe(true);
  });

  test('opening puts focus on the first writes switch', async () => {
    await bootOpen();
    expect(document.activeElement).toBe(toggleEl('standin'));
  });

  test('a re-render keeps focus on the same control', async () => {
    await bootOpen();
    toggleEl('live')?.focus();
    const before = toggleEl('live');

    await chipModule.refetch();
    await flush();

    expect(toggleEl('live')).not.toBe(before);
    expect(document.activeElement).toBe(toggleEl('live'));
  });

  test('Escape closes it and returns focus to the chip', async () => {
    await bootOpen();
    pressEscape();
    expect(isOpen()).toBe(false);
    expect(document.activeElement).toBe(chipEl());
  });
});

describe('beside the sidebar', () => {
  /**
   * Give the sidebar a right edge and the chip a top, as layout would.
   * @param {number} chipTop
   */
  function layout(chipTop) {
    const sidebar = /** @type {HTMLElement} */ (document.getElementById('sidebar'));
    vi.spyOn(sidebar, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({ left: 0, right: 220, top: 0, bottom: 700, width: 220, height: 700 })
    );
    const chip = /** @type {HTMLElement} */ (chipEl());
    vi.spyOn(chip, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({
        left: 8,
        right: 212,
        top: chipTop,
        bottom: chipTop + 40,
        width: 204,
        height: 40,
      })
    );
  }

  test('opens 8 px right of the sidebar, level with the chip', async () => {
    await boot();
    layout(64);
    chipEl()?.click();
    await flush();
    expect(isOpen()).toBe(true);
    expect(popEl()?.style.left).toBe('228px');
    expect(popEl()?.style.top).toBe('64px');
  });

  test('keeps its top inside the window', async () => {
    await boot();
    layout(window.innerHeight + 500);
    chipEl()?.click();
    await flush();
    // happy-dom lays nothing out, so the popover is 0 px tall: the clamp is
    // the window's height less the 8 px margin.
    expect(popEl()?.style.top).toBe(`${window.innerHeight - 8}px`);

    chipEl()?.click();
    await flush();
    layout(-30);
    chipEl()?.click();
    await flush();
    expect(popEl()?.style.top).toBe('8px');
  });

  test('a window resize closes it', async () => {
    await bootOpen();
    window.dispatchEvent(new Event('resize'));
    expect(isOpen()).toBe(false);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('false');
  });

  test('a change of sidebar width closes it', async () => {
    await bootOpen();
    document.dispatchEvent(
      new CustomEvent('osprey-collapsible-change', {
        detail: { attribute: 'data-sidebar', state: 'collapsed' },
      })
    );
    expect(isOpen()).toBe(false);
    expect(chipEl()?.getAttribute('aria-expanded')).toBe('false');
  });

  test('under a chip outside the sidebar it hangs below the chip, right-aligned', async () => {
    const bar = document.createElement('div');
    document.body.appendChild(bar);
    await boot(undefined, bar);
    const chip = /** @type {HTMLElement} */ (chipEl());
    vi.spyOn(chip, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({ left: 600, right: 760, top: 1, bottom: 29, width: 160, height: 28 })
    );
    chip.click();
    await flush();
    expect(isOpen()).toBe(true);
    expect(popEl()?.style.left).toBe('');
    expect(popEl()?.style.top).toBe('33px');
    expect(popEl()?.style.right).toBe(`${window.innerWidth - 760}px`);
  });
});

/* ---- the design-system primitives ---------------------------------------- */

describe('built on the design-system primitives', () => {
  test('titles, separators, rows, dots, switches and buttons', async () => {
    await bootOpen();
    const pop = /** @type {HTMLElement} */ (popEl());
    const titles = [...pop.querySelectorAll(':scope > .osprey-popover-title')].map(
      (n) => n.textContent
    );
    expect(titles).toEqual(['The agent is on', 'Other machines']);
    expect(pop.querySelectorAll(':scope > .osprey-popover-sep').length).toBeGreaterThan(0);

    const row = /** @type {HTMLElement} */ (rowEl('live'));
    expect(row.classList.contains('osprey-popover-item')).toBe(true);
    expect(row.querySelector('.ctc-name-line > .osprey-dot')).not.toBeNull();
    expect(cardEl()?.querySelector('.ctc-card-title > .osprey-dot')).not.toBeNull();
    expect(toggleEl('live')?.querySelector('.osprey-switch')).not.toBeNull();
    expect(allOffEl().classList.contains('osprey-btn')).toBe(true);
  });

  test('Switch to is the primary button', async () => {
    await bootOpen();
    const swap = document.querySelector('.ctc-switch');
    expect(swap?.classList.contains('osprey-btn')).toBe(true);
    expect(swap?.classList.contains('primary')).toBe(true);
  });
});

/* ---- the card ------------------------------------------------------------ */

describe('the card', () => {
  test('the machine the agent stands on is the card, not a row', async () => {
    await bootOpen();
    const card = /** @type {HTMLElement} */ (cardEl());
    expect(card.dataset.target).toBe('standin');
    expect(card.dataset.targetKind).toBe('rehearsal');
    expect(card.dataset.state).toBe('on');
    // `data-real` is retired: the kind alone keys the colours, and the
    // stand-in is not the real machine.
    expect(card.dataset.real).toBeUndefined();
    expect(rowEl('standin')).toBeNull();
    // The card sits under the popover's own title line, not an eyebrow of its own.
    expect(card.previousElementSibling?.className).toBe('osprey-popover-title');
    expect(card.previousElementSibling?.textContent).toBe('The agent is on');
  });

  test('the card names the machine by consequence and keeps the label on hover', async () => {
    await bootOpen();
    const title = /** @type {HTMLElement} */ (cardEl()?.querySelector('.ctc-card-title'));
    expect(title.querySelector('.ctc-name')?.textContent).toBe('Rehearsal');
    // The implementation vocabulary lives on the ⓘ tooltip, never at rest.
    const tip = /** @type {HTMLElement} */ (cardEl()?.querySelector('.ctc-tip'));
    expect(tip.textContent).toContain('Rehearsal — nothing moves');
    expect(tip.textContent).toContain('127.0.0.1:10090');
    expect(tip.querySelector('.ctc-tip-what')?.textContent).toBe(
      "Copy of the real machine's controls · nothing moves"
    );
    const toggle = /** @type {HTMLButtonElement} */ (toggleEl('standin'));
    expect(toggle.getAttribute('role')).toBe('switch');
    expect(toggle.getAttribute('aria-checked')).toBe('true');
    expect(toggle.getAttribute('aria-label')).toBe('writes on for Rehearsal');
  });

  test('writes off on the card reads off, and the same switch offers the way back', async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, { ...STATES.off }),
          rowOf(KINDS.standin, {
            ...STATES.off,
            active: true,
            available_now: false,
            reason: 'already_active',
          }),
          rowOf(KINDS.va),
        ],
      })
    );
    const toggle = /** @type {HTMLButtonElement} */ (toggleEl('standin'));
    expect(toggle.getAttribute('aria-checked')).toBe('false');
    expect(switchWord('standin')?.textContent).toBe('off');
    // The way back is the same widget, and its hover says it asks first.
    expect(toggle.title).toBe('Turn writes on for Rehearsal — asks first');
  });
});

/* ---- names and descriptors ----------------------------------------------- */

describe('names', () => {
  test('every machine is named by what it is, not how it is wired', async () => {
    await bootOpen();
    expect(rowEl('live')?.querySelector('.ctc-name')?.textContent).toBe('Real machine');
    expect(rowEl('va')?.querySelector('.ctc-name')?.textContent).toBe('Simulator');
    // No raw server label reaches a resting surface — the labels live only
    // inside the ⓘ tooltips.
    const atRest = /** @type {HTMLElement} */ (popEl()?.cloneNode(true));
    for (const tip of atRest.querySelectorAll('.ctc-tip')) tip.remove();
    expect(atRest.textContent).not.toContain('Real machine — writes move hardware');
    expect(atRest.textContent).not.toContain('Simulator — nothing moves');
  });

  test("the deployment's configured display_name wins over the default", async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, { ...STATES.off, display_name: 'Example storage ring' }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
          rowOf(KINDS.va),
        ],
      })
    );
    expect(rowEl('live')?.querySelector('.ctc-name')?.textContent).toBe('Example storage ring');
  });

  test('the live machine carries the one hazard descriptor, inside its tooltip', async () => {
    await bootOpen();
    const desc = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-tip-what'));
    expect(desc.textContent).toBe('Writes move hardware');
    expect(desc.dataset.tone).toBe('hazard');
    // Nothing-moves machines never carry the tone.
    const va = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-tip-what'));
    expect(va.textContent).toBe('Physics model · nothing moves');
    expect(va.dataset.tone).toBeUndefined();
  });

  test('the ⓘ is a keyboard-reachable button carrying both hidden facts', async () => {
    await bootOpen();
    const info = /** @type {HTMLButtonElement} */ (rowEl('va')?.querySelector('.ctc-info'));
    expect(info.getAttribute('aria-label')).toBe('About Simulator');
    const tip = /** @type {HTMLElement} */ (info.querySelector('.ctc-tip'));
    expect(tip.getAttribute('role')).toBe('tooltip');
    // Consequence first, then identity — one place for "what is this?".
    expect(tip.querySelector('.ctc-tip-what')?.textContent).toBe('Physics model · nothing moves');
    expect(tip.querySelector('.ctc-tip-ident')?.textContent).toContain('127.0.0.1:10064');
  });

  test('a live machine nothing has authored reads not set up, without the hazard tone', async () => {
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, {
            ...STATES.off,
            available_now: false,
            reason: 'gateways_missing',
          }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
        ],
      })
    );
    const desc = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-tip-what'));
    expect(desc.textContent).toBe('Not set up yet');
    expect(desc.dataset.tone).toBeUndefined();
  });

  test('a machine the switch cannot dial is still a machine, and reads as one', async () => {
    // Refused for its protocol, not for a key nobody filled in: the deployment
    // runs on that connector and the descriptor keeps saying what writing there
    // would do. "Not set up" would be the wrong sentence twice over.
    await bootOpen(
      viewOf({
        targets: [
          rowOf(KINDS.live, {
            ...STATES.off,
            available_now: false,
            reason: 'connector_not_switchable',
            reason_detail: "Target 'live' resolves to connector type 'doocs'.",
          }),
          rowOf(KINDS.va, { active: true, available_now: false, reason: 'already_active' }),
        ],
      })
    );
    const desc = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-tip-what'));
    expect(desc.textContent).toBe('Writes move hardware');
    expect(desc.dataset.tone).toBe('hazard');
    const reason = /** @type {HTMLElement} */ (rowEl('live')?.querySelector('.ctc-reason'));
    expect(reason.textContent).toBe('switching not supported');
    expect(reason.title).toBe("Target 'live' resolves to connector type 'doocs'.");
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
            ...STATES.off,
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
    await bootOpen(viewOf({ store_available: false }));
    expect(bannerEl()?.textContent).toContain('cannot be recorded');
    expect(bannerEl()?.dataset.tone).toBe('error');
  });

  test('a context another terminal holds names the port to go to', async () => {
    // The write routes answer 409 here, so the banner says so before a click
    // does — and names where the change can actually be made.
    await bootOpen(viewOf({ owner: { kind: 'web_terminal', pid: 77, port: 8123, self: false } }));
    expect(bannerEl()?.textContent).toBe(
      'Another terminal on port 8123 holds the control context.'
    );
    expect(bannerEl()?.dataset.tone).toBe('warn');
  });

  test('an owner with no port still says who holds it', async () => {
    await bootOpen(viewOf({ owner: { kind: 'web_terminal', pid: 77, port: null, self: false } }));
    expect(bannerEl()?.textContent).toBe('Another terminal holds the control context.');
  });

  test('nothing owning the context reads as unrecordable', async () => {
    // `owner: null` is "nothing owns it anywhere", which the write routes
    // answer 503 for — the same refusal an absent store gets.
    await bootOpen(viewOf({ owner: null }));
    expect(bannerEl()?.textContent).toContain('cannot be recorded');
    expect(bannerEl()?.dataset.tone).toBe('error');
  });

  test('a readonly run is named deployment-wide', async () => {
    await bootOpen(
      viewOf({
        readonly_run: true,
        locks: { readonly_run: true, tier: 'readwrite' },
        targets: [
          rowOf(KINDS.live, { ...STATES.locked }),
          rowOf(KINDS.standin, {
            ...STATES.locked,
            active: true,
            available_now: false,
            reason: 'already_active',
          }),
          rowOf(KINDS.va, { ...STATES.locked }),
        ],
      })
    );
    expect(bannerEl()?.textContent).toBe('The whole deployment is running read-only.');
    expect(bannerEl()?.dataset.tone).toBe('warn');
  });

  test('the published lock alone names the run, whatever each row says first', async () => {
    await bootOpen(
      viewOf({
        locks: { readonly_run: true, tier: 'readwrite' },
        targets: [
          rowOf(KINDS.live, { ...STATES.off }),
          rowOf(KINDS.standin, {
            ...STATES.locked,
            active: true,
            available_now: false,
            reason: 'already_active',
          }),
        ],
      })
    );
    expect(bannerEl()?.textContent).toBe('The whole deployment is running read-only.');
  });

  test('rows refused for their own reasons are not called a readonly run', async () => {
    await bootOpen(
      viewOf({
        targets: [rowOf(KINDS.live, { ...STATES.never }), rowOf(KINDS.va, { ...STATES.never })],
      })
    );
    expect(bannerEl()).toBeNull();
  });
});

/* ---- locks --------------------------------------------------------------- */

describe('the switch under a lock', () => {
  /** @param {any} view @param {string} target */
  async function lockOn(view, target) {
    await bootOpen(view);
    const toggle = toggleEl(target);
    return { toggle, reason: toggle?.title ?? '' };
  }

  test('an unrecordable store outranks everything else', async () => {
    const { toggle, reason } = await lockOn(
      viewOf({
        store_available: false,
        locks: { readonly_run: true, tier: 'never' },
        targets: [rowOf(KINDS.va, { ...STATES.never })],
      }),
      'va'
    );
    expect(toggle?.disabled).toBe(true);
    expect(reason).toBe('changes cannot be recorded right now');
  });

  test('a follower turns its login’s switch exactly as the holder does', async () => {
    // Only target switching is the holder's: the switch is the person's.
    await bootOpen(
      viewOf({
        owner: { kind: 'web_terminal', pid: 77, port: 8123, self: false },
        targets: [
          rowOf(KINDS.live, { ...STATES.off }),
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
          rowOf(KINDS.va),
        ],
      })
    );
    expect(toggleEl('live')?.disabled).toBe(false);
    expect(toggleEl('live')?.title).toBe('Turn writes on for Real machine — asks first');
    expect(toggleEl('va')?.disabled).toBe(false);
    // And no row offers a Switch it knows would be refused; the reason word
    // stands where the button was.
    expect(document.querySelectorAll('.ctc-switch')).toHaveLength(0);
    expect(reasonEl('va')?.textContent).toBe('another terminal');
  });

  test('a readonly run disables turning on, with the reason on hover', async () => {
    const { toggle, reason } = await lockOn(
      viewOf({
        locks: { readonly_run: true, tier: 'readwrite' },
        targets: [rowOf(KINDS.va, { ...STATES.off })],
      }),
      'va'
    );
    expect(toggle?.disabled).toBe(true);
    expect(reason).toBe('the whole deployment is running read-only');
    expect(switchWord('va')?.textContent).toBe('off');
  });

  test('a tier that never writes holds an off switch, and its sentence is shown', async () => {
    const { toggle, reason } = await lockOn(
      viewOf({ any_on: false, targets: [rowOf(KINDS.va, { ...STATES.never })] }),
      'va'
    );
    expect(toggle?.disabled).toBe(true);
    expect(reason).toBe(TIER_NEVER_PHRASE);
    expect(phraseEl('va')?.textContent).toBe(TIER_NEVER_PHRASE);
  });

  test('a machine that is on and refused reads locked, shows why, and keeps its off control', async () => {
    await bootOpen(
      viewOf({
        locks: { readonly_run: true, tier: 'readwrite' },
        targets: [
          rowOf(KINDS.standin, { active: true, available_now: false, reason: 'already_active' }),
          rowOf(KINDS.va, { ...STATES.locked }),
        ],
      })
    );
    const toggle = /** @type {HTMLButtonElement} */ (toggleEl('va'));
    expect(rowEl('va')?.dataset.state).toBe('locked');
    expect(switchWord('va')?.textContent).toBe('locked');
    expect(phraseEl('va')?.textContent).toBe(READONLY_PHRASE);
    expect(toggle.disabled).toBe(false);
    expect(toggle.getAttribute('aria-checked')).toBe('true');
    expect(toggle.title).toBe('Turn writes off for Simulator — applies now');

    toggle.click();
    await flush();
    expect(confirmEl()).toBeNull();
    expect(posts()[0].body).toEqual({ session_id: null, target: 'va', state: 'off', epoch: EPOCH });
  });

  test('a switch nobody can read reads locked and still offers the stop', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.unreadable })] }));
    const toggle = /** @type {HTMLButtonElement} */ (toggleEl('va'));
    expect(switchWord('va')?.textContent).toBe('locked');
    expect(phraseEl('va')?.textContent).toBe(STATES.unreadable.write_phrase);
    expect(toggle.disabled).toBe(false);
    toggle.click();
    await flush();
    expect(posts()[0].body).toMatchObject({ target: 'va', state: 'off' });
  });

  test('a machine that is merely off shows no sentence: the switch already says so', async () => {
    await bootOpen();
    expect(phraseEl('live')).toBeNull();
    expect(phraseEl('va')).toBeNull();
  });

  test('an unlocked switch stays live and names its outcome on hover', async () => {
    await bootOpen();
    const toggle = /** @type {HTMLButtonElement} */ (toggleEl('va'));
    expect(toggle.disabled).toBe(false);
    expect(toggle.getAttribute('aria-checked')).toBe('true');
    expect(toggle.title).toBe('Turn writes off for Simulator — applies now');
    expect(switchWord('va')?.textContent).toBe('on');
  });

  test('no surface carries data-real any more', async () => {
    await bootOpen();
    expect(popEl()?.querySelectorAll('[data-real]')).toHaveLength(0);
  });
});

/* ---- write-state gestures ------------------------------------------------ */

describe('turning writes off and on', () => {
  test('turning writes off applies on click, with no confirm', async () => {
    await bootOpen();
    postAnswers.push({ ok: true, body: { target: 'va', state: 'off', switch_epoch: EPOCH } });
    toggleEl('va')?.click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(1);
    expect(posts()[0].url).toContain('/api/terminal/posture');
    expect(posts()[0].body).toEqual({ session_id: null, target: 'va', state: 'off', epoch: EPOCH });
  });

  test('turning writes on raises a confirm and POSTs nothing until it is confirmed', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();

    expect(confirmTitle()).toBe('Turn writes on for Real machine?');
    expect(posts()).toHaveLength(0);
    // The facility's own machine, and only it, carries the hardware notice.
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Real machine — writes move hardware'
    );
    expect(confirmBtn()?.dataset.live).toBe('true');
    expect(confirmBtn()?.textContent).toBe('Turn writes on');
  });

  test('the confirm body states scope and claims nothing about process or endpoint', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();

    const body = inConfirm('.posture-modal-body').textContent ?? '';
    expect(body).toContain('your login');
    expect(body).not.toMatch(/approval|asks you|limits/i);
    expect(body).not.toMatch(/[\w.-]+:\d+/);
    const paragraphs = [...(confirmEl()?.querySelectorAll('.posture-modal-body p') ?? [])]
      .map((n) => n.textContent)
      .join(' ');
    expect(paragraphs).not.toContain('Real machine');
  });

  test('the simulator turns on without the hardware notice', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.off })] }));
    toggleEl('va')?.click();
    await flush();
    expect(confirmTitle()).toBe('Turn writes on for Simulator?');
    expect(confirmEl()?.querySelector('.posture-modal-live')).toBeNull();
  });

  test('cancelling the confirm changes nothing and leaves the popover open', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();
    inConfirm('.posture-modal-cancel').click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(0);
    expect(isOpen()).toBe(true);
  });

  test('while the turn-on confirm is up the switch parks mid-track, and springs back on cancel', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();

    expect(toggleEl('live')?.dataset.pending).toBe('on');
    expect(toggleEl('live')?.getAttribute('aria-checked')).toBe('false');

    inConfirm('.posture-modal-cancel').click();
    await flush();
    expect(toggleEl('live')?.dataset.pending).toBeUndefined();
    expect(toggleEl('live')?.getAttribute('aria-checked')).toBe('false');
  });

  test('confirming POSTs the epoch read, keeps the popover open, and re-renders in place', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();

    postAnswers.push({ ok: true, body: { target: 'live', state: 'on', switch_epoch: EPOCH } });
    // The next read is the truth the row repaints from — never what this click
    // intended.
    served = viewOf({
      targets: [
        rowOf(KINDS.live),
        rowOf(KINDS.standin, {
          active: true,
          is_baseline: true,
          available_now: false,
          reason: 'already_active',
        }),
        rowOf(KINDS.va),
      ],
    });
    confirmBtn()?.click();
    await flush();

    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'live',
      state: 'on',
      epoch: EPOCH,
    });
    expect(confirmEl()).toBeNull();
    expect(isOpen()).toBe(true);
    expect(rowEl('live')?.dataset.state).toBe('on');
    expect(switchWord('live')?.textContent).toBe('on');
    // A confirmed turn-on said its sentence in the dialog; no status line.
    expect(statusEl()).toBeNull();
  });

  test('a record that carried no epoch is sent as null', async () => {
    await bootOpen(viewOf({ switch_epoch: null }));
    toggleEl('va')?.click();
    await flush();
    expect(posts()[0].body).toEqual({ session_id: null, target: 'va', state: 'off', epoch: null });
  });

  test('a stop that was not taken says so, and the machine keeps reading on', async () => {
    await bootOpen();
    postAnswers.push({
      ok: false,
      status: 503,
      body: { detail: { reason: 'not_taken', taken: false, phrase: STOP_NOT_TAKEN } },
    });
    toggleEl('va')?.click();
    await flush();

    expect(outcomes('va')).toContain(`✗ ${STOP_NOT_TAKEN}`);
    // The re-read still says on, and nothing painted an off it does not hold.
    expect(switchWord('va')?.textContent).toBe('on');
    expect(toggleEl('va')?.getAttribute('aria-checked')).toBe('true');
    expect(fetchCalls.filter((c) => c.method === 'GET').length).toBeGreaterThan(1);
  });

  test('a 503 on a stop with no phrase still reads as the stop sentence', async () => {
    await bootOpen();
    postAnswers.push({ ok: false, status: 503, body: null });
    toggleEl('va')?.click();
    await flush();
    expect(outcomes('va')).toContain(`✗ ${STOP_NOT_TAKEN}`);
  });

  test('a refused turn-on keeps the dialog up carrying the route’s phrase', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();
    postAnswers.push({
      ok: false,
      status: 403,
      body: { detail: { reason: 'tier_never', phrase: TIER_NEVER_PHRASE } },
    });
    confirmBtn()?.click();
    await flush();

    expect(confirmEl()).not.toBeNull();
    expect(confirmEl()?.querySelector('.posture-modal-error')?.textContent).toBe(TIER_NEVER_PHRASE);
    expect(confirmBtn()?.disabled).toBe(false);
    expect(isOpen()).toBe(true);
  });

  test('a turn-on 503 shows its phrase in the dialog', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();
    const phrase =
      'Writes are refused because the switch record cannot be read; ask the deployment admin.';
    postAnswers.push({
      ok: false,
      status: 503,
      body: { detail: { reason: 'record_unavailable', phrase } },
    });
    confirmBtn()?.click();
    await flush();
    expect(confirmEl()?.querySelector('.posture-modal-error')?.textContent).toBe(phrase);
  });
});

describe('a stop that lands while a turn-on is asked', () => {
  test('a stamped 409 is retried once, with the epoch it returned', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();
    postAnswers.push(
      {
        ok: false,
        status: 409,
        body: { detail: { reason: 'stopped_since', epoch: '12', stamped: true } },
      },
      { ok: true, body: { target: 'live', state: 'on', switch_epoch: '12' } }
    );
    confirmBtn()?.click();
    await flush();

    expect(posts()).toHaveLength(2);
    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'live',
      state: 'on',
      epoch: EPOCH,
    });
    expect(posts()[1].body).toEqual({ session_id: null, target: 'live', state: 'on', epoch: '12' });
    expect(confirmEl()).toBeNull();
    expect(outcomes('live')).toHaveLength(0);
  });

  test('the retry is made once only', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();
    const stamped = { detail: { reason: 'stopped_since', epoch: '12', stamped: true } };
    postAnswers.push(
      { ok: false, status: 409, body: stamped },
      {
        ok: false,
        status: 409,
        body: { detail: { reason: 'stopped_since', epoch: '13', stamped: true } },
      },
      { ok: true, body: {} }
    );
    confirmBtn()?.click();
    await flush();

    expect(posts()).toHaveLength(2);
    expect(outcomes('live')).toContain(`✗ ${STOPPED_SINCE}`);
  });

  test('a 409 that stamped nothing re-reads and wants a new click', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();
    postAnswers.push({
      ok: false,
      status: 409,
      body: { detail: { reason: 'stopped_since', epoch: '9', stamped: false } },
    });
    served = viewOf({ switch_epoch: '9', any_on: false });
    confirmBtn()?.click();
    await flush();

    // Not retried: the turn-on never silently re-arms after a stop.
    expect(posts()).toHaveLength(1);
    expect(confirmEl()).toBeNull();
    expect(outcomes('live')).toContain(`✗ ${STOPPED_SINCE}`);
    expect(switchWord('live')?.textContent).toBe('off');

    // The new click carries the epoch the re-read brought.
    toggleEl('live')?.click();
    await flush();
    expect(confirmTitle()).toBe('Turn writes on for Real machine?');
    postAnswers.push({ ok: true, body: { target: 'live', state: 'on', switch_epoch: '9' } });
    confirmBtn()?.click();
    await flush();
    expect(posts()[1].body).toEqual({ session_id: null, target: 'live', state: 'on', epoch: '9' });
    // The next gesture cleared the last one's note.
    expect(outcomes('live')).toHaveLength(0);
  });
});

describe('one gesture at a time', () => {
  test('a second click while a POST is in flight sends nothing', async () => {
    await bootOpen();
    /** @type {(v: any) => void} */
    let release = () => {};
    const gate = new Promise((r) => {
      release = r;
    });
    const stubbed = /** @type {any} */ (globalThis.fetch);
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
        if ((init?.method ?? 'GET') === 'POST') await gate;
        return stubbed(url, init);
      })
    );
    toggleEl('va')?.click();
    allOffEl().click();
    release(undefined);
    await flush();
    expect(posts()).toHaveLength(1);
    expect(posts()[0].body).toMatchObject({ target: 'va', state: 'off' });
  });
});

describe('Turn all writes off', () => {
  test('POSTs all/off with the epoch the chip read', async () => {
    await bootOpen();
    postAnswers.push({ ok: true, body: { target: 'all', state: 'off', switch_epoch: EPOCH } });
    allOffEl().click();
    await flush();

    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'all',
      state: 'off',
      epoch: EPOCH,
    });
    expect(confirmEl()).toBeNull();
  });

  test('it is enabled exactly when the read says some switch is on', async () => {
    // A machine no longer configured can still be on: no row shows it, and the
    // read's own answer is the one that counts.
    await bootOpen(
      viewOf({
        any_on: true,
        targets: [rowOf(KINDS.live, { ...STATES.off }), rowOf(KINDS.va, { ...STATES.off })],
      })
    );
    expect(allOffEl().disabled).toBe(false);
  });

  test('it is disabled when nothing is on, whatever the rows say', async () => {
    await bootOpen(viewOf({ any_on: false }));
    expect(allOffEl().disabled).toBe(true);
  });

  test('it stays enabled under every lock', async () => {
    await bootOpen(
      viewOf({
        store_available: true,
        locks: { readonly_run: true, tier: 'never' },
        targets: [rowOf(KINDS.va, { ...STATES.locked })],
      })
    );
    expect(allOffEl().disabled).toBe(false);
  });

  test('a stop of everything that was not taken says so under the list', async () => {
    await bootOpen();
    postAnswers.push({
      ok: false,
      status: 503,
      body: { detail: { reason: 'not_taken', taken: false, phrase: STOP_NOT_TAKEN } },
    });
    allOffEl().click();
    await flush();
    const notes = [...document.querySelectorAll('.ctc-popover > .ctc-outcome')].map(
      (n) => n.textContent
    );
    expect(notes).toContain(`✗ ${STOP_NOT_TAKEN}`);
    expect(switchWord('va')?.textContent).toBe('on');
  });

  test('the foot states the popover’s scope, once', async () => {
    await bootOpen();
    expect(allOffEl().textContent).toBe('Turn all writes off');
    expect(document.querySelector('.ctc-foot-note')?.textContent).toBe(
      'Applies to every session of this login'
    );
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
    expect(body).toContain('Writes are on there.');
    expect(body).not.toMatch(/approval|asks you/i);

    postAnswers.push({ ok: true, body: { request_id: 'req-1', target: 'va' } });
    confirmBtn()?.click();
    await flush();

    expect(posts()[0].url).toContain('/api/terminal/target');
    expect(posts()[0].body).toEqual({ session_id: null, target: 'va' });
    expect(chipModule.isPending()).toBe(true);
    expect(outcomes('va')).toContain('switching…');
    // One outstanding gesture at a time: no row offers a second Switch.
    expect(document.querySelectorAll('.ctc-switch')).toHaveLength(0);
    expect(isOpen()).toBe(true);
  });

  test('the generation the 202 answers with is handed to the chip', async () => {
    // It is what ends the wait honestly: the record's terminus only says the
    // switch was accepted, and the chip waits until every live controls server
    // reports this number.
    await bootOpen(viewOf({ generation: 3 }));
    switchEl('va')?.click();
    await flush();
    postAnswers.push({ ok: true, body: { request_id: 'req-1', target: 'va', generation: 4 } });
    confirmBtn()?.click();
    await flush();
    expect(chipModule.isPending()).toBe(true);
    expect(outcomes('va')).toContain('switching…');

    /** @param {number} applied the generation the one live server has reached */
    const recordAccepted = (applied) =>
      viewOf({
        generation: 4,
        last_switch: {
          request_id: 'req-1',
          target: 'va',
          status: 'applied',
          generation: 4,
          age_s: 0,
        },
        servers: [{ pid: 4242, applied_generation: applied, last_switch: null }],
      });
    // The record accepted it, the server is still on 3: still switching.
    served = recordAccepted(3);
    await chipModule.refetch();
    await flush();
    expect(chipModule.isPending()).toBe(true);

    // The server reaches 4: landed. Without the 202's generation the chip
    // would have nothing to compare against and wait out its deadline.
    served = recordAccepted(4);
    await chipModule.refetch();
    await flush();
    expect(chipModule.isPending()).toBe(false);
    expect(outcomes('va')).toContain('✓ switched');
  });

  test('a request for the target already held arms no wait at all', async () => {
    // Same target, so the route accepts at the CURRENT generation and writes
    // no terminus: nothing moved. Arming a wait on that would sit in
    // `switching…` until the TTL and then report `request_expired` for a
    // request that succeeded.
    await bootOpen(viewOf({ generation: 3, control_target: 'va' }));
    switchEl('va')?.click();
    await flush();
    postAnswers.push({ ok: true, body: { request_id: 'req-1', target: 'va', generation: 3 } });
    confirmBtn()?.click();
    await flush();
    expect(posts()[0].url).toContain('/api/terminal/target');
    expect(chipModule.isPending()).toBe(false);
    expect(outcomes('va')).not.toContain('switching…');
  });

  test('arriving with writes off is said as the nothing-moves sentence', async () => {
    await bootOpen();
    switchEl('live')?.click();
    await flush();
    const body = inConfirm('.posture-modal-body').textContent ?? '';
    expect(body).toContain('Writes are off there — nothing moves until you turn them on.');
    // Off means off: no hardware notice until writes are actually on there.
    expect(confirmEl()?.querySelector('.posture-modal-live')).toBeNull();
  });

  test('a machine arrived at with writes off promises reads only', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.off })] }));
    switchEl('va')?.click();
    await flush();
    const body = inConfirm('.posture-modal-body').textContent ?? '';
    expect(body).toContain('All control reads go to 127.0.0.1:10064.');
    expect(body).not.toContain('reads and writes');
  });

  test('a locked machine promises reads only too', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.locked })] }));
    switchEl('va')?.click();
    await flush();
    const body = inConfirm('.posture-modal-body').textContent ?? '';
    expect(body).toContain('All control reads go to 127.0.0.1:10064.');
    expect(body).toContain('Writes are locked there.');
  });

  test('switching onto the live machine with writes on carries the hardware notice', async () => {
    await bootOpen(viewOf({ targets: [rowOf(KINDS.live)] }));
    switchEl('live')?.click();
    await flush();
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Real machine — writes move hardware'
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
      control_target: 'va',
      last_switch: {
        request_id: 'req-1',
        target: 'va',
        status: 'applied',
        reason: null,
        age_s: 4,
      },
      targets: [
        rowOf(KINDS.live, { ...STATES.off }),
        rowOf(KINDS.standin, { is_baseline: true }),
        rowOf(KINDS.va, { active: true, available_now: false, reason: 'already_active' }),
      ],
    });
    await chipModule.refetch();
    await flush();

    expect(chipModule.isPending()).toBe(false);
    // The outcome, with no counter: the age decides whether the line renders,
    // never what it says.
    expect(outcomes('va')).toContain('✓ switched');
    expect(outcomes('va').join(' ')).not.toMatch(/s ago/);
  });

  test('the record’s word for an accepted switch is applied, and it reads ✓ switched', async () => {
    // The record writes `applied` (control_context.SWITCH_APPLIED); nothing on
    // this path has ever published `success`. Matching the wrong word sent
    // every landed switch to the refusal branch, where it read `✗ applied`.
    await bootOpen(
      viewOf({
        last_switch: {
          request_id: 'req-9',
          target: 'va',
          status: 'applied',
          generation: 4,
          reason: null,
          age_s: 2,
        },
      })
    );
    expect(outcomes('va')).toContain('✓ switched');
    // `data-status` is the stylesheet's vocabulary, not the record's — the same
    // split kindAttr makes between the route's words and the CSS values.
    const line = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-outcome'));
    expect(line.dataset.status).toBe('success');
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
        last_switch: { request_id: 'req-9', status: 'applied', reason: null, age_s: 1 },
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
    // The client mints this code itself, so it always has a phrase.
    expect(outcomes('va')).toContain('✗ no answer');
    const line = /** @type {HTMLElement} */ (rowEl('va')?.querySelector('.ctc-outcome'));
    expect(line.dataset.status).toBe('expired');
  });

  test('an outcome older than the freshness window is history, not news', async () => {
    await bootOpen(
      viewOf({
        last_switch: {
          request_id: 'req-9',
          target: 'va',
          status: 'applied',
          reason: null,
          age_s: OUTCOME_MAX_AGE_S + 1,
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
          rowOf(KINDS.live, { ...STATES.off, available_now: false, reason: 'unreachable' }),
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
              ...STATES.off,
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
            rowOf(KINDS.live, { ...STATES.off, available_now: false, reason: code }),
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
          rowOf(KINDS.live, { ...STATES.off, reason: null }),
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

/* ---- don't ask again ----------------------------------------------------- */

describe("don't ask again", () => {
  /** The checkbox in the confirm that is up, or null. */
  const skipBox = () =>
    /** @type {HTMLInputElement|null} */ (
      confirmEl()?.querySelector('.posture-modal-skip input') ?? null
    );
  /** Click with Shift held — the waived dialog's re-show gesture. */
  const shiftClick = (/** @type {HTMLElement|null|undefined} */ node) =>
    node?.dispatchEvent(
      new MouseEvent('click', { shiftKey: true, bubbles: true, cancelable: true })
    );

  test('ticking it on a switch confirm mutes the NEXT switch to that machine only', async () => {
    await bootOpen();
    switchEl('va')?.click();
    await flush();
    /** @type {HTMLInputElement} */ (skipBox()).click();
    postAnswers.push({ ok: true, body: { request_id: 'req-1', target: 'va' } });
    confirmBtn()?.click();
    await flush();

    // The waiver is per gesture per machine, user-scoped (bare key here:
    // no data-osprey-storage-scope on a single-user page).
    expect(localStorage.getItem('osprey-ctc-skip-confirm:switch:va')).toBe('1');
    expect(localStorage.getItem('osprey-ctc-skip-confirm:switch:live')).toBeNull();

    // The outcome lands (clearing the pending request), the session comes
    // back; then switch to va again: no dialog, straight to the POST.
    served = viewOf({
      last_switch: { request_id: 'req-1', target: 'va', status: 'applied', reason: null, age_s: 2 },
    });
    await chipModule.refetch();
    await flush();
    postAnswers.push({ ok: true, body: { request_id: 'req-2', target: 'va' } });
    switchEl('va')?.click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(2);
    expect(posts()[1].body).toEqual({ session_id: null, target: 'va' });
    expect(outcomes('va')).toContain('switching…');
  });

  test('a waived switch to ANOTHER machine still asks', async () => {
    localStorage.setItem('osprey-ctc-skip-confirm:switch:va', '1');
    await bootOpen();
    switchEl('live')?.click();
    await flush();
    expect(confirmTitle()).toBe('Switch to Real machine?');
  });

  test('cancelling with the box ticked records nothing', async () => {
    await bootOpen();
    switchEl('va')?.click();
    await flush();
    /** @type {HTMLInputElement} */ (skipBox()).click();
    inConfirm('.posture-modal-cancel').click();
    await flush();

    expect(localStorage.getItem('osprey-ctc-skip-confirm:switch:va')).toBeNull();
    switchEl('va')?.click();
    await flush();
    expect(confirmTitle()).toBe('Switch to Simulator?');
  });

  test('Shift re-shows a waived dialog, pre-ticked, and unticking there is the undo', async () => {
    localStorage.setItem('osprey-ctc-skip-confirm:switch:va', '1');
    await bootOpen();
    shiftClick(switchEl('va'));
    await flush();

    expect(confirmTitle()).toBe('Switch to Simulator?');
    expect(skipBox()?.checked).toBe(true);

    /** @type {HTMLInputElement} */ (skipBox()).click();
    postAnswers.push({ ok: true, body: { request_id: 'req-1', target: 'va' } });
    confirmBtn()?.click();
    await flush();
    expect(localStorage.getItem('osprey-ctc-skip-confirm:switch:va')).toBeNull();
  });

  test('a waived turn-on applies directly and says the turn-on sentence once', async () => {
    localStorage.setItem('osprey-ctc-skip-confirm:writes-on:va:simulator', '1');
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.off })] }));
    postAnswers.push({ ok: true, body: { target: 'va', state: 'on', switch_epoch: EPOCH } });
    toggleEl('va')?.click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()[0].body).toEqual({ session_id: null, target: 'va', state: 'on', epoch: EPOCH });
    // No dialog said it, so the status line does — and never beside the off
    // control.
    expect(statusEl()?.textContent).toBe(TURN_ON);
    expect(surfaceEl('va')?.textContent ?? '').not.toContain(TURN_ON);

    // It survives the poll's re-render, and the next gesture clears it.
    await chipModule.refetch();
    await flush();
    expect(statusEl()?.textContent).toBe(TURN_ON);
    allOffEl().click();
    await flush();
    expect(statusEl()).toBeNull();
  });

  test('a waived turn-on that is refused shows no turn-on sentence', async () => {
    localStorage.setItem('osprey-ctc-skip-confirm:writes-on:va:simulator', '1');
    await bootOpen(viewOf({ targets: [rowOf(KINDS.va, { ...STATES.off })] }));
    postAnswers.push({
      ok: false,
      status: 403,
      body: { detail: { reason: 'tier_never', phrase: TIER_NEVER_PHRASE } },
    });
    toggleEl('va')?.click();
    await flush();
    expect(statusEl()).toBeNull();
    expect(outcomes('va')).toContain(`✗ ${TIER_NEVER_PHRASE}`);
  });

  test('turning writes on for the live machine offers no waiver and ignores a stale one', async () => {
    // The gesture after which hardware can move keeps asking, always — even a
    // hand-planted waiver key changes nothing.
    localStorage.setItem('osprey-ctc-skip-confirm:writes-on:live:real', '1');
    await bootOpen();
    toggleEl('live')?.click();
    await flush();

    expect(confirmTitle()).toBe('Turn writes on for Real machine?');
    expect(skipBox()).toBeNull();
    expect(posts()).toHaveLength(0);
  });

  test('a waived switch that is refused puts the server sentence on the row', async () => {
    // With no dialog up, the refusal's home is the row — same as a verb
    // click's.
    localStorage.setItem('osprey-ctc-skip-confirm:switch:va', '1');
    await bootOpen();
    postAnswers.push({
      ok: false,
      status: 409,
      body: { detail: { error: 'session_not_started', message: 'No controls server yet.' } },
    });
    switchEl('va')?.click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(outcomes('va')).toContain('✗ No controls server yet.');
  });
});

/* ---- the confirm sits ABOVE the popover, which stays open ---------------- */

describe('a confirm and the popover beneath it', () => {
  test('a press inside the confirm is not a press outside', async () => {
    await bootOpen();
    toggleEl('live')?.click();
    await flush();

    pointerDown(inConfirm('.posture-modal-body'));
    expect(isOpen()).toBe(true);
    expect(confirmEl()).not.toBeNull();
  });

  test('Escape dismisses the confirm first and the popover only after', async () => {
    await bootOpen();
    toggleEl('live')?.click();
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

/* ---- a roster no row can be switched from -------------------------------- */

describe('rows the route refuses', () => {
  const refusedView = () =>
    viewOf({
      targets: [
        rowOf(KINDS.live, { available_now: false, reason: 'gateways_missing' }),
        rowOf(KINDS.va, { available_now: false, reason: 'gateways_missing', active: true }),
      ],
    });

  test('offers no Switch, and says why in its place', async () => {
    await bootOpen(refusedView());
    expect(document.querySelectorAll('.ctc-switch')).toHaveLength(0);
    expect(reasonEl('live')?.textContent).toBe('not set up');
    // The rows are refused, not the context: nothing is wrong deployment-wide.
    expect(bannerEl()).toBeNull();
  });

  test('keeps their switches live — writes are a separate question from switching', async () => {
    await bootOpen(refusedView());
    expect(toggleEl('live')?.disabled).toBe(false);

    postAnswers.push({ ok: true, body: { target: 'live', state: 'off', switch_epoch: EPOCH } });
    toggleEl('live')?.click();
    await flush();
    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'live',
      state: 'off',
      epoch: EPOCH,
    });
  });
});

/* ---- realign and the execution in flight ---------------------------------- */

describe('a narrowing that has not reached the agent yet', () => {
  test('the card says it takes effect after the run', async () => {
    await bootOpen(
      viewOf({
        execution_in_flight: [
          {
            pid: 4242,
            target: 'standin',
            session: SESSION,
            surface: 'notebook_kernel',
            kernel_id: 'k-1',
            started_at: '2026-08-30T12:00:00+00:00',
            age_s: 4,
          },
        ],
        last_posture_realign: { state: 'pending' },
      })
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

    toggleEl('live')?.click();
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

  test('a locked, unreachable, followed roster is identical too', async () => {
    // The lock here is a context this terminal follows rather than owns: every
    // row is read-only and the banner names the holder. That is the widest
    // difference the two modes could have rendered differently, and they must
    // not — the modes differ in defaults, never in markup.
    const view = () =>
      viewOf({
        owner: { kind: 'web_terminal', pid: 77, port: 8123, self: false },
        targets: [
          rowOf(KINDS.live, {
            ...STATES.off,
            reachability: { state: 'down', role: 'read_only', age_s: 8 },
          }),
          rowOf(KINDS.va, { is_baseline: true, active: true }),
        ],
      });
    const simple = await markupUnder('simple', view());
    const expert = await markupUnder('expert', view());
    expect(simple).toBe(expert);
    // And the lock is really in the markup being compared — a seed that stopped
    // biting would leave two plain rosters agreeing about nothing.
    expect(simple).toContain('holds the control context');
  });
});

/* ---- request plumbing ----------------------------------------------------- */

describe('request plumbing', () => {
  test('every request goes through the per-user mount prefix', async () => {
    /** @type {any} */ (window).__OSPREY_PREFIX__ = '/u/alice';
    await bootOpen();
    postAnswers.push({ ok: true, body: { target: 'va', state: 'off', switch_epoch: EPOCH } });
    toggleEl('va')?.click();
    await flush();

    expect(fetchCalls.length).toBeGreaterThan(0);
    for (const call of fetchCalls) expect(call.url.startsWith('/u/alice/')).toBe(true);
  });
});
