// @ts-check
/**
 * Unit tests for the control-target chip (control-target-chip.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-chip.test.mjs
 *
 * The chip is the sidebar's one-line answer to "if the agent writes now, where
 * does it land, and will it be refused?". It reads ONE truth —
 * `GET /api/terminal/posture` — and derives nothing from what this page last
 * did, so a posture narrowed in another tab and a target the agent switched
 * mid-turn both reach the operator.
 *
 * What these tests pin down, in the order a reviewer would ask about it:
 *
 * - the chip mounts once, into the host its caller hands it (the hub hands it
 *   the sidebar's control-system block, `#sidebar-target`) and nowhere else:
 *   with no host it does nothing at all. A second init re-renders rather than
 *   mounting a second chip, and re-homes the same node into a new host;
 * - a caller may hand it a `host` of its own, which is what lets the
 *   JupyterLab bar mount it (that suite pins the import closure);
 * - `targetReading()` gives the chip's own words (name, write state, kind,
 *   state word) to the surfaces that repeat them;
 * - the roster is a fact about the DEPLOYMENT: the read carries no session id,
 *   and nothing here waits for a session to be settled;
 * - the pushed `{type: 'control_context'}` frame is what makes the chip
 *   current; the 5 s poll is the fallback behind it;
 * - a switch is done when every live controls server reports the generation
 *   the POST answered with — not when the record's terminus turns up;
 * - the full state matrix reaches the DOM as data attributes and words: three
 *   kinds (real / rehearsal / simulator, GET's `machine_kind`) × three states
 *   (on / off / locked, derived from `switch` and `write_allowed`). Not one
 *   colour is decided here — the stylesheet owns the map, so the attributes
 *   ARE the contract;
 * - `off` and `locked` stay separate words: one click undoes off, while a
 *   locked switch is on and still refused (or cannot be read);
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
 * Seams: nothing is mocked. `fetch` is stubbed the way the other suites here
 * stub it, and `EventSource` is stubbed as a class, since happy-dom has none —
 * so the chip's frames arrive through api.js's real shared stream. Module
 * state is isolated by vi.resetModules() + dynamic import per test and the
 * module's own teardown export.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { FakeEventSource, stubEventSource } from '../_support/event-source.mjs';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js')} */
let chipModule;

/* ---- roster fixtures ---------------------------------------------------- */

/**
 * The three machine kinds, in the shape the route publishes them. `machine_kind`
 * is the write verdict's kind (C-KIND), which is also the CSS value these
 * tests read.
 */
const KINDS = {
  live: {
    target: 'live',
    label: 'Real machine — writes move hardware',
    short_label: 'LIVE',
    machine_kind: 'real',
    endpoint: 'gw.example.org:5064',
    real_machine: true,
  },
  standin: {
    target: 'standin',
    label: 'Rehearsal — nothing moves',
    short_label: 'STAND-IN',
    machine_kind: 'rehearsal',
    endpoint: '127.0.0.1:10090',
    real_machine: true,
  },
  va: {
    target: 'va',
    label: 'Simulator — nothing moves',
    short_label: 'VIRTUAL',
    machine_kind: 'simulator',
    endpoint: '127.0.0.1:10064',
    real_machine: false,
  },
};

/**
 * The four states as C-GET's columns express them. `off` and `locked` are
 * both "no", and which "no" it is decides whether a toggle can undo it.
 * Each carries a legacy `posture` that DISAGREES with it, so a derivation
 * that still read `posture` would fail here.
 */
const STATES = {
  asked: {
    switch: 'asked',
    prompt: true,
    write_allowed: true,
    write_reason: null,
    write_phrase: null,
    posture: 'sandbox',
  },
  allowed: {
    switch: 'allowed',
    prompt: false,
    write_allowed: true,
    write_reason: null,
    write_phrase: null,
    posture: 'sandbox',
  },
  off: {
    switch: 'off',
    prompt: true,
    write_allowed: false,
    write_reason: 'switch_off',
    write_phrase: 'Writes are off; turn it on from the chip in the web terminal.',
    posture: 'writes',
    effective: true,
  },
  locked: {
    switch: 'asked',
    prompt: true,
    write_allowed: false,
    write_reason: 'readonly_run',
    write_phrase:
      'Writes are refused because the deployment is running read-only; the chip cannot lift this — ask the deployment admin.',
    posture: 'writes',
    effective: true,
  },
};

/** @param {object} [o] */
const rowOf = (o = {}) => ({
  ...KINDS.standin,
  ...STATES.asked,
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
    role_detail: { write_access: 'reached' },
  },
  ...o,
});

/**
 * One live controls server, as `servers[]` publishes it. The default row has
 * arrived at the record's generation, which is what a settled deployment looks
 * like: nothing outstanding for anyone to wait on.
 * @param {object} [o]
 */
const serverOf = (o = {}) => ({
  pid: 4242,
  session: null,
  applied_target: 'standin',
  applied_generation: 3,
  last_switch: null,
  last_posture_realign: null,
  updated_at: '2026-08-30T12:00:00+00:00',
  ...o,
});

/** @param {object} [o] */
const viewOf = (o = {}) => ({
  // The read carries no session id, so the route echoes none back. Seeding one
  // would let an assertion pass on a value production never sees.
  session_id: null,
  control_target: 'standin',
  generation: 3,
  owner: { kind: 'web_terminal', pid: 1000, port: 8080, self: true },
  servers: [serverOf()],
  store_available: true,
  readonly_run: false,
  switch_epoch: '1',
  any_armed: true,
  locks: { readonly_run: false, tier: 'allowed' },
  execution_in_flight: [],
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

/**
 * The sidebar the chip mounts into, in index.html's shape: the control-system
 * block `#sidebar-target` between the sidebar head and the palette trigger.
 * The block ships EMPTY; the chip is what fills it, and its neighbours are
 * never touched.
 */
function mountFixture() {
  document.body.innerHTML = `
    <nav class="sidebar" id="sidebar" aria-label="Panels">
      <div class="sidebar-head">
        <a class="sidebar-brand" href="/" aria-label="OSPREY"></a>
        <button class="osprey-icon-btn" id="sidebar-toggle" type="button"></button>
      </div>
      <div class="sidebar-target" id="sidebar-target"></div>
      <button class="osprey-search" id="command-palette-btn" type="button"></button>
    </nav>`;
}

/** The sidebar's control-system block, the host app.js hands the chip. */
const targetHost = () => /** @type {HTMLElement} */ (document.getElementById('sidebar-target'));

/** Drain the microtask/timer queue the async handlers chain through. */
async function flush() {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Boot the chip against a given roster, into `opts.host` when the key is
 * given (null included) and into the sidebar's block otherwise.
 * @param {any} [payload]
 * @param {{host?: HTMLElement|null}} [opts]
 */
async function boot(payload, opts = {}) {
  served = payload ?? viewOf();
  chipModule.initControlTargetChip({ host: 'host' in opts ? opts.host : targetHost() });
  await flush();
}

const chipEl = () =>
  /** @type {HTMLButtonElement|null} */ (document.querySelector('.control-target-chip'));
const anchorEl = () => /** @type {HTMLElement|null} */ (document.querySelector('.ctc-anchor'));
const shortText = () => document.querySelector('.ctc-short')?.textContent ?? '';
const stateText = () => document.querySelector('.ctc-state')?.textContent ?? '';

/** Deliver one frame on the page's event stream, as the browser would. */
function pushFrame(/** @type {any} */ frame) {
  const data = typeof frame === 'string' ? frame : JSON.stringify(frame);
  FakeEventSource.opened.at(-1)?.onmessage?.({ data });
}

beforeEach(async () => {
  stubEventSource({ open: true });
  vi.resetModules();
  served = viewOf();
  stubFetch();
  mountFixture();
  Reflect.deleteProperty(window, '__OSPREY_PREFIX__');
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
  test('mounts into the host it is handed, leaving its neighbours alone', async () => {
    await boot();
    const sidebar = /** @type {HTMLElement} */ (document.getElementById('sidebar'));
    const host = targetHost();
    // The block's neighbours keep their places: the chip only ever appends
    // into its host and never positions itself against the palette trigger.
    const order = [...sidebar.children].map((el) => el.id || el.className);
    expect(order).toEqual(['sidebar-head', 'sidebar-target', 'command-palette-btn']);
    expect(host.children).toHaveLength(1);
    expect(host.firstElementChild).toBe(anchorEl());
    // The chip lives inside its own wrapper, and that wrapper is what fills
    // the block.
    expect(chipEl()?.parentElement).toBe(anchorEl());
    expect(anchorEl()?.hidden).toBe(false);
  });

  test('is idempotent — a second init re-renders rather than mounting twice', async () => {
    await boot();
    chipModule.initControlTargetChip({ host: targetHost() });
    await flush();
    expect(document.querySelectorAll('.control-target-chip')).toHaveLength(1);
    expect(document.querySelectorAll('.ctc-anchor')).toHaveLength(1);
  });

  test('re-homes the SAME chip into a different host', async () => {
    await boot();
    const first = chipEl();
    const anchor = anchorEl();

    const other = document.createElement('div');
    document.body.appendChild(other);
    chipModule.initControlTargetChip({ host: other });
    await flush();

    // Moved, not rebuilt: the popover hangs its listeners on this exact node
    // and the chip's current paint rides along with it.
    expect(document.querySelectorAll('.control-target-chip')).toHaveLength(1);
    expect(chipEl()).toBe(first);
    expect(anchorEl()).toBe(anchor);
    expect(anchor?.parentElement).toBe(other);
    expect(targetHost().children).toHaveLength(0);
  });

  test('carries the popover trigger ARIA and the four spans the stylesheet keys on', async () => {
    await boot();
    const chip = /** @type {HTMLButtonElement} */ (chipEl());
    expect(chip.type).toBe('button');
    expect(chip.getAttribute('aria-haspopup')).toBe('dialog');
    expect(chip.getAttribute('aria-expanded')).toBe('false');
    for (const cls of ['.ctc-dot', '.ctc-short', '.ctc-sep', '.ctc-state', '.ctc-caret']) {
      expect(chip.querySelector(cls)).not.toBeNull();
    }
    expect(chip.querySelector('.ctc-sep')?.textContent).toBe('·');
    expect(chip.querySelector('.ctc-caret')?.textContent).toBe('▾');
  });

  test('reads and paints without waiting for any session to be settled', async () => {
    // The roster is a fact about the deployment: one record says where its
    // control system points. A chip that waited for a session id would never
    // paint on a page that has no terminal.
    await boot();
    expect(getCount()).toBe(1);
    expect(anchorEl()?.hidden).toBe(false);
    expect(chipEl()?.hidden).toBe(false);
  });

  test('does nothing at all without a host, even beside an empty block', async () => {
    // No discovery: `#sidebar-target` is on the page, but the chip mounts only
    // where its caller puts it (app.js skips the call when the picker is off).
    chipModule.initControlTargetChip();
    chipModule.initControlTargetChip({ host: null });
    await flush();
    expect(chipEl()).toBeNull();
    expect(anchorEl()).toBeNull();
    expect(targetHost().children).toHaveLength(0);
    expect(getCount()).toBe(0);
    expect(FakeEventSource.opened).toHaveLength(0);
  });

  test('mounts into a host the caller hands it, sidebar or no sidebar', async () => {
    // The JupyterLab page has no sidebar; the bar module that owns its own
    // element passes it.
    document.body.innerHTML = '<div id="lab-bar"></div>';
    const host = /** @type {HTMLElement} */ (document.querySelector('#lab-bar'));
    await boot(viewOf(), { host });
    expect(anchorEl()?.parentElement).toBe(host);
    expect(chipEl()?.hidden).toBe(false);
  });
});

/* ---- targetReading ------------------------------------------------------ */

describe('targetReading', () => {
  test('is null while nothing has been read', () => {
    expect(chipModule.targetReading()).toBeNull();
  });

  test('answers in the chip’s own words for every kind and state', async () => {
    /** @type {Record<string, [string, string]>} */
    const NAMED = {
      live: ['Real machine', 'real'],
      standin: ['Rehearsal', 'rehearsal'],
      va: ['Simulator', 'simulator'],
    };
    for (const [kindName, kind] of Object.entries(KINDS)) {
      for (const [stateName, state] of Object.entries(STATES)) {
        served = viewOf({
          control_target: kind.target,
          targets: [rowOf({ ...kind, ...state, active: true })],
        });
        if (chipModule.getState()) await chipModule.refetch();
        else await boot(served);
        const [name, kindAttr] = NAMED[kindName];
        expect(chipModule.targetReading()).toEqual({
          name,
          state: `writes ${stateName}`,
          kind: kindAttr,
          word: stateName,
        });
        // The one derivation the chip paints from.
        expect(shortText()).toBe(name);
        expect(stateText()).toBe(`writes ${stateName}`);
      }
    }
  });

  test("names the deployment's configured display_name", async () => {
    await boot(
      viewOf({
        targets: [rowOf({ ...KINDS.live, display_name: 'Example storage ring', active: true })],
      })
    );
    expect(chipModule.targetReading()?.name).toBe('Example storage ring');
    expect(chipModule.targetReading()?.kind).toBe('real');
  });

  test('says switching… while a request this browser wrote is outstanding', async () => {
    await boot();
    chipModule.markPending('req-1', 'va', 4);
    const reading = chipModule.targetReading();
    expect(reading?.state).toBe('switching…');
    // The row it speaks for has not changed yet.
    expect(reading?.word).toBe('asked');
    expect(stateText()).toBe('switching…');
  });

  test('is null for an empty roster, where the chip hides', async () => {
    await boot(viewOf({ targets: [] }));
    expect(chipModule.targetReading()).toBeNull();
  });
});

/* ---- the state matrix --------------------------------------------------- */

describe('state matrix', () => {
  /** @type {Record<string, string>} */
  const KIND_ATTR = { live: 'real', standin: 'rehearsal', va: 'simulator' };
  /** The consequence word each kind renders when no display_name is configured. */
  /** @type {Record<string, string>} */
  const WORDS = { live: 'Real machine', standin: 'Rehearsal', va: 'Simulator' };
  /** @type {Record<string, string>} */
  const PHRASES = {
    asked: 'writes asked',
    allowed: 'writes allowed',
    off: 'writes off',
    locked: 'writes locked',
  };

  for (const [kindName, kind] of Object.entries(KINDS)) {
    for (const [stateName, state] of Object.entries(STATES)) {
      test(`${kindName} × ${stateName} → data attributes and words`, async () => {
        await boot(
          viewOf({
            control_target: kind.target,
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

  test('a stored allowed capped at asked reads writes asked', async () => {
    // The route answers `prompt: true` from the effective position: the word
    // says what the next write meets, and the stored position stays `allowed`.
    await boot(
      viewOf({
        locks: { readonly_run: false, tier: 'asked' },
        targets: [rowOf({ ...STATES.allowed, prompt: true, active: true })],
      })
    );
    expect(chipEl()?.dataset.state).toBe('asked');
    expect(stateText()).toBe('writes asked');
  });

  test('a standin GET calls the real machine reads as the real machine', async () => {
    // C-KIND: a stand-in no live server vouched for is `real`. Neither the
    // target name nor the "(stand-in)" label may turn it back into the
    // rehearsal.
    await boot(
      viewOf({
        targets: [rowOf({ ...KINDS.standin, machine_kind: 'real', active: true })],
      })
    );
    expect(chipEl()?.dataset.targetKind).toBe('real');
    expect(shortText()).toBe('Real machine');
  });

  test('a row with no machine_kind falls back to the loud answer, whatever its label', async () => {
    for (const shape of [
      { ...KINDS.standin, machine_kind: undefined },
      { ...KINDS.va, machine_kind: undefined },
      { ...KINDS.live, machine_kind: 'live' },
    ]) {
      served = viewOf({ targets: [rowOf({ ...shape, active: true })] });
      if (chipModule.getState()) await chipModule.refetch();
      else await boot(served);
      expect(chipEl()?.dataset.targetKind).toBe('real');
      expect(shortText()).toBe('Real machine');
    }
  });

  test('the state reads switch and write_allowed, never posture or effective', async () => {
    await boot(
      viewOf({
        targets: [
          rowOf({
            switch: 'allowed',
            prompt: false,
            write_allowed: false,
            posture: 'writes',
            effective: true,
            active: true,
          }),
        ],
      })
    );
    expect(chipEl()?.dataset.state).toBe('locked');
    // An unreadable switch is locked, never off.
    served = viewOf({
      targets: [
        rowOf({
          switch: null,
          write_allowed: false,
          write_reason: 'record_unavailable',
          active: true,
        }),
      ],
    });
    await chipModule.refetch();
    expect(chipEl()?.dataset.state).toBe('locked');
    expect(stateText()).toBe('writes locked');
  });

  test('stateWord answers off | asked | allowed | locked and never "sandbox"', () => {
    /** @param {object} o */
    const wordOf = (o) => chipModule.stateWord(/** @type {any} */ (rowOf(o)));
    expect(wordOf(STATES.asked)).toBe('asked');
    expect(wordOf(STATES.allowed)).toBe('allowed');
    expect(wordOf(STATES.off)).toBe('off');
    expect(wordOf(STATES.locked)).toBe('locked');
    expect(wordOf({ switch: null, write_allowed: false })).toBe('locked');
    // The capped row: stored allowed, effective asked.
    expect(wordOf({ ...STATES.allowed, prompt: true })).toBe('asked');
    // An armed switch the verdict refuses is locked at either position.
    expect(wordOf({ ...STATES.allowed, write_allowed: false })).toBe('locked');
    // A switch word this module does not know, the retired `on` included, is
    // an unreadable switch, even with a verdict that would pass.
    expect(wordOf({ ...STATES.asked, switch: 'on' })).toBe('locked');
    expect(wordOf({ ...STATES.asked, switch: undefined })).toBe('locked');
    // A row that does not publish `prompt` reads the louder word.
    expect(wordOf({ ...STATES.asked, prompt: undefined })).toBe('allowed');
    // Not even a row whose legacy mirror still says sandbox.
    for (const o of [
      { posture: 'sandbox', effective: false },
      { switch: 'off', posture: 'sandbox' },
      { switch: undefined, posture: 'sandbox' },
    ]) {
      expect(wordOf(o)).not.toBe('sandbox');
    }
  });

  test("the deployment's configured display_name wins over the kind default", async () => {
    await boot(
      viewOf({
        targets: [rowOf({ ...KINDS.live, display_name: 'Example storage ring', active: true })],
      })
    );
    expect(shortText()).toBe('Example storage ring');
    expect(chipEl()?.dataset.targetKind).toBe('real');
  });

  test('data-enforceable answers whether a change here would be written', async () => {
    await boot();
    expect(chipEl()?.dataset.enforceable).toBe('true');

    // No context to write to: the write routes answer 503.
    served = viewOf({ store_available: false });
    await chipModule.refetch();
    expect(chipEl()?.dataset.enforceable).toBe('false');
    // Dimmed, never hidden — the roster is worth reading either way.
    expect(anchorEl()?.hidden).toBe(false);

    // Another terminal holds it: the write routes answer 409.
    served = viewOf({ owner: { kind: 'web_terminal', pid: 77, port: 8123, self: false } });
    await chipModule.refetch();
    expect(chipEl()?.dataset.enforceable).toBe('false');
    expect(anchorEl()?.hidden).toBe(false);

    // Nothing owns it anywhere.
    served = viewOf({ owner: null });
    await chipModule.refetch();
    expect(chipEl()?.dataset.enforceable).toBe('false');
  });
});

/* ---- which row the chip speaks for -------------------------------------- */

describe('active row', () => {
  test('speaks for the active row, not the first one', async () => {
    await boot(
      viewOf({
        control_target: 'va',
        targets: [
          rowOf({ ...KINDS.live, is_baseline: true }),
          rowOf({ ...KINDS.va, active: true }),
        ],
      })
    );
    expect(shortText()).toBe('Simulator');
    expect(chipEl()?.dataset.targetKind).toBe('simulator');
  });

  test('falls back to the baseline row when no row is marked active', async () => {
    await boot(
      viewOf({
        control_target: 'nothing-configured',
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

describe('refetch hints', () => {
  test('subscribes to the shared panel event stream, once', async () => {
    await boot();
    expect(FakeEventSource.opened).toHaveLength(1);
    expect(FakeEventSource.opened[0].url).toBe('/api/files/events');
    // A second init must not subscribe again: api.js would share the socket,
    // so what a second subscription costs is a second read per frame.
    chipModule.initControlTargetChip({ host: targetHost() });
    await flush();
    expect(FakeEventSource.opened).toHaveLength(1);
    const before = getCount();
    pushFrame({ type: 'control_context' });
    await flush();
    expect(getCount()).toBe(before + 1);
  });

  test('a pushed frame beats the idle poll to the news', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] });
    await boot(viewOf({ targets: [rowOf({ ...KINDS.standin, active: true })] }));
    expect(shortText()).toBe('Rehearsal');

    served = viewOf({
      control_target: 'live',
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    const before = getCount();
    pushFrame({ type: 'control_context' });
    await flush();
    // Repainted without a single tick of the 5 s fallback having passed.
    expect(getCount()).toBe(before + 1);
    expect(shortText()).toBe('Real machine');
  });

  test('a control_target_set frame triggers a re-read', async () => {
    await boot();
    const before = getCount();
    pushFrame({ type: 'agent_activity', tool: 'control_target_set', target: { kind: 'config' } });
    await flush();
    expect(getCount()).toBe(before + 1);
  });

  test('renders the ROUTE, never the frame text', async () => {
    await boot(viewOf({ targets: [rowOf({ ...KINDS.standin, active: true, is_baseline: true })] }));
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
    expect(chipEl()?.dataset.targetKind).toBe('rehearsal');
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
    // `data-state` still describes the machine the deployment is ON — the switch
    // has not landed, and the dot must not go quiet before it does.
    expect(chipEl()?.dataset.state).toBe('asked');

    const before = getCount();
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS * 3);
    await flush();
    expect(getCount()).toBe(before + 3);
  });

  test('the fast poll is torn down when the route publishes that request_id', async () => {
    await boot();
    chipModule.markPending('r-7f21', 'live');
    served = viewOf({
      control_target: 'live',
      last_switch: { request_id: 'r-7f21', status: 'applied', reason: null, age_s: 0 },
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();

    expect(chipModule.isPending()).toBe(false);
    expect(chipEl()?.dataset.pending).toBeUndefined();
    expect(stateText()).toBe('writes asked');
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
      last_switch: { request_id: 'r-someone-else', status: 'applied', age_s: 1 },
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

  test('the record accepting the switch is not yet the end of switching…', async () => {
    await boot();
    chipModule.markPending('r-7f21', 'live', 4);
    // The terminus says applied and the record is at 4; the one live controls
    // server is still on 3. Stopping here would name a machine it has not
    // reached.
    served = viewOf({
      control_target: 'live',
      generation: 4,
      last_switch: { request_id: 'r-7f21', status: 'applied', generation: 4, age_s: 0 },
      servers: [serverOf({ applied_generation: 3 })],
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();
    expect(chipModule.isPending()).toBe(true);
    expect(stateText()).toBe('switching…');

    // The server arrives.
    served = viewOf({
      control_target: 'live',
      generation: 4,
      last_switch: { request_id: 'r-7f21', status: 'applied', generation: 4, age_s: 1 },
      servers: [serverOf({ applied_target: 'live', applied_generation: 4 })],
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();
    expect(chipModule.isPending()).toBe(false);
    expect(stateText()).toBe('writes asked');
    expect(shortText()).toBe('Real machine');
  });

  test('a server that fails the switch ends the wait and is named', async () => {
    await boot();
    chipModule.markPending('r-7f21', 'live', 4);
    served = viewOf({
      generation: 4,
      last_switch: { request_id: 'r-7f21', status: 'applied', generation: 4, age_s: 0 },
      servers: [
        serverOf({
          pid: 5150,
          applied_generation: 3,
          last_switch: { status: 'failed', generation: 4, detail: 'gateway refused the bind' },
        }),
      ],
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();

    expect(chipModule.isPending()).toBe(false);
    // The record's own terminus says applied. Rendering that would report a
    // switch that did not land, so the client's reading wins here.
    expect(chipModule.getState()?.last_switch).toMatchObject({
      request_id: 'r-7f21',
      status: 'failed',
      reason: 'switch_failed',
      detail: 'Controls server pid 5150: gateway refused the bind',
      synthesized: true,
    });
  });

  test('the failure clears once the deployment moves on', async () => {
    await boot();
    chipModule.markPending('r-7f21', 'live', 4);
    served = viewOf({
      generation: 4,
      servers: [
        serverOf({
          pid: 5150,
          applied_generation: 3,
          last_switch: { status: 'failed', generation: 4 },
        }),
      ],
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();
    expect(chipModule.getState()?.last_switch?.synthesized).toBe(true);

    // Somebody switched again and it worked. The old failure is history.
    served = viewOf({ generation: 5, servers: [serverOf({ applied_generation: 5 })] });
    await chipModule.refetch();
    expect(chipModule.getState()?.last_switch).toBeNull();
  });

  test('a switch a refusal answered ends the wait on the terminus alone', async () => {
    await boot();
    chipModule.markPending('r-7f21', 'live', 4);
    // A refusal moves neither target nor generation — `generation: null` is
    // what says so — and no server will ever report it.
    served = viewOf({
      last_switch: {
        request_id: 'r-7f21',
        status: 'refused',
        reason: 'unreachable',
        generation: null,
        age_s: 0,
      },
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();
    expect(chipModule.isPending()).toBe(false);
    expect(chipModule.getState()?.last_switch?.status).toBe('refused');
  });

  test('a fleet stalled past the TTL keeps the expiry, terminus or no terminus', async () => {
    // The wait is on the FLEET, so the record's `applied` terminus is already
    // in view when the deadline passes. Retiring the expiry on that terminus
    // would flip the chip to ✓ switched while a server is still on the old
    // generation — the exact misreport this rule exists to prevent.
    await boot();
    chipModule.markPending('r-slow', 'live', 4);
    const stalled = () =>
      viewOf({
        generation: 4,
        last_switch: { request_id: 'r-slow', status: 'applied', generation: 4, age_s: 30 },
        servers: [serverOf({ applied_generation: 3 })],
      });
    served = stalled();
    vi.advanceTimersByTime(chipModule.REQUEST_TTL_S * 1000);
    await flush();
    expect(chipModule.isPending()).toBe(false);
    expect(chipModule.getState()?.last_switch).toMatchObject({
      request_id: 'r-slow',
      status: 'expired',
      synthesized: true,
    });

    // Another read, same lagging server: the expiry must still stand.
    served = stalled();
    await chipModule.refetch();
    expect(chipModule.getState()?.last_switch?.status).toBe('expired');

    // The rebuild finally finishes. NOW the terminus is the truth.
    served = viewOf({
      generation: 4,
      last_switch: { request_id: 'r-slow', status: 'applied', generation: 4, age_s: 40 },
      servers: [serverOf({ applied_generation: 4 })],
    });
    await chipModule.refetch();
    expect(chipModule.getState()?.last_switch?.status).toBe('applied');
  });

  test('a failure clears when the server retries the same generation and lands', async () => {
    // A controls server retries a failed launch on its next call, and
    // publish_last_switch replaces the block. Without this the chip would pin
    // `✗ not applied` on a converged fleet until somebody switched again.
    await boot();
    chipModule.markPending('r-7f21', 'live', 4);
    served = viewOf({
      generation: 4,
      servers: [
        serverOf({
          pid: 5150,
          applied_generation: 3,
          last_switch: { status: 'failed', generation: 4 },
        }),
      ],
    });
    vi.advanceTimersByTime(chipModule.FAST_POLL_MS);
    await flush();
    expect(chipModule.getState()?.last_switch?.status).toBe('failed');

    served = viewOf({
      generation: 4,
      servers: [
        serverOf({
          pid: 5150,
          applied_generation: 4,
          last_switch: { status: 'applied', generation: 4 },
        }),
      ],
    });
    await chipModule.refetch();
    expect(chipModule.getState()?.last_switch).toBeNull();
  });

  test('teardown releases both timers and the stream', async () => {
    await boot();
    chipModule.markPending('r-1');
    chipModule.teardownControlTargetChip();
    expect(FakeEventSource.opened[0].closed).toBe(true);
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
      control_target: 'live',
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    await chipModule.refetch();
    expect(shortText()).toBe('Real machine');

    // The parked read finally resolves, carrying history.
    heldGets.shift()?.();
    await slow;
    await flush();
    expect(shortText()).toBe('Real machine');
    expect(chipEl()?.dataset.targetKind).toBe('real');
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
    Object.assign(window, { __OSPREY_PREFIX__: '/u/alice' });
    await boot();
    const gets = fetchCalls.filter((c) => c.method === 'GET');
    expect(gets.length).toBeGreaterThan(0);
    for (const call of gets) {
      expect(call.url).toBe('/u/alice/api/terminal/posture');
    }
  });
});

/* ---- the API the popover consumes --------------------------------------- */

describe('popover API', () => {
  test('subscribers are called after each render and can unsubscribe', async () => {
    await boot();
    const seen = /** @type {any[]} */ ([]);
    const off = chipModule.subscribe((s) => seen.push(s));
    await chipModule.refetch();
    expect(seen).toHaveLength(1);
    expect(seen[0]?.control_target).toBe('standin');

    off();
    await chipModule.refetch();
    expect(seen).toHaveLength(1);
  });

  test('a throwing subscriber does not stop the others', async () => {
    await boot();
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const seen = /** @type {any[]} */ ([]);
    chipModule.subscribe(() => {
      throw new Error('boom');
    });
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
    chip.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(chip.getAttribute('aria-expanded')).toBe('false');
    expect(seen).toEqual([true, false]);
    // Status and action are separate gestures: opening changes nothing.
    await flush();
    expect(fetchCalls.filter((c) => c.method !== 'GET')).toHaveLength(0);
    expect(getCount()).toBe(before);
  });

  test('a refused request carries the sentence from all three refusal body shapes', async () => {
    /** @param {number} status @param {() => Promise<any>} json */
    const refuse = (status, json) =>
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({ ok: false, status, json }))
      );

    refuse(409, async () => ({ detail: { error: 'x', message: 'dict detail' } }));
    await expect(chipModule.targetRequest('/api/terminal/posture')).rejects.toThrow('dict detail');

    refuse(400, async () => ({ detail: 'string detail' }));
    await expect(chipModule.targetRequest('/api/terminal/posture')).rejects.toThrow(
      'string detail'
    );

    // A body that is not JSON at all: the status is all there is to say, and
    // the fallback names no session — one control target per deployment.
    refuse(503, async () => {
      throw new SyntaxError('not json');
    });
    await expect(chipModule.targetRequest('/api/terminal/posture')).rejects.toThrow(
      'Could not read the control target (HTTP 503).'
    );
  });

  test('a posture refusal carries its phrase, status and detail (C-POST)', async () => {
    const detail = { reason: 'not_taken', taken: false, phrase: 'The stop did not take effect.' };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 503, json: async () => ({ detail }) }))
    );
    const err = await chipModule
      .targetRequest('/api/terminal/posture', { method: 'POST', json: {} })
      .catch((e) => e);
    expect(err).toBeInstanceOf(chipModule.TargetRequestError);
    expect(err).toBeInstanceOf(Error);
    expect(err.message).toBe('The stop did not take effect.');
    expect(err.status).toBe(503);
    expect(err.detail).toEqual(detail);
  });

  test('a stamped 409 keeps the epoch it returned', async () => {
    const detail = { reason: 'stopped_since', epoch: '12', stamped: true };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 409, json: async () => ({ detail }) }))
    );
    const err = await chipModule.targetRequest('/api/terminal/posture').catch((e) => e);
    expect(err.status).toBe(409);
    expect(err.detail).toEqual(detail);
    // No phrase and no message: the status is all the sentence can say.
    expect(err.message).toBe('Could not read the control target (HTTP 409).');
  });

  test('a validation failure’s list detail reads as the generic failure', async () => {
    const detail = [{ loc: ['body', 'state'], msg: 'bad', type: 'literal_error' }];
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 422, json: async () => ({ detail }) }))
    );
    const err = await chipModule.targetRequest('/api/terminal/posture').catch((e) => e);
    expect(err.message).toBe('Could not read the control target (HTTP 422).');
    expect(err.detail).toEqual(detail);
  });

  test('a switch refusal says what the operator must not miss', () => {
    const { TargetRequestError, switchRefusal, STOPPED_SINCE_SENTENCE } = chipModule;
    /** @param {number} status @param {any} detail */
    const refused = (status, detail) => new TargetRequestError('x', status, detail);
    // The route's phrase wins, and leaves the confirm up.
    expect(switchRefusal(refused(403, { reason: 'tier_never', phrase: 'P.' }), 'asked')).toEqual({
      message: 'P.',
      close: false,
    });
    // A stop that overtook the gesture closes the dialog it asked from.
    expect(switchRefusal(refused(409, { reason: 'stopped_since', epoch: '3' }), 'asked')).toEqual({
      message: STOPPED_SINCE_SENTENCE,
      close: true,
    });
    // A stop that was not taken is the stop sentence even with no phrase.
    expect(switchRefusal(refused(503, null), 'off').message).toMatch(/writes are still on/);
    // A bare code reads as its phrase; anything else as the message.
    expect(switchRefusal(refused(400, { reason: 'bulk_on_refused' }), 'asked').message).toBe(
      'bulk_on_refused'
    );
    // The two refusals the route sends no phrase with read as the table's.
    expect(
      switchRefusal(refused(403, { reason: 'tier_caps', ceiling: 'asked' }), 'allowed').message
    ).toBe('your tier caps writes at asked');
    expect(switchRefusal(refused(400, { reason: 'waiver_not_allowed' }), 'allowed').message).toBe(
      'this move always confirms'
    );
    expect(switchRefusal(new Error('network down'), 'off').message).toBe('network down');
  });

  test('a body with no detail carries a null one', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 500,
        json: async () => {
          throw new SyntaxError('x');
        },
      }))
    );
    const err = await chipModule.targetRequest('/api/terminal/posture').catch((e) => e);
    expect(err.status).toBe(500);
    expect(err.detail).toBeNull();
  });
});

/* ---- the switch epoch on the wire ------------------------------------- */

describe('postSwitch', () => {
  /**
   * A nanosecond epoch with non-zero low digits: above 2**53, so as a JSON
   * number it would come back rounded (…148000 or …148128), and the route
   * would answer every turn-on 409 `stopped_since`.
   */
  const NS_EPOCH = '1793181592449148123';
  const STAMPED_EPOCH = '1793181592449148987';

  /** Answer each POST with the next `[status, bodyText]`, recording the raw request bodies. */
  function stubPosts(/** @type {[number, string][]} */ answers) {
    const sent = /** @type {string[]} */ ([]);
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ _url, /** @type {RequestInit} */ init) => {
        sent.push(String(init.body));
        const [status, text] = /** @type {[number, string]} */ (answers.shift());
        return { ok: status < 400, status, json: async () => JSON.parse(text) };
      })
    );
    return sent;
  }

  test('posts back the exact string it read', async () => {
    const sent = stubPosts([
      [200, `{"target":"live","state":"asked","switch_epoch":"${NS_EPOCH}"}`],
    ]);
    const answer = await chipModule.postSwitch(null, 'live', 'asked', NS_EPOCH);
    expect(sent).toHaveLength(1);
    expect(sent[0]).toContain(`"epoch":"${NS_EPOCH}"`);
    expect(answer.switch_epoch).toBe(NS_EPOCH);
  });

  test('a stamped 409 is retried with the exact string it carried', async () => {
    const sent = stubPosts([
      [409, `{"detail":{"reason":"stopped_since","epoch":"${STAMPED_EPOCH}","stamped":true}}`],
      [200, `{"target":"live","state":"allowed","switch_epoch":"${STAMPED_EPOCH}"}`],
    ]);
    await chipModule.postSwitch(null, 'live', 'allowed', null);
    expect(sent).toHaveLength(2);
    expect(JSON.parse(sent[0]).epoch).toBeNull();
    expect(sent[1]).toContain(`"epoch":"${STAMPED_EPOCH}"`);
  });

  test('sends the position and the waiver the route reads', async () => {
    const sent = stubPosts([
      [200, '{}'],
      [200, '{}'],
    ]);
    await chipModule.postSwitch('s-1', 'standin', 'asked', '7', true);
    expect(JSON.parse(sent[0])).toEqual({
      session_id: 's-1',
      target: 'standin',
      state: 'asked',
      epoch: '7',
      waived: true,
    });
    // No waiver named is `false`: the route's field is a strict boolean.
    await chipModule.postSwitch('s-1', 'standin', 'allowed', '7');
    expect(JSON.parse(sent[1]).state).toBe('allowed');
    expect(JSON.parse(sent[1]).waived).toBe(false);
  });

  test('the stamped retry keeps the waiver', async () => {
    const sent = stubPosts([
      [409, `{"detail":{"reason":"stopped_since","epoch":"${STAMPED_EPOCH}","stamped":true}}`],
      [200, '{}'],
    ]);
    await chipModule.postSwitch(null, 'standin', 'asked', null, true);
    expect(sent.map((body) => JSON.parse(body).waived)).toEqual([true, true]);
    expect(JSON.parse(sent[1]).state).toBe('asked');
  });

  test('a 409 that stamped nothing is thrown, never retried', async () => {
    const sent = stubPosts([[409, '{"detail":{"reason":"stopped_since","epoch":"3"}}']]);
    const err = await chipModule.postSwitch(null, 'live', 'asked', '1').catch((e) => e);
    expect(err.status).toBe(409);
    expect(sent).toHaveLength(1);
  });
});

/* ---- the active kind ---------------------------------------------------- */

describe('activeKind', () => {
  /** The CSS values the chip keys on, one per machine kind. */
  /** @type {Record<string, string>} */
  const ATTR = { live: 'real', standin: 'rehearsal', va: 'simulator' };

  test('answers null before anything has been read', () => {
    expect(chipModule.activeKind()).toBeNull();
  });

  test('answers null for a roster with no rows', async () => {
    await boot(viewOf({ targets: [] }));
    expect(chipModule.activeKind()).toBeNull();
  });

  for (const [kindName, kind] of Object.entries(KINDS)) {
    test(`answers ${kindName} for the row the chip speaks for`, async () => {
      await boot(
        viewOf({
          control_target: kind.target,
          targets: [rowOf({ ...kind, active: true, is_baseline: true })],
        })
      );
      expect(chipModule.activeKind()).toBe(ATTR[kindName]);
      // One derivation for the chip and for whoever phrases what a write means
      // on this machine: they cannot disagree about a stand-in.
      expect(chipModule.activeKind()).toBe(chipEl()?.dataset.targetKind);
    });
  }

  test('subscribers see the new kind once a switch has landed', async () => {
    await boot(viewOf({ targets: [rowOf({ ...KINDS.standin, active: true })] }));
    expect(chipModule.activeKind()).toBe('rehearsal');

    const seen = /** @type {(string|null)[]} */ ([]);
    const off = chipModule.subscribe(() => seen.push(chipModule.activeKind()));
    served = viewOf({
      control_target: KINDS.live.target,
      targets: [rowOf({ ...KINDS.live, active: true })],
    });
    await chipModule.refetch();
    expect(seen).toEqual(['real']);
    expect(chipModule.activeKind()).toBe('real');
    off();
  });
});
