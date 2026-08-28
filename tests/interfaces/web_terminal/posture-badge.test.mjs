// @ts-check
/**
 * Unit tests for the terminal card's session-posture badge (posture-badge.js):
 *   npx vitest run tests/interfaces/web_terminal/posture-badge.test.mjs
 *
 * The badge is the client half of the per-session runtime sandbox toggle. It
 * reads ONE truth — `GET /api/terminal/posture?session_id=` — and never
 * derives the posture from anything it did itself, so a posture set from
 * another tab (or persisted across a container restart) still shows here.
 *
 * What these tests pin down, in the order a reviewer would ask about it:
 *
 * - the badge renders the posture the server reports, in both states;
 * - BOTH toggle directions go through a confirm modal whose copy states the
 *   two consequences an operator must know before agreeing: the current turn
 *   is terminated, and the conversation history is preserved. There is no
 *   remembered acknowledgment — a sandbox toggle is never one silent click;
 * - the "writes" direction is not offered at all on a render whose config has
 *   writes off (`rendered_writes_enabled: false`), and the badge says why
 *   rather than failing at the server;
 * - the target line names the session's control target and whether THAT target
 *   is armed — `rendered_writes_enabled` is a union over every target, so on a
 *   mixed render it says nothing about the machine the operator is on — and it
 *   marks the value as the deployment baseline when no session target exists;
 * - the badge re-reads on a timer, so a target switch made mid-session (which
 *   fires no session change and no toggle) does not leave the line stale;
 * - a refused POST (403 writes-on-readonly-render, 409 unknown session id)
 *   surfaces the server's own detail in the still-open modal, and does NOT
 *   reconnect the terminal;
 * - a confirmed toggle reconnects with `startTerminal(sessionId, 'resume')` —
 *   the server terminated the session as part of storing the posture, so
 *   without this the card is left staring at a dead PTY.
 *
 * Seams: terminal.js is mocked (it owns the live PTY socket and the session
 * id) so `startTerminal`/`stopTerminal` are observable and the session-change
 * subscription can be fired by hand; `fetch` is stubbed the way the other
 * suites here stub it. Module-private state (the mounted badge, the last
 * payload) has no reset API, so each test gets a fresh module instance via
 * vi.resetModules() + dynamic import — same pattern as sessions.test.mjs.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SESSION = 'aaaaaaaa-1111-2222-3333-444444444444';

/** Mutable stand-in for terminal.js, reachable from the hoisted vi.mock factory. */
const term = vi.hoisted(() => ({
  /** @type {string|null} */
  sessionId: /** @type {string|null} */ (null),
  /** @type {((id: string) => void)[]} */
  listeners: [],
  /** @type {any} */
  startTerminal: null,
  /** @type {any} */
  stopTerminal: null,
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => term.sessionId,
  /** @param {(id: string) => void} fn */
  onSessionChange: (fn) => term.listeners.push(fn),
  /** @param {...any} args */
  startTerminal: (...args) => term.startTerminal(...args),
  /** @param {...any} args */
  stopTerminal: (...args) => term.stopTerminal(...args),
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/posture-badge.js')} */
let postureBadge;

/** What GET /api/terminal/posture answers; mutated by a successful POST. */
/** @type {{session_id: string, posture: string, rendered_writes_enabled: boolean,
 *           session_target: string|null, session_target_label?: string,
 *           target_writes_enabled: boolean, target_source: string}} */
let served = {
  session_id: SESSION,
  posture: 'writes',
  rendered_writes_enabled: true,
  session_target: 'live',
  target_writes_enabled: true,
  target_source: 'session',
};

/**
 * What the next POST /api/terminal/posture answers.
 *
 * The refusal fixtures below carry the routes' REAL body shape: FastAPI wraps
 * an HTTPException detail as `{detail: <detail>}`, and these routes raise a
 * DICT detail (`{error, message}` — routes/websocket.py). A fixture that
 * flattened that to a string would be testing a server that does not exist,
 * and would hide the fact that a naive `detail.detail` renders "[object
 * Object]" where the operator needs the sentence.
 */
/** @type {{ok: true} | {ok: false, status: number, detail?: any, body?: any}} */
let postOutcome = { ok: true };

/** @type {{url: string, method: string, body: any}[]} */
let fetchCalls = [];

/** Drain the microtask/timer queue the async handlers chain through. */
async function flush() {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * When true, the NEXT GET is parked in `heldGets` instead of resolving. Its
 * payload is snapshotted at call time — that is the whole point: a held read
 * carries the answer the server gave BEFORE whatever happened next.
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
      const body = init?.body ? JSON.parse(init.body) : null;
      fetchCalls.push({ url: String(url), method, body });
      if (method === 'GET') {
        if (holdNextGet) {
          holdNextGet = false;
          const parked = { ...served };
          return new Promise((resolve) => {
            heldGets.push(() =>
              resolve({ ok: true, status: 200, statusText: 'OK', json: async () => parked })
            );
          });
        }
        return { ok: true, status: 200, statusText: 'OK', json: async () => ({ ...served }) };
      }
      if (postOutcome.ok) {
        served = { ...served, posture: body.posture };
        return { ok: true, status: 200, statusText: 'OK', json: async () => ({ ...served }) };
      }
      const errorBody = postOutcome.body ?? { detail: postOutcome.detail };
      return {
        ok: false,
        status: postOutcome.status,
        statusText: 'Error',
        json: async () => errorBody,
      };
    })
  );
}

/** The terminal card header the badge mounts into (index.html's shape). */
function mountFixture() {
  document.body.innerHTML = `
    <div class="terminal-card">
      <div class="terminal-header">
        <span class="session-led" id="session-led"></span>
        <span class="terminal-label" id="terminal-label">Session</span>
        <div class="session-selector" id="session-selector"></div>
        <button class="new-session-btn" id="new-session-btn">+ New</button>
      </div>
    </div>`;
}

const badgeEl = () =>
  /** @type {HTMLButtonElement|null} */ (document.querySelector('.posture-badge'));
/**
 * The dialog that is UP. A dismissed one lingers in the DOM for its fade-out
 * (the shell's dialogs all animate; see settings.js) and is marked
 * `data-closing` the instant it is dismissed — that mark, not detachment, is
 * what "closed" means here. `removedOverlays()` covers the detachment.
 */
const overlayEl = () =>
  /** @type {HTMLElement|null} */ (
    document.querySelector('.posture-modal-overlay:not([data-closing])')
  );
const modalText = () => overlayEl()?.textContent ?? '';

/** Click the badge and let the (synchronous) modal build settle. */
async function clickBadge() {
  badgeEl()?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  await flush();
}

/** Click the modal's confirm button and let the POST round trip settle. */
async function confirmModal() {
  /** @type {HTMLButtonElement} */ (
    document.querySelector('.posture-modal-confirm')
  ).dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  await flush();
}

/**
 * Boot the badge for a given server state.
 * `session_target_label` is left ABSENT unless a case asks for one — that is
 * what an older server sends, and it is the fallback the line has to keep
 * working under.
 * @param {{posture?: string, rendered_writes_enabled?: boolean, sessionId?: string|null,
 *          session_target?: string|null, session_target_label?: string,
 *          target_writes_enabled?: boolean, target_source?: string}} [opts]
 */
async function boot(opts = {}) {
  served = {
    session_id: SESSION,
    posture: opts.posture ?? 'writes',
    rendered_writes_enabled: opts.rendered_writes_enabled ?? true,
    session_target: opts.session_target === undefined ? 'live' : opts.session_target,
    target_writes_enabled: opts.target_writes_enabled ?? true,
    target_source: opts.target_source ?? 'session',
  };
  if (opts.session_target_label !== undefined) {
    served.session_target_label = opts.session_target_label;
  }
  term.sessionId = opts.sessionId === undefined ? SESSION : opts.sessionId;
  postureBadge.initPostureBadge();
  await flush();
}

const targetEl = () =>
  /** @type {HTMLElement|null} */ (document.querySelector('.posture-badge-target'));

beforeEach(async () => {
  vi.resetModules();
  term.sessionId = SESSION;
  term.listeners = [];
  term.startTerminal = vi.fn();
  term.stopTerminal = vi.fn();
  postOutcome = { ok: true };
  stubFetch();
  mountFixture();
  postureBadge = await import(
    '../../../src/osprey/interfaces/web_terminal/static/js/posture-badge.js'
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('badge rendering', () => {
  test('renders the writes posture the server reports', async () => {
    await boot({ posture: 'writes' });

    const badge = badgeEl();
    expect(badge).not.toBeNull();
    expect(badge?.hidden).toBe(false);
    expect(badge?.dataset.posture).toBe('writes');
    expect(badge?.textContent?.toLowerCase()).toContain('writes');
  });

  test('renders the sandbox posture the server reports', async () => {
    await boot({ posture: 'sandbox' });

    const badge = badgeEl();
    expect(badge?.dataset.posture).toBe('sandbox');
    expect(badge?.textContent?.toLowerCase()).toContain('sandbox');
  });

  test('reads the posture for the current session id, with no cached guess', async () => {
    await boot({ posture: 'sandbox' });

    const gets = fetchCalls.filter((c) => c.method === 'GET');
    expect(gets).toHaveLength(1);
    expect(gets[0].url).toContain('/api/terminal/posture');
    expect(gets[0].url).toContain(`session_id=${SESSION}`);
  });

  test('stays hidden until a session id exists', async () => {
    await boot({ sessionId: null });

    expect(badgeEl()?.hidden).toBe(true);
    expect(fetchCalls.filter((c) => c.method === 'GET')).toHaveLength(0);
  });

  test('re-reads the posture when the terminal reports a new session', async () => {
    await boot({ sessionId: null });
    expect(term.listeners.length).toBeGreaterThan(0);

    term.sessionId = SESSION;
    served = { ...served, posture: 'sandbox' };
    for (const fn of term.listeners) fn(SESSION);
    await flush();

    expect(badgeEl()?.dataset.posture).toBe('sandbox');
    expect(badgeEl()?.hidden).toBe(false);
  });
});

describe('confirm modal', () => {
  test('the sandbox direction states turn termination and history preservation', async () => {
    await boot({ posture: 'writes' });
    await clickBadge();

    expect(overlayEl()).not.toBeNull();
    expect(modalText()).toMatch(/current turn is terminated/i);
    expect(modalText()).toMatch(/history is preserved/i);
    expect(modalText().toLowerCase()).toContain('sandbox');
  });

  test('the writes direction states turn termination and history preservation', async () => {
    await boot({ posture: 'sandbox' });
    await clickBadge();

    expect(overlayEl()).not.toBeNull();
    expect(modalText()).toMatch(/current turn is terminated/i);
    expect(modalText()).toMatch(/history is preserved/i);
    expect(modalText().toLowerCase()).toContain('write');
  });

  test('confirms every time — no remembered acknowledgment', async () => {
    await boot({ posture: 'writes' });

    await clickBadge();
    await confirmModal();
    expect(overlayEl()).toBeNull();

    // Second toggle (now sandbox -> writes) must put the dialog up again.
    await clickBadge();
    expect(overlayEl()).not.toBeNull();
  });

  test('cancel closes the modal and sends nothing', async () => {
    await boot({ posture: 'writes' });
    await clickBadge();

    /** @type {HTMLButtonElement} */ (
      document.querySelector('.posture-modal-cancel')
    ).dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await flush();

    expect(overlayEl()).toBeNull();
    expect(fetchCalls.filter((c) => c.method === 'POST')).toHaveLength(0);
    expect(term.startTerminal).not.toHaveBeenCalled();

    // …and the node itself goes once the fade is over — no overlay left
    // sitting invisibly over the page swallowing clicks.
    await new Promise((r) => setTimeout(r, 350));
    expect(document.querySelector('.posture-modal-overlay')).toBeNull();
  });
});

describe('a render with writes off', () => {
  test('does not offer the writes direction, and says why', async () => {
    await boot({ posture: 'sandbox', rendered_writes_enabled: false });

    const badge = badgeEl();
    expect(badge?.disabled).toBe(true);
    expect(badge?.dataset.writesAvailable).toBe('false');
    // The reason is on the badge itself, not left to a failed POST.
    expect(`${badge?.title} ${badge?.getAttribute('aria-label')}`).toMatch(/writes/i);
  });

  test('clicking the disabled badge opens no modal and sends nothing', async () => {
    await boot({ posture: 'sandbox', rendered_writes_enabled: false });
    await clickBadge();

    expect(overlayEl()).toBeNull();
    expect(fetchCalls.filter((c) => c.method === 'POST')).toHaveLength(0);
  });

  test('the sandbox direction is still offered when the render has writes', async () => {
    await boot({ posture: 'writes', rendered_writes_enabled: true });

    expect(badgeEl()?.disabled).toBe(false);
    expect(badgeEl()?.dataset.writesAvailable).toBe('true');
  });
});

describe('the control-target line', () => {
  test('names the session target and says it is armed', async () => {
    await boot({ session_target: 'va', target_writes_enabled: true, target_source: 'session' });

    const line = targetEl();
    expect(line).not.toBeNull();
    expect(line?.hidden).toBe(false);
    expect(line?.textContent).toContain('va');
    expect(line?.textContent?.toLowerCase()).toMatch(/\barmed\b/);
    expect(line?.textContent?.toLowerCase()).not.toMatch(/not armed/);
    expect(badgeEl()?.dataset.targetArmed).toBe('true');
  });

  test('makes an unarmed target unmistakable on a mixed render', async () => {
    // The motivating bug: the render arms SOME target (the VA), so
    // `rendered_writes_enabled` is true, while the target this session is on
    // refuses every write.
    await boot({
      rendered_writes_enabled: true,
      session_target: 'live',
      target_writes_enabled: false,
      target_source: 'session',
    });

    const line = targetEl();
    expect(line?.textContent).toContain('live');
    expect(line?.textContent?.toLowerCase()).toContain('not armed');
    expect(badgeEl()?.dataset.targetArmed).toBe('false');
    // …and the reason is spelled out where a screen reader reaches it.
    expect(badgeEl()?.getAttribute('aria-label')?.toLowerCase()).toContain('live');
  });

  test('marks a baseline fallback as the deployment default', async () => {
    await boot({ session_target: 'live', target_writes_enabled: false, target_source: 'baseline' });

    expect(targetEl()?.textContent?.toLowerCase()).toContain('baseline');
    expect(badgeEl()?.dataset.targetSource).toBe('baseline');
  });

  test('the baseline tooltip states what is known, not why', async () => {
    // `baseline` also covers "a target WAS published but could not be
    // resolved" — a foreign owner, two ambiguous records, a dead controls
    // server, a process table that could not be read. Telling an operator who
    // has actually switched that nothing has been published is the same
    // confident-but-wrong reading this line exists to end.
    await boot({ session_target: 'va', target_writes_enabled: true, target_source: 'baseline' });

    const title = `${badgeEl()?.title} ${badgeEl()?.getAttribute('aria-label')}`.toLowerCase();
    expect(title).toContain('no control target has been resolved');
    expect(title).not.toContain('published');
  });

  test('does not call a session target a baseline', async () => {
    await boot({ session_target: 'va', target_source: 'session' });

    expect(targetEl()?.textContent?.toLowerCase()).not.toContain('baseline');
    expect(badgeEl()?.dataset.targetSource).toBe('session');
  });

  test('stays out of the way when the server reports no target', async () => {
    // An older server, or one that could not read the render at all: say
    // nothing rather than inventing a target.
    await boot({ session_target: null });

    expect(targetEl()?.hidden).toBe(true);
    expect(badgeEl()?.hidden).toBe(false);
  });

  test('the writes/sandbox direction stays keyed on the render, not the target', async () => {
    // An unarmed CURRENT target does not disable the toggle: the operator may
    // legitimately step out of the sandbox before switching to the target they
    // intend to write on. Only `rendered_writes_enabled` gates the button.
    await boot({
      posture: 'sandbox',
      rendered_writes_enabled: true,
      session_target: 'live',
      target_writes_enabled: false,
    });

    expect(badgeEl()?.disabled).toBe(false);
    await clickBadge();
    expect(overlayEl()).not.toBeNull();
  });

  // ── what the target is CALLED ──────────────────────────────────────────
  //
  // `live` is the target's key, not its identity. A deployment running a
  // stand-in for its real machine publishes a label that says so, and the
  // badge shows the published one — it never works a name out for itself.

  test('names the target by the label the server published', async () => {
    await boot({
      session_target: 'live',
      session_target_label: 'LIVE MACHINE (stand-in)',
      target_source: 'session',
    });

    expect(targetEl()?.textContent).toContain('LIVE MACHINE (stand-in)');
    // The tooltip and the accessible name say the same word as the chip.
    const spoken = `${badgeEl()?.title} ${badgeEl()?.getAttribute('aria-label')}`;
    expect(spoken).toContain('LIVE MACHINE (stand-in)');
  });

  test('keeps the raw target name as the state key it styles from', async () => {
    // The label is display text and free to change; `data-target` is what the
    // stylesheet and the rest of the system key off, so it stays the bare name.
    await boot({ session_target: 'live', session_target_label: 'LIVE MACHINE (stand-in)' });

    expect(badgeEl()?.dataset.target).toBe('live');
  });

  test('falls back to the target name when the server sends no label', async () => {
    // An older server sends no label at all. Show the name it did send rather
    // than inventing one here — a second opinion about identity is the bug.
    await boot({ session_target: 'va', target_source: 'session' });

    expect(targetEl()?.textContent).toContain('va');
    expect(targetEl()?.textContent?.toLowerCase()).toContain('armed');
  });

  test('an empty label reads as no label', async () => {
    await boot({ session_target: 'va', session_target_label: '' });

    expect(targetEl()?.textContent).toContain('va');
  });

  test('a labelled baseline is still marked as the deployment default', async () => {
    await boot({
      session_target: 'live',
      session_target_label: 'LIVE MACHINE (stand-in)',
      target_source: 'baseline',
    });

    expect(targetEl()?.textContent).toContain('LIVE MACHINE (stand-in)');
    expect(targetEl()?.textContent?.toLowerCase()).toContain('baseline');
  });

  test('the label does not move the writes/sandbox direction', async () => {
    await boot({
      posture: 'sandbox',
      session_target: 'live',
      session_target_label: 'LIVE MACHINE (stand-in)',
    });

    expect(badgeEl()?.disabled).toBe(false);
    await clickBadge();
    expect(overlayEl()).not.toBeNull();
  });
});

describe('the colour state the badge hands the stylesheet', () => {
  // No colour name lives in this module: the (target, posture) pair travels as
  // data attributes and terminal.css maps it — grey for a sandboxed simulator,
  // green for one being driven, amber for a sandboxed live machine, red for a
  // live machine this session can write to. Asserting the attributes rather
  // than computed colours keeps the test about the contract between the two.
  test.each([
    ['va', 'writes'],
    ['va', 'sandbox'],
    ['live', 'writes'],
    ['live', 'sandbox'],
  ])('publishes (%s, %s) for the stylesheet to map', async (target, posture) => {
    await boot({
      posture,
      session_target: target,
      session_target_label: target === 'live' ? 'LIVE MACHINE (stand-in)' : 'sim',
    });

    expect(badgeEl()?.dataset.target).toBe(target);
    expect(badgeEl()?.dataset.posture).toBe(posture);
  });

  test('drops the target attribute when there is no target to colour', async () => {
    // Nothing published and nothing derived: the badge falls back to the plain
    // posture colours rather than styling a target it does not know.
    await boot({ session_target: null });

    expect(badgeEl()?.dataset.target).toBeUndefined();
    expect(badgeEl()?.dataset.posture).toBe('writes');
  });
});

describe('staying current through a mid-session target switch', () => {
  test('re-reads on a timer and repaints the target line', async () => {
    // A control-target switch fires neither a session change nor a toggle, so
    // without this poll the line would show the old target until the operator
    // happened to do something else.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      await boot({ session_target: 'live', target_writes_enabled: false });
      expect(targetEl()?.textContent).toContain('live');

      served = { ...served, session_target: 'va', target_writes_enabled: true };
      vi.advanceTimersByTime(30_000);
      await flush();

      expect(targetEl()?.textContent).toContain('va');
      expect(badgeEl()?.dataset.targetArmed).toBe('true');
    } finally {
      vi.useRealTimers();
    }
  });

  test('polls nothing while no session is attached', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      await boot({ sessionId: null });
      vi.advanceTimersByTime(30_000);
      await flush();

      expect(fetchCalls.filter((c) => c.method === 'GET')).toHaveLength(0);
    } finally {
      vi.useRealTimers();
    }
  });

  test('a slow tick cannot repaint the badge with the pre-toggle posture', async () => {
    // The race the poll introduces: a tick's GET leaves at t0 carrying
    // `writes`; the operator sandboxes the session at t1; the POST's own
    // re-read lands at t2; the t0 GET resolves at t3 > t2. Without a read
    // token it overwrites `state` and the badge reads `Writes` on a session
    // that is sandboxed — a wrong SAFETY readout on the surface this whole
    // change is making authoritative.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      await boot({ posture: 'writes' });

      // t0 — a tick fires and its GET is parked, holding `posture: writes`.
      holdNextGet = true;
      vi.advanceTimersByTime(30_000);
      await flush();
      expect(heldGets).toHaveLength(1);

      // t1/t2 — the operator sandboxes the session; the POST and its re-read
      // both complete while the tick's GET is still in flight.
      await clickBadge();
      await confirmModal();
      expect(badgeEl()?.dataset.posture).toBe('sandbox');

      // t3 — the stale answer finally arrives.
      heldGets.shift()?.();
      await flush();

      expect(badgeEl()?.dataset.posture).toBe('sandbox');
      expect(badgeEl()?.textContent?.toLowerCase()).toContain('sandbox');
    } finally {
      vi.useRealTimers();
    }
  });

  test('a stale target answer cannot overwrite a newer one either', async () => {
    // Same guard, the other payload: the target line is the fact this task
    // added, and it must not flip back to the target the operator left.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      await boot({ session_target: 'live', target_writes_enabled: false });

      holdNextGet = true;
      vi.advanceTimersByTime(30_000);
      await flush();

      served = { ...served, session_target: 'va', target_writes_enabled: true };
      vi.advanceTimersByTime(30_000);
      await flush();
      expect(targetEl()?.textContent).toContain('va');

      heldGets.shift()?.();
      await flush();

      expect(targetEl()?.textContent).toContain('va');
      expect(badgeEl()?.dataset.targetArmed).toBe('true');
    } finally {
      vi.useRealTimers();
    }
  });

  test('no tick fires while a confirm dialog is up', async () => {
    // The operator is mid-decision; a read started now can only race the
    // toggle's own re-read.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      await boot({ posture: 'writes' });
      await clickBadge();
      expect(overlayEl()).not.toBeNull();

      const before = fetchCalls.filter((c) => c.method === 'GET').length;
      vi.advanceTimersByTime(30_000);
      await flush();

      expect(fetchCalls.filter((c) => c.method === 'GET').length).toBe(before);
    } finally {
      vi.useRealTimers();
    }
  });

  test('stops polling once the badge leaves the page', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    try {
      await boot();
      document.body.innerHTML = '';
      const before = fetchCalls.length;
      vi.advanceTimersByTime(30_000);
      await flush();

      expect(fetchCalls.length).toBe(before);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('a refused toggle', () => {
  test('surfaces the 409 message and leaves the terminal alone', async () => {
    postOutcome = {
      ok: false,
      status: 409,
      detail: {
        error: 'session_not_started',
        message: 'This session has not started yet — send one prompt first, then set its posture. A chat session becomes addressable again on its next prompt.',
      },
    };
    await boot({ posture: 'writes' });
    await clickBadge();
    await confirmModal();

    const error = document.querySelector('.posture-modal-error');
    expect(overlayEl()).not.toBeNull(); // stays open so the message is readable
    expect(error?.textContent).toMatch(/send one prompt first/i);
    // The whole point of the dict-detail unwrap: never the stringified object.
    expect(error?.textContent).not.toContain('[object Object]');
    expect(term.startTerminal).not.toHaveBeenCalled();
    expect(term.stopTerminal).not.toHaveBeenCalled();
  });

  test('surfaces the 403 message and leaves the terminal alone', async () => {
    postOutcome = {
      ok: false,
      status: 403,
      detail: {
        error: 'writes_disabled',
        message:
          'This deployment is rendered with control_system.writes_enabled off; ' +
          'no session can step out of the sandbox.',
      },
    };
    await boot({ posture: 'sandbox' });
    await clickBadge();
    await confirmModal();

    const error = document.querySelector('.posture-modal-error');
    expect(error?.textContent).toMatch(/no session can step out of the sandbox/i);
    expect(error?.textContent).not.toContain('[object Object]');
    expect(term.startTerminal).not.toHaveBeenCalled();
  });

  test('surfaces the 400 message for a malformed session id', async () => {
    postOutcome = {
      ok: false,
      status: 400,
      detail: { error: 'invalid_session_id', message: 'session_id must be a Claude session UUID.' },
    };
    await boot({ posture: 'writes' });
    await clickBadge();
    await confirmModal();

    const error = document.querySelector('.posture-modal-error');
    expect(error?.textContent).toMatch(/must be a Claude session UUID/i);
    expect(term.startTerminal).not.toHaveBeenCalled();
  });

  test('falls back to the status when the body carries no usable message', async () => {
    // e.g. a proxy answering for the app: JSON, but nothing to read.
    postOutcome = { ok: false, status: 502, body: {} };
    await boot({ posture: 'writes' });
    await clickBadge();
    await confirmModal();

    const error = document.querySelector('.posture-modal-error');
    expect(error?.textContent).toMatch(/502/);
    expect(error?.textContent).not.toContain('[object Object]');
    expect(term.startTerminal).not.toHaveBeenCalled();
  });

  test('a string detail still reads through, for routes that send one', async () => {
    postOutcome = { ok: false, status: 409, detail: 'send one prompt first' };
    await boot({ posture: 'writes' });
    await clickBadge();
    await confirmModal();

    expect(document.querySelector('.posture-modal-error')?.textContent).toMatch(
      /send one prompt first/i
    );
  });
});

describe('a confirmed toggle', () => {
  test('posts the target posture for the current session', async () => {
    await boot({ posture: 'writes' });
    await clickBadge();
    await confirmModal();

    const posts = fetchCalls.filter((c) => c.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(posts[0].url).toContain('/api/terminal/posture');
    expect(posts[0].body).toEqual({ session_id: SESSION, posture: 'sandbox' });
  });

  test('reconnects the terminated session with startTerminal(sessionId, "resume")', async () => {
    await boot({ posture: 'writes' });
    await clickBadge();
    await confirmModal();

    expect(term.startTerminal).toHaveBeenCalledTimes(1);
    expect(term.startTerminal).toHaveBeenCalledWith(SESSION, 'resume');
    // The dead socket has to go first — startTerminal() no-ops on a live one.
    expect(term.stopTerminal).toHaveBeenCalled();
  });

  test('re-reads the posture from the server rather than assuming it', async () => {
    await boot({ posture: 'writes' });
    const getsBefore = fetchCalls.filter((c) => c.method === 'GET').length;
    await clickBadge();
    await confirmModal();

    expect(fetchCalls.filter((c) => c.method === 'GET').length).toBeGreaterThan(getsBefore);
    expect(badgeEl()?.dataset.posture).toBe('sandbox');
    expect(badgeEl()?.textContent?.toLowerCase()).toContain('sandbox');
  });

  test('toggles back to writes from sandbox', async () => {
    await boot({ posture: 'sandbox' });
    await clickBadge();
    await confirmModal();

    const posts = fetchCalls.filter((c) => c.method === 'POST');
    expect(posts[0].body).toEqual({ session_id: SESSION, posture: 'writes' });
    expect(badgeEl()?.dataset.posture).toBe('writes');
  });
});
