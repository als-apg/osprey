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
let served = {
  session_id: SESSION,
  posture: 'writes',
  rendered_writes_enabled: true,
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

function stubFetch() {
  fetchCalls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
      const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(init.body) : null;
      fetchCalls.push({ url: String(url), method, body });
      if (method === 'GET') {
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
 * @param {{posture?: string, rendered_writes_enabled?: boolean, sessionId?: string|null}} [opts]
 */
async function boot(opts = {}) {
  served = {
    session_id: SESSION,
    posture: opts.posture ?? 'writes',
    rendered_writes_enabled: opts.rendered_writes_enabled ?? true,
  };
  term.sessionId = opts.sessionId === undefined ? SESSION : opts.sessionId;
  postureBadge.initPostureBadge();
  await flush();
}

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
