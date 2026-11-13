// @ts-check
/**
 * Unit tests for the client session model (session-model.js):
 *   npx vitest run tests/interfaces/web_terminal/session-model.test.mjs
 *
 * The pointer is the single source of truth; the model hears every key change
 * from any path and PUTs last-active once, clears the pick, and synthesizes a
 * row for a key the listing has not caught up with. `switchTo` is the guarded
 * gesture entry: it asks before cutting a streaming chat turn, moves to a
 * current row's key, and enters any other transcript for a minted key.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import {
  PROMPT_TITLE_CHARS,
  TITLE_CHASE_MS,
  shortenedPrompt,
} from '../../../src/osprey/interfaces/web_terminal/static/js/session-title-chase.js';

const mocks = vi.hoisted(() => ({
  streaming: false,
  terminal: false,
  stopResult: true,
  stopTurn: vi.fn(),
  newConversation: vi.fn(),
  startNewSession: vi.fn(async () => {}),
  resumeSession: vi.fn(),
  showConfirm: vi.fn(),
  relayTranscriptMove: vi.fn(),
  /** @type {((data: any) => void)[]} */
  hubHandlers: [],
  /** @type {Record<string, ((detail: any) => void)[]>} */
  chatHandlers: {},
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat.js', () => ({
  isStreaming: () => mocks.streaming,
  stopTurn: mocks.stopTurn,
  newConversation: mocks.newConversation,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js', () => ({
  startNewSession: mocks.startNewSession,
  resumeSession: mocks.resumeSession,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => ({
  showConfirm: mocks.showConfirm,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js', () => ({
  isTerminalSurface: () => mocks.terminal,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js', () => ({
  onChat: (/** @type {string} */ name, /** @type {(d: any) => void} */ fn) => {
    (mocks.chatHandlers[name] ??= []).push(fn);
    return () => {};
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', () => ({
  relayTranscriptMove: mocks.relayTranscriptMove,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', async (orig) => ({
  ...(await orig()),
  createEventSource: (/** @type {string} */ _url, /** @type {any} */ { onMessage }) => {
    mocks.hubHandlers.push(onMessage);
    return { stop() {} };
  },
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js')} */
let model;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js')} */
let pointer;

const K1 = '11111111-1111-4111-8111-111111111111';
const K2 = '22222222-2222-4222-8222-222222222222';
const KNEW = '99999999-9999-4999-8999-999999999999';
const T1 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const T2 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const T3 = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';
const MINTED = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd';

/** @param {string} tid @param {Partial<any>} [extra] */
const row = (tid, extra = {}) => ({
  transcript_id: tid,
  session_key: null,
  current_for_key: false,
  title: `Session ${tid.slice(0, 4)}`,
  title_source: 'auto',
  last_artifact_id: null,
  ...extra,
});

/** @type {Record<string, any>} */
let pages;
/** @type {{url: string, method: string, body: any}[]} */
let calls;

beforeEach(async () => {
  vi.resetModules();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
  localStorage.clear();
  mocks.streaming = false;
  mocks.terminal = false;
  mocks.stopResult = true;
  mocks.hubHandlers = [];
  mocks.chatHandlers = {};
  // A stop that works ends the turn: the console goes idle and says so.
  mocks.stopTurn.mockReset().mockImplementation(async () => {
    if (!mocks.stopResult) return;
    mocks.streaming = false;
    for (const fn of mocks.chatHandlers.busy ?? []) fn({ busy: false });
  });
  mocks.newConversation.mockReset().mockImplementation(() => pointer.setPointer(KNEW));
  mocks.startNewSession.mockReset();
  mocks.showConfirm.mockReset();
  mocks.resumeSession.mockReset().mockImplementation(async (key) => {
    if (!mocks.terminal) pointer.setPointer(key);
  });
  pages = {
    first: {
      sessions: [
        row(T1, { session_key: K1, current_for_key: true, last_artifact_id: 'art-1' }),
        row(T2),
      ],
      more: true,
      next_cursor: 'c2',
      record: 'complete',
    },
    c2: {
      sessions: [row(T3, { session_key: K2, current_for_key: true })],
      more: false,
      next_cursor: null,
    },
  };
  calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url, /** @type {any} */ init = {}) => {
      const method = init.method ?? 'GET';
      calls.push({ url, method, body: init.body ? JSON.parse(init.body) : null });
      const parsed = new URL(url, 'http://hub');
      let body = {};
      if (parsed.pathname === '/api/sessions') {
        body = pages[parsed.searchParams.get('cursor') ?? 'first'];
      } else if (parsed.pathname.endsWith('/enter')) {
        body = { session_key: MINTED };
      } else if (parsed.pathname.endsWith('/pin')) {
        body = { pinned: JSON.parse(init.body).pinned ? '2026-11-06T11:00:00+00:00' : null };
      }
      return new Response(JSON.stringify(body), { status: 200 });
    })
  );
  pointer =
    await import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js');
  model = await import('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js');
});

afterEach(() => {
  model.resetSessionModel();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const lastActivePuts = () =>
  calls.filter((c) => c.method === 'PUT' && c.url.endsWith('/api/sessions/last-active'));
const listings = () => calls.filter((c) => c.method === 'GET' && c.url.includes('/api/sessions'));

async function boot(key = K1) {
  pointer.setPointer(key);
  await model.initSessionModel();
  await vi.advanceTimersByTimeAsync(400);
  calls.length = 0;
}

describe('the pointer subscription', () => {
  test.each([
    ['the pen button', () => mocks.newConversation()],
    ['the terminal "+ New"', () => pointer.setPointer(KNEW)],
    ['a session_switched', () => pointer.setPointer(KNEW)],
  ])('%s PUTs last-active once and clears the pick', async (_name, change) => {
    await boot();
    model.setPick('art-1');
    change();
    expect(model.pick()).toBeNull();
    await vi.advanceTimersByTimeAsync(model.LAST_ACTIVE_DEBOUNCE_MS + 10);
    expect(lastActivePuts().map((c) => c.body)).toEqual([{ session_key: KNEW }]);
  });

  test('a burst of key changes writes last-active once, for the last key', async () => {
    await boot();
    pointer.setPointer(K2);
    pointer.setPointer(KNEW);
    await vi.advanceTimersByTimeAsync(model.LAST_ACTIVE_DEBOUNCE_MS + 10);
    expect(lastActivePuts().map((c) => c.body)).toEqual([{ session_key: KNEW }]);
  });

  test('a new key gets a synthesized "New session" row until the listing has it', async () => {
    await boot();
    pointer.setPointer(KNEW);
    const active = model.activeRow();
    expect(active).toMatchObject({ session_key: KNEW, title: 'New session', synthesized: true });
    expect(model.rows()[0].session_key).toBe(KNEW);
    expect(model.rows()).toHaveLength(3);
  });

  test('a listed key is the active row', async () => {
    await boot();
    expect(model.activeRow()?.transcript_id).toBe(T1);
    expect(model.rows()).toHaveLength(2);
  });
});

describe('switchTo', () => {
  test('the active row is a no-op', async () => {
    await boot();
    const moved = await model.switchTo(model.rows()[0], { reason: 'gesture' });
    expect(moved).toBe(false);
    expect(mocks.resumeSession).not.toHaveBeenCalled();
    expect(mocks.showConfirm).not.toHaveBeenCalled();
  });

  test('a current row moves the pointer to its key', async () => {
    await boot();
    await model.loadMore();
    const target = model.rows().find((r) => r.transcript_id === T3);
    expect(await model.switchTo(/** @type {any} */ (target), { reason: 'gesture' })).toBe(true);
    expect(mocks.resumeSession).toHaveBeenCalledWith(K2);
    expect(pointer.getPointer()).toBe(K2);
    expect(calls.some((c) => c.url.endsWith('/enter'))).toBe(false);
  });

  test("a non-current row is entered and the pointer goes to the minted key K'", async () => {
    await boot();
    const target = model.rows().find((r) => r.transcript_id === T2);
    expect(await model.switchTo(/** @type {any} */ (target), { reason: 'gesture' })).toBe(true);
    const enter = calls.find((c) => c.url.endsWith(`/api/sessions/${T2}/enter`));
    expect(enter?.method).toBe('POST');
    expect(pointer.getPointer()).toBe(MINTED);
  });

  describe('while the chat streams', () => {
    beforeEach(() => {
      mocks.streaming = true;
    });

    test('Cancel stays put', async () => {
      await boot();
      mocks.showConfirm.mockImplementation((spec) => spec.onDismiss());
      const target = model.rows()[1];
      expect(await model.switchTo(target, { reason: 'gesture' })).toBe(false);
      expect(mocks.showConfirm).toHaveBeenCalledOnce();
      const spec = mocks.showConfirm.mock.calls[0][0];
      expect(spec.focus).toBe('cancel');
      expect(spec.confirmLabel).toBe('Stop & switch');
      expect(JSON.stringify(spec.body)).toContain('The Agent is answering');
      expect(mocks.stopTurn).not.toHaveBeenCalled();
      expect(pointer.getPointer()).toBe(K1);
    });

    test('Stop & switch stops the turn, then switches', async () => {
      await boot();
      const ui = { error: document.createElement('div'), confirm: null, done: vi.fn() };
      mocks.showConfirm.mockImplementation((spec) => spec.onConfirm(ui));
      expect(await model.switchTo(model.rows()[1], { reason: 'gesture' })).toBe(true);
      expect(mocks.stopTurn).toHaveBeenCalledOnce();
      expect(ui.done).toHaveBeenCalled();
      expect(pointer.getPointer()).toBe(MINTED);
    });

    test('a stop that times out leaves the pointer unchanged, with a notice', async () => {
      await boot();
      mocks.stopResult = false;
      const ui = { error: document.createElement('div'), confirm: null, done: vi.fn() };
      ui.error.hidden = true;
      mocks.showConfirm.mockImplementation((spec) => spec.onConfirm(ui));
      const moving = model.switchTo(model.rows()[1], { reason: 'gesture' });
      await vi.advanceTimersByTimeAsync(model.STOP_TIMEOUT_MS - 100);
      expect(ui.error.hidden).toBe(true);
      await vi.advanceTimersByTimeAsync(200);
      expect(await moving).toBe(false);
      expect(ui.error.hidden).toBe(false);
      expect(ui.error.textContent).toMatch(/did not stop/);
      expect(pointer.getPointer()).toBe(K1);
    });

    test('New session asks too, and a cancel tells nobody `new`', async () => {
      await boot();
      const seen = vi.fn();
      model.subscribe(seen);
      mocks.showConfirm.mockImplementation((spec) => spec.onDismiss());
      expect(await model.switchTo({ new: true }, { reason: 'gesture' })).toBe(false);
      expect(mocks.showConfirm).toHaveBeenCalledOnce();
      expect(mocks.newConversation).not.toHaveBeenCalled();
      expect(seen).not.toHaveBeenCalledWith({ reason: 'new' });
    });
  });

  test('transitioning only (not streaming) switches without a dialog', async () => {
    await boot();
    mocks.streaming = false;
    expect(await model.switchTo(model.rows()[1], { reason: 'gesture' })).toBe(true);
    expect(mocks.showConfirm).not.toHaveBeenCalled();
  });

  test('the terminal surface switches without a dialog, and the pick waits for the socket', async () => {
    await boot();
    mocks.streaming = true;
    mocks.terminal = true;
    await model.loadMore();
    const target = /** @type {any} */ (model.rows().find((r) => r.transcript_id === T3));
    target.last_artifact_id = 'art-3';
    expect(await model.switchTo(target, { reason: 'gesture' })).toBe(true);
    expect(mocks.showConfirm).not.toHaveBeenCalled();
    expect(pointer.getPointer()).toBe(K1);
    pointer.setPointer(K2); // session_switched lands
    expect(model.pick()).toBe('art-3');
  });

  test('New session on the terminal starts a new terminal session', async () => {
    await boot();
    mocks.terminal = true;
    expect(await model.switchTo({ new: true }, { reason: 'gesture' })).toBe(true);
    expect(mocks.startNewSession).toHaveBeenCalledOnce();
  });

  test.each([
    ['chat', false],
    ['terminal', true],
  ])(
    'New session on the %s tells listeners `new` once started, then `switched`',
    async (_, terminal) => {
      await boot();
      mocks.terminal = terminal;
      const start = terminal ? mocks.startNewSession : mocks.newConversation;
      /** @type {string[]} */
      const seen = [];
      model.subscribe(({ reason }) => {
        if (reason === 'new' || reason === 'switched')
          seen.push(`${reason}:${start.mock.calls.length}`);
      });
      expect(await model.switchTo({ new: true }, { reason: 'gesture' })).toBe(true);
      expect(seen).toEqual(['new:1', 'switched:1']);
    }
  );

  test('a switch to a listed session is not `new`', async () => {
    await boot();
    const seen = vi.fn();
    model.subscribe(seen);
    expect(await model.switchTo(model.rows()[1], { reason: 'gesture' })).toBe(true);
    expect(seen).toHaveBeenCalledWith({ reason: 'switched' });
    expect(seen).not.toHaveBeenCalledWith({ reason: 'new' });
  });
});

describe('the pick', () => {
  test('artifactId beats last_artifact_id', async () => {
    await boot(K2);
    const byId = (/** @type {string} */ id) =>
      /** @type {any} */ (model.rows().find((r) => r.transcript_id === id));
    expect(await model.switchTo(byId(T1), { artifactId: 'chosen', reason: 'gesture' })).toBe(true);
    expect(model.pick()).toBe('chosen');
  });

  test('last_artifact_id is next; with neither the pick is empty (the sidebar shows the newest)', async () => {
    await boot(K2);
    const byId = (/** @type {string} */ id) =>
      /** @type {any} */ (model.rows().find((r) => r.transcript_id === id));
    expect(await model.switchTo(byId(T1), { reason: 'gesture' })).toBe(true);
    expect(model.pick()).toBe('art-1');
    expect(await model.switchTo(byId(T2), { reason: 'gesture' })).toBe(true);
    expect(model.pick()).toBeNull();
  });

  test("a key change never carries the previous session's pick", async () => {
    await boot();
    model.setPick('old-session-artifact');
    pointer.setPointer(K2);
    expect(model.pick()).toBeNull();
  });
});

describe('listing and refresh', () => {
  test('only the first listing of a page drains', async () => {
    pointer.setPointer(K1);
    await model.initSessionModel();
    expect(listings().map((c) => c.url)).toEqual(['/api/sessions?drain=1']);
    calls.length = 0;
    await model.loadMore();
    expect(listings().map((c) => c.url)).toEqual(['/api/sessions?cursor=c2&drain=1']);
    calls.length = 0;
    model.refresh();
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(listings().map((c) => c.url)).toEqual(['/api/sessions']);
  });

  test('refresh merges page 1 and keeps the later pages', async () => {
    await boot();
    pages.c2 = { sessions: [row(T3)], more: true, next_cursor: 'c3' };
    pages.c3 = { sessions: [row(MINTED)], more: false, next_cursor: null };
    await model.loadMore();
    await model.loadMore();
    expect(model.rows().map((r) => r.transcript_id)).toEqual([T1, T2, T3, MINTED]);
    pages.first = {
      ...pages.first,
      sessions: [row(T1, { title: 'Renamed', session_key: K1, current_for_key: true }), row(T2)],
    };
    model.refresh();
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(model.rows().map((r) => r.transcript_id)).toEqual([T1, T2, T3, MINTED]);
    expect(model.rows()[0].title).toBe('Renamed');
    expect(model.hasMore()).toBe(false);
  });

  test('refresh is throttled', async () => {
    await boot();
    model.refresh();
    model.refresh();
    model.refresh();
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(listings()).toHaveLength(1);
  });

  test('chat streaming → idle refreshes', async () => {
    await boot();
    for (const fn of mocks.chatHandlers['turn-end'] ?? []) fn({});
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(listings()).toHaveLength(1);
  });

  test('session_transcript for a known key refreshes; for an unknown key it does not', async () => {
    await boot();
    for (const fn of mocks.hubHandlers) fn({ type: 'session_transcript', session_key: 'nobody' });
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(listings()).toHaveLength(0);
    for (const fn of mocks.hubHandlers) fn({ type: 'session_transcript', session_key: K1 });
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(listings()).toHaveLength(1);
  });

  test('session_transcript is relayed to the panels by its key, known or not; other frames are not', async () => {
    await boot();
    mocks.relayTranscriptMove.mockClear();
    for (const fn of mocks.hubHandlers) fn({ type: 'session_transcript', session_key: K1 });
    for (const fn of mocks.hubHandlers) fn({ type: 'session_transcript', session_key: 'nobody' });
    for (const fn of mocks.hubHandlers)
      fn({ type: 'session_named', transcript_id: T1, title: 'x', naming: 'auto' });
    // panel-manager decides whether the key is the one the panels show.
    await vi.waitFor(() =>
      expect(mocks.relayTranscriptMove.mock.calls).toEqual([[K1], ['nobody']])
    );
  });

  test('session_named retitles in place without a listing', async () => {
    await boot();
    const seen = vi.fn();
    model.subscribe(seen);
    for (const fn of mocks.hubHandlers)
      fn({ type: 'session_named', transcript_id: T2, title: 'Beam tuning', naming: 'done' });
    expect(model.rows()[1].title).toBe('Beam tuning');
    expect(seen).toHaveBeenCalledWith({ reason: 'named' });
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(listings()).toHaveLength(0);
  });

  test('session_deleted drops the row without a listing', async () => {
    await boot();
    const seen = vi.fn();
    model.subscribe(seen);
    for (const fn of mocks.hubHandlers) fn({ type: 'session_deleted', transcript_id: T2 });
    expect(model.rows().map((r) => r.transcript_id)).not.toContain(T2);
    expect(seen).toHaveBeenCalledWith({ reason: 'rows' });
    seen.mockClear();
    for (const fn of mocks.hubHandlers) fn({ type: 'session_deleted', transcript_id: T2 });
    expect(seen).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    expect(listings()).toHaveLength(0);
  });

  test('sessionOfArtifact finds the owner row, else null', async () => {
    await boot();
    expect(model.sessionOfArtifact({ owner: T2 })?.transcript_id).toBe(T2);
    expect(model.sessionOfArtifact({ owner: null })).toBeNull();
    expect(model.sessionOfArtifact({ owner: 'gone' })).toBeNull();
  });
});

describe('a new session’s title during its first turn', () => {
  const PROMPT =
    'The beam lifetime seemed to drop a few times during the night shift. Pull the beam current.';
  /** @param {string} text @param {boolean} [replay] */
  const send = (text, replay = false) => {
    for (const fn of mocks.chatHandlers['user-message'] ?? []) fn({ text, replay });
  };
  const titled = (/** @type {Partial<any>} */ extra) =>
    row(T3, { session_key: KNEW, current_for_key: true, title_source: 'prompt', ...extra });

  test('the sent prompt stands in at once, and re-listings stop once the listing titles it', async () => {
    await boot();
    pointer.setPointer(KNEW);
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    send(PROMPT);
    expect(model.activeRow()).toMatchObject({
      title: 'The beam lifetime seemed to drop a few times during the…',
      synthesized: true,
    });
    calls.length = 0;
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS[0] + model.REFRESH_THROTTLE_MS);
    expect(listings()).toHaveLength(1);
    // The record has the prompt now: the listing titles the session and the hub names it.
    pages.first = {
      ...pages.first,
      sessions: [titled({ title: 'From the hub' }), ...pages.first.sessions],
    };
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS[1] + model.REFRESH_THROTTLE_MS);
    expect(listings()).toHaveLength(2);
    expect(model.activeRow()?.title).toBe('From the hub');
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS[2] * 2);
    expect(listings()).toHaveLength(2);
  });

  test('a listed row without a title shows the sent prompt until the listing has one', async () => {
    pages.first = {
      ...pages.first,
      sessions: [
        titled({ title: 'Untitled session', title_source: null }),
        ...pages.first.sessions,
      ],
    };
    await boot(KNEW);
    send('anything weird overnight?');
    expect(model.rows()[0].title).toBe('anything weird overnight?');
    // A re-listing that still has no title keeps the stand-in on the fresh row.
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS[0] + model.REFRESH_THROTTLE_MS);
    expect(model.rows()[0].title).toBe('anything weird overnight?');
  });

  test('a titled session, a replayed prompt and a slash command change nothing', async () => {
    await boot();
    send(PROMPT);
    expect(model.activeRow()?.title).toBe('Session aaaa');
    pointer.setPointer(KNEW);
    send(PROMPT, true);
    send('/clear');
    expect(model.activeRow()?.title).toBe('New session');
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    calls.length = 0;
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS.reduce((a, b) => a + b, 0) + 5000);
    expect(listings()).toHaveLength(0);
  });

  test('moving to another session drops the stand-in and the re-listings', async () => {
    await boot();
    pointer.setPointer(KNEW);
    send(PROMPT);
    pointer.setPointer(K1);
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    calls.length = 0;
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS.reduce((a, b) => a + b, 0) + 5000);
    expect(listings()).toHaveLength(0);
    pointer.setPointer(KNEW);
    expect(model.activeRow()?.title).toBe('New session');
  });

  test('a re-listing that fails does not end the re-listings', async () => {
    await boot();
    pointer.setPointer(KNEW);
    await vi.advanceTimersByTimeAsync(model.REFRESH_THROTTLE_MS + 10);
    send(PROMPT);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const answer = /** @type {any} */ (vi.mocked(fetch).getMockImplementation());
    let down = true;
    vi.mocked(fetch).mockImplementation(async (/** @type {any} */ url, /** @type {any} */ init) => {
      const listing = (init?.method ?? 'GET') === 'GET' && String(url).includes('/api/sessions');
      if (down && listing) {
        down = false;
        throw new Error('hub unreachable');
      }
      return answer(url, init);
    });
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS[0] + model.REFRESH_THROTTLE_MS);
    expect(down).toBe(false);
    calls.length = 0;
    await vi.advanceTimersByTimeAsync(TITLE_CHASE_MS[1] + model.REFRESH_THROTTLE_MS);
    expect(listings()).toHaveLength(1);
    warn.mockRestore();
  });

  test('the shortened prompt matches the hub’s rule', () => {
    expect(shortenedPrompt('  short\n prompt ')).toBe('short prompt');
    const long = shortenedPrompt('word '.repeat(30));
    expect(long.length).toBeLessThanOrEqual(PROMPT_TITLE_CHARS);
    expect(long.endsWith('word…')).toBe(true);
  });
});

describe('pins', () => {
  const PINS = () => model.pinnedRows().map((r) => r.transcript_id);
  const pinPuts = () => calls.filter((c) => c.method === 'PUT' && c.url.endsWith('/pin'));

  test('the first page names every pinned session, a later page included, in pin order', async () => {
    pages.first.sessions[1] = row(T2, { pinned: '2026-11-06T10:00:00+00:00' });
    pages.first.pinned_sessions = [
      row(T3, { session_key: K2, current_for_key: true, pinned: '2026-11-06T09:00:00+00:00' }),
      pages.first.sessions[1],
    ];
    await boot();
    expect(PINS()).toEqual([T3, T2]);
    // A pinned session from a later page is the row a tab on its key is on.
    pointer.setPointer(K2);
    expect(model.activeRow()?.transcript_id).toBe(T3);
    expect(model.activeRow()?.synthesized).toBeUndefined();
  });

  test('setPinned flips the row at once, PUTs, and keeps the hub time', async () => {
    await boot();
    const done = model.setPinned(T2, true);
    expect(PINS()).toEqual([T2]);
    expect(await done).toBe(true);
    expect(pinPuts().map((c) => [c.url, c.body])).toEqual([
      [`/api/sessions/${T2}/pin`, { pinned: true }],
    ]);
    expect(model.pinnedRows()[0].pinned).toBe('2026-11-06T11:00:00+00:00');
    expect(await model.setPinned(T2, false)).toBe(true);
    expect(PINS()).toEqual([]);
  });

  test('a refused PUT puts the pin back', async () => {
    await boot();
    vi.mocked(fetch).mockImplementationOnce(
      async () => new Response(JSON.stringify({ detail: 'No such session.' }), { status: 404 })
    );
    expect(await model.setPinned(T2, true)).toBe(false);
    expect(PINS()).toEqual([]);
  });

  /** Hold the next pin PUT until the returned function answers it. */
  function holdPinPut() {
    const real = vi.mocked(fetch).getMockImplementation();
    /** @type {() => void} */
    let answer = () => {};
    const gate = new Promise((resolve) => (answer = () => resolve(undefined)));
    vi.mocked(fetch).mockImplementation(async (url, init) => {
      if (String(url).endsWith('/pin')) {
        await gate;
        return new Response(JSON.stringify({ pinned: '2026-11-06T11:00:00+00:00' }));
      }
      return /** @type {any} */ (real)(url, init);
    });
    return answer;
  }

  /** Run one refresh to its end. */
  async function refreshed() {
    model.refresh();
    await vi.advanceTimersByTimeAsync(1100);
  }

  test('a refresh decides the pins of loaded rows from later pages', async () => {
    await boot();
    await model.loadMore();
    expect(PINS()).toEqual([]);

    // Pinned from another tab: page 1 does not hold T3, the list names it.
    const stamp = '2026-11-06T10:00:00+00:00';
    pages.first.pinned_sessions = [row(T3, { pinned: stamp })];
    await refreshed();
    expect(model.rows().find((r) => r.transcript_id === T3)?.pinned).toBe(stamp);
    expect(PINS()).toEqual([T3]);

    // Unpinned from another tab: the list no longer names it.
    pages.first.pinned_sessions = [];
    await refreshed();
    expect(model.rows().find((r) => r.transcript_id === T3)?.pinned).toBeNull();
    expect(PINS()).toEqual([]);
  });

  test('a listing answered while the PUT is in flight does not undo the pin', async () => {
    pages.first.pinned_sessions = [];
    await boot();
    const answer = holdPinPut();
    const done = model.setPinned(T2, true);
    expect(model.pinPending(T2)).toBe(true);
    await refreshed();
    expect(PINS()).toEqual([T2]);
    answer();
    expect(await done).toBe(true);
    expect(model.pinPending(T2)).toBe(false);
    expect(PINS()).toEqual([T2]);
  });

  test('a listing read before the PUT settled and answered after does not undo the pin', async () => {
    pages.first.pinned_sessions = [];
    await boot();
    const real = vi.mocked(fetch).getMockImplementation();
    /** @type {() => void} */
    let release = () => {};
    const gate = new Promise((resolve) => (release = () => resolve(undefined)));
    vi.mocked(fetch).mockImplementationOnce(async (url, init) => {
      const stale = await /** @type {any} */ (real)(url, init);
      await gate;
      return stale;
    });
    model.refresh();
    await vi.advanceTimersByTimeAsync(1100);
    expect(await model.setPinned(T2, true)).toBe(true);
    release();
    await vi.advanceTimersByTimeAsync(0);
    expect(PINS()).toEqual([T2]);

    // The next listing was read after the pin, so it decides again.
    pages.first.pinned_sessions = [];
    await refreshed();
    expect(PINS()).toEqual([]);
  });

  test('a session_pinned frame from another tab flips the rows at once', async () => {
    await boot();
    const stamp = '2026-11-06T10:00:00+00:00';
    const frame = (/** @type {string|null} */ pinned) => {
      for (const fn of mocks.hubHandlers) fn({ type: 'session_pinned', transcript_id: T2, pinned });
    };
    frame(stamp);
    expect(PINS()).toEqual([T2]);
    expect(model.pinnedRows()[0].pinned).toBe(stamp);
    frame(null);
    expect(PINS()).toEqual([]);
  });

  test('a pin frame for a session not loaded here asks for a listing', async () => {
    await boot();
    pages.first.pinned_sessions = [row(T3, { pinned: '2026-11-06T10:00:00+00:00' })];
    for (const fn of mocks.hubHandlers)
      fn({ type: 'session_pinned', transcript_id: T3, pinned: '2026-11-06T10:00:00+00:00' });
    await vi.advanceTimersByTimeAsync(1100);
    expect(listings()).toHaveLength(1);
    expect(PINS()).toEqual([T3]);
  });

  test('a deleted session leaves the pins', async () => {
    pages.first.sessions[1] = row(T2, { pinned: '2026-11-06T10:00:00+00:00' });
    pages.first.pinned_sessions = [pages.first.sessions[1]];
    await boot();
    model.forgetSession(T2);
    expect(PINS()).toEqual([]);
  });
});
