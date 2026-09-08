// @ts-check
/**
 * Unit tests for the Simple-view chat controller (chat.js):
 *   ./node_modules/.bin/vitest run tests/interfaces/web_terminal/chat.test.mjs
 *
 * Two things are worth pinning here.
 *
 * The first is which sentence an operator reads when the endpoint refuses a
 * turn. That choice is exported as `transportNotice`, because the failure it
 * exists to prevent is not a crash: the endpoint answers 409 both for "a turn
 * is already running" and for "this chat was restarted by your own posture
 * flip", and keying on the status alone told the second operator the first
 * sentence — false, and with no cue to do the one thing that works (send the
 * prompt again).
 *
 * The second is the binding. The console has no conversation of its own: it
 * addresses the tab's session pointer, replays that key's transcript, and
 * arrives from the Expert view through a hand-off that the other view can
 * refuse. Those paths are DOM-and-transport glue, so they are driven here
 * against a real (happy-dom) container with chat-client.js mocked — what is
 * asserted is which key was addressed, what reached the log, and which one
 * action the operator is left with.
 *
 * Module isolation: the hand-off entry point is module-level state (app.js has
 * no handle on the console), so each binding test gets a fresh module instance
 * via vi.resetModules() + dynamic import, the same pattern as
 * terminal-resume.test.mjs.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { transportNotice } from '../../../src/osprey/interfaces/web_terminal/static/js/chat.js';

const CHAT_JS = '../../../src/osprey/interfaces/web_terminal/static/js/chat.js';
const STORAGE_KEY = 'osprey-pty-session';

/** Transport doubles, re-armed per test by `mountChat`. */
const transport = {
  fetchHistory: vi.fn(),
  sendPrompt: vi.fn(),
  interrupt: vi.fn(),
  requestHandoff: vi.fn(),
};

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js', () => ({
  fetchHistory: (/** @type {any} */ key) => transport.fetchHistory(key),
  sendPrompt: (/** @type {any} */ key, /** @type {any} */ prompt, /** @type {any} */ cb) =>
    transport.sendPrompt(key, prompt, cb),
  interrupt: (/** @type {any} */ key) => transport.interrupt(key),
  requestHandoff: (/** @type {any} */ key, /** @type {any} */ options) =>
    transport.requestHandoff(key, options),
}));

// terminal.js is xterm glue; the console only needs its panel broadcast, and
// first-contact.js (imported by the console) only needs its session hook.
const notifySessionChange = vi.fn();
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  notifySessionChange: (/** @type {any} */ id) => notifySessionChange(id),
  onSessionChange: () => {},
  focusTerminal: () => {},
  pasteToTerminal: () => {},
}));

/**
 * A transport failure shaped the way chat-client.js builds one.
 * @param {number} status
 * @param {string} [slug]
 */
function transportError(status, slug = '') {
  const err = /** @type {Error & { status: number, slug: string }} */ (
    new Error(`HTTP ${status}: Conflict`)
  );
  err.status = status;
  err.slug = slug;
  return err;
}

describe('transportNotice', () => {
  test('the terminated-chat 409 tells the operator to resend', () => {
    const notice = transportNotice(transportError(409, 'chat_terminated'));
    expect(notice).toContain('send your message again');
    expect(notice).not.toContain('already running');
  });

  test('the turn-in-progress 409 keeps its own copy', () => {
    expect(transportNotice(transportError(409, 'turn_in_progress'))).toBe(
      'A turn is already running.'
    );
  });

  test('two 409s with different slugs read differently', () => {
    expect(transportNotice(transportError(409, 'chat_terminated'))).not.toBe(
      transportNotice(transportError(409, 'turn_in_progress'))
    );
  });

  test('the capacity 429 is keyed on its slug', () => {
    expect(transportNotice(transportError(429, 'chat_capacity'))).toBe(
      'Server busy — please retry in a moment.'
    );
  });

  test('a rejection with no slug falls back to the status table', () => {
    expect(transportNotice(transportError(503))).toBe('Operator agent unavailable.');
    expect(transportNotice(transportError(409))).toBe('A turn is already running.');
  });

  test('an unknown slug falls back to the status table', () => {
    expect(transportNotice(transportError(503, 'something_new'))).toBe(
      'Operator agent unavailable.'
    );
  });

  test('a plain Error still yields its status notice, then the generic line', () => {
    expect(transportNotice(new Error('HTTP 429: Too Many Requests'))).toBe(
      'Server busy — please retry in a moment.'
    );
    expect(transportNotice(new Error('network down'))).toBe(
      'Connection to the operator agent failed.'
    );
  });

  test('a null failure does not throw', () => {
    expect(transportNotice(null)).toBe('Connection to the operator agent failed.');
  });
});

// ---- Binding, replay and hand-off ---- //

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat.js')} */
let chat;

/** The mounted console's container.
 * @type {HTMLElement} */
let container;

/**
 * Mount a fresh console into a fresh container.
 * @param {{ mode?: string, pointer?: string|null }} [options]
 */
async function mountChat({ mode = 'simple', pointer = null } = {}) {
  document.documentElement.setAttribute('data-ui-mode', mode);
  if (pointer) localStorage.setItem(STORAGE_KEY, pointer);
  container = document.createElement('div');
  container.id = 'operator-container';
  document.body.append(container);

  vi.resetModules();
  chat = await import(CHAT_JS);
  chat.initChat();
  // Let the boot-time bind's fetch settle before a test looks at the log.
  await Promise.resolve();
  await Promise.resolve();
}

/** @param {string} selector */
const one = (selector) => container.querySelector(selector);

/** @param {string} selector */
const textOf = (selector) => one(selector)?.textContent ?? '';

/** Every rendered log entry's text, in order. */
const logEntries = () =>
  [...container.querySelectorAll('.op-entry-body')].map((el) => el.textContent?.trim());

beforeEach(() => {
  localStorage.clear();
  document.body.replaceChildren();
  notifySessionChange.mockReset();
  transport.fetchHistory.mockReset().mockResolvedValue([]);
  transport.sendPrompt.mockReset().mockReturnValue({ abort: () => {}, aborted: false });
  transport.interrupt.mockReset().mockResolvedValue(undefined);
  transport.requestHandoff.mockReset().mockResolvedValue({ state: 'simple', session_id: 'k' });
});

afterEach(() => {
  vi.useRealTimers();
  document.documentElement.removeAttribute('data-ui-mode');
});

describe('initChat binding', () => {
  test('a tab with no pointer mints one and stores it', async () => {
    await mountChat({ mode: 'expert' });
    const stored = localStorage.getItem(STORAGE_KEY);
    expect(stored).toMatch(/^[0-9a-f-]{36}$/);
    // Nothing to replay on a key that has never existed.
    expect(transport.fetchHistory).not.toHaveBeenCalled();
  });

  test('an existing pointer is adopted, never replaced', async () => {
    await mountChat({ mode: 'expert', pointer: 'key-from-the-terminal' });
    expect(localStorage.getItem(STORAGE_KEY)).toBe('key-from-the-terminal');
  });

  test('a page that opens in Simple mode replays the pointer it adopted', async () => {
    transport.fetchHistory.mockResolvedValue([
      { role: 'user', content: 'status?' },
      { role: 'assistant', content: 'All quiet.' },
    ]);
    await mountChat({ pointer: 'K1' });

    expect(transport.fetchHistory).toHaveBeenCalledWith('K1');
    expect(logEntries()).toEqual(['status?', 'All quiet.']);
    expect(notifySessionChange).toHaveBeenCalledWith('K1');
  });

  test('a prompt is sent under the pointer key, not a minted chat id', async () => {
    await mountChat({ pointer: 'K1' });
    const textarea = /** @type {HTMLTextAreaElement} */ (one('textarea'));
    textarea.value = 'ping';
    /** @type {HTMLButtonElement} */ (one('.op-send-btn')).click();

    expect(transport.sendPrompt).toHaveBeenCalledWith('K1', 'ping', expect.anything());
  });

  test('a pointer change rebinds the log to the new key', async () => {
    transport.fetchHistory.mockResolvedValue([{ role: 'assistant', content: 'from K1' }]);
    await mountChat({ pointer: 'K1' });
    expect(logEntries()).toEqual(['from K1']);

    transport.fetchHistory.mockResolvedValue([{ role: 'assistant', content: 'from K2' }]);
    const pointer = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js'
    );
    pointer.setPointer('K2');
    await Promise.resolve();
    await Promise.resolve();

    // Reset then replay: the old key's turns are gone, not appended to.
    expect(logEntries()).toEqual(['from K2']);
    expect(notifySessionChange).toHaveBeenLastCalledWith('K2');
  });

  test('a cleared pointer leaves the log on the key it was bound to', async () => {
    transport.fetchHistory.mockResolvedValue([{ role: 'assistant', content: 'from K1' }]);
    await mountChat({ pointer: 'K1' });
    const pointer = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js'
    );

    transport.fetchHistory.mockClear();
    pointer.clearPointer();
    await Promise.resolve();
    await Promise.resolve();

    // A clear says the key is dead for both views, not that this console
    // moved to another conversation. There is nothing to replay.
    expect(logEntries()).toEqual(['from K1']);
    expect(transport.fetchHistory).not.toHaveBeenCalled();
  });

  test('overlapping pointer changes render only the newer key turns', async () => {
    await mountChat({ pointer: 'K1' });
    const pointer = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js'
    );

    /** @type {((value: any) => void)[]} */
    const admit = [];
    transport.fetchHistory.mockImplementation(() => new Promise((resolve) => admit.push(resolve)));
    pointer.setPointer('K2');
    pointer.setPointer('K3');
    // K2's transcript lands after K3 has already claimed the log.
    admit[1]([{ role: 'assistant', content: 'from K3' }]);
    admit[0]([{ role: 'assistant', content: 'from K2' }]);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(logEntries()).toEqual(['from K3']);
  });

  test('"New conversation" mints a key, points the tab at it, and clears the log', async () => {
    transport.fetchHistory.mockResolvedValue([{ role: 'assistant', content: 'old talk' }]);
    await mountChat({ pointer: 'K1' });
    expect(logEntries()).toEqual(['old talk']);

    transport.fetchHistory.mockClear();
    /** @type {HTMLButtonElement} */ (one('.op-session-new')).click();
    await Promise.resolve();

    const minted = localStorage.getItem(STORAGE_KEY);
    expect(minted).toMatch(/^[0-9a-f-]{36}$/);
    expect(minted).not.toBe('K1');
    expect(logEntries()).toEqual([]);
    expect(notifySessionChange).toHaveBeenLastCalledWith(minted);
    // A key minted this instant has no transcript worth asking for.
    expect(transport.fetchHistory).not.toHaveBeenCalled();
  });
});

describe('enterFromExpert', () => {
  test('shows the transitional state, then the replayed history and a live input', async () => {
    /** @type {(value: any) => void} */
    let admit = () => {};
    transport.requestHandoff.mockReturnValue(
      new Promise((resolve) => {
        admit = resolve;
      })
    );
    transport.fetchHistory.mockResolvedValue([{ role: 'assistant', content: 'carried over' }]);
    await mountChat({ pointer: 'K1' });

    const entered = chat.enterFromExpert();
    expect(/** @type {HTMLElement} */ (one('.op-handoff')).hidden).toBe(false);
    expect(textOf('.op-handoff-message')).toContain('Finishing in the other view');
    expect(textOf('.op-handoff-action')).toBe('Stop and switch now');
    expect(/** @type {HTMLTextAreaElement} */ (one('textarea')).disabled).toBe(true);

    admit({ state: 'simple', session_id: 'K1' });
    await entered;

    expect(transport.requestHandoff).toHaveBeenCalledWith('K1', expect.objectContaining({
      interrupt: false,
    }));
    expect(/** @type {HTMLElement} */ (one('.op-handoff')).hidden).toBe(true);
    expect(logEntries()).toEqual(['carried over']);
    expect(/** @type {HTMLTextAreaElement} */ (one('textarea')).disabled).toBe(false);
  });

  test('the elapsed counter ticks while the wait runs', async () => {
    vi.useFakeTimers();
    transport.requestHandoff.mockReturnValue(new Promise(() => {}));
    await mountChat({ pointer: 'K1' });

    void chat.enterFromExpert();
    vi.advanceTimersByTime(62_000);
    expect(textOf('.op-handoff-elapsed')).toBe('1:02');
  });

  test('a hook-less refusal offers only the stop, which re-asks with interrupt', async () => {
    transport.requestHandoff.mockRejectedValueOnce(transportError(409, 'handoff_needs_interrupt'));
    await mountChat({ pointer: 'K1' });

    await chat.enterFromExpert();
    const action = /** @type {HTMLButtonElement} */ (one('.op-handoff-action'));
    expect(textOf('.op-handoff-message')).toBe('The other view may still be working.');
    expect(action.hidden).toBe(false);
    expect(action.textContent).toBe('Stop and switch now');
    expect(/** @type {HTMLTextAreaElement} */ (one('textarea')).disabled).toBe(true);

    transport.requestHandoff.mockResolvedValue({ state: 'simple', session_id: 'K1' });
    action.click();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(transport.requestHandoff).toHaveBeenLastCalledWith('K1', expect.objectContaining({
      interrupt: true,
    }));
  });

  test('a session held elsewhere states the fact and offers nothing', async () => {
    transport.requestHandoff.mockRejectedValue(transportError(409, 'session_attached_elsewhere'));
    await mountChat({ pointer: 'K1' });

    await chat.enterFromExpert();
    expect(textOf('.op-handoff-message')).toBe('This session is in use in another tab or view.');
    expect(/** @type {HTMLButtonElement} */ (one('.op-handoff-action')).hidden).toBe(true);
  });

  test('a wait a newer request took over states the fact and offers a retry', async () => {
    transport.requestHandoff.mockRejectedValueOnce(transportError(409, 'handoff_superseded'));
    await mountChat({ pointer: 'K1' });

    await chat.enterFromExpert();
    const action = /** @type {HTMLButtonElement} */ (one('.op-handoff-action'));
    expect(textOf('.op-handoff-message')).toBe('Another request took over this session.');
    expect(action.hidden).toBe(false);
    expect(action.textContent).toBe('Retry');

    // The retry re-asks without an interrupt: the key's chat is pooled by now.
    transport.requestHandoff.mockResolvedValue({ state: 'simple', session_id: 'K1' });
    action.click();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(transport.requestHandoff).toHaveBeenLastCalledWith('K1', expect.objectContaining({
      interrupt: false,
    }));
  });

  test('a capacity 429 offers a retry that re-asks without interrupting', async () => {
    transport.requestHandoff.mockRejectedValueOnce(transportError(429, 'chat_capacity'));
    await mountChat({ pointer: 'K1' });

    await chat.enterFromExpert();
    const action = /** @type {HTMLButtonElement} */ (one('.op-handoff-action'));
    expect(textOf('.op-handoff-message')).toBe('No chat capacity right now.');
    expect(action.textContent).toBe('Retry');

    transport.requestHandoff.mockResolvedValue({ state: 'simple', session_id: 'K1' });
    action.click();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(transport.requestHandoff).toHaveBeenLastCalledWith('K1', expect.objectContaining({
      interrupt: false,
    }));
    expect(/** @type {HTMLElement} */ (one('.op-handoff')).hidden).toBe(true);
  });

  test('an outgoing agent still shutting down is retryable', async () => {
    transport.requestHandoff.mockRejectedValue(transportError(503, 'outgoing_still_running'));
    await mountChat({ pointer: 'K1' });

    await chat.enterFromExpert();
    expect(textOf('.op-handoff-message')).toBe('The previous agent is still shutting down.');
    expect(/** @type {HTMLButtonElement} */ (one('.op-handoff-action')).textContent).toBe('Retry');
  });

  test('an abandoned request (204, no body) leaves the wait exactly as it was', async () => {
    // The server saw this request's channel close and handed nothing over.
    // Nothing failed, so there is nothing to say — and the wait on screen is
    // still the true state, with its one action still open.
    transport.requestHandoff.mockResolvedValue(null);
    await mountChat({ pointer: 'K1' });
    transport.fetchHistory.mockClear();

    await chat.enterFromExpert();

    expect(/** @type {HTMLElement} */ (one('.op-handoff')).hidden).toBe(false);
    expect(textOf('.op-handoff-message')).toContain('Finishing in the other view');
    const action = /** @type {HTMLButtonElement} */ (one('.op-handoff-action'));
    expect(action.textContent).toBe('Stop and switch now');
    expect(action.hidden).toBe(false);
    expect(/** @type {HTMLTextAreaElement} */ (one('textarea')).disabled).toBe(true);
    // Nothing was handed over, so nothing is replayed.
    expect(transport.fetchHistory).not.toHaveBeenCalled();
  });

  test('a superseded attempt finishing last does not open the console over the newer wait', async () => {
    /** @type {((value: any) => void)[]} */
    const admit = [];
    transport.requestHandoff.mockImplementation(
      () => new Promise((resolve) => admit.push(resolve))
    );
    await mountChat({ pointer: 'K1' });

    const first = chat.enterFromExpert();
    const second = chat.enterFromExpert();
    // The newer attempt lands first; the older one is still in flight.
    admit[1]({ state: 'simple', session_id: 'K1' });
    await second;
    transport.requestHandoff.mockClear();

    // Nothing is waiting any more, so re-arm the wait the newer attempt would
    // be showing had the older one not been about to answer.
    const third = chat.enterFromExpert();
    admit[0]({ state: 'simple', session_id: 'K1' });
    await first;

    expect(/** @type {HTMLElement} */ (one('.op-handoff')).hidden).toBe(false);
    expect(/** @type {HTMLTextAreaElement} */ (one('textarea')).disabled).toBe(true);
    admit[2]({ state: 'simple', session_id: 'K1' });
    await third;
  });

  test('a failure with no slug says the hand-off failed and offers a retry', async () => {
    transport.requestHandoff.mockRejectedValue(new Error('network down'));
    await mountChat({ pointer: 'K1' });

    await chat.enterFromExpert();
    // The in-log transport copy is for the log; the overlay states its own
    // failure in its own voice.
    expect(textOf('.op-handoff-message')).toBe('The hand-off failed.');
    expect(/** @type {HTMLButtonElement} */ (one('.op-handoff-action')).textContent).toBe('Retry');
  });

  test('the elapsed counter is not announced on every tick', async () => {
    transport.requestHandoff.mockReturnValue(new Promise(() => {}));
    await mountChat({ pointer: 'K1' });

    void chat.enterFromExpert();
    expect(one('.op-handoff')?.getAttribute('aria-live')).toBe('polite');
    expect(one('.op-handoff-elapsed')?.getAttribute('aria-live')).toBe('off');
  });

  test('a console that was never mounted resolves rather than throwing', async () => {
    vi.resetModules();
    const fresh = await import(CHAT_JS);
    await expect(fresh.enterFromExpert()).resolves.toBeUndefined();
  });
});

describe('a prompt refused by the other view', () => {
  test('routes through the hand-off overlay and gives the text back', async () => {
    await mountChat({ pointer: 'K1' });
    const textarea = /** @type {HTMLTextAreaElement} */ (one('textarea'));
    textarea.value = 'open the shutter';
    /** @type {HTMLButtonElement} */ (one('.op-send-btn')).click();

    const callbacks = transport.sendPrompt.mock.calls[0][2];
    callbacks.onError(transportError(409, 'handoff_needs_interrupt'));

    expect(textOf('.op-handoff-message')).toBe('The other view may still be working.');
    // The prompt never ran, so neither its bubble nor its text is lost.
    expect(logEntries()).toEqual([]);
    expect(textarea.value).toBe('open the shutter');
  });

  test('a 409 held-elsewhere from the chat endpoint reads exactly as the hand-off one', async () => {
    await mountChat({ pointer: 'K1' });
    const textarea = /** @type {HTMLTextAreaElement} */ (one('textarea'));
    textarea.value = 'status?';
    /** @type {HTMLButtonElement} */ (one('.op-send-btn')).click();

    const callbacks = transport.sendPrompt.mock.calls[0][2];
    callbacks.onError(transportError(409, 'session_attached_elsewhere'));

    // One slug, one sentence, whichever endpoint answered it — a prompt
    // refused because another tab holds the session is the same fact as a
    // hand-off refused for it, and a second phrasing would read as a second
    // problem.
    expect(textOf('.op-handoff-message')).toBe('This session is in use in another tab or view.');
    expect(/** @type {HTMLButtonElement} */ (one('.op-handoff-action')).hidden).toBe(true);
    expect(logEntries()).toEqual([]);
    expect(textarea.value).toBe('status?');
  });

  test('a terminated chat keeps its resend notice rather than the overlay', async () => {
    await mountChat({ pointer: 'K1' });
    const textarea = /** @type {HTMLTextAreaElement} */ (one('textarea'));
    textarea.value = 'status?';
    /** @type {HTMLButtonElement} */ (one('.op-send-btn')).click();

    const callbacks = transport.sendPrompt.mock.calls[0][2];
    callbacks.onError(transportError(409, 'chat_terminated'));
    callbacks.onClose();

    // The one useful answer to this slug is the notice: send it again. An
    // overlay would take the console away and offer a retry of the wrong
    // thing — and it would leave the input disabled, so there would be no
    // sending it again.
    expect(textOf('.op-system')).toContain('send your message again');
    expect(/** @type {HTMLElement} */ (one('.op-handoff')).hidden).toBe(true);
    expect(/** @type {HTMLTextAreaElement} */ (one('textarea')).disabled).toBe(false);
  });

  test('an ordinary transport failure still reads as a system notice', async () => {
    await mountChat({ pointer: 'K1' });
    const textarea = /** @type {HTMLTextAreaElement} */ (one('textarea'));
    textarea.value = 'status?';
    /** @type {HTMLButtonElement} */ (one('.op-send-btn')).click();

    const callbacks = transport.sendPrompt.mock.calls[0][2];
    callbacks.onError(new Error('network down'));

    expect(textOf('.op-system')).toBe('Connection to the operator agent failed.');
    expect(/** @type {HTMLElement} */ (one('.op-handoff')).hidden).toBe(true);
    expect(logEntries()).toEqual(['status?']);
  });
});

describe('minting a session key without crypto.randomUUID', () => {
  /** @type {any} */
  let savedRandomUUID;

  beforeEach(() => {
    savedRandomUUID = globalThis.crypto?.randomUUID;
    if (globalThis.crypto) {
      // A plain-http, non-localhost origin is a documented OSPREY topology,
      // and there the property is simply absent.
      delete (/** @type {any} */ (globalThis.crypto).randomUUID);
    }
  });

  afterEach(() => {
    if (globalThis.crypto && savedRandomUUID) {
      /** @type {any} */ (globalThis.crypto).randomUUID = savedRandomUUID;
    }
  });

  test('the console still mounts', async () => {
    await mountChat();

    expect(container.querySelector('.op-send-btn')).toBeTruthy();
  });

  test('the minted key keeps the bare-UUID grammar the store keys on', async () => {
    await mountChat();

    const key = localStorage.getItem(STORAGE_KEY);
    expect(key).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });

  test('a new conversation mints one too', async () => {
    await mountChat();
    const first = localStorage.getItem(STORAGE_KEY);

    /** @type {HTMLButtonElement} */ (one('.op-session-new')).click();

    const second = localStorage.getItem(STORAGE_KEY);
    expect(second).not.toBe(first);
    expect(second).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });
});

describe('renderChatBootFailure', () => {
  test('a console that never mounted says so where it would have been', async () => {
    const { renderChatBootFailure } = await import(CHAT_JS);
    const host = document.createElement('div');
    host.id = 'operator-container';
    host.textContent = 'stale';
    document.body.append(host);

    renderChatBootFailure('operator-container');

    expect(host.textContent).toContain('failed to start');
    expect(host.textContent).not.toContain('stale');
  });

  test('a missing container is not an error of its own', async () => {
    const { renderChatBootFailure } = await import(CHAT_JS);

    expect(() => renderChatBootFailure('nothing-here')).not.toThrow();
  });
});
