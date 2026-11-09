/**
 * Unit tests for the chat's attention signal (chat-attention.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-attention.test.mjs
 *
 * The signal reads the chat's frames and turn ends from the bus, so the tests
 * publish the same details the renderer would. Notification and
 * document.visibilityState are stubbed: the browser is not asked anything.
 * The opt-in is a row in the Agent pane's ⋯ (agent-menu.js); the tests read
 * it as the menu would and run it as a pick does. The hub's menu is a double.
 */

import { test, expect, describe, vi, beforeEach, afterEach } from 'vitest';

import { agentMenuRows } from '../../../src/osprey/interfaces/web_terminal/static/js/agent-menu.js';
import { initChatAttention } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-attention.js';
import {
  emitChat,
  onChat,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js';

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  setServiceMenuExtras: () => {},
}));

const BASE_TITLE = 'Osprey Console';
const STORAGE_KEY = 'osprey-chat-notify';

/** @type {Array<{ title: string, options: any, close: ReturnType<typeof vi.fn>, onclick: (() => void) | null }>} */
let notes;
/** @type {() => void} */
let dispose;
/** @type {HTMLElement} */
let messages;

/** The console context the feature reads; only the log matters here. */
const ctxOf = () => /** @type {any} */ ({ messages });

/**
 * Stub the Notification constructor. The permission is what the browser
 * reports now; `answer` is what requestPermission resolves to.
 * @param {{ permission?: string, answer?: string | Error }} [options]
 */
function stubNotification({ permission = 'granted', answer = 'granted' } = {}) {
  notes = [];
  class FakeNotification {
    static permission = permission;
    static requestPermission = vi.fn(async () => {
      if (answer instanceof Error) throw answer;
      FakeNotification.permission = answer;
      return answer;
    });
    constructor(/** @type {string} */ title, /** @type {any} */ options) {
      this.title = title;
      this.options = options;
      this.close = vi.fn();
      /** @type {any} */ this.onclick = null;
      notes.push(this);
    }
  }
  vi.stubGlobal('Notification', FakeNotification);
  return FakeNotification;
}

/** @param {'visible' | 'hidden'} state */
function setVisibility(state) {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
}

/** @param {Record<string, any>} event @param {boolean} [replay] */
function frame(event, replay = false) {
  emitChat('frame', { event, replay });
}

/** @param {string} id @param {string} [tool] */
const approval = (id, tool = 'Execute') => ({
  type: 'permission_request',
  request_id: id,
  kind: 'tool',
  tool_name: tool,
  tool_name_raw: tool.toLowerCase(),
  input: { code: 'print(secret)' },
});

/** @param {string} id */
const question = (id) => ({
  type: 'permission_request',
  request_id: id,
  kind: 'question',
  tool_name: 'AskUserQuestion',
});

/** @param {string} id */
const resolved = (id) => ({ type: 'permission_resolved', request_id: id, allow: true });

/** The opt-in's row in the Agent pane's ⋯, as the menu reads it now; undefined when absent. */
const notifyRow = () =>
  agentMenuRows().find((row) => row.label?.startsWith('Desktop notifications'));

/** Pick the opt-in's row, as a click on it does. */
const pick = () => notifyRow()?.run?.();

/** Let the permission prompt and the title watcher settle. No fake time passes. */
const flush = () => vi.advanceTimersByTimeAsync(0);

beforeEach(() => {
  vi.useFakeTimers();
  document.title = BASE_TITLE;
  localStorage.clear();
  setVisibility('visible');
  stubNotification();
  messages = document.createElement('div');
  document.body.append(messages);
  dispose = initChatAttention(ctxOf());
});

afterEach(() => {
  dispose();
  messages.remove();
  Reflect.deleteProperty(document, 'visibilityState');
  vi.unstubAllGlobals();
  vi.useRealTimers();
  localStorage.clear();
});

/** Turn notifications on, with the browser already granting permission. */
function turnNotificationsOn() {
  localStorage.setItem(STORAGE_KEY, '1');
  dispose();
  dispose = initChatAttention(ctxOf());
}

describe('tab title count', () => {
  test('counts waiting approvals up and back down to the plain title', () => {
    frame(approval('a'));
    expect(document.title).toBe(`(1) Approval needed · ${BASE_TITLE}`);
    frame(approval('b'));
    expect(document.title).toBe(`(2) Approval needed · ${BASE_TITLE}`);
    frame(resolved('a'));
    expect(document.title).toBe(`(1) Approval needed · ${BASE_TITLE}`);
    frame(resolved('b'));
    expect(document.title).toBe(BASE_TITLE);
  });

  test('says Question for a question card and a plain prompt for a mix', () => {
    frame(question('q'));
    expect(document.title).toBe(`(1) Question · ${BASE_TITLE}`);
    frame(approval('a'));
    expect(document.title).toBe(`(2) Waiting on you · ${BASE_TITLE}`);
    frame(resolved('q'));
    expect(document.title).toBe(`(1) Approval needed · ${BASE_TITLE}`);
  });

  test('counts a request re-delivered on reconnect once', () => {
    // A hello.pending re-delivery arrives as a live frame, not a replay.
    frame(approval('p'));
    frame(approval('p'));
    expect(document.title).toBe(`(1) Approval needed · ${BASE_TITLE}`);
  });

  test('replayed history never badges the title, and never notifies, even when hidden and enabled', () => {
    turnNotificationsOn();
    setVisibility('hidden');
    frame(approval('old'), true);
    frame(question('oldq'), true);
    expect(document.title).toBe(BASE_TITLE);
    expect(notes).toHaveLength(0);
  });

  test('a turn ending clears the badge, and so does a reset', () => {
    frame(approval('a'));
    emitChat('turn-end', {});
    expect(document.title).toBe(BASE_TITLE);

    frame(approval('b'));
    emitChat('reset', {});
    expect(document.title).toBe(BASE_TITLE);
  });

  test.each([
    ['answering the card', resolved('a')],
    [
      'the server declining it for want of an answer',
      {
        type: 'permission_resolved',
        request_id: 'a',
        allow: false,
        for_session: false,
        answers: null,
        expired: true,
      },
    ],
  ])('%s clears the badge and closes its notification', (_name, resolution) => {
    turnNotificationsOn();
    setVisibility('hidden');
    frame(approval('a'));
    expect(document.title).toBe(`(1) Approval needed · ${BASE_TITLE}`);
    expect(notes).toHaveLength(1);

    frame(resolution);

    expect(document.title).toBe(BASE_TITLE);
    expect(notes[0].close).toHaveBeenCalled();
  });

  test('composes with another writer: the badge re-bases on the new title', async () => {
    frame(approval('a'));
    document.title = 'Other page';
    await flush();
    expect(document.title).toBe('(1) Approval needed · Other page');
    frame(approval('b'));
    expect(document.title).toBe('(2) Approval needed · Other page');
    frame(resolved('a'));
    frame(resolved('b'));
    expect(document.title).toBe('Other page');
  });
});

describe('waiting count on the chat bus', () => {
  /** @type {number[]} */
  let said;
  /** @type {() => void} */
  let stop;

  beforeEach(() => {
    said = [];
    stop = onChat('attention', ({ waiting }) => said.push(waiting));
  });
  afterEach(() => stop());

  test('follows every change of the pending-card count', () => {
    frame(approval('a'));
    frame(question('q'));
    frame(resolved('a'));
    frame(resolved('q'));
    expect(said).toEqual([1, 2, 1, 0]);
  });

  test('a re-delivered request and replayed history do not change it', () => {
    frame(approval('a'));
    frame(approval('a'));
    frame(approval('old'), true);
    expect(said).toEqual([1]);
  });

  test('a turn end and a reset say zero, even when nothing was waiting', () => {
    frame(approval('a'));
    emitChat('turn-end', {});
    emitChat('reset', {});
    expect(said).toEqual([1, 0, 0]);
  });
});

describe('desktop notifications', () => {
  test('a waiting card notifies once when hidden and enabled, and dedupes per request', () => {
    turnNotificationsOn();
    setVisibility('hidden');
    frame(approval('n1'));
    expect(notes).toHaveLength(1);
    // The title names the tool, and nothing of its input: it can show on a lock screen.
    expect(notes[0].title).toContain('Execute');
    expect(notes[0].title).not.toContain('secret');
    expect(notes[0].options.tag).toBe('osprey-attention-n1');

    frame(approval('n1'));
    expect(notes).toHaveLength(1);

    frame(approval('n2', 'Bash'));
    expect(notes).toHaveLength(2);
    expect(notes[1].title).toContain('Bash');
  });

  test('nothing notifies while the page is visible or notifications are off', () => {
    setVisibility('hidden');
    frame(approval('off'));
    expect(notes).toHaveLength(0);

    turnNotificationsOn();
    setVisibility('visible');
    frame(approval('visible'));
    expect(notes).toHaveLength(0);
    expect(document.title).toBe(`(1) Approval needed · ${BASE_TITLE}`);
  });

  test('clicking the notification focuses the window and the card', () => {
    turnNotificationsOn();
    setVisibility('hidden');
    const card = document.createElement('section');
    card.className = 'op-card';
    card.dataset.state = 'pending';
    card.tabIndex = -1;
    messages.append(card);
    frame(approval('c1'));

    const focusWindow = vi.spyOn(window, 'focus');
    notes[0].onclick?.();
    expect(focusWindow).toHaveBeenCalled();
    expect(document.activeElement).toBe(card);
    expect(notes[0].close).toHaveBeenCalled();
  });

  test('a turn that ends while hidden notifies once; a visible one does not', () => {
    turnNotificationsOn();
    setVisibility('visible');
    emitChat('turn-end', {});
    expect(notes).toHaveLength(0);

    setVisibility('hidden');
    emitChat('turn-end', {});
    expect(notes).toHaveLength(1);
    expect(notes[0].title).toMatch(/finished/);
  });
});

describe('the Desktop notifications row in the Agent pane ⋯', () => {
  test('is a checkable row, off until the operator turns it on', () => {
    expect(notifyRow()).toMatchObject({
      label: 'Desktop notifications',
      checked: false,
      disabled: false,
    });
  });

  test('a pick requests permission and turns notifications on when granted', async () => {
    const Fake = stubNotification({ permission: 'default', answer: 'granted' });
    dispose();
    dispose = initChatAttention(ctxOf());
    expect(notifyRow()?.checked).toBe(false);

    pick();
    await flush();
    expect(notifyRow()?.checked).toBe(true);
    expect(Fake.requestPermission).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem(STORAGE_KEY)).toBe('1');
  });

  test('a denied permission leaves notifications off with no throw', async () => {
    const Fake = stubNotification({ permission: 'default', answer: 'denied' });
    dispose();
    dispose = initChatAttention(ctxOf());

    expect(() => pick()).not.toThrow();
    await flush();
    expect(Fake.requestPermission).toHaveBeenCalledTimes(1);
    expect(notifyRow()?.checked).toBe(false);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();

    setVisibility('hidden');
    frame(approval('denied'));
    expect(notes).toHaveLength(0);
  });

  test('a browser that already blocks notifications shows the row inert and off, saying so', () => {
    stubNotification({ permission: 'denied' });
    localStorage.setItem(STORAGE_KEY, '1');
    dispose();
    dispose = initChatAttention(ctxOf());
    const row = notifyRow();
    expect(row?.disabled).toBe(true);
    expect(row?.checked).toBe(false);
    expect(row?.label).toMatch(/blocked/);
  });

  test('a second pick turns it off and forgets the choice', () => {
    turnNotificationsOn();
    expect(notifyRow()?.checked).toBe(true);
    pick();
    expect(notifyRow()?.checked).toBe(false);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  test('without a Notification API there is no row, and nothing throws', () => {
    vi.stubGlobal('Notification', undefined);
    dispose();
    dispose = initChatAttention(ctxOf());
    expect(notifyRow()).toBeUndefined();
    setVisibility('hidden');
    expect(() => frame(approval('nope'))).not.toThrow();
    expect(document.title).toBe(`(1) Approval needed · ${BASE_TITLE}`);
  });

  test('stopping the signal takes the row away', () => {
    dispose();
    expect(notifyRow()).toBeUndefined();
    dispose = initChatAttention(ctxOf());
    expect(notifyRow()).toBeDefined();
  });
});
