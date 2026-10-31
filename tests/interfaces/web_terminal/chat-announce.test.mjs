/**
 * Unit tests for the Simple chat's screen-reader layer (chat-announce.js and
 * the roles, labels and focus rules the console, blocks and cards carry):
 *   npx vitest run tests/interfaces/web_terminal/chat-announce.test.mjs
 *
 * What a screen-reader user hears is pinned here: each finished reply, each
 * approval or question that needs the operator, each error, and each notice
 * the console puts in the log, once and live only. Replayed history is silent.
 * Every control the keyboard reaches is a named native control, and a new
 * card never takes focus from the composer.
 */

import { test, expect, describe, vi, beforeEach, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { buildConsole } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-console.js';
import { emitChat } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js';
import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';
import {
  initChatAnnounce,
  replyLead,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-announce.js';

/** @type {Array<() => void>} */
let stops = [];
/** @type {HTMLElement} */
let container;
/** @type {HTMLElement} */
let messages;
/** @type {HTMLTextAreaElement} */
let textarea;
/** @type {HTMLElement} */
let region;
/** @type {ReturnType<typeof createChatRenderer>} */
let r;
/** @type {any} */
let answerPermission;
/** @type {MutationObserver} */
let observer;
/** Every text the live region was given since the last read, in order. */
let heard = /** @type {string[]} */ ([]);

/** @param {MutationRecord[]} records */
const addedText = (records) =>
  records.flatMap((record) => [...record.addedNodes].map((node) => node.textContent ?? ''));

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('marked', { parse: /** @param {string} t */ (t) => t });
  vi.stubGlobal('DOMPurify', { sanitize: /** @param {string} h */ (h) => h });
  heard = [];

  const handles = buildConsole();
  messages = handles.messages;
  textarea = handles.textarea;
  container = document.createElement('div');
  container.id = 'operator-container';
  container.append(handles.bar, messages, handles.overlay, handles.inputArea);
  document.body.append(container);

  stops.push(initChatAnnounce({ container }));
  region = qs(container, '.op-announce');

  observer = new MutationObserver((records) => heard.push(...addedText(records)));
  observer.observe(region, { childList: true });

  answerPermission = vi.fn(async () => {});
  r = createChatRenderer(messages, { answerPermission });
});

afterEach(() => {
  observer.disconnect();
  for (const stop of stops) stop();
  stops = [];
  container.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

/**
 * Let all queued work run: the reply a turn end defers, the region's mutation
 * records, and every step of a card's answer. No fake time passes.
 */
const settle = () => vi.advanceTimersByTimeAsync(0);

/** What the region was told since the last call, in order. */
async function spoken() {
  await settle();
  return heard.splice(0);
}

/** A tool call the agent starts, as the live stream draws it. */
const call = (/** @type {string} */ id, /** @type {string} */ name = 'Bash') => ({
  type: 'tool_use',
  tool_name: name,
  tool_name_raw: name,
  tool_use_id: id,
  input: { command: 'ls' },
});

/** A permission request filed under a call. */
const ask = (/** @type {string} */ toolUseId, /** @type {string} */ requestId, extra = {}) => ({
  type: 'permission_request',
  request_id: requestId,
  kind: 'tool',
  tool_name: 'Bash',
  tool_name_raw: 'Bash',
  tool_use_id: toolUseId,
  input: { command: 'ls' },
  reason: null,
  can_allow_session: false,
  ...extra,
});

/** A question from AskUserQuestion: one question with two options. */
const question = (/** @type {string} */ requestId) => ({
  type: 'permission_request',
  request_id: requestId,
  kind: 'question',
  input: {
    questions: [
      {
        question: 'Which lattice?',
        header: 'Lattice',
        multiSelect: false,
        options: [
          { label: 'Top-off', description: '' },
          { label: 'Hybrid', description: '' },
        ],
      },
    ],
  },
});

/** A plan the agent wants approved before it acts. */
const plan = (/** @type {string} */ requestId) => ({
  type: 'permission_request',
  request_id: requestId,
  kind: 'plan',
  input: { plan: '1. Read the orbit.' },
});

/** The newest live approval card on the page. */
const lastApproval = () => {
  const cards = container.querySelectorAll('.op-approval');
  return /** @type {HTMLElement} */ (cards[cards.length - 1]);
};

/** A live tool approval for call *id*, drawn on the page. */
function askLive(id = 't1', requestId = 'req-1', extra = {}) {
  r.handleEvent(call(id));
  r.handleEvent(ask(id, requestId, extra));
}

// ---- What the screen reader is told ---- //

describe('what the screen reader is told', () => {
  test('a live reply is spoken once, when its turn ends, not while it streams', async () => {
    r.beginTurn();
    r.handleEvent({ type: 'delta', content: 'The beam ' });
    r.handleEvent({ type: 'delta', content: 'is stable.' });
    r.handleEvent({ type: 'text', content: 'The beam is stable.' });
    expect(await spoken()).toEqual([]);

    r.handleEvent({ type: 'result' });
    expect(await spoken()).toEqual(['Agent replied: The beam is stable.']);

    r.endTurn();
    expect(await spoken()).toEqual([]);
  });

  test('a reply that ends in a turn of its own is spoken as its turn ends', async () => {
    r.beginTurn();
    r.handleEvent({ type: 'text', content: 'Checking the orbit.' });
    r.handleEvent(call('t1'));
    r.handleEvent({ type: 'result', is_error: false });
    expect(await spoken()).toEqual(['Agent replied: Checking the orbit.']);

    r.beginTurn();
    r.handleEvent({ type: 'text', content: 'The orbit is flat.' });
    r.handleEvent({ type: 'result' });
    expect(await spoken()).toEqual(['Agent replied: The orbit is flat.']);
  });

  test.each([
    ["a subagent's text", { type: 'text', content: 'Inside.', parent_tool_use_id: 'sub-1' }],
    ['a turn of tool calls alone', call('t1')],
  ])('%s is not a reply, so the turn ends silently', async (_name, event) => {
    r.beginTurn();
    r.handleEvent(event);
    r.endTurn();
    expect(await spoken()).toEqual([]);
  });

  test('a terminal error is spoken, and the reply of the turn it ends is not', async () => {
    r.beginTurn();
    r.handleEvent({ type: 'text', content: 'Partial answer.' });
    r.handleEvent({ type: 'error', message: 'API error: rate limited' });
    expect(await spoken()).toEqual(['Error: API error: rate limited']);
  });

  test('an assistant-message error is spoken, and the turn it sits in still speaks its reply', async () => {
    r.beginTurn();
    r.handleEvent({
      type: 'error',
      message: 'Partial message',
      error_type: 'AssistantMessageError',
    });
    expect(await spoken()).toEqual(['Error: Partial message']);

    r.handleEvent({ type: 'text', content: 'Recovered.' });
    r.handleEvent({ type: 'result' });
    expect(await spoken()).toEqual(['Agent replied: Recovered.']);
  });

  test('a tool approval is spoken by the tool it asks about, once', async () => {
    r.beginTurn();
    askLive();
    expect(await spoken()).toEqual(['Approval needed: Bash']);

    // A reconnect re-delivers the same request; it is not a second request.
    r.handleEvent(ask('t1', 'req-1'));
    expect(await spoken()).toEqual([]);
  });

  test.each([
    ['a question', question('q-1'), 'Question from the agent'],
    ['a plan awaiting approval', plan('plan-1'), 'Approval needed: plan'],
    ['an error with no message', { type: 'error' }, 'Error: An error occurred.'],
  ])('%s is spoken as what it is', async (_name, event, said) => {
    r.beginTurn();
    r.handleEvent(event);
    expect(await spoken()).toEqual([said]);
  });

  test('an approval nobody answered is spoken once when it expires, by its tool', async () => {
    r.beginTurn();
    askLive();
    await spoken();

    const expired = {
      type: 'permission_resolved',
      request_id: 'req-1',
      allow: false,
      for_session: false,
      answers: null,
      expired: true,
    };
    r.handleEvent(expired);
    const said = await spoken();
    expect(said).toHaveLength(1);
    expect(said[0]).toMatch(/no answer/i);
    expect(said[0]).toContain('Bash');

    // Re-delivered on a reconnect: not spoken again.
    r.handleEvent(expired);
    expect(await spoken()).toEqual([]);
  });

  test('an ordinary answer from another tab is not spoken', async () => {
    r.beginTurn();
    askLive();
    await spoken();

    r.handleEvent({
      type: 'permission_resolved',
      request_id: 'req-1',
      allow: true,
      for_session: false,
      answers: null,
      expired: false,
    });
    expect(await spoken()).toEqual([]);
  });

  test('a replayed history speaks nothing: not its replies, errors, approvals or questions', async () => {
    r.replay([
      { role: 'user', content: 'read the orbit' },
      {
        role: 'assistant',
        content: '',
        events: [
          call('t0'),
          ask('t0', 'old-1'),
          question('old-q'),
          { type: 'error', message: 'old failure' },
          { type: 'text', content: 'Old answer.' },
          { type: 'result' },
        ],
      },
      { role: 'assistant', content: 'Plain history.' },
    ]);
    expect(await spoken()).toEqual([]);

    // Live events after the replay still speak, so the silence is the replay's.
    r.beginTurn();
    r.handleEvent({ type: 'error', message: 'live failure' });
    expect(await spoken()).toEqual(['Error: live failure']);
  });

  test('a notice the console puts in the log is spoken as it reads', async () => {
    // A failed send draws a system line in a log that is aria-live="off"; the
    // console publishes it as a notice, and this region is what says it.
    emitChat('notice', { text: 'Server busy. Try again in a moment.' });
    expect(await spoken()).toEqual(['Server busy. Try again in a moment.']);
  });

  test('the region is a polite, atomic live region', () => {
    expect(region.getAttribute('aria-live')).toBe('polite');
    expect(region.getAttribute('aria-atomic')).toBe('true');
  });
});

describe('replyLead', () => {
  test.each([
    ['## Title\n\n**Bold** and [a link](http://x) and `code`', 'Title Bold and a link and code'],
    ['See:\n```js\nconst x = 1;\n```\nDone.', 'See: Done.'],
    ['Orbit is flat.', 'Orbit is flat.'],
  ])('%j is read as %j', (markdown, lead) => {
    expect(replyLead(markdown)).toBe(lead);
  });

  test('a long reply is cut near 200 characters, at a word, with an ellipsis', () => {
    const lead = replyLead('word '.repeat(100));
    expect(lead.endsWith('…')).toBe(true);
    expect(lead.length).toBeLessThanOrEqual(201);
    expect(lead.slice(0, -1).endsWith('word')).toBe(true);
  });
});

// ---- Roles and labels ---- //

describe('roles and labels', () => {
  test('the message log is a log, and it does not announce streamed text by itself', () => {
    expect(messages.getAttribute('role')).toBe('log');
    expect(messages.getAttribute('aria-live')).toBe('off');
    expect(messages.getAttribute('aria-label')).toBeTruthy();
  });

  test('the icon-only composer controls each have a name of their own', () => {
    const { cmdBtn, sendBtn, stopBtn } = buildConsole();
    const names = [cmdBtn, sendBtn, stopBtn, textarea].map((el) => el.getAttribute('aria-label'));
    for (const name of names) expect(name).toBeTruthy();
    expect(new Set(names).size).toBe(names.length);
  });

  test('each card is a group named by its own title', () => {
    r.beginTurn();
    askLive();
    r.handleEvent(question('q-1'));
    const titleOf = (/** @type {string} */ selector) => {
      const card = qs(container, selector);
      expect(card.getAttribute('role')).toBe('group');
      return document.getElementById(card.getAttribute('aria-labelledby') ?? '')?.textContent ?? '';
    };
    expect(titleOf('.op-approval')).toContain('Bash');
    expect(titleOf('.op-question')).toBe('Which lattice?');
  });

  test.each([
    ['a tool row', '.op-tool > .op-row-head', '.op-tool > .op-row-body', () => {}],
    [
      'a thinking row',
      '.op-thinking > .op-row-head',
      '.op-thinking > .op-row-body',
      () => r.handleEvent({ type: 'thinking', content: 'Let me see.' }),
    ],
    [
      'a folded run of tool calls',
      '.op-run > .op-row-head',
      '.op-run > .op-row-body',
      () => r.handleEvent(call('t2')),
    ],
  ])(
    '%s opens from the keyboard: its head is a button that reports whether it is open',
    (_name, headSel, bodySel, draw) => {
      r.beginTurn();
      r.handleEvent(call('t1'));
      draw();
      const head = qs(container, headSel);
      expect(head.tagName).toBe('BUTTON');
      expect(head.getAttribute('aria-expanded')).toBe('false');

      // What Enter or Space does on a focused button: a click.
      head.click();
      expect(head.getAttribute('aria-expanded')).toBe('true');
      expect(qs(container, bodySel).hidden).toBe(false);
    }
  );

  test.each([
    [
      'the error block carries no alert role, so the error is spoken once',
      '.op-error-block',
      { type: 'error', message: 'boom' },
    ],
    [
      'the loader is not a live status, so each step of its label stays silent',
      '.op-loader',
      { type: 'tool_use', tool_name: 'Read', tool_use_id: 't1', input: {} },
    ],
  ])('%s', (_name, selector, event) => {
    r.beginTurn();
    r.handleEvent(event);
    expect(qs(container, selector).getAttribute('role')).toBeNull();
  });
});

// ---- The keyboard path ---- //

/**
 * What a browser does for a keypress on a focused element: the keydown goes to
 * the element and bubbles, and a button that is not prevented from doing so is
 * clicked by Enter.
 * @param {HTMLElement} el
 * @param {string} key
 */
function press(el, key) {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
  el.dispatchEvent(event);
  if (!event.defaultPrevented && key === 'Enter' && el instanceof HTMLButtonElement) el.click();
}

describe('the keyboard path', () => {
  test('an approval card holds its controls as native buttons, in visual order', () => {
    r.beginTurn();
    askLive('t1', 'req-1', { can_allow_session: true });
    const card = qs(container, '.op-approval');
    const buttons = [...card.querySelectorAll('button')];
    expect(buttons.map((b) => b.textContent)).toEqual(['Deny', 'Allow for this session', 'Allow']);
    for (const b of buttons) {
      expect(b.disabled).toBe(false);
      expect(b.tabIndex).toBeGreaterThanOrEqual(0);
    }
  });

  test('a question card keeps its controls in order: pager, options, other, skip, submit', () => {
    r.beginTurn();
    r.handleEvent(question('q-1'));
    const card = qs(container, '.op-question');
    const order = [...card.querySelectorAll('button, input')].map((el) => {
      if (el instanceof HTMLInputElement) return 'other-input';
      if (el.classList.contains('op-option')) return qs(el, '.op-option-label').textContent;
      return el.getAttribute('aria-label') ?? el.textContent;
    });
    expect(order).toEqual([
      'Previous question',
      'Next question',
      'Top-off',
      'Hybrid',
      'other-input',
      'Skip',
      'Submit',
    ]);
  });

  test('Enter on a focused Deny button denies, and does not allow the call', async () => {
    r.beginTurn();
    r.handleEvent(call('t1'));
    r.handleEvent(ask('t1', 'req-1'));
    const deny = qs(container, '.op-btn:not(.op-btn-primary)');
    expect(deny.textContent).toBe('Deny');
    deny.focus();
    press(deny, 'Enter');
    await settle();
    expect(answerPermission).toHaveBeenCalledTimes(1);
    expect(answerPermission).toHaveBeenCalledWith('req-1', { allow: false });
  });

  test('Enter on the card itself still allows, and Escape still denies', async () => {
    r.beginTurn();
    askLive('t1', 'req-1');
    press(lastApproval(), 'Enter');
    await settle();
    expect(answerPermission).toHaveBeenLastCalledWith('req-1', { allow: true });

    askLive('t2', 'req-2');
    press(lastApproval(), 'Escape');
    await settle();
    expect(answerPermission).toHaveBeenLastCalledWith('req-2', { allow: false });
  });

  test('the other answer is a named text field, with no radio role in the way, and typing in it chooses its row', () => {
    r.beginTurn();
    r.handleEvent(question('q-1'));
    const row = qs(container, '.op-option-other');
    expect(row.hasAttribute('role')).toBe(false);
    expect(row.hasAttribute('tabindex')).toBe(false);
    const input = qs(row, 'input', HTMLInputElement);
    expect(input.getAttribute('aria-label')).toBeTruthy();
    input.focus();
    expect(document.activeElement).toBe(input);

    expect(row.getAttribute('data-checked')).toBe('false');
    input.value = 'Maybe the hybrid';
    input.dispatchEvent(new Event('input'));
    expect(row.getAttribute('data-checked')).toBe('true');
  });
});

// ---- Focus ---- //

describe('focus', () => {
  test.each([
    ['stays in the composer while it is typed in', { focused: true, disabled: false }, 'composer'],
    [
      'goes to the card when the composer is not in use',
      { focused: false, disabled: false },
      'card',
    ],
    [
      'goes to the card when the composer is disabled for the turn',
      { focused: true, disabled: true },
      'card',
    ],
  ])('a new card: focus %s', (_name, composer, owner) => {
    if (composer.focused) textarea.focus();
    else textarea.blur();
    textarea.disabled = composer.disabled;
    r.beginTurn();
    askLive();
    const expected = owner === 'composer' ? textarea : qs(container, '.op-btn-primary');
    expect(document.activeElement).toBe(expected);
  });

  test('a replayed approval takes no focus, because history is only drawn', () => {
    const focus = vi.spyOn(HTMLElement.prototype, 'focus');
    r.replay([
      {
        role: 'assistant',
        content: '',
        events: [call('t0'), ask('t0', 'old-1')],
      },
    ]);
    expect(focus).not.toHaveBeenCalled();
  });
});
