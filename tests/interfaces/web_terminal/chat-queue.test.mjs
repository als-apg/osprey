// @ts-check
/**
 * The follow-up queued during a turn (chat-queue.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-queue.test.mjs
 *
 * createQueuedFollowUp holds one slot and draws it: a second message joins the
 * first, a failed send comes back ahead of newer text, the bubble's buttons
 * call the actions they are given. createFollowUpQueue drives it for the
 * console: Enter-while-streaming queues, the end of a turn flushes the slot to
 * the session it was typed for (or returns it to the box when the console has
 * moved), "Interrupt and ask now" interrupts without sending twice, and a send
 * that never reaches the agent keeps the text queued for a retry.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { qs } from '../_support/dom.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const { createQueuedFollowUp, createFollowUpQueue } = await import(`${JS}/chat-queue.js`);

beforeEach(() => {
  document.body.innerHTML = '';
});
afterEach(() => vi.restoreAllMocks());

const actions = () => ({ askNow: vi.fn(), retry: vi.fn(), remove: vi.fn() });
/** @param {HTMLElement} root @param {string} cls */
const part = (root, cls) => qs(root, `.${cls}`);

describe('createQueuedFollowUp', () => {
  test('is hidden and empty until something is queued', () => {
    const view = createQueuedFollowUp(actions());
    expect(view.root.hidden).toBe(true);
    expect(view.peek()).toBeNull();
    expect(view.failed()).toBe(false);
  });

  test('shows the queued text and the waiting label', () => {
    const view = createQueuedFollowUp(actions());
    view.add('first', 'a');
    expect(view.root.hidden).toBe(false);
    expect(view.root.dataset.state).toBe('waiting');
    expect(part(view.root, 'op-queued-text').textContent).toBe('first');
    expect(part(view.root, 'op-queued-label').textContent).toContain('Queued');
    expect(part(view.root, 'op-queued-action').textContent).toBe('Interrupt and ask now');
  });

  test('a second message joins the first on its own line, one slot', () => {
    const view = createQueuedFollowUp(actions());
    view.add('first', 'a');
    view.add('second', 'b');
    expect(view.peek()).toEqual({ text: 'first\nsecond', key: 'b' });
  });

  test('take empties the slot, returns it, and clears failure and asking', () => {
    const view = createQueuedFollowUp(actions());
    view.add('x', 'a');
    view.asking(true);
    expect(view.take()).toEqual({ text: 'x', key: 'a' });
    expect(view.peek()).toBeNull();
    expect(view.root.hidden).toBe(true);
    expect(view.take()).toBeNull();
    view.add('y', 'a');
    expect(part(view.root, 'op-queued-action').textContent).toBe('Interrupt and ask now');
  });

  test('asking disables the button and relabels it', () => {
    const view = createQueuedFollowUp(actions());
    view.add('x', 'a');
    view.asking(true);
    const btn = /** @type {HTMLButtonElement} */ (part(view.root, 'op-queued-action'));
    expect(btn.disabled).toBe(true);
    expect(btn.textContent).toBe('Interrupting…');
    view.asking(false);
    expect(btn.disabled).toBe(false);
  });

  test('fail restores the held text ahead of newer text and offers Retry', () => {
    const view = createQueuedFollowUp(actions());
    view.add('newer', 'b');
    view.fail({ text: 'older', key: 'a' });
    expect(view.peek()).toEqual({ text: 'older\nnewer', key: 'a' });
    expect(view.failed()).toBe(true);
    expect(view.root.dataset.state).toBe('failed');
    expect(part(view.root, 'op-queued-label').textContent).toBe('Not sent.');
    expect(part(view.root, 'op-queued-action').textContent).toBe('Retry');
  });

  test('fail on an empty slot shows the held message; retry returns to waiting', () => {
    const view = createQueuedFollowUp(actions());
    view.fail({ text: 'lost', key: 'a' });
    expect(view.peek()).toEqual({ text: 'lost', key: 'a' });
    view.retry();
    expect(view.failed()).toBe(false);
    expect(view.root.dataset.state).toBe('waiting');
  });

  test('the action button asks now, or retries after a failure; remove calls remove', () => {
    const a = actions();
    const view = createQueuedFollowUp(a);
    view.add('x', 'k');
    qs(view.root, '.op-queued-action').click();
    expect(a.askNow).toHaveBeenCalledTimes(1);
    view.fail({ text: 'y', key: 'k' });
    qs(view.root, '.op-queued-action').click();
    expect(a.retry).toHaveBeenCalledTimes(1);
    qs(view.root, '.op-queued-remove').click();
    expect(a.remove).toHaveBeenCalledTimes(1);
  });
});

describe('createFollowUpQueue', () => {
  /** @param {Partial<{streaming: boolean, transitioning: boolean, key: string}>} [initial] */
  function setup(initial = {}) {
    const state = { streaming: true, transitioning: false, key: 's1', ...initial };
    const textarea = document.createElement('textarea');
    document.body.append(textarea);
    const deps = {
      textarea,
      autoResize: vi.fn(),
      send: vi.fn(),
      interrupt: vi.fn(() => Promise.resolve()),
      state: () => state,
    };
    const queue = createFollowUpQueue(deps);
    document.body.append(queue.root);
    return { state, textarea, deps, queue };
  }

  test('add queues for the session on screen and has() reports it', () => {
    const { queue } = setup();
    expect(queue.has()).toBe(false);
    queue.add('hello');
    expect(queue.has()).toBe(true);
    expect(queue.root.hidden).toBe(false);
  });

  test('flush sends the slot once the turn has ended, for the same session', () => {
    const { queue, state, deps } = setup();
    queue.add('one');
    queue.add('two');
    queue.flush();
    expect(deps.send).not.toHaveBeenCalled(); // still streaming
    state.streaming = false;
    queue.flush();
    expect(deps.send).toHaveBeenCalledTimes(1);
    expect(deps.send.mock.calls[0][0]).toBe('one\ntwo');
    expect(queue.has()).toBe(false);
  });

  test('flush waits while a hand-off holds the input', () => {
    const { queue, state, deps } = setup({ streaming: false, transitioning: true });
    queue.add('x');
    queue.flush();
    expect(deps.send).not.toHaveBeenCalled();
    state.transitioning = false;
    queue.flush();
    expect(deps.send).toHaveBeenCalledTimes(1);
  });

  test('flush returns the text to the box when the console moved to another session', () => {
    const { queue, state, deps, textarea } = setup();
    queue.add('stray');
    state.streaming = false;
    state.key = 's2';
    queue.flush();
    expect(deps.send).not.toHaveBeenCalled();
    expect(textarea.value).toBe('stray');
    expect(queue.has()).toBe(false);
  });

  test('a send that never reaches the agent keeps the slot, failed, and flush leaves it', () => {
    const { queue, state, deps } = setup();
    queue.add('x');
    state.streaming = false;
    queue.flush();
    const onUnsent = deps.send.mock.calls[0][1];
    onUnsent();
    expect(queue.has()).toBe(true);
    expect(queue.root.dataset.state).toBe('failed');
    queue.flush();
    expect(deps.send).toHaveBeenCalledTimes(1);
    // Retry while idle sends again.
    qs(queue.root, '.op-queued-action').click();
    expect(deps.send).toHaveBeenCalledTimes(2);
    expect(deps.send.mock.calls[1][0]).toBe('x');
  });

  test('Retry during a running turn goes back to waiting instead of sending', () => {
    const { queue, state, deps } = setup({ streaming: false });
    queue.add('x');
    queue.flush();
    deps.send.mock.calls[0][1]();
    state.streaming = true;
    qs(queue.root, '.op-queued-action').click();
    expect(deps.send).toHaveBeenCalledTimes(1);
    expect(queue.root.dataset.state).toBe('waiting');
  });

  test('Interrupt and ask now interrupts the turn and disables the button', () => {
    const { queue, deps } = setup();
    queue.add('now');
    const btn = /** @type {HTMLButtonElement} */ (qs(queue.root, '.op-queued-action'));
    btn.click();
    expect(deps.interrupt).toHaveBeenCalledTimes(1);
    expect(deps.send).not.toHaveBeenCalled();
    expect(btn.disabled).toBe(true);
    expect(btn.textContent).toBe('Interrupting…');
  });

  test('a failed interrupt leaves the slot waiting and re-enables the button', async () => {
    const { queue, deps } = setup();
    deps.interrupt.mockReturnValue(Promise.reject(new Error('down')));
    queue.add('now');
    const btn = /** @type {HTMLButtonElement} */ (qs(queue.root, '.op-queued-action'));
    btn.click();
    await Promise.resolve();
    await Promise.resolve();
    expect(btn.disabled).toBe(false);
    expect(queue.has()).toBe(true);
  });

  test('ask now when no turn runs sends the slot immediately', () => {
    const { queue, state, deps } = setup();
    queue.add('go');
    state.streaming = false;
    qs(queue.root, '.op-queued-action').click();
    expect(deps.interrupt).not.toHaveBeenCalled();
    expect(deps.send).toHaveBeenCalledTimes(1);
    expect(deps.send.mock.calls[0][0]).toBe('go');
  });

  test('remove recalls the slot into the box ahead of what was typed, and focuses it', () => {
    const { queue, textarea, deps } = setup();
    textarea.value = ' typed ';
    queue.add('queued');
    qs(queue.root, '.op-queued-remove').click();
    expect(textarea.value).toBe('queued\ntyped');
    expect(deps.autoResize).toHaveBeenCalled();
    expect(document.activeElement).toBe(textarea);
    expect(queue.has()).toBe(false);
  });

  test('remove does not focus a disabled box', () => {
    const { queue, textarea } = setup();
    textarea.disabled = true;
    queue.add('q');
    qs(queue.root, '.op-queued-remove').click();
    expect(textarea.value).toBe('q');
    expect(document.activeElement).not.toBe(textarea);
  });

  test('recall and returnToInput on an empty slot or empty box', () => {
    const { queue, textarea } = setup();
    queue.recall();
    expect(textarea.value).toBe('');
    queue.returnToInput('back');
    expect(textarea.value).toBe('back');
  });
});
