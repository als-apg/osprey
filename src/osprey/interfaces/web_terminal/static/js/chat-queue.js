// @ts-check
/* OSPREY Web Terminal — operator chat: the follow-up queued during a turn.
 *
 * The composer stays open while the agent replies, and Enter there does not
 * send: the message waits in one bubble just above the composer and goes out
 * when the turn ends. A second message typed in the meantime joins the first on
 * a line of its own — one slot, not a list. The bubble offers the two things an
 * operator may want instead: cut the turn short and ask now, or take the text
 * back into the box.
 *
 * This module holds the slot and draws it; it sends nothing. chat.js decides
 * when the slot goes out, where its text returns to, and what each button does.
 *
 * @module chat-queue
 */

import { button, elem } from './chat-dom.js';
import { chatIcon } from './chat-icons.js';

const WAITING = 'Queued · sends when the agent finishes';
const FAILED = 'Not sent.';
const ASK_NOW = 'Interrupt and ask now';
const ASKING = 'Interrupting…';
const RETRY = 'Retry';

/**
 * A queued message: its text, and the session it was typed for.
 * @typedef {{ text: string, key: string }} Queued
 */

/**
 * Build the queued-message bubble. Hidden while nothing waits.
 * @param {{ askNow: () => void, retry: () => void, remove: () => void }} actions
 */
export function createQueuedFollowUp(actions) {
  const root = elem('div', 'op-queued');
  root.setAttribute('role', 'group');
  root.setAttribute('aria-label', 'Queued message');
  const body = elem('div', 'op-queued-body');
  // The one line that changes with the slot's state is the one read out.
  const label = elem('p', 'op-queued-label');
  label.setAttribute('aria-live', 'polite');
  const text = elem('p', 'op-queued-text');
  body.append(label, text);
  const action = button('op-queued-action');
  const remove = button('op-queued-remove');
  remove.setAttribute('aria-label', 'Remove from queue, back to the input');
  remove.title = 'Back to the input';
  remove.append(chatIcon('x'));
  root.append(body, action, remove);

  /** @type {Queued | null} */
  let queued = null;
  /** The last send of the slot failed; it waits for a retry. */
  let failed = false;
  /** The operator asked to cut the turn short; the slot goes once it closes. */
  let asking = false;

  action.addEventListener('click', () => (failed ? actions.retry() : actions.askNow()));
  remove.addEventListener('click', () => actions.remove());

  function paint() {
    root.hidden = queued === null;
    root.dataset.state = failed ? 'failed' : 'waiting';
    label.textContent = queued === null ? '' : failed ? FAILED : WAITING;
    text.textContent = queued?.text ?? '';
    action.textContent = failed ? RETRY : asking ? ASKING : ASK_NOW;
    action.disabled = asking && !failed;
  }
  paint();

  return {
    root,
    /** What waits, or null. @returns {Queued | null} */
    peek: () => queued,
    /** Whether the slot's last send failed. */
    failed: () => failed,
    /**
     * Queue *message* for *key*, after anything already waiting.
     * @param {string} message
     * @param {string} key
     */
    add(message, key) {
      queued = queued ? { text: `${queued.text}\n${message}`, key } : { text: message, key };
      paint();
    },
    /** Empty the slot and return what it held. @returns {Queued | null} */
    take() {
      const held = queued;
      queued = null;
      failed = false;
      asking = false;
      paint();
      return held;
    },
    /**
     * A send of *held* never reached the agent: show it again with the failure,
     * ahead of anything queued since.
     * @param {Queued} held
     */
    fail(held) {
      queued = queued ? { text: `${held.text}\n${queued.text}`, key: held.key } : held;
      failed = true;
      asking = false;
      paint();
    },
    /** Go back to waiting for the turn to end. */
    retry() {
      failed = false;
      paint();
    },
    /**
     * Show whether the turn is being cut short.
     * @param {boolean} on
     */
    asking(on) {
      asking = on;
      paint();
    },
  };
}

/**
 * The queue as the console drives it: when the slot goes out, where its text
 * returns to, and what the bubble's buttons do. The console hands in its send
 * and interrupt paths and reads of its state; nothing here touches the transport.
 * @param {object} deps
 * @param {HTMLTextAreaElement} deps.textarea - the box the text returns to
 * @param {() => void} deps.autoResize - refit the box after a change
 * @param {(prompt: string, onUnsent: () => void) => void} deps.send - start a turn
 * @param {() => Promise<unknown>} deps.interrupt - ask the agent to end its turn
 * @param {() => { streaming: boolean, transitioning: boolean, key: string }} deps.state -
 *   whether a turn runs, whether a hand-off holds the input, and the session on screen
 */
export function createFollowUpQueue({ textarea, autoResize, send, interrupt, state }) {
  const streaming = () => state().streaming;
  const idle = () => !state().streaming && !state().transitioning;
  const key = () => state().key;
  const view = createQueuedFollowUp({
    // Interrupt only, never abort the turn's stream: the stream closes once the
    // server has let the turn go, so the send that follows is not refused as a
    // turn still running. A failed interrupt leaves the slot waiting as before.
    askNow: () => {
      if (!streaming()) return void sendNow();
      view.asking(true);
      interrupt().catch(() => view.asking(false));
    },
    retry: () => (idle() ? sendNow() : view.retry()),
    remove: () => {
      recall();
      if (!textarea.disabled) textarea.focus();
    },
  });

  /**
   * Put *text* back in the box, ahead of anything typed since.
   * @param {string} text
   */
  function returnToInput(text) {
    const typed = textarea.value.trim();
    textarea.value = typed ? `${text}\n${typed}` : text;
    autoResize();
  }

  /** Send the slot now; a send that never reaches the agent keeps it queued. */
  function sendNow() {
    const waiting = view.take();
    if (waiting) send(waiting.text, () => view.fail(waiting));
  }

  /** Empty the slot into the box. */
  function recall() {
    const waiting = view.take();
    if (waiting) returnToInput(waiting.text);
  }

  return {
    root: view.root,
    returnToInput,
    recall,
    /** @param {string} text - queue *text* for the session the console is on */
    add: (text) => view.add(text, key()),
    /**
     * The turn ended: send the slot, unless another turn runs, its last send
     * failed, or the console has moved to another session since.
     */
    flush() {
      const waiting = view.peek();
      if (!waiting || view.failed() || !idle()) return;
      if (waiting.key === key()) sendNow();
      else recall();
    },
    has: () => view.peek() !== null,
  };
}
