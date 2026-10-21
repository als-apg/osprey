// @ts-check
/* OSPREY Web Terminal — the Undo notice
 *
 * One short-lived line at the foot of the page that says what a one-click
 * removal just did — "Clock removed · Undo", "SYSTEM removed from the rail ·
 * Undo" — or why it was refused. Every surface that removes something on a
 * single right-click reports through here, so a removal from the header, the
 * status bar, the rail, a tile header or the rail's utility row reads the same
 * and is taken back the same way.
 *
 * WHY A NOTICE AND NOT A CONFIRM. The removals it reports are cheap to repeat
 * and cheap to reverse, and a confirm on each would teach operators to click
 * through confirms. The notice asks nothing; it offers the way back for long
 * enough to catch a mis-click and then gets out of the way.
 *
 * ONE AT A TIME. A second notice replaces the first: the newest removal is the
 * one the operator is looking at, and two stacked Undo buttons would leave it
 * unclear which removal each takes back.
 *
 * IT WAITS FOR THE OPERATOR. The countdown pauses while the pointer is over the
 * notice or the focus is inside it, so reaching for Undo never races the fade.
 *
 * THE UNDO IS THE CALLER'S. This module never knows what was removed. It runs
 * the closure it was handed, once, and closes; whether the restore was refused
 * is the closure's to report, which it does by showing a notice of its own.
 */

/** How long a notice stays up, in milliseconds. */
export const UNDO_NOTICE_MS = 3000;

/** Id of the live notice element, so tests and styles can find it. */
const NOTICE_ID = 'undo-notice';

/**
 * @typedef {object} UndoNoticeOptions
 * @property {string} message - what happened, in a few words
 * @property {(() => unknown) | null} [undo] - the way back; no Undo button without one
 * @property {number} [durationMs] - defaults to {@link UNDO_NOTICE_MS}
 * @property {Document} [doc] - the document to show it in
 */

/** @type {null | {node: HTMLElement, close: () => void}} */
let current = null;

/**
 * Show a notice, replacing any notice already up.
 * @param {UndoNoticeOptions} options
 * @returns {() => void} closes this notice (a no-op once it has closed)
 */
export function showUndoNotice({ message, undo = null, durationMs = UNDO_NOTICE_MS, doc }) {
  const owner = doc ?? document;
  closeUndoNotice();

  const node = owner.createElement('div');
  node.id = NOTICE_ID;
  node.className = 'undo-notice';
  node.setAttribute('role', 'status');
  node.setAttribute('aria-live', 'polite');

  const text = owner.createElement('span');
  text.className = 'undo-notice-text';
  text.textContent = message;
  node.appendChild(text);

  /** @type {ReturnType<typeof setTimeout> | null} */
  let timer = null;
  let hovered = false;
  let focused = false;
  const stop = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  };
  const arm = () => {
    stop();
    if (!hovered && !focused) timer = setTimeout(close, durationMs);
  };

  function close() {
    stop();
    node.remove();
    if (current?.node === node) current = null;
  }

  if (undo) {
    const button = owner.createElement('button');
    button.type = 'button';
    button.className = 'undo-notice-btn';
    button.textContent = 'Undo';
    button.addEventListener('click', () => {
      close();
      undo();
    });
    node.appendChild(button);
  }

  node.addEventListener('pointerenter', () => {
    hovered = true;
    arm();
  });
  node.addEventListener('pointerleave', () => {
    hovered = false;
    arm();
  });
  node.addEventListener('focusin', () => {
    focused = true;
    arm();
  });
  node.addEventListener('focusout', () => {
    focused = false;
    arm();
  });

  (owner.body ?? owner.documentElement).appendChild(node);
  current = { node, close };
  arm();
  return close;
}

/** Close the notice that is up, if any. */
export function closeUndoNotice() {
  current?.close();
}
