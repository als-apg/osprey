// @ts-check
/* OSPREY Web Terminal — Context menus of the chat log
 *
 * Three places, registered with the shell's place dispatcher (place-menu.js):
 *
 *   a message  — a user or assistant entry: Copy (its rendered text) and
 *                Quote in prompt (a Markdown quote at the composer's caret).
 *   a run      — the row of a script run (an `execute` call): the run's rows
 *                from artifact-verbs.js, the ones its sidebar row offers.
 *   the log    — the space between messages: Scroll to latest, Export
 *                conversation, New session.
 *
 * Each leaves the browser its own menu where the browser's is the useful one:
 * inside a link, a button, a form control, a code block or a tool row/card (a
 * run row only inside its opened body's code and cards), and whenever a
 * non-collapsed selection lies in the place — that is a request to
 * copy or look up the selected text. The composer is a text field, which the
 * dispatcher already leaves to the browser.
 *
 * The log and its entries take focus (tabindex -1, focused on click) so the
 * menu key and Shift+F10 reach them.
 *
 * No import-time side effects: chat.js installs it from initChat.
 */

import { registerPlace } from './place-menu.js';
import { heldRun } from './sidebar-artifact-listing.js';
import { runMenuItems } from './artifact-verbs.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/** The entries that carry a message menu. */
const ENTRY = '.op-entry.operator, .op-entry.assistant';

/** The row of a script run, named by its call's id. */
const RUN_ROW = '.op-tool[data-tool="execute"][data-call-id]';

/** Targets inside a run row where the browser keeps its own menu. */
const RUN_DECLINE = 'a, input, select, textarea, pre, code, .op-card';

/** Targets inside which the browser keeps its own menu. */
const DECLINE =
  'a, button, input, select, textarea, label, option, [contenteditable], pre, code, .op-row, .op-card';

/**
 * @typedef {object} ChatMenuOptions
 * @property {HTMLElement} messages - the chat log (.op-messages)
 * @property {HTMLTextAreaElement} textarea - the composer
 * @property {() => void} scrollToBottom - scrolls the log to its latest entry
 * @property {() => void} onNewSession - starts a conversation on a new session
 * @property {() => unknown} onExport - exports the conversation
 */

/**
 * Whether `node` is an element. Not `instanceof`: the chat moves into a pop-out
 * window, whose elements belong to that window's realm.
 * @param {unknown} node
 * @returns {node is HTMLElement}
 */
function isElement(node) {
  return !!node && /** @type {any} */ (node).nodeType === 1;
}

/**
 * The element an event points at (its target, or a text node's parent).
 * @param {Event} event
 * @returns {Element | null}
 */
function targetElement(event) {
  const t = /** @type {any} */ (event?.target);
  if (isElement(t)) return t;
  return isElement(t?.parentElement) ? t.parentElement : null;
}

/**
 * Whether `el` holds a non-collapsed part of its document's selection.
 * @param {Element} el
 * @returns {boolean}
 */
function holdsSelection(el) {
  const sel = el.ownerDocument?.getSelection?.();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return false;
  for (let i = 0; i < sel.rangeCount; i += 1) {
    const range = sel.getRangeAt(i);
    if (range.collapsed) continue;
    if (el.contains(range.commonAncestorContainer)) return true;
    if (range.intersectsNode?.(el)) return true;
  }
  return false;
}

/**
 * Whether the browser should keep its menu for `event` on the place `el`.
 * @param {Element} el
 * @param {Event} event
 * @returns {boolean}
 */
function declines(el, event) {
  const target = targetElement(event);
  const control = target?.closest(DECLINE);
  if (control && el.contains(control)) return true;
  return holdsSelection(el);
}

/**
 * The rendered text of a message: a user's bubble, or an assistant's text
 * blocks joined by a blank line (tool rows, cards and controls left out).
 * @param {Element} entry
 * @returns {string}
 */
export function messageText(entry) {
  if (entry.classList.contains('operator')) {
    const bubble = entry.querySelector('.op-bubble') ?? entry;
    return bubble.textContent ?? '';
  }
  return [...entry.querySelectorAll('.op-entry-body')]
    .map((node) => node.textContent?.trim() ?? '')
    .filter(Boolean)
    .join('\n\n');
}

/**
 * `text` as a Markdown block quote, followed by a blank line.
 * @param {string} text
 * @returns {string}
 */
export function markdownQuote(text) {
  const lines = text.replace(/\s+$/, '').split('\n');
  return `${lines.map((line) => (line ? `> ${line}` : '>')).join('\n')}\n\n`;
}

/**
 * Insert `text` as a quote at the composer's caret (replacing a selection),
 * leave the caret after it, focus the composer and fire `input` so it grows
 * and its listeners see the change.
 * @param {HTMLTextAreaElement} textarea
 * @param {string} text
 */
function quoteInto(textarea, text) {
  const start = textarea.selectionStart ?? textarea.value.length;
  const end = textarea.selectionEnd ?? start;
  const before = textarea.value.slice(0, start);
  const lead = before && !before.endsWith('\n') ? '\n' : '';
  const insert = lead + markdownQuote(text);
  textarea.value = before + insert + textarea.value.slice(end);
  const caret = start + insert.length;
  textarea.focus();
  textarea.setSelectionRange(caret, caret);
  textarea.dispatchEvent(new Event('input', { bubbles: true }));
}

/**
 * Copy `text` to the clipboard; a missing or refusing clipboard does nothing.
 * @param {string} text
 */
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // no clipboard here: the row does nothing
  }
}

/**
 * Register the chat's places and make the log and its entries focusable.
 * @param {ChatMenuOptions} options
 * @returns {() => void} removes the places and the focus listener
 */
export function installChatMenus({ messages, textarea, scrollToBottom, onNewSession, onExport }) {
  messages.setAttribute('tabindex', '-1');

  const unregisterEntry = registerPlace({
    match: (el) => el.matches(ENTRY) && messages.contains(el),
    ariaLabel: (el) => (el.classList.contains('operator') ? 'Your message' : 'Agent message'),
    build: (el, event) => {
      if (declines(el, event)) return [];
      const text = messageText(el);
      return /** @type {MenuItem[]} */ ([
        { label: 'Copy', disabled: !text, run: () => void copyText(text) },
        {
          label: 'Quote in prompt',
          disabled: !text || textarea.disabled,
          run: () => quoteInto(textarea, text),
        },
      ]);
    },
  });

  const unregisterRun = registerPlace({
    match: (el) => el.matches(RUN_ROW) && messages.contains(el),
    ariaLabel: 'Run actions',
    build: (el, event) => {
      const control = targetElement(event)?.closest(RUN_DECLINE);
      if ((control && el.contains(control)) || holdsSelection(el)) return [];
      const execId = /** @type {HTMLElement} */ (el).dataset.callId ?? '';
      return runMenuItems(execId, heldRun(execId));
    },
  });

  const unregisterLog = registerPlace({
    match: (el) => el === messages,
    ariaLabel: 'Conversation',
    build: (el, event) => {
      if (declines(el, event)) return [];
      return /** @type {MenuItem[]} */ ([
        { label: 'Scroll to latest', run: () => scrollToBottom() },
        { divider: true },
        { label: 'Export conversation', run: () => void onExport() },
        { label: 'New session', run: () => onNewSession() },
      ]);
    },
  });

  /**
   * Focus the entry clicked (or the log, between entries) so the menu key
   * reaches it — unless the click went to a control, which takes its own focus,
   * or ended a selection, which focusing would not disturb but means the
   * operator is working with text.
   * @param {MouseEvent} e
   */
  const onClick = (e) => {
    const target = targetElement(e);
    if (!target || !messages.contains(target)) return;
    if (target.closest(DECLINE)) return;
    if (holdsSelection(messages)) return;
    const entry = target.closest(ENTRY);
    const focusable =
      entry && messages.contains(entry) ? /** @type {HTMLElement} */ (entry) : messages;
    if (!focusable.hasAttribute('tabindex')) focusable.setAttribute('tabindex', '-1');
    focusable.focus({ preventScroll: true });
  };
  messages.addEventListener('click', onClick);

  return () => {
    unregisterEntry();
    unregisterRun();
    unregisterLog();
    messages.removeEventListener('click', onClick);
  };
}
