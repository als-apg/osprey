// @ts-check
/*
 * OSPREY Design System — the info popover.
 *
 * Details that are not the panel's content stay off the page (DESIGN.md §12,
 * "Panel chrome standard"): a description, a path, a size, a created date,
 * the issues behind a status chip. An Info icon or a status chip opens them
 * here, under the anchor, right edges aligned.
 *
 * One body-level `.osprey-popover.osprey-info-popover[role=dialog]` serves the
 * page, so a second info popover replaces the first. It is placed, focused and
 * dismissed by `popover.js` (outside press, Escape, resize, window blur), and
 * a second open on the same anchor closes it. Rows are a definition list; a
 * row with `copy: true` gets a small Copy button (`clipboard.js`) that shows a
 * check mark for a moment after a copy, and a row with `onClick` shows its
 * value as a link that closes the popover and runs it. A `notice` is a bar at
 * the top of the card in the warning colour: its text (whose parts may be
 * links, as a row's value) and one action button, which runs its handler and
 * leaves the popover open; the caller sends the content without the notice
 * through the handle's `update`. The visual half is `.osprey-info-popover` in
 * primitives.css.
 *
 * @module info-popover
 */

import { copyText } from './clipboard.js';
import { icon } from './icons.js';
import { closePopover, openPopover } from './popover.js';
import { writeTitle } from './tooltip.js';

const POPOVER_ID = 'osprey-info-popover';
const TITLE_ID = 'osprey-info-popover-title';
/** How long the Copy button shows its result. */
const COPIED_MS = 1500;

/**
 * @typedef {object} InfoRow
 * @property {string} label
 * @property {string | number} value
 * @property {boolean} [copy] add a Copy button for the value
 * @property {() => void} [onClick] show the value as a link that closes the
 *   popover and runs this
 */

/**
 * Text, or a link in the text: it closes the popover and runs `onClick`.
 * @typedef {string | {text: string, onClick: () => void}} InfoNoticePart
 */

/**
 * @typedef {object} InfoNotice
 * @property {InfoNoticePart[]} parts the notice's text, in order
 * @property {string} actionLabel the button at its end
 * @property {() => void} onAction runs on that button; the popover stays open
 */

/**
 * @typedef {object} InfoContent
 * @property {InfoNotice} [notice] a bar at the top of the card
 * @property {string} [title]
 * @property {string} [text] a paragraph; its line breaks are kept
 * @property {InfoRow[]} [rows]
 */

/** @type {HTMLElement | null} */
let popover = null;
/** @type {HTMLElement | null} the anchor the popover was last opened from */
let shownFor = null;

/** @returns {HTMLElement} the page's info popover, created on first use */
function infoPopover() {
  if (popover?.isConnected) return popover;
  popover = document.createElement('div');
  popover.id = POPOVER_ID;
  popover.className = 'osprey-popover osprey-info-popover';
  popover.setAttribute('role', 'dialog');
  // Focusable itself, for focus whose control new content dropped.
  popover.tabIndex = -1;
  popover.hidden = true;
  document.body.append(popover);
  return popover;
}

/**
 * A Copy button for one row.
 * @param {InfoRow} row
 * @returns {HTMLButtonElement}
 */
function copyButton({ label, value }) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'osprey-mini';
  button.setAttribute('aria-label', `Copy ${label}`);
  button.title = 'Copy';
  button.append(icon('copy'));
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let timer;
  button.addEventListener('click', async () => {
    const copied = await copyText(String(value));
    clearTimeout(timer);
    button.replaceChildren(icon(copied ? 'check' : 'copy'));
    writeTitle(button, copied ? 'Copied' : 'Copy failed');
    timer = setTimeout(() => {
      button.replaceChildren(icon('copy'));
      writeTitle(button, 'Copy');
    }, COPIED_MS);
  });
  return button;
}

/**
 * A value shown as a link: it closes the popover, then runs `onClick`.
 * @param {HTMLElement} pop
 * @param {string} text
 * @param {() => void} onClick
 * @returns {HTMLButtonElement}
 */
function link(pop, text, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'osprey-info-popover-link';
  button.textContent = text;
  button.addEventListener('click', () => {
    closePopover(pop);
    onClick();
  });
  return button;
}

/**
 * The notice bar: its text, then its action.
 * @param {HTMLElement} pop
 * @param {InfoNotice} notice
 * @returns {HTMLElement}
 */
function noticeBar(pop, { parts, actionLabel, onAction }) {
  const bar = document.createElement('div');
  bar.className = 'osprey-info-popover-notice';
  const text = document.createElement('p');
  text.className = 'osprey-info-popover-notice-text';
  for (const part of parts) {
    text.append(typeof part === 'string' ? part : link(pop, part.text, part.onClick));
  }
  const action = document.createElement('button');
  action.type = 'button';
  action.className = 'osprey-btn';
  action.textContent = actionLabel;
  action.addEventListener('click', () => onAction());
  bar.append(text, action);
  return bar;
}

/**
 * Fill the popover with `content`, replacing what it showed.
 * @param {HTMLElement} pop
 * @param {InfoContent} content
 */
function fill(pop, { notice, title, text, rows = [] }) {
  pop.replaceChildren();
  if (notice) pop.append(noticeBar(pop, notice));
  if (title) {
    const heading = document.createElement('div');
    heading.className = 'osprey-info-popover-title';
    heading.id = TITLE_ID;
    heading.textContent = title;
    pop.append(heading);
    pop.setAttribute('aria-labelledby', TITLE_ID);
    pop.removeAttribute('aria-label');
  } else {
    pop.removeAttribute('aria-labelledby');
    pop.setAttribute('aria-label', 'Details');
  }
  if (text) {
    const paragraph = document.createElement('p');
    paragraph.className = 'osprey-info-popover-text';
    paragraph.textContent = text;
    pop.append(paragraph);
  }
  if (!rows.length) return;
  const list = document.createElement('dl');
  list.className = 'osprey-info-popover-rows';
  for (const row of rows) {
    const term = document.createElement('dt');
    term.textContent = row.label;
    const detail = document.createElement('dd');
    const value = row.onClick
      ? link(pop, String(row.value), row.onClick)
      : document.createElement('span');
    value.classList.add('osprey-info-popover-value');
    value.textContent = String(row.value);
    detail.append(value);
    if (row.copy) {
      detail.classList.add('osprey-info-popover-copyable');
      detail.append(copyButton(row));
    }
    list.append(term, detail);
  }
  pop.append(list);
}

/**
 * @typedef {object} InfoPopoverHandle
 * @property {() => void} close closes the popover while it still shows for
 *   its anchor
 * @property {(content: InfoContent) => boolean} update shows `content` in
 *   place while the popover still shows for its anchor (focus that was on a
 *   control the new content drops moves to its first control, else to the
 *   popover); false, and nothing changes, once it does not
 */

/**
 * Open the info popover under `anchor`, or close it when it is already open
 * for `anchor`.
 *
 * @param {HTMLElement} anchor the Info icon or status chip; it gets
 *   `aria-expanded` and `aria-controls`
 * @param {InfoContent} [content]
 * @returns {InfoPopoverHandle}
 */
export function openInfoPopover(anchor, content = {}) {
  const pop = infoPopover();
  const showing = () => shownFor === anchor && !pop.hidden;
  /** @type {InfoPopoverHandle} */
  const handle = {
    close() {
      if (showing()) closePopover(pop);
    },
    update(next) {
      if (!showing()) return false;
      const doc = pop.ownerDocument;
      const hadFocus = pop.contains(doc.activeElement);
      fill(pop, next);
      if (hadFocus && !pop.contains(doc.activeElement)) {
        const control = /** @type {HTMLElement | null} */ (pop.querySelector('button, input'));
        (control ?? pop).focus({ preventScroll: true });
      }
      return true;
    },
  };
  const open = !pop.hidden;
  if (open && shownFor === anchor) {
    closePopover(pop);
    return handle;
  }
  if (open) closePopover(pop);
  fill(pop, content);
  shownFor = anchor;
  openPopover(pop, anchor, { place: 'below-end' });
  return handle;
}
