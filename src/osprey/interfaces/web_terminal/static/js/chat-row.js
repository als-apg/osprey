// @ts-check
/**
 * OSPREY Web Terminal — operator chat: the action row.
 *
 * One shape for every row an agent turn is made of — a thought, a tool call, a
 * run of tool calls, a subagent: a glyph in the gutter, a short verb phrase, a
 * dimmed detail, and the status at the row's right edge. The rows of a turn
 * share one gutter column, and operator.css draws a hairline through it, so the
 * turn reads as a timeline.
 *
 * The glyph is decorative; the row's text carries its meaning. Its colour
 * follows the row's `data-state` in CSS: a running row takes the live colour, a
 * settled row goes muted, a failed row takes the error tint. The one row that
 * pulses carries `pulse`; chat-tool-runs.js hands that class to the turn's
 * current action.
 *
 * @module chat-row
 */

import { chatIcon } from './chat-icons.js';
import { button, elem } from './chat-dom.js';

/** @typedef {import('./chat-icons.js').ChatIconName} ChatIconName */

/**
 * A row's state, as its glyph and status show it. `background` is a subagent
 * that reports later; `stopped` one the turn ended under.
 * @typedef {'running' | 'done' | 'error' | 'stopped' | 'background'} RowState
 */

/**
 * @typedef {object} ActionRow
 * @property {HTMLElement} el
 * @property {HTMLButtonElement} head - the row's one line; a click opens the body
 * @property {HTMLElement} glyph - the gutter's glyph box
 * @property {HTMLElement} label - the verb phrase
 * @property {HTMLElement} detail - the dimmed detail after it
 * @property {HTMLElement} status - the slot at the right edge
 * @property {HTMLElement} body - what the row opens to; hidden until it opens
 * @property {(state: RowState) => void} setState
 * @property {(open: boolean) => void} setOpen
 * @property {() => boolean} isOpen
 */

/** @returns {HTMLElement} the circle with one lit arc: a call still running */
export function spinner() {
  return elem('span', 'op-spin');
}

/**
 * The status mark for a state: a spinner while running, then a check, a red x,
 * or the stop square.
 * @param {RowState} state
 * @returns {HTMLElement | SVGSVGElement}
 */
export function statusMark(state) {
  if (state === 'running' || state === 'background') return spinner();
  if (state === 'error') return chatIcon('x');
  if (state === 'stopped') return chatIcon('stop');
  return chatIcon('check');
}

/**
 * Build a row. The caller fills `body` and decides what `status` holds; by
 * default the status follows the state through {@link statusMark}.
 * @param {object} spec
 * @param {string} spec.className - classes the row is styled by (`op-tool`, `op-thinking`, …)
 * @param {string} spec.kind - the row's `data-kind`: `thinking`, `run`, `agent`, or a tool's fold key
 * @param {SVGSVGElement | SVGSVGElement[]} spec.glyph - one icon, or a short stack for a run
 * @param {string} spec.label
 * @param {string} [spec.detail]
 * @param {boolean} [spec.ownStatus] - the caller fills the status slot itself
 * @returns {ActionRow}
 */
export function createRow({ className, kind, glyph, label, detail = '', ownStatus = false }) {
  const el = elem('div', `op-row ${className}`);
  el.dataset.kind = kind;
  const head = button('op-row-head');
  head.setAttribute('aria-expanded', 'false');
  const gutter = elem('span', 'op-row-gutter');
  const icons = Array.isArray(glyph) ? glyph : [glyph];
  const glyphBox = elem('span', icons.length > 1 ? 'op-row-glyph op-row-stack' : 'op-row-glyph');
  glyphBox.append(...icons);
  gutter.append(glyphBox);
  const labelEl = elem('span', 'op-row-label', label);
  const detailEl = elem('span', 'op-row-detail', detail);
  const status = elem('span', 'op-row-status');
  head.append(gutter, labelEl, detailEl, status);
  const body = elem('div', 'op-row-body');
  body.hidden = true;
  el.append(head, body);

  /** @param {boolean} open */
  function setOpen(open) {
    body.hidden = !open;
    el.classList.toggle('open', open);
    head.setAttribute('aria-expanded', String(open));
  }

  /** @param {RowState} state */
  function setState(state) {
    el.dataset.state = state;
    if (!ownStatus) status.replaceChildren(statusMark(state));
  }
  setState('running');

  return {
    el,
    head,
    glyph: glyphBox,
    label: labelEl,
    detail: detailEl,
    status,
    body,
    setState,
    setOpen,
    isOpen: () => !body.hidden,
  };
}

/**
 * A short stack of the same icon, as a run row shows up to four of its calls.
 * @param {ChatIconName} name
 * @param {number} count
 * @returns {SVGSVGElement[]}
 */
export function iconStack(name, count) {
  const n = Math.max(1, Math.min(4, count));
  return Array.from({ length: n }, () => chatIcon(name));
}
