// @ts-check
/* OSPREY Web Terminal — the control-target popover's writes control.
 *
 * One machine's three-position switch (`off` / `asked` / `allowed`), its
 * keyboard, its focus across a rebuild, and the verdict sentence beside it.
 * Split out of control-target-popover.js, which owns the gestures, the
 * confirms and the module state: this file builds DOM and decides nothing
 * about what a committed option does — the popover hands that in as `onMove`.
 *
 * The control is a `radiogroup` whose `aria-checked` is always the STORED
 * position — never the focus and never a dialog's question. Arrows and
 * Home/End move focus only (roving tabindex, skipping disabled options);
 * Space, Enter or a click commit.
 */

import {
  LIFTED_BY_TURN_ON,
  POSITIONS,
  displayName,
  lockReason,
  statePhrase,
} from './control-target-facts.js';
import { kindAttr } from './control-target-chip.js';

/** @param {string} tag @param {string} [cls] @param {string} [text] */
export function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * A button, typed and with its `type` set. Every button here lives inside no
 * form, but an unset `type` is `submit` and a stray form anywhere above would
 * make each of these navigate.
 * @param {string} cls @param {string} text @returns {HTMLButtonElement}
 */
export function button(cls, text) {
  const node = /** @type {HTMLButtonElement} */ (el('button', cls, text));
  node.type = 'button';
  return node;
}

/**
 * The control that both states and changes a machine's writes.
 *
 * `aria-checked` follows `row.switch`, the stored position: a capped row's
 * `allowed` stays checked and disabled while the word beside it reads `asked`,
 * and an unreadable row checks nothing. Each armed option asks
 * {@link lockReason} about itself, so the tier's ceiling holds `allowed` alone;
 * `off` is never asked, because no lock refuses a stop.
 *
 * Native `disabled` keeps an option out of the focus order, so the tab stop is
 * the checked option when it is enabled, else the next one down: `off` at the
 * latest.
 * @param {any} row
 * @param {any} state
 * @param {string} word
 * @param {{target: string, position: string}|null} pending  the widening whose
 *   confirm is on screen, if any: its option reads `data-pending`
 * @param {(position: 'off'|'asked'|'allowed', reshow: boolean) => void} onMove
 *   what committing an option does
 * @returns {HTMLElement}
 */
export function renderSwitch(row, state, word, pending, onMove) {
  const name = displayName(row, kindAttr(row));
  const stored = POSITIONS.indexOf(row.switch);
  /** @param {'off'|'asked'|'allowed'} position */
  const lockOf = (position) => (position === 'off' ? null : lockReason(row, state, position));

  const wrap = el('span', 'ctc-switchwrap');
  wrap.dataset.state = word;
  wrap.append(el('span', 'ctc-switch-word', 'Writes'));

  const group = el('span', 'ctc-positions');
  group.setAttribute('role', 'radiogroup');
  group.setAttribute('aria-label', `${statePhrase(word)} for ${name}`);
  group.addEventListener('keydown', (event) => onOptionKeydown(event, group, onMove));

  // The tab stop: the stored position, or the nearest one below it that no
  // lock holds.
  /** @type {'off'|'asked'|'allowed'} */
  let stop = 'off';
  for (let rank = Math.max(stored, 0); rank >= 0; rank -= 1) {
    if (!lockOf(POSITIONS[rank])) {
      stop = POSITIONS[rank];
      break;
    }
  }
  for (const [rank, position] of POSITIONS.entries()) {
    const lock = lockOf(position);
    const option = button('ctc-position', position);
    option.setAttribute('role', 'radio');
    option.setAttribute('aria-checked', String(rank === stored));
    option.dataset.target = row.target;
    option.dataset.position = position;
    option.disabled = Boolean(lock);
    option.tabIndex = position === stop ? 0 : -1;
    if (pending && pending.target === row.target && pending.position === position) {
      option.dataset.pending = 'true';
    }
    option.title = optionTitle(position, rank, stored, name, lock);
    option.addEventListener('click', (event) => {
      event.stopPropagation();
      onMove(position, event.shiftKey);
    });
    group.append(option);
  }
  wrap.append(group);

  const stateWordEl = el('span', 'ctc-switch-state', word);
  const lock = lockOf('asked');
  const capped = row.switch === 'allowed' && word === 'asked';
  const why =
    word === 'locked'
      ? row.write_phrase || lock
      : word === 'off'
        ? lock
        : capped && lockOf('allowed');
  if (why) stateWordEl.title = why;
  wrap.append(stateWordEl);
  return wrap;
}

/**
 * What hovering one option says: why it is held, or what committing it does.
 * @param {string} position
 * @param {number} rank
 * @param {number} stored  the stored position's rank, -1 when unreadable
 * @param {string} name
 * @param {string|null} lock
 * @returns {string}
 */
function optionTitle(position, rank, stored, name, lock) {
  if (lock) return lock;
  if (rank === stored) return `Writes are ${position} for ${name}`;
  if (position === 'off') return `Turn writes off for ${name} — applies now`;
  const outcome = rank > stored ? 'asks first' : 'applies now';
  return `Set writes to ${position} for ${name} — ${outcome}`;
}

/**
 * The radiogroup's keyboard: arrows and Home/End move focus and never POST,
 * Space and Enter commit. Handled on keydown, whose `preventDefault` stops the
 * button's native click from committing the same press a second time.
 * @param {KeyboardEvent} event
 * @param {HTMLElement} group
 * @param {(position: 'off'|'asked'|'allowed', reshow: boolean) => void} onMove
 */
function onOptionKeydown(event, group, onMove) {
  const options = /** @type {HTMLButtonElement[]} */ ([
    ...group.querySelectorAll('.ctc-position:not([disabled])'),
  ]);
  const current = options.indexOf(/** @type {HTMLButtonElement} */ (event.target));
  if (current < 0) return;

  let next = null;
  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
    next = options[(current + 1) % options.length];
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
    next = options[(current - 1 + options.length) % options.length];
  } else if (event.key === 'Home') {
    next = options[0];
  } else if (event.key === 'End') {
    next = options[options.length - 1];
  } else if (event.key !== ' ' && event.key !== 'Enter') {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  if (next) {
    for (const option of options) option.tabIndex = option === next ? 0 : -1;
    next.focus();
  } else if (!event.repeat) {
    const position = /** @type {'off'|'asked'|'allowed'} */ (options[current].dataset.position);
    onMove(position, event.shiftKey);
  }
}

/**
 * The option holding keyboard focus inside *root*, as the two values that
 * survive a rebuild.
 * @param {HTMLElement|null} root
 * @returns {{target: string, position: string}|null}
 */
export function focusedOption(root) {
  const node = document.activeElement;
  if (!(node instanceof HTMLElement) || !root?.contains(node)) return null;
  const { target, position } = node.dataset;
  if (!node.classList.contains('ctc-position') || !target || !position) return null;
  return { target, position };
}

/**
 * Focus one machine's option, or its tab stop when that option is now disabled.
 * Nothing happens when the machine has left the roster.
 * @param {HTMLElement|null} root
 * @param {string} target
 * @param {string} position
 */
export function focusOption(root, target, position) {
  const options = [...(root?.querySelectorAll('.ctc-position') ?? [])].filter(
    (node) => node instanceof HTMLButtonElement && node.dataset.target === target
  );
  const buttons = /** @type {HTMLButtonElement[]} */ (options);
  const wanted = buttons.find((node) => node.dataset.position === position && !node.disabled);
  const next = wanted ?? buttons.find((node) => node.tabIndex === 0);
  if (!next) return;
  // Keep the roving tab stop on the focused option, so Tab leaves the group.
  for (const node of buttons) node.tabIndex = node === next ? 0 : -1;
  next.focus();
}

/**
 * The verdict's own sentence for a refused machine, at rest: what stopped the
 * write and who lifts it (C-PHRASE). Shown for a locked row — on and still
 * refused, or unreadable — and for an off row held by a lock no turn-on lifts;
 * never for a row that is merely off, whose switch already says so.
 * @param {any} row
 * @param {string} word
 * @param {string|null} lock
 * @returns {HTMLElement|null}
 */
export function writePhraseLine(row, word, lock) {
  const held = word === 'off' && row.write_reason && !LIFTED_BY_TURN_ON.has(row.write_reason);
  const text = word === 'locked' || held ? row.write_phrase || lock : null;
  if (!text) return null;
  const line = el('div', 'ctc-write-phrase', text);
  line.dataset.state = word;
  return line;
}
