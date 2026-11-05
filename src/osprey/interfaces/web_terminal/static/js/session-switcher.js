/* OSPREY Web Terminal — Session switcher
 *
 * The Agent tile's header label on both surfaces — the chat's `.op-session-label`
 * and the terminal's `#terminal-label` — names the session the tab is on
 * (`activeRow()`'s title) with a ▾. Activating it opens one shared popover: the
 * operator's sessions (state, age, message count; the active one checked),
 * "+ New session", and under a separator the way to the other surface ("Open
 * terminal" in chat, "Back to chat" on the terminal) where a terminal is offered.
 *
 * Every pick goes through the session model's guarded `switchTo`, so a pick
 * while a chat turn streams asks first, and the active row is a no-op. The
 * title follows the model: a rename, a `session_named` frame, a switch. */

import { closePopover, openPopover } from '/design-system/js/popover.js';
import { terminalOffered } from '/design-system/js/frame-params.js';
import { formatFacilityTime } from '/design-system/js/facility-time.js';
import { icon } from '/design-system/js/icons.js';
import { isTerminalSurface, showChatSurface, showTerminalSurface } from './agent-surface.js';
import { activeRow, refresh, rows, subscribe, switchTo } from './session-model.js';

/** The label's text before the tab is on any session. */
export const FALLBACK_TITLE = 'Agent';

/** The label of the popover's new-session row. */
export const NEW_SESSION_LABEL = '+ New session';

/** @type {HTMLElement | null} */
let menu = null;
/** @type {HTMLElement | null} */
let openedFrom = null;
/** @type {Set<HTMLElement>} */
const labels = new Set();
/** @type {(() => void) | null} */
let unsubscribe = null;

/** @returns {string} the title the labels show. */
export function currentTitle() {
  return activeRow()?.title || FALLBACK_TITLE;
}

/** @param {HTMLElement} label */
function renderLabel(label) {
  const title = document.createElement('span');
  title.className = 'session-switcher-title';
  title.textContent = currentTitle();
  const caret = document.createElement('span');
  caret.className = 'session-switcher-caret';
  caret.setAttribute('aria-hidden', 'true');
  caret.textContent = '▾';
  label.replaceChildren(title, caret);
}

function onModelChange() {
  for (const label of labels) {
    if (label.isConnected) renderLabel(label);
    else labels.delete(label);
  }
  if (menu && !menu.hidden) renderMenu(menu);
}

/**
 * Make a header label the switcher's trigger: title + ▾, a menu button that
 * opens the shared session popover. Idempotent per element.
 * @param {HTMLElement | null} label
 */
export function attachSessionSwitcher(label) {
  if (!label || labels.has(label)) return;
  labels.add(label);
  label.classList.add('session-switcher');
  label.setAttribute('role', 'button');
  label.tabIndex = 0;
  label.setAttribute('aria-haspopup', 'menu');
  label.setAttribute('aria-expanded', 'false');
  label.setAttribute('aria-label', 'Switch session');
  label.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle(label);
  });
  label.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    if (!menu || menu.hidden || openedFrom !== label) toggle(label);
  });
  unsubscribe ??= subscribe(onModelChange);
  renderLabel(label);
}

/** @param {HTMLElement} label */
function toggle(label) {
  const pop = ensureMenu();
  if (!pop.hidden && openedFrom === label) {
    closePopover(pop);
    return;
  }
  openedFrom = label;
  renderMenu(pop);
  openPopover(pop, label, { place: 'below-end' });
  refresh();
}

/** @returns {HTMLElement} the one shared popover, hidden and in the page. */
function ensureMenu() {
  if (menu?.isConnected) return menu;
  menu = document.createElement('div');
  menu.className = 'osprey-popover session-switcher-menu';
  menu.id = 'session-switcher-menu';
  menu.setAttribute('role', 'menu');
  menu.setAttribute('aria-label', 'Sessions');
  menu.hidden = true;
  document.body.append(menu);
  return menu;
}

/**
 * The row's quiet facts: state, age, message count.
 * @param {import('./session-model.js').SessionRow & {last_ts?: string|null, message_count?: number}} row
 */
function rowFacts(row) {
  const facts = [];
  if (row.state === 'live') facts.push('live');
  if (row.last_ts) facts.push(relativeTime(row.last_ts));
  if (typeof row.message_count === 'number') {
    facts.push(`${row.message_count} msg${row.message_count === 1 ? '' : 's'}`);
  }
  return facts.join(' · ');
}

/** @param {HTMLElement} pop */
function renderMenu(pop) {
  const current = activeRow();
  const items = rows().map((r) => {
    const row = /** @type {any} */ (r);
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'osprey-popover-item session-switcher-row';
    item.setAttribute('role', 'menuitemradio');
    const isActive =
      !!current &&
      (row === current ||
        (row.transcript_id !== null && row.transcript_id === current.transcript_id));
    item.setAttribute('aria-checked', String(isActive));
    if (row.transcript_id) item.dataset.transcriptId = row.transcript_id;
    if (row.session_key) item.dataset.sessionKey = row.session_key;
    const label = document.createElement('span');
    label.className = 'session-switcher-row-title';
    label.textContent = row.title || 'Untitled session';
    const aside = document.createElement('span');
    aside.className = 'osprey-popover-aside';
    aside.textContent = rowFacts(row);
    const check = icon('check');
    check.classList.add('osprey-popover-check');
    item.append(label, aside, check);
    item.addEventListener('click', () => {
      closePopover(pop);
      void switchTo(r, { reason: 'gesture' });
    });
    return item;
  });

  const add = document.createElement('button');
  add.type = 'button';
  add.className = 'osprey-popover-item session-switcher-new';
  add.setAttribute('role', 'menuitem');
  add.textContent = NEW_SESSION_LABEL;
  add.addEventListener('click', () => {
    closePopover(pop);
    void switchTo({ new: true }, { reason: 'gesture' });
  });

  pop.replaceChildren(...items, add, ...surfaceFooter(pop));
}

/**
 * The quiet last row, under a separator: the way to the other surface.
 * @param {HTMLElement} pop
 * @returns {HTMLElement[]}
 */
function surfaceFooter(pop) {
  if (!terminalOffered()) return [];
  const sep = document.createElement('div');
  sep.className = 'osprey-popover-sep';
  sep.setAttribute('role', 'separator');
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'osprey-popover-item session-switcher-surface';
  row.setAttribute('role', 'menuitem');
  const onTerminal = isTerminalSurface();
  row.textContent = onTerminal ? 'Back to chat' : 'Open terminal';
  row.addEventListener('click', () => {
    closePopover(pop);
    if (onTerminal) showChatSurface();
    else showTerminalSurface();
  });
  return [sep, row];
}

/**
 * A human-readable relative time.
 * @param {string} isoString
 */
function relativeTime(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return '';
  const diffMin = Math.floor((Date.now() - date.getTime()) / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return formatFacilityTime(date, { year: 'numeric', month: 'numeric', day: 'numeric' });
}

/** Forget every label and the popover (tests, teardown). */
export function resetSessionSwitcher() {
  unsubscribe?.();
  unsubscribe = null;
  labels.clear();
  menu?.remove();
  menu = null;
  openedFrom = null;
}
