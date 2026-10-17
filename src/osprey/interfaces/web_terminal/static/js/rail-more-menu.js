// @ts-check
/* OSPREY Web Terminal — the rail's ⋯ More menu, and the way out of the rail
 *
 * The rail's utility cluster holds what an operator reaches for often: Docs and
 * Feedback. Everything reached for rarely lives one click further in, behind
 * ⋯ More: the tour, the activity log, and the SYSTEM panel when this deployment
 * serves it but has not put it on the rail. This module owns that popover and
 * the gesture that sends a control into it.
 *
 * ONE LOOP, ONE OWNER. Docs and Feedback each hide from their own right-click
 * menu ("Hide Docs"), and come back from a "Show Docs" row in ⋯ More. The two
 * halves are kept together here so the way out and the way back cannot drift.
 * ⋯ More itself has no hide gesture: it is where every hidden control returns
 * from, so hiding it would be a trap.
 *
 * WHERE A HIDDEN CONTROL'S STATE IS READ. The flags are the bar layout's
 * `rail_docs_visible` / `rail_feedback_visible`, written through
 * bar-customize.js. Their reflection on `<html>` — `data-rail-docs` and
 * `data-rail-feedback`, stamped by the server and kept live by bar-host.js — is
 * what the stylesheet withdraws the controls by, and it is what this menu reads
 * too. The menu and the stylesheet therefore cannot disagree about whether a
 * control is on screen, and the answer is right from the first paint, before
 * the layout document has been adopted.
 *
 * WHAT HIDING MEANS. The control is withdrawn by CSS and nothing else. The
 * Feedback button in particular stays in the DOM: the status bar's Feedback
 * item and the feedback boot forward to it by id. A Docs anchor this deployment
 * has no `web.docs_url` for is `hidden` already (rail-utility.js), and "Show
 * Docs" is offered only where there is a link to show.
 *
 * SYSTEM. The SYSTEM panel is enabled wherever the deployment context says
 * `systemHealthAvailable`, rail member or not; the status bar's health chip
 * opens it from there. The row here opens it through the same verb that
 * chip's "Open SYSTEM" uses. While the panel is not a rail member the row also
 * carries a pin (and a right-click "Pin to rail"), which adds it to the rail.
 * Taking it off again is the rail entry's own "Remove from rail".
 *
 * The popover follows panel-add-menu.js's grammar: rebuilt on every open so
 * every row is current, closed by an outside click, Escape (focus returns to
 * the button) or a pick, arrow keys between rows. It opens beside the column,
 * bottom-aligned to the button, or below the strip under a top rail; the
 * stylesheet does the placing.
 */

import { openContextMenu } from './panel-context-menu.js';
import { setPanelVisibility } from './panel-commands.js';
import { openPanelOffRail } from './panel-manager.js';
import { getEntry } from './panel-rail.js';
import { deploymentContext } from './bar-sync.js';
import { setRailUtilityVisible } from './bar-customize.js';
import { showUndoNotice } from './undo-notice.js';
import { startTour } from './tour.js';
import { openActivityLog } from './activity-log-link.js';

/** @typedef {import('./bar-customize.js').RailUtility} RailUtility */

/** The SYSTEM panel's id. */
export const SYSTEM_PANEL_ID = 'system-health';

/**
 * The hideable utility controls: the element each one is, the word the
 * operator sees on it, and the `<html>` dataset key that reflects its flag.
 * @type {Record<RailUtility, {id: string, label: string, flag: 'railDocs' | 'railFeedback'}>}
 */
const UTILITIES = {
  docs: { id: 'panel-docs-link', label: 'Docs', flag: 'railDocs' },
  feedback: { id: 'panel-feedback-btn', label: 'Feedback', flag: 'railFeedback' },
};

const SVG_NS = 'http://www.w3.org/2000/svg';

/** One path per row mark, drawn on the rail's 16-unit grid. */
const ICONS = {
  docs: 'M8 4.7C6.5 3.6 4.6 3.2 2.6 3.4L2.6 11.6C4.6 11.4 6.5 11.8 8 12.9M8 4.7C9.5 3.6 11.4 3.2 13.4 3.4L13.4 11.6C11.4 11.4 9.5 11.8 8 12.9M8 4.7L8 12.9',
  feedback: 'M4.6 2.7H11.4A2.1 2.1 0 0 1 13.5 4.8V9.3A2.1 2.1 0 0 1 11.4 11.4H7.0L3.5 14.1L4.4 11.4A2.1 2.1 0 0 1 2.5 9.3V4.8A2.1 2.1 0 0 1 4.6 2.7Z',
  system: 'M2 8H4.5L6 5L8 11L10 3L11.5 8H14',
  tour: 'M8 2A6 6 0 1 1 8 14A6 6 0 1 1 8 2ZM10.5 5.5L9 9L5.5 10.5L7 7Z',
  activity: 'M2 8.5H5L6.5 4.5L9.5 11.5L11 8.5H14',
  pin: 'M6 2.5H10M7 2.5V6.5L4.5 9H11.5L9 6.5V2.5M8 9V13.5',
};

/**
 * Whether the rail currently withdraws utility `name`, as the stylesheet sees it.
 * @param {RailUtility} name
 * @returns {boolean}
 */
export function utilityHidden(name) {
  return document.documentElement.dataset[UTILITIES[name].flag] === 'hidden';
}

/**
 * Whether the deployment has utility `name` to show at all. Feedback always
 * does; Docs only once rail-utility.js has given the anchor a target.
 * @param {RailUtility} name
 * @returns {boolean}
 */
function utilityOffered(name) {
  const node = document.getElementById(UTILITIES[name].id);
  return node instanceof HTMLElement && !node.hidden;
}

/**
 * Withdraw a utility control and offer it back. A refused write is reported by
 * bar-customize.js itself, so the notice is shown only for a hide that landed.
 * @param {RailUtility} name
 * @returns {Promise<void>}
 */
export async function hideUtility(name) {
  if (!(await setRailUtilityVisible(name, false))) return;
  showUndoNotice({
    message: `${UTILITIES[name].label} hidden`,
    undo: () => void setRailUtilityVisible(name, true),
  });
}

/**
 * Open the SYSTEM panel from ⋯ More. The one place the menu's System row
 * reaches the panel manager, through the same verb as the health chip's
 * "Open SYSTEM": the tile opens and the rail stays as it is, so looking at
 * SYSTEM never brings it back onto the rail. Pinning it is a separate choice.
 */
export function openSystemPanel() {
  openPanelOffRail(SYSTEM_PANEL_ID);
}

/** Whether the SYSTEM panel has a rail entry right now. @returns {boolean} */
function systemOnRail() {
  const rail = document.getElementById('panel-rail');
  return rail !== null && getEntry(rail, SYSTEM_PANEL_ID) !== null;
}

/**
 * Hand a context-menu request to `handler` from the mouse AND from the
 * keyboard. macOS synthesises no `contextmenu` from the menu key or Shift+F10,
 * so the keydown opens the menu itself, centred on the control — the same
 * route panel-rail.js gives its entries.
 * @param {HTMLElement} node
 * @param {(x: number, y: number) => boolean} handler - true when a menu opened
 */
function onMenuRequest(node, handler) {
  node.addEventListener('contextmenu', (e) => {
    if (handler(e.clientX, e.clientY)) e.preventDefault();
  });
  node.addEventListener('keydown', (e) => {
    if (e.key !== 'ContextMenu' && !(e.key === 'F10' && e.shiftKey)) return;
    const rect = node.getBoundingClientRect();
    if (handler(rect.left + rect.width / 2, rect.top + rect.height / 2)) e.preventDefault();
  });
}

/**
 * One ⋯ More row: the label, the mark beside it and the verb. The System row
 * carries a pin as well while the panel is off the rail.
 * @typedef {object} MoreRow
 * @property {string} key - stable name, stamped as `data-row`
 * @property {string} label
 * @property {keyof typeof ICONS} icon
 * @property {() => void} run
 * @property {() => void} [pin]
 */

/**
 * The rows ⋯ More offers at this moment, in order.
 * @param {() => string|null|undefined} getSessionId - the terminal's session
 * @returns {MoreRow[]}
 */
export function moreMenuRows(getSessionId) {
  /** @type {MoreRow[]} */
  const rows = [];
  for (const name of /** @type {RailUtility[]} */ (['docs', 'feedback'])) {
    if (utilityHidden(name) && utilityOffered(name)) {
      rows.push({
        key: `show-${name}`,
        label: `Show ${UTILITIES[name].label}`,
        icon: name,
        run: () => void setRailUtilityVisible(name, true),
      });
    }
  }
  if (deploymentContext(document).systemHealthAvailable) {
    /** @type {MoreRow} */
    const system = {
      key: 'system',
      label: 'System',
      icon: 'system',
      run: openSystemPanel,
    };
    if (!systemOnRail()) system.pin = () => setPanelVisibility(SYSTEM_PANEL_ID, true);
    rows.push(system);
  }
  rows.push({ key: 'tour', label: 'Take tour', icon: 'tour', run: () => startTour() });
  rows.push({
    key: 'activity',
    label: 'Activity log',
    icon: 'activity',
    run: () => openActivityLog(getSessionId),
  });
  return rows;
}

/**
 * A 16-unit stroke mark for a menu row.
 * @param {keyof typeof ICONS} name
 * @returns {SVGSVGElement}
 */
function mark(name) {
  const svg = /** @type {SVGSVGElement} */ (document.createElementNS(SVG_NS, 'svg'));
  svg.setAttribute('class', 'rail-more-icon');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', ICONS[name]);
  svg.appendChild(path);
  return svg;
}

/**
 * Wire ⋯ More and the hide gesture on Docs and Feedback. No-op for any piece
 * whose markup the page lacks, so a template without the rail degrades
 * quietly.
 * @param {{getSessionId: () => string|null|undefined}} deps - the terminal's
 *   session, read when the activity log is opened rather than now.
 */
export function initRailMoreMenu({ getSessionId }) {
  for (const name of /** @type {RailUtility[]} */ (['docs', 'feedback'])) {
    const node = document.getElementById(UTILITIES[name].id);
    if (!node) continue;
    onMenuRequest(node, (x, y) => {
      openContextMenu({
        x,
        y,
        anchorEl: node,
        ariaLabel: `${UTILITIES[name].label} actions`,
        items: [{ label: `Hide ${UTILITIES[name].label}`, run: () => void hideUtility(name) }],
      });
      return true;
    });
  }

  const rootEl = document.getElementById('rail-more');
  const buttonEl = document.getElementById('rail-more-btn');
  const menuEl = document.getElementById('rail-more-menu');
  if (!rootEl || !buttonEl || !menuEl) return;

  const isOpen = () => menuEl.classList.contains('open');
  const items = () =>
    /** @type {HTMLElement[]} */ (Array.from(menuEl.querySelectorAll('[role="menuitem"]')));

  /** @param {MouseEvent} e */
  function onDocClick(e) {
    if (e.target instanceof Node && !rootEl?.contains(e.target)) closeMenu();
  }

  /** @param {KeyboardEvent} e */
  function onKeydown(e) {
    // A row's own context menu, open on top, takes the keyboard while it is up.
    if (document.querySelector('.rail-context-menu')) return;
    if (e.key === 'Escape') {
      closeMenu();
      buttonEl?.focus();
      return;
    }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    const rows = items();
    if (!rows.length) return;
    e.preventDefault();
    const at = rows.indexOf(/** @type {HTMLElement} */ (document.activeElement));
    const step = e.key === 'ArrowDown' ? 1 : -1;
    rows[(at + step + rows.length) % rows.length].focus();
  }

  function openMenu() {
    render();
    menuEl?.classList.add('open');
    buttonEl?.setAttribute('aria-expanded', 'true');
    items()[0]?.focus();
    // Capture-phase so an outside click closes before it does anything else.
    document.addEventListener('click', onDocClick, true);
    document.addEventListener('keydown', onKeydown, true);
  }

  function closeMenu() {
    menuEl?.classList.remove('open');
    buttonEl?.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', onDocClick, true);
    document.removeEventListener('keydown', onKeydown, true);
  }

  /**
   * One menu item: closes the menu, then runs the verb.
   * @param {string} className
   * @param {() => void} run
   * @returns {HTMLButtonElement}
   */
  function item(className, run) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = className;
    btn.setAttribute('role', 'menuitem');
    btn.addEventListener('click', () => {
      closeMenu();
      run();
    });
    return btn;
  }

  function render() {
    menuEl?.replaceChildren();
    for (const row of moreMenuRows(getSessionId)) {
      const line = document.createElement('div');
      line.className = 'rail-more-row';
      line.dataset.row = row.key;
      const main = item('rail-more-item', row.run);
      const text = document.createElement('span');
      text.className = 'rail-more-label';
      text.textContent = row.label;
      main.append(mark(row.icon), text);
      line.appendChild(main);
      const pin = row.pin;
      if (pin) {
        const pinBtn = item('rail-more-pin', pin);
        pinBtn.title = 'Pin to rail';
        pinBtn.setAttribute('aria-label', `Pin ${row.label} to rail`);
        pinBtn.appendChild(mark('pin'));
        line.appendChild(pinBtn);
        onMenuRequest(main, (x, y) => {
          openContextMenu({
            x,
            y,
            anchorEl: main,
            ariaLabel: `${row.label} actions`,
            items: [
              {
                label: 'Pin to rail',
                // The row the menu hands focus back to is about to be hidden,
                // so focus goes where Escape would have sent it.
                run: () => {
                  closeMenu();
                  buttonEl?.focus();
                  pin();
                },
              },
            ],
          });
          return true;
        });
      }
      menuEl?.appendChild(line);
    }
  }

  buttonEl.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isOpen()) closeMenu();
    else openMenu();
  });
}
