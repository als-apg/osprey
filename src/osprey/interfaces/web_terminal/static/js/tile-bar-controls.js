// @ts-check
/* OSPREY Web Terminal — the tile bar's shared controls
 *
 * The deployment's controls on every SERVICE tile bar, right of the hairline
 * that separates them from the panel's own icons: `web.tile_bar.controls`
 * reaches the page as `<html data-tile-bar-controls="ask_agent reload help">`
 * (web_terminal/app.py), ids in the configured order, and this module turns
 * the list into buttons. They are hub chrome, so a panel cannot ask for or
 * refuse them, and the Agent tile never shows them: its bar is the chat's.
 *
 *   ask_agent — opens the Agent pane, puts the caret in the chat composer
 *     and asks the chat to insert `About the <Title> panel (<deep link>): `
 *     through the `osprey-agent-compose` document event (chat.js inserts it
 *     without sending). While the Agent pane shows the terminal there is no
 *     composer to write into, so the button is disabled and its tooltip says
 *     what to switch.
 *   reload — reloads the panel's iframe (panel-links.js).
 *   help   — opens the panel's documentation page in a new tab; absent when
 *     the deployment or the panel has none (panel-help.js).
 *
 * In a narrow tile these fold into the tile ⋯ after the panel's icons: the
 * fold ladder in tile-header-contrib.js hides them with `.contrib-hidden`,
 * and {@link foldedControlRows} offers the hidden ones as "This panel" rows.
 */

import { icon } from '/design-system/js/icons.js';
import { writeTitle } from '/design-system/js/tooltip.js';
import { isTerminalSurface, onSurfaceChange } from './agent-surface.js';
import { openTerminalPanel } from './dock-workspace.js';
import { panelLink, reloadPanel } from './panel-links.js';
import { helpUrl, openHelp } from './panel-help.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/** The ids `web.tile_bar.controls` may name, with the icon and tooltip of each. */
const CONTROLS = /** @type {const} */ ({
  ask_agent: { icon: 'agent', title: 'Ask the agent about this panel' },
  reload: { icon: 'refresh', title: 'Reload panel' },
  help: { icon: 'help', title: 'Help for this panel' },
});

/** The ask_agent tooltip while the Agent pane shows the terminal. */
export const ASK_AGENT_TERMINAL_TITLE = 'Switch the Agent pane back to chat to ask about a panel';

/** The document event ask_agent raises; `detail.text` is the composer text. */
export const AGENT_COMPOSE_EVENT = 'osprey-agent-compose';

/** @typedef {keyof typeof CONTROLS} ControlId */

/** @type {Map<string, HTMLElement>} the live controls element per service panel id */
const registry = new Map();

/**
 * The configured control ids, in order, unknown ones dropped. Read from the
 * page each call: the attribute is stamped before any script runs.
 * @param {HTMLElement} [root]
 * @returns {ControlId[]}
 */
export function configuredControls(root = document.documentElement) {
  const raw = root.getAttribute('data-tile-bar-controls') ?? '';
  /** @type {ControlId[]} */
  const ids = [];
  for (const id of raw.split(/\s+/)) {
    if (id in CONTROLS && !ids.includes(/** @type {ControlId} */ (id))) {
      ids.push(/** @type {ControlId} */ (id));
    }
  }
  return ids;
}

/**
 * Build a service tile's shared-controls element: the hairline, then one
 * `.osprey-icon-btn[data-control]` per configured id. Registered for the
 * panel so its folded controls can be offered in the tile ⋯; the TileTab
 * hands it to registerContribHost, whose ladder folds it, and unregisters it
 * on dispose.
 * @param {string} panelId  the service panel id
 * @param {() => string} titleOf  the tile's current title (ask_agent names it)
 * @returns {HTMLElement}
 */
export function buildSharedControls(panelId, titleOf) {
  const el = document.createElement('div');
  el.className = 'tile-tab-controls';
  const divider = document.createElement('span');
  divider.className = 'tile-tab-divider';
  divider.setAttribute('aria-hidden', 'true');
  divider.hidden = true;
  el.appendChild(divider);
  for (const id of configuredControls()) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'osprey-icon-btn';
    btn.dataset.control = id;
    btn.appendChild(icon(CONTROLS[id].icon));
    btn.setAttribute('aria-label', CONTROLS[id].title);
    writeTitle(btn, CONTROLS[id].title);
    if (id === 'ask_agent') {
      paintAskAgent(btn);
      btn.addEventListener('click', () => askAgentAbout(panelId, titleOf()));
    } else if (id === 'reload') {
      btn.addEventListener('click', () => reloadPanel(panelId));
    } else {
      // No page to open is no button: the slot does not keep a dead control.
      if (helpUrl(panelId) === null) btn.hidden = true;
      btn.addEventListener('click', () => openHelp(panelId));
    }
    el.appendChild(btn);
  }
  registry.set(panelId, el);
  return el;
}

/**
 * Forget a tile's controls on dispose (only if they are still the
 * registered ones — a rebuild registers the replacement first).
 * @param {string} panelId
 * @param {HTMLElement} el
 */
export function unregisterSharedControls(panelId, el) {
  if (registry.get(panelId) === el) registry.delete(panelId);
}

/**
 * The shared controls the fold ladder hid on a panel's tile, as "This panel"
 * rows for the tile ⋯: label = the control's tooltip, the same disabled
 * state, the same action.
 * @param {string} panelId
 * @returns {MenuItem[]}
 */
export function foldedControlRows(panelId) {
  const el = registry.get(panelId);
  if (!el) return [];
  /** @type {MenuItem[]} */
  const rows = [];
  for (const node of el.querySelectorAll('[data-control].contrib-hidden')) {
    const btn = /** @type {HTMLButtonElement} */ (node);
    if (btn.hidden) continue;
    rows.push({
      label: btn.getAttribute('aria-label') ?? '',
      disabled: btn.disabled,
      run: () => btn.click(),
    });
  }
  return rows;
}

/**
 * Open the Agent pane on the chat and hand the composer a line about the
 * panel, with the same deep link ⋯ › Copy link copies. Nothing is sent: the
 * operator finishes the sentence.
 * @param {string} panelId
 * @param {string} title  the tile's title
 */
export function askAgentAbout(panelId, title) {
  if (isTerminalSurface()) return;
  openTerminalPanel();
  /** @type {HTMLElement | null} */ (
    document.querySelector('#operator-container textarea')
  )?.focus();
  const link = panelLink(panelId);
  const text = link ? `About the ${title} panel (${link}): ` : `About the ${title} panel: `;
  document.dispatchEvent(new CustomEvent(AGENT_COMPOSE_EVENT, { detail: { text } }));
}

/**
 * The ask_agent button follows the Agent pane's surface: disabled, with the
 * reason as its tooltip, while the pane shows the terminal.
 * @param {HTMLButtonElement} btn
 */
function paintAskAgent(btn) {
  const terminal = isTerminalSurface();
  btn.disabled = terminal;
  writeTitle(btn, terminal ? ASK_AGENT_TERMINAL_TITLE : CONTROLS.ask_agent.title);
}

onSurfaceChange(() => {
  for (const el of registry.values()) {
    const btn = el.querySelector('[data-control="ask_agent"]');
    if (btn instanceof HTMLButtonElement) paintAskAgent(btn);
  }
});
