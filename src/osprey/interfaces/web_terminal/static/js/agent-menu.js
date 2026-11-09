// @ts-check
/* OSPREY Web Terminal — the Agent pane's rows in its tile ⋯
 *
 * The Agent pane's bar holds only the session switcher and `+` (agent-bar.js).
 * Its other actions are rows in the Agent tile's ⋯, section "Agent":
 * Export conversation (chat-export.js) and Desktop notifications
 * (chat-attention.js). Each chat feature registers its own row here, and this
 * module hands the set to the hub's menu (panel-menu-policy.js) under the
 * Agent pane's id, in one fixed order whatever order the features started in.
 * The docked bar heads the section with Switch to terminal / chat, which the
 * hub builds itself (the session switcher's handler, so the row is not
 * registered here); the pane's own window shows these rows alone.
 *
 * @module agent-menu
 */

import { TERMINAL_RAIL_ID } from './panel-catalog.js';
import { setServiceMenuExtras } from './panel-menu-policy.js';

/** @typedef {import('./tile-menu.js').PanelRow} PanelRow */

/** The keys of the pane's rows, in the order the ⋯ lists them. */
const ORDER = /** @type {const} */ (['export', 'notifications']);

/** @typedef {typeof ORDER[number]} AgentMenuKey */

/** @type {Map<AgentMenuKey, () => PanelRow | null>} */
const rows = new Map();

/**
 * Put one of the Agent pane's rows in its tile ⋯, or take it out with null.
 * `row()` is asked at every open, so the row follows the pane (the surface
 * showing, the browser's permission); a null answer leaves it out. The hub
 * learns of the pane's rows with the first one and forgets them with the last.
 * @param {AgentMenuKey} key
 * @param {(() => PanelRow | null) | null} row
 */
export function setAgentMenuRow(key, row) {
  if (row) rows.set(key, row);
  else rows.delete(key);
  setServiceMenuExtras(TERMINAL_RAIL_ID, rows.size ? agentMenuRows : null);
}

/**
 * The rows the Agent pane's ⋯ shows now, in the fixed order.
 * @returns {PanelRow[]}
 */
export function agentMenuRows() {
  return ORDER.flatMap((key) => {
    const row = rows.get(key)?.();
    return row ? [row] : [];
  });
}
