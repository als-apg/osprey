// @ts-check
/*
 * OSPREY Design System — the centred state of a panel or a pane.
 *
 * One look for loading, connecting, empty, error and unreachable (DESIGN.md
 * §12, "Panel chrome standard"): an icon, one message, an optional second
 * line, and at most one action (Retry, or the one obvious next step). Logs and
 * technical details (an env var name, an exit code) do not go here: put them
 * behind an Info icon (`info-popover.js`) or in the tile ⋯. The visual half is
 * `.osprey-state` in primitives.css.
 *
 * @module panel-state
 */

import { icon as buildIcon } from './icons.js';

/**
 * @typedef {object} PanelStateAction
 * @property {string} label the button text
 * @property {(event: MouseEvent) => void} onClick
 */

/**
 * @typedef {object} PanelState
 * @property {string} icon an `icons.js` name
 * @property {string} message one short line
 * @property {string} [detail] an optional second line
 * @property {PanelStateAction} [action] at most one action
 */

/**
 * Replace the content of `host` with a centred state.
 *
 * @param {HTMLElement} host the pane or panel body to fill
 * @param {PanelState} state
 * @returns {HTMLElement} the `.osprey-state` root
 * @throws {Error} on an unknown icon name (from `icons.js`)
 */
export function renderState(host, { icon, message, detail, action }) {
  const root = document.createElement('div');
  root.className = 'osprey-state';
  root.setAttribute('role', 'status');
  root.append(buildIcon(icon));

  const text = document.createElement('p');
  text.className = 'osprey-state-message';
  text.textContent = message;
  root.append(text);

  if (detail) {
    const more = document.createElement('p');
    more.className = 'osprey-state-detail';
    more.textContent = detail;
    root.append(more);
  }

  if (action) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'osprey-btn';
    button.textContent = action.label;
    button.addEventListener('click', (event) => action.onClick(event));
    root.append(button);
  }

  host.replaceChildren(root);
  return root;
}

/**
 * Remove the state `renderState` put in `host`, and nothing else.
 *
 * @param {HTMLElement} host
 * @returns {void}
 */
export function clearState(host) {
  for (const child of Array.from(host.children)) {
    if (child.classList.contains('osprey-state')) child.remove();
  }
}
