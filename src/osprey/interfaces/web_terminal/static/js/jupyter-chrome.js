// @ts-check
/* OSPREY Web Terminal — Jupyter tile chrome
 *
 * The hub's own chrome for the notebook sidecar's tile. JupyterLab brings its
 * own menu bar, file browser and launcher, so the panel has no toolbar and
 * sends no header contribution; the hub fills its tile bar and its ⋯ instead:
 *
 *   - the tile bar's panel icon: Reload, which reloads the Lab frame
 *     (contributeLocal, so the action comes here, not to the frame);
 *   - ⋯ "This panel": Restart kernel server (after a confirm) and Show server
 *     log (the sidecar's stderr tail in an info popover);
 *   - the tile body while the sidecar starts or after it failed: the design
 *     system's centred state, with Retry on the failed one
 *     (dock-iframe.js's setPanelBodyState, over the tile's frame).
 *
 * The start outcome comes from panel-lifecycle.js (onPanelStart), which also
 * keeps the sidebar row's starting and failed entry as before.
 */

import { fetchJSON } from './api.js';
import { setPanelBodyState } from './dock-iframe.js';
import { PANELS } from './panel-catalog.js';
import { onPanelStart, restartPanelServer, retryPanelStart } from './panel-lifecycle.js';
import { reloadPanel } from './panel-links.js';
import { setServiceMenuExtras } from './panel-menu-policy.js';
import { contributeLocal } from './tile-header-contrib.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { renderState } from '/design-system/js/panel-state.js';

/** @typedef {import('./panel-lifecycle.js').StartState} StartState */

export const JUPYTER_PANEL_ID = 'jupyter';
export const LOG_ENDPOINT = `/api/panels/${JUPYTER_PANEL_ID}/log`;
export const RESTART_ENDPOINT = `/api/panels/${JUPYTER_PANEL_ID}/restart`;
export const RESTART_CONFIRM = 'Restart JupyterLab? Running kernels stop.';

/**
 * The start this page follows now: `null` while the server runs (or never started).
 * @type {StartState | null}
 */
let start = null;
/** @type {(() => void) | null} */
let unfollow = null;

/** @returns {import('./panel-catalog.js').Panel | undefined} the notebook panel, when this page has one */
function jupyterPanel() {
  return PANELS.find((p) => p.id === JUPYTER_PANEL_ID);
}

/**
 * Give the Jupyter tile its chrome. A page served without the notebook panel
 * gets nothing. Called once from app.js, after the panel manager is up and
 * before any config read can answer.
 */
export function initJupyterChrome() {
  if (!jupyterPanel()) return;
  start = null;
  unfollow?.();
  unfollow = onPanelStart(onStart);
  setServiceMenuExtras(JUPYTER_PANEL_ID, menuRows);
  drawBar();
}

/**
 * The tile bar: Reload, disabled while there is no Lab to reload.
 */
function drawBar() {
  const item = { kind: 'icon', id: 'reload', icon: 'refresh', title: 'Reload JupyterLab' };
  contributeLocal(JUPYTER_PANEL_ID, [start ? { ...item, disabled: true } : item], onBarAction);
}

/** @param {string} id */
function onBarAction(id) {
  if (id === 'reload') reloadPanel(JUPYTER_PANEL_ID);
}

/**
 * The ⋯ "This panel" rows, asked at every open.
 * @returns {import('./tile-menu.js').PanelRow[]}
 */
function menuRows() {
  return [
    {
      label: 'Restart kernel server',
      run: confirmRestart,
      disabled: start?.state === 'starting',
    },
    { label: 'Show server log', run: () => void showServerLog() },
  ];
}

/** Restart the sidecar once the operator confirms; the tile shows the start. */
function confirmRestart() {
  const panel = jupyterPanel();
  if (!panel || !window.confirm(RESTART_CONFIRM)) return;
  restartPanelServer(panel, RESTART_ENDPOINT).catch((err) => {
    console.error('Could not restart JupyterLab:', err);
  });
}

/**
 * Where the log popover opens: the tile's ⋯ button, or the sidebar row when
 * the menu came from a row whose panel has no tile.
 * @returns {HTMLElement | null}
 */
function logAnchor() {
  const more = document.querySelector(
    `.tile-tab[data-panel-id="iframe:${JUPYTER_PANEL_ID}"] .tile-tab-more`
  );
  if (more instanceof HTMLElement && more.isConnected) return more;
  const row = document.querySelector(`.panel-rail-button[data-panel-id="${JUPYTER_PANEL_ID}"]`);
  return row instanceof HTMLElement ? row : null;
}

/** Read the sidecar's stderr tail and show it in an info popover. */
export async function showServerLog() {
  const anchor = logAnchor();
  if (!anchor) return;
  let text;
  try {
    const body = await fetchJSON(LOG_ENDPOINT);
    text = typeof body?.text === 'string' && body.text ? body.text : 'The server wrote no log yet.';
  } catch (err) {
    text = `Could not read the server log: ${err instanceof Error ? err.message : String(err)}`;
  }
  openInfoPopover(anchor, { title: 'JupyterLab server log', text });
}

/**
 * Follow the sidecar's start outcome: the tile body and the Reload icon.
 * @param {string} panelId
 * @param {StartState} next
 */
function onStart(panelId, next) {
  if (panelId !== JUPYTER_PANEL_ID) return;
  const was = start?.state ?? 'running';
  start = next.state === 'running' ? null : next;
  paintBody();
  if ((start?.state ?? 'running') !== was) drawBar();
}

/**
 * The server's sentence without its "<Label> failed to start: " lead, which
 * the state's own message already says.
 * @param {string | null} message
 * @returns {string | undefined}
 */
function reasonOf(message) {
  if (!message) return undefined;
  return message.replace(/^.*? failed to start: /, '') || undefined;
}

function paintBody() {
  if (!start) {
    setPanelBodyState(JUPYTER_PANEL_ID, null);
    return;
  }
  if (start.state === 'starting') {
    setPanelBodyState(JUPYTER_PANEL_ID, (host) => {
      renderState(host, { icon: 'refresh', message: 'Starting JupyterLab…' });
    });
    return;
  }
  const detail = reasonOf(start.message);
  setPanelBodyState(JUPYTER_PANEL_ID, (host) => {
    renderState(host, {
      icon: 'triangle-alert',
      message: 'JupyterLab did not start',
      detail,
      action: { label: 'Retry', onClick: retry },
    });
  });
}

function retry() {
  const panel = jupyterPanel();
  if (panel) void retryPanelStart(panel);
}
