// @ts-check
/* OSPREY Web Terminal — Direct visits
 *
 * A panel has no frontend of its own, so a page opened at a panel's address
 * comes here with `?open=<id>&at=<url>` (frame-params.js for a panel page,
 * routes/panels.py for a bundle's entry document, which hold the `at` contract).
 * The hub opens that panel on that page, the way an agent's link would.
 *
 * It waits for the session layout to settle, since that session's apply would
 * otherwise close or re-page the panel just opened, and then for the panel:
 * `at` is checked against the panel's mount, so it needs the address its config
 * read reports, and activating an unhealthy panel does nothing.
 */

import {
  getPanelStandaloneUrl,
  isPanelHealthy,
  isPanelKnown,
  navigatePanel,
  showPanel,
} from './panel-manager.js';
import { settled } from './session-layout.js';
import { showUndoNotice } from './undo-notice.js';

/** How long a panel gets to answer before the visit says it has not. */
export const ANSWER_TIMEOUT_MS = 30_000;

const POLL_MS = 100;

/**
 * Read `open` and `at` off this page's address and take them out of it, so a
 * reload is an ordinary reload.
 * @returns {{id: string, at: string | null} | null} the visit, if this is one
 */
export function takeDirectVisit() {
  const url = new URL(window.location.href);
  const id = url.searchParams.get('open');
  const at = url.searchParams.get('at');
  if (id === null && at === null) return null;
  url.searchParams.delete('open');
  url.searchParams.delete('at');
  window.history.replaceState(window.history.state, '', url);
  return id ? { id, at } : null;
}

/**
 * @param {() => boolean} condition
 * @param {number} deadline  a `Date.now()` time
 * @returns {Promise<boolean>} whether it came true in time
 */
async function until(condition, deadline) {
  while (!condition()) {
    if (Date.now() >= deadline) return false;
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
  return true;
}

/**
 * Open the panel a direct visit asked for.
 * @param {{id: string, at: string | null} | null} visit  {@link takeDirectVisit}'s answer
 * @param {Promise<unknown>} panelsReady  the panel manager's init: the panels are known
 * @returns {Promise<void>}
 */
export async function openDirectVisit(visit, panelsReady) {
  if (!visit) return;
  const { id, at } = visit;
  await panelsReady;
  await settled;
  if (!isPanelKnown(id)) {
    showUndoNotice({ message: `${id} is not a panel here.` });
    return;
  }
  const deadline = Date.now() + ANSWER_TIMEOUT_MS;
  const addressed = await until(() => getPanelStandaloneUrl(id) !== null, deadline);
  // Stored until the panel answers: its first frame loads at this page.
  if (addressed && at) navigatePanel(id, at);
  if (!addressed || !(await until(() => isPanelHealthy(id), deadline))) {
    showUndoNotice({ message: `${id} is not answering yet.` });
    return;
  }
  // On the rail as well as on screen: a panel this session had closed comes back.
  showPanel(id);
}
