// @ts-check
/* OSPREY Web Terminal — Config drawer tab gate
 *
 * The client half of `web.config_panel.enabled`. The server side is the real
 * enforcement: with the key off, `/api/config` and `/api/claude-setup` refuse
 * every verb with 403 and `GET /api/panels` never names `config` in any of its
 * panel lists (routes/panels.py). This module is what stops the browser from
 * painting a control for that refusing surface.
 *
 * Why a removal rather than server-rendered markup: the Config tab is STATIC
 * drawer markup in index.html, and index.html's Jinja context is the theme /
 * ui-mode / rail-position bundle resolved in `create_app`'s `/` handler. The
 * panel gate is already published on the payload every boot consumer reads
 * (`GET /api/panels` → `config_panel_enabled`), so gating off that one truth
 * keeps a single source: a deployment that flips the key needs no second
 * server-side render path to agree with the first.
 *
 * The node is REMOVED, not hidden. Every downstream consumer of the Config tab
 * already keys on element presence — `settings.js`'s `openDrawerTab` returns
 * false for a tab it cannot resolve, the command palette gates its rows the
 * way it gates the logout row (`#logout-btn` exists only in multi-user
 * deployments) — so absence is the vocabulary the page already speaks, and it
 * takes the button out of the tab order for free.
 *
 * Timing, stated plainly: the payload arrives one round trip after boot, so
 * there is a brief window in which the tab is still painted. That window is
 * cosmetic only — the routes behind it answer 403 from the first request — and
 * closing it would mean blocking the drawer's wiring on a network read.
 *
 * When the panel is closed because the deployment's config file exists but
 * could not be read, the payload names that file (`config_unreadable_path`)
 * and a notice under the drawer header says so. The tab still goes away — the
 * notice sits beside it rather than in a tab kept alive to hold it — and the
 * path is written as text, because it comes from the server.
 */

/** Drawer-tab id of the Config surface (index.html: the tab button + panel). */
export const CONFIG_TAB_ID = 'tab-config';

/** Id of the notice naming a config file the server could not read. */
export const CONFIG_UNREADABLE_NOTICE_ID = 'config-unreadable-notice';

/**
 * Remove the Config drawer tab when the deployment has gated it off.
 *
 * Only an explicit `false` withdraws the tab. `/api/panels` always sends the
 * flag, so a payload without it is not a panels response and says nothing
 * about the deployment's posture. Neither does a null payload (a failed or
 * hung `/api/panels`), because a read that never landed is not a statement
 * either.
 *
 * @param {any} panelsPayload - the `GET /api/panels` response, or null.
 * @returns {boolean} true when the tab was present and has been removed.
 */
export function applyConfigTabGate(panelsPayload) {
  if (!panelsPayload || panelsPayload.config_panel_enabled !== false) return false;
  return removeConfigTab(document);
}

/**
 * Name the unreadable config file in a notice under the drawer header.
 *
 * Only a non-empty `config_unreadable_path` adds the notice; a null payload,
 * a missing key or an empty string leave the drawer as it is. Applying twice
 * keeps one notice.
 *
 * @param {any} panelsPayload - the `GET /api/panels` response, or null.
 * @returns {boolean} true when a notice is present afterwards.
 */
export function applyConfigUnreadableNotice(panelsPayload) {
  const path = panelsPayload?.config_unreadable_path;
  if (typeof path !== 'string' || !path) return false;
  if (document.querySelector(`#${CONFIG_UNREADABLE_NOTICE_ID}`)) return true;
  const header = document.querySelector('#settings-drawer .drawer-header');
  if (!header) return false;
  const notice = document.createElement('p');
  notice.className = 'drawer-notice';
  notice.id = CONFIG_UNREADABLE_NOTICE_ID;
  notice.setAttribute('role', 'status');
  notice.textContent =
    `Config editing is off: ${path} could not be read. Fix the file and restart the web terminal.`;
  header.after(notice);
  return true;
}

/**
 * Detach the Config tab button and its panel, keeping the drawer coherent.
 *
 * If the removed tab happened to be the active one, the first surviving tab is
 * promoted — a drawer whose only active panel has just been deleted opens
 * blank, which reads as a broken drawer rather than a gated one. The Config
 * tab is not the boot-active tab today (Behavior is), so this is insurance
 * against a future reorder, not a path exercised at boot.
 *
 * @param {ParentNode & { querySelector: typeof document.querySelector }} root
 * @returns {boolean} true when a tab or panel was actually removed.
 */
function removeConfigTab(root) {
  const tab = /** @type {HTMLElement|null} */ (
    root.querySelector(`.drawer-tab[data-tab="${CONFIG_TAB_ID}"]`)
  );
  const panel = /** @type {HTMLElement|null} */ (root.querySelector(`#${CONFIG_TAB_ID}`));
  if (!tab && !panel) return false;

  const wasActive = !!tab && tab.classList.contains('active');
  tab?.remove();
  panel?.remove();
  if (wasActive) promoteFirstTab(root);
  return true;
}

/**
 * Make the first remaining drawer tab (and its panel) the active one.
 *
 * @param {ParentNode & { querySelector: typeof document.querySelector }} root
 * @returns {void}
 */
function promoteFirstTab(root) {
  const next = /** @type {HTMLElement|null} */ (root.querySelector('.drawer-tab'));
  if (!next) return;
  next.classList.add('active');
  const nextId = next.getAttribute('data-tab');
  if (!nextId) return;
  const nextPanel = /** @type {HTMLElement|null} */ (root.querySelector(`#${nextId}`));
  nextPanel?.classList.add('active');
}
