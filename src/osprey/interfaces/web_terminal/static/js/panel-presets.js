// @ts-check
/* OSPREY Web Terminal — Panel Presets ("Layouts")
 *
 * A preset is a config-defined, named set of panels a human applies in one
 * click from the "+" popover's "Layouts" section (or from the command
 * palette). Applying a preset is EXCLUSIVE: the preset's panels replace the
 * dock's service tiles, and every non-member leaves the rail ("those panels
 * open and the rest close").
 *
 * The click is a one-line request: the preset NAME goes to
 * /api/panel-op/open-arrangement, the server resolves its members from
 * `web.presets` and relays one `panel_op` frame to the current dock
 * client, whose op runner (panel-placement.js) applies it and answers with the
 * resulting layout; the server then prunes the rail. A human "Layouts" click
 * and an agent `arrange_workspace(preset=...)` call are therefore literally the
 * same operation, with no local orchestration to drift from it.
 */

import { initPanelAddMenu } from './panel-add-menu.js';
import { requestPanelOp } from './panel-commands.js';

/**
 * Apply a config-defined preset by NAME: one open-arrangement op, run by the
 * current dock client's op runner like any agent op.
 *
 * Nothing is orchestrated locally. The server resolves the preset's members
 * (filtered fail-safe to known ids), the dock client applies the arrangement,
 * and the server prunes rail membership to its members.
 * Fire-and-forget: a refusal changes nothing and never rejects.
 * @param {string} name  a `web.presets` entry name
 */
export function applyPreset(name) {
  void requestPanelOp('open-arrangement', { name });
}

/**
 * @typedef {object} HeaderControlsDeps
 * @property {() => {id: string, label: string}[]} getHiddenPanels - known-but-hidden panels, tab order
 * @property {() => boolean} allowUrlPanels - whether runtime URL registration is on
 * @property {(id: string) => void} onShowPanel - reveal + focus a hidden panel
 * @property {(fields: {id: string, label: string, url: string}) => Promise<{ok: boolean, error?: string}>} onRegisterUrl
 * @property {() => {name: string, panels: string[]}[]} getPresets - config-defined layouts, in config order
 * @property {(name: string) => void} onApplyPreset - apply a named layout exclusively
 */

/**
 * Wire the sidebar's "+ Add panel" control (add-panel menu + Layouts).
 *
 * Absorbs the getElementById lookups for ``#panel-add-btn``/``#panel-add-menu``
 * and the {@link initPanelAddMenu} call that previously lived inline in
 * panel-manager, now including the preset options — so relocating this wiring
 * nets a line reduction there. No-op (returns) if the "+" DOM is absent, so a
 * template without the control degrades gracefully.
 *
 * @param {HeaderControlsDeps} deps
 */
export function wirePanelHeaderControls(deps) {
  const buttonEl = document.getElementById('panel-add-btn');
  const menuEl = document.getElementById('panel-add-menu');
  if (!buttonEl || !menuEl) return;
  initPanelAddMenu({
    buttonEl: /** @type {HTMLButtonElement} */ (buttonEl),
    menuEl,
    getHiddenPanels: deps.getHiddenPanels,
    allowUrlPanels: deps.allowUrlPanels,
    onShowPanel: deps.onShowPanel,
    onRegisterUrl: deps.onRegisterUrl,
    getPresets: deps.getPresets,
    onApplyPreset: deps.onApplyPreset,
  });
}
