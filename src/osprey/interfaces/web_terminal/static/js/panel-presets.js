// @ts-check
/* OSPREY Web Terminal — Panel Presets ("Layouts")
 *
 * A preset is a config-defined, named set of panel ids a human applies in one
 * click from the "+" popover's "Layouts" section (or from the command palette).
 * Applying a preset is EXCLUSIVE: exactly the preset's members end up open, and
 * every non-member leaves the rail ("those panels open and the rest close").
 *
 * The click is a one-line request: the preset NAME goes to /api/panel-arrange,
 * the server resolves its members from `web.presets` and broadcasts a single
 * panel_arrange frame, and every client — this one included — applies the
 * arrangement from that echo (panel-placement.js). A human "Layouts" click and
 * an agent `arrange_workspace(preset=...)` call are therefore literally the same
 * server operation, with no local orchestration to drift from it.
 */

import { initPanelAddMenu } from './panel-add-menu.js';
import { arrangePanels } from './panel-commands.js';

/**
 * Apply a config-defined preset by NAME: one arrange request, applied on every
 * client by the panel_arrange handler.
 *
 * Nothing is orchestrated locally. The server resolves the preset's members
 * (filtered fail-safe to known ids), prunes rail membership to them, and
 * broadcasts the arrangement; the echo then opens exactly those tiles and
 * focuses the first healthy one, on every client alike.
 * @param {string} name  a `web.presets` entry name
 */
export function applyPreset(name) {
  arrangePanels({ preset: name });
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
 * Wire the header "+" control (add-panel menu + Layouts).
 *
 * Absorbs the getElementById lookups for ``#panel-add``/``#panel-add-btn``/
 * ``#panel-add-menu`` and the {@link initPanelAddMenu} call that previously
 * lived inline in panel-manager, now including the preset options — so
 * relocating this wiring nets a line reduction there. No-op (returns) if the
 * "+" DOM is absent, so a template without the control degrades gracefully.
 *
 * @param {HeaderControlsDeps} deps
 */
export function wirePanelHeaderControls(deps) {
  const rootEl = document.getElementById('panel-add');
  const buttonEl = document.getElementById('panel-add-btn');
  const menuEl = document.getElementById('panel-add-menu');
  if (!rootEl || !buttonEl || !menuEl) return;
  initPanelAddMenu({
    rootEl,
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
