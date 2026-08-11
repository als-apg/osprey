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
 *
 * {@link computePresetDiff} stays here as the executable statement of those
 * exclusive semantics — the resolution the route performs mirrors its fail-safe
 * filtering, and its tests are where that contract is pinned.
 */

import { initPanelAddMenu } from './panel-add-menu.js';
import { arrangePanels } from './panel-commands.js';

/**
 * @typedef {object} PresetDiff
 * @property {string[]} toShow  - member ids not currently visible
 * @property {string[]} toHide  - visible ids that are not preset members
 * @property {string | null} focus - first known member to focus, or null (no-op guard)
 */

/**
 * Compute the exclusive show/hide diff for applying a preset.
 *
 * Members are first filtered to knownSet (enabled built-ins + custom ids) so a
 * typo'd or disabled id is skipped fail-safe; a preset where no member survives
 * that filtering reports `focus: null`.
 *
 * That last case is ENFORCED server-side, not here: routes/panels.py's
 * `_resolve_preset_tiles` applies the same filtering and rejects an empty
 * result with a 422 naming the valid ids, so the arrangement is never
 * broadcast. Since {@link applyPreset} is fire-and-forget the rejection is
 * dropped silently, which lands on the intended fail-safe — an inapplicable
 * preset leaves the workspace exactly as it was, rather than stranding the
 * operator on a blank one.
 *
 * @param {string[]} members - the preset's member panel ids, in config order
 * @param {Set<string>} visibleSet - currently-visible panel ids
 * @param {Set<string>} knownSet - all known panel ids (enabled + custom)
 * @returns {PresetDiff}
 */
export function computePresetDiff(members, visibleSet, knownSet) {
  const filtered = members.filter((id) => knownSet.has(id));
  if (filtered.length === 0) {
    return { toShow: [], toHide: [], focus: null };
  }
  const memberSet = new Set(filtered);
  const toShow = filtered.filter((id) => !visibleSet.has(id));
  const toHide = [...visibleSet].filter((id) => !memberSet.has(id));
  return { toShow, toHide, focus: filtered[0] };
}

/**
 * Apply a config-defined preset by NAME: one arrange request, applied on every
 * client by the panel_arrange handler.
 *
 * Nothing is orchestrated locally. The server resolves the preset's members
 * (filtered fail-safe to known ids, as {@link computePresetDiff} describes),
 * prunes rail membership to them, and broadcasts the arrangement; the echo then
 * opens exactly those tiles and focuses the first healthy one — the same focus
 * rule this module used to apply by hand, now applied once for everyone.
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
