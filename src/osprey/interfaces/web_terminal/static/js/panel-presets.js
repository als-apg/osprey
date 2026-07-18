// @ts-check
/* OSPREY Web Terminal — Panel Presets ("Layouts")
 *
 * A preset is a config-defined, named set of panel ids a human applies in one
 * click from the "+" popover's "Layouts" section. Applying a preset is
 * EXCLUSIVE: show exactly the preset's members, hide every currently-visible
 * non-member ("those panels open and the rest close").
 *
 * This is a thin layer over the existing panel-visibility path — each show/hide
 * is the same setPanelVisibility() POST the "×"/"+" controls already use, so a
 * preset click and an agent show/hide call are indistinguishable downstream.
 * There is no parallel state system: the canonical visible set still lives on
 * the server and is broadcast over the panel_visibility SSE echo.
 */

import { initPanelAddMenu } from './panel-add-menu.js';

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
 * typo'd or disabled id is skipped fail-safe. If no member survives filtering,
 * ``focus`` is null and {@link applyPreset} no-ops — never strand the user on a
 * blank "No panels visible" tabset.
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
 * @typedef {object} ApplyPresetDeps
 * @property {() => Set<string>} getVisible - current visible-panel id set
 * @property {() => Set<string>} getKnown   - all known panel id set
 * @property {(id: string) => boolean} isHealthy - whether a panel can be focused (loaded/reachable)
 * @property {(id: string, visible: boolean) => void} setVisibility - the visibility POST helper
 * @property {(id: string) => void} focus   - focus a panel LOCALLY (no visibility POST)
 */

/**
 * Apply a preset EXCLUSIVELY: show its members, hide every visible non-member.
 *
 * Ordering avoids a transient all-hidden flash: show every member first, then
 * focus LOCALLY (not waiting for the SSE echo), then hide the non-members. The
 * focus target is the preset's primary member when it is healthy, else the first
 * healthy member — focusing an unhealthy panel is a no-op, so an offline primary
 * would otherwise strand focus on the outgoing panel. No-op entirely when no
 * member is known (empty guard).
 *
 * @param {string[]} members
 * @param {ApplyPresetDeps} deps
 */
export function applyPreset(members, { getVisible, getKnown, isHealthy, setVisibility, focus }) {
  const known = getKnown();
  const { toShow, toHide, focus: primary } = computePresetDiff(members, getVisible(), known);
  if (primary === null) return;
  for (const id of toShow) setVisibility(id, true);
  const target = isHealthy(primary) ? primary : members.find((id) => known.has(id) && isHealthy(id));
  if (target) focus(target);
  for (const id of toHide) setVisibility(id, false);
}

/**
 * @typedef {object} HeaderControlsDeps
 * @property {() => {id: string, label: string}[]} getHiddenPanels - known-but-hidden panels, tab order
 * @property {() => boolean} allowUrlPanels - whether runtime URL registration is on
 * @property {(id: string) => void} onShowPanel - reveal + focus a hidden panel
 * @property {(fields: {id: string, label: string, url: string}) => Promise<{ok: boolean, error?: string}>} onRegisterUrl
 * @property {() => {name: string, panels: string[]}[]} getPresets - config-defined layouts, in config order
 * @property {(panels: string[]) => void} onApplyPreset - apply a layout exclusively
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
