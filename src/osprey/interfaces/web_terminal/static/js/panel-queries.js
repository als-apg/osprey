// @ts-check
/* OSPREY Web Terminal — Panel Enumeration Queries
 *
 * The derivations behind panel-manager's command-palette accessors. Each takes
 * the panel catalog and the state it reads as arguments rather than closing
 * over panel-manager's module-private variables, so the enumeration rules are
 * testable on their own and panel-manager keeps only the one-line bindings that
 * supply its private state.
 *
 * Pure: no DOM, no mutation, no module state of its own.
 */

/**
 * Known-but-hidden panels, in catalog order.
 * @param {Array<{id: string, label: string}>} panels
 * @param {Set<string>} visible
 * @returns {Array<{id: string, label: string}>}
 */
export function hiddenPanels(panels, visible) {
  return panels.filter(p => !visible.has(p.id)).map(p => ({ id: p.id, label: p.label }));
}

/**
 * Visible panels excluding the active one, in catalog order. "Focus" on the
 * already-active panel is a no-op, so the active id is filtered out.
 * @param {Array<{id: string, label: string}>} panels
 * @param {Set<string>} visible
 * @param {string | null} activeId
 * @returns {Array<{id: string, label: string}>}
 */
export function visiblePanelsExcept(panels, visible, activeId) {
  return panels
    .filter(p => visible.has(p.id) && p.id !== activeId)
    .map(p => ({ id: p.id, label: p.label }));
}

/**
 * Standalone (non-embedded) URL for a service panel. `state.url` is the
 * already-proxied root-relative base; the optional catalog path suffixes custom
 * panels' UI root. Null until the panel's config fetch has resolved a URL.
 * @param {Array<{id: string, path?: string}>} panels
 * @param {{url?: string | null} | undefined} state
 * @param {string} panelId
 * @returns {string | null}
 */
export function standaloneUrl(panels, state, panelId) {
  if (!state?.url) return null;
  const panel = panels.find(p => p.id === panelId);
  const path = panel?.path && panel.path !== '/' ? panel.path : '';
  return state.url + path;
}
