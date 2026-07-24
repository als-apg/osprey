// @ts-check
/* OSPREY Web Terminal — Command Palette Registry Builder
 *
 * Pure, dependency-injected builder that turns live app data into a flat,
 * grouped list of command-palette items (Settings / Panels / Layouts /
 * Actions, in that order). Everything it needs — the /api/config snapshot,
 * panel/preset getters, action closures, and the navigation callbacks
 * (revealSetting / showPanel / focusPanel / applyPreset) — is passed in via
 * `deps`, so this module imports nothing, touches no DOM, and is unit-testable
 * in isolation. The caller (palette.js) rebuilds the registry each time the
 * palette opens, then fuzzy-filters the returned items by their `searchText`.
 */

/**
 * A navigable palette item, or a non-navigable Settings status decoration.
 *
 * Navigable items carry `run` (invoked on selection) and `searchText` (scored
 * by the fuzzy matcher). Settings status rows are static decorations with
 * neither — the UI renders them and excludes them from keyboard navigation.
 *
 * @typedef {(
 *   {
 *     group: 'Settings' | 'Panels' | 'Layouts' | 'Actions',
 *     label: string,
 *     detail?: string,
 *     searchText: string,
 *     run: () => void,
 *   }
 *   |
 *   {
 *     group: 'Settings',
 *     status: 'loading' | 'error',
 *     label: string,
 *   }
 * )} Item
 */

/**
 * Injected dependencies. Every field is provided by the caller; optional
 * getters may be absent and are treated as empty. No field is read from any
 * global or module import — that is what keeps this builder pure.
 *
 * @typedef {{
 *   config?: (
 *     { state: 'ok', sections: Record<string, unknown> }
 *     | { state: 'loading' }
 *     | { state: 'error' }
 *   ),
 *   getHiddenPanels?: () => Array<{ id: string, label: string }>,
 *   getVisiblePanels?: () => Array<{ id: string, label: string }>,
 *   getPresets?: () => Array<{ name: string, panels: string[] }>,
 *   actions?: Array<{ label: string, detail?: string, run: () => void }>,
 *   showPanel?: (id: string) => void,
 *   focusPanel?: (id: string) => void,
 *   applyPreset?: (panels: string[]) => void,
 *   revealSetting?: (dotKey: string) => void,
 * }} PaletteDeps
 */

/**
 * Invoke an optional getter defensively, always returning an array. Absent or
 * non-function getters, and getters that return a non-array, yield [].
 *
 * @template T
 * @param {(() => Array<T>) | undefined} getter
 * @returns {Array<T>}
 */
function safeList(getter) {
  if (typeof getter !== 'function') {
    return [];
  }
  const out = getter();
  return Array.isArray(out) ? out : [];
}

/**
 * Recursively flatten a config `sections` tree into leaf dot-paths. Nested
 * plain-object keys are joined with '.'; arrays and scalars are treated as
 * leaves (never recursed into). Intermediate object nodes are not emitted —
 * only the paths that terminate at a non-object leaf. Already-dotted flat keys
 * (leaf values) fall through as-is.
 *
 * @param {Record<string, unknown>} sections
 * @returns {string[]} leaf dot-keys, in source (insertion) order
 */
function flattenSections(sections) {
  /** @type {string[]} */
  const keys = [];

  /**
   * @param {string} prefix
   * @param {unknown} value
   */
  const walk = (prefix, value) => {
    if (
      value !== null &&
      typeof value === 'object' &&
      !Array.isArray(value)
    ) {
      for (const [k, v] of Object.entries(value)) {
        walk(prefix ? `${prefix}.${k}` : k, v);
      }
      return;
    }
    // Leaf: scalar, null, or array.
    keys.push(prefix);
  };

  for (const [k, v] of Object.entries(sections)) {
    walk(k, v);
  }
  return keys;
}

/**
 * Build the Settings group from the config snapshot. Loading/error states each
 * emit a single non-navigable decoration; the ok state emits one navigable
 * Item per leaf dot-key, whose `run` calls the injected `revealSetting`.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildSettings(deps) {
  const config = deps.config;
  if (!config || config.state === 'loading') {
    return [{ group: 'Settings', status: 'loading', label: 'Loading settings…' }];
  }
  if (config.state === 'error') {
    return [{ group: 'Settings', status: 'error', label: 'Settings unavailable' }];
  }

  const sections = config.sections && typeof config.sections === 'object' ? config.sections : {};
  const revealSetting = deps.revealSetting;
  return flattenSections(sections).map((dotKey) => ({
    group: 'Settings',
    label: dotKey,
    searchText: dotKey,
    run: () => {
      if (typeof revealSetting === 'function') {
        revealSetting(dotKey);
      }
    },
  }));
}

/**
 * Build the Panels group: a "Show <label>" item per hidden panel, then a
 * "Focus <label>" item per visible (non-active) panel.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildPanels(deps) {
  const showPanel = deps.showPanel;
  const focusPanel = deps.focusPanel;

  /** @type {Item[]} */
  const items = [];

  for (const panel of safeList(deps.getHiddenPanels)) {
    const id = panel.id;
    items.push({
      group: 'Panels',
      label: `Show ${panel.label}`,
      searchText: `show ${panel.label} ${id}`,
      run: () => {
        if (typeof showPanel === 'function') {
          showPanel(id);
        }
      },
    });
  }

  for (const panel of safeList(deps.getVisiblePanels)) {
    const id = panel.id;
    items.push({
      group: 'Panels',
      label: `Focus ${panel.label}`,
      searchText: `focus ${panel.label} ${id}`,
      run: () => {
        if (typeof focusPanel === 'function') {
          focusPanel(id);
        }
      },
    });
  }

  return items;
}

/**
 * Build the Layouts group: one item per preset, whose `run` applies the
 * preset's panel set via the injected `applyPreset`.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildLayouts(deps) {
  const applyPreset = deps.applyPreset;
  return safeList(deps.getPresets).map((preset) => {
    const panels = Array.isArray(preset.panels) ? preset.panels : [];
    return {
      group: 'Layouts',
      label: `Layout: ${preset.name}`,
      searchText: preset.name,
      run: () => {
        if (typeof applyPreset === 'function') {
          applyPreset(panels);
        }
      },
    };
  });
}

/**
 * Build the Actions group by wrapping each injected action closure, preserving
 * source order. Each entry's own `run` is passed straight through.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildActions(deps) {
  const actions = Array.isArray(deps.actions) ? deps.actions : [];
  return actions.map((action) => {
    /** @type {Item} */
    const item = {
      group: 'Actions',
      label: action.label,
      searchText: action.label,
      run: action.run,
    };
    if (action.detail !== undefined) {
      item.detail = action.detail;
    }
    return item;
  });
}

/**
 * Build the full command-palette registry from injected app data. Returns a
 * flat array grouped Settings → Panels → Layouts → Actions; within each group,
 * source order is preserved. Deterministic for a given input, and never throws
 * on absent optional deps (missing getters default to empty).
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
export function buildRegistry(deps) {
  const d = deps || {};
  return [
    ...buildSettings(d),
    ...buildPanels(d),
    ...buildLayouts(d),
    ...buildActions(d),
  ];
}
