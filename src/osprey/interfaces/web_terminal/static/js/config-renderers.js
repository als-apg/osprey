// @ts-check
/* OSPREY Web Terminal — Rich renderer for settings.json (+ shared helpers)
 *
 * Instead of dumping raw JSON, renderSettingsJson parses the content and
 * presents a structured, scannable view: Environment, Model, Permissions
 * (allow/deny/ask), Hooks.
 *
 * Also provides an interactive editor for settings.json that allows editing
 * model, permissions (tri-state allow/ask/deny per entry), and adding/removing
 * permission entries without touching raw JSON.
 *
 * Two sibling modules re-export back through this file so every import
 * site (e.g. scaffold-gallery.js) keeps the same `./config-renderers.js`
 * path regardless of which file actually implements a given export:
 *
 *   - settings-editor.js: the interactive settings.json editor.
 *   - mcp-renderer.js: the .mcp.json renderer.
 *
 * Both re-exports are one-directional: `_section`/`_groupPermissions`/
 * `_renderHookEvents` (needed by this file, settings-editor.js, and
 * mcp-renderer.js alike) live in the neutral leaf module
 * config-render-helpers.js, so none of the three renderer modules need to
 * import from each other — no circular imports.
 */

import { el as _el } from '/design-system/js/dom.js';
import { _section, _groupPermissions, _renderHookEvents } from './config-render-helpers.js';

export { renderSettingsJsonEditor } from './settings-editor.js';
export { renderMcpJson } from './mcp-renderer.js';

// ---------------------------------------------------------------------------
// settings.json renderer
// ---------------------------------------------------------------------------

/** @param {string} jsonString */
export function renderSettingsJson(jsonString) {
  let data;
  try {
    data = JSON.parse(jsonString);
  } catch {
    return null;
  }

  const container = document.createElement('div');
  container.className = 'config-structured-view';

  // ---- Environment & Model ----
  if (data.env || data.model) {
    const section = _section('Environment');
    const grid = _el('div', 'config-env-grid');

    if (data.model) {
      grid.appendChild(_envRow('Model', data.model));
    }
    if (data.env) {
      for (const [key, value] of Object.entries(data.env)) {
        grid.appendChild(_envRow(_humanizeEnvKey(key), value));
      }
    }
    section.appendChild(grid);
    container.appendChild(section);
  }

  // ---- Permissions ----
  if (data.permissions) {
    const section = _section('Permissions');
    const columns = _el('div', 'config-permissions-columns');

    if (data.permissions.allow) {
      columns.appendChild(_permissionColumn('allow', data.permissions.allow));
    }
    if (data.permissions.ask) {
      columns.appendChild(_permissionColumn('ask', data.permissions.ask));
    }
    if (data.permissions.deny) {
      columns.appendChild(_permissionColumn('deny', data.permissions.deny));
    }

    section.appendChild(columns);
    container.appendChild(section);
  }

  // ---- Hooks ----
  if (data.hooks) {
    const section = _section('Hooks');
    section.appendChild(_renderHookEvents(data.hooks));
    container.appendChild(section);
  }

  return container;
}


// ---------------------------------------------------------------------------
// Permission column builder
// ---------------------------------------------------------------------------

/** @param {string} level @param {string[]} entries */
function _permissionColumn(level, entries) {
  const col = _el('div', `config-perm-col config-perm-${level}`);

  const header = _el('div', 'config-perm-header');
  header.textContent = level.toUpperCase();
  col.appendChild(header);

  // Group entries by prefix (mcp server, file path, task, etc.)
  const groups = _groupPermissions(entries);

  for (const [groupName, items] of Object.entries(groups)) {
    if (groupName !== '_ungrouped') {
      const groupLabel = _el('div', 'config-perm-group-label');
      groupLabel.textContent = groupName;
      col.appendChild(groupLabel);
    }

    for (const item of items) {
      const row = _el('div', 'config-perm-entry');
      row.textContent = item.display;
      row.title = item.raw;
      col.appendChild(row);
    }
  }

  return col;
}


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** @param {string} label @param {string} value */
function _envRow(label, value) {
  const row = _el('div', 'config-env-row');
  const labelEl = _el('span', 'config-env-label');
  labelEl.textContent = label;
  const valueEl = _el('span', 'config-env-value');
  valueEl.textContent = value;
  valueEl.title = value;
  row.appendChild(labelEl);
  row.appendChild(valueEl);
  return row;
}

/** @param {string} key */
function _humanizeEnvKey(key) {
  return key
    .replace(/^ANTHROPIC_/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}
