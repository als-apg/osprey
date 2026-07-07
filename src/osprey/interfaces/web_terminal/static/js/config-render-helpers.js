// @ts-check
/* OSPREY Web Terminal — shared config-renderer helpers
 *
 * A small neutral leaf module for the three helpers `renderSettingsJson`
 * (config-renderers.js), `renderSettingsJsonEditor` (settings-editor.js),
 * and `renderMcpJson` (mcp-renderer.js) all need: a section-header builder,
 * the permission-entry grouping logic, and a hook counter. Keeping these in
 * a neutral leaf dissolves what would otherwise be two circular imports
 * (config-renderers.js <-> settings-editor.js and config-renderers.js <->
 * mcp-renderer.js) into plain one-directional dependencies: this module has
 * no dependency on any of its three consumers, so none of them need to
 * import from each other to share this logic.
 *
 * @module config-render-helpers
 */

import { el as _el } from '/design-system/js/dom.js';

/**
 * Build a titled section container (`.config-section` with a
 * `.config-section-header` child). Shared by settings.json's
 * Environment/Permissions/Hooks sections and .mcp.json's "MCP Servers (N)"
 * section.
 *
 * @param {string} title
 * @returns {HTMLElement}
 */
export function _section(title) {
  const section = _el('div', 'config-section');
  const header = _el('div', 'config-section-header');
  header.textContent = title;
  section.appendChild(header);
  return section;
}

/**
 * Group permission entries by prefix (mcp server, file path, task, etc.) so
 * the read-only renderer's columns and the interactive editor's
 * drag-and-drop columns present entries the same way.
 *
 * @param {string[]} entries
 * @returns {Record<string, Array<{raw: string, display: string}>>}
 */
export function _groupPermissions(entries) {
  /** @type {Record<string, Array<{raw: string, display: string}>>} */
  const groups = {};
  /** @param {string} group @param {string} raw @param {string} display */
  const addTo = (group, raw, display) => {
    if (!groups[group]) groups[group] = [];
    groups[group].push({ raw, display });
  };

  for (const entry of entries) {
    if (entry.startsWith('mcp__')) {
      const parts = entry.split('__');
      const server = parts[1] || 'unknown';
      const tool = parts.slice(2).join('__') || '*';
      addTo(server, entry, tool);
    } else if (entry.startsWith('Task(')) {
      const agentName = entry.replace(/^Task\(/, '').replace(/\)$/, '');
      addTo('agents', entry, agentName);
    } else if (entry.startsWith('Read(') || entry.startsWith('NotebookEdit(')) {
      const match = entry.match(/^(\w+)\((.+)\)$/);
      if (match) {
        addTo('file access', entry, `${match[1]}: ${match[2]}`);
      } else {
        addTo('_ungrouped', entry, entry);
      }
    } else {
      addTo('_ungrouped', entry, entry);
    }
  }

  return groups;
}

/**
 * Count the total number of hooks across every matcher group of a single
 * hook event (e.g. `PreToolUse`), for the event header's count badge.
 *
 * @param {Array<{hooks?: Array<unknown>}>} hookGroups
 * @returns {number}
 */
export function _countHooks(hookGroups) {
  let count = 0;
  for (const g of hookGroups) {
    count += (g.hooks || []).length;
  }
  return count;
}
