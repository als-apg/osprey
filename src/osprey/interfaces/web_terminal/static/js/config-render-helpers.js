// @ts-check
/* OSPREY Web Terminal — shared config-renderer helpers
 *
 * A small neutral leaf module for the helpers `renderSettingsJson`
 * (config-renderers.js), `renderSettingsJsonEditor` (settings-editor.js),
 * and `renderMcpJson` (mcp-renderer.js) need: a section-header builder, the
 * permission-entry grouping logic, and the per-event hooks tree renderer.
 * Keeping these in a neutral leaf dissolves what would otherwise be two
 * circular imports (config-renderers.js <-> settings-editor.js and
 * config-renderers.js <-> mcp-renderer.js) into plain one-directional
 * dependencies: this module has no dependency on any of its three
 * consumers, so none of them need to import from each other to share this
 * logic.
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
 * hook event (e.g. `PreToolUse`), for the event header's count badge. Only
 * {@link _renderHookEvents} needs it, so it stays module-private.
 *
 * @param {Array<{hooks?: Array<unknown>}>} hookGroups
 * @returns {number}
 */
function _countHooks(hookGroups) {
  let count = 0;
  for (const g of hookGroups) {
    count += (g.hooks || []).length;
  }
  return count;
}

/**
 * @typedef {object} HookGroup
 * @property {string} [matcher]
 * @property {Array<{command?: string, timeout?: number}>} [hooks]
 */

/**
 * Render the collapsible per-event hooks tree: one `.config-hook-event` per
 * event with a chevron + name + count-badge header (click toggles
 * `expanded`), matcher groups, and per-hook script-name/timeout entries.
 * Shared by the read-only settings.json renderer (config-renderers.js) and
 * the editor's read-only Hooks section (settings-editor.js); each caller
 * wraps the fragment in its own titled section. Text lands via textContent
 * only — no HTML sink.
 *
 * @param {Record<string, HookGroup[]>} hooks
 * @returns {DocumentFragment}
 */
export function _renderHookEvents(hooks) {
  const frag = document.createDocumentFragment();

  for (const [eventName, hookGroups] of Object.entries(hooks)) {
    const eventSection = _el('div', 'config-hook-event');
    const eventHeader = _el('div', 'config-hook-event-header');

    const chevron = _el('span', 'config-hook-chevron');
    chevron.textContent = '▶';
    eventHeader.appendChild(chevron);

    const nameSpan = _el('span', '');
    nameSpan.textContent = eventName;
    eventHeader.appendChild(nameSpan);

    const countSpan = _el('span', 'config-hook-count');
    countSpan.textContent = String(_countHooks(hookGroups));
    eventHeader.appendChild(countSpan);

    eventHeader.addEventListener('click', () => eventSection.classList.toggle('expanded'));
    eventSection.appendChild(eventHeader);

    const eventBody = _el('div', 'config-hook-event-body');
    for (const group of hookGroups) {
      const matcherEl = _el('div', 'config-hook-matcher');
      const matcherLabel = _el('span', 'config-hook-matcher-label');
      matcherLabel.textContent = group.matcher || '*';
      matcherEl.appendChild(matcherLabel);

      for (const hook of (group.hooks || [])) {
        const hookEl = _el('div', 'config-hook-entry');
        const cmd = hook.command || '';
        const scriptName = (cmd.split('/').pop() || '').replace(/"/g, '').replace(/\.py$/, '');

        const scriptSpan = _el('span', 'config-hook-script');
        scriptSpan.textContent = scriptName;
        hookEl.appendChild(scriptSpan);

        if (hook.timeout) {
          const timeoutSpan = _el('span', 'config-hook-timeout');
          timeoutSpan.textContent = hook.timeout + 's';
          hookEl.appendChild(timeoutSpan);
        }
        matcherEl.appendChild(hookEl);
      }
      eventBody.appendChild(matcherEl);
    }
    eventSection.appendChild(eventBody);
    frag.appendChild(eventSection);
  }

  return frag;
}
