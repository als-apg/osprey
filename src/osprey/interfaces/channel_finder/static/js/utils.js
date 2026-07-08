// @ts-check
/**
 * OSPREY Channel Finder — Shared Utilities
 *
 * Common helpers used across multiple view modules.
 */

import { escapeHtml as esc } from '/design-system/js/dom.js';

/**
 * HTML-escape a string to prevent XSS.
 *
 * Re-exported from the shared design-system `dom.js` module under the historical
 * `esc` name so existing importers (explore-hierarchical.js, etc.) keep working.
 * Semantics are identical: textContent -> innerHTML, nullish -> "", plus
 * double/single-quote escaping (safe inside quoted attribute values).
 */
export { esc };

/**
 * Normalize an unknown thrown value into a human-readable message string.
 *
 * Shared by every error-path sink so each `catch` binding — typed `unknown`
 * under strict checkJs — reads its message uniformly. `Error`/`ApiError`
 * instances yield their `.message`; anything else is stringified.
 * @param {unknown} e
 * @returns {string}
 */
export function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Render a pipeline schema diagram into a container element.
 * @param {HTMLElement} container - Target element.
 * @param {string|null|undefined} pipelineType - 'hierarchical' | 'middle_layer' | 'in_context'.
 * @param {any} metadata - Pipeline metadata (e.g., hierarchy_levels).
 */
export function renderSchema(container, pipelineType, metadata) {
  if (pipelineType === 'hierarchical' && metadata?.hierarchy_levels) {
    const levels = metadata.hierarchy_levels;
    const rows = levels.map((/** @type {any} */ lvl, /** @type {number} */ i) => {
      const arrow = i < levels.length - 1 ? '<span class="schema-arrow">&rarr;</span>' : '';
      const name = typeof lvl === 'string' ? lvl : (lvl.name || lvl.level || '?');
      return `<span class="schema-node${i === 0 ? ' active' : ''}">${esc(name)}</span>${arrow}`;
    }).join('');
    container.innerHTML = `
      <div class="schema-row">${rows}</div>
      ${metadata.naming_pattern ? `<div style="margin-top: var(--space-3); color: var(--text-muted); font-size: var(--text-sm);">
        <strong>Naming pattern:</strong> <code class="pv-name">${esc(metadata.naming_pattern)}</code>
      </div>` : ''}
    `;
  } else if (pipelineType === 'middle_layer') {
    container.innerHTML = `
      <div class="schema-row">
        <span class="schema-node active">System</span>
        <span class="schema-arrow">&rarr;</span>
        <span class="schema-node">Family</span>
        <span class="schema-arrow">&rarr;</span>
        <span class="schema-node">Field</span>
        <span class="schema-arrow">&rarr;</span>
        <span class="schema-node">Channel PV</span>
      </div>
    `;
  } else {
    container.innerHTML = `
      <div class="schema-row">
        <span class="schema-node active">Database</span>
        <span class="schema-arrow">&rarr;</span>
        <span class="schema-node">Channels (chunked)</span>
        <span class="schema-arrow">&rarr;</span>
        <span class="schema-node">Name + Address</span>
      </div>
    `;
  }
}
