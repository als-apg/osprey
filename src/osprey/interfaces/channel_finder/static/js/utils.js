// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * OSPREY Channel Finder — Shared Utilities
 *
 * Common helpers used across multiple view modules.
 */

/**
 * HTML-escape a string to prevent XSS.
 *
 * Re-exported from the shared design-system `dom.js` module under the historical
 * `esc` name so existing importers (explore-hierarchical.js, etc.) keep working.
 * Semantics are identical: textContent -> innerHTML, nullish -> "", no
 * quote-escaping.
 */
export { escapeHtml as esc } from '/design-system/js/dom.js';

/**
 * Render a pipeline schema diagram into a container element.
 * @param {HTMLElement} container - Target element.
 * @param {string} pipelineType - 'hierarchical' | 'middle_layer' | 'in_context'.
 * @param {object} metadata - Pipeline metadata (e.g., hierarchy_levels).
 */
export function renderSchema(container, pipelineType, metadata) {
  if (pipelineType === 'hierarchical' && metadata?.hierarchy_levels) {
    const levels = metadata.hierarchy_levels;
    const rows = levels.map((lvl, i) => {
      const arrow = i < levels.length - 1 ? '<span class="schema-arrow">&rarr;</span>' : '';
      const name = typeof lvl === 'string' ? lvl : (lvl.name || lvl.level || '?');
      return `<span class="schema-node${i === 0 ? ' active' : ''}">${name}</span>${arrow}`;
    }).join('');
    container.innerHTML = `
      <div class="schema-row">${rows}</div>
      ${metadata.naming_pattern ? `<div style="margin-top: var(--space-3); color: var(--text-muted); font-size: var(--text-sm);">
        <strong>Naming pattern:</strong> <code class="pv-name">${metadata.naming_pattern}</code>
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
