// @ts-check
/**
 * OSPREY Channel Finder — Explore View (dispatcher)
 *
 * Detects pipeline type and mounts the correct explore renderer.
 * Shows database source path and schema diagram for structured pipelines; the
 * graph paradigm carries neither, and its renderer names its own store.
 */

import { state } from './state.js';
import { esc, renderSchema } from './utils.js';
import { mountHierarchical, unmountHierarchical, setShowDescriptions as setHierDescriptions } from './explore-hierarchical.js';
import { mountInContext, unmountInContext } from './explore-in-context.js';
import { mountMiddleLayer, unmountMiddleLayer, setShowDescriptions as setMLDescriptions } from './explore-middle-layer.js';
import { mountGraph, unmountGraph } from './explore-graph.js';

/** @type {string|null} */
let currentRenderer = null;

function _dbSourceBadge() {
  const dbPath = state.dbPath;
  if (!dbPath) return '';
  const filename = dbPath.split('/').pop();
  return `<div class="db-source-badge" title="${esc(dbPath)}">
            <span class="db-source-icon">&#128193;</span>
            <code>${esc(filename)}</code>
          </div>`;
}

/**
 * Show a pipeline type this view has no renderer for. Loud on purpose: a
 * pipeline the server never accepted must not look like a working view.
 * @param {HTMLElement} content
 * @param {string|null} pipelineType
 */
function mountUnknown(content, pipelineType) {
  content.innerHTML = `
    <div class="explore-unknown" role="alert">
      <div class="explore-unknown-title">
        Unknown pipeline '${esc(String(pipelineType))}' &mdash; the server rejected this configuration
      </div>
      <div class="explore-unknown-body">
        Explore has no view for this pipeline type. Check the channel finder
        pipeline setting in your configuration, then reload this panel.
      </div>
    </div>
  `;
}

/** Clear the unknown-pipeline pane. */
function clearStaticPane() {
  const content = document.getElementById('explore-content');
  if (content) content.innerHTML = '';
}

/**
 * @param {HTMLElement} container
 */
export function mountExplore(container) {
  const pt = state.pipelineType;

  const hasSchema = (pt === 'hierarchical' || pt === 'middle_layer');

  const subtitle = pt === 'graph'
    ? 'Channels are resolved from the facility graph'
    : 'Browse the channel database structure';

  const descToggle = hasSchema
    ? `<label class="miller-toggle" style="margin-top: var(--cf-space-1)">
         <input type="checkbox" id="show-desc-toggle"> Show full descriptions
       </label>`
    : '';

  // Graph mode names its own provenance inside the panel (the graph store, not
  // a database file), so the db-source badge is left to the file-backed modes.
  container.innerHTML = `
    <div class="section-header">
      <div>
        <div class="section-title">Explore Channels</div>
        <div class="section-subtitle">${subtitle}</div>
        ${pt === 'graph' ? '' : _dbSourceBadge()}
        ${descToggle}
      </div>
    </div>
    ${hasSchema ? `<div class="db-schema-section">
      <div class="schema-diagram" id="explore-schema"></div>
    </div>` : ''}
    <div id="explore-content"></div>
  `;

  // Render schema diagram
  const schemaEl = document.getElementById('explore-schema');
  if (schemaEl) {
    renderSchema(schemaEl, pt, state.pipelineMetadata);
  }

  // Wire description toggle
  container.querySelector('#show-desc-toggle')?.addEventListener('change', (e) => {
    const checked = /** @type {HTMLInputElement} */ (e.target).checked;
    if (pt === 'hierarchical') setHierDescriptions(checked);
    else if (pt === 'middle_layer') setMLDescriptions(checked);
  });

  const content = document.getElementById('explore-content');
  if (!content) return;

  if (pt === 'hierarchical') {
    currentRenderer = 'hierarchical';
    mountHierarchical(content);
  } else if (pt === 'middle_layer') {
    currentRenderer = 'middle_layer';
    mountMiddleLayer(content);
  } else if (pt === 'in_context') {
    currentRenderer = 'in_context';
    mountInContext(content);
  } else if (pt === 'graph') {
    currentRenderer = 'graph';
    void mountGraph(content);
  } else {
    currentRenderer = 'unknown';
    mountUnknown(content, pt);
  }
}

export function unmountExplore() {
  if (currentRenderer === 'hierarchical') unmountHierarchical();
  else if (currentRenderer === 'middle_layer') unmountMiddleLayer();
  else if (currentRenderer === 'in_context') unmountInContext();
  else if (currentRenderer === 'graph') unmountGraph();
  else if (currentRenderer === 'unknown') clearStaticPane();
  currentRenderer = null;
}
