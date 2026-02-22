/**
 * OSPREY Channel Finder — Explore View (dispatcher)
 *
 * Detects pipeline type and mounts the correct explore renderer.
 * Shows database source path and schema diagram for structured pipelines.
 */

import { state } from './state.js';
import { esc, renderSchema } from './utils.js';
import { mountHierarchical, unmountHierarchical, setShowDescriptions as setHierDescriptions } from './explore-hierarchical.js';
import { mountInContext, unmountInContext } from './explore-in-context.js';
import { mountMiddleLayer, unmountMiddleLayer, setShowDescriptions as setMLDescriptions } from './explore-middle-layer.js';

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

export function mountExplore(container) {
  const pt = state.pipelineType;

  const hasSchema = (pt === 'hierarchical' || pt === 'middle_layer');

  const descToggle = hasSchema
    ? `<label class="miller-toggle" style="margin-top: var(--space-1)">
         <input type="checkbox" id="show-desc-toggle"> Show full descriptions
       </label>`
    : '';

  container.innerHTML = `
    <div class="section-header">
      <div>
        <div class="section-title">Explore Channels</div>
        <div class="section-subtitle">Browse the channel database structure</div>
        ${_dbSourceBadge()}
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
    if (pt === 'hierarchical') setHierDescriptions(e.target.checked);
    else if (pt === 'middle_layer') setMLDescriptions(e.target.checked);
  });

  const content = document.getElementById('explore-content');

  if (pt === 'hierarchical') {
    currentRenderer = 'hierarchical';
    mountHierarchical(content);
  } else if (pt === 'middle_layer') {
    currentRenderer = 'middle_layer';
    mountMiddleLayer(content);
  } else {
    currentRenderer = 'in_context';
    mountInContext(content);
  }
}

export function unmountExplore() {
  if (currentRenderer === 'hierarchical') unmountHierarchical();
  else if (currentRenderer === 'middle_layer') unmountMiddleLayer();
  else if (currentRenderer === 'in_context') unmountInContext();
  currentRenderer = null;
}
