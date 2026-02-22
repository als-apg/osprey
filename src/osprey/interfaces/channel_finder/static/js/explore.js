/**
 * OSPREY Channel Finder — Explore View (dispatcher)
 *
 * Detects pipeline type and mounts the correct explore renderer.
 * Includes collapsible Database Info section with schema diagram.
 */

import { state } from './state.js';
import { renderSchema } from './utils.js';
import { mountHierarchical, unmountHierarchical, setShowDescriptions as setHierDescriptions } from './explore-hierarchical.js';
import { mountInContext, unmountInContext } from './explore-in-context.js';
import { mountMiddleLayer, unmountMiddleLayer, setShowDescriptions as setMLDescriptions } from './explore-middle-layer.js';

let currentRenderer = null;

export function mountExplore(container) {
  const pt = state.pipelineType;

  const descToggle = (pt === 'hierarchical' || pt === 'middle_layer')
    ? `<label class="miller-toggle" style="margin-top: var(--space-1)">
         <input type="checkbox" id="show-desc-toggle"> Show full descriptions
       </label>`
    : '';

  container.innerHTML = `
    <div class="section-header">
      <div>
        <div class="section-title">Explore Channels</div>
        <div class="section-subtitle">Browse the channel database structure</div>
        ${descToggle}
      </div>
      <div>
        <span class="db-info-toggle" id="db-info-toggle">
          <span class="chevron">&#9654;</span> Database Info
        </span>
      </div>
    </div>
    <div class="db-info-content" id="db-info-content">
      <div style="padding: var(--space-3) 0;">
        <div class="schema-diagram" id="explore-schema"></div>
      </div>
    </div>
    <div id="explore-content"></div>
  `;

  // Wire up Database Info toggle
  const toggle = document.getElementById('db-info-toggle');
  const infoContent = document.getElementById('db-info-content');
  toggle?.addEventListener('click', () => {
    toggle.classList.toggle('open');
    infoContent.classList.toggle('open');
  });

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
