/**
 * ARIEL Advanced Options
 *
 * Dynamically renders search mode tabs and advanced parameter panels
 * based on capabilities discovered from the backend API.
 */

// --- State ---
let capabilities = null;
let currentMode = 'rag';
let isPanelOpen = false;
let paramValues = {};
const dynamicOptionsCache = {};

// --- Fallback ---
const FALLBACK_CAPABILITIES = {
  categories: {
    llm: {
      label: 'LLM',
      modes: [
        { name: 'rag', label: 'RAG', description: 'AI-powered search', parameters: [] },
        { name: 'agent', label: 'Agent', description: 'Autonomous agent', parameters: [] },
      ],
    },
    direct: {
      label: 'Direct',
      modes: [
        { name: 'keyword', label: 'Keyword', description: 'Text search', parameters: [] },
      ],
    },
  },
  shared_parameters: [],
};

// --- Public API ---

/**
 * Initialize the advanced options system.
 * @param {Object|null} caps - Capabilities from /api/capabilities (or null for fallback)
 */
export function initAdvancedOptions(caps) {
  capabilities = caps || FALLBACK_CAPABILITIES;

  // Set defaults from capabilities
  resetToDefaults();

  // Render mode tabs
  renderModeTabs();
  selectMode(currentMode);

  // Wire up toggle button (try both IDs for backward compat)
  const toggleBtn = document.getElementById('advanced-toggle-btn')
    || document.getElementById('advanced-toggle');
  toggleBtn?.addEventListener('click', () => {
    isPanelOpen = !isPanelOpen;
    const panel = document.getElementById('advanced-panel');
    if (panel) {
      panel.classList.toggle('hidden', !isPanelOpen);
      if (isPanelOpen) {
        renderAdvancedPanel();
      }
    }
    toggleBtn.classList.toggle('active', isPanelOpen);
  });

  // Close button
  const closeBtn = document.getElementById('advanced-close-btn');
  closeBtn?.addEventListener('click', () => {
    isPanelOpen = false;
    document.getElementById('advanced-panel')?.classList.add('hidden');
    document.getElementById('advanced-toggle-btn')?.classList.remove('active');
  });

  // Reset button
  const resetBtn = document.getElementById('advanced-reset-btn');
  resetBtn?.addEventListener('click', () => {
    resetToDefaults();
    if (isPanelOpen) {
      renderAdvancedPanel();
    }
  });
}

/**
 * Get the currently selected search mode.
 * @returns {string} Mode name (e.g. "rag", "keyword")
 */
export function getCurrentMode() {
  return currentMode;
}

/**
 * Get current advanced parameter values for the selected mode.
 * Returns only params relevant to the current mode + shared params.
 * @returns {Object} Parameter values keyed by name
 */
export function getAdvancedParams() {
  const modeParams = getModeParameters(currentMode);
  const sharedParams = capabilities?.shared_parameters || [];
  const allParamNames = new Set([
    ...modeParams.map(p => p.name),
    ...sharedParams.map(p => p.name),
  ]);

  const result = {};
  for (const name of allParamNames) {
    if (name in paramValues && paramValues[name] !== undefined && paramValues[name] !== null) {
      result[name] = paramValues[name];
    }
  }
  return result;
}

/**
 * Legacy export for backwards compatibility with search.js.
 * @param {string} mode - Search mode (ignored, uses currentMode)
 * @returns {Object} Advanced options
 */
export function getAdvancedOptions(mode) {
  return getAdvancedParams();
}

/**
 * Check if advanced panel is currently open.
 * @returns {boolean}
 */
export function isAdvancedPanelOpen() {
  return isPanelOpen;
}

/**
 * Close the advanced panel if it is open.
 */
export function closeAdvancedPanel() {
  if (!isPanelOpen) return;
  isPanelOpen = false;
  document.getElementById('advanced-panel')?.classList.add('hidden');
  const btn = document.getElementById('advanced-toggle-btn')
    || document.getElementById('advanced-toggle');
  btn?.classList.remove('active');
}

// --- Internal ---

/**
 * Render mode tabs into #search-mode-tabs.
 */
function renderModeTabs() {
  const container = document.getElementById('search-mode-tabs');
  if (!container) return;

  let html = '';
  const categories = capabilities?.categories || {};

  // Render LLM group first, then Direct
  for (const catKey of ['llm', 'direct']) {
    const cat = categories[catKey];
    if (!cat || !cat.modes?.length) continue;

    html += `<div class="mode-tab-group">`;
    html += `<span class="mode-tab-group-label">${escapeHtml(cat.label)}</span>`;

    for (const mode of cat.modes) {
      const active = mode.name === currentMode ? ' active' : '';
      html += `<button class="mode-tab${active}" data-mode="${escapeHtml(mode.name)}" title="${escapeHtml(mode.description)}">${escapeHtml(mode.label)}</button>`;
    }

    html += `</div>`;
  }

  container.innerHTML = html;

  // Attach click handlers
  container.querySelectorAll('.mode-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      selectMode(btn.dataset.mode);
    });
  });
}

/**
 * Select a mode and update UI.
 * @param {string} mode - Mode name
 */
function selectMode(mode) {
  currentMode = mode;

  // Update active class on tabs
  document.querySelectorAll('.mode-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  // Re-render advanced panel if open
  if (isPanelOpen) {
    renderAdvancedPanel();
  }
}

/**
 * Get parameter descriptors for a mode.
 * @param {string} modeName - Mode name
 * @returns {Array} Parameter descriptors
 */
function getModeParameters(modeName) {
  const categories = capabilities?.categories || {};
  for (const cat of Object.values(categories)) {
    for (const mode of (cat.modes || [])) {
      if (mode.name === modeName) {
        return mode.parameters || [];
      }
    }
  }
  return [];
}

/**
 * Render the advanced options panel for the current mode.
 */
function renderAdvancedPanel() {
  const container = document.getElementById('advanced-sections');
  if (!container) return;

  const modeParams = getModeParameters(currentMode);
  const sharedParams = capabilities?.shared_parameters || [];

  // Combine mode-specific and shared params
  const allParams = [...modeParams, ...sharedParams];

  if (allParams.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding: var(--space-6);">
        <p class="empty-state-text">No advanced options for this mode.</p>
      </div>
    `;
    return;
  }

  // Group by section, separate "Filters" from the rest
  const filterParams = [];
  const otherSections = {};
  for (const param of allParams) {
    const section = param.section || 'General';
    if (section === 'Filters') {
      filterParams.push(param);
    } else {
      if (!otherSections[section]) otherSections[section] = [];
      otherSections[section].push(param);
    }
  }

  let html = '';

  // Render Filters section first (full-width)
  if (filterParams.length > 0) {
    html += `
      <div class="advanced-section filters-section">
        <div class="advanced-section-header">
          <span class="section-title">Filters</span>
        </div>
        <div class="advanced-section-body">
    `;
    for (const param of filterParams) {
      html += renderParameter(param);
    }
    html += `</div></div>`;
  }

  // Render remaining sections
  for (const [sectionName, params] of Object.entries(otherSections)) {
    html += `
      <div class="advanced-section">
        <div class="advanced-section-header">
          <span class="section-title">${escapeHtml(sectionName)}</span>
        </div>
        <div class="advanced-section-body">
    `;

    for (const param of params) {
      html += renderParameter(param);
    }

    html += `</div></div>`;
  }

  container.innerHTML = html;

  // Attach event listeners
  attachParamListeners(container);

  // Load dynamic select options asynchronously
  loadDynamicSelectOptions(container);
}

/**
 * Render a single parameter control.
 * @param {Object} param - Parameter descriptor
 * @returns {string} HTML string
 */
function renderParameter(param) {
  const value = paramValues[param.name] ?? param.default;

  switch (param.type) {
    case 'float':
    case 'int':
      return renderSlider(param, value);
    case 'bool':
      return renderToggle(param, value);
    case 'select':
      return renderSelect(param, value);
    case 'date':
      return renderDateInput(param, value);
    case 'text':
      return renderTextInput(param, value);
    case 'dynamic_select':
      return renderDynamicSelect(param, value);
    default:
      return '';
  }
}

/**
 * Render a slider control for float/int params.
 */
function renderSlider(param, value) {
  const displayValue = param.type === 'float' ? Number(value).toFixed(2) : value;
  return `
    <div class="slider-control">
      <div class="slider-header">
        <label class="slider-label" for="param-${param.name}" title="${escapeHtml(param.description)}">${escapeHtml(param.label)}</label>
        <span class="slider-value" id="param-${param.name}-value">${displayValue}</span>
      </div>
      <input type="range" id="param-${param.name}" class="slider"
        data-param="${param.name}" data-type="${param.type}"
        min="${param.min ?? 0}" max="${param.max ?? 100}"
        value="${value}" step="${param.step ?? 1}">
    </div>
  `;
}

/**
 * Render a toggle switch for bool params.
 */
function renderToggle(param, value) {
  const checked = value ? 'checked' : '';
  return `
    <div class="toggle-control">
      <label class="toggle-label" for="param-${param.name}" title="${escapeHtml(param.description)}">${escapeHtml(param.label)}</label>
      <label class="toggle-switch">
        <input type="checkbox" id="param-${param.name}"
          data-param="${param.name}" data-type="bool" ${checked}>
        <span class="toggle-slider"></span>
      </label>
    </div>
  `;
}

/**
 * Render a select dropdown for select params.
 */
function renderSelect(param, value) {
  let optionsHtml = '';
  for (const opt of (param.options || [])) {
    const selected = opt.value === value ? 'selected' : '';
    optionsHtml += `<option value="${escapeHtml(opt.value)}" ${selected}>${escapeHtml(opt.label)}</option>`;
  }

  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}" title="${escapeHtml(param.description)}">${escapeHtml(param.label)}</label>
      <select id="param-${param.name}" class="select"
        data-param="${param.name}" data-type="select">
        ${optionsHtml}
      </select>
    </div>
  `;
}

/**
 * Render a date input control.
 */
function renderDateInput(param, value) {
  const val = value || '';
  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}" title="${escapeHtml(param.description)}">${escapeHtml(param.label)}</label>
      <input type="date" id="param-${param.name}" class="input"
        data-param="${param.name}" data-type="date" value="${escapeHtml(val)}">
    </div>
  `;
}

/**
 * Render a text input control.
 */
function renderTextInput(param, value) {
  const val = value || '';
  const placeholder = param.placeholder || '';
  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}" title="${escapeHtml(param.description)}">${escapeHtml(param.label)}</label>
      <input type="text" id="param-${param.name}" class="input"
        data-param="${param.name}" data-type="text"
        value="${escapeHtml(val)}" placeholder="${escapeHtml(placeholder)}">
    </div>
  `;
}

/**
 * Render a dynamic select that fetches options from an endpoint.
 */
function renderDynamicSelect(param, value) {
  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}" title="${escapeHtml(param.description)}">${escapeHtml(param.label)}</label>
      <select id="param-${param.name}" class="select"
        data-param="${param.name}" data-type="dynamic_select"
        data-endpoint="${escapeHtml(param.options_endpoint || '')}">
        <option value="">All</option>
      </select>
    </div>
  `;
}

/**
 * Fetch and populate options for dynamic_select elements.
 */
async function loadDynamicSelectOptions(container) {
  const selects = container.querySelectorAll('select[data-type="dynamic_select"]');
  for (const select of selects) {
    const endpoint = select.dataset.endpoint;
    if (!endpoint) continue;

    const currentValue = paramValues[select.dataset.param] || '';

    try {
      let options;
      if (dynamicOptionsCache[endpoint]) {
        options = dynamicOptionsCache[endpoint];
      } else {
        const response = await fetch(endpoint);
        if (!response.ok) continue;
        const data = await response.json();
        options = data.options || [];
        dynamicOptionsCache[endpoint] = options;
      }

      // Preserve the "All" option and append fetched options
      let html = '<option value="">All</option>';
      for (const opt of options) {
        const selected = opt.value === currentValue ? 'selected' : '';
        html += `<option value="${escapeHtml(opt.value)}" ${selected}>${escapeHtml(opt.label)}</option>`;
      }
      select.innerHTML = html;
    } catch (err) {
      console.warn(`Failed to load options from ${endpoint}:`, err);
    }
  }
}

/**
 * Attach change listeners to parameter controls.
 */
function attachParamListeners(container) {
  // Sliders
  container.querySelectorAll('input[type="range"][data-param]').forEach(slider => {
    slider.addEventListener('input', () => {
      const name = slider.dataset.param;
      const type = slider.dataset.type;
      const val = type === 'float' ? parseFloat(slider.value) : parseInt(slider.value, 10);
      paramValues[name] = val;

      // Update displayed value
      const display = document.getElementById(`param-${name}-value`);
      if (display) {
        display.textContent = type === 'float' ? val.toFixed(2) : val;
      }
    });
  });

  // Toggles
  container.querySelectorAll('input[type="checkbox"][data-param]').forEach(toggle => {
    toggle.addEventListener('change', () => {
      paramValues[toggle.dataset.param] = toggle.checked;
    });
  });

  // Selects (including dynamic_select)
  container.querySelectorAll('select[data-param]').forEach(select => {
    select.addEventListener('change', () => {
      paramValues[select.dataset.param] = select.value || null;
    });
  });

  // Date inputs
  container.querySelectorAll('input[type="date"][data-param]').forEach(input => {
    input.addEventListener('change', () => {
      paramValues[input.dataset.param] = input.value || null;
    });
  });

  // Text inputs
  container.querySelectorAll('input[type="text"][data-param]').forEach(input => {
    input.addEventListener('input', () => {
      paramValues[input.dataset.param] = input.value.trim() || null;
    });
  });
}

/**
 * Reset all param values to their defaults from capabilities.
 */
function resetToDefaults() {
  paramValues = {};
  const categories = capabilities?.categories || {};

  // Collect defaults from all modes
  for (const cat of Object.values(categories)) {
    for (const mode of (cat.modes || [])) {
      for (const param of (mode.parameters || [])) {
        if (param.default !== null && param.default !== undefined) {
          paramValues[param.name] = param.default;
        }
      }
    }
  }

  // Collect defaults from shared params
  for (const param of (capabilities?.shared_parameters || [])) {
    if (param.default !== null && param.default !== undefined) {
      paramValues[param.name] = param.default;
    }
  }
}

/**
 * Escape HTML special characters.
 */
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

export default {
  initAdvancedOptions,
  getCurrentMode,
  getAdvancedParams,
  getAdvancedOptions,
  isAdvancedPanelOpen,
  closeAdvancedPanel,
};
