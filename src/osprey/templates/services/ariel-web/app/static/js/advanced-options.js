/**
 * ARIEL Advanced Options Module
 *
 * State management and UI for advanced search parameters.
 * Implements progressive disclosure with mode-based visibility.
 */

// Default values for advanced options
const DEFAULTS = {
  maxResults: 20,
  similarityThreshold: 0.7,
  includeHighlights: true,
  fuzzyFallback: true,
  assemblyMaxItems: 10,
  assemblyMaxChars: 12000,
  assemblyMaxCharsPerItem: 2000,
  temperature: 0.1,
  maxTokens: 1024,
  fusionStrategy: 'rrf',
  keywordWeight: 0.5,
  semanticWeight: 0.5,
};

// Current advanced options state
let advancedState = { ...DEFAULTS };

// Track if panel is open
let isPanelOpen = false;

/**
 * Initialize advanced options module.
 */
export function initAdvancedOptions() {
  // Get elements
  const advancedToggleBtn = document.getElementById('advanced-toggle-btn');
  const advancedCloseBtn = document.getElementById('advanced-close-btn');
  const advancedResetBtn = document.getElementById('advanced-reset-btn');
  const advancedPanel = document.getElementById('advanced-panel');
  const modeSelect = document.getElementById('search-mode');
  const filtersShowBtn = document.getElementById('filters-show-btn');
  const filtersPanel = document.getElementById('filters-panel');

  // Toggle advanced panel
  advancedToggleBtn?.addEventListener('click', () => {
    toggleAdvancedPanel();
  });

  // Close advanced panel
  advancedCloseBtn?.addEventListener('click', () => {
    closeAdvancedPanel();
  });

  // Reset to defaults
  advancedResetBtn?.addEventListener('click', () => {
    resetToDefaults();
  });

  // Mode change - update visibility
  modeSelect?.addEventListener('change', (e) => {
    updateVisibilityForMode(e.target.value);
  });

  // Filters show button
  filtersShowBtn?.addEventListener('click', () => {
    filtersPanel?.classList.toggle('hidden');
  });

  // Setup slider handlers
  setupSliderHandlers();

  // Setup toggle handlers
  setupToggleHandlers();

  // Setup fusion strategy change handler
  setupFusionStrategyHandler();

  // Set initial visibility based on current mode
  const currentMode = modeSelect?.value || 'auto';
  updateVisibilityForMode(currentMode);
}

/**
 * Toggle advanced panel visibility.
 */
function toggleAdvancedPanel() {
  const panel = document.getElementById('advanced-panel');
  const btn = document.getElementById('advanced-toggle-btn');

  if (panel) {
    isPanelOpen = !isPanelOpen;
    panel.classList.toggle('hidden', !isPanelOpen);
    btn?.classList.toggle('active', isPanelOpen);
  }
}

/**
 * Close advanced panel.
 */
function closeAdvancedPanel() {
  const panel = document.getElementById('advanced-panel');
  const btn = document.getElementById('advanced-toggle-btn');

  isPanelOpen = false;
  panel?.classList.add('hidden');
  btn?.classList.remove('active');
}

/**
 * Reset all values to defaults.
 */
function resetToDefaults() {
  advancedState = { ...DEFAULTS };

  // Update all slider and input values
  setSliderValue('adv-max-results', DEFAULTS.maxResults);
  setSliderValue('adv-similarity-threshold', DEFAULTS.similarityThreshold.toFixed(2));
  setSliderValue('adv-assembly-max-items', DEFAULTS.assemblyMaxItems);
  setSliderValue('adv-assembly-max-chars', DEFAULTS.assemblyMaxChars);
  setSliderValue('adv-assembly-max-chars-per-item', DEFAULTS.assemblyMaxCharsPerItem);
  setSliderValue('adv-temperature', DEFAULTS.temperature.toFixed(2));
  setSliderValue('adv-max-tokens', DEFAULTS.maxTokens);
  setSliderValue('adv-keyword-weight', DEFAULTS.keywordWeight.toFixed(2));
  setSliderValue('adv-semantic-weight', DEFAULTS.semanticWeight.toFixed(2));

  // Update toggles
  const highlightsToggle = document.getElementById('adv-include-highlights');
  const fuzzyToggle = document.getElementById('adv-fuzzy-fallback');
  if (highlightsToggle) highlightsToggle.checked = DEFAULTS.includeHighlights;
  if (fuzzyToggle) fuzzyToggle.checked = DEFAULTS.fuzzyFallback;

  // Update fusion strategy
  const fusionSelect = document.getElementById('adv-fusion-strategy');
  if (fusionSelect) fusionSelect.value = DEFAULTS.fusionStrategy;
  updateFusionWeightsVisibility(DEFAULTS.fusionStrategy);
}

/**
 * Set slider value and update display.
 */
function setSliderValue(sliderId, value) {
  const slider = document.getElementById(sliderId);
  const valueDisplay = document.getElementById(`${sliderId}-value`);

  if (slider) {
    slider.value = value;
  }
  if (valueDisplay) {
    valueDisplay.textContent = value;
  }
}

/**
 * Setup all slider input handlers.
 */
function setupSliderHandlers() {
  const sliders = [
    { id: 'adv-max-results', key: 'maxResults', format: (v) => parseInt(v) },
    { id: 'adv-similarity-threshold', key: 'similarityThreshold', format: (v) => parseFloat(v).toFixed(2) },
    { id: 'adv-assembly-max-items', key: 'assemblyMaxItems', format: (v) => parseInt(v) },
    { id: 'adv-assembly-max-chars', key: 'assemblyMaxChars', format: (v) => parseInt(v) },
    { id: 'adv-assembly-max-chars-per-item', key: 'assemblyMaxCharsPerItem', format: (v) => parseInt(v) },
    { id: 'adv-temperature', key: 'temperature', format: (v) => parseFloat(v).toFixed(2) },
    { id: 'adv-max-tokens', key: 'maxTokens', format: (v) => parseInt(v) },
    { id: 'adv-keyword-weight', key: 'keywordWeight', format: (v) => parseFloat(v).toFixed(2) },
    { id: 'adv-semantic-weight', key: 'semanticWeight', format: (v) => parseFloat(v).toFixed(2) },
  ];

  sliders.forEach(({ id, key, format }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(`${id}-value`);

    if (slider) {
      slider.addEventListener('input', (e) => {
        const formatted = format(e.target.value);
        advancedState[key] = parseFloat(formatted) || parseInt(formatted);
        if (valueDisplay) {
          valueDisplay.textContent = formatted;
        }
      });
    }
  });
}

/**
 * Setup toggle switch handlers.
 */
function setupToggleHandlers() {
  const toggles = [
    { id: 'adv-include-highlights', key: 'includeHighlights' },
    { id: 'adv-fuzzy-fallback', key: 'fuzzyFallback' },
  ];

  toggles.forEach(({ id, key }) => {
    const toggle = document.getElementById(id);
    if (toggle) {
      toggle.addEventListener('change', (e) => {
        advancedState[key] = e.target.checked;
      });
    }
  });
}

/**
 * Setup fusion strategy dropdown handler.
 */
function setupFusionStrategyHandler() {
  const fusionSelect = document.getElementById('adv-fusion-strategy');

  if (fusionSelect) {
    fusionSelect.addEventListener('change', (e) => {
      advancedState.fusionStrategy = e.target.value;
      updateFusionWeightsVisibility(e.target.value);
    });
  }
}

/**
 * Update visibility of fusion weight controls based on strategy.
 */
function updateFusionWeightsVisibility(strategy) {
  const keywordControl = document.getElementById('keyword-weight-control');
  const semanticControl = document.getElementById('semantic-weight-control');

  const showWeights = strategy === 'weighted';

  if (keywordControl) {
    keywordControl.style.display = showWeights ? '' : 'none';
  }
  if (semanticControl) {
    semanticControl.style.display = showWeights ? '' : 'none';
  }
}

/**
 * Update visibility of controls based on selected mode.
 * @param {string} mode - Current search mode
 */
function updateVisibilityForMode(mode) {
  // Get all elements with data-modes attribute
  const modeElements = document.querySelectorAll('[data-modes]');

  modeElements.forEach((element) => {
    const allowedModes = element.dataset.modes.split(',').map(m => m.trim().toLowerCase());
    const shouldShow = allowedModes.includes(mode.toLowerCase());

    // Check if this is a section or a control
    if (element.classList.contains('advanced-section')) {
      element.style.display = shouldShow ? '' : 'none';
    } else {
      element.style.display = shouldShow ? '' : 'none';
    }
  });

  // Update fusion weights visibility based on current strategy
  if (mode === 'multi') {
    const fusionSelect = document.getElementById('adv-fusion-strategy');
    updateFusionWeightsVisibility(fusionSelect?.value || 'rrf');
  }
}

/**
 * Get current advanced options for search.
 * Returns only options relevant to the current mode.
 * @param {string} mode - Current search mode
 * @returns {Object} Advanced options object
 */
export function getAdvancedOptions(mode) {
  const options = {
    maxResults: advancedState.maxResults,
  };

  // Mode-specific options
  const modeLC = mode.toLowerCase();

  // Similarity threshold for semantic-based modes
  if (['semantic', 'rag', 'multi'].includes(modeLC)) {
    options.similarityThreshold = advancedState.similarityThreshold;
  }

  // Keyword options
  if (['keyword', 'multi'].includes(modeLC)) {
    options.includeHighlights = advancedState.includeHighlights;
    options.fuzzyFallback = advancedState.fuzzyFallback;
  }

  // Assembly options (all modes except agent benefit from this)
  if (modeLC !== 'agent') {
    options.assemblyMaxItems = advancedState.assemblyMaxItems;
    options.assemblyMaxChars = advancedState.assemblyMaxChars;
    options.assemblyMaxCharsPerItem = advancedState.assemblyMaxCharsPerItem;
  }

  // RAG processing options
  if (modeLC === 'rag') {
    options.temperature = advancedState.temperature;
    options.maxTokens = advancedState.maxTokens;
  }

  // Fusion options for multi mode
  if (modeLC === 'multi') {
    options.fusionStrategy = advancedState.fusionStrategy;
    if (advancedState.fusionStrategy === 'weighted') {
      options.keywordWeight = advancedState.keywordWeight;
      options.semanticWeight = advancedState.semanticWeight;
    }
  }

  return options;
}

/**
 * Check if advanced panel is currently open.
 * @returns {boolean}
 */
export function isAdvancedPanelOpen() {
  return isPanelOpen;
}

export default {
  initAdvancedOptions,
  getAdvancedOptions,
  isAdvancedPanelOpen,
};
