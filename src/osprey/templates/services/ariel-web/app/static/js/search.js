/**
 * ARIEL Search Module
 *
 * Search functionality and UI management.
 */

import { searchApi } from './api.js';
import {
  renderEntryCard,
  renderAnswerBox,
  renderLoading,
  renderEmptyState,
  escapeHtml,
} from './components.js';
import { getAdvancedOptions } from './advanced-options.js';

// Search state
let currentQuery = '';
let currentMode = 'auto';
let isSearching = false;
let lastResults = null;

/**
 * Initialize search module.
 */
export function initSearch() {
  const searchInput = document.getElementById('search-input');
  const searchBtn = document.getElementById('search-btn');
  const modeSelect = document.getElementById('search-mode');
  const filtersToggle = document.getElementById('filters-toggle');
  const filtersPanel = document.getElementById('filters-panel');

  // Search input enter key
  searchInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      performSearch();
    }
  });

  // Search button click
  searchBtn?.addEventListener('click', () => {
    performSearch();
  });

  // Mode change
  modeSelect?.addEventListener('change', (e) => {
    currentMode = e.target.value;
  });

  // Filters toggle
  filtersToggle?.addEventListener('click', () => {
    filtersPanel?.classList.toggle('hidden');
  });

  // Focus search on page load
  searchInput?.focus();

  // Keyboard shortcut: / to focus search
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
      e.preventDefault();
      searchInput?.focus();
    }
    // Escape to clear search
    if (e.key === 'Escape' && document.activeElement === searchInput) {
      searchInput.value = '';
      searchInput.blur();
    }
  });
}

/**
 * Perform a search.
 * @param {string} query - Optional query override
 */
export async function performSearch(query = null) {
  const searchInput = document.getElementById('search-input');
  const resultsContainer = document.getElementById('search-results');

  query = query || searchInput?.value?.trim();
  if (!query || isSearching) return;

  currentQuery = query;
  isSearching = true;

  // Show loading state
  if (resultsContainer) {
    resultsContainer.innerHTML = renderLoading('Searching...');
  }

  // Get filter values
  const startDate = document.getElementById('filter-start-date')?.value || null;
  const endDate = document.getElementById('filter-end-date')?.value || null;
  const author = document.getElementById('filter-author')?.value?.trim() || null;
  const sourceSystem = document.getElementById('filter-source')?.value || null;

  // Get advanced options for current mode
  const advancedOptions = getAdvancedOptions(currentMode);

  try {
    const results = await searchApi.search({
      query,
      mode: currentMode,
      maxResults: advancedOptions.maxResults,
      startDate: startDate ? new Date(startDate).toISOString() : null,
      endDate: endDate ? new Date(endDate).toISOString() : null,
      author,
      sourceSystem,
      // Advanced options
      similarityThreshold: advancedOptions.similarityThreshold,
      includeHighlights: advancedOptions.includeHighlights,
      fuzzyFallback: advancedOptions.fuzzyFallback,
      assemblyMaxItems: advancedOptions.assemblyMaxItems,
      assemblyMaxChars: advancedOptions.assemblyMaxChars,
      assemblyMaxCharsPerItem: advancedOptions.assemblyMaxCharsPerItem,
      temperature: advancedOptions.temperature,
      maxTokens: advancedOptions.maxTokens,
      fusionStrategy: advancedOptions.fusionStrategy,
      keywordWeight: advancedOptions.keywordWeight,
      semanticWeight: advancedOptions.semanticWeight,
    });

    lastResults = results;
    renderSearchResults(results);
  } catch (error) {
    console.error('Search failed:', error);
    if (resultsContainer) {
      resultsContainer.innerHTML = `
        <div class="empty-state">
          <h3 class="empty-state-title text-error">Search Failed</h3>
          <p class="empty-state-text">${escapeHtml(error.message)}</p>
        </div>
      `;
    }
  } finally {
    isSearching = false;
  }
}

/**
 * Render search results.
 * @param {Object} results - Search results from API
 */
function renderSearchResults(results) {
  const resultsContainer = document.getElementById('search-results');
  if (!resultsContainer) return;

  // Build results header
  const modesUsed = results.search_modes_used?.join(', ') || 'none';
  const execTime = results.execution_time_ms || 0;

  let html = '';

  // RAG answer if present
  if (results.answer) {
    html += renderAnswerBox(results.answer, results.sources);
  }

  // Results header
  html += `
    <div class="results-header">
      <span class="results-count">
        <strong>${results.total_results}</strong> results
        <span class="text-muted">(${execTime}ms)</span>
      </span>
      <span class="results-modes">
        Modes: ${escapeHtml(modesUsed)}
      </span>
    </div>
  `;

  // Results list
  if (results.entries?.length > 0) {
    html += '<div class="results-list">';
    results.entries.forEach(entry => {
      html += renderEntryCard(entry);
    });
    html += '</div>';
  } else {
    html += renderEmptyState(
      'No Results Found',
      'Try adjusting your search terms or filters.'
    );
  }

  resultsContainer.innerHTML = html;
}

/**
 * Clear search results.
 */
export function clearSearch() {
  const searchInput = document.getElementById('search-input');
  const resultsContainer = document.getElementById('search-results');

  if (searchInput) searchInput.value = '';
  if (resultsContainer) resultsContainer.innerHTML = '';

  currentQuery = '';
  lastResults = null;
}

/**
 * Get current search state.
 * @returns {Object} Current state
 */
export function getSearchState() {
  return {
    query: currentQuery,
    mode: currentMode,
    isSearching,
    results: lastResults,
  };
}

export default {
  initSearch,
  performSearch,
  clearSearch,
  getSearchState,
};
