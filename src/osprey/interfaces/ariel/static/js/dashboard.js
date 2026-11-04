// @ts-check
/**
 * ARIEL Status drawer.
 *
 * Renders system health into #status-content inside #status-drawer.
 * Store keys: reads `drawer` only, never writes it. layout.js alone opens and
 * closes the drawer element; this module loads Status when `drawer` becomes
 * 'status' and auto-refreshes only while it stays 'status'.
 */

import { statusApi } from './api.js';
import {
  formatTimestamp,
  formatRelativeTime,
  renderStatCard,
  renderLoading,
  renderErrorState,
  escapeHtml,
} from './components.js';
import { getState, subscribe } from './store.js';

// Refresh interval
/** @type {ReturnType<typeof setInterval> | null} */
let refreshInterval = null;

/**
 * Wire the Status drawer: the refresh button, and load + auto-refresh while
 * the store's `drawer` is 'status'.
 * @param {ParentNode} [root]
 * @returns {{destroy: () => void}}
 */
export function initDashboard(root = document) {
  const controller = new AbortController();
  const refreshBtn = root.querySelector('#refresh-status');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => loadStatus(), { signal: controller.signal });
  }

  /** @param {unknown} drawer */
  const apply = (drawer) => {
    if (drawer === 'status') {
      loadStatus();
      startAutoRefresh();
    } else {
      stopAutoRefresh();
    }
  };

  const unsubscribe = subscribe((next, _prev, changed) => {
    if (changed.has('drawer')) apply(next.drawer);
  });
  if (getState().drawer === 'status') apply('status');

  return {
    destroy: () => {
      controller.abort();
      unsubscribe();
      stopAutoRefresh();
    },
  };
}

/**
 * Load and display status.
 */
export async function loadStatus() {
  const container = document.getElementById('status-content');
  if (!container) return;

  container.innerHTML = renderLoading('Loading status...');

  try {
    const status = await statusApi.get();
    renderStatus(container, status);
  } catch (error) {
    console.error('Failed to load status:', error);
    container.innerHTML = renderErrorState('Failed to Load Status', error);
  }
}

/**
 * Render status dashboard.
 * @param {HTMLElement} container - Container element
 * @param {any} status - Status data
 */
function renderStatus(container, status) {
  const healthStatus = status.healthy ? 'healthy' : 'error';
  const healthLabel = status.healthy ? 'ARIEL is healthy' : 'ARIEL has issues';

  container.innerHTML = `
    <!-- Overall Health -->
    <div class="card" style="margin-bottom: 24px;">
      <div class="card-body" style="display: flex; align-items: center; gap: 16px;">
        <span class="status-dot ${healthStatus}" style="width: 16px; height: 16px;"></span>
        <div>
          <div style="font-size: var(--text-xl); font-weight: var(--weight-semibold); color: var(--text-primary);">
            ${healthLabel}
          </div>
          <div class="text-muted">
            Database: ${status.database_connected ? 'Connected' : 'Disconnected'}
            ${status.entry_count !== null ? ` | Entries: ${status.entry_count.toLocaleString()}` : ''}
            ${status.last_ingestion ? ` | Last Ingestion: ${formatRelativeTime(status.last_ingestion)}` : ''}
          </div>
        </div>
      </div>
    </div>

    <!-- Stats Grid -->
    <div class="dashboard-grid">
      ${renderStatCard('Total Entries', status.entry_count?.toLocaleString() || '0', 'in database')}
      ${renderStatCard('Database', status.database_connected ? 'Connected' : 'Disconnected', status.database_uri, status.database_connected ? 'healthy' : 'error')}
      ${renderStatCard('Active Model', status.active_embedding_model || 'None', 'for semantic search')}
      ${renderStatCard('Last Ingestion', status.last_ingestion ? formatRelativeTime(status.last_ingestion) : 'Never', status.last_ingestion ? formatTimestamp(status.last_ingestion) : '')}
    </div>

    <!-- Modules -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; margin-top: 24px;">
      <!-- Search Modules -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Search Modules</span>
        </div>
        <div class="card-body">
          ${renderModuleList([
            { name: 'Keyword Search', enabled: status.enabled_search_modules?.includes('keyword') },
            {
              name: 'Semantic Search',
              enabled: status.enabled_search_modules?.includes('semantic'),
            },
          ])}
        </div>
      </div>
    </div>

    <!-- Enhancement Coverage: full width, one column per state -->
    <div class="card" style="margin-top: 24px;">
      <div class="card-header">
        <span class="card-title">Enhancement Modules</span>
      </div>
      <div class="card-body" style="padding: 0;">
        ${renderCoverageTable(status.enhancement_coverage)}
      </div>
    </div>

    <!-- Embedding Tables -->
    ${
      status.embedding_tables?.length > 0
        ? `
      <div class="card" style="margin-top: 24px;">
        <div class="card-header">
          <span class="card-title">Embedding Models</span>
        </div>
        <div class="card-body" style="padding: 0;">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Entries</th>
                <th>Dimension</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              ${status.embedding_tables
                .map(
                  (/** @type {any} */ table) => `
                <tr>
                  <td class="font-mono">${escapeHtml(table.table_name.replace('text_embeddings_', ''))}</td>
                  <td class="font-mono">${table.entry_count.toLocaleString()}</td>
                  <td class="font-mono">${table.dimension || '-'}</td>
                  <td>
                    ${
                      table.is_active
                        ? '<span class="status-indicator"><span class="status-dot healthy"></span> Active</span>'
                        : '<span class="text-muted">-</span>'
                    }
                  </td>
                </tr>
              `
                )
                .join('')}
            </tbody>
          </table>
        </div>
      </div>
    `
        : ''
    }

    <!-- Errors -->
    ${
      status.errors?.length > 0
        ? `
      <div class="card" style="margin-top: 24px; border-color: var(--color-error);">
        <div class="card-header" style="background: var(--color-error-bg);">
          <span class="card-title text-error">Errors</span>
        </div>
        <div class="card-body">
          <ul style="margin: 0; padding-left: 20px;">
            ${status.errors.map((/** @type {any} */ err) => `<li class="text-error">${escapeHtml(err)}</li>`).join('')}
          </ul>
        </div>
      </div>
    `
        : ''
    }
  `;
}

/** Coverage columns, in display order: [state key, column heading]. */
const COVERAGE_COLUMNS = /** @type {const} */ ([
  ['complete', 'Complete'],
  ['pending', 'Pending'],
  ['stale', 'Stale'],
  ['failed', 'Failed'],
  ['n/a', 'N/A'],
]);

/**
 * One count cell: the number, or an em dash when the server sent none.
 * @param {unknown} value
 * @returns {string}
 */
function formatCount(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString() : '\u2014';
}

/**
 * Render the per-module enhancement coverage table: one row per module the
 * server lists (it lists only enabled ones), labelled with the server's label.
 * @param {import('./api.js').EnhancementCoverage | undefined | null} coverage
 * @returns {string} HTML string
 */
function renderCoverageTable(coverage) {
  const modules = coverage ? Object.entries(coverage) : [];
  if (modules.length === 0) {
    return '<div class="text-muted" style="padding: 16px;">No enhancement modules enabled</div>';
  }
  return `
    <table class="coverage-table">
      <thead>
        <tr>
          <th>Module</th>
          ${COVERAGE_COLUMNS.map(([, heading]) => `<th>${heading}</th>`).join('')}
        </tr>
      </thead>
      <tbody>
        ${modules
          .map(
            ([name, counts]) => `
          <tr data-module="${escapeHtml(name)}">
            <td>${escapeHtml(counts?.label || name)}</td>
            ${COVERAGE_COLUMNS.map(
              ([key]) =>
                `<td class="font-mono" data-state="${key}">${formatCount(counts?.[key])}</td>`
            ).join('')}
          </tr>
        `
          )
          .join('')}
      </tbody>
    </table>
  `;
}

/**
 * Render module list.
 * @param {Array<any>} modules - Module definitions
 * @returns {string} HTML string
 */
function renderModuleList(modules) {
  return `
    <div style="display: flex; flex-direction: column; gap: 12px;">
      ${modules
        .map((/** @type {any} */ m) => {
          const dot = m.enabled ? 'healthy' : 'inactive';
          const label = m.enabled ? 'Enabled' : 'Disabled';
          const labelColor = m.enabled ? 'text-success' : 'text-muted';
          return `
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <span class="status-indicator">
              <span class="status-dot ${dot}"></span>
              <span>${escapeHtml(m.name)}</span>
            </span>
            <span class="${labelColor}" style="font-size: var(--text-xl);">${label}</span>
          </div>
        `;
        })
        .join('')}
    </div>
  `;
}

/**
 * Start auto-refresh.
 * @param {number} interval - Refresh interval in ms
 */
export function startAutoRefresh(interval = 30000) {
  stopAutoRefresh();
  refreshInterval = setInterval(() => loadStatus(), interval);
}

/**
 * Stop auto-refresh.
 */
export function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}
