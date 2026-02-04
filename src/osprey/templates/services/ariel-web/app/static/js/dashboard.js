/**
 * ARIEL Dashboard Module
 *
 * Status monitoring and system health display.
 */

import { statusApi } from './api.js';
import {
  formatTimestamp,
  formatRelativeTime,
  renderStatCard,
  renderLoading,
  escapeHtml,
} from './components.js';

// Refresh interval
let refreshInterval = null;

/**
 * Initialize dashboard module.
 */
export function initDashboard() {
  const refreshBtn = document.getElementById('refresh-status');
  refreshBtn?.addEventListener('click', () => loadStatus());
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
    container.innerHTML = `
      <div class="empty-state">
        <h3 class="empty-state-title text-error">Failed to Load Status</h3>
        <p class="empty-state-text">${escapeHtml(error.message)}</p>
      </div>
    `;
  }
}

/**
 * Render status dashboard.
 * @param {HTMLElement} container - Container element
 * @param {Object} status - Status data
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
          <div style="font-size: var(--text-lg); font-weight: 600; color: var(--text-primary);">
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
            { name: 'Semantic Search', enabled: status.enabled_search_modules?.includes('semantic') },
            { name: 'RAG Search', enabled: status.enabled_search_modules?.includes('rag') },
            { name: 'Vision Search', enabled: false, future: true },
          ])}
        </div>
      </div>

      <!-- Enhancement Modules -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Enhancement Modules</span>
        </div>
        <div class="card-body">
          ${renderModuleList([
            { name: 'Text Embedding', enabled: status.enabled_enhancement_modules?.includes('text_embedding') },
            { name: 'Semantic Processor', enabled: status.enabled_enhancement_modules?.includes('semantic_processor') },
          ])}
        </div>
      </div>
    </div>

    <!-- Embedding Tables -->
    ${status.embedding_tables?.length > 0 ? `
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
              ${status.embedding_tables.map(table => `
                <tr>
                  <td class="font-mono">${escapeHtml(table.table_name.replace('text_embeddings_', ''))}</td>
                  <td class="font-mono">${table.entry_count.toLocaleString()}</td>
                  <td class="font-mono">${table.dimension || '-'}</td>
                  <td>
                    ${table.is_active
                      ? '<span class="status-indicator"><span class="status-dot healthy"></span> Active</span>'
                      : '<span class="text-muted">-</span>'
                    }
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    ` : ''}

    <!-- Errors -->
    ${status.errors?.length > 0 ? `
      <div class="card" style="margin-top: 24px; border-color: var(--color-error);">
        <div class="card-header" style="background: var(--color-error-bg);">
          <span class="card-title text-error">Errors</span>
        </div>
        <div class="card-body">
          <ul style="margin: 0; padding-left: 20px;">
            ${status.errors.map(err => `<li class="text-error">${escapeHtml(err)}</li>`).join('')}
          </ul>
        </div>
      </div>
    ` : ''}
  `;
}

/**
 * Render module list.
 * @param {Array} modules - Module definitions
 * @returns {string} HTML string
 */
function renderModuleList(modules) {
  return `
    <div style="display: flex; flex-direction: column; gap: 12px;">
      ${modules.map(m => {
        const dot = m.enabled ? 'healthy' : (m.future ? 'inactive' : 'inactive');
        const label = m.enabled ? 'Enabled' : (m.future ? '(Future)' : 'Disabled');
        const labelColor = m.enabled ? 'text-success' : 'text-muted';
        return `
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <span class="status-indicator">
              <span class="status-dot ${dot}"></span>
              <span>${escapeHtml(m.name)}</span>
            </span>
            <span class="${labelColor}" style="font-size: var(--text-sm);">${label}</span>
          </div>
        `;
      }).join('')}
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

export default {
  initDashboard,
  loadStatus,
  startAutoRefresh,
  stopAutoRefresh,
};
