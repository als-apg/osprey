/**
 * ARIEL UI Components
 *
 * Reusable component rendering functions.
 */

/**
 * Format a timestamp for display.
 * @param {string} timestamp - ISO timestamp
 * @returns {string} Formatted date/time
 */
export function formatTimestamp(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format a relative time.
 * @param {string} timestamp - ISO timestamp
 * @returns {string} Relative time string
 */
export function formatRelativeTime(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now - date;

  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return formatTimestamp(timestamp);
}

/**
 * Get score class based on value.
 * @param {number} score - Score value (0-1)
 * @returns {string} CSS class name
 */
export function getScoreClass(score) {
  if (score >= 0.8) return 'score-high';
  if (score >= 0.6) return 'score-medium';
  return 'score-low';
}

/**
 * Render a score badge.
 * @param {number} score - Score value
 * @returns {string} HTML string
 */
export function renderScoreBadge(score) {
  if (score === null || score === undefined) return '';
  const cls = getScoreClass(score);
  return `<span class="score-badge ${cls}">${(score * 100).toFixed(0)}%</span>`;
}

/**
 * Render a status indicator.
 * @param {boolean} healthy - Health status
 * @param {string} label - Status label
 * @returns {string} HTML string
 */
export function renderStatusIndicator(healthy, label) {
  const status = healthy ? 'healthy' : 'error';
  return `
    <span class="status-indicator">
      <span class="status-dot ${status}"></span>
      <span>${label}</span>
    </span>
  `;
}

/**
 * Render a tag list.
 * @param {string[]} tags - Tag values
 * @param {string} type - Tag type (default, accent, amber)
 * @returns {string} HTML string
 */
export function renderTags(tags, type = '') {
  if (!tags || tags.length === 0) return '';
  const cls = type ? `tag-${type}` : '';
  return tags.map(tag => `<span class="tag ${cls}">${escapeHtml(tag)}</span>`).join('');
}

/**
 * Render an entry card.
 * @param {Object} entry - Entry data
 * @returns {string} HTML string
 */
export function renderEntryCard(entry) {
  const score = entry.score !== null ? renderScoreBadge(entry.score) : '';
  const attachmentCount = entry.attachments?.length || 0;
  const keywords = entry.keywords?.slice(0, 5) || [];

  // Extract subject from raw_text (first line or first 100 chars)
  const rawText = entry.raw_text || '';
  const lines = rawText.split('\n');
  const subject = lines[0]?.slice(0, 100) || 'Untitled';
  const preview = lines.slice(1).join('\n').trim() || rawText;

  return `
    <article class="entry-card" data-entry-id="${escapeHtml(entry.entry_id)}" onclick="window.app.showEntry('${escapeHtml(entry.entry_id)}')">
      <div class="entry-card-header">
        <div class="entry-card-meta">
          <span class="entry-id">${escapeHtml(entry.entry_id)}</span>
          <span class="timestamp">${formatTimestamp(entry.timestamp)}</span>
          <span>${escapeHtml(entry.author || 'Unknown')}</span>
          <span class="text-muted">${escapeHtml(entry.source_system)}</span>
        </div>
        ${score}
      </div>
      <div class="entry-card-content">
        ${escapeHtml(preview).slice(0, 300)}${preview.length > 300 ? '...' : ''}
      </div>
      <div class="entry-card-footer">
        ${attachmentCount > 0 ? `<span class="text-muted">📎 ${attachmentCount}</span>` : ''}
        ${keywords.length > 0 ? `<span class="text-muted">🏷️ ${keywords.join(', ')}</span>` : ''}
      </div>
    </article>
  `;
}

/**
 * Render the RAG answer box.
 * @param {string} answer - Generated answer
 * @param {string[]} sources - Source entry IDs
 * @returns {string} HTML string
 */
export function renderAnswerBox(answer, sources = []) {
  if (!answer) return '';

  const sourceLinks = sources.map(id =>
    `<a href="#" onclick="window.app.showEntry('${escapeHtml(id)}'); return false;">${escapeHtml(id)}</a>`
  ).join(', ');

  return `
    <div class="answer-box animate-fade-in">
      <div class="answer-box-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 16v-4M12 8h.01"/>
        </svg>
        <span>RAG Answer</span>
      </div>
      <div class="answer-box-content">
        ${escapeHtml(answer).replace(/\n/g, '<br>')}
      </div>
      ${sources.length > 0 ? `
        <div class="answer-box-sources">
          <strong>Sources:</strong> ${sourceLinks}
        </div>
      ` : ''}
    </div>
  `;
}

/**
 * Render a loading spinner.
 * @param {string} text - Loading text
 * @returns {string} HTML string
 */
export function renderLoading(text = 'Loading...') {
  return `
    <div class="loading-overlay">
      <div class="spinner spinner-lg"></div>
      <p class="loading-text">${escapeHtml(text)}</p>
    </div>
  `;
}

/**
 * Render an empty state.
 * @param {string} title - Empty state title
 * @param {string} text - Empty state description
 * @returns {string} HTML string
 */
export function renderEmptyState(title, text) {
  return `
    <div class="empty-state">
      <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
      </svg>
      <h3 class="empty-state-title">${escapeHtml(title)}</h3>
      <p class="empty-state-text">${escapeHtml(text)}</p>
    </div>
  `;
}

/**
 * Render a stat card for the dashboard.
 * @param {string} title - Stat title
 * @param {string|number} value - Stat value
 * @param {string} subtitle - Stat subtitle
 * @param {string} status - Status (healthy, warning, error)
 * @returns {string} HTML string
 */
export function renderStatCard(title, value, subtitle = '', status = null) {
  const statusDot = status ? `<span class="status-dot ${status}"></span>` : '';
  return `
    <div class="stat-card">
      <div class="stat-card-header">
        <span class="stat-card-title">${escapeHtml(title)}</span>
        ${statusDot}
      </div>
      <div class="stat-card-value">${escapeHtml(String(value))}</div>
      ${subtitle ? `<div class="stat-card-subtitle">${escapeHtml(subtitle)}</div>` : ''}
    </div>
  `;
}

/**
 * Escape HTML special characters.
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
export function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Create an element from HTML string.
 * @param {string} html - HTML string
 * @returns {Element} DOM element
 */
export function createElement(html) {
  const template = document.createElement('template');
  template.innerHTML = html.trim();
  return template.content.firstChild;
}

export default {
  formatTimestamp,
  formatRelativeTime,
  getScoreClass,
  renderScoreBadge,
  renderStatusIndicator,
  renderTags,
  renderEntryCard,
  renderAnswerBox,
  renderLoading,
  renderEmptyState,
  renderStatCard,
  escapeHtml,
  createElement,
};
