// @ts-check
/**
 * ARIEL UI Components
 *
 * Reusable component rendering functions.
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { formatFacilityTime } from '/design-system/js/facility-time.js';
import { parseEntryText } from './entries-helpers.js';
import { messageOf } from './utils.js';
import { getStripPref } from './display-prefs.js';

/**
 * Stamp the zone the capabilities name on `<html>`, where the shared
 * facility-time formatter reads it. ARIEL's page is a static file, so this is
 * how the zone reaches it. A payload without a zone (a failed fetch) leaves
 * the page unstamped, and times then read in the viewer's own zone, named.
 * @param {{ facility_timezone?: string }|null} [capabilities]
 */
export function applyFacilityTimezone(capabilities) {
  const zone = capabilities?.facility_timezone;
  if (typeof zone === 'string' && zone) {
    document.documentElement.setAttribute('data-facility-timezone', zone);
  }
}

/**
 * A list row (C-ROW, `view=row`).
 * @typedef {Object} Entry
 * @property {string} entry_id
 * @property {string|null} [timestamp]
 * @property {string} [author]
 * @property {string} [source_system]
 * @property {string|null} [subject]
 * @property {string} [snippet]
 * @property {string} [raw_text] - A search result's full text (rows from the listing carry `snippet`)
 * @property {number|null} [score]
 * @property {number} [attachment_count]
 * @property {Record<string, string>} [enhancements]
 * @property {string[]} [keywords]
 * @property {string[]} [highlights]
 * @property {string[]|null} [matched_via]
 */

/**
 * One enhancement module, in registry order (capabilities.enhancement_modules).
 * @typedef {{name: string, label: string}} EnhancementModule
 */

/** @type {EnhancementModule[]} */
let enhancementModules = [];

/**
 * Set the modules the badge strip shows, once at boot.
 * @param {EnhancementModule[]|null|undefined} modules
 */
export function setEnhancementModules(modules) {
  enhancementModules = Array.isArray(modules)
    ? modules.filter((m) => m && typeof m.name === 'string')
    : [];
}

/** Strip cell states and the class token each maps to (`n/a` is no class token). */
const STRIP_STATES = /** @type {Record<string, string>} */ ({
  complete: 'complete',
  pending: 'pending',
  stale: 'stale',
  failed: 'failed',
  'n/a': 'na',
});

/**
 * The enhancement badge strip: one cell per module that applies to the entry,
 * in module order, styled by the entry's state for it. A module that does not
 * apply (`n/a`, or a missing or unknown state) renders no cell, and an entry
 * no module applies to renders no strip. Each cell carries its text
 * alternative (`role="img"` with `aria-label`).
 * @param {Record<string, string>|null|undefined} enhancements
 * @param {EnhancementModule[]} modules
 * @returns {string}
 */
export function renderBadgeStrip(enhancements, modules) {
  if (!Array.isArray(modules) || modules.length === 0) return '';
  const cells = modules.flatMap((m) => {
    const raw = enhancements?.[m.name];
    const state = typeof raw === 'string' && raw in STRIP_STATES ? raw : 'n/a';
    if (state === 'n/a') return [];
    const text = `${m.label || m.name}: ${state}`;
    return `<span class="ariel-strip-cell ariel-strip-cell-${STRIP_STATES[state]}" role="img" data-module="${escapeHtml(m.name)}" data-state="${escapeHtml(state)}" title="${escapeHtml(text)}" aria-label="${escapeHtml(text)}"></span>`;
  });
  if (cells.length === 0) return '';
  return `<span class="ariel-strip" role="group" aria-label="Enhancements">${cells.join('')}</span>`;
}

/**
 * @typedef {Object} Diagnostic
 * @property {string} level - Severity ('info', 'warning', 'error')
 * @property {string} source
 * @property {string} message
 * @property {string} [category]
 */

/**
 * One vocabulary expansion that was actually applied to the query: the term
 * the operator typed and every canonical form it was expanded to.
 * @typedef {Object} ExpandedTerm
 * @property {string} original
 * @property {string[]} alternatives
 */

/**
 * What operators call each writer that stamps `created_via` (and a draft's
 * `staged_by`): the stored value names the program, the screen names who. A
 * value not listed here is shown as stored.
 */
const SOURCE_NAMES = /** @type {Record<string, string>} */ ({
  'ariel-mcp': 'Agent',
  'ariel-web': 'Logbook',
  'gallery-compose': 'Artifact gallery',
});

/**
 * The display name of a `created_via` value, e.g. `ariel-mcp` -> `Agent`.
 * @param {string} createdVia
 * @returns {string}
 */
export function sourceName(createdVia) {
  return SOURCE_NAMES[createdVia] ?? createdVia;
}

/**
 * A draft's "staged by" phrase: `staged by the agent` for the agent's
 * drafts, else `staged by` and the source's display name.
 * @param {string} stagedBy - The draft's stored `staged_by`
 * @returns {string}
 */
export function stagedByText(stagedBy) {
  return stagedBy === 'ariel-mcp' ? 'staged by the agent' : `staged by ${sourceName(stagedBy)}`;
}

/**
 * Format a timestamp for display, in the facility zone and the viewer's date
 * convention. Falls back to the raw string if the timestamp can't be parsed.
 * @param {string} [timestamp] - ISO timestamp
 * @returns {string} Formatted date/time
 */
export function formatTimestamp(timestamp) {
  if (!timestamp) return '';
  return (
    formatFacilityTime(timestamp, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }) || timestamp
  );
}

/**
 * Format a timestamp's time of day only, for a row under a day heading that
 * already names the date. Same zone and convention as `formatTimestamp`.
 * @param {string} [timestamp] - ISO timestamp
 * @returns {string} Formatted time
 */
export function formatTime(timestamp) {
  if (!timestamp) return '';
  return formatFacilityTime(timestamp, { hour: '2-digit', minute: '2-digit' }) || timestamp;
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
  const diff = now.getTime() - date.getTime();

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
 * @param {number|null|undefined} score - Score value
 * @returns {string} HTML string
 */
export function renderScoreBadge(score) {
  if (score === null || score === undefined) return '';
  const cls = getScoreClass(score);
  return `<span class="score-badge ${cls}">${(score * 100).toFixed(0)}%</span>`;
}

/**
 * Render a tag list.
 * @param {string[]} tags - Tag values
 * @param {string} type - Tag type (default, accent)
 * @returns {string} HTML string
 */
export function renderTags(tags, type = '') {
  if (!tags || tags.length === 0) return '';
  const cls = type ? `tag-${type}` : '';
  return tags.map((tag) => `<span class="tag ${cls}">${escapeHtml(tag)}</span>`).join('');
}

/**
 * Sanitize a highlight snippet from ts_headline, allowing only <b> and </b> tags.
 * All other HTML is escaped for defense-in-depth.
 * @param {string} html - Raw highlight string from PostgreSQL ts_headline
 * @returns {string} Sanitized HTML safe for innerHTML
 */
export function sanitizeHighlight(html) {
  if (!html) return '';
  // Replace <b> and </b> with placeholders, escape everything else, restore placeholders
  return (
    html
      .replace(/<b>/gi, '\x00B_OPEN\x00')
      .replace(/<\/b>/gi, '\x00B_CLOSE\x00')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;') // hygiene-allow-color: HTML entity, not a hex color (scanner false positive)
      // eslint-disable-next-line no-control-regex -- NUL sentinels are intentional bold-marker placeholders swapped back to tags
      .replace(/\x00B_OPEN\x00/g, '<b>')
      // eslint-disable-next-line no-control-regex -- NUL sentinels are intentional bold-marker placeholders swapped back to tags
      .replace(/\x00B_CLOSE\x00/g, '</b>')
  );
}

/**
 * One list row from the C-ROW fields: subject (else the snippet's first
 * line), time, author, score badge, highlights or snippet, attachment count,
 * picture-match marker, keywords, and the badge strip when `strip` is on
 * (default: the display preference). A row with neither highlights nor a
 * snippet (a search result, which carries the full text) previews the text
 * after its first line. `timeOnly` is for a row under a day heading: the row
 * shows the time and keeps the full date as its tooltip. Every interpolated
 * value is escaped; the entry id is carried only as a `data-*` attribute.
 * @param {Entry} entry
 * @param {string|null} [selected] - The selected entry id
 * @param {{strip?: boolean, timeOnly?: boolean}} [options]
 * @returns {string}
 */
export function renderListRow(
  entry,
  selected = null,
  { strip = getStripPref(), timeOnly = false } = {}
) {
  const id = String(entry.entry_id ?? '');
  const isSelected = selected !== null && selected === id;
  const score = renderScoreBadge(entry.score);
  const attachmentCount = Number(entry.attachment_count) || 0;
  const keywords = entry.keywords?.slice(0, 5) || [];
  const pictureMatch = Array.isArray(entry.matched_via) && entry.matched_via.includes('image');
  const snippet = entry.snippet || (entry.raw_text ? parseEntryText(entry.raw_text).details : '');
  const subject = entry.subject || parseEntryText(entry.snippet || entry.raw_text || '').subject;

  const highlights = entry.highlights || [];
  const preview =
    highlights.length > 0
      ? highlights.map((/** @type {string} */ h) => sanitizeHighlight(h)).join(' &hellip; ')
      : escapeHtml(snippet);

  const foot = [
    strip ? renderBadgeStrip(entry.enhancements, enhancementModules) : '',
    attachmentCount > 0
      ? `<span class="text-muted" data-attachment-count="${attachmentCount}">📎 ${attachmentCount}</span>`
      : '',
    pictureMatch
      ? '<span class="text-muted" data-match-marker="image" title="Matched through a picture in this entry">🖼️ picture match</span>'
      : '',
    keywords.length > 0
      ? `<span class="keyword-list">🏷️ ${keywords
          .map((/** @type {string} */ kw) => `<span class="keyword-tag">${escapeHtml(kw)}</span>`)
          .join('')}</span>`
      : '',
  ].join('');

  return `
    <div class="osprey-list-row multiline ariel-row" role="option" tabindex="-1"
         data-entry-id="${escapeHtml(id)}" aria-selected="${isSelected}">
      <div class="ariel-row-top">
        <span class="osprey-list-title ariel-row-subject">${escapeHtml(subject)}</span>
        ${renderRowTime(entry.timestamp, timeOnly)}
      </div>
      <div class="ariel-row-meta">
        <span class="ariel-row-author">${escapeHtml(entry.author || 'Unknown')}</span>
        <span class="text-muted">${escapeHtml(entry.source_system || '')}</span>
        ${score}
      </div>
      <div class="ariel-row-preview">${preview}</div>
      ${foot ? `<div class="ariel-row-foot">${foot}</div>` : ''}
    </div>
  `;
}

/**
 * A row's `<time>`: the full date and time, or under a day heading the time
 * only with the full form as its tooltip.
 * @param {string|null|undefined} timestamp
 * @param {boolean} timeOnly
 * @returns {string}
 */
function renderRowTime(timestamp, timeOnly) {
  const full = formatTimestamp(timestamp || undefined);
  const datetime = escapeHtml(timestamp || '');
  if (!timeOnly)
    return `<time class="ariel-row-time" datetime="${datetime}">${escapeHtml(full)}</time>`;
  return `<time class="ariel-row-time" datetime="${datetime}" title="${escapeHtml(full)}">${escapeHtml(formatTime(timestamp || undefined))}</time>`;
}

/**
 * The calendar day an entry falls on in the facility time zone; the day
 * separator's label and the key consecutive rows are grouped by.
 * @param {string} timestamp
 * @returns {string}
 */
function dayLabel(timestamp) {
  if (!timestamp) return '';
  return (
    formatFacilityTime(timestamp, {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    }) || ''
  );
}

/**
 * Rows for a page of listing entries, with a day separator wherever the day
 * changes — including against the last row already on screen (`lastDay`).
 * @param {any[]} entries
 * @param {string|null} lastDay
 * @param {string|null} selected
 * @returns {{html: string, lastDay: string|null}}
 */
export function renderDayGroupedRows(entries, lastDay, selected) {
  let html = '';
  let day = lastDay;
  for (const entry of entries) {
    const label = dayLabel(entry.timestamp);
    if (label !== day) {
      html += `<div class="ariel-day-sep" role="presentation" data-day="${escapeHtml(label)}">${escapeHtml(label)}</div>`;
      day = label;
    }
    html += renderListRow(entry, selected, { timeOnly: true });
  }
  return { html, lastDay: day };
}

/** The rerank status line's text per driver status. */
export const RERANK_STATUS_TEXT = {
  reranking: 'Search complete — reranking with LLM…',
  updated: 'Results updated after reranking',
  fallback: 'Could not improve the ranking — showing fast results',
};

/**
 * The rerank status line, or nothing.
 * @param {'reranking'|'updated'|'fallback'|null} status
 * @returns {string}
 */
export function renderRerankStatus(status) {
  if (!status) return '';
  return `
    <div class="results-modes" data-testid="rerank-status" role="status" aria-live="polite">
      ${escapeHtml(RERANK_STATUS_TEXT[status])}
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
 * Render an error state for a failed load, matching renderEmptyState's markup
 * minus the icon, with the message sourced from a caught error via messageOf().
 * Shared by every module's load-failure catch block (dashboard, list, reader)
 * so they render identical markup.
 * @param {string} title - Error heading
 * @param {unknown} error - The caught error
 * @returns {string} HTML string
 */
export function renderErrorState(title, error) {
  return `
    <div class="empty-state">
      <h3 class="empty-state-title text-error">${escapeHtml(title)}</h3>
      <p class="empty-state-text">${escapeHtml(messageOf(error))}</p>
    </div>
  `;
}

/**
 * Render a stat card for the dashboard.
 * @param {string} title - Stat title
 * @param {string|number} value - Stat value
 * @param {string} subtitle - Stat subtitle
 * @param {string|null} status - Status (healthy, error)
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
 * The most severe level among a search's diagnostics.
 * @param {Diagnostic[]} diagnostics
 * @returns {'info'|'warning'|'error'}
 */
export function diagnosticsLevel(diagnostics) {
  const levels = /** @type {const} */ (['info', 'warning', 'error']);
  let maxLevel = /** @type {'info'|'warning'|'error'} */ ('info');
  for (const d of diagnostics) {
    const level = levels.find((l) => l === d.level);
    if (level && levels.indexOf(level) > levels.indexOf(maxLevel)) maxLevel = level;
  }
  return maxLevel;
}

/**
 * A search's diagnostics as one plain sentence, for a tooltip or an
 * accessible description: `1 search issue: hybrid: Picture search unavailable`.
 * @param {Diagnostic[]} diagnostics
 * @returns {string}
 */
export function diagnosticsText(diagnostics) {
  const count = diagnostics.length;
  const head = count === 1 ? '1 search issue' : `${count} search issues`;
  return `${head}: ${diagnostics.map((d) => `${d.source}: ${d.message}`).join('; ')}`;
}

/**
 * Render a diagnostics bar for search issues.
 * Uses native <details>/<summary> for zero-JS collapsibility.
 * @param {Diagnostic[]} diagnostics - Array of {level, source, message, category}
 * @param {{open?: boolean}} [options] - `open` renders it expanded
 * @returns {string} HTML string (empty if no diagnostics)
 */
export function renderDiagnosticsBar(diagnostics, { open = false } = {}) {
  if (!diagnostics || diagnostics.length === 0) return '';

  const maxLevel = diagnosticsLevel(diagnostics);

  const icon =
    maxLevel === 'info'
      ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><path d="M12 9v4M12 17h.01"/></svg>';

  const count = diagnostics.length;
  const label = count === 1 ? '1 issue detected' : `${count} issues detected`;

  const items = diagnostics
    .map((d) => {
      const msgLevel = escapeHtml(d.level);
      return (
        `<li class="diagnostics-msg diagnostics-msg-${msgLevel}">` +
        `<span class="diagnostics-source">${escapeHtml(d.source)}</span> ` +
        `${escapeHtml(d.message)}</li>`
      );
    })
    .join('');

  return `
    <details class="diagnostics-bar diagnostics-${escapeHtml(maxLevel)} animate-fade-in"${open ? ' open' : ''}>
      <summary class="diagnostics-toggle">
        ${icon}
        <span>${label}</span>
      </summary>
      <ul class="diagnostics-messages">${items}</ul>
    </details>
  `;
}

/**
 * Render the query-expansion strip for the results header: one pair per
 * applied expansion, e.g. `ts -> troubleshoot, timing system`. An entry whose
 * term expanded to several canonicals is ONE pair with the canonicals joined by
 * commas, never one pair per canonical, so the strip mirrors the backend's
 * `expanded_terms` groups exactly.
 * @param {ExpandedTerm[]} [expandedTerms] - Applied expansions from the search response
 * @returns {string} HTML string (empty when nothing was expanded)
 */
export function renderExpansionBanner(expandedTerms) {
  const groups = (expandedTerms ?? []).filter(
    (t) => t && t.original && (t.alternatives?.length ?? 0) > 0
  );
  if (groups.length === 0) return '';

  const pairs = groups
    .map((t) => {
      const alternatives = t.alternatives.map((a) => escapeHtml(a)).join(', ');
      return (
        '<span class="expansion-pair">' +
        `<code class="expansion-original">${escapeHtml(t.original)}</code>` +
        ` &rarr; ${alternatives}</span>`
      );
    })
    .join('');

  return `
    <div class="expansion-banner" data-testid="expansion-banner">
      <span class="expansion-label">Expanded</span>
      ${pairs}
    </div>
  `;
}

/**
 * Render the degraded-configuration banner shown above the search form.
 *
 * Blocking (`configuration_invalid`) renders `#config-invalid-banner` and the
 * caller disables the search controls; non-blocking (`configuration_warning`)
 * renders `#config-warning-banner` with the form left usable. Both carry
 * `data-testid="config-banner"`.
 * @param {string[]} [configErrors] - Configuration errors reported by /api/capabilities
 * @param {string|null} [remedy] - Operator-facing fix derived from the error class
 * @param {boolean} [blocking] - True for configuration_invalid
 * @returns {string} HTML string
 */
export function renderConfigBanner(configErrors, remedy, blocking = false) {
  const id = blocking ? 'config-invalid-banner' : 'config-warning-banner';
  const variant = blocking ? 'config-banner-invalid' : 'config-banner-warning';
  const title = blocking
    ? 'Search unavailable \u2014 configuration invalid'
    : 'Configuration warning \u2014 search still available';
  const items = (configErrors ?? [])
    .map((e) => `<li class="config-banner-error">${escapeHtml(e)}</li>`)
    .join('');
  const remedyHtml = remedy ? `<p class="config-banner-remedy">${escapeHtml(remedy)}</p>` : '';

  return `
    <div id="${id}" class="config-banner ${variant}" role="alert" data-testid="config-banner">
      <p class="config-banner-title">${title}</p>
      <ul class="config-banner-errors">${items}</ul>
      ${remedyHtml}
    </div>
  `;
}

// Re-exported from the shared design-system dom module, so the panes import it
// with the rest of their rendering helpers.
export { escapeHtml };
