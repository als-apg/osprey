// @ts-check
/**
 * OSPREY Channel Finder — Hierarchical Explore (Miller Columns)
 *
 * Progressive drill-down through hierarchy levels with multi-select
 * support on instance levels and "Build Channels" action.
 * Read-only: corrections go into data/facility/fixes.yaml and take effect with
 * `osprey build`.
 */

import { fetchJSON } from './api.js';
import { showToast } from './app.js';
import { esc, messageOf } from './utils.js';
import { computeSelection } from './explore-selection.js';

/**
 * @typedef {object} Column
 * @property {string} level
 * @property {any[]} options
 * @property {Set<string>} selectedValues
 */

/** @type {any} */
let hierInfo = null;
/** @type {Record<string, string|string[]>} */
let selections = {};  // level -> value(s)
/** @type {Column[]} */
let columns = [];     // array of { level, options, selectedValues }
let showDescriptions = false;

/**
 * Toggle description visibility and re-render (called from explore.js).
 * @param {boolean} val
 */
export function setShowDescriptions(val) {
  showDescriptions = val;
  renderColumns();
}

/**
 * @param {HTMLElement} container
 */
export async function mountHierarchical(container) {
  container.innerHTML = `
    <div class="cf-corrections-info" style="color: var(--text-muted); font-size: var(--cf-text-sm); margin-bottom: var(--cf-space-2);">
      Corrections go in <code>data/facility/fixes.yaml</code> and take effect with <code>osprey build</code>.
    </div>
    <div class="miller-container" id="miller-container">
      <div class="loading-center"><div class="loading-spinner"></div> Loading hierarchy...</div>
    </div>
  `;

  try {
    hierInfo = await fetchJSON('/api/explore/hierarchy-info');
    selections = {};
    columns = [];
    await loadLevel(0);
  } catch (e) {
    container.innerHTML = `<div class="empty-state">Failed to load hierarchy: ${esc(messageOf(e))}</div>`;
  }

}

export function unmountHierarchical() {
  hierInfo = null;
  selections = {};
  columns = [];
}

/**
 * @param {number} levelIdx
 */
async function loadLevel(levelIdx) {
  if (!hierInfo?.hierarchy_levels) return;
  const levels = hierInfo.hierarchy_levels;
  if (levelIdx >= levels.length) return;

  const level = typeof levels[levelIdx] === 'string'
    ? levels[levelIdx]
    : (levels[levelIdx].name || levels[levelIdx].level);

  // Build selections dict for API call
  /** @type {Record<string, any>} */
  const apiSelections = {};
  for (let i = 0; i < levelIdx; i++) {
    const prevLevel = typeof levels[i] === 'string' ? levels[i] : (levels[i].name || levels[i].level);
    if (selections[prevLevel] !== undefined) {
      apiSelections[prevLevel] = selections[prevLevel];
    }
  }

  try {
    const selParam = Object.keys(apiSelections).length > 0
      ? `&selections=${encodeURIComponent(JSON.stringify(apiSelections))}`
      : '';
    const data = await fetchJSON(`/api/explore/options?level=${encodeURIComponent(level)}${selParam}`);

    // Trim columns beyond current level
    columns = columns.slice(0, levelIdx);

    columns.push({
      level,
      options: data.options || [],
      selectedValues: new Set(),
    });

    renderColumns();
  } catch (e) {
    showToast(`Failed to load ${level}: ${messageOf(e)}`, 'error');
  }
}

function renderColumns() {
  const mc = document.getElementById('miller-container');
  if (!mc) return;

  mc.innerHTML = columns.map((col, colIdx) => {
    const items = (col.options || []).map((/** @type {any} */ opt) => {
      const name = typeof opt === 'string' ? opt : (opt.name || opt.label || opt.value || '');
      const count = (typeof opt === 'object' && opt.count !== null && opt.count !== undefined) ? opt.count : null;
      const desc = (typeof opt === 'object' && opt.description) ? opt.description : '';
      const isSelected = col.selectedValues.has(name);

      const descHtml = desc
        ? (showDescriptions
            ? `<div class="item-desc item-desc-full">${esc(desc)}</div>`
            : `<div class="item-desc">${esc(desc)}</div>`)
        : '';

      return `
        <div class="miller-item${isSelected ? ' selected' : ''}"
             data-col="${colIdx}" data-value="${esc(name)}">
          <div class="item-name-group">
            <span class="item-label">${esc(name)}</span>
            ${descHtml}
          </div>
          <span>${count !== null && count !== undefined ? `<span class="item-count">${esc(count)}</span>` : ''}</span>
        </div>
      `;
    }).join('');

    return `
      <div class="miller-column">
        <div class="miller-column-header">${esc(col.level)}</div>
        <div class="miller-column-body">${items || '<div class="empty-state">No options</div>'}</div>
      </div>
    `;
  }).join('');

  // Attach click handlers for item selection
  mc.querySelectorAll('.miller-item').forEach(item => {
    item.addEventListener('click', () => {
      const el = /** @type {HTMLElement} */ (item);
      const colIdx = parseInt(el.dataset.col ?? '', 10);
      const value = el.dataset.value;
      if (value === undefined) return;
      handleSelect(colIdx, value);
    });
  });
}

// ---- Selection ----

/**
 * @param {number} colIdx
 * @param {string} value
 */
function handleSelect(colIdx, value) {
  const col = columns[colIdx];
  if (!col) return;

  const isLastLevel = colIdx === columns.length - 1 &&
    hierInfo?.hierarchy_levels?.length === columns.length;

  const { selectedValues, selectionValue, loadNext } =
    computeSelection([...col.selectedValues], value, isLastLevel);

  // Apply the computed selection to column + module state
  col.selectedValues = new Set(selectedValues);
  if (selectionValue === null) {
    delete selections[col.level];
  } else {
    selections[col.level] = selectionValue;
  }

  // Remove deeper selections
  for (let i = colIdx + 1; i < columns.length; i++) {
    delete selections[columns[i].level];
  }

  // Re-render current columns
  renderColumns();

  // Load next level (only for single selection on non-terminal)
  if (loadNext) {
    loadLevel(colIdx + 1);
  }
}
