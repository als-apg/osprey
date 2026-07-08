// @ts-check
/**
 * OSPREY Tuning — Progress Display
 *
 * Status badge, Plotly graph, results table, logs, polling.
 * @module tuning/progress-display
 */

import { api } from './api.js';
import { state } from './state.js';
import { createOptimizationPlot } from './plots.js';
import { escapeHtml } from '/design-system/js/dom.js';

/** @typedef {import('./state.js').OptimizationState} OptimizationState */

/**
 * One row of the results table, built by merging a raw LHS/BO data point with
 * its display phase and iteration index.
 * @typedef {object} ResultRow
 * @property {string} phase
 * @property {number} iter
 * @property {number} [objective_value]
 * @property {*} [objective]
 */

/** Extract a message from an unknown catch binding.
 *  @param {unknown} e  @returns {string} */
function messageOf(e) { return e instanceof Error ? e.message : String(e); }

/** @type {ReturnType<typeof setInterval> | null} */
let pollTimer = null;
/** @type {HTMLElement | null} */
let plotContainer = null;
/** @type {HTMLTextAreaElement | null} */
let logOutput = null;
/** @type {HTMLElement | null} */
let resultsTableBody = null;
/** @type {HTMLElement | null} */
let resultsPagination = null;
let currentPage = 0;
const PAGE_SIZE = 10;

/** @returns {void} */
export function initProgressDisplay() {
  plotContainer = document.getElementById('optimization-plot');
  logOutput = /** @type {HTMLTextAreaElement | null} */ (document.getElementById('log-output'));
  resultsTableBody = document.getElementById('results-table-body');
  resultsPagination = document.getElementById('results-pagination');

  // Display mode selector
  const displayMode = document.getElementById('display-mode');
  if (displayMode) {
    displayMode.addEventListener('change', (e) => {
      const target = /** @type {HTMLSelectElement} */ (e.target);
      state.setDisplayMode(target.value);
    });
  }

  // Apply selected point
  const applyBtn = document.getElementById('apply-point-btn');
  if (applyBtn) {
    applyBtn.addEventListener('click', applySelectedPoint);
  }

  // Subscribe to state
  state.on('optimizationStateChanged', onStateChanged);
  state.on('pageStateChanged', onPageStateChanged);
  state.on('displayModeChanged', onDisplayModeChanged);
  state.on('selectedPointChanged', onSelectedPointChanged);
}

/** @param {OptimizationState} optState  @returns {void} */
function onStateChanged(optState) {
  updateStatusBadge(optState.status);
  updatePlot(optState);
  updateResultsTable(optState);
  updateLogs(optState);
}

/** @param {any} ps  @returns {void} */
function onPageStateChanged(ps) {
  if (ps.pollingEnabled && !pollTimer) {
    startPolling();
  } else if (!ps.pollingEnabled && pollTimer) {
    stopPolling();
  }
}

/** @returns {void} */
function onDisplayModeChanged() {
  updatePlot(state.optimizationState);
}

/** @param {any} point  @returns {void} */
function onSelectedPointChanged(point) {
  const el = document.getElementById('selected-point');
  const textEl = document.getElementById('selected-point-text');
  if (!el || !textEl) return;

  if (point) {
    el.style.display = '';
    const vars = point.variables
      ? Object.entries(point.variables).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(4) : v}`).join(', ')
      : '';
    textEl.textContent = `Iteration ${point.idx}: Objective=${typeof point.objective_value === 'number' ? point.objective_value.toFixed(6) : point.objective_value}${vars ? ' | ' + vars : ''}`;
  } else {
    el.style.display = 'none';
  }
}

// ---- Status Badge ----

/** @param {string} status  @returns {void} */
function updateStatusBadge(status) {
  const badge = document.getElementById('run-status-badge');
  if (!badge) return;

  const dot = /** @type {HTMLElement} */ (badge.querySelector('.status-dot'));
  const text = /** @type {HTMLElement} */ (badge.querySelector('.status-text'));

  dot.className = 'status-dot';
  switch (status) {
    case 'RUNNING':
      dot.classList.add('running');
      text.textContent = 'Running';
      break;
    case 'PAUSED':
      dot.classList.add('paused');
      text.textContent = 'Paused';
      break;
    case 'COMPLETED':
      dot.classList.add('live');
      text.textContent = 'Completed';
      break;
    case 'CANCELLED':
      dot.classList.add('warning');
      text.textContent = 'Cancelled';
      break;
    case 'ERROR':
      dot.classList.add('error');
      text.textContent = 'Error';
      break;
    default:
      text.textContent = 'Idle';
  }
}

// ---- Plot ----

/** @param {OptimizationState} optState  @returns {void} */
function updatePlot(optState) {
  if (!plotContainer) return;

  const hasData = (optState.lhs_data?.length > 0) ||
                  (optState.bo_data?.length > 0) ||
                  (optState.snapshots?.length > 0);

  if (!hasData) {
    if (!plotContainer.querySelector('.plot-empty')) {
      plotContainer.innerHTML = '<div class="plot-empty">Start an optimization to see results</div>';
    }
    return;
  }

  createOptimizationPlot(plotContainer, optState, state.displayMode);

  // Click handler for point selection
  const plotly = /** @type {any} */ (plotContainer);
  plotly.removeAllListeners?.('plotly_click');
  plotly.on?.('plotly_click', (/** @type {any} */ data) => {
    if (!data?.points?.[0]) return;
    const pt = data.points[0];
    const allData = [
      ...(optState.snapshots || []).map((s, i) => ({ ...s, idx: i })),
      ...(optState.lhs_data || []).map((d, i) => ({ ...d, idx: (optState.snapshots?.length || 0) + i })),
      ...(optState.bo_data || []).map((d, i) => ({ ...d, idx: (optState.snapshots?.length || 0) + (optState.lhs_data?.length || 0) + i })),
    ];
    const point = allData[pt.pointIndex] || null;
    state.setSelectedPoint(point);
  });
}

// ---- Results Table ----

/** @param {OptimizationState} optState  @returns {void} */
function updateResultsTable(optState) {
  if (!resultsTableBody) return;

  /** @type {ResultRow[]} */
  const allData = [
    ...(optState.lhs_data || []).map((d, i) => ({ ...d, phase: 'LHS', iter: i + 1 })),
    ...(optState.bo_data || []).map((d, i) => ({ ...d, phase: 'BO', iter: (optState.lhs_data?.length || 0) + i + 1 })),
  ];

  if (allData.length === 0) {
    resultsTableBody.innerHTML = '<tr class="empty-row"><td colspan="3">No results yet</td></tr>';
    if (resultsPagination) resultsPagination.innerHTML = '';
    return;
  }

  const totalPages = Math.ceil(allData.length / PAGE_SIZE);
  currentPage = Math.min(currentPage, totalPages - 1);
  const start = currentPage * PAGE_SIZE;
  const pageData = allData.slice(start, start + PAGE_SIZE);

  resultsTableBody.innerHTML = '';
  for (const d of pageData) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${d.iter}</td>
      <td class="muted">${typeof d.objective_value === 'number' ? d.objective_value.toFixed(6) : escapeHtml(d.objective ?? '--')}</td>
      <td>${escapeHtml(d.phase)}</td>
    `;
    resultsTableBody.appendChild(tr);
  }

  // Pagination
  if (resultsPagination && totalPages > 1) {
    resultsPagination.innerHTML = '';

    const prevBtn = document.createElement('button');
    prevBtn.textContent = 'Prev';
    prevBtn.disabled = currentPage === 0;
    prevBtn.addEventListener('click', () => { currentPage--; updateResultsTable(optState); });
    resultsPagination.appendChild(prevBtn);

    for (let i = 0; i < totalPages; i++) {
      const btn = document.createElement('button');
      btn.textContent = String(i + 1);
      btn.className = i === currentPage ? 'active' : '';
      btn.addEventListener('click', () => { currentPage = i; updateResultsTable(optState); });
      resultsPagination.appendChild(btn);
    }

    const nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next';
    nextBtn.disabled = currentPage >= totalPages - 1;
    nextBtn.addEventListener('click', () => { currentPage++; updateResultsTable(optState); });
    resultsPagination.appendChild(nextBtn);
  } else if (resultsPagination) {
    resultsPagination.innerHTML = '';
  }
}

// ---- Logs ----

/** @param {OptimizationState} optState  @returns {void} */
function updateLogs(optState) {
  if (!logOutput) return;
  const logs = optState.logs || [];
  const text = logs.join('\n');
  if (logOutput.value !== text) {
    logOutput.value = text;
    logOutput.scrollTop = logOutput.scrollHeight;
  }
}

// ---- Apply Selected Point ----

/** @returns {Promise<void>} */
async function applySelectedPoint() {
  const point = /** @type {any} */ (state.selectedPoint);
  if (!point?.variables || !state.environment) return;

  try {
    await api.setMachineVariables(state.environment, point.variables);
    // Flash success
    const btn = /** @type {HTMLButtonElement | null} */ (document.getElementById('apply-point-btn'));
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = 'Applied!';
      btn.disabled = true;
      setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 2000);
    }
  } catch (err) {
    const alert = document.getElementById('validation-alert');
    const text = document.getElementById('validation-text');
    if (alert && text) {
      text.textContent = `Failed to apply: ${messageOf(err)}`;
      alert.style.display = '';
    }
  }
}

// ---- Polling ----

/** @returns {void} */
function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(pollState, 500);
}

/** @returns {void} */
function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

/** @returns {Promise<void>} */
async function pollState() {
  if (!state.jobId) return;

  try {
    const result = await api.getState(state.jobId);
    if (result) {
      state.setOptimizationState(result);
    }
  } catch (err) {
    console.warn('Poll error:', err);
  }
}
