// @ts-check
/**
 * OSPREY Tuning — Navbar (Environment selector, status, run selector)
 * @module navbar
 */

import { api } from './api.js';
import { state } from './state.js';

/** @type {HTMLSelectElement} */
let envSelect;
/** @type {HTMLSelectElement} */
let runSelect;
/** @type {HTMLElement | null} */
let statusBadge;
/** @type {Element | null | undefined} */
let statusDot;
/** @type {Element | null | undefined} */
let statusText;
/** @type {HTMLButtonElement} */
let newRunBtn;
/** @type {ReturnType<typeof setInterval> | null} */
let statusPollTimer = null;

/**
 * Wire up the navbar controls and initial data loads.
 * @returns {void}
 */
export function initNavbar() {
  envSelect = /** @type {HTMLSelectElement} */ (document.getElementById('env-select'));
  runSelect = /** @type {HTMLSelectElement} */ (document.getElementById('run-select'));
  statusBadge = document.getElementById('env-status-badge');
  statusDot = statusBadge?.querySelector('.status-dot');
  statusText = statusBadge?.querySelector('.status-text');
  newRunBtn = /** @type {HTMLButtonElement} */ (document.getElementById('new-run-btn'));

  loadEnvironments();
  loadRuns();

  envSelect.addEventListener('change', onEnvironmentChange);
  runSelect.addEventListener('change', onRunChange);
  newRunBtn.addEventListener('click', onNewRun);

  // Update controls when page state changes
  state.on('pageStateChanged', (ps) => {
    envSelect.disabled = ps.envSelectorDisabled;
    newRunBtn.disabled = ps.envSelectorDisabled;
  });
}

/**
 * Fetch environments and populate the environment `<select>`.
 * @returns {Promise<void>}
 */
async function loadEnvironments() {
  try {
    const result = await api.listEnvironments();
    const envs = result.environments || result || [];

    envSelect.innerHTML = '<option value="">Select environment...</option>';
    for (const env of envs) {
      const name = typeof env === 'string' ? env : env.name;
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      envSelect.appendChild(opt);
    }
    envSelect.disabled = false;
  } catch (err) {
    console.warn('Failed to load environments:', err);
    envSelect.innerHTML = '<option value="">No environments available</option>';
  }
}

/**
 * Fetch available runs and populate the run `<select>`.
 * @returns {Promise<void>}
 */
async function loadRuns() {
  try {
    const result = await api.getAvailableRuns();
    const runs = result.runs || result || [];

    runSelect.innerHTML = '<option value="">Select run...</option>';
    for (const run of runs) {
      const ts = typeof run === 'string' ? run : run.timestamp;
      const opt = document.createElement('option');
      opt.value = ts;
      opt.textContent = ts;
      runSelect.appendChild(opt);
    }
    runSelect.disabled = runs.length === 0;
  } catch {
    runSelect.innerHTML = '<option value="">No runs</option>';
  }
}

/**
 * Handle environment selection: load details, toggle testing indicator, poll status.
 * @returns {Promise<void>}
 */
async function onEnvironmentChange() {
  const name = envSelect.value;
  if (!name) {
    state.setEnvironment(null);
    stopStatusPolling();
    return;
  }

  try {
    const details = await api.getEnvironmentDetails(name);
    state.setEnvironment(name, details);

    // Show testing mode if applicable
    const testingIndicator = document.getElementById('testing-indicator');
    if (testingIndicator) {
      testingIndicator.style.display = details?.testing_mode ? '' : 'none';
    }

    checkStatus(name);
    startStatusPolling(name);
  } catch (err) {
    console.error('Failed to load environment:', err);
    state.setEnvironment(name);
  }
}

/**
 * Handle run selection: emit a historical-run load and activate the analysis tab.
 * @returns {void}
 */
function onRunChange() {
  const ts = runSelect.value;
  if (!ts) return;

  // Switch to Analysis tab and load run
  state.emit('loadHistoricalRun', ts);

  // Activate analysis tab
  const analysisTab = /** @type {HTMLElement | null} */ (
    document.querySelector('[data-tab="analysis-tab"]')
  );
  if (analysisTab) analysisTab.click();
}

/**
 * Reset optimization state for a fresh run and clear the run selector.
 * @returns {void}
 */
function onNewRun() {
  state.resetForNewRun();
  runSelect.value = '';
}

/**
 * Query connection status for an environment and update the badge.
 * @param {string} envName
 * @returns {Promise<void>}
 */
async function checkStatus(envName) {
  try {
    const result = await api.checkEnvironmentStatus(envName);
    const status = result?.status || result?.connected ? 'connected' : 'offline';
    updateStatusDisplay(status);
  } catch {
    updateStatusDisplay('offline');
  }
}

/**
 * Reflect a connection status in the status dot and text.
 * @param {string} status
 * @returns {void}
 */
function updateStatusDisplay(status) {
  if (!statusDot || !statusText) return;

  statusDot.className = 'status-dot';
  if (status === 'connected' || status === 'online') {
    statusDot.classList.add('live');
    statusText.textContent = 'Connected';
  } else {
    statusDot.classList.add('error');
    statusText.textContent = 'Offline';
  }
}

/**
 * Begin polling environment status on a fixed interval.
 * @param {string} envName
 * @returns {void}
 */
function startStatusPolling(envName) {
  stopStatusPolling();
  statusPollTimer = setInterval(() => checkStatus(envName), 60000);
}

/**
 * Stop any active status polling timer.
 * @returns {void}
 */
function stopStatusPolling() {
  if (statusPollTimer) {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
  }
}
