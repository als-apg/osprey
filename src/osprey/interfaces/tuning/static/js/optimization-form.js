/**
 * OSPREY Tuning — Optimization Form
 *
 * Accordion phases, variable tables, add-variable forms, control buttons.
 * LHS and BO tables stay synced.
 */

import { api } from './api.js';
import { state } from './state.js';

// ---- DOM refs ----

const els = {};

export function initOptimizationForm() {
  // LHS elements
  els.lhsObjective = document.getElementById('lhs-objective');
  els.lhsSamples = document.getElementById('lhs-samples');
  els.lhsTableBody = document.getElementById('lhs-table-body');
  els.lhsAddPv = document.getElementById('lhs-add-pv');
  els.lhsCheckValue = document.getElementById('lhs-check-value');
  els.lhsAddBtn = document.getElementById('lhs-add-btn');
  els.lhsAddResult = document.getElementById('lhs-add-result');
  els.lhsSelectAll = document.getElementById('lhs-select-all');
  els.lhsDeselectAll = document.getElementById('lhs-deselect-all');
  els.lhsCheckAll = document.getElementById('lhs-check-all');
  els.lhsResetBtn = document.getElementById('lhs-reset-btn');
  els.lhsUpdateBtn = document.getElementById('lhs-update-btn');

  // BO elements
  els.boObjective = document.getElementById('bo-objective');
  els.boAlgorithm = document.getElementById('bo-algorithm');
  els.boTopPoints = document.getElementById('bo-top-points');
  els.boIterations = document.getElementById('bo-iterations');
  els.boTableBody = document.getElementById('bo-table-body');
  els.boAddPv = document.getElementById('bo-add-pv');
  els.boCheckValue = document.getElementById('bo-check-value');
  els.boAddBtn = document.getElementById('bo-add-btn');
  els.boAddResult = document.getElementById('bo-add-result');
  els.boSelectAll = document.getElementById('bo-select-all');
  els.boDeselectAll = document.getElementById('bo-deselect-all');
  els.boCheckAll = document.getElementById('bo-check-all');
  els.boResetBtn = document.getElementById('bo-reset-btn');
  els.boUpdateBtn = document.getElementById('bo-update-btn');

  // Control buttons
  els.startBtn = document.getElementById('start-btn');
  els.pauseBtn = document.getElementById('pause-btn');
  els.resumeBtn = document.getElementById('resume-btn');
  els.cancelBtn = document.getElementById('cancel-btn');

  // Event listeners — add variable
  els.lhsCheckValue.addEventListener('click', () => checkValue('lhs'));
  els.lhsAddBtn.addEventListener('click', () => addVariable('lhs'));
  els.boCheckValue.addEventListener('click', () => checkValue('bo'));
  els.boAddBtn.addEventListener('click', () => addVariable('bo'));

  // Select all / deselect all
  els.lhsSelectAll.addEventListener('click', () => toggleAllVariables(true));
  els.lhsDeselectAll.addEventListener('click', () => toggleAllVariables(false));
  els.boSelectAll.addEventListener('click', () => toggleAllVariables(true));
  els.boDeselectAll.addEventListener('click', () => toggleAllVariables(false));
  els.lhsCheckAll.addEventListener('change', (e) => toggleAllVariables(e.target.checked));
  els.boCheckAll.addEventListener('change', (e) => toggleAllVariables(e.target.checked));

  // Reset
  els.lhsResetBtn.addEventListener('click', resetVariables);
  els.boResetBtn.addEventListener('click', resetVariables);

  // Objective sync
  els.lhsObjective.addEventListener('change', () => {
    els.boObjective.value = els.lhsObjective.value;
  });
  els.boObjective.addEventListener('change', () => {
    els.lhsObjective.value = els.boObjective.value;
  });

  // Control buttons
  els.startBtn.addEventListener('click', startOptimization);
  els.pauseBtn.addEventListener('click', pauseOptimization);
  els.resumeBtn.addEventListener('click', resumeOptimization);
  els.cancelBtn.addEventListener('click', cancelOptimization);

  // State subscriptions
  state.on('variableTableChanged', renderTables);
  state.on('pageStateChanged', updateControlButtons);
  state.on('environmentChanged', onEnvironmentChanged);
  state.on('resetForNewRun', resetForm);

  // Populate objective dropdowns when environment loads variables
  state.on('environmentChanged', populateObjectiveDropdowns);
}

// ---- Objective Dropdowns ----

function populateObjectiveDropdowns({ details }) {
  if (!details?.variables) return;

  const vars = details.variables;
  for (const sel of [els.lhsObjective, els.boObjective]) {
    sel.innerHTML = '<option value="">Select variable...</option>';
    for (const v of vars) {
      const name = typeof v === 'string' ? v : v.name || v.pv_name;
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
  }
}

// ---- Variable Table Rendering ----

function renderTables() {
  renderTable(els.lhsTableBody, false);
  renderTable(els.boTableBody, true);
}

function renderTable(tbody, showBoRange) {
  const data = state.variableTableData;

  if (data.length === 0) {
    const cols = showBoRange ? 7 : 6;
    tbody.innerHTML = `<tr class="empty-row"><td colspan="${cols}">No variables added</td></tr>`;
    return;
  }

  tbody.innerHTML = '';
  for (const v of data) {
    const tr = document.createElement('tr');
    tr.className = v.selected ? 'selected' : '';

    tr.innerHTML = `
      <td class="col-check"><input type="checkbox" data-pv="${v.pv_name}" ${v.selected ? 'checked' : ''}></td>
      <td class="pv-name">${v.pv_name}</td>
      <td class="muted">${formatNum(v.current_value)}</td>
      <td><input class="input" type="number" data-pv="${v.pv_name}" data-field="min" value="${v.min ?? ''}" step="any"></td>
      <td><input class="input" type="number" data-pv="${v.pv_name}" data-field="max" value="${v.max ?? ''}" step="any"></td>
      <td><input class="input" type="number" data-pv="${v.pv_name}" data-field="step_size" value="${v.step_size ?? ''}" step="any"></td>
      ${showBoRange ? `<td><input class="input" type="number" data-pv="${v.pv_name}" data-field="bo_range_factor" value="${v.bo_range_factor ?? 1}" step="0.1"></td>` : ''}
    `;

    // Checkbox toggle
    tr.querySelector('input[type="checkbox"]').addEventListener('change', (e) => {
      state.updateVariable(v.pv_name, { selected: e.target.checked });
    });

    // Editable fields
    tr.querySelectorAll('input[data-field]').forEach(input => {
      input.addEventListener('change', (e) => {
        const field = e.target.dataset.field;
        const val = e.target.value === '' ? null : parseFloat(e.target.value);
        state.updateVariable(v.pv_name, { [field]: val });
      });
    });

    tbody.appendChild(tr);
  }
}

function formatNum(v) {
  if (v === null || v === undefined) return '--';
  return typeof v === 'number' ? v.toPrecision(6) : String(v);
}

// ---- Add Variable ----

let pendingValue = null;

async function checkValue(phase) {
  const pvInput = phase === 'lhs' ? els.lhsAddPv : els.boAddPv;
  const resultEl = phase === 'lhs' ? els.lhsAddResult : els.boAddResult;
  const addBtn = phase === 'lhs' ? els.lhsAddBtn : els.boAddBtn;
  const pvName = pvInput.value.trim();

  if (!pvName || !state.environment) return;

  resultEl.style.display = '';
  resultEl.textContent = 'Checking...';

  try {
    const result = await api.getVariableValue(state.environment, pvName);
    pendingValue = {
      pv_name: pvName,
      current_value: result.value ?? result.current_value ?? null,
      min: result.min ?? result.lower_limit ?? null,
      max: result.max ?? result.upper_limit ?? null,
      step_size: result.step_size ?? null,
      bo_range_factor: 1,
      selected: true,
    };
    resultEl.textContent = `Value: ${formatNum(pendingValue.current_value)} | Range: [${pendingValue.min ?? '?'}, ${pendingValue.max ?? '?'}]`;
    addBtn.disabled = false;
  } catch (err) {
    resultEl.textContent = `Error: ${err.message}`;
    pendingValue = null;
    addBtn.disabled = true;
  }
}

function addVariable(phase) {
  if (!pendingValue) return;

  // Prevent duplicates
  if (state.variableTableData.some(v => v.pv_name === pendingValue.pv_name)) {
    showValidation(`Variable ${pendingValue.pv_name} already exists`);
    return;
  }

  state.addVariable({ ...pendingValue });

  // Clear form
  const pvInput = phase === 'lhs' ? els.lhsAddPv : els.boAddPv;
  const resultEl = phase === 'lhs' ? els.lhsAddResult : els.boAddResult;
  const addBtn = phase === 'lhs' ? els.lhsAddBtn : els.boAddBtn;
  pvInput.value = '';
  resultEl.style.display = 'none';
  addBtn.disabled = true;
  pendingValue = null;

  // Sync the other phase's add form
  const otherPvInput = phase === 'lhs' ? els.boAddPv : els.lhsAddPv;
  otherPvInput.value = '';
}

// ---- Select All / Reset ----

function toggleAllVariables(selected) {
  const data = state.variableTableData.map(v => ({ ...v, selected }));
  state.setVariableTableData(data);
}

function resetVariables() {
  state.setVariableTableData([]);
}

function resetForm() {
  els.lhsObjective.value = '';
  els.boObjective.value = '';
  els.lhsSamples.value = '20';
  els.boTopPoints.value = '5';
  els.boIterations.value = '30';
  els.boAlgorithm.value = 'expected_improvement';
}

// ---- Control Buttons ----

function updateControlButtons(ps) {
  els.startBtn.disabled = !ps.canStart;
  els.pauseBtn.disabled = !ps.canPause;
  els.resumeBtn.disabled = !ps.canResume;
  els.cancelBtn.disabled = !ps.canCancel;

  els.pauseBtn.style.display = ps.showPause ? '' : 'none';
  els.resumeBtn.style.display = ps.showResume ? '' : 'none';
  els.cancelBtn.style.display = ps.showCancel ? '' : 'none';

  // Disable form inputs when running
  const formInputs = document.querySelectorAll('#optimization-form input, #optimization-form select');
  formInputs.forEach(el => {
    if (el.id === 'display-mode') return; // Display mode always enabled
    el.disabled = ps.formDisabled;
  });
}

function onEnvironmentChanged() {
  els.startBtn.disabled = !state.environment;
}

// ---- Start / Pause / Resume / Cancel ----

async function startOptimization() {
  if (!validateForm()) return;

  const selectedVars = state.variableTableData.filter(v => v.selected);
  const config = {
    environment: state.environment,
    objective: els.lhsObjective.value,
    lhs_samples: parseInt(els.lhsSamples.value),
    bo_algorithm: els.boAlgorithm.value,
    bo_top_points: parseInt(els.boTopPoints.value),
    bo_iterations: parseInt(els.boIterations.value),
    variables: selectedVars.map(v => ({
      name: v.pv_name,
      min: v.min,
      max: v.max,
      step_size: v.step_size,
      bo_range_factor: v.bo_range_factor,
    })),
  };

  try {
    els.startBtn.disabled = true;
    const result = await api.startOptimization(config);
    state.setJobId(result.job_id || result.id);
    state.setOptimizationState({ status: 'RUNNING', phase: 'LHS' });
  } catch (err) {
    showValidation(`Failed to start: ${err.message}`);
    els.startBtn.disabled = false;
  }
}

async function pauseOptimization() {
  if (!state.jobId) return;
  try {
    await api.pause(state.jobId);
    state.setOptimizationState({ status: 'PAUSED' });
  } catch (err) {
    showValidation(`Failed to pause: ${err.message}`);
  }
}

async function resumeOptimization() {
  if (!state.jobId) return;
  try {
    await api.resume(state.jobId);
    state.setOptimizationState({ status: 'RUNNING' });
  } catch (err) {
    showValidation(`Failed to resume: ${err.message}`);
  }
}

async function cancelOptimization() {
  if (!state.jobId) return;
  try {
    await api.cancel(state.jobId);
    state.setOptimizationState({ status: 'CANCELLED' });
  } catch (err) {
    showValidation(`Failed to cancel: ${err.message}`);
  }
}

// ---- Validation ----

function validateForm() {
  const objective = els.lhsObjective.value;
  if (!objective) {
    showValidation('Please select an objective variable.');
    return false;
  }

  const selectedVars = state.variableTableData.filter(v => v.selected);
  if (selectedVars.length === 0) {
    showValidation('Please add and select at least one variable.');
    return false;
  }

  for (const v of selectedVars) {
    if (v.min === null || v.max === null) {
      showValidation(`Variable ${v.pv_name} is missing min/max bounds.`);
      return false;
    }
    if (v.min >= v.max) {
      showValidation(`Variable ${v.pv_name}: min must be less than max.`);
      return false;
    }
  }

  return true;
}

function showValidation(msg) {
  const alert = document.getElementById('validation-alert');
  const text = document.getElementById('validation-text');
  if (alert && text) {
    text.textContent = msg;
    alert.style.display = '';
  }
}
