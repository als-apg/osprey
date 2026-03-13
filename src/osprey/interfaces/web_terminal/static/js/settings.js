/* OSPREY Web Terminal — Agent Settings Panel */

import { fetchJSON } from './api.js';
import { closeDrawer } from './drawer.js';
import { restartTerminal } from './terminal.js';

let currentConfig = null;  // { sections, raw, path }
let isDirty = false;
let currentMode = 'form';  // 'form' | 'raw'

// The agent tab panel — all DOM queries are scoped to this element
let agentPanel = null;

// Known enum values for select dropdowns
const ENUM_FIELDS = {
  'control_system.write_verification': ['none', 'callback', 'readback'],
  'approval.default_policy': ['always', 'selective', 'skip'],
  'approval.tools.channel_write': ['always', 'selective', 'skip'],
  'approval.tools.channel_read': ['always', 'selective', 'skip'],
  'approval.tools.archiver_read': ['always', 'selective', 'skip'],
  'approval.tools.execute': ['always', 'selective', 'skip'],
  'approval.tools.setup_patch': ['always', 'selective', 'skip'],
  'approval.tools.entry_create': ['always', 'selective', 'skip'],
};

// Fields that should render as toggles (boolean)
const BOOLEAN_FIELDS = new Set([
  'approval.enabled',
  'control_system.writes_enabled',
  'control_system.read_only',
  'python_execution.enabled',
  'ariel.enabled',
  'artifact_server.auto_launch',
  'screen_capture.enabled',
]);

/**
 * Initialize the settings panel. Call once on DOMContentLoaded.
 */
export function initSettings() {
  const drawer = document.getElementById('settings-drawer');
  agentPanel = document.getElementById('tab-config');
  if (!drawer || !agentPanel) return;

  // Load config when agent tab becomes active (covers both drawer open and tab switch)
  agentPanel.addEventListener('drawer:tab-activate', () => loadConfig());

  // Mode toggle buttons (scoped to agent panel)
  agentPanel.querySelectorAll('.settings-mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => switchMode(btn.dataset.mode));
  });

  // Apply button
  const applyBtn = agentPanel.querySelector('.settings-apply-btn');
  if (applyBtn) applyBtn.addEventListener('click', showConfirmDialog);

  // Confirm dialog buttons
  const confirmBtn = agentPanel.querySelector('.settings-confirm-btn');
  const cancelBtn = agentPanel.querySelector('.settings-cancel-btn');
  if (confirmBtn) confirmBtn.addEventListener('click', applySettings);
  if (cancelBtn) cancelBtn.addEventListener('click', hideConfirmDialog);

  // Raw editor dirty tracking
  const rawEditor = document.getElementById('settings-raw-editor');
  if (rawEditor) rawEditor.addEventListener('input', markDirty);
}

async function loadConfig() {
  const formContainer = document.getElementById('settings-form');
  const rawTextarea = document.getElementById('settings-raw-editor');
  const loading = document.getElementById('settings-loading');
  const error = document.getElementById('settings-error');

  if (loading) loading.style.display = 'flex';
  if (error) error.style.display = 'none';
  if (formContainer) formContainer.innerHTML = '';

  try {
    currentConfig = await fetchJSON('/api/config');
    if (loading) loading.style.display = 'none';

    // Populate form view
    renderFormSections(currentConfig.sections);

    // Populate raw view
    if (rawTextarea) rawTextarea.value = currentConfig.raw;

    isDirty = false;
    updateSaveBar();
  } catch (e) {
    if (loading) loading.style.display = 'none';
    if (error) {
      error.style.display = 'flex';
      error.textContent = `Failed to load config: ${e.message}`;
    }
  }
}

function switchMode(mode) {
  currentMode = mode;
  if (!agentPanel) return;

  agentPanel.querySelectorAll('.settings-mode-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  const formView = document.getElementById('settings-form-view');
  const rawView = document.getElementById('settings-raw-view');

  if (formView) formView.style.display = mode === 'form' ? '' : 'none';
  if (rawView) rawView.classList.toggle('active', mode === 'raw');

  // When switching to raw, reload the raw content from disk so users see the
  // current file (including comments).  Do NOT rebuild from the form — that
  // would strip comments.  If the form is dirty the user is warned by the
  // unsaved-changes guard on tab/drawer close.
}

function renderFormSections(sections) {
  const container = document.getElementById('settings-form');
  if (!container) return;
  container.innerHTML = '';

  for (const [sectionKey, sectionValue] of Object.entries(sections)) {
    if (sectionValue == null || typeof sectionValue !== 'object') continue;

    const section = document.createElement('div');
    section.className = 'settings-section';

    const header = document.createElement('div');
    header.className = 'settings-section-header';
    header.innerHTML = `
      <span class="settings-section-chevron">\u25B6</span>
      <span class="settings-section-label">${formatLabel(sectionKey)}</span>
    `;
    header.addEventListener('click', () => {
      section.classList.toggle('expanded');
    });

    const body = document.createElement('div');
    body.className = 'settings-section-body';

    renderFields(body, sectionValue, sectionKey);

    section.appendChild(header);
    section.appendChild(body);
    container.appendChild(section);
  }
}

function renderFields(container, obj, prefix, depth = 0) {
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = `${prefix}.${key}`;

    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      const group = document.createElement('div');
      group.className = 'settings-subgroup';
      group.style.setProperty('--depth', depth);

      const groupLabel = document.createElement('div');
      groupLabel.className = 'settings-subgroup-label';
      groupLabel.textContent = formatLabel(key);
      groupLabel.title = fullKey;
      group.appendChild(groupLabel);

      renderFields(group, value, fullKey, depth + 1);

      container.appendChild(group);
      continue;
    }

    const field = document.createElement('div');
    field.className = 'settings-field';

    const label = document.createElement('span');
    label.className = 'settings-field-label';
    label.textContent = key;
    label.title = fullKey;

    const inputWrap = document.createElement('div');
    inputWrap.className = 'settings-field-input';

    const input = createInputForValue(fullKey, value);
    inputWrap.appendChild(input);

    field.appendChild(label);
    field.appendChild(inputWrap);
    container.appendChild(field);
  }
}

function createInputForValue(fullKey, value) {
  if (ENUM_FIELDS[fullKey]) {
    const select = document.createElement('select');
    select.className = 'settings-select';
    select.dataset.key = fullKey;
    for (const opt of ENUM_FIELDS[fullKey]) {
      const option = document.createElement('option');
      option.value = opt;
      option.textContent = opt;
      if (opt === value) option.selected = true;
      select.appendChild(option);
    }
    select.addEventListener('change', markDirty);
    return select;
  }

  if (typeof value === 'boolean' || BOOLEAN_FIELDS.has(fullKey)) {
    const toggle = document.createElement('label');
    toggle.className = 'toggle-switch';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = !!value;
    checkbox.dataset.key = fullKey;
    checkbox.addEventListener('change', markDirty);
    const slider = document.createElement('span');
    slider.className = 'toggle-slider';
    toggle.appendChild(checkbox);
    toggle.appendChild(slider);
    return toggle;
  }

  if (typeof value === 'number') {
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'settings-input';
    input.value = value;
    input.dataset.key = fullKey;
    input.addEventListener('input', markDirty);
    return input;
  }

  if (Array.isArray(value)) {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'settings-input';
    input.value = value.join(', ');
    input.dataset.key = fullKey;
    input.dataset.type = 'array';
    input.addEventListener('input', markDirty);
    return input;
  }

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'settings-input';
  input.value = value ?? '';
  input.dataset.key = fullKey;
  input.addEventListener('input', markDirty);
  return input;
}

function markDirty() {
  isDirty = true;
  updateSaveBar();
}

function updateSaveBar() {
  const bar = agentPanel ? agentPanel.querySelector('.settings-save-bar') : null;
  if (bar) bar.classList.toggle('visible', isDirty);
}

/**
 * Collect all form field values that differ from the original config into a
 * dot-keyed updates object suitable for PATCH /api/config.
 */
function collectFormUpdates() {
  const updates = {};
  const inputs = document.querySelectorAll('#settings-form [data-key]');

  for (const input of inputs) {
    const key = input.dataset.key;
    let newValue;

    if (input.type === 'checkbox') {
      newValue = input.checked;
    } else if (input.type === 'number') {
      newValue = Number(input.value);
    } else if (input.dataset.type === 'array') {
      newValue = input.value.split(',').map((s) => s.trim()).filter(Boolean);
    } else {
      newValue = input.value;
    }

    // Compare against original to only send changed fields
    const originalValue = getNestedValue(currentConfig.sections, key);
    if (!deepEqual(originalValue, newValue)) {
      updates[key] = newValue;
    }
  }

  return updates;
}

function getNestedValue(obj, dottedKey) {
  const parts = dottedKey.split('.');
  let node = obj;
  for (const part of parts) {
    if (node == null || typeof node !== 'object') return undefined;
    node = node[part];
  }
  return node;
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }
  if (typeof a === 'object') {
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    if (keysA.length !== keysB.length) return false;
    return keysA.every((k) => deepEqual(a[k], b[k]));
  }
  return false;
}

function showConfirmDialog() {
  if (!agentPanel) return;
  const overlay = agentPanel.querySelector('.settings-confirm-overlay');
  if (overlay) overlay.classList.add('visible');
}

function hideConfirmDialog() {
  if (!agentPanel) return;
  const overlay = agentPanel.querySelector('.settings-confirm-overlay');
  if (overlay) overlay.classList.remove('visible');
}

async function applySettings() {
  hideConfirmDialog();

  const status = agentPanel ? agentPanel.querySelector('.settings-status') : null;
  const applyBtn = agentPanel ? agentPanel.querySelector('.settings-apply-btn') : null;
  if (applyBtn) applyBtn.disabled = true;
  if (status) status.textContent = 'Saving...';

  let configSaved = false;
  try {
    if (currentMode === 'raw') {
      // Raw mode: send the full YAML text as-is (user is responsible for content)
      const textarea = document.getElementById('settings-raw-editor');
      const yamlContent = textarea ? textarea.value : '';
      const saveResp = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw: yamlContent }),
      });
      if (!saveResp.ok) {
        const detail = await saveResp.json().catch(() => ({}));
        throw new Error(detail.detail || `Save failed (HTTP ${saveResp.status})`);
      }
    } else {
      // Form mode: send only changed fields via PATCH — server uses ruamel.yaml
      // to apply them without stripping comments or reordering the file.
      const updates = collectFormUpdates();
      if (Object.keys(updates).length === 0) {
        if (status) status.textContent = 'No changes to apply';
        if (applyBtn) applyBtn.disabled = false;
        return;
      }
      const patchResp = await fetch('/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ updates }),
      });
      if (!patchResp.ok) {
        const detail = await patchResp.json().catch(() => ({}));
        throw new Error(detail.detail || `Patch failed (HTTP ${patchResp.status})`);
      }
    }
    configSaved = true;

    isDirty = false;
    updateSaveBar();
    if (status) status.textContent = '';
    if (applyBtn) applyBtn.disabled = false;

    closeDrawer();
    await restartTerminal();
  } catch (e) {
    const prefix = configSaved ? 'Config saved, but: ' : '';
    if (status) status.textContent = `${prefix}${e.message}`;
    if (applyBtn) applyBtn.disabled = false;
  }
}

function formatLabel(key) {
  return key.replace(/_/g, ' ');
}
