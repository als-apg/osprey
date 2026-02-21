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
  'approval.mode': ['disabled', 'selective', 'all_capabilities'],
  'control_system.write_verification': ['none', 'callback', 'readback'],
};

// Fields that should render as toggles (boolean)
const BOOLEAN_FIELDS = new Set([
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

  // Sync form → raw when switching to raw
  if (mode === 'raw' && currentConfig && isDirty) {
    const updated = buildConfigFromForm();
    const rawTextarea = document.getElementById('settings-raw-editor');
    if (rawTextarea) rawTextarea.value = updated;
  }
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
      // Nested object — render as a labeled sub-group
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
  // Check enum first
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

  // Boolean toggle
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

  // Number
  if (typeof value === 'number') {
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'settings-input';
    input.value = value;
    input.dataset.key = fullKey;
    input.addEventListener('input', markDirty);
    return input;
  }

  // Array — render as comma-separated text
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

  // Default: string input
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

function buildConfigFromForm() {
  if (!currentConfig) return '';

  // Deep clone the original full config
  const raw = currentConfig.raw;
  let fullConfig;
  try {
    fullConfig = jsyaml_parse(raw);
  } catch {
    return raw;
  }

  // Apply form values
  const inputs = document.querySelectorAll(
    '#settings-form [data-key]'
  );
  for (const input of inputs) {
    const key = input.dataset.key;
    const parts = key.split('.');
    let obj = fullConfig;
    for (let i = 0; i < parts.length - 1; i++) {
      if (obj[parts[i]] == null) obj[parts[i]] = {};
      obj = obj[parts[i]];
    }
    const lastKey = parts[parts.length - 1];

    if (input.type === 'checkbox') {
      obj[lastKey] = input.checked;
    } else if (input.type === 'number') {
      obj[lastKey] = Number(input.value);
    } else if (input.dataset.type === 'array') {
      obj[lastKey] = input.value.split(',').map((s) => s.trim()).filter(Boolean);
    } else {
      obj[lastKey] = input.value;
    }
  }

  return jsyaml_dump(fullConfig);
}

/**
 * Simple YAML parse/dump using the raw text approach:
 * We send raw YAML to the server which validates it.
 * For form→raw sync we do a basic reconstruction.
 */
function jsyaml_parse(text) {
  // We don't have js-yaml in the browser — parse server-side.
  // For local form editing, we work with the sections object directly.
  // Return a basic object from current config.
  try {
    return JSON.parse(JSON.stringify(currentConfig.sections));
  } catch {
    return {};
  }
}

function jsyaml_dump(obj) {
  // Reconstruct YAML from object — basic implementation
  // The raw textarea is the source of truth for saving.
  return yamlStringify(obj, 0);
}

function yamlStringify(obj, indent) {
  const lines = [];
  const prefix = '  '.repeat(indent);

  for (const [key, value] of Object.entries(obj)) {
    if (value === null || value === undefined) {
      lines.push(`${prefix}${key}: null`);
    } else if (typeof value === 'boolean') {
      lines.push(`${prefix}${key}: ${value}`);
    } else if (typeof value === 'number') {
      lines.push(`${prefix}${key}: ${value}`);
    } else if (Array.isArray(value)) {
      if (value.length === 0) {
        lines.push(`${prefix}${key}: []`);
      } else {
        lines.push(`${prefix}${key}:`);
        for (const item of value) {
          lines.push(`${prefix}  - ${item}`);
        }
      }
    } else if (typeof value === 'object') {
      lines.push(`${prefix}${key}:`);
      lines.push(yamlStringify(value, indent + 1));
    } else {
      // String — quote if it contains special chars
      const str = String(value);
      if (str.includes(':') || str.includes('#') || str.includes('"') || str === '') {
        lines.push(`${prefix}${key}: "${str.replace(/"/g, '\\"')}"`);
      } else {
        lines.push(`${prefix}${key}: ${str}`);
      }
    }
  }

  return lines.join('\n');
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

  // Get the raw YAML to save
  let yamlContent;
  if (currentMode === 'raw') {
    const textarea = document.getElementById('settings-raw-editor');
    yamlContent = textarea ? textarea.value : '';
  } else {
    // In form mode, use the original raw with form modifications
    // The simplest correct approach: use the raw editor content if dirty,
    // otherwise reconstruct from form
    yamlContent = buildRawFromForm();
  }

  let configSaved = false;
  try {
    // Save config
    const saveResp = await fetch('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ raw: yamlContent }),
    });
    if (!saveResp.ok) {
      const detail = await saveResp.json().catch(() => ({}));
      throw new Error(detail.detail || `Save failed (HTTP ${saveResp.status})`);
    }
    configSaved = true;

    isDirty = false;
    updateSaveBar();
    if (status) status.textContent = '';
    if (applyBtn) applyBtn.disabled = false;

    // Close drawer, then restart terminal (same as the Refresh button)
    closeDrawer();
    await restartTerminal();
  } catch (e) {
    const prefix = configSaved ? 'Config saved, but: ' : '';
    if (status) status.textContent = `${prefix}${e.message}`;
    if (applyBtn) applyBtn.disabled = false;
  }
}

function buildRawFromForm() {
  if (!currentConfig) return '';

  // Parse the original raw YAML lines
  const lines = currentConfig.raw.split('\n');

  // Collect form values by key
  const formValues = {};
  const inputs = document.querySelectorAll('#settings-form [data-key]');
  for (const input of inputs) {
    const key = input.dataset.key;
    if (input.type === 'checkbox') {
      formValues[key] = input.checked;
    } else if (input.type === 'number') {
      formValues[key] = Number(input.value);
    } else if (input.dataset.type === 'array') {
      formValues[key] = input.value.split(',').map((s) => s.trim()).filter(Boolean);
    } else {
      formValues[key] = input.value;
    }
  }

  // For simplicity, update values in the original YAML by matching keys.
  // This preserves comments and formatting for unchanged values.
  // For changed values, do simple line replacement.
  const result = [];
  const keyStack = [];

  for (const line of lines) {
    const trimmed = line.trimStart();
    const indent = line.length - trimmed.length;
    const indentLevel = Math.floor(indent / 2);

    // Track nesting
    const match = trimmed.match(/^(\w[\w_]*):\s*(.*)/);
    if (match) {
      const [, key, rest] = match;

      // Adjust key stack to current indent level
      while (keyStack.length > indentLevel) keyStack.pop();
      keyStack.push(key);

      const fullKey = keyStack.join('.');
      const hasValue = rest !== '' && !rest.startsWith('#');

      if (hasValue && fullKey in formValues) {
        const newVal = formValues[fullKey];
        const formatted = formatYamlValue(newVal);
        result.push(`${' '.repeat(indent)}${key}: ${formatted}`);
        delete formValues[fullKey]; // consumed
        continue;
      }
    }

    result.push(line);
  }

  return result.join('\n');
}

function formatYamlValue(value) {
  if (typeof value === 'boolean') return String(value);
  if (typeof value === 'number') return String(value);
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    return '[' + value.join(', ') + ']';
  }
  const str = String(value);
  if (str.includes(':') || str.includes('#') || str === '') {
    return `"${str.replace(/"/g, '\\"')}"`;
  }
  return str;
}

function formatLabel(key) {
  return key.replace(/_/g, ' ');
}
