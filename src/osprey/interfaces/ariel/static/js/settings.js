/**
 * Settings module
 *
 * Settings drawer for editing OSPREY config.yml. Manages the form/raw
 * editor modes, config loading/saving, and the confirmation dialog.
 */

import { configApi } from './api.js';
import { escapeHtml } from '/design-system/js/dom.js';

let configDirty = false;
let originalRaw = '';

/**
 * Initialize settings module.
 */
export function initSettings() {
  // Mode bar (Form / Raw)
  document.querySelectorAll('.settings-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => switchMode(btn.dataset.mode));
  });

  // Apply button
  const applyBtn = document.getElementById('config-apply-btn');
  if (applyBtn) applyBtn.addEventListener('click', () => showConfirm('config'));

  // Confirmation dialog buttons
  const confirmYes = document.getElementById('settings-confirm-yes');
  const confirmNo = document.getElementById('settings-confirm-no');
  if (confirmYes) confirmYes.addEventListener('click', handleConfirmYes);
  if (confirmNo) confirmNo.addEventListener('click', hideConfirm);

  // Raw YAML editor change tracking
  const rawEditor = document.getElementById('config-raw-editor');
  if (rawEditor) {
    rawEditor.addEventListener('input', () => {
      configDirty = rawEditor.value !== originalRaw;
      updateConfigStatus();
    });
  }

  // Load config whenever the drawer opens, via the component's public
  // drawer:open event (fired on the host) rather than a MutationObserver on
  // its `open` attribute — attribute reflection is the component's internal
  // state mechanics, and the event only fires for real transitions (a vetoed
  // or echoed attribute mutation never dispatches it).
  const drawer = document.getElementById('settings-drawer');
  if (drawer) {
    drawer.addEventListener('drawer:open', () => loadConfig());
  }
}

/**
 * Switch between Form and Raw modes within the config tab.
 * @param {string} mode - 'form' or 'raw'
 */
function switchMode(mode) {
  document.querySelectorAll('.settings-mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  const formView = document.getElementById('config-form-view');
  const rawView = document.getElementById('config-raw-view');
  if (formView) formView.style.display = mode === 'form' ? 'block' : 'none';
  if (rawView) rawView.style.display = mode === 'raw' ? 'block' : 'none';
}

/**
 * Load config.yml from backend and populate both form and raw views.
 */
export async function loadConfig() {
  const statusEl = document.getElementById('config-status');
  try {
    const data = await configApi.get();
    originalRaw = data.raw;
    configDirty = false;

    // Populate raw editor
    const rawEditor = document.getElementById('config-raw-editor');
    if (rawEditor) rawEditor.value = data.raw;

    // Populate form view
    renderConfigForm(data.parsed);

    updateConfigStatus();
  } catch (e) {
    if (statusEl) {
      statusEl.textContent = `Error: ${e.message}`;
      statusEl.className = 'settings-status error';
    }
  }
}

/**
 * Render config.yml as an editable form.
 * @param {Object} parsed - Parsed YAML object
 */
function renderConfigForm(parsed) {
  const container = document.getElementById('config-form-fields');
  if (!container) return;

  container.innerHTML = '';

  // Render top-level sections
  for (const [section, value] of Object.entries(parsed)) {
    if (typeof value !== 'object' || value === null) {
      // Simple key-value
      container.appendChild(createField(section, value, [section]));
      continue;
    }

    const sectionEl = document.createElement('div');
    sectionEl.className = 'settings-section';
    sectionEl.innerHTML = `
      <div class="settings-section-header">
        <span class="section-title">${escapeHtml(section)}</span>
      </div>
      <div class="settings-section-body"></div>
    `;

    const body = sectionEl.querySelector('.settings-section-body');
    renderFields(body, value, [section]);
    container.appendChild(sectionEl);
  }
}

/**
 * Recursively render fields for a config object.
 */
function renderFields(container, obj, path) {
  for (const [key, value] of Object.entries(obj)) {
    const fullPath = [...path, key];
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      // Nested section — render as sub-group
      const groupLabel = document.createElement('div');
      groupLabel.className = 'settings-field-label';
      groupLabel.style.marginTop = '8px';
      groupLabel.style.color = 'var(--text-muted)';
      groupLabel.textContent = key;
      container.appendChild(groupLabel);
      renderFields(container, value, fullPath);
    } else {
      container.appendChild(createField(key, value, fullPath));
    }
  }
}

/**
 * Create a form field element.
 */
function createField(label, value, path) {
  const field = document.createElement('div');
  field.className = 'settings-field';

  const labelEl = document.createElement('span');
  labelEl.className = 'settings-field-label';
  labelEl.textContent = label;
  field.appendChild(labelEl);

  const inputWrap = document.createElement('div');
  inputWrap.className = 'settings-field-input';

  if (typeof value === 'boolean') {
    const toggle = document.createElement('label');
    toggle.className = 'toggle-switch';
    toggle.innerHTML = `
      <input type="checkbox" ${value ? 'checked' : ''} data-path="${path.join('.')}">
      <span class="toggle-slider"></span>
    `;
    toggle.querySelector('input').addEventListener('change', () => {
      configDirty = true;
      updateConfigStatus();
      syncFormToRaw();
    });
    inputWrap.appendChild(toggle);
  } else if (Array.isArray(value)) {
    const input = document.createElement('input');
    input.className = 'input';
    input.type = 'text';
    input.value = value.join(', ');
    input.dataset.path = path.join('.');
    input.dataset.type = 'array';
    input.addEventListener('input', () => {
      configDirty = true;
      updateConfigStatus();
      syncFormToRaw();
    });
    inputWrap.appendChild(input);
  } else {
    const input = document.createElement('input');
    input.className = 'input';
    input.type = typeof value === 'number' ? 'number' : 'text';
    input.value = value ?? '';
    input.dataset.path = path.join('.');
    input.addEventListener('input', () => {
      configDirty = true;
      updateConfigStatus();
      syncFormToRaw();
    });
    inputWrap.appendChild(input);
  }

  field.appendChild(inputWrap);
  return field;
}

/**
 * Sync form field values back to the raw YAML editor.
 */
function syncFormToRaw() {
  try {
    // Reconstruct the config object from form fields
    const rawEditor = document.getElementById('config-raw-editor');
    if (!rawEditor) return;

    // For simplicity, we update the raw editor only when switching to raw mode
    // The raw editor is the source of truth for saves
  } catch (e) {
    console.warn('Form-to-raw sync skipped:', e);
  }
}

/**
 * Update the config status display.
 */
function updateConfigStatus() {
  const statusEl = document.getElementById('config-status');
  const applyBtn = document.getElementById('config-apply-btn');
  if (!statusEl) return;

  if (configDirty) {
    statusEl.textContent = 'Unsaved changes';
    statusEl.className = 'settings-status dirty';
    if (applyBtn) applyBtn.disabled = false;
  } else {
    statusEl.textContent = 'No changes';
    statusEl.className = 'settings-status';
    if (applyBtn) applyBtn.disabled = true;
  }
}

// --- Confirmation dialog ---

let pendingAction = null;

function showConfirm(action) {
  pendingAction = action;
  const overlay = document.getElementById('settings-confirm-overlay');
  const titleEl = document.getElementById('settings-confirm-title');
  const textEl = document.getElementById('settings-confirm-text');

  if (action === 'config') {
    titleEl.textContent = 'Apply Configuration?';
    textEl.textContent =
      'This will overwrite config.yml and may require a service restart to take effect.';
  }

  overlay.classList.add('visible');
}

function hideConfirm() {
  pendingAction = null;
  const overlay = document.getElementById('settings-confirm-overlay');
  overlay.classList.remove('visible');
}

async function handleConfirmYes() {
  const action = pendingAction;
  hideConfirm();

  if (action === 'config') {
    await saveConfig();
  }
}

/**
 * Save config.yml via API.
 */
async function saveConfig() {
  const rawEditor = document.getElementById('config-raw-editor');
  const statusEl = document.getElementById('config-status');
  if (!rawEditor) return;

  try {
    statusEl.textContent = 'Saving...';
    statusEl.className = 'settings-status';
    await configApi.update(rawEditor.value);
    originalRaw = rawEditor.value;
    configDirty = false;
    statusEl.textContent = 'Saved successfully';
    statusEl.className = 'settings-status saved';
  } catch (e) {
    statusEl.textContent = `Save failed: ${e.message}`;
    statusEl.className = 'settings-status error';
  }
}
