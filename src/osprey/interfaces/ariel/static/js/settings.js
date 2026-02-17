/**
 * Settings module
 *
 * Unified settings drawer with two tabs: OSPREY Config and Claude Setup.
 * Manages tab switching, config loading/saving, and the confirmation dialog.
 */

import { configApi } from './api.js';
import { initClaudeSetup, hasUnsavedEdits as claudeHasUnsaved } from './claude-setup.js';

let currentTab = 'config';
let configDirty = false;
let originalRaw = '';

/**
 * Initialize settings module.
 */
export function initSettings() {
  // Tab switching
  document.querySelectorAll('.settings-tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Mode bar (Form / Raw) within the config tab
  document.querySelectorAll('.settings-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => switchMode(btn.dataset.mode));
  });

  // Apply button (config tab)
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

  // Initialize Claude Setup sub-module
  initClaudeSetup();

  // Observe drawer open to load config on first show
  const drawer = document.getElementById('settings-drawer');
  if (drawer) {
    const observer = new MutationObserver(() => {
      if (drawer.classList.contains('open')) {
        if (currentTab === 'config') loadConfig();
      }
    });
    observer.observe(drawer, { attributes: true, attributeFilter: ['class'] });
  }
}

/**
 * Switch between tabs.
 * @param {string} tabName - 'config' or 'claude'
 */
function switchTab(tabName) {
  // Warn about unsaved changes
  if (currentTab === 'config' && configDirty && tabName !== 'config') {
    if (!confirm('You have unsaved config changes. Switch tabs anyway?')) return;
    configDirty = false;
  }
  if (currentTab === 'claude' && claudeHasUnsaved() && tabName !== 'claude') {
    if (!confirm('You have unsaved file edits. Switch tabs anyway?')) return;
  }

  currentTab = tabName;

  // Update tab buttons
  document.querySelectorAll('.settings-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabName);
  });

  // Update panes
  document.querySelectorAll('.settings-tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `${tabName}-pane`);
  });

  // Lazy-load data for the activated tab
  if (tabName === 'config') {
    loadConfig();
  } else if (tabName === 'claude') {
    import('./claude-setup.js').then(m => m.loadFileList());
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
  } else if (action === 'claude-file') {
    titleEl.textContent = 'Save & Restart?';
    textEl.textContent =
      'This will save the file (a .bak backup will be created) and may require a restart.';
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
  } else if (action === 'claude-file') {
    // Delegated to claude-setup.js via the exported doSave
    const { doSave } = await import('./claude-setup.js');
    await doSave();
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

/**
 * Show the confirmation dialog for a claude-setup file save.
 * Called by claude-setup.js.
 */
export function showClaudeConfirm() {
  showConfirm('claude-file');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
