/* OSPREY Web Terminal — Agent Settings Panel */

import { fetchJSON, apiRequest } from './api.js';
import { restartTerminal, startTerminal } from './terminal.js';
import { showSettingsNotice } from './settings-notice.js';
import { hideApplyConfirm, initApplyConfirmGate } from './settings-apply-confirm.js';
import { isSectionProtected, markProtected, protectedNote } from './settings-protected.js';
import { initSettingsWarningGate, runWarningGate } from './settings-warning-gate.js';

/**
 * The config document returned by GET /api/config.
 * @typedef {object} ConfigPayload
 * @property {Record<string, any>} sections  nested config tree, keyed by section
 * @property {string} raw  the raw YAML file text
 * @property {string} [path]
 * @property {string[]} [protected]  dotted leaf keys the form shows read-only
 */

/**
 * A dot-keyed map of changed fields sent to PATCH /api/config.
 * @typedef {Record<string, any>} SettingsFormUpdates
 */

/**
 * The osprey-drawer custom element: an HTMLElement superset that adds
 * imperative open()/close()/toggle() methods over the `open` attribute.
 * @typedef {HTMLElement & { open(): void; close(): void; toggle(): void }} DrawerElement
 */

/** @type {ConfigPayload|null} */
let currentConfig = null;
// Dotted leaf keys the build owns: rendered disabled and never sent on Apply.
/** @type {Set<string>} */
let protectedKeys = new Set();
let isDirty = false;
let currentMode = 'form'; // 'form' | 'raw'

// The agent tab panel — all DOM queries are scoped to this element
/** @type {HTMLElement|null} */
let agentPanel = null;

// The settings drawer element — resolved once in initSettings(), handed to the
// warning gate, and reused by applySettings() to close it and by
// openDrawerTab() to open it.
/** @type {DrawerElement|null} */
let settingsDrawer = null;

// The policy vocabulary a per-tool approval entry offers. The approval hook
// resolves a policy for ANY tool it gates, by short name, so which tools a
// deployment states here is the deployment's business — the drawer renders the
// select for whatever keys the config carries and never decides the roster.
const APPROVAL_POLICIES = ['always', 'selective', 'skip'];

// Every leaf under this prefix is a per-tool approval policy.
const APPROVAL_TOOL_PREFIX = 'approval.tools.';

// Known enum values for select dropdowns, keyed by the full dotted path of a
// LEAF field. Per-tool approval policies are not here: they are the prefix
// rule above.
/** @type {Record<string, string[]>} */
const ENUM_FIELDS = {
  'claude_code.effort': ['low', 'medium', 'high', 'max'],
  'approval.default_policy': APPROVAL_POLICIES,
  // The write ceiling: one of three words, never a boolean.
  'control_system.writes': ['never', 'asked', 'allowed'],
};

/**
 * The option list a key renders as a select over, or null for a plain input.
 *
 * @param {string} fullKey
 * @returns {string[]|null}
 */
function enumOptionsFor(fullKey) {
  if (fullKey.startsWith(APPROVAL_TOOL_PREFIX)) return APPROVAL_POLICIES;
  return ENUM_FIELDS[fullKey] ?? null;
}

// Fields that should render as toggles even when the current value is null or
// absent. A field whose value is already a boolean gets a toggle regardless
// (see createInputForValue), so this set only matters for unset keys.
const BOOLEAN_FIELDS = new Set([
  'approval.enabled',
  'ariel.enabled',
  'artifact_server.auto_launch',
]);

/**
 * Initialize the settings panel. Call once on DOMContentLoaded.
 */
export function initSettings() {
  settingsDrawer = /** @type {DrawerElement|null} */ (document.getElementById('settings-drawer'));
  // Invariant: the warning gate must never be gated on elements unrelated to
  // the drawer (fail-closed) — install it as soon as the drawer itself is
  // resolved, before the unrelated `#tab-config` guard clause below, so a
  // missing/renamed config tab can never silently leave the drawer ungated.
  if (settingsDrawer) initSettingsWarningGate(settingsDrawer);

  agentPanel = document.getElementById('tab-config');
  if (!settingsDrawer || !agentPanel) return;

  // Load config when agent tab becomes active (covers both drawer open and tab switch)
  agentPanel.addEventListener('drawer:tab-activate', () => loadConfig());

  // Mode toggle buttons (scoped to agent panel)
  /** @type {NodeListOf<HTMLElement>} */ (
    agentPanel.querySelectorAll('.settings-mode-btn')
  ).forEach((btn) => {
    btn.addEventListener('click', () => switchMode(/** @type {string} */ (btn.dataset.mode)));
  });

  // Apply button and its confirm dialog, behind the don't-ask-again gate
  // (settings-apply-confirm.js).
  initApplyConfirmGate(agentPanel, () => void applySettings());

  // Raw editor dirty tracking
  const rawEditor = document.getElementById('settings-raw-editor');
  if (rawEditor) rawEditor.addEventListener('input', markDirty);
}

/**
 * Open the settings drawer to a specific tab, always THROUGH the warning gate
 * (never bypassing it). The single gated seam the command palette builds on.
 * @param {string} tabId  e.g. 'tab-config' | 'tab-behavior' | 'tab-safety'
 * @returns {Promise<boolean>} true once the target tab is active; false if the
 *   gate was declined, the tab is missing, or the drawer's unsaved-changes
 *   guard vetoed the switch (observable as the tab NOT gaining `.active`).
 */
export async function openDrawerTab(tabId) {
  if (!(await runWarningGate())) return false;
  // Resolve the tab (scoped to this drawer) BEFORE opening, so a bogus tabId
  // returns false without leaving an empty drawer visibly open.
  const tab = /** @type {HTMLElement|null} */ (
    settingsDrawer?.querySelector(`.drawer-tab[data-tab="${tabId}"]`) ?? null
  );
  if (!tab) return false;
  settingsDrawer?.open();
  // The drawer component activates a tab via its delegated click handler; a
  // vetoed switch returns early WITHOUT adding `.active`. No polling, no retry.
  tab.click();
  return tab.classList.contains('active');
}

// revealSetting waits this long for the lazily-rendered config form to paint
// the target field before degrading gracefully (Config tab left open, no jump).
const REVEAL_POLL_MS = 100;
const REVEAL_MAX_POLLS = 20;

// Short-lived class flashed on a revealed field. The keyframe/transition
// styling lives in CSS; this module only adds it and removes it after a beat.
const FIELD_FLASH_CLASS = 'settings-field-flash';
const FIELD_FLASH_MS = 1600;

/**
 * Open the settings drawer to the Config tab and jump to a single config
 * field: expand its section, scroll it into view, and flash-highlight it.
 * Routes through openDrawerTab (hence the warning gate). Degrades gracefully
 * — no throw, no highlight — if the gate is declined, the tab is vetoed, or
 * the field never renders.
 * @param {string} dotKey  e.g. 'approval.tools.channel_write'
 * @returns {Promise<void>}
 */
export async function revealSetting(dotKey) {
  const ok = await openDrawerTab('tab-config');
  if (!ok) return;

  // Force form view — [data-key] fields exist ONLY there, and currentMode
  // persists across tab activations (the tab may have been left in raw mode).
  switchMode('form');

  const field = await waitForConfigField(dotKey);
  if (!field) return; // lazy render timed out; leave the Config tab open

  const section = field.closest('.settings-section');
  if (section) section.classList.add('expanded');
  const target = field.closest('.settings-field') || field;
  target.scrollIntoView({ block: 'center', behavior: 'smooth' });
  target.classList.add(FIELD_FLASH_CLASS);
  setTimeout(() => target.classList.remove(FIELD_FLASH_CLASS), FIELD_FLASH_MS);
}

/**
 * Poll for the lazily-rendered form input carrying data-key === dotKey.
 * Matches by dataset.key rather than an attribute selector so dotted keys
 * need no CSS escaping and no selector injection is possible.
 * @param {string} dotKey
 * @returns {Promise<HTMLElement|null>} the element, or null on timeout
 */
function waitForConfigField(dotKey) {
  return new Promise((resolve) => {
    let polls = 0;
    const check = () => {
      const form = document.getElementById('settings-form');
      if (form) {
        const fields = /** @type {NodeListOf<HTMLElement>} */ (form.querySelectorAll('[data-key]'));
        const match = Array.from(fields).find((el) => el.dataset.key === dotKey);
        if (match) {
          resolve(match);
          return;
        }
      }
      if (++polls >= REVEAL_MAX_POLLS) {
        resolve(null);
        return;
      }
      setTimeout(check, REVEAL_POLL_MS);
    };
    check();
  });
}

async function loadConfig() {
  const formContainer = document.getElementById('settings-form');
  const rawTextarea = /** @type {HTMLTextAreaElement|null} */ (
    document.getElementById('settings-raw-editor')
  );
  const loading = document.getElementById('settings-loading');
  const error = document.getElementById('settings-error');

  if (loading) loading.style.display = 'flex';
  if (error) error.style.display = 'none';
  if (formContainer) formContainer.innerHTML = '';

  try {
    currentConfig = /** @type {ConfigPayload} */ (await fetchJSON('/api/config')); // fetchJSON prefixes internally
    if (loading) loading.style.display = 'none';

    protectedKeys = new Set(currentConfig.protected ?? []);

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
      error.textContent = `Failed to load config: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
}

/** @param {string} mode */
function switchMode(mode) {
  currentMode = mode;
  if (!agentPanel) return;

  /** @type {NodeListOf<HTMLElement>} */ (
    agentPanel.querySelectorAll('.settings-mode-btn')
  ).forEach((btn) => {
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

/** @param {Record<string, any>} sections */
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

    const sectionProtected = isSectionProtected(sectionValue, sectionKey, protectedKeys);
    renderFields(body, sectionValue, sectionKey, 0, sectionProtected);

    section.appendChild(header);
    if (sectionProtected) section.appendChild(protectedNote());
    section.appendChild(body);
    container.appendChild(section);
  }
}

/**
 * @param {HTMLElement} container
 * @param {Record<string, any>} obj
 * @param {string} prefix
 * @param {number} [depth]
 * @param {boolean} [sectionProtected]  the section carries the note for every leaf
 */
function renderFields(container, obj, prefix, depth = 0, sectionProtected = false) {
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = `${prefix}.${key}`;

    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      const group = document.createElement('div');
      group.className = 'settings-subgroup';
      group.style.setProperty('--depth', String(depth));

      const groupLabel = document.createElement('div');
      groupLabel.className = 'settings-subgroup-label';
      groupLabel.textContent = formatLabel(key);
      groupLabel.title = fullKey;
      group.appendChild(groupLabel);

      renderFields(group, value, fullKey, depth + 1, sectionProtected);

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
    if (protectedKeys.has(fullKey)) markProtected(field, input, !sectionProtected);
    container.appendChild(field);
  }
}

/**
 * @param {string} fullKey
 * @param {any} value
 * @returns {HTMLElement}
 */
function createInputForValue(fullKey, value) {
  const options = enumOptionsFor(fullKey);
  if (options) {
    const select = document.createElement('select');
    select.className = 'settings-select';
    select.dataset.key = fullKey;
    for (const opt of options) {
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
    input.value = String(value);
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
/** @returns {SettingsFormUpdates} */
function collectFormUpdates() {
  /** @type {SettingsFormUpdates} */
  const updates = {};
  const inputs = /** @type {NodeListOf<HTMLInputElement>} */ (
    document.querySelectorAll('#settings-form [data-key]')
  );

  for (const input of inputs) {
    const key = /** @type {string} */ (input.dataset.key);
    if (protectedKeys.has(key)) continue;
    let newValue;

    if (input.type === 'checkbox') {
      newValue = input.checked;
    } else if (input.type === 'number') {
      newValue = Number(input.value);
    } else if (input.dataset.type === 'array') {
      newValue = input.value
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
    } else {
      newValue = input.value;
    }

    // Compare against original to only send changed fields
    const originalValue = getNestedValue(currentConfig?.sections, key);
    if (!deepEqual(originalValue, newValue)) {
      updates[key] = newValue;
    }
  }

  return updates;
}

/**
 * @param {any} obj
 * @param {string} dottedKey
 * @returns {any}
 */
function getNestedValue(obj, dottedKey) {
  const parts = dottedKey.split('.');
  let node = obj;
  for (const part of parts) {
    if (node == null || typeof node !== 'object') return undefined;
    node = node[part];
  }
  return node;
}

/**
 * @param {any} a
 * @param {any} b
 * @returns {boolean}
 */
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

async function applySettings() {
  if (agentPanel) hideApplyConfirm(agentPanel);

  const status = agentPanel ? agentPanel.querySelector('.settings-status') : null;
  const applyBtn = /** @type {HTMLButtonElement|null} */ (
    agentPanel ? agentPanel.querySelector('.settings-apply-btn') : null
  );
  if (applyBtn) applyBtn.disabled = true;
  if (status) status.textContent = 'Saving...';

  let configSaved = false;
  /** @type {any} */
  let result = null;
  try {
    if (currentMode === 'raw') {
      // Raw mode: send the full YAML text as-is (user is responsible for content)
      const textarea = /** @type {HTMLTextAreaElement|null} */ (
        document.getElementById('settings-raw-editor')
      );
      const yamlContent = textarea ? textarea.value : '';
      result = await apiRequest('/api/config', {
        method: 'PUT',
        json: { raw: yamlContent },
        errorPrefix: 'Save failed',
      });
    } else {
      // Form mode: send only changed fields via PATCH — server uses ruamel.yaml
      // to apply them without stripping comments or reordering the file.
      const updates = collectFormUpdates();
      if (Object.keys(updates).length === 0) {
        if (status) status.textContent = 'No changes to apply';
        if (applyBtn) applyBtn.disabled = false;
        return;
      }
      result = await apiRequest('/api/config', {
        method: 'PATCH',
        json: { updates },
        errorPrefix: 'Patch failed',
      });
    }
    configSaved = true;

    // A write can succeed and still leave the operator with something they must
    // know. `detail` is how the server says so -- today, a read-only render zone
    // where config.yml took the edit but the artifacts derived from it wait for
    // the container restart. Shown, not awaited: it acknowledges what already
    // happened, and the terminal restart below must not queue behind a dialog.
    if (result && typeof result.detail === 'string' && result.detail)
      showSettingsNotice(result.detail);

    isDirty = false;
    updateSaveBar();
    if (status) status.textContent = '';
    if (applyBtn) applyBtn.disabled = false;

    settingsDrawer?.close();
    await restartTerminal();
    startTerminal();
  } catch (e) {
    const prefix = configSaved ? 'Config saved, but: ' : '';
    if (status) status.textContent = `${prefix}${e instanceof Error ? e.message : String(e)}`;
    if (applyBtn) applyBtn.disabled = false;
  }
}

/** @param {string} key */
function formatLabel(key) {
  return key.replace(/_/g, ' ');
}
