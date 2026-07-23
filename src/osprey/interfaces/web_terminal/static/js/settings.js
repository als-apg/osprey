/* OSPREY Web Terminal — Agent Settings Panel */

import { fetchJSON, apiRequest } from './api.js';
import { restartTerminal, startTerminal } from './terminal.js';

/**
 * The config document returned by GET /api/config.
 * @typedef {object} ConfigPayload
 * @property {Record<string, any>} sections  nested config tree, keyed by section
 * @property {string} raw  the raw YAML file text
 * @property {string} [path]
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
let isDirty = false;
let currentMode = 'form';  // 'form' | 'raw'

// The agent tab panel — all DOM queries are scoped to this element
/** @type {HTMLElement|null} */
let agentPanel = null;

// The settings drawer element — resolved once in initSettings(), reused by
// applySettings() to close it and by the warning gate below to open it.
/** @type {DrawerElement|null} */
let settingsDrawer = null;

// Per-session warning for the settings drawer (resets on server restart)
const SETTINGS_WARNING_KEY = 'osprey-settings-warning-ack';

// True from the moment a trigger click starts the gate (health check in
// flight, or the warning dialog itself is up) until it resolves one way or
// another. Guards against a rapid second click spawning a second dialog.
let warningGatePending = false;

// Known enum values for select dropdowns
/** @type {Record<string, string[]>} */
const ENUM_FIELDS = {
  'claude_code.effort': ['low', 'medium', 'high', 'max'],
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
  settingsDrawer = /** @type {DrawerElement|null} */ (document.getElementById('settings-drawer'));
  // Invariant: the warning gate must never be gated on elements unrelated to
  // the drawer (fail-closed) — install it as soon as the drawer itself is
  // resolved, before the unrelated `#tab-config` guard clause below, so a
  // missing/renamed config tab can never silently leave the drawer ungated.
  if (settingsDrawer) initSettingsWarningGate();

  agentPanel = document.getElementById('tab-config');
  if (!settingsDrawer || !agentPanel) return;

  // Load config when agent tab becomes active (covers both drawer open and tab switch)
  agentPanel.addEventListener('drawer:tab-activate', () => loadConfig());

  // Mode toggle buttons (scoped to agent panel)
  /** @type {NodeListOf<HTMLElement>} */ (agentPanel.querySelectorAll('.settings-mode-btn')).forEach((btn) => {
    btn.addEventListener('click', () => switchMode(/** @type {string} */ (btn.dataset.mode)));
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

/**
 * Gate opening the settings drawer behind a first-time-per-session warning
 * (these settings control agent behavior, safety hooks, and security
 * policies). The trigger button intentionally carries `data-drawer-trigger`
 * rather than osprey-drawer's own `[data-drawer]` marker, so the component's
 * delegated handler never matches it and never toggles the drawer directly —
 * this gate is the sole open path, and the click reaches every other
 * document-level listener (e.g. sessions.js's outside-click dropdown close)
 * completely unmodified, with no propagation tampering.
 */
function initSettingsWarningGate() {
  const trigger = document.querySelector('[data-drawer-trigger="settings-drawer"]');
  if (!trigger) return;

  trigger.addEventListener('click', async () => {
    // Toggle-close stays FIRST, ahead of the gate: an already-open drawer
    // closes without re-running the warning. The gate never owns the toggle.
    if (settingsDrawer?.hasAttribute('open')) {
      settingsDrawer.close();
      return;
    }
    if (await runWarningGate()) settingsDrawer?.open();
  });
}

// Bounds how long a trigger click can leave `warningGatePending` true while
// waiting on /health -- a stalled backend or a dropped connection with no
// RST would otherwise never settle the fetch and permanently brick the
// gear (every later click a no-op, only a reload recovering).
const HEALTH_CHECK_TIMEOUT_MS = 4000;

/**
 * Encapsulate ONLY the gate decision — the re-entrancy guard, the /health
 * check, and the first-per-session warning dialog. It never opens or closes
 * the drawer; callers do that on a true result. This is the sole gate path;
 * every gated seam (the gear trigger, openDrawerTab, revealSetting) routes
 * through it.
 *
 * Resolves true when the user may proceed: acknowledgment for this server
 * session is already stored, OR the user clicks Proceed. Resolves false on
 * Cancel, Escape, dialog-build failure, or a re-entrant call made while a
 * gate is already pending (the second caller does not spawn a second dialog).
 * @returns {Promise<boolean>}
 */
async function runWarningGate() {
  if (warningGatePending) return false; // a check or the dialog is already in flight
  warningGatePending = true;
  /** @type {string|null} */
  let serverSession = null;
  let timeoutId;
  try {
    const timeout = new Promise((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error('health check timed out')), HEALTH_CHECK_TIMEOUT_MS);
    });
    const healthCheck = fetchJSON('/health');
    // If the timeout wins the race, the still-pending fetch may reject later;
    // swallow that late rejection so it doesn't surface as an unhandled one.
    healthCheck.catch(() => {});
    const health = await Promise.race([healthCheck, timeout]);
    clearTimeout(timeoutId);
    serverSession = health.session_id ?? null;
    // Already acknowledged this server session — proceed without the dialog.
    if (serverSession && localStorage.getItem(SETTINGS_WARNING_KEY) === serverSession) {
      warningGatePending = false;
      return true;
    }
  } catch {
    // Health endpoint unreachable, or slow enough to hit the timeout above —
    // show the warning to be safe (fail-safe-to-warning; the flag clears via
    // the dialog's own cleanup() once dismissed).
    clearTimeout(timeoutId);
    serverSession = null;
  }
  // Show the warning; its dismissal resolves the gate. Defense-in-depth: if
  // building/showing the dialog throws, reset the gate and decline rather
  // than leaving it permanently pending.
  try {
    return await showSettingsWarning(serverSession);
  } catch (error) {
    warningGatePending = false;
    console.error('osprey web_terminal: failed to show the settings warning dialog; gate reset', error);
    return false;
  }
}

/**
 * Build the first-time warning dialog. Persists acknowledgment per server
 * session so it reappears on restart. Resolves true if the user clicks
 * Proceed, false on Cancel or Escape. The DOM/animation/Escape-cleanup are
 * unchanged from the pre-Promise gate; only the terminal action differs —
 * each dismissal path settles this promise instead of opening the drawer.
 * @param {string|null} serverSession
 * @returns {Promise<boolean>}
 */
function showSettingsWarning(serverSession) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'settings-warning-overlay';

    const dialog = document.createElement('div');
    dialog.className = 'settings-warning-dialog';

    dialog.innerHTML = `
      <div class="settings-warning-icon">⚠</div>
      <div class="settings-warning-title">Expert Configuration Area</div>
      <div class="settings-warning-body">
        <p>These settings directly control <strong>agent behavior</strong>,
        <strong>safety hooks</strong>, and <strong>security policies</strong>.</p>
        <p>Incorrect changes can <strong>disable safety checks</strong>,
        bypass human approval requirements, or allow unvalidated writes
        to control system hardware.</p>
        <p>Only modify these settings if you understand the safety
        implications of each option.</p>
      </div>
      <div class="settings-warning-actions">
        <button class="settings-warning-cancel">Cancel</button>
        <button class="settings-warning-proceed">I Understand, Proceed</button>
      </div>
    `;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // Animate in
    requestAnimationFrame(() => overlay.classList.add('visible'));

    // Unconditional on every dismissal path (Cancel, Proceed, Escape) — leaves
    // no stale `keydown` listener behind and always clears the pending gate.
    const cleanup = () => {
      document.removeEventListener('keydown', onKey);
      warningGatePending = false;
      overlay.classList.remove('visible');
      overlay.addEventListener('transitionend', () => overlay.remove(), { once: true });
      // Fallback removal if transition doesn't fire
      setTimeout(() => { if (overlay.parentNode) overlay.remove(); }, 300);
    };

    // Cancel/Escape decline the gate; Proceed acknowledges + accepts. The gate
    // decision is all this resolves — the caller owns opening the drawer.
    const cancel = () => {
      cleanup();
      resolve(false);
    };

    /** @type {HTMLElement} */ (dialog.querySelector('.settings-warning-cancel')).addEventListener('click', cancel);

    /** @type {HTMLElement} */ (dialog.querySelector('.settings-warning-proceed')).addEventListener('click', () => {
      if (serverSession) {
        localStorage.setItem(SETTINGS_WARNING_KEY, serverSession);
      }
      cleanup();
      resolve(true);
    });

    // Escape key cancels
    /** @param {KeyboardEvent} e */
    const onKey = (e) => {
      if (e.key === 'Escape') cancel();
    };
    document.addEventListener('keydown', onKey);
  });
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
 * @param {string} dotKey  e.g. 'control_system.write_verification'
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
  const rawTextarea = /** @type {HTMLTextAreaElement|null} */ (document.getElementById('settings-raw-editor'));
  const loading = document.getElementById('settings-loading');
  const error = document.getElementById('settings-error');

  if (loading) loading.style.display = 'flex';
  if (error) error.style.display = 'none';
  if (formContainer) formContainer.innerHTML = '';

  try {
    currentConfig = /** @type {ConfigPayload} */ (await fetchJSON('/api/config')); // fetchJSON prefixes internally
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
      error.textContent = `Failed to load config: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
}

/** @param {string} mode */
function switchMode(mode) {
  currentMode = mode;
  if (!agentPanel) return;

  /** @type {NodeListOf<HTMLElement>} */ (agentPanel.querySelectorAll('.settings-mode-btn')).forEach((btn) => {
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

    renderFields(body, sectionValue, sectionKey);

    section.appendChild(header);
    section.appendChild(body);
    container.appendChild(section);
  }
}

/**
 * @param {HTMLElement} container
 * @param {Record<string, any>} obj
 * @param {string} prefix
 * @param {number} [depth]
 */
function renderFields(container, obj, prefix, depth = 0) {
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

/**
 * @param {string} fullKey
 * @param {any} value
 * @returns {HTMLElement}
 */
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
  const applyBtn = /** @type {HTMLButtonElement|null} */ (agentPanel ? agentPanel.querySelector('.settings-apply-btn') : null);
  if (applyBtn) applyBtn.disabled = true;
  if (status) status.textContent = 'Saving...';

  let configSaved = false;
  try {
    if (currentMode === 'raw') {
      // Raw mode: send the full YAML text as-is (user is responsible for content)
      const textarea = /** @type {HTMLTextAreaElement|null} */ (document.getElementById('settings-raw-editor'));
      const yamlContent = textarea ? textarea.value : '';
      await apiRequest('/api/config', {
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
      await apiRequest('/api/config', {
        method: 'PATCH',
        json: { updates },
        errorPrefix: 'Patch failed',
      });
    }
    configSaved = true;

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
