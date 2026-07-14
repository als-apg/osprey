// @ts-check
/* OSPREY Lattice Dashboard — Computation Settings Layer
 *
 * The declarative per-figure settings schema (SETTINGS_FIELDS) and the
 * settings form built from it: load/render/collect/apply/reset. The schema
 * stays a plain data structure; the form builder stays createElement()-based
 * to match the rest of the dashboard's DOM style.
 *
 * Only dependency is net.js's generic apiFetch() REST helper — no
 * callback injection needed, unlike net.js/render.js/ui.js, since nothing
 * here needs to call back into another module's effects.
 */

import { apiFetch } from './net.js';

/**
 * @typedef {{key: string, label: string, unit: string, type: 'int'|'float'|'int_or_null', min: number, max: number, step: number}} SettingsField
 * @typedef {{label: string, fields: SettingsField[]}} SettingsGroup
 */

/** @type {Record<string, SettingsGroup>} */
export const SETTINGS_FIELDS = {
  da: {
    label: 'DYNAMIC APERTURE',
    fields: [
      { key: 'nturns', label: 'Turns', unit: '', type: 'int', min: 64, max: 8192, step: 64 },
      { key: 'n_angles', label: 'Angles', unit: '', type: 'int', min: 5, max: 72, step: 1 },
      { key: 'amp_max_mm', label: 'Max amp', unit: 'mm', type: 'float', min: 1, max: 100, step: 1 },
      { key: 'n_bisect', label: 'Bisect steps', unit: '', type: 'int', min: 5, max: 30, step: 1 },
    ],
  },
  lma: {
    label: 'MOMENTUM APERTURE',
    fields: [
      { key: 'nturns', label: 'Turns', unit: '', type: 'int', min: 64, max: 8192, step: 64 },
      { key: 'n_refpts', label: 'Ref pts', unit: '', type: 'int', min: 10, max: 500, step: 10 },
      { key: 'dp_max_pct', label: 'dp max', unit: '%', type: 'float', min: 0.5, max: 20, step: 0.5 },
      { key: 'n_sectors', label: 'Sectors', unit: '', type: 'int_or_null', min: 1, max: 100, step: 1 },
      { key: 'n_bisect', label: 'Bisect steps', unit: '', type: 'int', min: 5, max: 30, step: 1 },
    ],
  },
  chromaticity: {
    label: 'CHROMATICITY',
    fields: [
      { key: 'dp_min_pct', label: 'dp min', unit: '%', type: 'float', min: -20, max: 0, step: 0.5 },
      { key: 'dp_max_pct', label: 'dp max', unit: '%', type: 'float', min: 0, max: 20, step: 0.5 },
      { key: 'n_steps', label: 'Steps', unit: '', type: 'int', min: 5, max: 200, step: 5 },
    ],
  },
  footprint: {
    label: 'TUNE FOOTPRINT',
    fields: [
      { key: 'n_amp', label: 'Grid pts', unit: '', type: 'int', min: 3, max: 30, step: 1 },
      { key: 'x_max_mm', label: 'x max', unit: 'mm', type: 'float', min: 0.1, max: 50, step: 0.5 },
      { key: 'y_max_mm', label: 'y max', unit: 'mm', type: 'float', min: 0.1, max: 50, step: 0.5 },
      { key: 'n_half', label: 'Half-turns', unit: '', type: 'int', min: 32, max: 2048, step: 32 },
    ],
  },
};

/** Fetch the current settings from the backend and (re)render the form. */
export async function loadSettings() {
  try {
    const settings = await apiFetch('/api/settings');
    renderSettingsForm(settings);
  } catch (err) {
    console.warn('Failed to load settings:', err);
  }
}

/**
 * Build the settings form from SETTINGS_FIELDS, one numeric input per
 * schema field, populated from `settings` (falling back to a placeholder
 * for an unset `int_or_null` field).
 * @param {Record<string, Record<string, number|null>>} settings
 */
export function renderSettingsForm(settings) {
  const container = document.getElementById('settings-container');
  if (!container) return;
  container.textContent = '';

  for (const [group, meta] of Object.entries(SETTINGS_FIELDS)) {
    const groupEl = document.createElement('div');
    groupEl.className = 'settings-group';

    const header = document.createElement('div');
    header.className = 'settings-group-header';
    header.textContent = meta.label;
    groupEl.appendChild(header);

    const groupSettings = settings[group] || {};

    for (const field of meta.fields) {
      const row = document.createElement('div');
      row.className = 'settings-field';

      const label = document.createElement('span');
      label.className = 'settings-field-label';
      label.textContent = field.label;
      row.appendChild(label);

      const input = document.createElement('input');
      input.type = 'number';
      input.className = 'settings-field-input';
      input.id = `setting-${group}-${field.key}`;
      input.min = String(field.min);
      input.max = String(field.max);
      input.step = String(field.step);
      input.dataset.group = group;
      input.dataset.key = field.key;
      input.dataset.fieldType = field.type;

      const val = groupSettings[field.key];
      if (val != null) {
        input.value = String(field.type === 'int' || field.type === 'int_or_null' ? Math.round(val) : val);
      } else if (field.type === 'int_or_null') {
        input.placeholder = 'auto';
      }

      row.appendChild(input);

      if (field.unit) {
        const unit = document.createElement('span');
        unit.className = 'settings-field-unit';
        unit.textContent = field.unit;
        row.appendChild(unit);
      }

      groupEl.appendChild(row);
    }

    container.appendChild(groupEl);
  }

  // Action buttons
  const actions = document.createElement('div');
  actions.className = 'settings-actions';

  const resetBtn = document.createElement('button');
  resetBtn.className = 'settings-btn';
  resetBtn.textContent = 'RESET';
  resetBtn.addEventListener('click', resetSettings);
  actions.appendChild(resetBtn);

  const applyBtn = document.createElement('button');
  applyBtn.className = 'settings-btn settings-btn--apply';
  applyBtn.textContent = 'APPLY';
  applyBtn.addEventListener('click', applySettings);
  actions.appendChild(applyBtn);

  container.appendChild(actions);
}

/**
 * Read every rendered settings input back into a nested
 * `{group: {key: value}}` map, honoring each field's declared type.
 * @returns {Record<string, Record<string, number|null>>}
 */
export function collectSettingsFromForm() {
  /** @type {Record<string, Record<string, number|null>>} */
  const settings = {};
  const inputs = document.querySelectorAll('.settings-field-input');
  for (const inputEl of inputs) {
    const input = /** @type {HTMLInputElement} */ (inputEl);
    const group = /** @type {string} */ (input.dataset.group);
    const key = /** @type {string} */ (input.dataset.key);
    const fieldType = input.dataset.fieldType;

    if (!settings[group]) settings[group] = {};

    if (fieldType === 'int_or_null') {
      const raw = input.value.trim();
      settings[group][key] = raw === '' ? null : parseInt(raw, 10);
    } else if (fieldType === 'int') {
      settings[group][key] = parseInt(input.value, 10);
    } else {
      settings[group][key] = parseFloat(input.value);
    }
  }
  return settings;
}

/** Collect the form, PUT it to the backend, then refresh the fast figures. */
export async function applySettings() {
  const settings = collectSettingsFromForm();
  try {
    await apiFetch('/api/settings', {
      method: 'PUT',
      body: JSON.stringify({ settings }),
    });
    // Refresh fast figures (cheap, ~1-2s each)
    await apiFetch('/api/refresh', { method: 'POST' });
  } catch (err) {
    console.error('Apply settings failed:', err);
  }
}

/** Delete the backend's settings overrides and re-render the defaults returned. */
export async function resetSettings() {
  try {
    const result = await apiFetch('/api/settings', { method: 'DELETE' });
    if (result.settings) renderSettingsForm(result.settings);
  } catch (err) {
    console.error('Reset settings failed:', err);
  }
}
