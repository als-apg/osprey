// @ts-check
/**
 * ARIEL Advanced Options
 *
 * The Filters view: pure reads of the /api/capabilities descriptor
 * (`effectiveParam`, `declaredParamNames`, `defaultMode`) and render functions
 * for the mode choices and the Filters & Options controls. The view renders
 * from the values passed in and reports changes; the chosen mode and the
 * touched filter values live in the store, never here.
 */

import { escapeHtml } from '/design-system/js/dom.js';

/**
 * @typedef {Object} AdvancedParamOption
 * @property {string} value
 * @property {string} label
 */

/**
 * @typedef {Object} AdvancedParam
 * @property {string} name
 * @property {string} label
 * @property {string} [description]
 * @property {string} type - "float"|"int"|"bool"|"select"|"date"|"text"|"dynamic_select"
 * @property {*} [default]
 * @property {number} [min]
 * @property {number} [max]
 * @property {number} [step]
 * @property {string} [section]
 * @property {string} [placeholder]
 * @property {string} [options_endpoint]
 * @property {AdvancedParamOption[]} [options]
 */

/**
 * @typedef {Object} AdvancedMode
 * @property {string} name
 * @property {string} label
 * @property {string} [description]
 * @property {AdvancedParam[]} [parameters]
 */

/**
 * @typedef {Object} AdvancedCategory
 * @property {string} label
 * @property {AdvancedMode[]} [modes]
 */

/**
 * @typedef {Object} VocabularyCapability
 * @property {boolean} enabled
 * @property {number} concepts
 * @property {boolean} expand_by_default
 */

/**
 * The /api/capabilities payload. `status`/`config_errors`/`remedy` are absent
 * on a healthy panel and present only when the backend reports a degraded
 * configuration: `configuration_invalid` means search is dead (the search form
 * is blocked), `configuration_warning` means search still works.
 * `facility_timezone` is the IANA zone the entry timestamps are rendered in.
 * @typedef {Object} Capabilities
 * @property {Object<string, AdvancedCategory>} categories
 * @property {AdvancedParam[]} [shared_parameters]
 * @property {string|null} [default_mode]
 * @property {'ok'|'configuration_warning'|'configuration_invalid'} [status]
 * @property {string[]} [config_errors]
 * @property {string|null} [remedy]
 * @property {VocabularyCapability} [vocabulary]
 * @property {boolean} [config_panel_enabled]
 * @property {string} [facility_timezone]
 */

// --- Fallback ---
// Used when /api/capabilities could not be fetched at all. The annotation is
// deliberate: it makes tsc prove the literal still satisfies the typedef every
// time a capabilities field is added.
/** @type {Capabilities} */
export const FALLBACK_CAPABILITIES = {
  categories: {
    direct: {
      label: 'Direct',
      modes: [{ name: 'keyword', label: 'Keyword', description: 'Text search', parameters: [] }],
    },
  },
  shared_parameters: [],
  status: 'ok',
  config_errors: [],
  remedy: null,
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
};

// The only module-scope data: fetched dynamic_select options, keyed by
// endpoint. A cache of the backend's answer, not a choice anyone made.
/** @type {Object<string, AdvancedParamOption[]>} */
const dynamicOptionsCache = {};

// --- Pure capabilities reads ---

/**
 * Every mode the capabilities payload advertises, in category order.
 * @param {Capabilities|null|undefined} capabilities
 * @returns {AdvancedMode[]}
 */
export function allModes(capabilities) {
  /** @type {AdvancedMode[]} */
  const modes = [];
  for (const cat of Object.values(capabilities?.categories || {})) {
    modes.push(...(cat.modes || []));
  }
  return modes;
}

/**
 * The mode a search runs in when nobody picked one: the deployment's
 * `default_mode` when it names an advertised mode, else the first advertised
 * mode, else `keyword`.
 * @param {Capabilities|null|undefined} capabilities
 * @returns {string}
 */
export function defaultMode(capabilities) {
  const modes = allModes(capabilities);
  const advertised = capabilities?.default_mode;
  if (advertised && modes.some((m) => m.name === advertised)) return advertised;
  return modes[0]?.name ?? 'keyword';
}

/**
 * Parameter descriptors a mode declares (not the shared ones).
 * @param {Capabilities|null|undefined} capabilities
 * @param {string|null|undefined} mode - Mode name
 * @returns {AdvancedParam[]}
 */
export function modeParameters(capabilities, mode) {
  return allModes(capabilities).find((m) => m.name === mode)?.parameters || [];
}

/**
 * The parameter names a search in `mode` may send: the mode's own plus the
 * shared ones. Anything else in `filters` belongs to another mode and stays
 * off the wire, so hybrid override validation never sees a foreign knob.
 * @param {Capabilities|null|undefined} capabilities
 * @param {string|null|undefined} mode - Mode name
 * @returns {Set<string>}
 */
export function declaredParamNames(capabilities, mode) {
  return new Set([
    ...modeParameters(capabilities, mode).map((p) => p.name),
    ...(capabilities?.shared_parameters || []).map((p) => p.name),
  ]);
}

/**
 * The value a search in `mode` would actually run with for one of that mode's
 * parameters: the touched value from `filters` when there is one, otherwise the
 * deployment's configured default from the mode descriptor. Pure.
 * @param {Capabilities|null|undefined} capabilities
 * @param {string|null|undefined} mode - Mode name
 * @param {Object<string, *>|null|undefined} filters - Touched values, any mode
 * @param {string} name - Parameter name
 * @returns {*} Effective value, or undefined when the mode does not declare `name`
 */
export function effectiveParam(capabilities, mode, filters, name) {
  const descriptor = modeParameters(capabilities, mode).find((p) => p.name === name);
  if (!descriptor) return undefined;
  return filters?.[name] ?? descriptor.default;
}

// --- View: mode choices ---

/**
 * Render the mode choices into `root`. The selected mode is whatever the
 * caller passes; a click only reports through `onSelect`.
 * @param {HTMLElement} root
 * @param {{capabilities: Capabilities|null, mode: string|null, onSelect: (mode: string) => void}} opts
 */
export function renderModeChoices(root, { capabilities, mode, onSelect }) {
  const categories = (capabilities || FALLBACK_CAPABILITIES).categories || {};
  let html = '';
  for (const catKey of ['direct']) {
    const cat = categories[catKey];
    if (!cat || !cat.modes?.length) continue;

    html += `<div class="mode-tab-group">`;
    html += `<span class="mode-tab-group-label">${escapeHtml(cat.label)}</span>`;
    for (const m of cat.modes) {
      const selected = m.name === mode;
      html += `<button type="button" class="mode-tab${selected ? ' active' : ''}" data-mode="${escapeHtml(m.name)}" aria-pressed="${selected}">${escapeHtml(m.label)}</button>`;
    }
    html += `</div>`;
  }
  root.innerHTML = html;

  const tabs = /** @type {NodeListOf<HTMLElement>} */ (root.querySelectorAll('.mode-tab'));
  tabs.forEach((btn) => {
    btn.addEventListener('click', () => {
      const chosen = btn.dataset.mode;
      if (chosen && chosen !== mode) onSelect(chosen);
    });
  });
}

// --- View: filters & options ---

/**
 * @typedef {Object} RenderFiltersOptions
 * @property {Capabilities|null} capabilities
 * @property {string|null} mode - The mode whose parameters (plus the shared ones) are shown
 * @property {Object<string, *>} filters - Touched values, any mode; rendered, never mutated
 * @property {Object<string, *>} [locked] - Name → value rendered disabled and never reported
 * @property {(filters: Object<string, *>) => void} onChange - Receives the whole new filters object
 */

/**
 * Render the Filters & Options controls for `mode` into `root`.
 *
 * The view renders from the values passed in and reports changes; it holds no
 * authoritative state. Each change reports a fresh copy of `filters` with that
 * one name set (or removed, when a text/select/date control is cleared), and
 * successive changes before the caller re-renders accumulate on that copy.
 * Reset reports `{}`.
 * @param {HTMLElement} root
 * @param {RenderFiltersOptions} opts
 */
export function renderFilters(root, { capabilities, mode, filters, locked = {}, onChange }) {
  const caps = capabilities || FALLBACK_CAPABILITIES;
  /** @type {Object<string, *>} */
  let current = { ...(filters || {}) };
  /** @param {*} name */
  const isLocked = (name) => Object.prototype.hasOwnProperty.call(locked, name);
  /** @param {string} name */
  const valueOf = (name) => (isLocked(name) ? locked[name] : current[name]);

  const allParams = [...modeParameters(caps, mode), ...(caps.shared_parameters || [])];

  if (allParams.length === 0) {
    root.innerHTML = `
      <div class="empty-state" style="padding: var(--space-5);">
        <p class="empty-state-text">No advanced options for this mode.</p>
      </div>
    `;
    return;
  }

  // Group by section, separate "Filters" from the rest
  /** @type {AdvancedParam[]} */
  const filterParams = [];
  /** @type {Object<string, AdvancedParam[]>} */
  const otherSections = {};
  for (const param of allParams) {
    const section = param.section || 'General';
    if (section === 'Filters') {
      filterParams.push(param);
    } else {
      if (!otherSections[section]) otherSections[section] = [];
      otherSections[section].push(param);
    }
  }

  /** @param {AdvancedParam} param */
  const control = (param) => renderParameter(param, valueOf(param.name), isLocked(param.name));

  /**
   * @param {string} title
   * @param {AdvancedParam[]} params
   * @param {string} [modifier] - Extra class on the section
   */
  const section = (title, params, modifier = '') => `
      <div class="advanced-section${modifier}">
        <div class="advanced-section-header">
          <span class="section-title">${escapeHtml(title)}</span>
        </div>
        <div class="advanced-section-body">
    ${params.map(control).join('')}</div></div>`;

  let html = filterParams.length > 0 ? section('Filters', filterParams, ' filters-section') : '';
  for (const [sectionName, params] of Object.entries(otherSections)) {
    html += section(sectionName, params);
  }
  html += `
    <div class="advanced-actions">
      <button type="button" class="btn btn-ghost btn-sm" data-action="reset-filters">Reset</button>
    </div>
  `;
  root.innerHTML = html;

  /**
   * Report one control's new value; `null` removes the name.
   * @param {string} name
   * @param {*} value
   */
  const report = (name, value) => {
    if (isLocked(name)) return;
    current = { ...current };
    if (value === null || value === undefined) {
      delete current[name];
    } else {
      current[name] = value;
    }
    onChange({ ...current });
  };

  attachParamListeners(root, report);
  root.querySelector('[data-action="reset-filters"]')?.addEventListener('click', () => {
    current = {};
    onChange({});
  });

  loadDynamicSelectOptions(root, valueOf);
}

/**
 * Render a single parameter control.
 * @param {AdvancedParam} param - Parameter descriptor
 * @param {*} touchedValue - The touched (or locked) value, undefined when untouched
 * @param {boolean} isLocked - Render disabled
 * @returns {string} HTML string
 */
function renderParameter(param, touchedValue, isLocked) {
  const value = touchedValue ?? param.default;
  const disabled = isLocked ? ' disabled' : '';
  // The description is the knob's documentation, so it is shown rather than
  // hidden behind a hover. escapeHtml() stringifies undefined, so a descriptor
  // without a description is skipped here instead of rendering an empty hint.
  const hint = param.description
    ? `<p class="param-hint">${escapeHtml(param.description)}</p>`
    : '';

  switch (param.type) {
    case 'float':
    case 'int':
      return renderSlider(param, value, disabled) + hint;
    case 'bool':
      return renderToggle(param, value, disabled) + hint;
    case 'select':
      return renderSelect(param, value, disabled) + hint;
    case 'date':
      return renderDateInput(param, value, disabled) + hint;
    case 'text':
      return renderTextInput(param, value, disabled) + hint;
    case 'dynamic_select':
      return renderDynamicSelect(param, value, disabled) + hint;
    default:
      return '';
  }
}

/**
 * Render a slider control for float/int params.
 * @param {AdvancedParam} param - Parameter descriptor
 * @param {*} value - Current value
 * @param {string} disabled - '' or ' disabled'
 * @returns {string} HTML string
 */
function renderSlider(param, value, disabled) {
  const displayValue = param.type === 'float' ? Number(value).toFixed(2) : value;
  return `
    <div class="slider-control">
      <div class="slider-header">
        <label class="slider-label" for="param-${param.name}">${escapeHtml(param.label)}</label>
        <span class="slider-value" id="param-${param.name}-value">${displayValue}</span>
      </div>
      <input type="range" id="param-${param.name}" class="slider"
        data-param="${param.name}" data-type="${param.type}"
        min="${param.min ?? 0}" max="${param.max ?? 100}"
        value="${value}" step="${param.step ?? 1}"${disabled}>
    </div>
  `;
}

/**
 * Render a toggle switch for bool params.
 * @param {AdvancedParam} param - Parameter descriptor
 * @param {*} value - Current value
 * @param {string} disabled - '' or ' disabled'
 * @returns {string} HTML string
 */
function renderToggle(param, value, disabled) {
  const checked = value ? 'checked' : '';
  return `
    <div class="toggle-control">
      <label class="toggle-label" for="param-${param.name}">${escapeHtml(param.label)}</label>
      <label class="toggle-switch">
        <input type="checkbox" id="param-${param.name}"
          data-param="${param.name}" data-type="bool" ${checked}${disabled}>
        <span class="toggle-slider"></span>
      </label>
    </div>
  `;
}

/**
 * @param {AdvancedParamOption} opt
 * @param {*} value - The control's current value
 * @returns {string} HTML string
 */
function renderOption(opt, value) {
  const selected = opt.value === value ? 'selected' : '';
  return `<option value="${escapeHtml(opt.value)}" ${selected}>${escapeHtml(opt.label)}</option>`;
}

/**
 * Render a select dropdown for select params.
 * @param {AdvancedParam} param - Parameter descriptor
 * @param {*} value - Current value
 * @param {string} disabled - '' or ' disabled'
 * @returns {string} HTML string
 */
function renderSelect(param, value, disabled) {
  const optionsHtml = (param.options || []).map((opt) => renderOption(opt, value)).join('');

  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}">${escapeHtml(param.label)}</label>
      <select id="param-${param.name}" class="select"
        data-param="${param.name}" data-type="select"${disabled}>
        ${optionsHtml}
      </select>
    </div>
  `;
}

/**
 * Render a date input control.
 * @param {AdvancedParam} param - Parameter descriptor
 * @param {*} value - Current value
 * @param {string} disabled - '' or ' disabled'
 * @returns {string} HTML string
 */
function renderDateInput(param, value, disabled) {
  const val = value || '';
  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}">${escapeHtml(param.label)}</label>
      <input type="date" id="param-${param.name}" class="input"
        data-param="${param.name}" data-type="date" value="${escapeHtml(val)}"${disabled}>
    </div>
  `;
}

/**
 * Render a text input control.
 * @param {AdvancedParam} param - Parameter descriptor
 * @param {*} value - Current value
 * @param {string} disabled - '' or ' disabled'
 * @returns {string} HTML string
 */
function renderTextInput(param, value, disabled) {
  const val = value || '';
  const placeholder = param.placeholder || '';
  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}">${escapeHtml(param.label)}</label>
      <input type="text" id="param-${param.name}" class="input"
        data-param="${param.name}" data-type="text"
        value="${escapeHtml(val)}" placeholder="${escapeHtml(placeholder)}"${disabled}>
    </div>
  `;
}

/**
 * Render a dynamic select that fetches options from an endpoint. Until they
 * arrive, a set value is kept as its own option so the control shows it.
 * @param {AdvancedParam} param - Parameter descriptor
 * @param {*} value - Current value
 * @param {string} disabled - '' or ' disabled'
 * @returns {string} HTML string
 */
function renderDynamicSelect(param, value, disabled) {
  const pending = value
    ? `<option value="${escapeHtml(value)}" selected>${escapeHtml(value)}</option>`
    : '';
  return `
    <div class="input-group">
      <label class="input-label" for="param-${param.name}">${escapeHtml(param.label)}</label>
      <select id="param-${param.name}" class="select"
        data-param="${param.name}" data-type="dynamic_select"
        data-endpoint="${escapeHtml(param.options_endpoint || '')}"${disabled}>
        <option value="">All</option>${pending}
      </select>
    </div>
  `;
}

/**
 * Fetch and populate options for dynamic_select elements.
 * @param {HTMLElement} container - Container to search for dynamic_select elements
 * @param {(name: string) => *} valueOf - The value each control currently shows
 */
async function loadDynamicSelectOptions(container, valueOf) {
  const selects = /** @type {NodeListOf<HTMLSelectElement>} */ (
    container.querySelectorAll('select[data-type="dynamic_select"]')
  );
  for (const select of selects) {
    const endpoint = select.dataset.endpoint;
    if (!endpoint) continue;

    const currentValue = valueOf(select.dataset.param || '') || '';

    try {
      /** @type {AdvancedParamOption[]} */
      let options;
      if (dynamicOptionsCache[endpoint]) {
        options = dynamicOptionsCache[endpoint];
      } else {
        const response = await fetch(endpoint);
        if (!response.ok) continue;
        const data = await response.json();
        options = data.options || [];
        dynamicOptionsCache[endpoint] = options;
      }

      // Preserve the "All" option and append fetched options
      select.innerHTML =
        '<option value="">All</option>' +
        options.map((opt) => renderOption(opt, currentValue)).join('');
    } catch (err) {
      console.warn(`Failed to load options from ${endpoint}:`, err);
    }
  }
}

/**
 * Attach change listeners to parameter controls. Disabled (locked) controls
 * get none, and `report` refuses locked names besides.
 * @param {HTMLElement} container - Container holding the rendered parameter controls
 * @param {(name: string, value: *) => void} report - Receives a name and its new value (null = cleared)
 */
function attachParamListeners(container, report) {
  /**
   * @param {string} selector
   * @param {string} event
   * @param {(el: HTMLInputElement & HTMLSelectElement) => *} read
   */
  const bind = (selector, event, read) => {
    const controls = /** @type {NodeListOf<HTMLInputElement & HTMLSelectElement>} */ (
      container.querySelectorAll(selector)
    );
    controls.forEach((ctl) => {
      if (ctl.disabled) return;
      ctl.addEventListener(event, () => {
        const name = ctl.dataset.param;
        if (name) report(name, read(ctl));
      });
    });
  };

  bind('input[type="range"][data-param]', 'input', (slider) => {
    const type = slider.dataset.type;
    const val = type === 'float' ? parseFloat(slider.value) : parseInt(slider.value, 10);
    const display = container.querySelector(`#param-${slider.dataset.param}-value`);
    if (display) display.textContent = type === 'float' ? val.toFixed(2) : String(val);
    return val;
  });
  bind('input[type="checkbox"][data-param]', 'change', (toggle) => toggle.checked);
  // A cleared select/date/text control means "no value": the name is removed.
  bind('select[data-param]', 'change', (select) => select.value || null);
  bind('input[type="date"][data-param]', 'change', (input) => input.value || null);
  bind('input[type="text"][data-param]', 'input', (input) => input.value.trim() || null);
}
