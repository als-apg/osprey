// @ts-check
/**
 * ARIEL Entry Fields
 *
 * Renders the entry fields a facility adapter declares (`publish-info`'s
 * `entry_fields`) into the create form, one section per descriptor `section`,
 * placed in the mount after the "Metadata" section. A declaration named
 * `logbook` or `shift` replaces that built-in input in place and keeps its form
 * name, so the built-in submit path reads the same value.
 *
 * Public surface:
 *   - renderEntryFields(fields, mount?)  render (or clear) the declared fields
 *   - entryFieldsRendered()              settles once fields are rendered, or
 *                                        known to be none
 *   - collectEntryFieldValues()          the declared values, native-typed
 *   - markEntryFieldError(field, msg)    mark one declared input as invalid
 */

import { entriesApi } from './api.js';

/** Id of the element the declared sections render into. */
export const ENTRY_FIELDS_MOUNT_ID = 'entry-fields';

/** Declared names that replace a built-in create-form input, mapped to its id. */
const BUILT_IN_OVERRIDES = /** @type {Record<string, string>} */ ({
  logbook: 'entry-logbook',
  shift: 'entry-shift',
});

/**
 * @typedef {Object} EntryFieldDescriptor
 * @property {string} name
 * @property {string} label
 * @property {string} [description]
 * @property {string} type - "text"|"int"|"float"|"bool"|"date"|"select"|"dynamic_select"
 * @property {any} [default]
 * @property {string} [section]
 * @property {number} [min]
 * @property {number} [max]
 * @property {number} [step]
 * @property {Array<{value: string, label: string}>} [options]
 * @property {string} [placeholder]
 * @property {string} [options_endpoint]
 * @property {boolean} [required]
 * @property {string[]} [depends_on]
 */

/**
 * @typedef {Object} RenderedField
 * @property {EntryFieldDescriptor} descriptor
 * @property {HTMLInputElement|HTMLSelectElement} control
 * @property {number} requestSeq - Sequence of the latest options request.
 */

/** @type {RenderedField[]} */
let rendered = [];

/** Built-in input groups currently replaced by a declared field. */
/** @type {Array<{original: Element, replacement: Element}>} */
let overrides = [];

/** Sections this module added to the mount. */
/** @type {Element[]} */
let sections = [];

/** @type {(value?: void) => void} */
let resolveRendered = () => {};
let renderedSettled = false;
/** @type {Promise<void>} */
let renderedPromise = newRenderedPromise();

/** @returns {Promise<void>} */
function newRenderedPromise() {
  renderedSettled = false;
  return new Promise((resolve) => {
    resolveRendered = () => {
      renderedSettled = true;
      resolve();
    };
  });
}

/**
 * A promise that settles once the declared fields are rendered — including the
 * first load of every dynamic select's choices — or once the form is known to
 * declare none (`renderEntryFields([])`).
 * @returns {Promise<void>}
 */
export function entryFieldsRendered() {
  return renderedPromise;
}

/**
 * Render the declared entry fields, replacing whatever an earlier call rendered.
 *
 * Call with `[]` when the facility declares none, or when the declarations
 * could not be read: that restores the form and settles the rendered-promise.
 * @param {EntryFieldDescriptor[]} fields - Declarations in form order
 * @param {HTMLElement|null} [mount] - Where the sections go; defaults to `#entry-fields`
 * @returns {Promise<void>} Settles when the fields and their first choices are in place
 */
export async function renderEntryFields(fields, mount) {
  if (renderedSettled) renderedPromise = newRenderedPromise();
  clearRendered();

  /** @type {Map<string, HTMLElement>} */
  const sectionRows = new Map();
  for (const descriptor of fields || []) {
    const { group, control } = buildInputGroup(descriptor);
    rendered.push({ descriptor, control, requestSeq: 0 });
    const overrideId = BUILT_IN_OVERRIDES[descriptor.name];
    const builtIn = overrideId ? document.getElementById(overrideId) : null;
    const builtInGroup = builtIn?.closest('.input-group') || builtIn;
    if (overrideId && builtIn && builtInGroup) {
      control.id = overrideId;
      control.name = descriptor.name;
      const label = group.querySelector('label');
      if (label) label.htmlFor = overrideId;
      builtInGroup.replaceWith(group);
      overrides.push({ original: builtInGroup, replacement: group });
      continue;
    }
    const sectionName = descriptor.section || 'General';
    let row = sectionRows.get(sectionName);
    if (!row) {
      const target = mount || resolveMount();
      if (!target) continue;
      row = appendSection(target, sectionName);
      sectionRows.set(sectionName, row);
    }
    row.appendChild(group);
  }

  wireDependencies();
  const firstLoads = rendered
    .filter((f) => f.descriptor.type === 'dynamic_select')
    .map((f) => loadOptions(f));
  await Promise.allSettled(firstLoads);
  resolveRendered();
}

/**
 * The declared values, keyed by field name. Numbers and booleans come back
 * native; blank inputs are left out.
 * @returns {Record<string, string|number|boolean>}
 */
export function collectEntryFieldValues() {
  /** @type {Record<string, string|number|boolean>} */
  const values = {};
  for (const { descriptor, control } of rendered) {
    const value = readValue(descriptor, control);
    if (value !== undefined) values[descriptor.name] = value;
  }
  return values;
}

/**
 * Mark one declared input as invalid, show `message` beside it and focus it.
 * The mark clears when the operator changes the input.
 * @param {string} field - Name of the declared field
 * @param {string} message - What is wrong
 * @returns {boolean} Whether a rendered input carries that name
 */
export function markEntryFieldError(field, message) {
  const entry = rendered.find((f) => f.descriptor.name === field);
  if (!entry) return false;
  const { control } = entry;
  clearError(control);
  control.setAttribute('aria-invalid', 'true');
  control.style.borderColor = 'var(--color-error)';
  const note = document.createElement('div');
  note.className = 'text-xs text-error';
  note.dataset.entryFieldError = field;
  note.textContent = message;
  control.insertAdjacentElement('afterend', note);
  control.focus();
  return true;
}

/** Remove everything an earlier render added and restore replaced built-ins. */
function clearRendered() {
  for (const { original, replacement } of overrides) replacement.replaceWith(original);
  for (const section of sections) section.remove();
  overrides = [];
  sections = [];
  rendered = [];
}

/**
 * The mount element; created after the "Metadata" section when absent.
 * @returns {HTMLElement|null}
 */
function resolveMount() {
  const existing = document.getElementById(ENTRY_FIELDS_MOUNT_ID);
  if (existing) return existing;
  const titles = document.querySelectorAll('.form-section-title');
  const metadata = [...titles].find((t) => t.textContent?.trim() === 'Metadata');
  const section = metadata?.closest('.form-section');
  if (!section) return null;
  const mount = document.createElement('div');
  mount.id = ENTRY_FIELDS_MOUNT_ID;
  section.insertAdjacentElement('afterend', mount);
  return mount;
}

/**
 * Append one titled section to `mount` and return its row.
 * @param {HTMLElement} mount
 * @param {string} title
 * @returns {HTMLElement} The row fields go into
 */
function appendSection(mount, title) {
  const section = document.createElement('div');
  section.className = 'form-section';
  section.dataset.entryFieldSection = title;
  const heading = document.createElement('div');
  heading.className = 'form-section-title';
  heading.textContent = title;
  const row = document.createElement('div');
  row.className = 'form-row';
  section.append(heading, row);
  mount.appendChild(section);
  sections.push(section);
  return row;
}

/**
 * Build the labelled input group of one declaration.
 * @param {EntryFieldDescriptor} descriptor
 * @returns {{group: HTMLElement, control: HTMLInputElement|HTMLSelectElement}}
 */
function buildInputGroup(descriptor) {
  const control = buildControl(descriptor);
  control.id = `entry-field-${descriptor.name}`;
  control.dataset.entryField = descriptor.name;
  control.dataset.type = descriptor.type;
  if (descriptor.description) control.title = descriptor.description;
  if (descriptor.required) {
    control.setAttribute('aria-required', 'true');
    if (descriptor.type !== 'bool') control.required = true;
  }
  control.addEventListener('change', () => clearError(control));

  const label = document.createElement('label');
  label.className = 'input-label';
  label.htmlFor = control.id;
  label.textContent = descriptor.required ? `${descriptor.label} *` : descriptor.label;

  const group = document.createElement('div');
  group.className = 'input-group';
  group.dataset.entryFieldGroup = descriptor.name;
  group.append(label, control);
  return { group, control };
}

/**
 * The plain input for one declaration, with its default pre-filled.
 * @param {EntryFieldDescriptor} descriptor
 * @returns {HTMLInputElement|HTMLSelectElement}
 */
function buildControl(descriptor) {
  const fallback = descriptor.default;
  const hasDefault = fallback !== null && fallback !== undefined;
  switch (descriptor.type) {
    case 'select':
    case 'dynamic_select': {
      const select = document.createElement('select');
      select.className = 'input';
      setOptions(select, descriptor.options || [], hasDefault ? String(fallback) : '');
      return select;
    }
    case 'bool': {
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = fallback === true;
      return input;
    }
    case 'int':
    case 'float': {
      const input = document.createElement('input');
      input.type = 'number';
      input.className = 'input';
      if (descriptor.min != null) input.min = String(descriptor.min);
      if (descriptor.max != null) input.max = String(descriptor.max);
      input.step =
        descriptor.step != null ? String(descriptor.step) : descriptor.type === 'int' ? '1' : 'any';
      if (hasDefault) input.value = String(fallback);
      return input;
    }
    default: {
      const input = document.createElement('input');
      input.type = descriptor.type === 'date' ? 'date' : 'text';
      input.className = 'input';
      if (descriptor.placeholder) input.placeholder = descriptor.placeholder;
      if (hasDefault) input.value = String(fallback);
      return input;
    }
  }
}

/**
 * Replace a select's choices, after a blank first choice, selecting `value`
 * when it is among them.
 * @param {HTMLSelectElement} select
 * @param {Array<{value: string, label: string}>} options
 * @param {string} value
 */
function setOptions(select, options, value) {
  select.replaceChildren(makeOption('', ''));
  for (const opt of options) {
    select.appendChild(makeOption(String(opt.value), String(opt.label ?? opt.value)));
  }
  select.value = options.some((o) => String(o.value) === value) ? value : '';
}

/**
 * One `<option>`, its label set as text.
 * @param {string} value
 * @param {string} label
 * @returns {HTMLOptionElement}
 */
function makeOption(value, label) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  return option;
}

/** Refetch a dynamic select's choices whenever one of its parents changes. */
function wireDependencies() {
  for (const child of rendered) {
    for (const parentName of child.descriptor.depends_on || []) {
      const parent = rendered.find((f) => f.descriptor.name === parentName);
      parent?.control.addEventListener('change', () => {
        loadOptions(child);
      });
    }
  }
}

/**
 * Load a dynamic select's choices with its parents' current values. A response
 * that arrives after a newer request was issued is dropped.
 * @param {RenderedField} field
 * @returns {Promise<void>}
 */
async function loadOptions(field) {
  const { descriptor } = field;
  const select = /** @type {HTMLSelectElement} */ (field.control);
  if (!descriptor.options_endpoint) return;

  /** @type {Record<string, string|number|boolean>} */
  const params = {};
  for (const parentName of descriptor.depends_on || []) {
    const parent = rendered.find((f) => f.descriptor.name === parentName);
    const value = parent ? readValue(parent.descriptor, parent.control) : undefined;
    if (value !== undefined) params[parentName] = value;
  }

  const seq = ++field.requestSeq;
  try {
    const options = await entriesApi.getEntryFieldOptions(descriptor.options_endpoint, params);
    if (seq !== field.requestSeq) return;
    const fallback = descriptor.default;
    const wanted = select.value || (fallback != null ? String(fallback) : '');
    setOptions(select, options, wanted);
  } catch (err) {
    if (seq !== field.requestSeq) return;
    console.warn(`Failed to load choices for entry field '${descriptor.name}':`, err);
  }
}

/**
 * One control's value as the server expects it, or undefined when blank.
 * @param {EntryFieldDescriptor} descriptor
 * @param {HTMLInputElement|HTMLSelectElement} control
 * @returns {string|number|boolean|undefined}
 */
function readValue(descriptor, control) {
  if (descriptor.type === 'bool') return /** @type {HTMLInputElement} */ (control).checked;
  const raw = control.value;
  if (raw === '') return undefined;
  if (descriptor.type === 'int' || descriptor.type === 'float') {
    const num = Number(raw);
    return Number.isFinite(num) ? num : raw;
  }
  return raw;
}

/**
 * Remove an error mark from a control.
 * @param {HTMLInputElement|HTMLSelectElement} control
 */
function clearError(control) {
  control.removeAttribute('aria-invalid');
  control.style.borderColor = '';
  const next = control.nextElementSibling;
  if (next instanceof HTMLElement && next.dataset.entryFieldError !== undefined) next.remove();
}
