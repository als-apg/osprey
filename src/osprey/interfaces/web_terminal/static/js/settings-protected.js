// @ts-check
/* OSPREY Web Terminal — Protected Settings Fields
 *
 * The leaves of the config the build owns: GET /api/config lists them as
 * dotted keys in `protected`, and the write gates refuse a change to any of
 * them. The settings form renders them as disabled controls with one note —
 * on the section when every leaf under it is protected, else on the row — and
 * never puts them in an Apply body. settings.js owns rendering and writing;
 * this module owns the read-only marking.
 */

export const PROTECTED_NOTE = "Set in the profile's config: block — re-run osprey build to change.";

/** @param {any} value @returns {boolean} */
function isBlock(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * The dotted keys of every leaf the form renders a control for.
 * @param {Record<string, any>} obj
 * @param {string} prefix
 * @returns {string[]}
 */
function leafKeys(obj, prefix) {
  return Object.entries(obj).flatMap(([key, value]) => (
    isBlock(value) ? leafKeys(value, `${prefix}.${key}`) : [`${prefix}.${key}`]
  ));
}

/** @returns {HTMLElement} */
export function protectedNote() {
  const note = document.createElement('span');
  note.className = 'settings-field-note';
  note.textContent = PROTECTED_NOTE;
  return note;
}

/**
 * True when the section renders at least one leaf and every one is protected.
 * @param {Record<string, any>} section
 * @param {string} sectionKey
 * @param {Set<string>} protectedKeys
 * @returns {boolean}
 */
export function isSectionProtected(section, sectionKey, protectedKeys) {
  const leaves = leafKeys(section, sectionKey);
  return leaves.length > 0 && leaves.every((key) => protectedKeys.has(key));
}

/**
 * Mark a rendered field row read-only: grey row, disabled control (the inner
 * checkbox of a toggle), and the note unless its section already carries it.
 * The control keeps its `data-key`, so reveal and lookup still find it.
 * @param {HTMLElement} field  the `.settings-field` row
 * @param {HTMLElement} input  what createInputForValue returned
 * @param {boolean} withNote
 */
export function markProtected(field, input, withNote) {
  field.classList.add('settings-field-protected');
  const control = /** @type {HTMLInputElement|HTMLSelectElement} */ (
    input.matches('[data-key]') ? input : input.querySelector('[data-key]')
  );
  control.disabled = true;
  if (withNote) field.appendChild(protectedNote());
}
