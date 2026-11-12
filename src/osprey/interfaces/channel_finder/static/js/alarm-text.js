// @ts-check
/**
 * OSPREY Channel Finder — alarm configuration as text
 *
 * The guidance and displays of an alarm configuration arrive in whatever
 * shape the alarm server keeps them: a list of `{title, details}` mappings
 * (Phoebus), a string, or anything else. {@link configEntries} turns each
 * into title + details text, so nothing prints as `[object Object]`.
 */

/** @typedef {{ title: string, details: string }} ConfigEntry */

/**
 * A field as text: a string as is, nothing as empty, anything else as JSON.
 * @param {unknown} value
 * @returns {string}
 */
function text(value) {
  if (value === null || value === undefined) return '';
  return typeof value === 'string' ? value : JSON.stringify(value);
}

/**
 * One entry: a mapping gives its title and details, anything else is details.
 * @param {unknown} value
 * @returns {ConfigEntry}
 */
function entry(value) {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    const fields = /** @type {Record<string, unknown>} */ (value);
    if ('title' in fields || 'details' in fields) {
      return { title: text(fields.title), details: text(fields.details) };
    }
  }
  return { title: '', details: text(value) };
}

/**
 * The entries of a guidance or displays value: one per list item, one for a
 * string or any other value, none for null.
 * @param {unknown} value
 * @returns {ConfigEntry[]}
 */
export function configEntries(value) {
  if (value === null || value === undefined) return [];
  return Array.isArray(value) ? value.map(entry) : [entry(value)];
}
