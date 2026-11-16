// @ts-check
/**
 * OSPREY Channel Finder — a value read, as text
 *
 * {@link formatValue} turns one row of the values route into the text a cell
 * shows, its unit, and the hover title: "no answer", the reason a read did
 * not happen, a number rounded to 6 significant digits (or to the record's
 * precision when the row carries one), an enum as its label with the index on
 * hover, text as is, an array or object as JSON.
 */

/**
 * A row of the values route; `units`, `precision` and `enum_label` are read
 * when present.
 * @typedef {import('./directory-api.js').ValueRow} FormattableRow
 */

/** @typedef {{ text: string, unit: string, title: string }} FormattedValue */

/** Significant digits a number is shown with when the row has no precision. */
const SIGNIFICANT = 6;

/**
 * A number as text: to `precision` places when given, else a non-integer to
 * at most {@link SIGNIFICANT} significant digits with trailing zeros dropped.
 * An integer is a count and is shown whole.
 * @param {number} value
 * @param {unknown} precision
 * @returns {string}
 */
function numberText(value, precision) {
  if (!Number.isFinite(value)) return String(value);
  if (typeof precision === 'number' && Number.isInteger(precision) && precision >= 0) {
    return value.toFixed(Math.min(precision, 100));
  }
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toPrecision(SIGNIFICANT)));
}

/**
 * @param {FormattableRow} row
 * @returns {FormattedValue}
 */
export function formatValue(row) {
  const unit = typeof row.units === 'string' ? row.units : '';
  if (row.outcome === 'no_answer') return { text: 'no answer', unit: '', title: row.reason ?? '' };
  if (row.outcome !== 'found') {
    return { text: row.reason || 'no answer', unit: '', title: row.reason ?? '' };
  }
  const { value } = row;
  if (typeof row.enum_label === 'string') {
    return { text: row.enum_label, unit: '', title: String(value) };
  }
  if (typeof value === 'number') {
    const text = numberText(value, row.precision);
    return { text, unit, title: text === String(value) ? '' : String(value) };
  }
  if (value === null || value === undefined) return { text: '', unit, title: '' };
  const text = typeof value === 'object' ? JSON.stringify(value) : String(value);
  return { text, unit, title: '' };
}
