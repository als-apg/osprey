// @ts-check
/**
 * OSPREY Artifact Gallery — a served table as text: Copy table and Export CSV.
 *
 * A `table_html` artifact is a sandboxed page on an opaque origin, so the
 * gallery cannot read the table in its frame. These two actions of the
 * table's content strip (artifact-strip.js) fetch the artifact's file
 * instead, parse it inert (`DOMParser` runs no script) and read its first
 * table: Copy table puts it on the clipboard as tab-separated text, for the
 * logbook or a spreadsheet; Export CSV downloads it as a `.csv` file. A
 * header cell that spans columns is followed by empty cells, so the columns
 * stay in line.
 *
 * @module table-export
 */

import { copyText } from '/design-system/js/clipboard.js';
import { fileUrl } from './state.js';

/**
 * The cells of the first table in an HTML page, row by row, as plain text.
 * @param {string} html
 * @returns {string[][]} no rows when the page holds no table
 */
export function tableRows(html) {
  const table = new DOMParser().parseFromString(html, 'text/html').querySelector('table');
  if (!table) return [];
  return Array.from(table.rows, (row) =>
    Array.from(row.cells).flatMap((cell) => {
      const text = (cell.textContent ?? '').replace(/\s+/g, ' ').trim();
      return [text, ...Array(Math.max(0, cell.colSpan - 1)).fill('')];
    })
  );
}

/**
 * @param {string[][]} rows
 * @returns {string} tab-separated text, one line per row
 */
export function toTsv(rows) {
  return rows.map((row) => row.join('\t')).join('\n');
}

/**
 * @param {string[][]} rows
 * @returns {string} RFC 4180 CSV: a field with a comma, a quote or a line
 *   break is quoted
 */
export function toCsv(rows) {
  /** @param {string} value */
  const field = (value) => (/[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value);
  return rows.map((row) => row.map(field).join(',')).join('\r\n') + '\r\n';
}

/**
 * The CSV file name for an artifact: its file name with `.csv` for its
 * extension.
 * @param {{filename?: string, title?: string}} a
 * @returns {string}
 */
export function csvName(a) {
  const stem = (a.filename || a.title || 'table').replace(/\.[^./]*$/, '');
  return `${stem}.csv`;
}

/**
 * @param {any} a
 * @returns {Promise<string[][]>} the artifact's table, or no rows when it
 *   cannot be read
 */
async function readTable(a) {
  try {
    const resp = await fetch(fileUrl(a));
    return resp.ok ? tableRows(await resp.text()) : [];
  } catch {
    return [];
  }
}

/**
 * Copy table: the artifact's table on the clipboard, tab-separated.
 * @param {any} a
 * @returns {Promise<boolean>} whether it was copied
 */
export async function copyTable(a) {
  const rows = await readTable(a);
  return rows.length > 0 && copyText(toTsv(rows));
}

/**
 * Export CSV: download the artifact's table as a CSV file.
 * @param {any} a
 * @returns {Promise<boolean>} whether there was a table to download
 */
export async function exportCsv(a) {
  const rows = await readTable(a);
  if (!rows.length) return false;
  const url = URL.createObjectURL(new Blob([toCsv(rows)], { type: 'text/csv' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = csvName(a);
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  // Revoked once the browser has taken the download, not before.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}
