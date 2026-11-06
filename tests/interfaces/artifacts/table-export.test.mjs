// @ts-check
/**
 * Unit tests for a served table as text (table-export.js), behind the table
 * strip's Copy table and Export CSV:
 *   npx vitest run tests/interfaces/artifacts/table-export.test.mjs
 *
 * The table page is sandboxed, so both actions fetch the artifact's file and
 * read its first table; `fetch`, the clipboard and the object URLs are
 * stubbed here.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  copyTable,
  csvName,
  exportCsv,
  tableRows,
  toCsv,
  toTsv,
} from '../../../src/osprey/interfaces/artifacts/static/js/table-export.js';

/** A pandas DataFrame as serialize_object() saves it: two header rows. */
const PANDAS = `<html><head><style>table { color: red; }</style></head><body>
<table border="1" class="dataframe artifact-table">
  <thead>
    <tr style="text-align: right;"><th></th><th>s (m)</th><th>x (mm)</th><th>in bump</th></tr>
    <tr><th>BPM</th><th></th><th></th><th></th></tr>
  </thead>
  <tbody>
    <tr><th>70</th><td>56.8</td><td>0.014</td><td>no</td></tr>
    <tr><th>74</th><td>60.1</td><td>0.030</td><td>yes</td></tr>
  </tbody>
</table></body></html>`;

const ROWS = [
  ['', 's (m)', 'x (mm)', 'in bump'],
  ['BPM', '', '', ''],
  ['70', '56.8', '0.014', 'no'],
  ['74', '60.1', '0.030', 'yes'],
];

const artifact = { id: 't 1', filename: 'orbit table.html', artifact_type: 'table_html' };

/**
 * Serve `body` for every fetch.
 * @param {string} body
 * @param {boolean} [ok]
 */
function serve(body, ok = true) {
  const fetchMock = vi.fn(async () => ({ ok, text: async () => body }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
  document.body.replaceChildren();
});

describe('reading the table', () => {
  test('the first table, row by row, header rows included', () => {
    expect(tableRows(PANDAS)).toEqual(ROWS);
  });

  test('a spanning cell keeps the columns in line, and white space collapses', () => {
    const html =
      '<table><tr><th colspan="2">Orbit\n   x</th><th>y</th></tr>' +
      '<tr><td>1</td><td>2</td><td>3</td></tr></table>';
    expect(tableRows(html)).toEqual([
      ['Orbit x', '', 'y'],
      ['1', '2', '3'],
    ]);
  });

  test('a page without a table has no rows, and runs no script', () => {
    expect(tableRows('<p>No table</p><script>window.__ran = true</script>')).toEqual([]);
    expect(/** @type {any} */ (window).__ran).toBeUndefined();
  });
});

describe('writing it out', () => {
  test('tab-separated text, one line per row', () => {
    expect(toTsv(ROWS.slice(2))).toBe('70\t56.8\t0.014\tno\n74\t60.1\t0.030\tyes');
  });

  test('CSV quotes a field with a comma, a quote or a line break', () => {
    expect(
      toCsv([
        ['a,b', 'say "hi"', 'plain'],
        ['line\nbreak', '', 'x'],
      ])
    ).toBe('"a,b","say ""hi""",plain\r\n"line\nbreak",,x\r\n');
  });

  test("the CSV file is named after the artifact's file", () => {
    expect(csvName(artifact)).toBe('orbit table.csv');
    expect(csvName({ title: 'Orbit' })).toBe('Orbit.csv');
    expect(csvName({})).toBe('table.csv');
  });
});

describe('Copy table', () => {
  test("fetches the artifact's file and copies its table as tab-separated text", async () => {
    const fetchMock = serve(PANDAS);
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });

    expect(await copyTable(artifact)).toBe(true);

    expect(fetchMock).toHaveBeenCalledWith('/files/t%201/orbit%20table.html');
    expect(writeText).toHaveBeenCalledWith(toTsv(ROWS));
  });

  test('copies nothing when the file holds no table or cannot be read', async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });

    serve('<p>none</p>');
    expect(await copyTable(artifact)).toBe(false);
    serve(PANDAS, false);
    expect(await copyTable(artifact)).toBe(false);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('offline');
      })
    );
    expect(await copyTable(artifact)).toBe(false);
    expect(writeText).not.toHaveBeenCalled();
  });
});

describe('Export CSV', () => {
  test('downloads the table as a CSV file and leaves nothing behind', async () => {
    vi.useFakeTimers();
    serve(PANDAS);
    const create = vi.fn(() => 'blob:csv');
    const revoke = vi.fn();
    vi.stubGlobal(
      'URL',
      Object.assign(class extends URL {}, { createObjectURL: create, revokeObjectURL: revoke })
    );
    /** @type {HTMLAnchorElement[]} */
    const clicked = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(
      /** @this {HTMLAnchorElement} */ function () {
        clicked.push(this);
      }
    );

    expect(await exportCsv(artifact)).toBe(true);

    expect(clicked).toHaveLength(1);
    expect(clicked[0].download).toBe('orbit table.csv');
    expect(clicked[0].getAttribute('href')).toBe('blob:csv');
    const [[blob]] = /** @type {any[][]} */ (create.mock.calls);
    expect(blob.type).toBe('text/csv');
    expect(await blob.text()).toBe(toCsv(ROWS));
    expect(document.querySelector('a')).toBeNull();
    vi.advanceTimersByTime(1000);
    expect(revoke).toHaveBeenCalledWith('blob:csv');
  });

  test('downloads nothing for a file without a table', async () => {
    serve('<p>none</p>');
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click');

    expect(await exportCsv(artifact)).toBe(false);
    expect(click).not.toHaveBeenCalled();
  });
});
