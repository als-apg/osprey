/**
 * Unit tests for the BLUESKY panel's CSV export.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/bluesky_web/csv-export.test.mjs
 *
 * Two layers:
 *  - The pure builders (`csvField`, `toCsv`, `exportFilename`) against literal
 *    values. These carry the whole correctness story: the CSV is an ANALYSIS
 *    artifact, so it must round-trip the wire values, not the display strings
 *    `formatCell` produces for the table.
 *  - `saveCsv` against a stubbed picker and a stubbed anchor, which is the only
 *    way to pin the three outcomes an operator can actually hit: saved through
 *    the OS dialog, cancelled at it, and saved to Downloads on a browser that
 *    has no dialog to offer.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  csvField,
  exportFilename,
  saveCsv,
  toCsv,
} from '../../../src/osprey/interfaces/bluesky_web/panels/bluesky/csv-export.js';

describe('csvField', () => {
  test('an absent value is an EMPTY field, never the table’s em dash', () => {
    // formatCell renders null as '—' because a human is reading it. A parser
    // is reading this, and '—' is a string where a gap belongs.
    expect(csvField(null)).toBe('');
    expect(csvField(undefined)).toBe('');
  });

  test('numbers keep full precision', () => {
    // formatCell rounds to 6 decimals for display. Rounding here would put
    // silently lossy values into a file someone fits a model to.
    expect(csvField(0.1234567891234)).toBe('0.1234567891234');
    expect(csvField(-2.5e-9)).toBe('-2.5e-9');
    expect(csvField(0)).toBe('0');
  });

  test('non-finite numbers are written as-is rather than blanked', () => {
    // A NaN in a run is a real reading that failed; erasing it to an empty
    // field would make it indistinguishable from a column that was never read.
    expect(csvField(NaN)).toBe('NaN');
    expect(csvField(Infinity)).toBe('Infinity');
  });

  test('plain strings pass through unquoted', () => {
    expect(csvField('BPM:72:POSITION:X')).toBe('BPM:72:POSITION:X');
    expect(csvField('')).toBe('');
  });

  test('separators, quotes and newlines force RFC-4180 quoting', () => {
    expect(csvField('a,b')).toBe('"a,b"');
    expect(csvField('say "hi"')).toBe('"say ""hi"""');
    expect(csvField('line1\nline2')).toBe('"line1\nline2"');
    expect(csvField('line1\r\nline2')).toBe('"line1\r\nline2"');
  });

  test('booleans and objects stringify rather than throw', () => {
    expect(csvField(true)).toBe('true');
    expect(csvField({ a: 1 })).toBe('[object Object]');
  });
});

describe('toCsv', () => {
  test('header row then data rows, CRLF-terminated', () => {
    const csv = toCsv(
      ['time', 'signal'],
      [
        [1, 10],
        [2, 20],
      ]
    );
    expect(csv).toBe('time,signal\r\n1,10\r\n2,20\r\n');
  });

  test('a column name needing quotes is quoted in the header too', () => {
    expect(toCsv(['a,b'], [[1]])).toBe('"a,b"\r\n1\r\n');
  });

  test('no rows still yields the header, so the file says what it is', () => {
    expect(toCsv(['time', 'signal'], [])).toBe('time,signal\r\n');
  });

  test('no columns yields an empty string, not a stray newline', () => {
    expect(toCsv([], [])).toBe('');
  });

  test('a short row is padded so columns stay aligned', () => {
    // The bridge should never serve a ragged window, but a row that lost a
    // field must not silently shift every later column left by one.
    expect(toCsv(['a', 'b', 'c'], [[1, 2]])).toBe('a,b,c\r\n1,2,\r\n');
  });
});

describe('exportFilename', () => {
  test('plan name and a short run id', () => {
    expect(exportFilename('grid_scan', '3f9a1c2e-7b44-4c1a-9f0e-1d2c3b4a5e6f')).toBe(
      'grid_scan-3f9a1c2e.csv'
    );
  });

  test('a missing plan name still produces a usable name', () => {
    expect(exportFilename(null, 'abcdef123456')).toBe('run-abcdef12.csv');
    expect(exportFilename('', 'abcdef123456')).toBe('run-abcdef12.csv');
  });

  test('path separators and spaces cannot escape into the filename', () => {
    // A plan name reaches this from the bridge. It must not be able to steer
    // where the file lands.
    expect(exportFilename('../../etc/hosts', 'abcdef123456')).toBe('etc-hosts-abcdef12.csv');
    expect(exportFilename('my scan v2', 'abcdef123456')).toBe('my-scan-v2-abcdef12.csv');
  });

  test('a short run id is used whole', () => {
    expect(exportFilename('orm', 'r1')).toBe('orm-r1.csv');
  });
});

describe('saveCsv', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  /** A File System Access handle that records what was written to it. */
  function fakeHandle() {
    /** @type {string[]} */
    const written = [];
    return {
      written,
      createWritable: async () => ({
        write: async (/** @type {string} */ text) => void written.push(text),
        close: async () => {},
      }),
    };
  }

  test('the native picker is used when the browser has one', async () => {
    const handle = fakeHandle();
    const picker = vi.fn(async () => handle);

    const result = await saveCsv('run.csv', 'a,b\r\n1,2\r\n', { picker });

    expect(result).toEqual({ saved: true, method: 'picker' });
    expect(handle.written).toEqual(['a,b\r\n1,2\r\n']);
    expect(picker).toHaveBeenCalledWith(
      expect.objectContaining({ suggestedName: 'run.csv' })
    );
  });

  test('cancelling the picker saves NOTHING — no silent download behind it', async () => {
    // The fallback path exists for browsers without a picker. Reaching it
    // because the operator said "no" would drop an unwanted file in Downloads
    // every time they changed their mind.
    const abort = Object.assign(new Error('cancelled'), { name: 'AbortError' });
    const picker = vi.fn(async () => {
      throw abort;
    });
    const anchor = { href: '', download: '', click: vi.fn(), remove: vi.fn() };
    vi.spyOn(document, 'createElement').mockReturnValue(/** @type {any} */ (anchor));

    const result = await saveCsv('run.csv', 'a\r\n', { picker });

    expect(result).toEqual({ saved: false, method: 'picker' });
    expect(anchor.click).not.toHaveBeenCalled();
  });

  test('no picker falls back to a download', async () => {
    const anchor = { href: '', download: '', click: vi.fn(), remove: vi.fn() };
    vi.spyOn(document, 'createElement').mockReturnValue(/** @type {any} */ (anchor));

    const result = await saveCsv('run.csv', 'a,b\r\n', { picker: undefined });

    expect(result).toEqual({ saved: true, method: 'download' });
    expect(anchor.download).toBe('run.csv');
    expect(anchor.click).toHaveBeenCalledOnce();
  });

  test('a picker that is blocked rather than cancelled falls back to a download', async () => {
    // An embedded panel can be denied the picker outright (SecurityError).
    // That is an environment limit, not an operator decision, so the export
    // must still produce the file.
    const picker = vi.fn(async () => {
      throw Object.assign(new Error('blocked'), { name: 'SecurityError' });
    });
    const anchor = { href: '', download: '', click: vi.fn(), remove: vi.fn() };
    vi.spyOn(document, 'createElement').mockReturnValue(/** @type {any} */ (anchor));

    const result = await saveCsv('run.csv', 'a,b\r\n', { picker });

    expect(result).toEqual({ saved: true, method: 'download' });
    expect(anchor.click).toHaveBeenCalledOnce();
  });
});
