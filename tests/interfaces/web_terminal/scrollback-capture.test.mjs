/**
 * Unit tests for the terminal scrollback capture (scrollback-capture.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/scrollback-capture.test.mjs
 *
 * captureScrollback() walks an xterm buffer duck-type: rows 0..length-1, each
 * exposing `isWrapped` and `translateToString(trim)`. A row whose `isWrapped`
 * is true is a continuation of the row above it, so the capture rejoins the
 * hard wraps into one logical line before applying its caps. Both caps keep
 * the NEWEST content: the tail of the buffer is what a bug report needs.
 *
 * The module never imports xterm — the fakes below are the whole contract,
 * which is exactly what makes it testable without a browser.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe } from 'vitest';

import { captureScrollback } from '../../../src/osprey/interfaces/web_terminal/static/js/scrollback-capture.js';

const MARKER = '[scrollback truncated]';

/** UTF-8 byte length, the unit the byte cap is measured in. */
/** @param {string} text @returns {number} */
function bytes(text) {
  return new TextEncoder().encode(text).length;
}

/**
 * Build a fake xterm terminal over a list of physical rows.
 *
 * Mirrors the real API shapes: `term.buffer.active.length`, `getLine(i)`, and
 * `line.translateToString(trim)` where trim=true drops trailing whitespace —
 * xterm pads every row out to the terminal width with spaces, so untrimmed
 * rows come back full-width.
 *
 * @param {Array<{text: string, wrapped?: boolean}>} rows
 * @param {number} [width] - column count each row is padded out to
 */
function fakeTerm(rows, width = 80) {
  return {
    buffer: {
      active: {
        length: rows.length,
        /** @param {number} i */
        getLine(i) {
          const row = rows[i];
          if (!row) return undefined;
          const padded = row.text.padEnd(width, ' ');
          return {
            isWrapped: row.wrapped === true,
            /** @param {boolean} [trim] */
            translateToString(trim) {
              return trim === true ? padded.replace(/\s+$/, '') : padded;
            },
          };
        },
      },
    },
  };
}

/** A fake whose rows are plain unwrapped lines. @param {string[]} texts */
function fakeTermFromLines(texts) {
  return fakeTerm(texts.map((text) => ({ text })));
}

describe('logical line reconstruction', () => {
  test('joins isWrapped continuation rows into one logical line', () => {
    const term = fakeTerm([
      { text: 'first line' },
      { text: 'a wrapped command that ran' },
      { text: ' off the right edge', wrapped: true },
      { text: ' and kept going', wrapped: true },
      { text: 'last line' },
    ]);

    const out = captureScrollback(term);

    expect(out).toBe(
      'first line\n' +
        'a wrapped command that ran off the right edge and kept going\n' +
        'last line'
    );
    expect(out.split('\n')).toHaveLength(3);
  });

  test('a wrapped flag on the very first row starts a line rather than crashing', () => {
    const term = fakeTerm([
      { text: 'scrolled-in continuation', wrapped: true },
      { text: 'plain row' },
    ]);

    expect(captureScrollback(term)).toBe('scrolled-in continuation\nplain row');
  });

  test('trims the column padding xterm leaves on every row', () => {
    const term = fakeTerm([{ text: 'padded' }, { text: 'also padded' }], 120);

    const out = captureScrollback(term);

    expect(out).toBe('padded\nalso padded');
    expect(out).not.toMatch(/ \n/);
    expect(out.endsWith('padded')).toBe(true);
  });

  test('blank rows inside the capture survive as empty lines', () => {
    const term = fakeTermFromLines(['top', '', 'bottom']);

    expect(captureScrollback(term)).toBe('top\n\nbottom');
  });

  test('trailing blank rows below the cursor are dropped', () => {
    const term = fakeTermFromLines(['output', '', '   ', '']);

    expect(captureScrollback(term)).toBe('output');
  });

  test('an all-blank buffer captures nothing', () => {
    expect(captureScrollback(fakeTermFromLines(['', '   ', '']))).toBe('');
  });
});

describe('line cap', () => {
  test('a 3,000-line buffer keeps the newest 2,000 with a marker', () => {
    const lines = Array.from({ length: 3000 }, (_, i) => `line ${i}`);

    const out = captureScrollback(fakeTermFromLines(lines));
    const captured = out.split('\n');

    expect(captured[0]).toBe(MARKER);
    expect(captured).toHaveLength(2001);
    expect(captured[1]).toBe('line 1000');
    expect(captured[captured.length - 1]).toBe('line 2999');
    expect(out).not.toContain('line 999\n');
  });

  test('exactly at the cap keeps everything and adds no marker', () => {
    const lines = Array.from({ length: 2000 }, (_, i) => `line ${i}`);

    const out = captureScrollback(fakeTermFromLines(lines));

    expect(out).not.toContain(MARKER);
    expect(out.split('\n')).toHaveLength(2000);
    expect(out.startsWith('line 0\n')).toBe(true);
  });

  test('the cap counts logical lines, not physical rows', () => {
    const term = fakeTerm([
      { text: 'dropped' },
      { text: 'kept' },
      { text: ' and wrapped', wrapped: true },
    ]);

    const out = captureScrollback(term, { maxLines: 1 });

    expect(out).toBe(`${MARKER}\nkept and wrapped`);
  });

  test('maxLines is configurable', () => {
    const out = captureScrollback(fakeTermFromLines(['a', 'b', 'c', 'd']), { maxLines: 2 });

    expect(out).toBe(`${MARKER}\nc\nd`);
  });
});

describe('byte cap', () => {
  test('drops the oldest logical lines until the capture fits', () => {
    const lines = ['a', 'b', 'c', 'd'].map((ch) => ch.repeat(40));

    // 110 bytes holds the newest two lines (81) plus the marker line (23).
    const out = captureScrollback(fakeTermFromLines(lines), { maxBytes: 110 });

    expect(out).toBe(`${MARKER}\n${'c'.repeat(40)}\n${'d'.repeat(40)}`);
    expect(bytes(out)).toBeLessThanOrEqual(110);
  });

  test('the whole capture, marker included, stays inside the cap', () => {
    const lines = Array.from({ length: 500 }, (_, i) => `line ${i} ${'x'.repeat(50)}`);

    const out = captureScrollback(fakeTermFromLines(lines), { maxBytes: 2000 });

    expect(bytes(out)).toBeLessThanOrEqual(2000);
    expect(out.startsWith(`${MARKER}\n`)).toBe(true);
    expect(out.endsWith(`line 499 ${'x'.repeat(50)}`)).toBe(true);
  });

  test('bytes are UTF-8, not UTF-16 code units', () => {
    // Each line is 4 three-byte characters = 12 bytes, not 4.
    const term = fakeTermFromLines(['€€€€', '€€€€', '€€€€']);

    const out = captureScrollback(term, { maxBytes: bytes(`${MARKER}\n€€€€`) });

    expect(out).toBe(`${MARKER}\n€€€€`);
  });

  test('one oversized logical line is kept as its newest tail', () => {
    const huge = `${'a'.repeat(2_000_000)}THE-NEWEST-END`;

    const out = captureScrollback(fakeTerm([{ text: huge }], huge.length));

    expect(bytes(out)).toBeLessThanOrEqual(1_000_000);
    expect(out.startsWith(`${MARKER}\n`)).toBe(true);
    expect(out.endsWith('THE-NEWEST-END')).toBe(true);
    // The tail should fill the budget rather than collapsing to a stub.
    expect(bytes(out)).toBeGreaterThan(900_000);
  });

  test('an oversized tail is never cut mid-character', () => {
    // 20 x 3 bytes = 60 bytes of text; the budget forces a cut that would land
    // inside a multi-byte character if the slice were taken naively.
    const out = captureScrollback(fakeTerm([{ text: '€'.repeat(20) }], 20), { maxBytes: 40 });

    expect(out.startsWith(`${MARKER}\n`)).toBe(true);
    expect(out).not.toContain('�');
    expect(out.slice(MARKER.length + 1)).toBe('€'.repeat(5));
    expect(bytes(out)).toBeLessThanOrEqual(40);
  });

  test('a budget smaller than the marker yields the marker alone', () => {
    const out = captureScrollback(fakeTermFromLines(['content']), { maxBytes: 4 });

    expect(out).toBe(MARKER);
  });
});

describe('degenerate terminals', () => {
  test('an absent terminal captures nothing', () => {
    expect(captureScrollback(null)).toBe('');
    expect(captureScrollback(undefined)).toBe('');
  });

  test('a terminal without a buffer captures nothing', () => {
    expect(captureScrollback({})).toBe('');
    expect(captureScrollback({ buffer: {} })).toBe('');
  });

  test('an empty buffer captures nothing', () => {
    expect(captureScrollback(fakeTermFromLines([]))).toBe('');
  });

  test('a row the buffer refuses to hand back reads as an empty row', () => {
    const term = {
      buffer: {
        active: {
          length: 3,
          /** @param {number} i */
          getLine(i) {
            if (i === 1) return undefined;
            return {
              isWrapped: false,
              translateToString() {
                return `row ${i}`;
              },
            };
          },
        },
      },
    };

    expect(captureScrollback(term)).toBe('row 0\n\nrow 2');
  });
});
