// @ts-check
/**
 * The panel's one severity vocabulary (severity.js): a severity from any
 * spelling the routes carry, the worst of a set, and its mark.
 *   npx vitest run tests/interfaces/channel_finder/severity.test.mjs
 */

import { test, expect, describe } from 'vitest';
import {
  BAND_SEVERITIES,
  SEVERITIES,
  inBand,
  severityMark,
  severityOf,
  worstSeverity,
} from '../../../src/osprey/interfaces/channel_finder/static/js/severity.js';

describe('severityOf', () => {
  test.each([
    ['MAJOR', 'MAJOR'],
    ['INVALID', 'INVALID'],
    ['UNDEFINED', 'UNDEFINED'],
    ['MINOR', 'MINOR'],
    ['OK', 'OK'],
    ['MAJOR_ACK', 'MAJOR'],
    ['MINOR_ACK', 'MINOR'],
    ['INVALID_ACK', 'INVALID'],
    ['UNDEFINED_ACK', 'UNDEFINED'],
    [0, 'OK'],
    [1, 'MINOR'],
    [2, 'MAJOR'],
    [3, 'INVALID'],
    [null, 'NONE'],
    [undefined, 'NONE'],
  ])('%j is %s', (value, severity) => {
    expect(severityOf(value)).toBe(severity);
  });

  test.each(['major', 'HIHI_ALARM', 'OK_ACK', '', 4, -1, 1.5, true, {}, []])(
    '%j is UNDEFINED',
    (value) => {
      expect(severityOf(value)).toBe('UNDEFINED');
    }
  );

  test('every answer is one of the vocabulary', () => {
    for (const value of ['MAJOR', 'MINOR_ACK', 2, null, 'junk']) {
      expect(SEVERITIES).toContain(severityOf(value));
    }
  });
});

describe('the band of counts', () => {
  test('shows the server severities in the mockup order, UNDEFINED only when counted', () => {
    expect(BAND_SEVERITIES).toEqual(['MAJOR', 'INVALID', 'MINOR', 'UNDEFINED']);
    expect(BAND_SEVERITIES.filter((s) => inBand(s, 0))).toEqual(['MAJOR', 'INVALID', 'MINOR']);
    expect(inBand('UNDEFINED', 1)).toBe(true);
  });
});

describe('the worst-first order', () => {
  test('matches the connectors rank', () => {
    expect(SEVERITIES).toEqual(['UNDEFINED', 'INVALID', 'MAJOR', 'MINOR', 'OK', 'NONE']);
  });

  test.each([
    [['MAJOR', 'INVALID'], 'INVALID'],
    [['MAJOR', 'UNDEFINED'], 'UNDEFINED'],
    [['MINOR', 'MAJOR', 'OK'], 'MAJOR'],
    [['OK', 'NONE'], 'OK'],
    [[], 'NONE'],
  ])('the worst of %j is %s', (severities, worst) => {
    expect(worstSeverity(/** @type {any} */ (severities))).toBe(worst);
  });
});

describe('severityMark', () => {
  /** @param {string} html */
  function parse(html) {
    const host = document.createElement('div');
    host.innerHTML = html;
    return /** @type {HTMLElement} */ (host.firstElementChild);
  }

  test.each(SEVERITIES)('%s draws one class and its word as text', (severity) => {
    const mark = parse(severityMark(severity));
    expect(mark.classList.contains('cf-sev')).toBe(true);
    const modifiers = [...mark.classList].filter((name) => name.startsWith('cf-sev--'));
    expect(modifiers).toEqual([`cf-sev--${severity.toLowerCase()}`]);
    expect(mark.querySelector('i')).not.toBeNull();
    expect(mark.textContent?.trim()).not.toBe('');
  });

  test('the word is the severity, or the given text', () => {
    expect(parse(severityMark('MAJOR')).textContent).toBe('MAJOR');
    expect(parse(severityMark('MAJOR', 'MAJOR · HIHI_ALARM')).textContent).toBe(
      'MAJOR · HIHI_ALARM'
    );
  });

  test('the given text is text, never markup', () => {
    const mark = parse(severityMark('MINOR', '<img src=x onerror=alert(1)>'));
    expect(mark.querySelector('img')).toBeNull();
    expect(mark.textContent).toBe('<img src=x onerror=alert(1)>');
  });

  test('OK draws the check icon', () => {
    expect(parse(severityMark('OK')).querySelector('i svg.osprey-ic')).not.toBeNull();
  });

  test('a severity outside the vocabulary draws as UNDEFINED', () => {
    const mark = parse(severityMark(/** @type {any} */ ('HIHI')));
    expect(mark.classList.contains('cf-sev--undefined')).toBe(true);
  });
});
