// @ts-check
/**
 * Unit tests for the OKF Knowledge Panel's pure DOM-leaf helpers
 * (helpers.js) -- the interface's first JS unit test:
 *   npx vitest run tests/interfaces/okf_panel/helpers.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';
import { marked } from 'marked';

import {
  STRUCTURE_MARKER,
  el,
  facilityLinkView,
  hashChangeTarget,
  isFallback,
  readPanelParams,
} from '../../../src/osprey/interfaces/okf_panel/static/js/helpers.js';

describe('el', () => {
  test('creates an element with the given tag', () => {
    const node = el('span');
    expect(node.tagName).toBe('SPAN');
  });

  test('the "class" key sets className, not a class attribute', () => {
    const node = el('div', { class: 'health-dot ok' });
    expect(node.className).toBe('health-dot ok');
  });

  test('the "text" key sets textContent', () => {
    const node = el('span', { text: 'hello world' });
    expect(node.textContent).toBe('hello world');
  });

  test('text is set via textContent, never parsed as HTML', () => {
    const node = el('div', { text: '<b>x</b>' });
    expect(node.textContent).toBe('<b>x</b>');
    expect(node.querySelector('b')).toBe(null);
    expect(node.children.length).toBe(0);
  });

  test('other keys are set as attributes, readable via getAttribute', () => {
    const node = el('a', { title: 'A title', href: '#control-system/x' });
    expect(node.getAttribute('title')).toBe('A title');
    expect(node.getAttribute('href')).toBe('#control-system/x');
  });

  test('children are appended in order', () => {
    const a = el('span', { text: 'first' });
    const b = el('span', { text: 'second' });
    const parent = el('div', null, [a, b]);
    expect(parent.children.length).toBe(2);
    expect(parent.children[0]).toBe(a);
    expect(parent.children[1]).toBe(b);
  });

  test('falsy children entries are skipped without throwing', () => {
    const a = el('span', { text: 'only' });
    expect(() => el('div', null, [null, a, undefined])).not.toThrow();
    const parent = el('div', null, [null, a, undefined]);
    expect(parent.children.length).toBe(1);
    expect(parent.children[0]).toBe(a);
  });

  test('attrs may be null, as in el("li", null, [link])', () => {
    const link = el('a', { href: '#x', text: 'link' });
    const node = el('li', null, [link]);
    expect(node.tagName).toBe('LI');
    expect(node.children[0]).toBe(link);
  });

  test('children may be omitted entirely, as in el("span", { class: "health-dot ok" })', () => {
    const node = el('span', { class: 'health-dot ok' });
    expect(node.tagName).toBe('SPAN');
    expect(node.className).toBe('health-dot ok');
    expect(node.children.length).toBe(0);
  });
});

describe('isFallback', () => {
  test('true when the title equals the last "/"-separated segment of the id', () => {
    expect(isFallback('control-system/channel-finding', 'channel-finding')).toBe(true);
  });

  test('false when the title is a real human title', () => {
    expect(isFallback('control-system/channel-finding', 'Channel Finding')).toBe(false);
  });

  test('false for empty or missing id/title (guard clause)', () => {
    expect(isFallback('', 'channel-finding')).toBe(false);
    expect(isFallback('control-system/channel-finding', '')).toBe(false);
    expect(isFallback(null, 'channel-finding')).toBe(false);
    expect(isFallback('control-system/channel-finding', null)).toBe(false);
    expect(isFallback(undefined, undefined)).toBe(false);
  });
});

describe('readPanelParams', () => {
  beforeEach(() => {
    location.hash = '';
  });

  test('empty hash yields concept: null, structure: false, raw: ""', () => {
    location.hash = '';
    expect(readPanelParams()).toEqual({ concept: null, structure: false, raw: '' });
  });

  test('the structure marker hash yields structure: true, concept: null', () => {
    location.hash = `#${STRUCTURE_MARKER}`;
    const params = readPanelParams();
    expect(params.structure).toBe(true);
    expect(params.concept).toBe(null);
  });

  test('a concept hash preserves slashes and sets structure: false', () => {
    location.hash = '#control-system/channel-finding';
    const params = readPanelParams();
    expect(params.concept).toBe('control-system/channel-finding');
    expect(params.structure).toBe(false);
  });

  test('a malformed percent-escape falls back to the raw hash via the catch branch', () => {
    // Verified to throw URIError in decodeURIComponent, and happy-dom does
    // not normalize it away when assigned to location.hash:
    //   node -e 'decodeURIComponent("%E0%A4%A")' -> URIError: URI malformed
    const malformed = '%E0%A4%A';
    expect(() => decodeURIComponent(malformed)).toThrow(URIError);
    location.hash = `#${malformed}`;
    const params = readPanelParams();
    expect(params.concept).toBe(malformed);
    expect(params.raw).toBe(malformed);
  });
});

describe('hashChangeTarget', () => {
  /**
   * @param {string} hash
   */
  function paramsFor(hash) {
    location.hash = hash;
    return readPanelParams();
  }

  const known = new Set(['devices/bpm', 'procedures/orbit-correction']);

  beforeEach(() => {
    location.hash = '';
  });

  test('an empty hash yields null', () => {
    expect(hashChangeTarget(paramsFor(''), null, known)).toBe(null);
  });

  test('a history state already on the target yields null', () => {
    expect(hashChangeTarget(paramsFor('#devices/bpm'), { id: 'devices/bpm' }, known)).toBe(null);
  });

  test('the structure marker with a null state yields the marker', () => {
    expect(hashChangeTarget(paramsFor(`#${STRUCTURE_MARKER}`), null, known)).toBe(STRUCTURE_MARKER);
  });

  test('a listed concept id with a null state yields that id', () => {
    expect(hashChangeTarget(paramsFor('#devices/bpm'), null, known)).toBe('devices/bpm');
  });

  test('a percent-encoded listed id yields the decoded id', () => {
    expect(hashChangeTarget(paramsFor('#devices%2Fbpm'), null, known)).toBe('devices/bpm');
  });

  test('an id the tree does not list yields null', () => {
    expect(hashChangeTarget(paramsFor('#some-heading'), null, known)).toBe(null);
  });

  test('any id while the tree has not loaded yields that id', () => {
    expect(hashChangeTarget(paramsFor('#some-heading'), null, null)).toBe('some-heading');
  });
});

describe('facilityLinkView', () => {
  /** @typedef {import('../../../src/osprey/interfaces/okf_panel/static/js/helpers.js').FacilityRecordView} RecordView */
  /**
   * @param {string} target
   * @param {RecordView[] | null} records
   */
  const span = (target, records) => ({ start: 0, end: 1, target, kinds: [], label: 'x', records });
  const references = [
    span('SR:DIAG:BPM:01:POSITION:X', [
      { kind: 'channel', id: 'SR:DIAG:BPM:01:POSITION:X', unit: 'mm', description: 'Beam X.' },
    ]),
    span('TWIN', [
      { kind: 'channel', id: 'TWIN', unit: 'A' },
      { kind: 'device', id: 'TWIN' },
    ]),
    span('SR GAUGE 1', [{ kind: 'device', id: 'SR GAUGE 1' }]),
    span('NO/SUCH', []),
  ];

  test('a held channel shows its kind, unit and description', () => {
    expect(facilityLinkView('facility:SR:DIAG:BPM:01:POSITION:X', references)).toEqual({
      text: 'SR:DIAG:BPM:01:POSITION:X',
      title: 'channel · mm · Beam X.',
      dangling: false,
    });
  });

  test('a target held as channel and device names both records', () => {
    expect(facilityLinkView('facility:TWIN', references).title).toBe('channel · A\ndevice');
  });

  test('an empty list is a dangling link', () => {
    const view = facilityLinkView('facility:NO/SUCH', references);
    expect(view.dangling).toBe(true);
    expect(view.title).toBe('NO/SUCH · not in the facility file');
  });

  test('without a facility file the link is unresolved, not dangling', () => {
    expect(facilityLinkView('facility:NO/SUCH', null)).toEqual({
      text: 'NO/SUCH',
      title: 'NO/SUCH · facility file not loaded',
      dangling: false,
    });
    expect(facilityLinkView('facility:TWIN', [span('TWIN', null)]).dangling).toBe(false);
  });

  test('a percent-encoded space is decoded before the lookup', () => {
    expect(facilityLinkView('facility:SR%20GAUGE%201', references)).toEqual({
      text: 'SR GAUGE 1',
      title: 'device',
      dangling: false,
    });
  });

  test('marked keeps the facility href and percent-encodes an angle-bracket target', () => {
    const holder = document.createElement('div');
    holder.innerHTML = /** @type {string} */ (
      marked.parse('[x](facility:SR:DIAG:BPM:01:POSITION:X) [g](<facility:SR GAUGE 1>)')
    );
    const hrefs = Array.from(holder.querySelectorAll('a')).map((a) => a.getAttribute('href'));
    expect(hrefs).toEqual(['facility:SR:DIAG:BPM:01:POSITION:X', 'facility:SR%20GAUGE%201']);
    expect(facilityLinkView(hrefs[1] || '', references).text).toBe('SR GAUGE 1');
  });
});
