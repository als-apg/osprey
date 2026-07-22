// @ts-check
/**
 * Unit tests for the System Health dashboard's pure helpers (helpers.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/helpers.test.mjs
 */

import { test, expect, describe } from 'vitest';

import {
  STATUS_PRIORITY,
  esc,
  el,
  worst,
  fmtName,
  fmtMs,
  msCls,
  byCategory,
} from './helpers.js';

describe('esc', () => {
  test('escapes HTML metacharacters so server data cannot inject markup', () => {
    expect(esc('<img src=x onerror=alert(1)>')).toBe(
      '&lt;img src=x onerror=alert(1)&gt;',
    );
    expect(esc('a & b')).toBe('a &amp; b');
  });

  test('null and undefined become the empty string', () => {
    expect(esc(null)).toBe('');
    expect(esc(undefined)).toBe('');
  });

  test('non-string values are stringified before escaping', () => {
    expect(esc(42)).toBe('42');
    expect(esc(0)).toBe('0');
  });

  test('plain text is returned unchanged', () => {
    expect(esc('Beam Current')).toBe('Beam Current');
  });
});

describe('el', () => {
  test('creates an element with the given tag', () => {
    expect(el('span').tagName).toBe('SPAN');
  });

  test('the "class" key sets className; "text" sets textContent', () => {
    const node = el('div', { class: 'dot s-ok', text: 'hi' });
    expect(node.className).toBe('dot s-ok');
    expect(node.textContent).toBe('hi');
  });

  test('the "text" key is never parsed as HTML', () => {
    const node = el('div', { text: '<b>x</b>' });
    expect(node.textContent).toBe('<b>x</b>');
    expect(node.querySelector('b')).toBe(null);
  });

  test('other keys become attributes readable via getAttribute', () => {
    const node = el('a', { title: 'A title', 'data-s': 'error' });
    expect(node.getAttribute('title')).toBe('A title');
    expect(node.getAttribute('data-s')).toBe('error');
  });

  test('children are appended in order and falsy entries skipped', () => {
    const a = el('span', { text: 'first' });
    const b = el('span', { text: 'second' });
    const parent = el('div', null, [null, a, undefined, b]);
    expect(parent.children.length).toBe(2);
    expect(parent.children[0]).toBe(a);
    expect(parent.children[1]).toBe(b);
  });

  test('attrs may be null and children omitted', () => {
    const node = el('li', null);
    expect(node.tagName).toBe('LI');
    expect(node.children.length).toBe(0);
  });
});

describe('worst', () => {
  test('reduces to the single most-severe status present', () => {
    expect(worst([{ status: 'ok' }, { status: 'warning' }, { status: 'ok' }])).toBe('warning');
    expect(worst([{ status: 'warning' }, { status: 'error' }])).toBe('error');
    expect(worst([{ status: 'ok' }, { status: 'skip' }])).toBe('skip');
  });

  test('an empty list is ok', () => {
    expect(worst([])).toBe('ok');
  });

  test('priority order is error > warning > skip > ok', () => {
    expect(STATUS_PRIORITY.error).toBeGreaterThan(STATUS_PRIORITY.warning);
    expect(STATUS_PRIORITY.warning).toBeGreaterThan(STATUS_PRIORITY.skip);
    expect(STATUS_PRIORITY.skip).toBeGreaterThan(STATUS_PRIORITY.ok);
    // error must win regardless of position in the list.
    expect(worst([{ status: 'error' }, { status: 'warning' }, { status: 'skip' }])).toBe('error');
    expect(worst([{ status: 'skip' }, { status: 'warning' }, { status: 'error' }])).toBe('error');
  });

  test('an unknown status is treated as ok-like (priority 0)', () => {
    expect(worst([{ status: 'ok' }, { status: 'mystery' }])).toBe('ok');
    expect(worst([{ status: 'mystery' }, { status: 'warning' }])).toBe('warning');
  });
});

describe('fmtName', () => {
  test('strips the leading "<category>." prefix and title-cases the rest', () => {
    expect(fmtName('epics.beam_current')).toBe('Beam Current');
  });

  test('a name with no dot is title-cased as-is', () => {
    expect(fmtName('disk_space')).toBe('Disk Space');
  });

  test('only the first dot delimits the prefix', () => {
    expect(fmtName('a.b_c')).toBe('B C');
  });

  test('a single word is capitalized', () => {
    expect(fmtName('latency')).toBe('Latency');
  });
});

describe('fmtMs', () => {
  test('sub-second latencies render as rounded milliseconds', () => {
    expect(fmtMs(42)).toBe('42ms');
    expect(fmtMs(42.6)).toBe('43ms');
    expect(fmtMs(999)).toBe('999ms');
  });

  test('one second and above renders as fixed-1 seconds', () => {
    expect(fmtMs(1000)).toBe('1.0s');
    expect(fmtMs(2500)).toBe('2.5s');
  });

  test('zero, negative, or missing latency renders empty (per to_dict omission)', () => {
    expect(fmtMs(0)).toBe('');
    expect(fmtMs(-5)).toBe('');
    expect(fmtMs(undefined)).toBe('');
  });
});

describe('msCls', () => {
  test('buckets latency into fast / medium / slow classes', () => {
    expect(msCls(50)).toBe('ms-f');
    expect(msCls(99)).toBe('ms-f');
    expect(msCls(100)).toBe('ms-m');
    expect(msCls(499)).toBe('ms-m');
    expect(msCls(500)).toBe('ms-s');
    expect(msCls(5000)).toBe('ms-s');
  });

  test('zero, negative, or missing latency yields no class', () => {
    expect(msCls(0)).toBe('');
    expect(msCls(-1)).toBe('');
    expect(msCls(undefined)).toBe('');
  });
});

describe('byCategory', () => {
  test('groups results by category, preserving first-seen order', () => {
    const results = [
      { category: 'file_system', name: 'a' },
      { category: 'providers', name: 'b' },
      { category: 'file_system', name: 'c' },
    ];
    const grouped = byCategory(results);
    expect([...grouped.keys()]).toEqual(['file_system', 'providers']);
    expect(grouped.get('file_system')?.map((r) => r.name)).toEqual(['a', 'c']);
    expect(grouped.get('providers')?.map((r) => r.name)).toEqual(['b']);
  });

  test('an empty result list yields an empty map', () => {
    expect(byCategory([]).size).toBe(0);
  });
});
