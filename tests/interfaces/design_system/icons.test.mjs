// @ts-check
/**
 * Unit tests for the shared chrome icons (design_system/static/js/icons.js):
 *   npx vitest run tests/interfaces/design_system/icons.test.mjs
 *
 * Pins the contract the shell relies on: every listed name builds an
 * `svg.osprey-ic` from SVG-namespace nodes (no innerHTML), the glyph is
 * hidden from assistive tech, and an unknown name throws at the call site.
 */

import { test, expect, describe } from 'vitest';

import { ICON_NAMES, icon } from '/design-system/js/icons.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** The names the layout F contract lists, in its order. */
const CONTRACT_NAMES = [
  'panel-left',
  'chevron-down',
  'chevron-right',
  'chevron-left',
  'chevrons-ud',
  'search',
  'workspace',
  'activity',
  'terminal',
  'notebook',
  'book-open',
  'hexagon',
  'library',
  'radio',
  'flask',
  'plus',
  'help',
  'message',
  'more',
  'sliders',
  'file-text',
  'chart-line',
  'bash',
  'maximize',
  'save',
  'x',
  'pen',
  'bell',
  'share',
  'bulb',
  'clip',
  'up',
  'agent',
  'crosshair',
  'menu',
  'keyboard',
  'compass',
  'info',
  'logout',
  'sun-moon',
  'check',
  'types',
  'history',
  'table',
  'database',
  'refresh',
  'layers',
];

describe('icons.js', () => {
  test('ICON_NAMES is exactly the contract set, frozen', () => {
    expect([...ICON_NAMES]).toEqual(CONTRACT_NAMES);
    expect(Object.isFrozen(ICON_NAMES)).toBe(true);
  });

  test.each(CONTRACT_NAMES)('%s builds an svg.osprey-ic with drawn children', (name) => {
    const svg = icon(name);
    expect(svg.namespaceURI).toBe(SVG_NS);
    expect(svg.tagName.toLowerCase()).toBe('svg');
    expect(svg.getAttribute('class')).toBe('osprey-ic');
    expect(svg.getAttribute('viewBox')).toBe('0 0 24 24');
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.children.length).toBeGreaterThan(0);
    for (const child of svg.children) {
      expect(child.namespaceURI).toBe(SVG_NS);
      expect(['path', 'rect', 'circle', 'ellipse']).toContain(child.tagName.toLowerCase());
    }
  });

  test('a path child carries its d attribute', () => {
    const [path] = icon('x').children;
    expect(path.getAttribute('d')).toBe('M18 6 6 18');
  });

  test('each call returns a fresh node', () => {
    expect(icon('plus')).not.toBe(icon('plus'));
  });

  test('the default size leaves sizing to CSS; another size sets it inline', () => {
    expect(icon('search').style.width).toBe('');
    const big = icon('search', { size: 20 });
    expect(big.style.width).toBe('20px');
    expect(big.style.height).toBe('20px');
  });

  test('up keeps its heavier stroke on every child', () => {
    for (const child of icon('up').children) {
      expect(child.getAttribute('stroke-width')).toBe('2');
    }
  });

  test('an unknown name throws', () => {
    expect(() => icon('no-such-icon')).toThrow(/unknown icon name/);
    expect(() => icon('toString')).toThrow(/unknown icon name/);
  });
});
