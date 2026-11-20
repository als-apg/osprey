// @ts-check
/**
 * Contract tests for shell-dress.js, which runs ahead of app.js:
 *   npx vitest run tests/interfaces/web_terminal/shell-dress.test.mjs
 *
 * It does its whole job on import: the glyphs and key hints of the document as
 * it is parsed so far. shell-icons.test.mjs pins the fill itself.
 */

import { test, expect } from 'vitest';

const DRESS = '../../../src/osprey/interfaces/web_terminal/static/js/shell-dress.js';
const SVG_NS = 'http://www.w3.org/2000/svg';

test('draws the glyphs and key hints already in the document, on import', async () => {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('data-icon', 'search');
  const kbd = document.createElement('kbd');
  kbd.dataset.chord = 'K';
  document.body.replaceChildren(svg, kbd);

  await import(DRESS);

  expect(svg.childElementCount).toBeGreaterThan(0);
  expect(kbd.textContent).toMatch(/^(⌘|Ctrl\+)K$/);
});

test('imports the glyph fill and nothing of the application', async () => {
  const { readFileSync } = await import('node:fs');
  const { join } = await import('node:path');
  const source = readFileSync(join(import.meta.dirname, DRESS), 'utf8');
  const imports = [...source.matchAll(/^import .* from '(.+)';$/gm)].map((match) => match[1]);

  expect(imports).toEqual(['./shell-icons.js']);
});
