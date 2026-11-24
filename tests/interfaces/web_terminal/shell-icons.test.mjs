// @ts-check
/**
 * Contract tests for shell-icons.js, the fill of the page's glyph placeholders
 * and key hints:
 *   npx vitest run tests/interfaces/web_terminal/shell-icons.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import {
  hydrateIcons,
  paintChords,
} from '../../../src/osprey/interfaces/web_terminal/static/js/shell-icons.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** @param {string} name */
function placeholder(name) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'osprey-ic');
  svg.setAttribute('data-icon', name);
  document.body.appendChild(svg);
  return svg;
}

/** @param {string} platform */
function stubPlatform(platform) {
  vi.stubGlobal('navigator', { platform, userAgent: platform });
}

beforeEach(() => {
  document.body.replaceChildren();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('hydrateIcons', () => {
  test('fills an empty placeholder in place, keeping the element and its class', () => {
    const svg = placeholder('search');

    hydrateIcons(document);

    expect(document.querySelector('svg[data-icon="search"]')).toBe(svg);
    expect(svg.childElementCount).toBeGreaterThan(0);
    expect(svg.getAttribute('viewBox')).toBeTruthy();
    expect(svg.getAttribute('class')).toBe('osprey-ic');
  });

  test('leaves a glyph that is already drawn alone', () => {
    const svg = placeholder('search');
    hydrateIcons(document);
    const drawn = [...svg.childNodes];

    hydrateIcons(document);

    expect([...svg.childNodes]).toEqual(drawn);
    expect(svg.childNodes[0]).toBe(drawn[0]);
  });

  test('fills only under the root it is given', () => {
    const inside = document.createElement('div');
    document.body.appendChild(inside);
    const within = placeholder('plus');
    inside.appendChild(within);
    const outside = placeholder('menu');

    hydrateIcons(inside);

    expect(within.childElementCount).toBeGreaterThan(0);
    expect(outside.childElementCount).toBe(0);
  });
});

describe('paintChords', () => {
  /** @returns {HTMLElement} */
  function hint() {
    const kbd = document.createElement('kbd');
    kbd.dataset.chord = 'K';
    document.body.appendChild(kbd);
    return kbd;
  }

  test('writes the command key on macOS', () => {
    stubPlatform('MacIntel');
    const kbd = hint();

    paintChords(document);

    expect(kbd.textContent).toBe('⌘K');
  });

  test('writes Ctrl+ elsewhere', () => {
    stubPlatform('Linux x86_64');
    const kbd = hint();

    paintChords(document);

    expect(kbd.textContent).toBe('Ctrl+K');
  });
});
