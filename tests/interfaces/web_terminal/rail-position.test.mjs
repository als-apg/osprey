/**
 * Unit tests for the rail-position runtime axis (rail-position.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/rail-position.test.mjs
 *
 * getRailPosition reads html[data-rail-position] (defaulting left);
 * setRailPosition stamps the attribute, persists the explicit choice,
 * strips a one-shot ?rail= from the URL, and dispatches a window resize so
 * dockview + xterm re-fit to the new geometry. Invalid positions are
 * rejected wholesale — no attribute write, no persistence.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import {
  getRailPosition,
  setRailPosition,
} from '../../../src/osprey/interfaces/web_terminal/static/js/rail-position.js';

describe('rail-position runtime setter', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.setAttribute('data-rail-position', 'left');
    window.history.replaceState(null, '', '/?rail=top&theme=dark');
  });

  test('getRailPosition reads the attribute, defaulting left', () => {
    expect(getRailPosition()).toBe('left');
    document.documentElement.setAttribute('data-rail-position', 'top');
    expect(getRailPosition()).toBe('top');
    document.documentElement.removeAttribute('data-rail-position');
    expect(getRailPosition()).toBe('left');
    document.documentElement.setAttribute('data-rail-position', 'bogus');
    expect(getRailPosition()).toBe('left');
  });

  test('setRailPosition stamps, persists, strips ?rail, fires resize', () => {
    const resized = vi.fn();
    window.addEventListener('resize', resized);

    setRailPosition('top');

    expect(document.documentElement.getAttribute('data-rail-position')).toBe('top');
    expect(localStorage.getItem('osprey-rail-position')).toBe('top');
    // Only the one-shot rail param is dropped — other params survive.
    expect(window.location.search).toBe('?theme=dark');
    expect(resized).toHaveBeenCalled();
    window.removeEventListener('resize', resized);
  });

  test('strip is a no-op when the URL carries no ?rail=', () => {
    window.history.replaceState(null, '', '/?theme=dark');

    setRailPosition('top');

    expect(window.location.search).toBe('?theme=dark');
    expect(document.documentElement.getAttribute('data-rail-position')).toBe('top');
  });

  test('invalid position is a no-op', () => {
    setRailPosition('diagonal');

    expect(document.documentElement.getAttribute('data-rail-position')).toBe('left');
    expect(localStorage.getItem('osprey-rail-position')).toBe(null);
    // The one-shot param survives too — nothing about the call took effect.
    expect(window.location.search).toBe('?rail=top&theme=dark');
  });
});
