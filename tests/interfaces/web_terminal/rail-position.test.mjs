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
  followThemeFamily,
  getRailPosition,
  initRailThemeCoupling,
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
    setRailPosition(/** @type {any} */ ('diagonal'));

    expect(document.documentElement.getAttribute('data-rail-position')).toBe('left');
    expect(localStorage.getItem('osprey-rail-position')).toBe(null);
    // The one-shot param survives too — nothing about the call took effect.
    expect(window.location.search).toBe('?rail=top&theme=dark');
  });
});

/**
 * The theme coupling: the retro family restores the pre-redesign look, and
 * the horizontal tab strip is part of that look. The coupling map itself is
 * served by GET /api/panels (app.FAMILY_RAIL_DEFAULTS) — this module only
 * applies it, and only when nobody has stated a position of their own.
 */
describe('rail-position theme coupling', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.setAttribute('data-rail-position', 'left');
    window.history.replaceState(null, '', '/');
    initRailThemeCoupling({ family_rail_defaults: { retro: 'top' }, rail_position_configured: false });
  });

  test('the retro family moves the rail to the top', () => {
    const resized = vi.fn();
    window.addEventListener('resize', resized);

    followThemeFamily('retro');

    expect(getRailPosition()).toBe('top');
    expect(resized).toHaveBeenCalled();
    window.removeEventListener('resize', resized);
  });

  test('following a family never persists — leaving retro moves the rail back', () => {
    followThemeFamily('retro');
    expect(getRailPosition()).toBe('top');
    // Nothing pinned: the move was a consequence of the theme, not a choice.
    expect(localStorage.getItem('osprey-rail-position')).toBeNull();

    followThemeFamily('main');
    expect(getRailPosition()).toBe('left');
  });

  test('an unmapped family goes to the default position', () => {
    followThemeFamily('retro');
    followThemeFamily('high-contrast');
    expect(getRailPosition()).toBe('left');
  });

  test('a user pin outranks the family', () => {
    setRailPosition('left'); // explicit choice — pinned
    followThemeFamily('retro');
    expect(getRailPosition()).toBe('left');
  });

  test('a user pin of top survives leaving the retro family', () => {
    setRailPosition('top');
    followThemeFamily('main');
    expect(getRailPosition()).toBe('top');
  });

  test('a config-pinned rail outranks the family', () => {
    initRailThemeCoupling({ family_rail_defaults: { retro: 'top' }, rail_position_configured: true });
    followThemeFamily('retro');
    expect(getRailPosition()).toBe('left');
  });

  test('an absent or partial payload leaves the coupling inert', () => {
    initRailThemeCoupling({});
    followThemeFamily('retro');
    expect(getRailPosition()).toBe('left');

    initRailThemeCoupling(undefined);
    followThemeFamily('retro');
    expect(getRailPosition()).toBe('left');
  });

  test('a no-op follow does not fire a spurious resize', () => {
    const resized = vi.fn();
    window.addEventListener('resize', resized);

    followThemeFamily('main'); // already left

    expect(resized).not.toHaveBeenCalled();
    window.removeEventListener('resize', resized);
  });
});
