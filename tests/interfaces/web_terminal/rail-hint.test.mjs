/**
 * Unit tests for the one-time rail hint (rail-hint.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/rail-hint.test.mjs
 *
 * maybeShowRailHint() shows a small dismissible callout anchored to the rail
 * region — at most once EVER (localStorage flag, deliberately not keyed to
 * the server session). Suppressed when the flag is
 * set, the rail is already top, the page is embedded, or the rail region is
 * absent. "Move to top" flips the rail AND sets the flag; "Got it" just sets
 * the flag.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { maybeShowRailHint } from '../../../src/osprey/interfaces/web_terminal/static/js/rail-hint.js';

describe('first-visit rail hint', () => {
  beforeEach(() => {
    localStorage.clear();
    document.body.innerHTML =
      '<div class="shell-body"><div class="panel-rail-region"></div></div>';
    document.documentElement.setAttribute('data-rail-position', 'left');
    document.body.classList.remove('embedded');
    window.history.replaceState(null, '', '/');
  });

  test('shows once, then never again after dismiss', () => {
    expect(maybeShowRailHint()).toBe(true);
    expect(document.querySelector('.rail-hint')).not.toBe(null);

    const dismiss = document.querySelector('.rail-hint-dismiss');
    if (!(dismiss instanceof HTMLElement)) throw new Error('expected the rail-hint dismiss button');
    dismiss.click();

    expect(document.querySelector('.rail-hint')).toBe(null);
    expect(localStorage.getItem('osprey-rail-hint-dismissed-v1')).toBe('1');
    expect(maybeShowRailHint()).toBe(false);
  });

  test("'Move to top' flips the rail and dismisses", () => {
    maybeShowRailHint();

    const move = document.querySelector('.rail-hint-move');
    if (!(move instanceof HTMLElement)) throw new Error('expected the rail-hint move button');
    move.click();

    expect(document.documentElement.getAttribute('data-rail-position')).toBe('top');
    expect(document.querySelector('.rail-hint')).toBe(null);
    expect(localStorage.getItem('osprey-rail-hint-dismissed-v1')).toBe('1');
    // The flip went through setRailPosition, so it persisted as an explicit choice.
    expect(localStorage.getItem('osprey-rail-position')).toBe('top');
  });

  test('suppressed when rail already top or flag set', () => {
    document.documentElement.setAttribute('data-rail-position', 'top');
    expect(maybeShowRailHint()).toBe(false);

    document.documentElement.setAttribute('data-rail-position', 'left');
    localStorage.setItem('osprey-rail-hint-dismissed-v1', '1');
    expect(maybeShowRailHint()).toBe(false);
    expect(document.querySelector('.rail-hint')).toBe(null);
  });

  test('suppressed in embedded pages and without a rail region', () => {
    document.body.classList.add('embedded');
    expect(maybeShowRailHint()).toBe(false);

    document.body.classList.remove('embedded');
    document.body.innerHTML = '<div class="shell-body"></div>';
    expect(maybeShowRailHint()).toBe(false);
  });
});
