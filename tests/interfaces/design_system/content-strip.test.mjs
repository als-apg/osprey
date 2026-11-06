// @ts-check
/**
 * Unit tests for the content strip (design_system/static/js/content-strip.js):
 *   npx vitest run tests/interfaces/design_system/content-strip.test.mjs
 *
 * Visibility is CSS: the strip is visible with `[data-visible]` (a timed
 * show), `[data-pinned]` (a mode other than its group's default is pressed),
 * or while its host is hovered or holds focus. These tests check the two
 * attributes the module owns.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { createContentStrip } from '/design-system/js/content-strip.js';

/** @returns {HTMLElement} */
const host = () => /** @type {HTMLElement} */ (document.getElementById('stage'));

/** @returns {HTMLElement} */
const strip = () => /** @type {HTMLElement} */ (host().querySelector('.osprey-content-strip'));

/** @param {string} label @returns {HTMLButtonElement} */
const button = (label) =>
  /** @type {HTMLButtonElement} */ (host().querySelector(`button[aria-label="${label}"]`));

const PLOT_GROUPS = [
  {
    id: 'mode',
    items: [
      { id: 'zoom', icon: 'box-zoom', title: 'Box zoom', mode: true },
      { id: 'pan', icon: 'move', title: 'Pan', mode: true },
    ],
  },
  {
    id: 'view',
    items: [
      { id: 'zoom-in', icon: 'zoom-in', title: 'Zoom in' },
      { id: 'autoscale', icon: 'autoscale', title: 'Autoscale' },
    ],
  },
  { id: 'output', items: [{ id: 'png', icon: 'image-down', title: 'Export PNG' }] },
];

beforeEach(() => {
  document.body.innerHTML = '<div id="stage"><iframe title="plot"></iframe></div>';
});

afterEach(() => {
  vi.useRealTimers();
});

describe('createContentStrip', () => {
  test('renders one toolbar with a group per entry, in order', () => {
    createContentStrip(host(), { groups: PLOT_GROUPS, onAction: () => {} });
    expect(strip().getAttribute('role')).toBe('toolbar');
    const groups = strip().querySelectorAll('.osprey-content-strip-group');
    expect(groups).toHaveLength(3);
    expect(groups[0].querySelectorAll('button')).toHaveLength(2);
    expect(host().classList.contains('osprey-content-strip-host')).toBe(true);
  });

  test('an action calls back with its id only', () => {
    const onAction = vi.fn();
    createContentStrip(host(), { groups: PLOT_GROUPS, onAction });
    button('Export PNG').click();
    expect(onAction).toHaveBeenCalledWith('png');
    expect(onAction.mock.calls[0]).toHaveLength(1);
    expect(button('Export PNG').hasAttribute('aria-pressed')).toBe(false);
  });

  test('a mode toggles, one per group, and keeps the strip visible while pressed', () => {
    const onAction = vi.fn();
    const api = createContentStrip(host(), { groups: PLOT_GROUPS, onAction });
    expect(strip().hasAttribute('data-pinned')).toBe(false);

    button('Pan').click();
    expect(onAction).toHaveBeenLastCalledWith('pan', true);
    expect(button('Pan').getAttribute('aria-pressed')).toBe('true');
    expect(strip().hasAttribute('data-pinned')).toBe(true);

    button('Box zoom').click();
    expect(onAction).toHaveBeenLastCalledWith('zoom', true);
    expect(button('Pan').getAttribute('aria-pressed')).toBe('false');
    expect(button('Box zoom').getAttribute('aria-pressed')).toBe('true');

    api.hide();
    expect(strip().hasAttribute('data-pinned')).toBe(true);

    button('Box zoom').click();
    expect(onAction).toHaveBeenLastCalledWith('zoom', false);
    expect(strip().hasAttribute('data-pinned')).toBe(false);
  });

  test("a group's default mode is a radio set and does not pin the strip", () => {
    const onAction = vi.fn();
    const groups = [
      {
        id: 'mode',
        items: [
          { id: 'zoom', icon: 'box-zoom', title: 'Box zoom', mode: true, pressed: true },
          { id: 'pan', icon: 'move', title: 'Pan', mode: true },
        ],
      },
    ];
    createContentStrip(host(), { groups, onAction });
    expect(button('Box zoom').getAttribute('aria-pressed')).toBe('true');
    expect(strip().hasAttribute('data-pinned')).toBe(false);

    button('Box zoom').click();
    expect(onAction).not.toHaveBeenCalled();
    expect(button('Box zoom').getAttribute('aria-pressed')).toBe('true');

    button('Pan').click();
    expect(strip().hasAttribute('data-pinned')).toBe(true);
    button('Pan').click();
    expect(button('Pan').getAttribute('aria-pressed')).toBe('true');
  });

  test('show() makes it visible for about 3 s', () => {
    vi.useFakeTimers();
    const api = createContentStrip(host(), { groups: PLOT_GROUPS, onAction: () => {} });
    expect(strip().hasAttribute('data-visible')).toBe(false);
    api.show();
    expect(strip().hasAttribute('data-visible')).toBe(true);
    vi.advanceTimersByTime(2900);
    expect(strip().hasAttribute('data-visible')).toBe(true);
    vi.advanceTimersByTime(200);
    expect(strip().hasAttribute('data-visible')).toBe(false);
  });

  test('setPressed follows the content without calling back', () => {
    const onAction = vi.fn();
    const api = createContentStrip(host(), { groups: PLOT_GROUPS, onAction });
    api.setPressed('pan', true);
    expect(button('Pan').getAttribute('aria-pressed')).toBe('true');
    expect(strip().hasAttribute('data-pinned')).toBe(true);
    api.setPressed('zoom', true);
    expect(button('Pan').getAttribute('aria-pressed')).toBe('false');
    expect(onAction).not.toHaveBeenCalled();
    expect(() => api.setPressed('png', true)).toThrow(/not a mode/);
  });

  test('destroy removes the strip and gives the host back', () => {
    const api = createContentStrip(host(), { groups: PLOT_GROUPS, onAction: () => {} });
    api.destroy();
    expect(strip()).toBeNull();
    expect(host().classList.contains('osprey-content-strip-host')).toBe(false);
    expect(host().style.position).toBe('');
  });
});
