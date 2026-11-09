// @ts-check
/**
 * The ARIEL list pane head (list-head.js):
 *   npx vitest run tests/interfaces/ariel/list-head.test.mjs
 *
 * `buildHead()` builds the mode chip; the tile bar holds search and compose.
 * `buildFiltersPopover()` replaces the body-level
 * popover with a fresh hidden one. `paintChipIssues` paints a search's
 * diagnostics on the chip (dot, tooltip, accessible description) and the top of
 * the popover; an empty list clears all three. The head wired into the pane
 * lives in list.test.mjs.
 *
 * Runs under happy-dom. Nothing is mocked. The store is a module singleton, so
 * every case resets modules and imports the store and the head from the same
 * reset.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const STORE_PATH = '../../../src/osprey/interfaces/ariel/static/js/store.js';
const HEAD_PATH = '../../../src/osprey/interfaces/ariel/static/js/list-head.js';

/** @param {string} url */
async function load(url = '/') {
  history.replaceState(null, '', url);
  vi.resetModules();
  const store = await import(STORE_PATH);
  const head = await import(HEAD_PATH);
  return { store, head };
}

beforeEach(() => {
  document.body.innerHTML = '';
});

afterEach(() => {
  history.replaceState(null, '', '/');
});

describe('buildHead', () => {
  test('only the mode chip', async () => {
    const { head } = await load('/?embedded=true');
    const { head: el, chip } = head.buildHead();

    expect(chip.id).toBe('ariel-mode-chip');
    expect(chip.getAttribute('aria-haspopup')).toBe('dialog');
    expect(chip.querySelector('.ariel-mode-chip-label')).not.toBeNull();
    expect(el.contains(chip)).toBe(true);
    expect(el.querySelector('input')).toBeNull();
    expect(el.querySelector('#ariel-compose-btn')).toBeNull();
  });
});

describe('buildFiltersPopover', () => {
  test('replaces any earlier popover with one hidden dialog on the body', async () => {
    const { head } = await load();
    const first = head.buildFiltersPopover();
    const second = head.buildFiltersPopover();

    expect(document.querySelectorAll('#ariel-filters-popover')).toHaveLength(1);
    expect(first.isConnected).toBe(false);
    expect(second.parentElement).toBe(document.body);
    expect(second.hidden).toBe(true);
    expect(second.getAttribute('role')).toBe('dialog');
    for (const slot of ['data-search-issues', 'data-mode-choices', 'data-filters']) {
      expect(second.querySelector(`[${slot}]`)).not.toBeNull();
    }
  });
});

describe('paintChipIssues', () => {
  /** @type {import('../../../src/osprey/interfaces/ariel/static/js/components.js').Diagnostic[]} */
  const WARNING = [{ level: 'warning', source: 'semantic', message: 'Embedding service down' }];

  test('issues paint the dot, tooltip, description and popover top', async () => {
    const { head } = await load();
    const { chip } = head.buildHead();
    const popover = head.buildFiltersPopover();

    head.paintChipIssues(chip, popover, WARNING);

    const dot = /** @type {HTMLElement} */ (chip.querySelector('.ariel-mode-chip-dot'));
    expect(dot.dataset.level).toBe('warning');
    expect(dot.getAttribute('aria-hidden')).toBe('true');
    expect(chip.title).toContain('Embedding service down');
    expect(chip.getAttribute('aria-description')).toContain('Embedding service down');
    const issues = /** @type {HTMLElement} */ (popover.querySelector('[data-search-issues]'));
    expect(issues.hidden).toBe(false);
    expect(issues.textContent).toContain('Embedding service down');
  });

  test('a repaint keeps one dot, and an empty list clears all three', async () => {
    const { head } = await load();
    const { chip } = head.buildHead();
    const popover = head.buildFiltersPopover();
    const title = chip.title;

    head.paintChipIssues(chip, popover, WARNING);
    head.paintChipIssues(chip, popover, WARNING);
    expect(chip.querySelectorAll('.ariel-mode-chip-dot')).toHaveLength(1);

    head.paintChipIssues(chip, popover, []);
    expect(chip.querySelector('.ariel-mode-chip-dot')).toBeNull();
    expect(chip.title).toBe(title);
    expect(chip.hasAttribute('aria-description')).toBe(false);
    const issues = /** @type {HTMLElement} */ (popover.querySelector('[data-search-issues]'));
    expect(issues.hidden).toBe(true);
    expect(issues.innerHTML).toBe('');
  });
});
