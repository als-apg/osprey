/**
 * Unit tests for the add-panel menu's pure id-derivation helper.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-add-menu.test.mjs
 *
 * derivePanelId turns human input (a label or a URL) into a URL-safe panel id.
 * The DOM/popover behavior of initPanelAddMenu is covered end-to-end by the
 * Playwright suite (test_panels_browser.py); here we pin the slug contract.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe } from 'vitest';

import { derivePanelId } from '../../../src/osprey/interfaces/web_terminal/static/js/panel-add-menu.js';

describe('derivePanelId', () => {
  test('slugs a plain label', () => {
    expect(derivePanelId('My Dashboard')).toBe('my-dashboard');
  });

  test('uses the hostname for a URL input', () => {
    expect(derivePanelId('http://grafana.internal:3000')).toBe('grafana-internal');
    expect(derivePanelId('https://Metrics.Lan/path?q=1')).toBe('metrics-lan');
  });

  test('collapses runs of punctuation and trims edge dashes', () => {
    expect(derivePanelId('  Beam   Position!! ')).toBe('beam-position');
    expect(derivePanelId('--weird__name--')).toBe('weird-name');
  });

  test('never returns an empty id', () => {
    expect(derivePanelId('')).toBe('panel');
    expect(derivePanelId('!!!')).toBe('panel');
    expect(derivePanelId('   ')).toBe('panel');
  });

  test('is idempotent on an already-clean slug', () => {
    expect(derivePanelId('lattice')).toBe('lattice');
  });
});
