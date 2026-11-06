// @ts-check
/* global document -- vitest runs these in happy-dom */
/**
 * Overview: ring, headline, scan line, bands (overview.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/overview.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { demoEnvelope } from './demo.js';
import { store, mcp, indexModel } from './state.js';
import {
  renderRing,
  renderHeadline,
  renderScan,
  renderBands,
  renderWarming,
  renderError,
} from './overview.js';

const SKELETON = ['ring', 'headline', 'scaninfo', 'bands'];

beforeEach(() => {
  document.body.textContent = '';
  for (const id of SKELETON) {
    const n = document.createElement('div');
    n.id = id;
    document.body.append(n);
  }
  const d = demoEnvelope();
  store.env = d;
  store.R = d.results;
  store.LAYERS = d.layers;
  mcp.data = null;
  indexModel();
});

describe('bands', () => {
  test('one band per layer, one tile per component; a MINOR tile shows its issue', () => {
    renderBands();
    expect(document.querySelectorAll('.band').length).toBe(8);
    expect(document.querySelectorAll('.drawer').length).toBe(8);
    const t = /** @type {HTMLElement} */ (document.getElementById('tile-telemetry-store'));
    expect(t.className).toContain('s-MINOR');
    expect(t.querySelector('.f')?.textContent).toBe('1 / 2 transcript roots readable');
    expect(document.getElementById('tile-container-runtime')?.className).toContain('s-DISABLED');
    expect(document.querySelectorAll('.t[tabindex="0"]').length).toBe(1);
    expect(document.querySelector('#band-data .bl')?.className).toContain('s-MINOR');
  });
});

describe('ring + headline', () => {
  test('the ring centers the worst count and draws one segment per layer', () => {
    renderRing();
    expect(document.querySelector('.ctr .big')?.textContent).toBe('1');
    expect(document.querySelector('.ctr .lab')?.textContent).toBe('MINOR');
    expect(document.querySelectorAll('.seg').length).toBe(8);
  });

  test('the headline counts by severity and links the faulty tile', () => {
    renderHeadline();
    const p = /** @type {HTMLElement} */ (document.getElementById('headline'));
    expect(p.textContent).toBe(
      '1 MINOR · 0 MAJOR — telemetry store: 1 / 2 transcript roots readable'
    );
    expect(p.querySelector('a')?.getAttribute('href')).toBe('#telemetry-store');
  });
});

describe('scan line and states', () => {
  test('scan line names time, elapsed, interval, and flags stale', () => {
    store.env = { ...store.env, stale: true };
    renderScan();
    const t = document.getElementById('scaninfo')?.textContent || '';
    expect(t).toMatch(/^scan \d\d:\d\d:\d\dZ · .* · 1736 ms · every 60 s · stale$/);
  });

  test('warming and error states replace the bands', () => {
    store.env = { ...store.env, warming: true };
    renderWarming();
    expect(document.getElementById('bands')?.textContent).toBe('scanning');
    renderError('HTTP 502');
    expect(document.getElementById('bands')?.textContent).toContain('HTTP 502');
  });
});
