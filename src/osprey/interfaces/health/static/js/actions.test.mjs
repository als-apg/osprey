// @ts-check
/* global document -- vitest runs these in happy-dom */
/**
 * Data changes: envelope, poll, re-probe, ping (actions.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/actions.test.mjs
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import { demoApi, demoEnvelope } from './demo.js';
import { store, pings, busyRows, S, DS, mcp } from './state.js';
import { applyEnvelope, reprobe, ping, poll } from './actions.js';
import { openTile } from './drawer.js';

const SKELETON = ['ring', 'headline', 'scaninfo', 'bands', 'note', 'rescan'];

beforeEach(() => {
  document.body.textContent = '';
  for (const id of SKELETON) {
    const n = document.createElement('div');
    n.id = id;
    document.body.append(n);
  }
  store.env = null;
  store.api = demoApi();
  pings.clear();
  busyRows.clear();
  mcp.data = null;
  S.band = null;
  S.key = null;
  for (const k of Object.keys(DS)) delete DS[k];
});

describe('applyEnvelope', () => {
  test('renders bands and keeps an open drawer open across a refresh', () => {
    applyEnvelope(demoEnvelope());
    expect(document.querySelectorAll('.band').length).toBe(8);
    expect(document.title).toBe('System Health');
    openTile('hooks');
    applyEnvelope(demoEnvelope());
    expect(S.key).toBe('hooks');
    expect(document.querySelector('#drawer-chain.open .dh h2')?.textContent).toBe('hooks');
  });

  test('a warming envelope shows the first-scan state', () => {
    applyEnvelope({ ...demoEnvelope(), warming: true });
    expect(document.getElementById('headline')?.textContent).toBe('first scan in progress');
  });
});

describe('re-probe and ping', () => {
  test('reprobe clears busy rows and applies the answer', async () => {
    applyEnvelope(demoEnvelope());
    const p = reprobe({ checks: ['disk_space'] }, ['disk_space']);
    expect(busyRows.has('disk_space')).toBe(true);
    await p;
    expect(busyRows.has('disk_space')).toBe(false);
  });

  test('a refused re-probe shows the reason in the note line', async () => {
    applyEnvelope(demoEnvelope());
    store.api = { ...store.api, reprobe: () => Promise.reject(new Error('refresh suppressed')) };
    await reprobe({}, ['disk_space']);
    expect(document.getElementById('note')?.textContent).toBe('refresh suppressed');
  });

  test('ping stores the provider result shown in its row', async () => {
    applyEnvelope(demoEnvelope());
    openTile('model-provider');
    await ping('als-apg');
    expect(pings.get('als-apg')?.value).toMatch(/^claude-haiku-4-5 · \d+ ms · 14 in \/ 1 out$/);
    const row = document.querySelector('.row[data-name="als-apg"] .v');
    expect(row?.textContent).toBe(pings.get('als-apg')?.value);
  });
});

describe('poll', () => {
  test('fetches, renders and schedules the next poll at interval_s', async () => {
    vi.useFakeTimers();
    const api = demoApi();
    const spy = vi.fn(api.checks);
    store.api = { ...api, checks: spy };
    poll();
    await vi.waitFor(() => expect(document.querySelectorAll('.band').length).toBe(8));
    vi.advanceTimersByTime(60_000);
    expect(spy).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });
});
