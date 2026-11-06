// @ts-check
/* global document, history, location -- vitest runs these in happy-dom */
/**
 * The in-place drawer (drawer.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/drawer.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { demoEnvelope } from './demo.js';
import { store, mcp, S, DS, indexModel } from './state.js';
import { renderBands } from './overview.js';
import {
  show,
  closeDrawer,
  openTile,
  toggleTile,
  toggleBand,
  fromHash,
  drawerSignature,
} from './drawer.js';

beforeEach(() => {
  document.body.textContent = '';
  const b = document.createElement('section');
  b.id = 'bands';
  document.body.append(b);
  const d = demoEnvelope();
  store.env = d;
  store.R = d.results;
  store.LAYERS = d.layers;
  mcp.data = null;
  S.band = null;
  S.key = null;
  for (const k of Object.keys(DS)) delete DS[k];
  history.replaceState(null, '', location.pathname);
  indexModel();
  renderBands();
});

describe('open / close', () => {
  test('a tile opens its drawer under its band, selected, with a deep link', () => {
    openTile('hooks');
    expect(S).toEqual({ band: 'chain', key: 'hooks' });
    expect(document.querySelector('#band-chain.open')).not.toBeNull();
    expect(document.getElementById('tile-hooks')?.getAttribute('aria-expanded')).toBe('true');
    expect(document.querySelector('#drawer-chain .dh h2')?.textContent).toBe('hooks');
    expect(document.querySelectorAll('#drawer-chain .row').length).toBe(3);
    expect(location.hash).toBe('#hooks');
  });

  test('re-click closes; another band moves the drawer; band label opens the band', () => {
    toggleTile('hooks');
    toggleTile('hooks');
    expect(S.key).toBeNull();
    expect(location.hash).toBe('');
    openTile('hooks');
    openTile('disk');
    expect(S.band).toBe('host');
    toggleBand('data');
    expect(S.key).toBe('band:data');
    expect(location.hash).toBe('#band-data');
    closeDrawer();
    expect(document.querySelectorAll('.band.open').length).toBe(0);
  });

  test('a grouped tile carries its drill path in the hash', () => {
    openTile('telemetry-store');
    expect(location.hash).toBe('#telemetry-store/transcripts');
    expect(document.querySelector('#drawer-data .dh .sub')?.textContent).toBe(
      'agent record · clickhouse'
    );
  });
});

describe('band drawer', () => {
  test('lists every tile of the band, faulty first, groups as links', () => {
    show('data', 'band:data');
    const heads = [...document.querySelectorAll('#drawer-data .grph')].map((e) => e.textContent);
    expect(heads[0]).toContain('telemetry store');
    expect(
      [...document.querySelectorAll('#drawer-data .it .nm')].map((e) => e.textContent)
    ).toEqual(['store', 'transcripts']);
  });
});

describe('deep links', () => {
  test('fromHash opens a tile at a group and a band by id', () => {
    history.replaceState(null, '', '#telemetry-store/store');
    fromHash();
    expect(DS['telemetry-store'].path).toEqual(['telemetry-store/store']);
    history.replaceState(null, '', '#band-host');
    fromHash();
    expect(S.key).toBe('band:host');
  });

  test('drawerSignature changes when a row of the tile changes', () => {
    const a = drawerSignature('hooks');
    const i = store.R.findIndex((r) => r.name === 'hooks.osprey_limits');
    store.R[i] = { ...store.R[i], status: 'error' };
    expect(drawerSignature('hooks')).not.toBe(a);
  });
});
