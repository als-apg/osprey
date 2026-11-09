// @ts-check
/* global document -- vitest runs these in happy-dom */
/**
 * Check rows (rows.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/rows.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { demoEnvelope } from './demo.js';
import { store, busyRows, indexModel } from './state.js';
import { rowEl, rowsBlock, methodTag, paintBusy } from './rows.js';

beforeEach(() => {
  const d = demoEnvelope();
  store.env = d;
  store.R = d.results;
  store.LAYERS = d.layers;
  busyRows.clear();
  indexModel();
});

/** @param {string} name */
const idx = (name) => store.R.findIndex((r) => r.name === name);

describe('rowsBlock', () => {
  test('problem rows first, then the fix, then OK, then DISABLED', () => {
    const rows = [
      idx('agent_record_transcripts.alice'),
      idx('model_configs'),
      idx('agent_record_transcripts.logbook'),
    ];
    const out = rowsBlock(rows, 'docker compose exec agent-record ls /data/transcripts');
    expect(out.map((e) => e.className.split(' ')[0] + ':' + (e.dataset.name || ''))).toEqual([
      'row:agent_record_transcripts.logbook',
      'fix:',
      'row:agent_record_transcripts.alice',
      'row:model_configs',
    ]);
    expect(out[1].textContent).toContain('$ docker compose exec agent-record ls /data/transcripts');
  });

  test('a method shared by every row is dropped from the rows and shown once', () => {
    const rows = [idx('agent_record_transcripts.alice'), idx('agent_record_transcripts.logbook')];
    for (const e of rowsBlock(rows))
      if (e.classList.contains('row')) expect(e.querySelector('.m')?.textContent).toBe('');
    expect(methodTag(rows)?.textContent).toBe('dir stat');
  });
});

describe('rowEl', () => {
  test('provider rows offer ping, other rows re-probe', () => {
    expect(rowEl(idx('als-apg'), false).querySelector('.rp')?.textContent).toBe('ping');
    expect(rowEl(idx('disk_space'), false).querySelector('.rp')?.textContent).toBe('re-probe');
  });

  test('a busy row shows a spinner and keeps label, value and method', () => {
    busyRows.add('disk_space');
    const e = rowEl(idx('disk_space'), false);
    expect(e.classList.contains('busy')).toBe(true);
    expect(e.querySelector('.spin')).not.toBeNull();
    expect([...e.querySelectorAll('.w, .v, .m')].map((x) => x.textContent)).toEqual([
      'project volume',
      '30 % used · 1308.4 GB free',
      'statvfs',
    ]);
  });

  test('paintBusy swaps the action for a spinner in place', () => {
    document.body.textContent = '';
    const e = rowEl(idx('disk_space'), false);
    document.body.append(e);
    busyRows.add('disk_space');
    paintBusy(['disk_space']);
    expect(e.classList.contains('busy')).toBe(true);
    expect(e.querySelector('.hov .spin')).not.toBeNull();
  });
});
