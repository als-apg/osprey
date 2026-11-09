// @ts-check
/**
 * Demo rows and the component builder (demo-data.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/demo-data.test.mjs
 */

import { test, expect, describe } from 'vitest';

import { row, demoRows, comp } from './demo-data.js';

describe('row', () => {
  test('carries the panel fields and a category from the name', () => {
    const r = row('host', 'disk', 'disk_space.root', 'project volume', 'ok', '30 %', 'statvfs');
    expect(r).toMatchObject({
      layer: 'host',
      component: 'disk',
      category: 'disk_space',
      label: 'project volume',
      method: 'statvfs',
      group: null,
    });
    expect(r.probed_at).toBeGreaterThan(0);
  });
});

describe('demoRows', () => {
  test('names are unique and statuses use the wire vocabulary', () => {
    const rows = demoRows();
    expect(new Set(rows.map((r) => r.name)).size).toBe(rows.length);
    for (const r of rows) expect(['ok', 'warning', 'error', 'skip']).toContain(r.status);
  });
});

describe('comp', () => {
  test('status is the worst of its rows; all-skip stays skip', () => {
    const rows = demoRows();
    const warn = rows.findIndex((r) => r.status === 'warning');
    const ok = rows.findIndex((r) => r.status === 'ok');
    const skip = rows.findIndex((r) => r.status === 'skip');
    expect(
      comp(rows, 'x', 'x', '', { rows: [ok], groups: [{ id: 'g', name: 'g', rows: [warn] }] })
        .status
    ).toBe('warning');
    expect(comp(rows, 'x', 'x', '', { rows: [ok, skip] }).status).toBe('ok');
    expect(comp(rows, 'x', 'x', '', { unused: [skip] }).status).toBe('skip');
  });

  test('groups carry their own status and default fix', () => {
    const rows = demoRows();
    const c = comp(rows, 'x', 'x', 'f', { groups: [{ id: 'g', name: 'g', rows: [0] }] });
    expect(c.groups[0]).toMatchObject({ id: 'g', status: rows[0].status, fix: null });
    expect(c).toMatchObject({ fact: 'f', issue: null, header: null, used_by: [], unused: [] });
  });
});
