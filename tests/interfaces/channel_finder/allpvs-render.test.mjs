// @ts-check
/**
 * The All PVs markup (allpvs-render.js): escaping, counts, the value and alarm
 * cells, the filters, the pager and the selection bar each model draws.
 *   npx vitest run tests/interfaces/channel_finder/allpvs-render.test.mjs
 */

import { test, expect, describe } from 'vitest';
import {
  chipsHtml,
  countHtml,
  facetsHtml,
  pagerHtml,
  selectionBarHtml,
  tableHtml,
} from '../../../src/osprey/interfaces/channel_finder/static/js/allpvs-render.js';

const ROW = {
  name: 'SR:<b>X</b>',
  origin: /** @type {const} */ ('curated'),
  status: /** @type {const} */ ('Active'),
  description: '<script>x</script>',
  record_type: 'ai',
  ioc: 'ioc one',
  host: null,
  last_announced: null,
};

/** @param {Partial<import('../../../src/osprey/interfaces/channel_finder/static/js/directory-api.js').DirectoryPage>} [over] */
function page(over = {}) {
  return {
    outcome: /** @type {const} */ ('found'),
    searched: 'name matches SR',
    request: '',
    source: {
      target: /** @type {const} */ ('va'),
      service: 'sim',
      label: 'All PVs',
      rehearsal: false,
    },
    total: 1,
    rows: [ROW],
    truncated: false,
    partial: [],
    facets: { origin: [{ value: 'directory', count: 3 }], ioc: [], record_type: [] },
    facets_reason: null,
    ...over,
  };
}

/** @param {string} html */
function parse(html) {
  const box = document.createElement('div');
  box.innerHTML = html;
  return box;
}

const NO_FILTERS = () => ({
  origin: new Set(),
  ioc: new Set(),
  record_type: new Set(),
  alarm: new Set(),
});

describe('allpvs-render', () => {
  test('the count reads PV or PVs, and nothing without a total', () => {
    expect(countHtml(1)).toBe('<strong>1</strong> PV');
    expect(countHtml(5000)).toBe('<strong>5,000</strong> PVs');
    expect(countHtml(null)).toBe('');
  });

  test('directory text is escaped and the links are encoded', () => {
    const box = parse(tableHtml(page(), new Set(), null, ''));
    expect(box.querySelector('script, b')).toBeNull();
    expect(box.querySelector('td.desc')?.textContent).toBe('<script>x</script>');
    const [pv, ioc] = [...box.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(pv).toBe(`#/pv/${encodeURIComponent(ROW.name)}`);
    expect(ioc).toBe('#/ioc/ioc%20one');
  });

  test('a value cell is pending, read, silent or failed', () => {
    const cell = (/** @type {any} */ values, reason = '') =>
      parse(tableHtml(page(), new Set(), values, reason)).querySelector('td.allpvs-value');
    expect(cell(null)?.textContent).toBe('…');
    const found = cell(
      new Map([
        [
          ROW.name,
          {
            name: ROW.name,
            outcome: 'found',
            value: 2,
            timestamp: 't',
            severity: 1,
            status: 'HIGH',
          },
        ],
      ])
    );
    expect(found?.textContent).toBe('2');
    expect(found?.hasAttribute('data-alarm')).toBe(false);
    const silent = cell(
      new Map([
        [
          ROW.name,
          {
            name: ROW.name,
            outcome: 'no_answer',
            value: null,
            timestamp: null,
            severity: null,
            status: null,
          },
        ],
      ])
    );
    expect(silent?.textContent).toBe('no answer');
    expect(cell(new Map(), 'values read failed')?.getAttribute('title')).toBe('values read failed');
  });

  test('a value reads rounded with its unit, the full value and read time on hover', () => {
    const values = new Map([
      [
        ROW.name,
        {
          name: ROW.name,
          outcome: /** @type {const} */ ('found'),
          value: 46.23456,
          timestamp: '2026-11-12T08:00:00Z',
          severity: 2,
          status: 'HIHI',
          units: 'degC',
          precision: 1,
        },
      ],
    ]);
    const cell = parse(tableHtml(page(), new Set(), values, '')).querySelector('td.allpvs-value');
    expect(cell?.textContent).toBe('46.2 degC');
    expect(cell?.querySelector('.allpvs-unit')?.textContent).toBe('degC');
    expect(cell?.getAttribute('title')).toBe('46.23456 · 2026-11-12T08:00:00Z');
  });

  test('no Status column; an inactive row carries its chip, an active one nothing', () => {
    const inactive = { ...ROW, name: 'SR:OFF', status: /** @type {const} */ ('Inactive') };
    const box = parse(tableHtml(page({ rows: [ROW, inactive] }), new Set(), null, ''));
    const heads = [...box.querySelectorAll('th')].map((th) => th.textContent);
    expect(heads).toEqual(['', '', 'Name', 'Description', 'Value', 'Record', 'IOC']);
    const [active, off] = [...box.querySelectorAll('tbody tr')];
    expect(active.querySelector('.allpvs-inactive')).toBeNull();
    expect(active.textContent).not.toContain('Active');
    expect(off.querySelector('td.name .allpvs-inactive')?.textContent).toBe('Inactive');
    expect(off.classList.contains('inactive')).toBe(true);
  });

  test('the alarm cell marks a state, dots OK or none, and dashes a missing answer', () => {
    const major = {
      severity: 'MAJOR',
      status: 'HIHI_ALARM',
      latched: true,
      acknowledged: false,
      time: '2026-03-02T08:12:30Z',
    };
    const ok = { ...major, severity: 'OK', status: 'NO_ALARM', latched: false };
    const rows = [
      { ...ROW, name: 'A', alarm: major },
      { ...ROW, name: 'B', alarm: ok },
      { ...ROW, name: 'C', alarm: null },
    ];
    const box = parse(tableHtml(page({ rows }), new Set(), null, '', { reason: '' }));
    expect([...box.querySelectorAll('th')].map((th) => th.textContent)).toContain('Alarm');
    const cells = [...box.querySelectorAll('td.allpvs-alarm')];
    expect(cells[0].querySelector('.cf-sev--major')?.textContent).toBe('MAJOR');
    expect(cells[0].getAttribute('title')).toBe(
      'HIHI_ALARM · latched · not acknowledged · since 2026-03-02T08:12:30Z'
    );
    expect(cells.slice(1).map((td) => td.textContent)).toEqual(['·', '·']);

    const missing = parse(
      tableHtml(page(), new Set(), null, '', { reason: 'alarm feed still loading' })
    ).querySelector('td.allpvs-alarm');
    expect(missing?.textContent).toBe('—');
    expect(missing?.getAttribute('title')).toBe('alarm feed still loading');
    expect(missing?.querySelector('.cf-sev')).toBeNull();

    const off = parse(tableHtml(page({ rows }), new Set(), null, ''));
    expect(off.querySelector('td.allpvs-alarm')).toBeNull();
    expect([...off.querySelectorAll('th')].map((th) => th.textContent)).not.toContain('Alarm');
  });

  test('the sidebar lists Alarm first, each severity with its mark', () => {
    const one = [{ value: 'v', count: 1 }];
    const alarm = [
      { value: 'MAJOR', count: 2 },
      { value: 'none', count: 5 },
    ];
    const facets = { origin: one, ioc: one, record_type: one, alarm };
    const box = parse(facetsHtml(page({ facets }), NO_FILTERS()));
    const groups = [...box.querySelectorAll('.allpvs-facet-group')];
    expect(groups.map((g) => g.querySelector('.osprey-section-label')?.textContent)).toEqual([
      'Alarm',
      'Origin',
      'IOC',
      'Record type',
    ]);
    const marks = [...groups[0].querySelectorAll('.cf-sev')];
    expect(marks.map((m) => [m.className, m.textContent])).toEqual([
      ['cf-sev cf-sev--major', 'MAJOR'],
      ['cf-sev cf-sev--none', 'No alarm'],
    ]);
  });

  test('a chip per filter in force, each removing its own value, then clear all', () => {
    const filters = NO_FILTERS();
    filters.ioc.add('sim-SR');
    filters.alarm.add('none');
    const box = parse(chipsHtml(filters, false));
    const chips = [...box.querySelectorAll('.allpvs-chip')];
    expect(chips.map((c) => c.firstChild?.textContent)).toEqual(['Alarm: No alarm', 'IOC: sim-SR']);
    const remove = chips[1].querySelector('button');
    expect([remove?.dataset.facet, remove?.dataset.value]).toEqual(['ioc', 'sim-SR']);
    expect(box.querySelector('[data-action="clear-all"]')).not.toBeNull();
    expect(chipsHtml(NO_FILTERS(), false)).toBe('');
    expect(parse(chipsHtml(NO_FILTERS(), true)).textContent).toBe('clear all');
  });

  test('a filter in force stays listed with no count for it', () => {
    const filters = NO_FILTERS();
    filters.ioc.add('gone');
    const box = parse(facetsHtml(page(), filters));
    const kept = box.querySelector('[data-facet="ioc"][data-value="gone"]');
    expect(kept?.getAttribute('aria-pressed')).toBe('true');
    expect(kept?.textContent).toContain('0');
    expect(facetsHtml(page({ outcome: 'unavailable' }), NO_FILTERS())).toBe('');
  });

  test('the pager names the rows shown and the page', () => {
    const rows = Array.from({ length: 50 }, (_, i) => ({ ...ROW, name: `PV:${i}` }));
    const second = parse(pagerHtml(page({ total: 4750, rows }), 1));
    expect(second.textContent).toContain('rows 51–100 of 4,750');
    expect(second.textContent).toContain('page 2 of 95');
    expect(second.querySelector('[data-action="prev"]')?.hasAttribute('disabled')).toBe(false);
    const only = parse(pagerHtml(page(), 0));
    expect(only.textContent).toContain('rows 1–1 of 1');
    expect(only.querySelector('[data-action="next"]')?.hasAttribute('disabled')).toBe(true);
    expect(pagerHtml(page({ rows: [] }), 0)).toBe('');
  });

  test('the selection bar shows only with a selection, and follows the model', () => {
    expect(selectionBarHtml({ selected: 0, plot: null, send: true })).toBe('');

    const bare = parse(selectionBarHtml({ selected: 2, plot: null, send: false }));
    expect(bare.textContent).toContain('2 selected');
    expect(bare.querySelector('[data-action="plot"], [data-action="send"]')).toBeNull();
    const actions = [...bare.querySelectorAll('button')].map((b) => b.dataset.action);
    expect(actions).toEqual(['copy-names', 'clear']);

    const plot = { disabled: true, label: 'Plot (max 12)', title: 'at most 12' };
    const full = parse(selectionBarHtml({ selected: 13, plot, send: true }));
    expect([...full.querySelectorAll('button')].map((b) => b.textContent)).toEqual([
      'Plot (max 12)',
      'Send to assistant',
      'Copy names',
      'Clear',
    ]);
    expect(full.querySelector('[data-action="plot"]')?.hasAttribute('disabled')).toBe(true);
  });
});
