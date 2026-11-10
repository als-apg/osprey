// @ts-check
/**
 * The All PVs markup (allpvs-render.js): escaping, counts, the value cells and
 * the actions each model draws.
 *   npx vitest run tests/interfaces/channel_finder/allpvs-render.test.mjs
 */

import { test, expect, describe } from 'vitest';
import {
  actionsHtml,
  countHtml,
  facetsHtml,
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

const NO_FILTERS = () => ({ origin: new Set(), ioc: new Set(), record_type: new Set() });

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
    expect(found?.getAttribute('data-alarm')).toBe('1');
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

  test('a filter in force stays listed with no count for it', () => {
    const filters = NO_FILTERS();
    filters.ioc.add('gone');
    const box = parse(facetsHtml(page(), filters));
    const kept = box.querySelector('[data-facet="ioc"][data-value="gone"]');
    expect(kept?.getAttribute('aria-pressed')).toBe('true');
    expect(kept?.textContent).toContain('0');
    expect(facetsHtml(page({ outcome: 'unavailable' }), NO_FILTERS())).toBe('');
  });

  test('the actions follow the model', () => {
    const model = { answer: page(), page: 0, selected: 0, csvStatus: '', plot: null, send: false };
    const bare = parse(actionsHtml(model));
    expect(bare.querySelector('[data-action="csv"]')).not.toBeNull();
    expect(bare.querySelector('[data-action="plot"], [data-action="send"]')).toBeNull();

    const plot = { disabled: false, label: 'Plot', title: '' };
    const none = parse(actionsHtml({ ...model, plot, send: true }));
    expect(none.querySelector('[data-action="plot"]')?.getAttribute('title')).toBe(
      'Select PVs to plot'
    );
    expect(none.querySelector('[data-action="send"]')?.hasAttribute('disabled')).toBe(true);

    const paged = parse(
      actionsHtml({ ...model, answer: page({ total: 120 }), page: 1, selected: 2 })
    );
    expect(paged.textContent).toContain('page 2 of 3');
    expect(paged.textContent).toContain('2 selected');
  });
});
