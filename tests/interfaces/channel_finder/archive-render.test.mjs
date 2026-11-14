// @ts-check
/**
 * The Archive markup (archive-render.js): the tiles per store, the Archived PVs
 * unknown line, the coverage links, the configuration as text, the chip.
 *   npx vitest run tests/interfaces/channel_finder/archive-render.test.mjs
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test, expect, describe } from 'vitest';
import {
  READ_ONLY_NOTE,
  configurationText,
  freshnessChip,
  pageHtml,
} from '../../../src/osprey/interfaces/channel_finder/static/js/archive-render.js';

/** @type {Record<string, any>} */
const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);
/** The deployment's own store on the VA, as the archive route answers it. */
const OWN = ROUTES.archive;
/** Eight seconds after the fixture's newest sample. */
const NOW = Date.parse(OWN.newest_sample) + 8000;

/**
 * An EPICS Archiver Appliance on `live`: it lists no channels and no newest sample.
 * @type {import('../../../src/osprey/interfaces/channel_finder/static/js/directory-api.js').ArchiveSummary}
 */
const APPLIANCE = {
  outcome: 'found',
  source: { target: 'live', service: 'epics_archiver', label: 'epics_archiver', rehearsal: false },
  archived: { count: null, of: 4750, reason: 'this archiver does not report what it holds' },
  by_ioc: [],
  by_ioc_reason: 'this archiver does not report what it holds',
  newest_sample: null,
  oldest_sample: null,
  freshness_bound_s: null,
  configuration: { archiver: { type: 'epics_archiver' } },
  targets: [{ target: 'live', archive: true }],
};

/** @param {string} html */
function parse(html) {
  const box = document.createElement('div');
  box.innerHTML = html;
  return box;
}

/** @param {HTMLElement} box */
const tiles = (box) =>
  Object.fromEntries(
    [...box.querySelectorAll('[data-tile]')].map((tile) => [
      /** @type {HTMLElement} */ (tile).dataset.tile,
      [
        tile.querySelector('.archive-tile-value')?.textContent,
        tile.querySelector('.archive-tile-sub')?.textContent,
      ],
    ])
  );

describe('tiles', () => {
  test("the VA store's tiles: counts, retention, cadence, newest sample, store", () => {
    expect(tiles(parse(pageHtml(OWN, NOW)))).toEqual({
      archived: ['4,731', 'of 4,750 · 19 not archived'],
      retention: ['30 d', 'oldest sample 2026-10-13'],
      cadence: ['10 s', 'for 48 h, then 60 s'],
      newest: ['8 s ago', 'recorder writes every 10 s'],
      store: ['mongodb_archiver', ''],
    });
  });

  test('an Appliance answer reads Archived PVs unknown with the reason, no retention or cadence', () => {
    const box = parse(pageHtml(APPLIANCE, NOW));
    expect(tiles(box)).toEqual({
      archived: ['unknown', 'this archiver does not report what it holds'],
      newest: ['unknown', ''],
      store: ['epics_archiver', ''],
    });
    expect(box.querySelector('[data-card="retention"]')).toBeNull();
    expect(box.querySelector('[data-card="coverage"]')?.textContent).toContain(
      'this archiver does not report what it holds'
    );
    expect(box.querySelector('[data-role="newest"]')).toBeNull();
  });
});

describe('coverage', () => {
  test('a row per IOC; its link and a not-archived count filtered to that IOC', () => {
    const card = /** @type {HTMLElement} */ (
      parse(pageHtml(OWN, NOW)).querySelector('[data-card="coverage"]')
    );
    const rows = [...card.querySelectorAll('tbody tr')];
    expect(rows.map((row) => [...row.querySelectorAll('td')].map((td) => td.textContent))).toEqual([
      ['sim-SR', '2,400', '2,390', '10', ''],
      ['sim-BR', '2,350', '2,341', '9', ''],
    ]);
    const links = [...rows[0].querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(links).toEqual(['#/ioc/sim-SR', '#/allpvs?ioc=sim-SR&archived=no']);
  });

  test('an IOC with every PV archived has no not-archived link', () => {
    const summary = { ...OWN, by_ioc: [{ ioc: 'sim-LTB', pvs: 4, archived: 4 }] };
    const row = parse(pageHtml(summary, NOW)).querySelector('[data-card="coverage"] tbody tr');
    expect([...(row?.querySelectorAll('a') ?? [])].map((a) => a.textContent)).toEqual(['sim-LTB']);
  });

  test('IOC names are text, never markup', () => {
    const summary = {
      ...OWN,
      by_ioc: [{ ioc: '<img src=x onerror=alert(1)>', pvs: 1, archived: 0 }],
    };
    const box = parse(pageHtml(summary, NOW));
    expect(box.querySelector('img')).toBeNull();
  });
});

describe('configuration and targets', () => {
  test('the configuration is text with Copy and the read-only note, and no write control', () => {
    const card = /** @type {HTMLElement} */ (
      parse(pageHtml(OWN, NOW)).querySelector('[data-card="configuration"]')
    );
    expect(card.querySelector('pre')?.textContent).toBe(configurationText(OWN.configuration));
    expect(configurationText(OWN.configuration).split('\n').slice(0, 3)).toEqual([
      'archiver:',
      '  type: mongodb_archiver',
      'va_archiver:',
    ]);
    expect(card.textContent).toContain(READ_ONLY_NOTE);
    expect(card.querySelectorAll('input, textarea, select, form')).toHaveLength(0);
    expect(
      [...card.querySelectorAll('button')].map((b) => /** @type {HTMLElement} */ (b).dataset.action)
    ).toEqual(['copy']);
  });

  test('each configured target says whether it has an archive', () => {
    const card = parse(pageHtml(OWN, NOW)).querySelector('[data-card="targets"]');
    const items = [...(card?.querySelectorAll('dt') ?? [])].map((dt) => [
      dt.textContent,
      dt.nextElementSibling?.textContent,
    ]);
    expect(items).toEqual([
      ['va', 'archive'],
      ['live', 'no archive configured'],
    ]);
  });
});

describe('the newest-sample chip', () => {
  test('ok within the bound, warn beyond it, absent when unknown', () => {
    expect(freshnessChip(OWN, NOW)).toEqual({ text: 'newest sample 8 s ago', tone: 'ok' });
    expect(freshnessChip(OWN, NOW + 120_000)?.tone).toBe('warn');
    expect(freshnessChip(APPLIANCE, NOW)).toBeNull();
    expect(freshnessChip({ ...APPLIANCE, newest_sample: OWN.newest_sample }, NOW)?.tone).toBe(
      'info'
    );
  });
});
