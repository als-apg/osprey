import { afterEach, describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  LABELS,
  NEUTRAL_WORDS,
  heading,
  useWords,
} from '../../../src/osprey/interfaces/channel_finder/static/js/labels.js';
import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';
import * as alarmsRender from '../../../src/osprey/interfaces/channel_finder/static/js/alarms-render.js';
import { pageHtml } from '../../../src/osprey/interfaces/channel_finder/static/js/archive-render.js';
import * as channelsRender from '../../../src/osprey/interfaces/channel_finder/static/js/channels-render.js';
import { plotButton } from '../../../src/osprey/interfaces/channel_finder/static/js/directory-api.js';
import {
  aliveness,
  fleetStatement,
} from '../../../src/osprey/interfaces/channel_finder/static/js/server-page.js';
import * as serversRender from '../../../src/osprey/interfaces/channel_finder/static/js/servers-render.js';
import { EPICS_WORDS } from './epics-words.mjs';

/** The shared route fixture, its directory label in the neutral words. */
const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8').replaceAll(
    '"All PVs"',
    '"Channels"'
  )
);

const TANGO = {
  channel: 'attribute',
  channels: 'attributes',
  server: 'device server',
  servers: 'device servers',
  directory: 'Attributes',
  servers_view: 'Device servers',
};

afterEach(() => {
  useWords(null);
  state.directory = null;
});

describe('the channel and server words', () => {
  test('each word is stated once, as a non-empty string', () => {
    expect(Object.keys(LABELS).sort()).toEqual([
      'channel',
      'channels',
      'directory',
      'server',
      'servers',
      'serversView',
    ]);
    for (const word of Object.values(LABELS)) {
      expect(typeof word).toBe('string');
      expect(word.length).toBeGreaterThan(0);
    }
  });

  test('the words cannot be changed at run time', () => {
    expect(Object.isFrozen(LABELS)).toBe(true);
    expect(Object.isFrozen(NEUTRAL_WORDS)).toBe(true);
  });

  test('before any words are served, the neutral words hold', () => {
    useWords(null);
    expect({ ...LABELS }).toEqual({
      channel: 'channel',
      channels: 'channels',
      server: 'server',
      servers: 'servers',
      directory: 'Channels',
      serversView: 'Servers',
    });
  });

  test('served words are shown', () => {
    useWords(EPICS_WORDS);
    expect({ ...LABELS }).toEqual({
      channel: 'PV',
      channels: 'PVs',
      server: 'IOC',
      servers: 'IOCs',
      directory: 'All PVs',
      serversView: 'IOCs',
    });
  });

  test('a set missing a word shows the neutral words', () => {
    useWords(EPICS_WORDS);
    useWords(/** @type {any} */ ({ ...TANGO, servers_view: '' }));
    expect(LABELS.server).toBe('server');
  });

  test('a heading capitalises the first letter and keeps the rest', () => {
    expect(heading('IOC')).toBe('IOC');
    expect(heading('server')).toBe('Server');
    expect(heading('device server')).toBe('Device server');
  });
});

describe('the words follow the target an answer comes from', () => {
  test('the descriptor shows its target’s words', () => {
    state.directory = /** @type {any} */ ({
      label: 'Attributes',
      target: 'live',
      words: { live: TANGO, va: EPICS_WORDS, standin: EPICS_WORDS },
      configured_services: [],
      canopy: false,
      ariel: false,
    });
    expect(LABELS.server).toBe('device server');
  });

  test('an answer from another target moves the words, the label and says so', () => {
    state.directory = /** @type {any} */ ({
      label: 'Attributes',
      target: 'live',
      words: { live: TANGO, va: EPICS_WORDS, standin: EPICS_WORDS },
      configured_services: [],
      canopy: false,
      ariel: false,
    });
    /** @type {string[]} */
    const moved = [];
    const listener = (/** @type {string} */ target) => moved.push(target);
    state.on('directoryWords', listener);

    state.followTarget('va');
    state.followTarget('va');

    state.off('directoryWords', listener);
    expect(moved).toEqual(['va']);
    expect(LABELS.server).toBe('IOC');
    expect(state.directory?.label).toBe('All PVs');
    expect(state.directory?.target).toBe('va');
  });
});

/** Every operator-visible string the directory's render functions build. */
function renderedStrings() {
  const fleet = ROUTES.fleet;
  const snapshot = fleet.snapshot;
  const alarms = ROUTES.alarms;
  const outcomes = ['no_snapshot', 'not_refreshed', 'cannot_list', 'stale', 'unavailable'].flatMap(
    (outcome) => {
      const said = serversRender.outcomeStatement({ ...fleet, outcome, reason: 'r' });
      return [said.message, said.detail];
    }
  );
  const readPaths = ['ok', 'down', 'unverified'].map((status) =>
    serversRender.headerHtml({ ...snapshot, read_path: { status, channel: 'SR:PROBE' } })
  );
  const fleetOutcomes = [
    'no_snapshot',
    'not_refreshed',
    'cannot_list',
    'unavailable',
    'stale',
    'not_in_snapshot',
    'other',
  ].map((outcome) =>
    fleetStatement(
      /** @type {any} */ ({ outcome, target: 'live', generated_at: null, row: null }),
      'demo-ioc-bpm'
    )
  );
  return [
    channelsRender.countHtml(1),
    channelsRender.countHtml(2),
    channelsRender.wordsText(''),
    channelsRender.tableHtml(ROUTES.search, new Set(), null, ''),
    ...alarmsRender.feedRows(alarms).map((row) => `${row.label} ${row.value}`),
    alarmsRender.listHtml(alarms.rows, { selected: null, values: null, stale: false, cut: null }),
    alarmsRender.inspectorHtml({
      row: alarms.rows[0],
      facts: null,
      factsReason: 'r',
      value: undefined,
      plot: true,
      send: false,
    }),
    pageHtml(ROUTES.archive, Date.parse(ROUTES.archive.newest_sample)),
    pageHtml({ ...ROUTES.archive, by_server: [], by_server_reason: null }, 0),
    ...outcomes,
    ...readPaths,
    serversRender.countHtml(1, 2),
    serversRender.countHtml(2, 2),
    serversRender.tableHtml(snapshot.servers, null),
    ...fleetOutcomes,
    aliveness('', undefined, undefined).reason,
    plotButton(99).title,
  ];
}

describe('nothing an operator reads names an EPICS word on a control system without one', () => {
  test('the neutral words render no PV and no IOC', () => {
    useWords(null);
    for (const text of renderedStrings()) {
      expect(text).not.toMatch(/\bPVs?\b|\bIOCs?\b/);
    }
  });

  test('a Tango control target renders its own words', () => {
    useWords(TANGO);
    const all = renderedStrings().join('\n');
    expect(all).not.toMatch(/\bPVs?\b|\bIOCs?\b/);
    expect(all).toContain('device servers');
    expect(all).toContain('attributes');
  });

  test('the EPICS words render today’s strings', () => {
    useWords(EPICS_WORDS);
    expect(channelsRender.wordsText('')).toBe('every PV');
    expect(channelsRender.countHtml(2)).toBe('<strong>2</strong> PVs');
    expect(serversRender.countHtml(2, 2)).toBe('<strong>2</strong> IOCs');
    expect(aliveness('', undefined, undefined).reason).toBe('no heartbeat PV listed for this IOC');
    expect(plotButton(99).title).toBe('Data Browser takes at most 12 PVs');
  });
});
