// @ts-check
/**
 * Unit tests for the Middle-Layer channel grouping logic (explore-grouping.js),
 * extracted from explore-middle-layer.js during the P3 retrofit. Pure logic:
 *   npx vitest run tests/interfaces/channel_finder/explore-grouping.test.mjs
 */

import { test, expect } from 'vitest';

import { groupFieldChannels } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-grouping.js';

/** @type {Set<string>} */
const NONE = new Set();

test('groups channels by sector with positional device/common-name alignment', () => {
  const channels = ['c1', 'c2', 'c3'];
  const deviceList = [['3', '1'], ['1', '2'], ['3', '3']];
  const commonNames = ['name-1', 'name-2', 'name-3'];

  const { sectors, visibleCount } = groupFieldChannels(channels, deviceList, commonNames, NONE, NONE);

  expect(visibleCount).toBe(3);
  // Sorted ascending by sector key.
  expect(sectors.map(s => s.key)).toEqual(['1', '3']);
  expect(sectors[0].label).toBe('Sector 1');
  // Positional alignment: channel index 1 -> device '2', common 'name-2'.
  expect(sectors[0].shown[0]).toEqual({ name: 'c2', device: '2', commonName: 'name-2' });
  // Sector 3 keeps both of its channels in channel order.
  expect(sectors[1].shown.map(i => i.name)).toEqual(['c1', 'c3']);
  expect(sectors[1].total).toBe(2);
});

test('channels without a device entry fall into the _unknown group, ordered last', () => {
  const channels = ['a', 'b', 'c'];
  // Only two device entries -> third channel has no sector -> _unknown.
  const deviceList = [['5', '1'], ['2', '2']];

  const { sectors } = groupFieldChannels(channels, deviceList, null, NONE, NONE);

  expect(sectors.map(s => s.key)).toEqual(['2', '5', '_unknown']);
  expect(sectors[sectors.length - 1].key).toBe('_unknown');
  expect(sectors[sectors.length - 1].label).toBe('Unknown');
});

test('active sector/device filters restrict the visible set', () => {
  const channels = ['a', 'b', 'c'];
  const deviceList = [['3', '1'], ['1', '2'], ['3', '2']];

  const bySector = groupFieldChannels(channels, deviceList, null, new Set(['3']), NONE);
  expect(bySector.visibleCount).toBe(2);
  expect(bySector.sectors.map(s => s.key)).toEqual(['3']);

  const byDevice = groupFieldChannels(channels, deviceList, null, NONE, new Set(['2']));
  expect(byDevice.visibleCount).toBe(2);
  expect(byDevice.sectors.flatMap(s => s.shown.map(i => i.name)).sort()).toEqual(['b', 'c']);
});

test('each sector truncates at the display cap of 50 (49/50/51 boundary)', () => {
  const build = (/** @type {number} */ n) => {
    const channels = [];
    const deviceList = [];
    for (let i = 0; i < n; i++) {
      channels.push(`ch-${i}`);
      deviceList.push(['7', String(i)]);
    }
    return groupFieldChannels(channels, deviceList, null, NONE, NONE).sectors[0];
  };

  const at49 = build(49);
  expect(at49.shown.length).toBe(49);
  expect(at49.total).toBe(49);
  expect(at49.hidden).toBe(0);

  const at50 = build(50);
  expect(at50.shown.length).toBe(50);
  expect(at50.hidden).toBe(0);

  const at51 = build(51);
  expect(at51.shown.length).toBe(50);
  expect(at51.total).toBe(51);
  expect(at51.hidden).toBe(1);
});
