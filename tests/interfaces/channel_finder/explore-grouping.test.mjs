// @ts-check
/**
 * Unit tests for the Middle-Layer channel grouping logic (explore-grouping.js).
 * Pure logic:
 *   npx vitest run tests/interfaces/channel_finder/explore-grouping.test.mjs
 */

import { test, expect } from 'vitest';

import { groupFieldChannels, placeChips } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-grouping.js';

/** @type {Set<string>} */
const NONE = new Set();

test('without a place list, groups channels by the bare place index', () => {
  const channels = ['c1', 'c2', 'c3'];
  const deviceList = [['3', '1'], ['1', '2'], ['3', '3']];
  const commonNames = ['name-1', 'name-2', 'name-3'];

  const { places, visibleCount } = groupFieldChannels(channels, deviceList, null, commonNames, NONE, NONE);

  expect(visibleCount).toBe(3);
  expect(places.map(p => p.key)).toEqual(['1', '3']);
  expect(places[0].label).toBe('1');
  // Positional alignment: channel index 1 -> device '2', common 'name-2'.
  expect(places[0].shown[0]).toEqual({ name: 'c2', device: '2', commonName: 'name-2' });
  // Group 3 keeps both of its channels in channel order.
  expect(places[1].shown.map(i => i.name)).toEqual(['c1', 'c3']);
  expect(places[1].total).toBe(2);
});

test('bare place indices order as numbers', () => {
  const deviceList = [[10, 1], [2, 1], [0, 1]];

  const { places } = groupFieldChannels(['a', 'b', 'c'], deviceList, undefined, null, NONE, NONE);

  expect(places.map(p => p.key)).toEqual(['0', '2', '10']);
});

test('channels without a device entry fall into the _unknown group, ordered last', () => {
  const channels = ['a', 'b', 'c'];
  // Only two device entries -> third channel has no row -> _unknown.
  const deviceList = [['5', '1'], ['2', '2']];

  const { places } = groupFieldChannels(channels, deviceList, null, null, NONE, NONE);

  expect(places.map(p => p.key)).toEqual(['2', '5', '_unknown']);
  expect(places[places.length - 1].label).toBe('Unknown');
});

test('with a place list, groups are keyed by place id and labelled by its last segment', () => {
  const channels = ['a', 'b', 'c', 'd', 'e'];
  const deviceList = [[10, 1], [0, 1], [3, 1], [2, 1], [0, 2]];
  const placeList = ['SR/SECT10', 'SR', 'SR/SECT3', 'SR/SECT2', 'SR'];

  const { places, visibleCount } = groupFieldChannels(channels, deviceList, placeList, null, NONE, NONE);

  expect(visibleCount).toBe(5);
  // The index-0 group first, then by place index: SECT10 after SECT2 and SECT3.
  expect(places.map(p => p.key)).toEqual(['SR', 'SR/SECT2', 'SR/SECT3', 'SR/SECT10']);
  expect(places.map(p => p.label)).toEqual(['SR', 'SECT2', 'SECT3', 'SECT10']);
  expect(places[0].shown.map(i => i.name)).toEqual(['b', 'e']);
});

test('index-0 groups of different places stay apart, in natural label order', () => {
  const deviceList = [[0, 1], [0, 2], [0, 3]];
  const placeList = ['M/S10', 'M', 'M/S2'];

  const { places } = groupFieldChannels(['a', 'b', 'c'], deviceList, placeList, null, NONE, NONE);

  expect(places.map(p => p.label)).toEqual(['M', 'S2', 'S10']);
});

test('a device with no place is grouped under -', () => {
  const { places } = groupFieldChannels(['a', 'b'], [[0, 1], [0, 2]], [null, null], null, NONE, NONE);

  expect(places.map(p => [p.key, p.label, p.total])).toEqual([['-', '-', 2]]);
});

test('active place/device filters restrict the visible set', () => {
  const channels = ['a', 'b', 'c'];
  const deviceList = [[3, 1], [0, 1], [3, 2]];
  const placeList = ['SR/SECT3', 'SR', 'SR/SECT3'];

  const byPlace = groupFieldChannels(channels, deviceList, placeList, null, new Set(['SR/SECT3']), NONE);
  expect(byPlace.visibleCount).toBe(2);
  expect(byPlace.places.map(p => p.key)).toEqual(['SR/SECT3']);

  const byDevice = groupFieldChannels(channels, deviceList, placeList, null, NONE, new Set(['1']));
  expect(byDevice.visibleCount).toBe(2);
  expect(byDevice.places.flatMap(p => p.shown.map(i => i.name)).sort()).toEqual(['a', 'b']);

  const byIndex = groupFieldChannels(channels, deviceList, null, null, new Set(['3']), NONE);
  expect(byIndex.places.map(p => p.key)).toEqual(['3']);
});

test('placeChips lists the groups in the order groupFieldChannels gives them', () => {
  const deviceList = [[10, 1], [0, 1], [3, 1], [2, 1], [0, 2]];
  const placeList = ['SR/SECT10', 'SR', 'SR/SECT3', 'SR/SECT2', null];
  const channels = ['a', 'b', 'c', 'd', 'e'];

  const chips = placeChips(deviceList, placeList);
  const { places } = groupFieldChannels(channels, deviceList, placeList, null, NONE, NONE);

  expect(chips).toEqual([
    { key: '-', label: '-' },
    { key: 'SR', label: 'SR' },
    { key: 'SR/SECT2', label: 'SECT2' },
    { key: 'SR/SECT3', label: 'SECT3' },
    { key: 'SR/SECT10', label: 'SECT10' },
  ]);
  expect(places.map(({ key, label }) => ({ key, label }))).toEqual(chips);

  expect(placeChips([[2, 1], [1, 1], [2, 2]], null)).toEqual([
    { key: '1', label: '1' },
    { key: '2', label: '2' },
  ]);
});

test('each group truncates at the display cap of 50 (49/50/51 boundary)', () => {
  const build = (/** @type {number} */ n) => {
    const channels = [];
    const deviceList = [];
    for (let i = 0; i < n; i++) {
      channels.push(`ch-${i}`);
      deviceList.push(['7', String(i)]);
    }
    return groupFieldChannels(channels, deviceList, null, null, NONE, NONE).places[0];
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
