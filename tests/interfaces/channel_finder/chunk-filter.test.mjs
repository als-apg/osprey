/**
 * Regression tests for the in-context Explore filter/chunk logic (issue #299).
 *
 * Pure-logic guard, no DOM/network:
 *   npx vitest run tests/interfaces/channel_finder/chunk-filter.test.mjs
 *
 * The bug: chunking happened server-side BEFORE filtering, so a search match on
 * a later page read as "No channels match the filter". The fix filters the whole
 * DB first, then re-chunks the filtered set. These tests pin that behavior.
 */

import { test, expect } from 'vitest';

import {
  filterChannels,
  totalChunksFor,
  clampChunkIdx,
  pageSlice,
} from '../../../src/osprey/interfaces/channel_finder/static/js/chunk-filter.js';

const CHUNK_SIZE = 50;

// 120 channels with one unique marker parked deep on page 3 (index 110).
function makeDb() {
  const db = [];
  for (let i = 0; i < 120; i++) {
    if (i === 110) {
      db.push({
        channel: 'SR:SPECIAL:ZEBRA',
        address: 'SR:SPECIAL:ZEBRA',
        description: 'the unique zebra magnet the operator is hunting for',
      });
    } else {
      db.push({
        channel: `SR:BPM:${String(i).padStart(3, '0')}:X`,
        address: `SR:BPM:${String(i).padStart(3, '0')}:X`,
        description: `beam position monitor ${i}`,
      });
    }
  }
  return db;
}

test('#299 regression: a match deep on page 3 is found from page 1', () => {
  const db = makeDb();
  // Operator was on page 1; the OLD code filtered only that loaded chunk.
  let chunkIdx = 0;

  // New flow: filter the WHOLE db, reset to page 0, re-chunk the filtered set.
  const filtered = filterChannels(db, 'zebra');
  chunkIdx = 0;
  chunkIdx = clampChunkIdx(chunkIdx, filtered.length, CHUNK_SIZE);

  expect(filtered.length, 'the zebra is found across the whole DB').toBe(1);
  expect(totalChunksFor(filtered.length, CHUNK_SIZE), 'filtered set is one chunk').toBe(1);

  const page = pageSlice(filtered, chunkIdx, CHUNK_SIZE);
  expect(page.length).toBe(1);
  expect(page[0].channel, 'the match is on the visible page').toBe('SR:SPECIAL:ZEBRA');
});

test('empty filter returns the full set unchanged (same reference)', () => {
  const db = makeDb();
  expect(filterChannels(db, '')).toBe(db);
});

test('filter matches across name, address, and description (case-insensitive)', () => {
  const db = [
    { channel: 'SR:BPM:01:X', address: 'SR:BPM:01:X', description: 'horizontal position' },
    { channel: 'SR:DCCT', address: 'SR:CURRENT:MON', description: 'stored beam current' },
    { channel: 'BR:VAC:01', address: 'BR:VAC:01', description: 'booster vacuum gauge' },
  ];
  expect(filterChannels(db, 'dcct').length, 'name match').toBe(1);
  expect(filterChannels(db, 'current:mon').length, 'address match').toBe(1);
  expect(filterChannels(db, 'vacuum').length, 'description match').toBe(1);
  expect(filterChannels(db, 'sr:').length, 'shared prefix matches two').toBe(2);
});

test('totalChunksFor never reports zero chunks', () => {
  expect(totalChunksFor(0, CHUNK_SIZE)).toBe(1);
  expect(totalChunksFor(1, CHUNK_SIZE)).toBe(1);
  expect(totalChunksFor(50, CHUNK_SIZE)).toBe(1);
  expect(totalChunksFor(51, CHUNK_SIZE)).toBe(2);
  expect(totalChunksFor(120, CHUNK_SIZE)).toBe(3);
});

test('clampChunkIdx keeps the operator on a valid page when the set shrinks', () => {
  // Was on page 3 (idx 2) of 120; a narrowing filter leaves 10 matches (1 page).
  expect(clampChunkIdx(2, 10, CHUNK_SIZE), 'clamped down to the only page').toBe(0);
  // Still valid pages stay put.
  expect(clampChunkIdx(1, 120, CHUNK_SIZE)).toBe(1);
  // Negative input floors at 0.
  expect(clampChunkIdx(-3, 120, CHUNK_SIZE)).toBe(0);
});

test('pageSlice returns the correct window per page', () => {
  const db = makeDb();
  expect(pageSlice(db, 0, CHUNK_SIZE).length).toBe(50);
  expect(pageSlice(db, 1, CHUNK_SIZE).length).toBe(50);
  expect(pageSlice(db, 2, CHUNK_SIZE).length, 'last page is the remainder').toBe(20);
  expect(pageSlice(db, 0, CHUNK_SIZE)[0].channel).toBe('SR:BPM:000:X');
  expect(pageSlice(db, 1, CHUNK_SIZE)[0].channel).toBe('SR:BPM:050:X');
});
