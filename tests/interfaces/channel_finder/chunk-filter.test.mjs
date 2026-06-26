/**
 * Regression tests for the in-context Explore filter/chunk logic (issue #299).
 *
 * Pure-logic guard, no DOM/network — run with zero extra dependencies:
 *   node --test tests/interfaces/channel_finder/chunk-filter.test.mjs
 *
 * The bug: chunking happened server-side BEFORE filtering, so a search match on
 * a later page read as "No channels match the filter". The fix filters the whole
 * DB first, then re-chunks the filtered set. These tests pin that behavior.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

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

  assert.equal(filtered.length, 1, 'the zebra is found across the whole DB');
  assert.equal(totalChunksFor(filtered.length, CHUNK_SIZE), 1, 'filtered set is one chunk');

  const page = pageSlice(filtered, chunkIdx, CHUNK_SIZE);
  assert.equal(page.length, 1);
  assert.equal(page[0].channel, 'SR:SPECIAL:ZEBRA', 'the match is on the visible page');
});

test('empty filter returns the full set unchanged (same reference)', () => {
  const db = makeDb();
  assert.equal(filterChannels(db, ''), db);
});

test('filter matches across name, address, and description (case-insensitive)', () => {
  const db = [
    { channel: 'SR:BPM:01:X', address: 'SR:BPM:01:X', description: 'horizontal position' },
    { channel: 'SR:DCCT', address: 'SR:CURRENT:MON', description: 'stored beam current' },
    { channel: 'BR:VAC:01', address: 'BR:VAC:01', description: 'booster vacuum gauge' },
  ];
  assert.equal(filterChannels(db, 'dcct').length, 1, 'name match');
  assert.equal(filterChannels(db, 'current:mon').length, 1, 'address match');
  assert.equal(filterChannels(db, 'vacuum').length, 1, 'description match');
  assert.equal(filterChannels(db, 'sr:').length, 2, 'shared prefix matches two');
});

test('totalChunksFor never reports zero chunks', () => {
  assert.equal(totalChunksFor(0, CHUNK_SIZE), 1);
  assert.equal(totalChunksFor(1, CHUNK_SIZE), 1);
  assert.equal(totalChunksFor(50, CHUNK_SIZE), 1);
  assert.equal(totalChunksFor(51, CHUNK_SIZE), 2);
  assert.equal(totalChunksFor(120, CHUNK_SIZE), 3);
});

test('clampChunkIdx keeps the operator on a valid page when the set shrinks', () => {
  // Was on page 3 (idx 2) of 120; a narrowing filter leaves 10 matches (1 page).
  assert.equal(clampChunkIdx(2, 10, CHUNK_SIZE), 0, 'clamped down to the only page');
  // Still valid pages stay put.
  assert.equal(clampChunkIdx(1, 120, CHUNK_SIZE), 1);
  // Negative input floors at 0.
  assert.equal(clampChunkIdx(-3, 120, CHUNK_SIZE), 0);
});

test('pageSlice returns the correct window per page', () => {
  const db = makeDb();
  assert.equal(pageSlice(db, 0, CHUNK_SIZE).length, 50);
  assert.equal(pageSlice(db, 1, CHUNK_SIZE).length, 50);
  assert.equal(pageSlice(db, 2, CHUNK_SIZE).length, 20, 'last page is the remainder');
  assert.equal(pageSlice(db, 0, CHUNK_SIZE)[0].channel, 'SR:BPM:000:X');
  assert.equal(pageSlice(db, 1, CHUNK_SIZE)[0].channel, 'SR:BPM:050:X');
});
