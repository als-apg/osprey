// @ts-check
/**
 * Unit tests for the feedback pure render/parse helpers (feedback-render.js),
 * extracted from feedback.js during the P3 retrofit. Pure logic, no DOM/network:
 *   npx vitest run tests/interfaces/channel_finder/feedback-render.test.mjs
 */

import { test, expect } from 'vitest';

import {
  _toolLabel,
  _buildSelectionPath,
  _buildCardSummary,
  _parseChannelsFromResponse,
  _parseSelections,
  _formatTime,
} from '../../../src/osprey/interfaces/channel_finder/static/js/feedback-render.js';

test('_toolLabel maps tool names to short badges', () => {
  expect(_toolLabel('mcp__cf__build_channels')).toBe('build');
  expect(_toolLabel('mcp__cf__get_channels')).toBe('get');
  expect(_toolLabel('something_else')).toBe('');
  expect(_toolLabel('')).toBe('');
  expect(_toolLabel(undefined)).toBe('');
});

test('_buildSelectionPath orders canonical levels and drops the rest to the end', () => {
  expect(_buildSelectionPath({ system: 'MAG', device: 'QF1' })).toBe('MAG / QF1');
  // Non-standard keys are appended as key:value after the ordered levels.
  expect(_buildSelectionPath({ system: 'MAG', custom: 'C' })).toBe('MAG / custom:C');
  expect(_buildSelectionPath({})).toBe('');
  expect(_buildSelectionPath(undefined)).toBe('');
});

test('_buildSelectionPath merges a trailing subfield into the field part', () => {
  expect(_buildSelectionPath({ field: 'X', subfield: 'Y' })).toBe('X:Y');
  // Subfield with no preceding field stands alone.
  expect(_buildSelectionPath({ subfield: 'Y' })).toBe('Y');
});

test('_buildSelectionPath truncates arrays longer than two entries', () => {
  expect(_buildSelectionPath({ system: ['a', 'b'] })).toBe('a, b');
  expect(_buildSelectionPath({ system: ['a', 'b', 'c', 'd'] })).toBe('a, b +2');
});

test('_buildCardSummary prefers query, then path, then tool label, then default', () => {
  expect(_buildCardSummary({ query: '  find magnets  ' })).toBe('find magnets');
  expect(_buildCardSummary({ selections: { system: 'MAG' } })).toBe('MAG');
  expect(_buildCardSummary({ tool_name: 'x_get_channels' })).toBe('get');
  expect(_buildCardSummary({})).toBe('Agent-captured search');
});

test('_parseChannelsFromResponse handles strings, objects, and the double-encoded envelope', () => {
  // Direct object.
  expect(_parseChannelsFromResponse({ channels: ['a', 'b'] })).toEqual(['a', 'b']);
  // JSON string.
  expect(_parseChannelsFromResponse('{"channels":["a"]}')).toEqual(['a']);
  // Double-encoded {result: "<json>"} envelope.
  expect(_parseChannelsFromResponse(JSON.stringify({ result: JSON.stringify({ channels: ['z'] }) }))).toEqual(['z']);
  // Channel entries may be strings or {name|pv} objects.
  expect(_parseChannelsFromResponse({ channels: [{ name: 'n' }, { pv: 'p' }, 'raw'] })).toEqual(['n', 'p', 'raw']);
});

test('_parseChannelsFromResponse returns [] for missing/invalid input', () => {
  expect(_parseChannelsFromResponse(undefined)).toEqual([]);
  expect(_parseChannelsFromResponse('not json')).toEqual([]);
  expect(_parseChannelsFromResponse({ channels: 'not-an-array' })).toEqual([]);
});

test('_parseSelections parses one key:value per line, skipping blanks and keyless lines', () => {
  expect(_parseSelections('system: MAG\ndevice: QF1')).toEqual({ system: 'MAG', device: 'QF1' });
  expect(_parseSelections('system: MAG\n\n  \nnocolon')).toEqual({ system: 'MAG' });
  expect(_parseSelections('')).toEqual({});
  expect(_parseSelections(undefined)).toEqual({});
});

test('_formatTime returns "" for empty input and a non-empty label for a valid ISO string', () => {
  expect(_formatTime('')).toBe('');
  expect(_formatTime(undefined)).toBe('');
  expect(_formatTime('2024-01-15T10:30:00Z').length).toBeGreaterThan(0);
});
