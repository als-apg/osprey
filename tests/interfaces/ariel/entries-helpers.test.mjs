// @ts-check
/**
 * Unit tests for the pure, DOM-free entries helpers (entries-helpers.js),
 * extracted from entries.js/components.js during the P4 retrofit:
 *   npx vitest run tests/interfaces/ariel/entries-helpers.test.mjs
 */

import { test, expect } from 'vitest';

import {
  isImageAttachment,
  formatFileSize,
  parseEntryText,
} from '../../../src/osprey/interfaces/ariel/static/js/entries-helpers.js';

test('isImageAttachment trusts an image/* type over filename', () => {
  expect(isImageAttachment({ type: 'image/png' })).toBe(true);
  expect(isImageAttachment({ type: 'image/png', filename: 'notes.pdf' })).toBe(true);
  expect(isImageAttachment({ type: 'application/pdf' })).toBe(false);
});

test('isImageAttachment falls back to filename extension when type is missing', () => {
  expect(isImageAttachment({ filename: 'photo.jpg' })).toBe(true);
  expect(isImageAttachment({ filename: 'photo.JPEG' })).toBe(true);
  expect(isImageAttachment({ filename: 'scan.PNG' })).toBe(true);
  expect(isImageAttachment({ filename: 'diagram.svg' })).toBe(true);
  expect(isImageAttachment({ filename: 'report.pdf' })).toBe(false);
});

test('isImageAttachment handles edge cases with neither/empty type nor filename', () => {
  expect(isImageAttachment({})).toBe(false);
  expect(isImageAttachment({ filename: '' })).toBe(false);
  expect(isImageAttachment({ filename: 'noextension' })).toBe(false);
  expect(isImageAttachment({ type: '' })).toBe(false);
});

test('formatFileSize renders bytes below 1024 as B', () => {
  expect(formatFileSize(0)).toBe('0 B');
  expect(formatFileSize(1)).toBe('1 B');
  expect(formatFileSize(1023)).toBe('1023 B');
});

test('formatFileSize renders 1024..1MB-1 as KB', () => {
  expect(formatFileSize(1024)).toBe('1.0 KB');
  expect(formatFileSize(1536)).toBe('1.5 KB');
  expect(formatFileSize(1024 * 1024 - 1)).toBe('1024.0 KB');
});

test('formatFileSize renders 1MB and above as MB', () => {
  expect(formatFileSize(1024 * 1024)).toBe('1.0 MB');
  expect(formatFileSize(1024 * 1024 * 2.5)).toBe('2.5 MB');
});

test('parseEntryText splits the first line as subject and the rest as details', () => {
  expect(parseEntryText('Subject line\nBody line one\nBody line two')).toEqual({
    subject: 'Subject line',
    details: 'Body line one\nBody line two',
  });
});

test('parseEntryText falls back to the full text as details when there is no remainder', () => {
  expect(parseEntryText('Just one line')).toEqual({ subject: 'Just one line', details: 'Just one line' });
});

test('parseEntryText handles empty/malformed input', () => {
  expect(parseEntryText('')).toEqual({ subject: 'Untitled', details: '' });
  expect(parseEntryText(undefined)).toEqual({ subject: 'Untitled', details: '' });
  expect(parseEntryText('\n\n')).toEqual({ subject: 'Untitled', details: '\n\n' });
});
