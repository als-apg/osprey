// @ts-check
/**
 * Unit tests for the one hover text of a facility reference:
 *   npx vitest run tests/interfaces/design_system/facility-references.test.mjs
 */

import { test, expect } from 'vitest';

import { facilityReferenceView } from '../../../src/osprey/interfaces/design_system/static/js/facility-references.js';

test('without records the facility file was not loaded and nothing dangles', () => {
  expect(facilityReferenceView('X', null)).toEqual({
    title: 'X · facility file not loaded',
    dangling: false,
  });
});

test('an empty record list means the target is not in the facility file', () => {
  expect(facilityReferenceView('X', [])).toEqual({
    title: 'X · not in the facility file',
    dangling: true,
  });
});

test('each record gets one line of its stated fields', () => {
  const records = [{ kind: 'channel', unit: 'A', description: 'Current' }, { kind: 'device' }];

  expect(facilityReferenceView('X', records)).toEqual({
    title: 'channel · A · Current\ndevice',
    dangling: false,
  });
});
