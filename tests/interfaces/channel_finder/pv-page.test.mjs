// @ts-check
/**
 * The PV page (pv-page.js).
 *   npx vitest run tests/interfaces/channel_finder/pv-page.test.mjs
 */

import { test, expect } from 'vitest';
import {
  mountPvPage,
  unmountPvPage,
} from '../../../src/osprey/interfaces/channel_finder/static/js/pv-page.js';

test('the view exports what the router mounts and unmounts', () => {
  expect(mountPvPage).toBeTypeOf('function');
  expect(unmountPvPage).toBeTypeOf('function');
});
