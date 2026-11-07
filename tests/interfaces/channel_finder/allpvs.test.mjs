// @ts-check
/**
 * The All PVs view (allpvs.js).
 *   npx vitest run tests/interfaces/channel_finder/allpvs.test.mjs
 */

import { test, expect } from 'vitest';
import {
  mountAllPvs,
  unmountAllPvs,
} from '../../../src/osprey/interfaces/channel_finder/static/js/allpvs.js';

test('the view exports what the router mounts and unmounts', () => {
  expect(mountAllPvs).toBeTypeOf('function');
  expect(unmountAllPvs).toBeTypeOf('function');
});
