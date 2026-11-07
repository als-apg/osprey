// @ts-check
/**
 * The IOC page (ioc-page.js).
 *   npx vitest run tests/interfaces/channel_finder/ioc-page.test.mjs
 */

import { test, expect } from 'vitest';
import {
  mountIocPage,
  unmountIocPage,
} from '../../../src/osprey/interfaces/channel_finder/static/js/ioc-page.js';

test('the view exports what the router mounts and unmounts', () => {
  expect(mountIocPage).toBeTypeOf('function');
  expect(unmountIocPage).toBeTypeOf('function');
});
