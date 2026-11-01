// @ts-check
/**
 * Which panels are served (panel-availability.js):
 *   npx vitest run tests/interfaces/web_terminal/panel-availability.test.mjs
 *
 * What a reader of a panel's routes gets: its start when the panel becomes
 * served, now included, its stop when it stops, and each once per change. Who
 * writes it is pinned in panel-manager.test.mjs, what the readers do with it in
 * sidebar-target.test.mjs and the browser suite.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/panel-availability.js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/panel-availability.js')} */
let mod;

beforeEach(async () => {
  vi.resetModules();
  mod = await import(MODULE);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('whilePanelServed', () => {
  test('nothing runs while the panel is not served', () => {
    const start = vi.fn();
    const stop = vi.fn();
    mod.whilePanelServed('artifacts', start, stop);
    expect(start).not.toHaveBeenCalled();
    expect(stop).not.toHaveBeenCalled();
  });

  test('a follower that arrives late starts at once', () => {
    mod.setPanelServed('artifacts', true);
    const start = vi.fn();
    mod.whilePanelServed('artifacts', start, vi.fn());
    expect(start).toHaveBeenCalledTimes(1);
  });

  test('each change runs its half once; a repeat runs nothing', () => {
    const start = vi.fn();
    const stop = vi.fn();
    mod.whilePanelServed('artifacts', start, stop);

    mod.setPanelServed('artifacts', true);
    mod.setPanelServed('artifacts', true);
    expect(start).toHaveBeenCalledTimes(1);

    mod.setPanelServed('artifacts', false);
    mod.setPanelServed('artifacts', false);
    expect(stop).toHaveBeenCalledTimes(1);

    mod.setPanelServed('artifacts', true);
    expect(start).toHaveBeenCalledTimes(2);
  });

  test('another panel changing runs nothing', () => {
    const start = vi.fn();
    mod.whilePanelServed('artifacts', start, vi.fn());
    mod.setPanelServed('system-health', true);
    expect(start).not.toHaveBeenCalled();
  });

  test('the returned function stops following, and calls neither half', () => {
    const start = vi.fn();
    const stop = vi.fn();
    const unfollow = mod.whilePanelServed('artifacts', start, stop);
    unfollow();
    mod.setPanelServed('artifacts', true);
    mod.setPanelServed('artifacts', false);
    expect(start).not.toHaveBeenCalled();
    expect(stop).not.toHaveBeenCalled();
  });

  test('a follower that throws costs only itself', () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const healthy = vi.fn();
    mod.whilePanelServed(
      'artifacts',
      () => {
        throw new Error('boom');
      },
      vi.fn()
    );
    mod.whilePanelServed('artifacts', healthy, vi.fn());

    expect(() => mod.setPanelServed('artifacts', true)).not.toThrow();
    expect(healthy).toHaveBeenCalledTimes(1);
    expect(errors).toHaveBeenCalledTimes(1);
  });
});
