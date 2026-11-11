/** Unit tests for the OPTIMIZER panel's polling cadence (poller.js). */

import { test, expect, describe, vi } from 'vitest';

import {
  ACTIVE_MS,
  DRAFT_MS,
  FAILED_MS,
  IDLE_MS,
  createPoller,
  isActive,
  pollDelay,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/poller.js';

describe('pollDelay', () => {
  test('about once a second while a run holds the machine', () => {
    for (const status of ['submitted', 'running', 'paused', 'cancelling']) {
      expect(isActive(status)).toBe(true);
      expect(pollDelay({ status })).toBe(ACTIVE_MS);
    }
    expect(ACTIVE_MS).toBe(1000);
  });
  test('a waiting draft is polled faster than idle, slower than a run', () => {
    expect(pollDelay({ status: 'created' })).toBe(DRAFT_MS);
    expect(DRAFT_MS).toBeGreaterThan(ACTIVE_MS);
    expect(DRAFT_MS).toBeLessThan(IDLE_MS);
  });
  test('idle, terminal and failed reads back off', () => {
    expect(pollDelay({ status: null })).toBe(IDLE_MS);
    expect(pollDelay({ status: 'completed' })).toBe(IDLE_MS);
    expect(pollDelay({ status: 'running', failed: true })).toBe(FAILED_MS);
  });
});

describe('createPoller', () => {
  /** A manual timer: records the delay, runs on demand. */
  function manualTimer() {
    /** @type {{ fn: () => void, ms: number }[]} */
    const pending = [];
    return {
      pending,
      setTimer: (/** @type {() => void} */ fn, /** @type {number} */ ms) => {
        const entry = { fn, ms };
        pending.push(entry);
        return entry;
      },
      clearTimer: (/** @type {any} */ entry) => {
        const i = pending.indexOf(entry);
        if (i >= 0) pending.splice(i, 1);
      },
      async fire() {
        const entry = pending.shift();
        entry?.fn();
        await vi.waitFor(() => {});
      },
    };
  }

  test('schedules each next tick at the delay the last status asks for', async () => {
    const timer = manualTimer();
    const statuses = ['created', 'running', null];
    const tick = vi.fn(async () => ({ status: statuses.shift() ?? null }));
    const poller = createPoller({ tick, setTimer: timer.setTimer, clearTimer: timer.clearTimer });
    poller.start();
    await vi.waitFor(() => expect(timer.pending).toHaveLength(1));
    expect(timer.pending[0].ms).toBe(DRAFT_MS);
    await timer.fire();
    await vi.waitFor(() => expect(timer.pending[0]?.ms).toBe(ACTIVE_MS));
    await timer.fire();
    await vi.waitFor(() => expect(timer.pending[0]?.ms).toBe(IDLE_MS));
    expect(tick).toHaveBeenCalledTimes(3);
  });

  test('a throwing tick backs off instead of stopping', async () => {
    const timer = manualTimer();
    const poller = createPoller({
      tick: async () => {
        throw new Error('boom');
      },
      setTimer: timer.setTimer,
      clearTimer: timer.clearTimer,
    });
    poller.start();
    await vi.waitFor(() => expect(timer.pending[0]?.ms).toBe(FAILED_MS));
  });

  test('now() during a tick re-runs once after it, never concurrently', async () => {
    const timer = manualTimer();
    /** @type {(v: any) => void} */
    let release = () => {};
    let concurrent = 0;
    let maxConcurrent = 0;
    const tick = vi.fn(async () => {
      concurrent += 1;
      maxConcurrent = Math.max(maxConcurrent, concurrent);
      await new Promise((resolve) => (release = resolve));
      concurrent -= 1;
      return { status: 'running' };
    });
    const poller = createPoller({ tick, setTimer: timer.setTimer, clearTimer: timer.clearTimer });
    poller.start();
    void poller.now();
    void poller.now();
    release(null);
    await vi.waitFor(() => expect(tick).toHaveBeenCalledTimes(2));
    release(null);
    await vi.waitFor(() => expect(timer.pending).toHaveLength(1));
    expect(maxConcurrent).toBe(1);
  });

  test('stop() cancels the pending tick', async () => {
    const timer = manualTimer();
    const poller = createPoller({
      tick: async () => ({ status: null }),
      setTimer: timer.setTimer,
      clearTimer: timer.clearTimer,
    });
    poller.start();
    await vi.waitFor(() => expect(timer.pending).toHaveLength(1));
    poller.stop();
    expect(timer.pending).toHaveLength(0);
  });
});
