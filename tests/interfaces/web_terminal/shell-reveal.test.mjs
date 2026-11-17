// @ts-check
/**
 * Contract tests for the module half of the hub's loading shell
 * (shell-reveal.js): the one reveal, the early reads start-up takes, and the
 * two small waits the reveal is built from.
 *   npx vitest run tests/interfaces/web_terminal/shell-reveal.test.mjs
 *
 * shell-boot.test.mjs pins the classic script that runs before any module.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import {
  BOOT_ATTRIBUTE,
  SETTLE_BUDGET_MS,
  latch,
  revealShell,
  settledWithin,
  takeEarlyRead,
} from '../../../src/osprey/interfaces/web_terminal/static/js/shell-reveal.js';

const root = document.documentElement;
const boot = () => root.getAttribute(BOOT_ATTRIBUTE);

beforeEach(() => {
  root.setAttribute(BOOT_ATTRIBUTE, 'loading');
  /** @type {any} */ (globalThis).__OSPREY_EARLY__ = undefined;
});

afterEach(() => {
  vi.useRealTimers();
});

describe('revealShell', () => {
  test('reveals a loading page once', () => {
    expect(revealShell()).toBe(true);
    expect(boot()).toBe('revealed');
    expect(revealShell()).toBe(false);
  });

  test('leaves a failed page failed', () => {
    root.setAttribute(BOOT_ATTRIBUTE, 'failed');
    expect(revealShell()).toBe(false);
    expect(boot()).toBe('failed');
  });

  test('is a no-op on a page that declares no loading state', () => {
    root.removeAttribute(BOOT_ATTRIBUTE);
    expect(revealShell()).toBe(false);
    expect(root.hasAttribute(BOOT_ATTRIBUTE)).toBe(false);
  });

  test('ends the early reads: later reads ask the server', () => {
    /** @type {any} */ (globalThis).__OSPREY_EARLY__ = { '/api/panels': Promise.resolve({}) };
    revealShell();
    expect(takeEarlyRead('/api/panels')).toBeNull();
  });
});

describe('takeEarlyRead', () => {
  test('is null for a path nobody read early', () => {
    expect(takeEarlyRead('/api/panels')).toBeNull();
    /** @type {any} */ (globalThis).__OSPREY_EARLY__ = {};
    expect(takeEarlyRead('/api/panels')).toBeNull();
    expect(takeEarlyRead('constructor')).toBeNull();
  });

  test('gives each taker its own copy of the one result', async () => {
    const body = { panels: [{ id: 'a' }] };
    /** @type {any} */ (globalThis).__OSPREY_EARLY__ = { '/api/panels': Promise.resolve(body) };

    const first = await takeEarlyRead('/api/panels');
    first.panels.push({ id: 'b' });
    const second = await takeEarlyRead('/api/panels');

    expect(second).toEqual({ panels: [{ id: 'a' }] });
    expect(second).not.toBe(body);
  });

  test('hands the taker the failure of a read that failed', async () => {
    const failed = Promise.reject(new Error('HTTP 503: Service Unavailable'));
    failed.catch(() => {});
    /** @type {any} */ (globalThis).__OSPREY_EARLY__ = { '/api/panels': failed };

    await expect(takeEarlyRead('/api/panels')).rejects.toThrow('HTTP 503');
  });
});

describe('latch', () => {
  test('opens once and stays open', async () => {
    const { opened, open } = latch();
    let settled = false;
    void opened.then(() => {
      settled = true;
    });
    await Promise.resolve();
    expect(settled).toBe(false);

    open();
    open();
    await opened;
    expect(settled).toBe(true);
  });
});

describe('settledWithin', () => {
  test('resolves as soon as every wait has settled, rejections included', async () => {
    vi.useFakeTimers();
    const failed = Promise.reject(new Error('no layout'));
    let done = false;
    void settledWithin([Promise.resolve(), failed]).then(() => {
      done = true;
    });

    await vi.advanceTimersByTimeAsync(0);
    expect(done).toBe(true);
  });

  test('gives up on a wait that never settles when the budget runs out', async () => {
    vi.useFakeTimers();
    let done = false;
    void settledWithin([new Promise(() => {})]).then(() => {
      done = true;
    });

    await vi.advanceTimersByTimeAsync(SETTLE_BUDGET_MS - 1);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(done).toBe(true);
  });
});
