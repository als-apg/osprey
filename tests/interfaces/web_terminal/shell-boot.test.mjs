// @ts-check
/**
 * Contract tests for shell-boot.js, the loading shell's watch before any
 * module: the early reads, the slow notice, the load failure, the fail-open.
 *   npx vitest run tests/interfaces/web_terminal/shell-boot.test.mjs
 *
 * Like theme-boot.js it is a non-module IIFE that exports nothing, so each
 * scenario arranges the document and re-executes the on-disk source
 * (boot-harness.mjs). shell-reveal.test.mjs pins the module half.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { runBootScript } from '../design_system/js/boot-harness.mjs';
import {
  BOOT_ATTRIBUTE,
  revealShell,
  takeEarlyRead,
} from '../../../src/osprey/interfaces/web_terminal/static/js/shell-reveal.js';

const runBoot = () => runBootScript('../../../web_terminal/static/js/shell-boot.js');

const root = document.documentElement;
const boot = () => root.getAttribute(BOOT_ATTRIBUTE);
const early = () => /** @type {any} */ (globalThis).__OSPREY_EARLY__;

/** The veil as index.html ships it, built node by node. */
function mountVeil() {
  const veil = document.createElement('div');
  veil.id = 'boot-veil';
  const status = document.createElement('p');
  status.setAttribute('data-boot-status', '');
  status.textContent = 'Loading workspace';
  const detail = document.createElement('p');
  detail.setAttribute('data-boot-detail', '');
  detail.hidden = true;
  const reload = document.createElement('button');
  reload.setAttribute('data-boot-reload', '');
  reload.hidden = true;
  veil.append(status, detail, reload);
  document.body.replaceChildren(veil);
}

const text = (/** @type {string} */ selector) =>
  document.querySelector(selector)?.textContent ?? '';
const hidden = (/** @type {string} */ selector) =>
  /** @type {HTMLElement} */ (document.querySelector(selector)).hidden;

/** @param {any} body @param {boolean} [ok] */
function stubFetch(body, ok = true) {
  /** @type {(url: string) => Promise<any>} */
  const answer = async () => ({
    ok,
    status: ok ? 200 : 503,
    statusText: ok ? 'OK' : 'Service Unavailable',
    json: async () => body,
  });
  const fetchMock = vi.fn(answer);
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/**
 * Report a failed load of `url` the way the browser does: an `error` event on
 * the element, seen on the way down. The element stays out of the document so
 * the test environment does not try to fetch it.
 * @param {'script' | 'link'} tag
 * @param {string} url
 */
function failToLoad(tag, url) {
  const el = document.createElement(tag);
  if (el instanceof HTMLScriptElement) el.src = url;
  else if (el instanceof HTMLLinkElement) {
    el.rel = 'stylesheet';
    el.href = url;
  }
  const event = new Event('error');
  Object.defineProperty(event, 'target', { value: el });
  window.dispatchEvent(event);
}

beforeEach(() => {
  root.setAttribute(BOOT_ATTRIBUTE, 'loading');
  mountVeil();
  /** @type {any} */ (globalThis).__OSPREY_EARLY__ = undefined;
  /** @type {any} */ (window).__OSPREY_PREFIX__ = '';
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('early reads', () => {
  test('starts the two reads start-up opens with, under the prefix', async () => {
    const fetchMock = stubFetch({ ok: 1 });
    /** @type {any} */ (window).__OSPREY_PREFIX__ = '/u/alice';

    runBoot();

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      '/u/alice/api/sessions/last-active',
      '/u/alice/api/panels',
    ]);
    await expect(takeEarlyRead('/api/panels')).resolves.toEqual({ ok: 1 });
    await expect(takeEarlyRead('/api/sessions/last-active')).resolves.toEqual({ ok: 1 });
  });

  test('a read the server refused fails for its taker, as fetchJSON would', async () => {
    stubFetch({}, false);

    runBoot();

    await expect(takeEarlyRead('/api/panels')).rejects.toThrow('HTTP 503: Service Unavailable');
  });

  test('does nothing on a page that declares no loading state', () => {
    const fetchMock = stubFetch({});
    root.removeAttribute(BOOT_ATTRIBUTE);

    runBoot();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(early()).toBeUndefined();
  });
});

describe('slow notice', () => {
  test('says the connection is slow after eight seconds', async () => {
    vi.useFakeTimers();
    stubFetch({});
    runBoot();

    await vi.advanceTimersByTimeAsync(7999);
    expect(hidden('[data-boot-detail]')).toBe(true);

    await vi.advanceTimersByTimeAsync(1);
    expect(text('[data-boot-detail]')).toBe(
      'Still loading. The connection to this server is slow.'
    );
    expect(hidden('[data-boot-detail]')).toBe(false);
    expect(text('[data-boot-status]')).toBe('Loading workspace');
    expect(hidden('[data-boot-reload]')).toBe(true);
  });

  test('a revealed page is not told it is slow', async () => {
    vi.useFakeTimers();
    stubFetch({});
    runBoot();
    revealShell();

    await vi.advanceTimersByTimeAsync(8000);
    expect(hidden('[data-boot-detail]')).toBe(true);
  });
});

describe('load failure', () => {
  test.each(/** @type {const} */ (['script', 'link']))(
    'a %s of the page that cannot be fetched ends the wait with a reload',
    (tag) => {
      stubFetch({});
      runBoot();

      failToLoad(tag, `${window.location.origin}/static/js/app.js`);

      expect(boot()).toBe('failed');
      expect(text('[data-boot-status]')).toBe('The workspace did not load');
      expect(text('[data-boot-detail]')).toBe(
        'A file of this page could not be fetched from the server.'
      );
      expect(hidden('[data-boot-reload]')).toBe(false);
    }
  );

  test("another origin's file does not end the wait", () => {
    stubFetch({});
    runBoot();

    failToLoad('script', 'https://cdn.example/xterm.js');

    expect(boot()).toBe('loading');
  });

  test('a failure after the reveal leaves the page revealed', () => {
    stubFetch({});
    runBoot();
    revealShell();

    failToLoad('script', `${window.location.origin}/static/js/late.js`);

    expect(boot()).toBe('revealed');
  });
});

describe('fail-open', () => {
  test('lifts the veil ten seconds after the document is parsed, whatever is pending', async () => {
    vi.useFakeTimers();
    stubFetch({});
    runBoot();
    document.dispatchEvent(new Event('DOMContentLoaded'));

    await vi.advanceTimersByTimeAsync(9999);
    expect(boot()).toBe('loading');
    await vi.advanceTimersByTimeAsync(1);
    expect(boot()).toBe('revealed');
  });

  test('leaves a failed page failed', async () => {
    vi.useFakeTimers();
    stubFetch({});
    runBoot();
    document.dispatchEvent(new Event('DOMContentLoaded'));
    failToLoad('script', `${window.location.origin}/static/js/app.js`);

    await vi.advanceTimersByTimeAsync(10000);
    expect(boot()).toBe('failed');
  });
});
