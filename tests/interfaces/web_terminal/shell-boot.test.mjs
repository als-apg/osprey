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
  const bar = document.createElement('div');
  bar.setAttribute('data-boot-bar', '');
  veil.append(status, bar, detail, reload);
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
  if (el instanceof HTMLScriptElement) {
    if (url) el.src = url;
  } else if (el instanceof HTMLLinkElement) {
    el.rel = 'stylesheet';
    el.href = url;
  }
  const event = new Event('error');
  Object.defineProperty(event, 'target', { value: el });
  window.dispatchEvent(event);
}

beforeEach(() => {
  root.setAttribute(BOOT_ATTRIBUTE, 'loading');
  root.removeAttribute('data-page-ready');
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

describe('progress', () => {
  const FILES = ['a.js', 'b.js', 'c.js', 'd.js'].map(
    (name) => `${window.location.origin}/static/${name}`
  );
  const bar = () => /** @type {HTMLElement} */ (document.querySelector('[data-boot-bar]'));
  const filled = () => bar().style.getPropertyValue('--boot-progress');

  /** Name FILES as the page's modules; returns how the browser reports arrivals. */
  function nameFiles() {
    /** @type {(list: { getEntries: () => Array<{ name: string }> }) => void} */
    let report = () => {};
    vi.stubGlobal(
      'PerformanceObserver',
      class {
        /** @param {typeof report} callback */
        constructor(callback) {
          report = callback;
        }
        observe() {}
      }
    );
    for (const href of FILES) {
      const link = document.createElement('link');
      link.rel = 'modulepreload';
      link.href = href;
      document.head.appendChild(link);
    }
    return (/** @type {string[]} */ ...names) =>
      report({ getEntries: () => names.map((name) => ({ name })) });
  }

  afterEach(() => {
    for (const link of document.head.querySelectorAll('link[rel="modulepreload"]')) link.remove();
  });

  test('the bar is indeterminate until a file is reported', () => {
    stubFetch({});
    nameFiles();
    runBoot();

    expect(bar().hasAttribute('data-progress')).toBe(false);
  });

  test('fills by the share of named files that arrived, up to nine tenths', () => {
    stubFetch({});
    const arrive = nameFiles();
    runBoot();

    arrive(FILES[0], FILES[1]);
    expect(filled()).toBe('0.450');
    expect(bar().getAttribute('aria-valuenow')).toBe('45');

    arrive(FILES[2], FILES[3]);
    expect(filled()).toBe('0.900');
  });

  test('a file the page did not name does not move the bar', () => {
    stubFetch({});
    const arrive = nameFiles();
    runBoot();

    arrive(FILES[0]);
    arrive(`${window.location.origin}/api/panels`);

    expect(filled()).toBe('0.225');
  });

  test('ends full at the reveal', () => {
    stubFetch({});
    const arrive = nameFiles();
    runBoot();
    arrive(FILES[0]);

    revealShell();

    expect(filled()).toBe('1');
    expect(bar().getAttribute('aria-valuenow')).toBe('100');
  });

  test('a browser without resource timings keeps the indeterminate bar', () => {
    stubFetch({});
    vi.stubGlobal('PerformanceObserver', undefined);

    runBoot();

    expect(boot()).toBe('loading');
    expect(bar().hasAttribute('data-progress')).toBe(false);
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

  test('a vendor bundle of this server does not end the wait: its user has a fallback', () => {
    stubFetch({});
    runBoot();

    failToLoad(
      'script',
      `${window.location.origin}/static/_v/0123456789abcdef/vendor/dockview-core.min.js`
    );

    expect(boot()).toBe('loading');
  });

  test('an inline module of the page whose import failed ends the wait', () => {
    stubFetch({});
    runBoot();

    failToLoad('script', '');

    expect(boot()).toBe('failed');
  });

  test.each(/** @type {const} */ (['script', 'link']))(
    'a pending page whose own %s fails to load is marked failed',
    (tag) => {
      root.setAttribute('data-page-ready', 'pending');
      stubFetch({});
      runBoot();

      failToLoad(tag, `${window.location.origin}/static/js/app.js`);

      expect(root.getAttribute('data-page-ready')).toBe('failed');
    }
  );

  test('a page already marked ready keeps its mark', () => {
    root.setAttribute('data-page-ready', 'true');
    stubFetch({});
    runBoot();

    failToLoad('script', `${window.location.origin}/static/js/app.js`);

    expect(root.getAttribute('data-page-ready')).toBe('true');
  });

  test('a vendor bundle leaves the mark pending', () => {
    root.setAttribute('data-page-ready', 'pending');
    stubFetch({});
    runBoot();

    failToLoad('script', `${window.location.origin}/static/vendor/xterm.min.js`);

    expect(root.getAttribute('data-page-ready')).toBe('pending');
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
