/**
 * The page-ready mark: one writer, and the shared start-up's rule.
 *
 *   npx vitest run tests/interfaces/design_system/js/page-ready.test.mjs
 *
 * A plain page is marked after its DOMContentLoaded handlers have run; a page
 * whose markup says `pending` is not, until it calls markPageReady() itself,
 * or until it throws, which marks it failed; an end state stands.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  PAGE_READY_ATTRIBUTE,
  markPageReady,
  markPageReadyOnceParsed,
} from '/design-system/js/page-ready.js';

const root = () => document.documentElement;

/** A stand-in document still being parsed, with its own DOMContentLoaded. */
function parsingDocument() {
  const doc = document.implementation.createHTMLDocument('');
  Object.defineProperty(doc, 'readyState', { configurable: true, value: 'loading' });
  const fireParsed = () => {
    Object.defineProperty(doc, 'readyState', { configurable: true, value: 'interactive' });
    doc.dispatchEvent(new Event('DOMContentLoaded'));
  };
  return { doc, fireParsed };
}

beforeEach(() => {
  vi.useFakeTimers();
  root().removeAttribute(PAGE_READY_ATTRIBUTE);
});

afterEach(() => {
  vi.useRealTimers();
  root().removeAttribute(PAGE_READY_ATTRIBUTE);
});

describe('markPageReady', () => {
  test('writes data-page-ready="true" on <html>', () => {
    expect(PAGE_READY_ATTRIBUTE).toBe('data-page-ready');
    markPageReady();
    expect(root().getAttribute('data-page-ready')).toBe('true');
  });

  test('a second call changes nothing', () => {
    markPageReady();
    const observer = new MutationObserver(() => {});
    observer.observe(root(), { attributes: true });
    markPageReady();
    expect(observer.takeRecords()).toEqual([]);
    observer.disconnect();
    expect(root().getAttribute('data-page-ready')).toBe('true');
  });

  test('marks the document it is given, not the current one', () => {
    const { doc } = parsingDocument();
    markPageReady(doc);
    expect(doc.documentElement.getAttribute('data-page-ready')).toBe('true');
    expect(root().hasAttribute('data-page-ready')).toBe(false);
  });
});

describe('a pending page that throws', () => {
  const uncaughtError = () => window.dispatchEvent(new ErrorEvent('error', { message: 'boom' }));
  const unhandledRejection = () => window.dispatchEvent(new Event('unhandledrejection'));

  test('an uncaught error marks a pending page failed', () => {
    root().setAttribute('data-page-ready', 'pending');
    uncaughtError();
    expect(root().getAttribute('data-page-ready')).toBe('failed');
  });

  test('an unhandled rejection marks a pending page failed', () => {
    root().setAttribute('data-page-ready', 'pending');
    unhandledRejection();
    expect(root().getAttribute('data-page-ready')).toBe('failed');
  });

  test('a page that is not pending is left alone', () => {
    for (const fire of [uncaughtError, unhandledRejection]) {
      root().removeAttribute('data-page-ready');
      fire();
      expect(root().hasAttribute('data-page-ready')).toBe(false);

      root().setAttribute('data-page-ready', 'true');
      fire();
      expect(root().getAttribute('data-page-ready')).toBe('true');
    }
  });

  test('failed stands: a later markPageReady() changes nothing', () => {
    root().setAttribute('data-page-ready', 'pending');
    uncaughtError();
    markPageReady();
    expect(root().getAttribute('data-page-ready')).toBe('failed');
  });

  test('ready stands: an error after markPageReady() changes nothing', () => {
    root().setAttribute('data-page-ready', 'pending');
    markPageReady();
    uncaughtError();
    unhandledRejection();
    expect(root().getAttribute('data-page-ready')).toBe('true');
  });

  test('a resource that fails to load does not fail the page', () => {
    root().setAttribute('data-page-ready', 'pending');
    const img = document.createElement('img');
    document.body.append(img);
    img.dispatchEvent(new Event('error'));
    img.remove();
    expect(root().getAttribute('data-page-ready')).toBe('pending');
  });

  test('a script that fails to load does not fail the page', () => {
    root().setAttribute('data-page-ready', 'pending');
    const script = document.createElement('script');
    document.body.append(script);
    script.dispatchEvent(new Event('error'));
    script.remove();
    expect(root().getAttribute('data-page-ready')).toBe('pending');
  });
});

describe('markPageReadyOnceParsed (the shared start-up)', () => {
  test('marks a plain page only after its DOMContentLoaded handlers have run', () => {
    const { doc, fireParsed } = parsingDocument();
    /** @type {(string | null)[]} */
    const seenByLaterHandler = [];
    markPageReadyOnceParsed(doc);
    // Registered after the start-up's own listener: it still runs first.
    doc.addEventListener('DOMContentLoaded', () => {
      seenByLaterHandler.push(doc.documentElement.getAttribute('data-page-ready'));
    });
    vi.runAllTimers();
    expect(doc.documentElement.hasAttribute('data-page-ready')).toBe(false);

    fireParsed();
    expect(seenByLaterHandler).toEqual([null]);
    vi.runAllTimers();
    expect(doc.documentElement.getAttribute('data-page-ready')).toBe('true');
  });

  test('called from a DOMContentLoaded handler, marks after that dispatch', () => {
    markPageReadyOnceParsed();
    expect(root().hasAttribute('data-page-ready')).toBe(false);
    vi.runAllTimers();
    expect(root().getAttribute('data-page-ready')).toBe('true');
  });

  test('leaves a pending page alone until it marks itself', () => {
    root().setAttribute('data-page-ready', 'pending');
    markPageReadyOnceParsed();
    vi.runAllTimers();
    expect(root().getAttribute('data-page-ready')).toBe('pending');

    markPageReady();
    expect(root().getAttribute('data-page-ready')).toBe('true');
  });

  test('initTheme() is the shared start-up', async () => {
    vi.resetModules();
    const { initTheme } = await import('/design-system/js/theme-manager.js');
    initTheme({ role: 'follower' });
    vi.runAllTimers();
    expect(root().getAttribute('data-page-ready')).toBe('true');
  });
});
