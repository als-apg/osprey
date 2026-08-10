/**
 * Unit tests for the panel side of the tile header-bar contribution contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/header-contrib.test.js
 *
 * Covers HEADER_CONTRACT_VERSION, contributeHeader()'s standalone no-op
 * (in this environment window.parent === window, i.e. the not-embedded
 * case by construction), and onHeaderAction() (origin check, id/value
 * forwarding, malformed-payload rejection).
 *
 * NOTE: header-contrib.js is imported by RELATIVE path, not the absolute
 * `/design-system/js/header-contrib.js` runtime specifier — this mirrors
 * frame-params.test.js in this directory.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  contributeHeader,
  onHeaderAction,
  HEADER_CONTRACT_VERSION,
} from '../../../../src/osprey/interfaces/design_system/static/js/header-contrib.js';

/**
 * Deliver a message event synchronously (window.postMessage queues a task,
 * which awkwardly interleaves with test teardown; dispatchEvent is the
 * established pattern for exercising message listeners in this suite).
 * @param {any} data
 * @param {string} [origin]
 */
function deliverMessage(data, origin = window.location.origin) {
  window.dispatchEvent(new MessageEvent('message', { data, origin }));
}

describe('HEADER_CONTRACT_VERSION', () => {
  test('is version 1', () => {
    expect(HEADER_CONTRACT_VERSION).toBe(1);
  });
});

describe('contributeHeader', () => {
  afterEach(() => {
    document.body.className = '';
    vi.restoreAllMocks();
  });

  test('is a no-op when the page is not embedded', () => {
    const post = vi.spyOn(window, 'postMessage');

    contributeHeader([{ kind: 'text', id: 't', text: 'hello' }]);

    expect(post).not.toHaveBeenCalled();
  });

  test('is a no-op even with body.embedded when there is no parent frame', () => {
    // body.embedded set but window.parent === window (standalone tab that
    // somehow carries the class): still must not self-message.
    document.body.classList.add('embedded');
    const post = vi.spyOn(window, 'postMessage');

    contributeHeader([{ kind: 'text', id: 't', text: 'hello' }]);

    expect(post).not.toHaveBeenCalled();
  });
});

describe('onHeaderAction', () => {
  test('forwards the item id and nav value to the callback', () => {
    const callback = vi.fn();
    onHeaderAction(callback);

    deliverMessage({ type: 'osprey-header-action', id: 'view', value: 'browse' });

    expect(callback).toHaveBeenCalledWith('view', 'browse');
  });

  test('forwards undefined value for plain button actions', () => {
    const callback = vi.fn();
    onHeaderAction(callback);

    deliverMessage({ type: 'osprey-header-action', id: 'refresh' });

    expect(callback).toHaveBeenCalledWith('refresh', undefined);
  });

  test('ignores messages from a foreign origin', () => {
    const callback = vi.fn();
    onHeaderAction(callback);

    deliverMessage({ type: 'osprey-header-action', id: 'view' }, 'https://evil.example');

    expect(callback).not.toHaveBeenCalled();
  });

  test('ignores unrelated types and malformed payloads', () => {
    const callback = vi.fn();
    onHeaderAction(callback);

    deliverMessage({ type: 'osprey-mode-change', mode: 'simple' });
    deliverMessage({ type: 'osprey-header-action' }); // no id
    deliverMessage({ type: 'osprey-header-action', id: 42 }); // non-string id
    deliverMessage(null);

    expect(callback).not.toHaveBeenCalled();
  });

  test('coerces a non-string value to undefined', () => {
    const callback = vi.fn();
    onHeaderAction(callback);

    deliverMessage({ type: 'osprey-header-action', id: 'view', value: 7 });

    expect(callback).toHaveBeenCalledWith('view', undefined);
  });
});
