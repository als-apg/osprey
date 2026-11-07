// @ts-check
/**
 * In-place navigation of a panel document (panel-navigate.js): a document
 * that says it routes urls itself is sent them by message, anything else is
 * left to a src load.
 */
import { describe, expect, test, vi } from 'vitest';

import {
  navigateInPlace,
  probeNavigateReady,
  watchNavigateReady,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-navigate.js';

/** @returns {any} an iframe stand-in whose window records what it was sent */
function fakeIframe() {
  return { ownerDocument: document, contentWindow: { postMessage: vi.fn() } };
}

describe('panel-navigate', () => {
  test('a ready document is navigated by message', () => {
    const iframe = fakeIframe();
    const state = { iframe, inPlace: true };

    expect(navigateInPlace(state, '/panel/canopy/s/sr/overview')).toBe(true);
    expect(iframe.contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-navigate', url: '/panel/canopy/s/sr/overview' },
      window.location.origin
    );
  });

  test('a document that never said it was ready is left to a src load', () => {
    const iframe = fakeIframe();
    const state = { iframe, inPlace: false };

    expect(navigateInPlace(state, '/panel/canopy/s/sr/overview')).toBe(false);
    expect(iframe.contentWindow.postMessage).not.toHaveBeenCalled();
  });

  test('a frame moved to another document is not navigated in place: its parent is not the hub', () => {
    const iframe = document.createElement('iframe');
    document.implementation.createHTMLDocument('').body.append(iframe);
    const postMessage = vi.fn();
    Object.defineProperty(iframe, 'contentWindow', { value: { postMessage } });
    const state = { iframe, inPlace: true, pendingUrl: '/panel/ariel/x' };
    expect(navigateInPlace(state, '/panel/ariel/x')).toBe(false);
    expect(postMessage).not.toHaveBeenCalled();
    expect(state.inPlace).toBe(false);
  });

  test('a url a src would refuse is not sent, and readiness is forgotten', () => {
    const iframe = fakeIframe();
    const state = { iframe, inPlace: true };

    expect(navigateInPlace(state, 'javascript:alert(1)')).toBe(false);
    expect(iframe.contentWindow.postMessage).not.toHaveBeenCalled();
    expect(state.inPlace).toBe(false);
  });

  test('a fresh document is probed and starts not ready', () => {
    const iframe = fakeIframe();
    const state = { iframe, inPlace: true };

    probeNavigateReady(state, iframe);

    expect(state.inPlace).toBe(false);
    expect(iframe.contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-navigate-probe' },
      window.location.origin
    );
  });

  test('a ready message from a watched panel frame marks it, in either name', () => {
    /** @param {any} iframe */
    const watchedState = (iframe) => {
      const state = { iframe: { ...iframe, addEventListener: vi.fn() }, inPlace: true };
      watchNavigateReady(state);
      return state;
    };
    const a = watchedState(fakeIframe());
    const b = watchedState(fakeIframe());
    expect(a.inPlace).toBe(false);
    const ready = (
      /** @type {string} */ type,
      /** @type {any} */ source,
      origin = window.location.origin
    ) => window.dispatchEvent(new MessageEvent('message', { data: { type }, origin, source }));

    ready('canopy-host-navigate-ready', b.iframe.contentWindow, 'https://evil.example');
    expect(b.inPlace).toBe(false);
    ready('canopy-host-navigate-ready', b.iframe.contentWindow);
    expect(b.inPlace).toBe(true);
    ready('osprey-navigate-ready', a.iframe.contentWindow);
    expect(a.inPlace).toBe(true);
  });
});
