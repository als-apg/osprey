// @ts-check
/**
 * The service panel iframe's sandbox: exactly the tokens a hub tile grants.
 * `allow-downloads` lets a panel hand the operator a file (a saved record,
 * an exported table) from inside its tile.
 */
import { describe, expect, it, vi } from 'vitest';

const adopted = vi.hoisted(() => vi.fn());
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  adoptIframe: adopted,
}));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js', () => ({
  sendThemeToIframe: vi.fn(),
  sendSessionToIframe: vi.fn(),
  sendModeToIframe: vi.fn(),
  buildEmbedSrc: (/** @type {string} */ url) => url,
}));

if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = /** @type {any} */ (
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  );
}

import { createPanelIframe } from '../../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-factory.js';

describe('createPanelIframe', () => {
  it('sandboxes the tile with exactly the granted tokens, downloads included', () => {
    /** @type {{url: string | null, pendingUrl: string | null, iframe: HTMLIFrameElement | null}} */
    const state = { url: '/svc/tuning', pendingUrl: null, iframe: null };
    createPanelIframe(
      { id: 'tuning', label: 'Tuning' },
      'tuning',
      state,
      document.createElement('div')
    );

    const iframe = /** @type {HTMLIFrameElement} */ (state.iframe);
    expect(adopted).toHaveBeenCalledWith('tuning', iframe, { title: 'Tuning' });
    const tokens = (iframe.getAttribute('sandbox') ?? '').split(/\s+/).filter(Boolean).sort();
    expect(tokens).toEqual([
      'allow-downloads',
      'allow-forms',
      'allow-modals',
      'allow-popups',
      'allow-same-origin',
      'allow-scripts',
    ]);
  });

  it('builds no iframe before the panel has a URL', () => {
    const state = { url: null, pendingUrl: null, iframe: null };
    createPanelIframe(undefined, 'tuning', state, document.createElement('div'));
    expect(state.iframe).toBeNull();
  });
});
