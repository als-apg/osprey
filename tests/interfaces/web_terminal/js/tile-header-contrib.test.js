/**
 * Unit tests for the hub side of the tile header-bar contribution contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/tile-header-contrib.test.js
 *
 * Covers the receive pipeline (origin check, source→panel-id resolution
 * against live iframes, vocabulary sanitization), the per-panel contribution
 * cache vs. host-registration ordering, rendering (textContent only — no
 * markup injection), and the action round-trip back to the iframe.
 *
 * The module keeps page-lifetime state (wired listener, contribution cache),
 * matching production where it lives as long as the shell does — tests use
 * distinct panel ids to stay independent.
 */

import { test, expect, describe, beforeAll, afterEach, vi } from 'vitest';

// happy-dom provides no ResizeObserver; the module observes contribution
// hosts for overflow re-measuring. Stub before import.
beforeAll(() => {
  if (typeof globalThis.ResizeObserver === 'undefined') {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

import {
  initHeaderContrib,
  registerContribHost,
  unregisterContribHost,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js';

/** Monotonic id source so each test gets an isolated panel id. */
let seq = 0;
/** @returns {string} */
function freshPanelId() {
  seq += 1;
  return `panel-${seq}`;
}

/**
 * Create a fake service-panel iframe whose contentWindow identity the module
 * matches messages against, with a spyable postMessage for the action
 * round-trip.
 * @param {string} panelId
 */
function addPanelIframe(panelId) {
  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  iframe.dataset.panelId = panelId;
  const contentWindow = { postMessage: vi.fn() };
  Object.defineProperty(iframe, 'contentWindow', { value: contentWindow });
  document.body.appendChild(iframe);
  return { iframe, contentWindow };
}

/**
 * Deliver a contribution message as if it came from the given source window.
 * @param {any} items
 * @param {any} source
 * @param {string} [origin]
 */
function deliverContribution(items, source, origin = window.location.origin) {
  window.dispatchEvent(
    new MessageEvent('message', {
      data: { type: 'osprey-header-contribution', version: 1, items },
      origin,
      source,
    })
  );
}

initHeaderContrib();
initHeaderContrib(); // idempotent — a second call must not double-listen

afterEach(() => {
  document.body.replaceChildren();
});

describe('receive pipeline', () => {
  test('renders a contribution from a matching iframe into a registered host', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution([{ kind: 'text', id: 'name', text: 'bl4c_v2' }], contentWindow);

    expect(host.querySelector('.contrib-text')?.textContent).toBe('bl4c_v2');
    expect(host.classList.contains('has-items')).toBe(true);
  });

  test('caches a contribution that arrives before the host registers', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);

    deliverContribution([{ kind: 'text', id: 'name', text: 'early' }], contentWindow);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    expect(host.querySelector('.contrib-text')?.textContent).toBe('early');
  });

  test('a later contribution replaces the earlier one wholesale', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [
        { kind: 'button', id: 'a', label: 'A' },
        { kind: 'button', id: 'b', label: 'B' },
      ],
      contentWindow
    );
    deliverContribution([{ kind: 'button', id: 'c', label: 'C' }], contentWindow);

    const labels = [...host.querySelectorAll('.contrib-btn')].map((b) => b.textContent);
    expect(labels).toEqual(['C']);
  });

  test('an empty contribution clears the region and the has-items class', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution([{ kind: 'text', id: 'name', text: 'x' }], contentWindow);
    deliverContribution([], contentWindow);

    expect(host.children.length).toBe(0);
    expect(host.classList.contains('has-items')).toBe(false);
  });

  test('ignores messages from a foreign origin', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [{ kind: 'text', id: 'name', text: 'evil' }],
      contentWindow,
      'https://evil.example'
    );

    expect(host.children.length).toBe(0);
  });

  test('ignores messages whose source matches no panel iframe', () => {
    const panelId = freshPanelId();
    addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution([{ kind: 'text', id: 'name', text: 'spoof' }], { postMessage() {} });

    expect(host.children.length).toBe(0);
  });

  test('a panel cannot inject into another tile: identity comes from the source', () => {
    const panelA = freshPanelId();
    const panelB = freshPanelId();
    const { contentWindow: windowB } = addPanelIframe(panelB);
    addPanelIframe(panelA);
    const hostA = document.createElement('div');
    const hostB = document.createElement('div');
    registerContribHost(panelA, hostA);
    registerContribHost(panelB, hostB);

    deliverContribution([{ kind: 'text', id: 'name', text: 'from B' }], windowB);

    expect(hostA.children.length).toBe(0);
    expect(hostB.querySelector('.contrib-text')?.textContent).toBe('from B');
  });
});

describe('sanitization', () => {
  test('drops unknown kinds and malformed items, keeps valid ones', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [
        { kind: 'chip', id: 'x', text: 'unknown kind' },
        { kind: 'button', id: 'no-label' },
        { kind: 'button', label: 'no id' },
        'not-an-object',
        { kind: 'button', id: 'ok', label: 'OK' },
      ],
      contentWindow
    );

    expect(host.children.length).toBe(1);
    expect(host.querySelector('.contrib-btn')?.textContent).toBe('OK');
  });

  test('renders markup-bearing text as literal text, never as elements', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [{ kind: 'text', id: 't', text: '<img src=x onerror=alert(1)>' }],
      contentWindow
    );

    expect(host.querySelector('img')).toBeNull();
    expect(host.querySelector('.contrib-text')?.textContent).toBe(
      '<img src=x onerror=alert(1)>'
    );
  });

  test('clips oversized text and caps item count', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    const flood = Array.from({ length: 40 }, (_, i) => ({
      kind: 'text',
      id: `t${i}`,
      text: 'x'.repeat(500),
    }));
    deliverContribution(flood, contentWindow);

    expect(host.children.length).toBeLessThanOrEqual(12);
    expect(host.querySelector('.contrib-text')?.textContent?.length).toBeLessThanOrEqual(120);
  });

  test('nav entries missing id or label are dropped; empty navs vanish', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [
        { kind: 'nav', id: 'v', items: [{ id: 'a', label: 'A', active: true }, { id: 'bad' }] },
        { kind: 'nav', id: 'empty', items: [{}] },
      ],
      contentWindow
    );

    const links = [...host.querySelectorAll('.contrib-nav-link')];
    expect(links.map((l) => l.textContent)).toEqual(['A']);
    expect(links[0].classList.contains('active')).toBe(true);
    expect(host.querySelectorAll('.contrib-nav').length).toBe(1);
  });
});

describe('action round-trip', () => {
  test('button click posts osprey-header-action to the panel iframe', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    deliverContribution([{ kind: 'button', id: 'refresh', label: 'Refresh' }], contentWindow);

    /** @type {HTMLButtonElement} */ (host.querySelector('.contrib-btn')).click();

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'osprey-header-action', id: 'refresh' },
      window.location.origin
    );
  });

  test('nav click includes the entry id as value', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    deliverContribution(
      [{ kind: 'nav', id: 'view', items: [{ id: 'browse', label: 'Browse' }] }],
      contentWindow
    );

    /** @type {HTMLButtonElement} */ (host.querySelector('.contrib-nav-link')).click();

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'osprey-header-action', id: 'view', value: 'browse' },
      window.location.origin
    );
  });

  test('disabled buttons render disabled', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    deliverContribution(
      [{ kind: 'button', id: 'verify', label: 'Verify', disabled: true }],
      contentWindow
    );

    expect(/** @type {HTMLButtonElement} */ (host.querySelector('.contrib-btn')).disabled).toBe(
      true
    );
  });
});

describe('host lifecycle', () => {
  test('a rebuilt tab re-renders from the cache; the stale host is dropped', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const oldHost = document.createElement('div');
    registerContribHost(panelId, oldHost);
    deliverContribution([{ kind: 'text', id: 'name', text: 'persist' }], contentWindow);

    // Tab rebuild: the replacement registers first, then the old tab disposes.
    const newHost = document.createElement('div');
    registerContribHost(panelId, newHost);
    unregisterContribHost(panelId, oldHost);

    expect(newHost.querySelector('.contrib-text')?.textContent).toBe('persist');

    // A later contribution reaches only the live host.
    deliverContribution([{ kind: 'text', id: 'name', text: 'updated' }], contentWindow);
    expect(newHost.querySelector('.contrib-text')?.textContent).toBe('updated');
    expect(oldHost.querySelector('.contrib-text')?.textContent).toBe('persist');
  });
});
