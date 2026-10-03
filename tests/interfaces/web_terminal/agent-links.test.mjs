// @ts-check
/**
 * Links in an agent answer (agent-links.js):
 *   npx vitest run tests/interfaces/web_terminal/agent-links.test.mjs
 *
 * A link in an agent answer never navigates the hub page away. A link to a
 * panel this hub hosts opens that panel in the workspace; any other http(s)
 * link opens in a new tab without an opener; any other scheme does nothing.
 *
 * The click tests render through the real `createChatRenderer`, with the
 * passthrough `marked` and identity `DOMPurify` stubs chat-render.test.mjs
 * uses, so the anchors are the ones the renderer writes. The passthrough
 * parser is handed the HTML marked emits for `[BPM](panel/okf#devices/bpm)`.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  classifyAgentLink,
  wireAgentLinks,
} from '../../../src/osprey/interfaces/web_terminal/static/js/agent-links.js';
import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';

const ORIGIN = 'http://hub.test';

/** @param {string} id */
const hostsOkf = (id) => id === 'okf';

/**
 * @param {string} base
 * @param {string} prefix
 * @param {(id: string) => boolean} [isHostedPanel]
 */
function ctx(base, prefix, isHostedPanel = hostsOkf) {
  return { base, origin: ORIGIN, prefix, isHostedPanel };
}

describe('classifyAgentLink', () => {
  test('a panel link at the root opens that panel at the root-relative url', () => {
    expect(classifyAgentLink('panel/okf#devices/bpm', ctx('http://hub.test/', ''))).toEqual({
      kind: 'panel',
      panel: 'okf',
      url: '/panel/okf#devices/bpm',
    });
  });

  test('a panel link under a per-user mount keeps the prefix', () => {
    expect(
      classifyAgentLink('panel/okf#devices/bpm', ctx('http://hub.test/u/alice/', '/u/alice'))
    ).toEqual({ kind: 'panel', panel: 'okf', url: '/u/alice/panel/okf#devices/bpm' });
  });

  test('a panel sub-path keeps its query', () => {
    expect(classifyAgentLink('/panel/okf/api/concept?id=x', ctx('http://hub.test/', ''))).toEqual({
      kind: 'panel',
      panel: 'okf',
      url: '/panel/okf/api/concept?id=x',
    });
  });

  test('a panel this hub does not host opens in a new tab', () => {
    expect(classifyAgentLink('panel/other#x', ctx('http://hub.test/', ''))).toEqual({
      kind: 'window',
      url: 'http://hub.test/panel/other#x',
    });
  });

  test('a same-origin panel path outside the prefix opens in a new tab', () => {
    expect(
      classifyAgentLink('/u/bob/panel/okf', ctx('http://hub.test/u/alice/', '/u/alice'))
    ).toEqual({ kind: 'window', url: 'http://hub.test/u/bob/panel/okf' });
  });

  test('an external https link opens in a new tab', () => {
    expect(classifyAgentLink('https://example.org/doc', ctx('http://hub.test/', ''))).toEqual({
      kind: 'window',
      url: 'https://example.org/doc',
    });
  });

  test('a protocol-relative link is cross origin, never a panel', () => {
    expect(classifyAgentLink('//evil.example/panel/okf', ctx('http://hub.test/', ''))).toEqual({
      kind: 'window',
      url: 'http://evil.example/panel/okf',
    });
  });

  test('a javascript: link is inert', () => {
    expect(classifyAgentLink('javascript:alert(1)', ctx('http://hub.test/', ''))).toEqual({
      kind: 'inert',
    });
  });

  test('a data: link is inert', () => {
    expect(classifyAgentLink('data:text/html,x', ctx('http://hub.test/', ''))).toEqual({
      kind: 'inert',
    });
  });

  test('a mailto: link is inert', () => {
    expect(classifyAgentLink('mailto:a@b', ctx('http://hub.test/', ''))).toEqual({
      kind: 'inert',
    });
  });

  test('an href URL cannot parse is inert', () => {
    expect(classifyAgentLink('http://[::1', ctx('http://hub.test/', ''))).toEqual({
      kind: 'inert',
    });
  });

  test('a prefix with regex metacharacters matches only itself', () => {
    expect(classifyAgentLink('/uXalice/panel/okf', ctx('http://hub.test/', '/u.alice'))).toEqual({
      kind: 'window',
      url: 'http://hub.test/uXalice/panel/okf',
    });
  });
});

describe('wireAgentLinks', () => {
  /** @type {HTMLElement} */
  let root;
  /** @type {import('vitest').Mock<(id: string, url: string) => void>} */
  let openPanel;
  /** @type {import('vitest').Mock<(url: string, target: string, features: string) => void>} */
  let openWindow;

  beforeEach(() => {
    vi.stubGlobal('marked', { parse: (/** @type {string} */ t) => t });
    vi.stubGlobal('DOMPurify', { sanitize: (/** @type {string} */ h) => h });
    root = document.createElement('div');
    document.body.appendChild(root);
    openPanel = vi.fn();
    openWindow = vi.fn();
    wireAgentLinks(root, { prefix: '', isHostedPanel: hostsOkf, openPanel, openWindow });
  });

  afterEach(() => {
    root.remove();
    vi.unstubAllGlobals();
  });

  /**
   * Render one agent answer holding a single link and return that anchor.
   * @param {string} href
   * @returns {HTMLAnchorElement}
   */
  function agentAnchor(href) {
    const entry = createChatRenderer(root).addAgentMessage(`<p><a href="${href}">BPM</a></p>`);
    const a = entry.querySelector('a');
    if (!a) throw new Error('renderer wrote no anchor');
    return /** @type {HTMLAnchorElement} */ (a);
  }

  /**
   * @param {HTMLElement} a
   * @param {MouseEventInit} [init]
   */
  function click(a, init = {}) {
    const ev = new MouseEvent('click', { bubbles: true, cancelable: true, button: 0, ...init });
    a.dispatchEvent(ev);
    return ev;
  }

  test('a panel link opens that panel in the workspace', () => {
    const ev = click(agentAnchor('panel/okf#devices/bpm'));
    expect(openPanel).toHaveBeenCalledTimes(1);
    expect(openPanel).toHaveBeenCalledWith('okf', '/panel/okf#devices/bpm');
    expect(openWindow).not.toHaveBeenCalled();
    expect(ev.defaultPrevented).toBe(true);
  });

  test('an external link opens in a new tab without an opener', () => {
    const ev = click(agentAnchor('https://example.org/doc'));
    expect(openWindow).toHaveBeenCalledWith('https://example.org/doc', '_blank', 'noopener');
    expect(openPanel).not.toHaveBeenCalled();
    expect(ev.defaultPrevented).toBe(true);
  });

  test('a javascript: link does nothing', () => {
    const ev = click(agentAnchor('javascript:alert(1)'));
    expect(ev.defaultPrevented).toBe(true);
    expect(openPanel).not.toHaveBeenCalled();
    expect(openWindow).not.toHaveBeenCalled();
  });

  for (const key of /** @type {const} */ (['metaKey', 'ctrlKey', 'shiftKey', 'altKey'])) {
    test(`a click with ${key} is left to the browser`, () => {
      const ev = click(agentAnchor('panel/okf#devices/bpm'), { [key]: true });
      expect(ev.defaultPrevented).toBe(false);
      expect(openPanel).not.toHaveBeenCalled();
      expect(openWindow).not.toHaveBeenCalled();
    });
  }

  test('a non-primary button is left to the browser', () => {
    const ev = click(agentAnchor('panel/okf#devices/bpm'), { button: 1 });
    expect(ev.defaultPrevented).toBe(false);
    expect(openPanel).not.toHaveBeenCalled();
    expect(openWindow).not.toHaveBeenCalled();
  });

  test('an anchor outside an agent answer is not intercepted', () => {
    const a = document.createElement('a');
    a.setAttribute('href', 'panel/okf#devices/bpm');
    root.appendChild(a);
    const ev = click(a);
    expect(ev.defaultPrevented).toBe(false);
    expect(openPanel).not.toHaveBeenCalled();
  });
});
