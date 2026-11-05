/**
 * Unit tests for the shared panel link verbs (panel-links.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-links.test.mjs
 *
 * The iframe lookup is stubbed with a map of fake frames, so each test shapes
 * `contentWindow` (readable, throwing, blank) without a real navigation.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @type {Map<string, any>} */
const frames = new Map();
const copyText = vi.hoisted(() => vi.fn());

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  iframeForPanel: (/** @type {string} */ id) => frames.get(id) ?? null,
}));
vi.mock('/design-system/js/clipboard.js', () => ({ copyText }));

import {
  copyPanelLink,
  openPanelInTab,
  panelLink,
  reloadPanel,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-links.js';

/**
 * A fake iframe.
 * @param {{href?: string | (() => string), src?: string | null, reload?: () => void}} opts
 */
function fakeFrame({ href, src = null, reload = () => {} }) {
  const location = {
    reload,
    get href() {
      return typeof href === 'function' ? href() : (href ?? 'about:blank');
    },
  };
  return {
    contentWindow: { location },
    src: src ?? '',
    getAttribute: (/** @type {string} */ name) => (name === 'src' ? src : null),
  };
}

const throwing = () => {
  throw new DOMException('Blocked a frame', 'SecurityError');
};

beforeEach(() => {
  frames.clear();
  copyText.mockReset();
});
afterEach(() => vi.restoreAllMocks());

describe('panelLink', () => {
  test('reads the live location, strips frame params, keeps the rest', () => {
    frames.set(
      'a',
      fakeFrame({
        href: 'http://host/panel/ariel/?embedded=true&q=beam&popout=1&theme=dark#hit-3',
        src: '/panel/ariel/?embedded=true',
      })
    );
    expect(panelLink('a')).toBe('http://host/panel/ariel/?q=beam#hit-3');
  });

  test('a link with no frame params is returned unchanged', () => {
    frames.set('a', fakeFrame({ href: 'http://host/panel/ariel/?q=a%20b#x' }));
    expect(panelLink('a')).toBe('http://host/panel/ariel/?q=a%20b#x');
  });

  test('falls back to a root-relative src when the frame throws', () => {
    frames.set('a', fakeFrame({ href: throwing, src: '/panel/lattice/?embedded=true&theme=dark' }));
    expect(panelLink('a')).toBe(`${window.location.origin}/panel/lattice/`);
  });

  test('falls back to src when the frame is about:blank', () => {
    frames.set('a', fakeFrame({ href: 'about:blank', src: '/panel/lattice/?x=1&popout=true' }));
    expect(panelLink('a')).toBe(`${window.location.origin}/panel/lattice/?x=1`);
  });

  test('keeps an absolute src as it is', () => {
    frames.set('a', fakeFrame({ href: throwing, src: 'https://svc.example/dash?theme=light' }));
    expect(panelLink('a')).toBe('https://svc.example/dash');
  });

  test('is null without an iframe', () => {
    expect(panelLink('missing')).toBeNull();
  });

  test('is null when the frame is unreadable and has no src', () => {
    frames.set('a', fakeFrame({ href: throwing, src: null }));
    expect(panelLink('a')).toBeNull();
  });
});

describe('reloadPanel', () => {
  test('reloads a same-origin frame in place', () => {
    const reload = vi.fn();
    frames.set('a', fakeFrame({ href: 'http://host/p/', src: '/p/', reload }));
    expect(reloadPanel('a')).toBe(true);
    expect(reload).toHaveBeenCalledOnce();
  });

  test('re-sets src when the frame refuses the reload', () => {
    const frame = fakeFrame({ href: 'http://x/', src: '/p/?embedded=true', reload: throwing });
    let assigned = '';
    Object.defineProperty(frame, 'src', {
      get: () => 'resolved-absolute-value',
      set: (v) => {
        assigned = v;
      },
    });
    frames.set('a', frame);
    expect(reloadPanel('a')).toBe(true);
    expect(assigned).toBe('/p/?embedded=true');
  });

  test('is false without an iframe', () => {
    expect(reloadPanel('missing')).toBe(false);
  });
});

describe('copyPanelLink and openPanelInTab', () => {
  test('copyPanelLink copies the link', async () => {
    frames.set('a', fakeFrame({ href: 'http://host/p/?theme=dark&k=v' }));
    copyText.mockResolvedValue(true);
    await expect(copyPanelLink('a')).resolves.toBe(true);
    expect(copyText).toHaveBeenCalledWith('http://host/p/?k=v');
  });

  test('copyPanelLink passes a failed copy on', async () => {
    frames.set('a', fakeFrame({ href: 'http://host/p/' }));
    copyText.mockResolvedValue(false);
    await expect(copyPanelLink('a')).resolves.toBe(false);
  });

  test('copyPanelLink resolves false without a link', async () => {
    await expect(copyPanelLink('missing')).resolves.toBe(false);
    expect(copyText).not.toHaveBeenCalled();
  });

  test('openPanelInTab opens the link in a new tab without an opener', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    frames.set('a', fakeFrame({ href: 'http://host/p/?popout=1&k=v' }));
    expect(openPanelInTab('a')).toBe(true);
    expect(open).toHaveBeenCalledWith('http://host/p/?k=v', '_blank', 'noopener');
  });

  test('openPanelInTab is false without a link', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    expect(openPanelInTab('missing')).toBe(false);
    expect(open).not.toHaveBeenCalled();
  });
});
