// @ts-check
/**
 * Zooming one pane (pane-zoom.js): Ctrl+wheel and a pinch over a pane zoom
 * that pane alone and leave the browser's zoom to everywhere else; a zoom
 * belongs to a kind of pane and is kept per user; a docked frame is placed so
 * that its zoomed box still fills its tile; the ⋯ row and the bar chip.
 *
 *   npx vitest run tests/interfaces/web_terminal/pane-zoom.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/pane-zoom.js')} */
let zoom;

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  document.body.innerHTML = `
    <div class="terminal-panel"><div id="operator-container"><p class="msg">hi</p></div></div>
    <nav class="sidebar"><span class="row">Sessions</span></nav>`;
  zoom = await import(`${JS}/pane-zoom.js`);
});

afterEach(() => {
  zoom.resetPaneZoom();
  document.body.innerHTML = '';
});

/**
 * @param {Element} target  @param {Partial<WheelEventInit>} init
 * @returns {WheelEvent} the dispatched event
 */
function wheel(target, init) {
  const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, ...init });
  // happy-dom's WheelEvent drops the modifier keys of its init dictionary.
  Object.defineProperty(event, 'ctrlKey', { value: Boolean(init.ctrlKey) });
  Object.defineProperty(event, 'metaKey', { value: Boolean(init.metaKey) });
  target.dispatchEvent(event);
  return event;
}

const $ = (/** @type {string} */ sel) => /** @type {HTMLElement} */ (document.querySelector(sel));

describe('what a zoom belongs to', () => {
  test('the Agent pane, each panel by its id, and each artifact by itself, at 100% until zoomed', () => {
    expect(zoom.zoomKindOf('terminal')).toBe('agent');
    expect(zoom.zoomKindOf('logbook')).toBe('logbook');
    zoom.setPaneZoom('artifact:a1', 1.25);
    expect(zoom.paneZoom('artifact:a1')).toBe(1.25);
    expect(zoom.paneZoom('artifact:b2')).toBe(1);
    expect(zoom.paneZoom('logbook')).toBe(1);
  });

  test('the Artifacts tile zooms as the artifact it shows, and as itself while it shows none', () => {
    zoom.setPaneZoom('artifact:a1', 1.5);
    zoom.setPaneZoom('artifacts', 0.9);
    zoom.setGalleryArtifact('artifacts', 'artifact:a1');
    expect(zoom.paneZoom('artifacts')).toBe(1.5);
    zoom.setPaneZoom('artifacts', 1.75);
    expect(zoom.paneZoom('artifact:a1')).toBe(1.75);
    zoom.setGalleryArtifact('artifacts', 'artifact:b2');
    expect(zoom.paneZoom('artifacts')).toBe(1);
    zoom.setGalleryArtifact('artifacts', null);
    expect(zoom.paneZoom('artifacts')).toBe(0.9);
  });

  test('a new artifact on the Artifacts tile re-zooms its frame and tells the bar', () => {
    const frame = document.createElement('iframe');
    frame.className = 'panel-iframe';
    frame.dataset.panelId = 'artifacts';
    document.body.append(frame);
    zoom.sizeFrame(frame, 0, 0, 600, 400);
    zoom.setPaneZoom('artifact:a1', 1.5);
    const told = vi.fn();
    zoom.onPaneZoomChange(told);

    zoom.setGalleryArtifact('artifacts', 'artifact:a1');
    expect([frame.style.zoom, frame.style.width]).toEqual(['1.5', '400px']);
    expect(told).toHaveBeenCalledExactlyOnceWith('artifact:a1');

    zoom.setGalleryArtifact('artifacts', 'artifact:b2');
    expect([frame.style.zoom, frame.style.width]).toEqual(['', '600px']);
  });

  test('a zoom is clamped to 50–200% and kept to the percent', () => {
    zoom.setPaneZoom('logbook', 5);
    expect(zoom.paneZoom('logbook')).toBe(2);
    zoom.setPaneZoom('logbook', 0.1);
    expect(zoom.paneZoom('logbook')).toBe(0.5);
    zoom.setPaneZoom('logbook', 1.23456);
    expect(zoom.paneZoom('logbook')).toBe(1.23);
  });
});

describe('the stored zooms', () => {
  test('only the most recently set are kept; setting one again makes it recent', () => {
    zoom.setPaneZoom('logbook', 1.1);
    for (let i = 0; i < zoom.MAX_STORED - 1; i++) zoom.setPaneZoom(`artifact:x${i}`, 1.25);
    zoom.setPaneZoom('logbook', 1.25);
    zoom.setPaneZoom('artifact:new', 1.5);
    const stored = JSON.parse(localStorage.getItem(zoom.ZOOM_STORAGE_KEY) ?? '{}');
    expect(Object.keys(stored)).toHaveLength(zoom.MAX_STORED);
    expect(stored['artifact:x0']).toBeUndefined();
    expect(stored.logbook).toBe(1.25);
  });

  test('are kept for bob only; another user’s are never read; 100% is not stored', async () => {
    localStorage.setItem(zoom.ZOOM_STORAGE_KEY, JSON.stringify({ logbook: 1.5 }));
    document.documentElement.dataset.ospreyStorageScope = 'bob';
    expect(zoom.paneZoom('logbook')).toBe(1);

    zoom.setPaneZoom('terminal', 1.1);
    zoom.setPaneZoom('logbook', 1.5);
    zoom.setPaneZoom('logbook', 1);

    const stored = localStorage.getItem(`${zoom.ZOOM_STORAGE_KEY}--bob`);
    expect(JSON.parse(stored ?? 'null')).toEqual({ agent: 1.1 });
    vi.resetModules();
    const again = await import(`${JS}/pane-zoom.js`);
    expect(again.paneZoom('terminal')).toBe(1.1);
  });
});

describe('Ctrl+wheel and pinch', () => {
  beforeEach(() => zoom.initPaneZoom());

  test('a mouse notch over the Agent steps through the browser’s levels, and the chat zooms', () => {
    const msg = $('.msg');
    expect(wheel(msg, { ctrlKey: true, deltaY: -100 }).defaultPrevented).toBe(true);
    expect(zoom.paneZoom('terminal')).toBe(1.1);
    wheel(msg, { ctrlKey: true, deltaY: -100 });
    expect(zoom.paneZoom('terminal')).toBe(1.25);
    expect($('#operator-container').style.zoom).toBe('1.25');
    wheel(msg, { ctrlKey: true, deltaY: 100 });
    wheel(msg, { ctrlKey: true, deltaY: 100 });
    expect(zoom.paneZoom('terminal')).toBe(1);
    expect($('#operator-container').style.zoom).toBe('');
  });

  test('a line-mode notch (Firefox) steps as well', () => {
    wheel($('.msg'), { ctrlKey: true, deltaY: 3, deltaMode: 1 });
    expect(zoom.paneZoom('terminal')).toBe(0.9);
  });

  test('a Firefox mouse notch steps one level: the unit is asked for before the delta', () => {
    // Firefox answers in lines to a page that reads deltaMode first, and in
    // pixels (3 lines of text, under a notch's 50) to one that reads a delta first.
    const event = new WheelEvent('wheel', { bubbles: true, cancelable: true });
    let askedMode = false;
    Object.defineProperty(event, 'ctrlKey', { value: true });
    Object.defineProperty(event, 'deltaMode', {
      get: () => ((askedMode = true), 1),
    });
    Object.defineProperty(event, 'deltaY', { get: () => (askedMode ? -3 : -48) });
    $('.msg').dispatchEvent(event);
    expect(zoom.paneZoom('terminal')).toBe(1.1);
  });

  test('a pinch (small Ctrl+wheel deltas) zooms smoothly', () => {
    wheel($('.msg'), { ctrlKey: true, deltaY: -10 });
    expect(zoom.paneZoom('terminal')).toBe(Math.round(Math.exp(0.1) * 100) / 100);
  });

  test('on macOS Cmd+wheel zooms a pane too; elsewhere Cmd (Super) +wheel is left alone', () => {
    const platform = Object.getOwnPropertyDescriptor(navigator, 'platform');
    try {
      Object.defineProperty(navigator, 'platform', { value: 'MacIntel', configurable: true });
      expect(wheel($('.msg'), { metaKey: true, deltaY: -100 }).defaultPrevented).toBe(true);
      expect(zoom.paneZoom('terminal')).toBe(1.1);
      Object.defineProperty(navigator, 'platform', { value: 'Linux x86_64', configurable: true });
      expect(wheel($('.msg'), { metaKey: true, deltaY: -100 }).defaultPrevented).toBe(false);
      expect(zoom.paneZoom('terminal')).toBe(1.1);
    } finally {
      if (platform) Object.defineProperty(navigator, 'platform', platform);
      else delete (/** @type {any} */ (navigator).platform);
    }
  });

  test('a plain wheel scrolls, and Ctrl+wheel outside a pane is the browser’s zoom', () => {
    expect(wheel($('.msg'), { deltaY: -100 }).defaultPrevented).toBe(false);
    expect(wheel($('.row'), { ctrlKey: true, deltaY: -100 }).defaultPrevented).toBe(false);
    expect(zoom.paneZoom('terminal')).toBe(1);
  });

  test('Safari’s pinch: scale from where the gesture began', () => {
    zoom.setPaneZoom('terminal', 1.1);
    const gesture = (/** @type {string} */ type, /** @type {number} */ scale) => {
      const event = Object.assign(new Event(type, { bubbles: true, cancelable: true }), { scale });
      $('.msg').dispatchEvent(event);
      return event;
    };
    expect(gesture('gesturestart', 1).defaultPrevented).toBe(true);
    gesture('gesturechange', 1.5);
    expect(zoom.paneZoom('terminal')).toBe(1.65);
  });

  test('inside a panel frame on this origin, Ctrl+wheel zooms that panel', () => {
    const frame = document.createElement('iframe');
    frame.className = 'panel-iframe';
    frame.dataset.panelId = 'logbook';
    document.body.append(frame);
    frame.dispatchEvent(new Event('load'));
    const inner = /** @type {Document} */ (frame.contentDocument);
    expect(wheel(inner.body, { ctrlKey: true, deltaY: -100 }).defaultPrevented).toBe(true);
    expect(zoom.paneZoom('logbook')).toBe(1.1);
    expect(zoom.paneZoom('terminal')).toBe(1);
  });
});

describe('a panel frame', () => {
  /** @returns {HTMLIFrameElement} */
  const frame = (id = 'logbook') => {
    const f = document.createElement('iframe');
    f.className = 'panel-iframe';
    f.dataset.panelId = id;
    document.body.append(f);
    return f;
  };

  test('docked: its zoomed box fills the tile, and a new zoom re-places it', () => {
    const f = frame();
    zoom.sizeFrame(f, 10, 50, 600, 400);
    expect([f.style.zoom, f.style.left, f.style.width]).toEqual(['', '10px', '600px']);
    zoom.setPaneZoom('logbook', 1.25);
    expect(f.style.zoom).toBe('1.25');
    expect([f.style.left, f.style.top, f.style.width, f.style.height]).toEqual([
      '8px',
      '40px',
      '480px',
      '320px',
    ]);
  });

  test('in its own window it keeps its percentages and takes the zoom alone', () => {
    const f = frame('artifact:a1');
    f.style.width = '100%';
    zoom.fillFrame(f);
    zoom.setPaneZoom('artifact:a1', 1.5);
    expect([f.style.zoom, f.style.width]).toEqual(['1.5', '100%']);
  });
});

describe('the ⋯ row and the bar chip', () => {
  test('Zoom ▸ names the level and checks it; a pick sets it', () => {
    zoom.setPaneZoom('logbook', 1.25);
    const row = zoom.zoomMenuRow('logbook');
    expect([row.label, row.value]).toEqual(['Zoom', '125%']);
    const levels = row.submenu ?? [];
    expect(levels.filter((l) => l.checked).map((l) => l.label)).toEqual(['125%']);
    levels.find((l) => l.label === '90%')?.run?.();
    expect(zoom.paneZoom('logbook')).toBe(0.9);
  });

  test('the chip leads the bar while zoomed, and a click goes back to 100% and takes it away', () => {
    const actions = document.createElement('div');
    actions.innerHTML = '<button class="tile-tab-more"></button>';
    document.body.append(actions);
    /** @type {{dispose: () => void}[]} */
    const disposables = [];
    zoom.watchZoomChip(actions, 'logbook', disposables);
    expect(actions.querySelector('.tile-tab-zoom')).toBeNull();

    zoom.setPaneZoom('logbook', 1.5);
    const chip = /** @type {HTMLButtonElement} */ (actions.firstElementChild);
    expect([chip.className, chip.textContent]).toEqual(['tile-tab-zoom', '150%']);

    chip.click();
    expect(zoom.paneZoom('logbook')).toBe(1);
    expect(actions.querySelector('.tile-tab-zoom')).toBeNull();

    for (const d of disposables) d.dispose();
    zoom.setPaneZoom('logbook', 1.5);
    expect(actions.querySelector('.tile-tab-zoom')).toBeNull();
  });
});
