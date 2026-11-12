// @ts-check
/**
 * The KNOWLEDGE panel's Graph view (graph-view.js):
 *   npx vitest run tests/interfaces/okf_panel/graph-view.test.mjs
 *
 * Each test mounts the real page shell and a fresh module, and serves the
 * small graph of graph-fixture.mjs. happy-dom does no layout, so nothing here
 * is about where a node lands: the tests read what is drawn, what the
 * sidebar and the inspector say, and what the view reports.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { payload } from './graph-fixture.mjs';
import { mountShell } from './page-shell.mjs';

/** @typedef {import('../../../src/osprey/interfaces/okf_panel/static/js/graph-view.js').GraphViewApi} GraphViewApi */

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

class FakeResizeObserver {
  observe() {}
  disconnect() {}
}

/** @type {import('vitest').Mock<(conceptId: string) => void>} */
let onPage;
/** @type {import('vitest').Mock<(nodeId: string | null) => void>} */
let onLocation;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

/**
 * @param {unknown} [served] what `/api/graph` answers; an Error fails it
 * @returns {Promise<GraphViewApi>}
 */
async function load(served = payload()) {
  mountShell();
  $('app').dataset.graph = 'true';
  fetchMock = vi.fn(async (/** @type {string} */ url) => {
    if (url !== '/api/graph') throw new Error(`unstubbed fetch: ${url}`);
    if (served instanceof Error) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => served };
  });
  vi.stubGlobal('fetch', fetchMock);
  vi.resetModules();
  const { createGraphView } =
    await import('../../../src/osprey/interfaces/okf_panel/static/js/graph-view.js');
  onPage = vi.fn();
  onLocation = vi.fn();
  return createGraphView({ onPage, onLocation });
}

/** The labels of the drawn nodes. */
function drawnLabels() {
  return [...$('graph-stage').querySelectorAll('.okf-graph-node title')].map(
    (title) => title.textContent
  );
}

/**
 * @param {string} key
 * @returns {Element}
 */
function nodeEl(key) {
  const node = $('graph-stage').querySelector(`.okf-graph-node[data-key="${key}"]`);
  if (!node) throw new Error('no drawn node ' + key);
  return node;
}

/** @param {Element} target @param {string} type */
function fire(target, type) {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true }));
}

/**
 * @param {HTMLElement} host
 * @param {string} text
 * @returns {HTMLElement}
 */
function button(host, text) {
  const found = [...host.querySelectorAll('button')].find(
    (candidate) => (candidate.textContent || '').trim() === text
  );
  if (!found) throw new Error('no button ' + text);
  return found;
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loading', () => {
  test('nothing is fetched until the view is shown', async () => {
    const view = await load();
    expect(fetchMock).not.toHaveBeenCalled();
    await view.search('');
    expect(fetchMock).not.toHaveBeenCalled();
    await view.show(null);
    await view.show(null);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('a graph that does not load says so and can be retried', async () => {
    const view = await load(new Error('down'));
    await view.show(null);
    expect($('graph-stage').textContent).toContain('The graph did not load');
    expect($('graph-stage').querySelector('svg.okf-graph-svg')).toBeNull();
    button($('graph-stage'), 'Retry').click();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test('an empty graph says so', async () => {
    const view = await load({ nodes: [], edges: [] });
    await view.show(null);
    expect($('graph-stage').textContent).toContain('The facility graph is empty');
  });
});

describe('the overview', () => {
  /** @type {GraphViewApi} */
  let view;
  beforeEach(async () => {
    view = await load();
    await view.show(null);
  });

  test('draws the groups and says what the graph holds', () => {
    expect(drawnLabels()).toEqual([
      'Places · 2',
      'Channels · 2',
      'Signals · 2',
      'BeamPositionMonitor · 1',
      'Quadrupole · 1',
    ]);
    expect($('graph-context').textContent).toBe('Overview');
    expect($('graph-stage').querySelector('.okf-graph-status')?.textContent).toBe(
      '10 nodes and 9 relations, grouped by type'
    );
    expect($('graph-chip').textContent?.trim()).toBe('10 nodes · built from the facility file');
    expect(view.location()).toBeNull();
    expect(onLocation).toHaveBeenLastCalledWith(null);
    expect($('graph-depth').hidden).toBe(true);
  });

  test('the legend lists every relation with its count, and tags a derived one', () => {
    const rows = [...$('graph-relations').querySelectorAll('.okf-graph-legend-row')].map((row) =>
      [...row.querySelectorAll('span')].map((span) => span.textContent).join(' ')
    );
    expect(rows).toEqual([
      'hasBinding 2',
      'readsSignal 1',
      'writesSignal 1',
      'type 2',
      'inPlace derived 2',
      'calibrates 1',
    ]);
    const kinds = [...$('graph-kinds').querySelectorAll('.okf-graph-legend-name')];
    expect(kinds.map((kind) => kind.textContent)).toEqual([
      'Place',
      'Device',
      'Channel',
      'Signal',
      'Class',
    ]);
  });

  test('switching a relation off takes its edges out of the drawing', () => {
    const edges = () => $('graph-stage').querySelectorAll('path.okf-graph-edge[data-a]').length;
    const before = edges();
    const box = /** @type {HTMLInputElement} */ (
      $('graph-relations').querySelector('.okf-graph-legend-row input')
    );
    box.checked = false;
    box.dispatchEvent(new Event('change', { bubbles: true }));
    expect(edges()).toBeLessThan(before);
  });

  test('a click on a group lists its members', () => {
    fire(nodeEl('g:channel'), 'click');
    const inspector = $('graph-inspector');
    expect(inspector.querySelector('.okf-graph-title')?.textContent).toBe('Channels');
    expect(inspector.textContent).toContain('2 in the graph');
    expect(
      [...inspector.querySelectorAll('.okf-graph-row-name')].map((row) => row.textContent)
    ).toEqual(['M:BPM1:X', 'M:Q1:SP']);
  });

  test('a class group names its class and what it is also called', () => {
    fire(nodeEl('c:8'), 'click');
    expect($('graph-inspector').querySelector('.okf-graph-title')?.textContent).toBe(
      'BeamPositionMonitor'
    );
    expect($('graph-inspector').textContent).toContain('Also called BPM');
  });

  test('a member row draws that node’s neighbourhood', () => {
    fire(nodeEl('g:channel'), 'click');
    button($('graph-inspector'), 'M:BPM1:XChannel').click();
    expect(view.location()).toBe('c:M:BPM1:X');
    expect(onLocation).toHaveBeenLastCalledWith('c:M:BPM1:X');
    expect($('graph-context').textContent).toBe('Neighbourhood of M:BPM1:X');
  });

  test('a double-click on a group focuses nothing', () => {
    fire(nodeEl('g:channel'), 'dblclick');
    expect(view.location()).toBeNull();
  });

  test('the Neighbourhood drawing starts on the first device', () => {
    button($('graph-modes'), 'Neighbourhood').click();
    expect(view.location()).toBe('d:M/BPM1');
    expect($('graph-depth').hidden).toBe(false);
    button($('graph-modes'), 'Overview').click();
    expect(view.location()).toBeNull();
    expect(onLocation).toHaveBeenLastCalledWith(null);
  });
});

describe('a neighbourhood', () => {
  /** @type {GraphViewApi} */
  let view;
  beforeEach(async () => {
    view = await load();
    await view.show('d:M/BPM1');
  });

  test('draws the nodes around the focus, and marks it', () => {
    expect(drawnLabels().sort()).toEqual(
      ['BeamPositionMonitor', 'M/BPM1', 'M/Q1', 'M/S1', 'M:BPM1:X', 'position_x'].sort()
    );
    expect(nodeEl('2').querySelector('.okf-graph-focus')).not.toBeNull();
    expect($('graph-stage').querySelector('.okf-graph-status')?.textContent).toBe(
      '6 nodes · 6 relations around M/BPM1 (2 hops)'
    );
    expect($('graph-modes').querySelector('[aria-pressed="true"]')?.textContent).toBe(
      'Neighbourhood'
    );
  });

  test('the depth changes how far it reaches', () => {
    button($('graph-hops'), '1 hop').click();
    expect(drawnLabels().sort()).toEqual(
      ['BeamPositionMonitor', 'M/BPM1', 'M/S1', 'M:BPM1:X'].sort()
    );
    expect($('graph-stage').querySelector('.okf-graph-status')?.textContent).toContain('(1 hop)');
  });

  test('the inspector describes the focus node', () => {
    const inspector = $('graph-inspector');
    expect(inspector.querySelector('.osprey-chip-label')?.textContent).toBe(
      'Device · BeamPositionMonitor'
    );
    expect(inspector.querySelector('.okf-graph-title')?.textContent).toBe('M/BPM1');
    expect(inspector.querySelector('.okf-graph-description')?.textContent).toBe('A monitor.');
    expect([...inspector.querySelectorAll('dt')].map((dt) => dt.textContent)).toEqual([
      'placePath',
    ]);
    expect(
      [...inspector.querySelectorAll('.osprey-section-label')].map((label) => label.textContent)
    ).toEqual(['→ hasBinding · 1', '→ type · 1', '→ inPlace · 1']);
    // The focus needs no "Focus here".
    expect([...inspector.querySelectorAll('.osprey-btn')].map((b) => b.textContent)).toEqual([
      'Knowledge page',
    ]);
  });

  test('Knowledge page opens the page that names the node', () => {
    button($('graph-inspector'), 'Knowledge page').click();
    expect(onPage).toHaveBeenCalledWith('devices/bpm');
  });

  test('a click picks a node without moving the focus; Focus here moves it', () => {
    fire(nodeEl('4'), 'click');
    expect($('graph-inspector').querySelector('.okf-graph-title')?.textContent).toBe('M:BPM1:X');
    expect(nodeEl('4').hasAttribute('data-selected')).toBe(true);
    expect(view.location()).toBe('d:M/BPM1');
    button($('graph-inspector'), 'Focus here').click();
    expect(view.location()).toBe('c:M:BPM1:X');
  });

  test('a click on the background picks nothing', () => {
    fire(nodeEl('4'), 'click');
    fire(/** @type {Element} */ ($('graph-stage').querySelector('svg.okf-graph-svg')), 'click');
    expect($('graph-inspector').hidden).toBe(true);
  });

  test('a double-click focuses the node', () => {
    fire(nodeEl('3'), 'dblclick');
    expect(view.location()).toBe('d:M/Q1');
    expect(onLocation).toHaveBeenLastCalledWith('d:M/Q1');
  });

  test('an id the graph does not hold draws the overview', async () => {
    await view.show('no-such-node');
    expect(view.location()).toBeNull();
    expect($('graph-context').textContent).toBe('Overview');
  });
});

describe('the right-click menus', () => {
  /** @type {GraphViewApi} */
  let view;
  beforeEach(async () => {
    view = await load();
    await view.show('d:M/BPM1');
  });

  /** @param {Element} target */
  const rightClick = (target) =>
    target.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
  const svg = () => /** @type {Element} */ ($('graph-stage').querySelector('svg.okf-graph-svg'));
  const heading = () => document.querySelector('.osprey-context-heading')?.textContent;
  /** @param {string} text */
  function pick(text) {
    const found = [...document.querySelectorAll('.osprey-context-item')].find((item) =>
      (item.querySelector('.osprey-context-label')?.textContent || '').startsWith(text)
    );
    if (!found) throw new Error('no menu row ' + text);
    /** @type {HTMLElement} */ (found).click();
  }
  const boxes = () =>
    /** @type {HTMLInputElement[]} */ ([
      ...$('graph-relations').querySelectorAll('.okf-graph-legend-row input'),
    ]);
  const edges = () => $('graph-stage').querySelectorAll('path.okf-graph-edge[data-a]').length;

  test('a node’s menu names it, and Focus here moves the focus', () => {
    rightClick(nodeEl('4'));
    expect(heading()).toBe('M:BPM1:X');
    pick('Focus here');
    expect(document.querySelector('.osprey-context-menu')).toBeNull();
    expect(view.location()).toBe('c:M:BPM1:X');
  });

  test('a node’s page opens from its menu', () => {
    rightClick(nodeEl('2'));
    pick('Open knowledge page');
    expect(onPage).toHaveBeenCalledWith('devices/bpm');
  });

  test('a copied name is said in the status line, which then comes back', async () => {
    vi.useFakeTimers();
    try {
      const writeText = vi.fn(async () => {});
      vi.stubGlobal('navigator', { clipboard: { writeText } });
      const status = () => $('graph-stage').querySelector('.okf-graph-status')?.textContent;
      const before = status();
      rightClick(nodeEl('4'));
      pick('Copy name');
      await vi.advanceTimersByTimeAsync(0);
      expect(writeText).toHaveBeenCalledWith('M:BPM1:X');
      expect(status()).toBe('Copied M:BPM1:X');
      await vi.advanceTimersByTimeAsync(2000);
      expect(status()).toBe(before);
    } finally {
      vi.useRealTimers();
    }
  });

  test('a relation is hidden from its menu, and the legend follows', () => {
    const hit = /** @type {Element} */ ($('graph-stage').querySelector('.okf-graph-edge-hit'));
    const before = edges();
    rightClick(hit);
    const relation = heading();
    pick('Hide');
    expect(edges()).toBeLessThan(before);
    const off = boxes().filter((box) => !box.checked);
    expect(off).toHaveLength(1);
    expect(off[0].closest('.okf-graph-legend-row')?.textContent).toContain(relation);
  });

  test('a relation is shown alone, and the stage’s menu shows all again', () => {
    rightClick(/** @type {Element} */ ($('graph-stage').querySelector('.okf-graph-edge-hit')));
    pick('Show only');
    expect(boxes().filter((box) => box.checked)).toHaveLength(1);
    rightClick(svg());
    pick('Show all relations');
    expect(boxes().every((box) => box.checked)).toBe(true);
  });

  test('an end of a relation is picked from its menu', () => {
    rightClick(/** @type {Element} */ ($('graph-stage').querySelector('.okf-graph-edge-hit')));
    pick('Select target');
    expect($('graph-inspector').hidden).toBe(false);
    expect($('graph-stage').querySelector('.okf-graph-node[data-selected]')).not.toBeNull();
  });

  test('Relation labels is the strip’s toggle', () => {
    const toggle = () => $('graph-stage').querySelector('[aria-pressed]');
    rightClick(svg());
    pick('Relation labels');
    expect(toggle()?.getAttribute('aria-pressed')).toBe('false');
    rightClick(svg());
    const row = [...document.querySelectorAll('.osprey-context-item')].find((item) =>
      item.textContent?.includes('Relation labels')
    );
    expect(row?.getAttribute('aria-checked')).toBe('false');
  });

  test('Reset view brings back every relation, the depth and the labels', () => {
    boxes()[0].checked = false;
    boxes()[0].dispatchEvent(new Event('change', { bubbles: true }));
    button($('graph-hops'), '1 hop').click();
    fire(nodeEl('2'), 'click');
    rightClick(svg());
    pick('Relation labels');

    rightClick(svg());
    pick('Reset view');
    expect(boxes().every((box) => box.checked)).toBe(true);
    expect($('graph-hops').querySelector('[aria-pressed="true"]')?.textContent).toBe('2 hops');
    expect($('graph-stage').querySelector('[aria-pressed]')?.getAttribute('aria-pressed')).toBe(
      'true'
    );
    expect($('graph-inspector').hidden).toBe(true);
    expect(view.location()).toBe('d:M/BPM1');
    expect($('graph-stage').querySelector('.okf-graph-status')?.textContent).toBe(
      '6 nodes · 6 relations around M/BPM1 (2 hops)'
    );
  });

  test('a group of the overview lists its members from its menu', async () => {
    await view.show(null);
    rightClick(nodeEl('g:channel'));
    expect(heading()).toBe('Channels · 2');
    pick('List members');
    expect($('graph-inspector').hidden).toBe(false);
  });
});

describe('search and records', () => {
  test('search lists the nodes a query names, and a hit focuses its node', async () => {
    const view = await load();
    await view.show(null);
    await view.search('q1');
    const hits = $('graph-hits');
    expect(hits.hidden).toBe(false);
    expect([...hits.querySelectorAll('.okf-graph-row-name')].map((row) => row.textContent)).toEqual(
      ['M/Q1', 'M:Q1:SP']
    );
    expect($('graph-sidebar').getAttribute('data-open')).toBe('true');
    /** @type {HTMLElement} */ (hits.querySelector('button')).click();
    expect(view.location()).toBe('d:M/Q1');

    await view.search('zzz');
    expect(hits.textContent).toContain('No node matches');
    await view.search('');
    expect(hits.hidden).toBe(true);
  });

  test('showRecord focuses the node of a facility record', async () => {
    const view = await load();
    expect(await view.showRecord('channel', 'M:Q1:SP')).toBe(true);
    expect(view.location()).toBe('c:M:Q1:SP');
  });

  test('a record the graph does not hold leaves the overview', async () => {
    const view = await load();
    expect(await view.showRecord('channel', 'NOPE')).toBe(false);
    expect(view.location()).toBeNull();
    expect(drawnLabels()).toContain('Channels · 2');
  });
});
