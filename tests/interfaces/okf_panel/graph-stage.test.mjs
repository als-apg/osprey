// @ts-check
/**
 * The Graph view's stage (graph-stage.js): a scene drawn as SVG, and the
 * pointer on it:
 *   npx vitest run tests/interfaces/okf_panel/graph-stage.test.mjs
 *
 * happy-dom does no layout, so the stage has no size here: a fit is owed, not
 * done, and the tests read the elements and their attributes.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  createGraphStage,
  shape,
} from '../../../src/osprey/interfaces/okf_panel/static/js/graph-stage.js';

/** @typedef {import('../../../src/osprey/interfaces/okf_panel/static/js/graph-stage.js').Scene} Scene */
/** @typedef {import('../../../src/osprey/interfaces/okf_panel/static/js/graph-stage.js').GraphStage} GraphStage */

/** The stage's resize callback, to call as a resize would. */
let resized = () => {};

class FakeResizeObserver {
  /** @param {() => void} callback */
  constructor(callback) {
    resized = callback;
  }
  observe() {}
  disconnect() {}
}

/** @type {HTMLElement} */
let host;
/** @type {GraphStage} */
let stage;
const onSelect = vi.fn();
const onFocus = vi.fn();
const onRelayout = vi.fn();
const onMenu = vi.fn();

/**
 * Three nodes; two edges join the first pair, one in each direction.
 * @param {Partial<Scene>} [over]
 * @returns {Scene}
 */
function scene(over = {}) {
  return {
    drawn: {
      nodes: [
        { key: 1, label: 'M/BPM1', kind: 'device', r: 8 },
        { key: 2, label: 'M:BPM1:X', kind: 'channel', r: 5, badge: 14 },
        { key: 'g:signal', label: 'Signals · 2', kind: 'signal', r: 12 },
      ],
      edges: [
        { a: 1, b: 2, relation: 'hasBinding', label: 'hasBinding' },
        { a: 2, b: 1, relation: 'calibrates', label: 'calibrates' },
        { a: 2, b: 'g:signal', relation: 'readsSignal', label: 'readsSignal ×2', width: 2.5 },
      ],
      hidden: 0,
    },
    positions: new Map(
      /** @type {[number | string, {x: number, y: number}][]} */ ([
        [1, { x: 0, y: 0 }],
        [2, { x: 100, y: 0 }],
        ['g:signal', { x: 100, y: 80 }],
      ])
    ),
    grouped: false,
    focus: 1,
    selected: null,
    ...over,
  };
}

/** @param {string} selector */
const all = (selector) => [...host.querySelectorAll(selector)];

/** @param {string | number} key */
function node(key) {
  return /** @type {Element} */ (host.querySelector(`.okf-graph-node[data-key="${key}"]`));
}

/**
 * @param {Element} target
 * @param {string} type
 * @param {object} [init]
 */
function fire(target, type, init = {}) {
  const event = new MouseEvent(type, { bubbles: true, button: 0, ...init });
  target.dispatchEvent(event);
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  vi.clearAllMocks();
  document.body.innerHTML = '<div id="stage"></div>';
  host = /** @type {HTMLElement} */ (document.getElementById('stage'));
  stage = createGraphStage(host, { onSelect, onFocus, onRelayout, onMenu });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('building', () => {
  test('the stage is not built until asked', () => {
    expect(stage.built()).toBe(false);
    stage.render(scene());
    expect(host.querySelector('svg')).toBeNull();
  });

  test('build lays down the drawing, one arrowhead per line style, and the status', () => {
    stage.build(['hasBinding', 'calibrates', 'made-up']);
    expect(stage.built()).toBe(true);
    expect(all('marker').map((marker) => marker.id)).toEqual([
      'okf-graph-arrow-other',
      'okf-graph-arrow-hasBinding',
    ]);
    stage.setStatus('3 nodes');
    expect(host.querySelector('.okf-graph-status')?.textContent).toBe('3 nodes');
    expect(host.querySelector('.okf-graph-hint')).not.toBeNull();
  });
});

describe('drawing a scene', () => {
  beforeEach(() => {
    stage.build(['hasBinding', 'readsSignal', 'calibrates']);
  });

  test('each node is drawn where it is, with its kind, label and badge', () => {
    stage.render(scene());
    expect(all('.okf-graph-node').map((g) => g.getAttribute('data-kind'))).toEqual([
      'device',
      'channel',
      'signal',
    ]);
    expect(node(2).getAttribute('transform')).toBe('translate(100.0 0.0)');
    expect(node(2).querySelector('.okf-graph-label')?.textContent).toBe('M:BPM1:X');
    expect(node(2).querySelector('.okf-graph-badge')?.textContent).toBe('+14');
    expect(node(1).querySelector('.okf-graph-badge')).toBeNull();
  });

  test('the focus and the picked node are marked', () => {
    stage.render(scene({ selected: 2 }));
    expect(node(1).querySelector('.okf-graph-focus')).not.toBeNull();
    expect(node(2).querySelector('.okf-graph-focus')).toBeNull();
    expect(node(2).hasAttribute('data-selected')).toBe(true);
    expect(node(1).hasAttribute('data-selected')).toBe(false);
  });

  test('the overview marks no focus', () => {
    stage.render(scene({ grouped: true }));
    expect(host.querySelector('.okf-graph-focus')).toBeNull();
  });

  test('each edge carries its line style, its arrowhead and its label', () => {
    stage.render(scene());
    const edges = all('path.okf-graph-edge');
    expect(edges.map((edge) => edge.getAttribute('data-relation'))).toEqual([
      'hasBinding',
      'other',
      'readsSignal',
    ]);
    expect(edges[1].getAttribute('marker-end')).toBe('url(#okf-graph-arrow-other)');
    expect(all('.okf-graph-edge-label').map((label) => label.textContent)).toEqual([
      'hasBinding',
      'calibrates',
      'readsSignal ×2',
    ]);
  });

  test('a counted edge says how heavy it is; the stylesheet draws it', () => {
    stage.render(scene());
    const edges = /** @type {SVGElement[]} */ (all('path.okf-graph-edge'));
    expect(edges[2].style.getPropertyValue('--okf-edge-width')).toBe('2.50');
    expect(edges[0].style.getPropertyValue('--okf-edge-width')).toBe('');
    expect(edges[0].getAttribute('stroke')).toBeNull();
  });

  test('two edges between one pair of nodes bend apart', () => {
    stage.render(scene());
    const [there, back, single] = all('path.okf-graph-edge').map((edge) => edge.getAttribute('d'));
    expect(there).not.toBe(back);
    // A lone edge is straight: its control point is on the chord.
    expect(single).toMatch(/Q100\.0 40\.0 /);
    expect(there).not.toMatch(/Q50\.0 0\.0 /);
  });

  test('the labels control takes the relation labels off', () => {
    stage.render(scene());
    /** @type {HTMLElement} */ (host.querySelector('[title="Relation labels"]')).click();
    expect(all('.okf-graph-edge-label')).toHaveLength(0);
    expect(all('path.okf-graph-edge')).toHaveLength(3);
  });

  test('a crowded neighbourhood drops its channel labels, but not the picked one’s', () => {
    const crowd = scene();
    for (let i = 10; i < 85; i++) {
      crowd.drawn.nodes.push({ key: i, label: 'ch' + i, kind: 'channel', r: 5 });
      crowd.positions.set(i, { x: i, y: i });
    }
    stage.render({ ...crowd, selected: 12 });
    expect(node(11).querySelector('.okf-graph-label')).toBeNull();
    expect(node(12).querySelector('.okf-graph-label')).not.toBeNull();
    expect(node(1).querySelector('.okf-graph-label')).not.toBeNull();
  });
});

describe('the pointer', () => {
  /** @type {Scene} */
  let shown;
  /** @type {Element} */
  let surface;
  beforeEach(() => {
    stage.build(['hasBinding']);
    shown = scene();
    stage.render(shown);
    surface = /** @type {Element} */ (host.querySelector('svg.okf-graph-svg'));
  });

  test('a click reports the node, or null for the background', () => {
    fire(node(2), 'click');
    expect(onSelect).toHaveBeenLastCalledWith(2);
    fire(node('g:signal'), 'click');
    expect(onSelect).toHaveBeenLastCalledWith('g:signal');
    fire(surface, 'click');
    expect(onSelect).toHaveBeenLastCalledWith(null);
  });

  test('a click delivered to the drawing picks the node that was pressed', () => {
    // A captured pointer's click goes to the capturing element.
    fire(node(2), 'pointerdown', { clientX: 10, clientY: 10 });
    fire(surface, 'pointerup');
    fire(surface, 'click');
    expect(onSelect).toHaveBeenLastCalledWith(2);
    fire(surface, 'dblclick');
    expect(onFocus).toHaveBeenLastCalledWith(2);

    fire(surface, 'pointerdown', { clientX: 300, clientY: 300 });
    fire(surface, 'pointerup');
    fire(surface, 'click');
    expect(onSelect).toHaveBeenLastCalledWith(null);
  });

  test('a double-click reports the node, and nothing for the background', () => {
    fire(node(2), 'dblclick');
    expect(onFocus).toHaveBeenCalledWith(2);
    fire(surface, 'dblclick');
    expect(onFocus).toHaveBeenCalledTimes(1);
  });

  test('dragging a node moves it, and the click that ends the drag picks nothing', () => {
    fire(node(2), 'pointerdown', { clientX: 10, clientY: 10 });
    fire(node(2), 'pointermove', { clientX: 40, clientY: 30 });
    fire(node(2), 'pointerup');
    expect(shown.positions.get(2)).toEqual({ x: 130, y: 20 });
    expect(node(2).getAttribute('transform')).toBe('translate(130.0 20.0)');
    fire(node(2), 'click');
    expect(onSelect).not.toHaveBeenCalled();
    fire(node(2), 'click');
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  test('a press that barely moves is a click, not a drag', () => {
    fire(node(2), 'pointerdown', { clientX: 10, clientY: 10 });
    fire(node(2), 'pointermove', { clientX: 11, clientY: 11 });
    fire(node(2), 'pointerup');
    expect(shown.positions.get(2)).toEqual({ x: 100, y: 0 });
    fire(node(2), 'click');
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  test('dragging the background pans the drawing', () => {
    fire(surface, 'pointerdown', { clientX: 0, clientY: 0 });
    fire(surface, 'pointermove', { clientX: 25, clientY: -10 });
    fire(surface, 'pointerup');
    expect(surface.querySelector('g')?.getAttribute('transform')).toBe(
      'translate(25 -10) scale(1)'
    );
    expect(shown.positions.get(2)).toEqual({ x: 100, y: 0 });
  });

  test('the zoom controls scale the drawing', () => {
    /** @type {HTMLElement} */ (host.querySelector('[title="Zoom in"]')).click();
    expect(surface.querySelector('g')?.getAttribute('transform')).toBe(
      'translate(0 0) scale(1.25)'
    );
  });

  test('pointing at a node dims what is not joined to it', () => {
    fire(node(1), 'mouseover');
    expect(node(1).hasAttribute('data-dim')).toBe(false);
    expect(node(2).hasAttribute('data-dim')).toBe(false);
    expect(node('g:signal').hasAttribute('data-dim')).toBe(true);
    const edges = all('path.okf-graph-edge');
    expect(edges.map((edge) => edge.hasAttribute('data-dim'))).toEqual([false, false, true]);
    fire(surface, 'mouseover');
    expect(all('[data-dim]')).toHaveLength(0);
  });

  test('a stage that narrows over the picked node pans it back into view', () => {
    // The drawing was fitted once; node 2 sits at x = 100.
    Object.defineProperty(host, 'clientWidth', { value: 400, configurable: true });
    Object.defineProperty(host, 'clientHeight', { value: 300, configurable: true });
    resized();
    const group = /** @type {Element} */ (surface.querySelector('g'));
    stage.render({ ...shown, selected: 2 });
    const before = group.getAttribute('transform');
    resized();
    expect(group.getAttribute('transform')).toBe(before);

    Object.defineProperty(host, 'clientWidth', { value: 120, configurable: true });
    resized();
    const x = Number(/translate\(([-\d.]+)/.exec(group.getAttribute('transform') ?? '')?.[1]);
    const k = Number(/scale\(([-\d.]+)/.exec(group.getAttribute('transform') ?? '')?.[1]);
    expect(100 * k + x).toBeCloseTo(40);
  });

  test('Lay out again is the caller’s', () => {
    /** @type {HTMLElement} */ (host.querySelector('[title="Lay out again"]')).click();
    expect(onRelayout).toHaveBeenCalledTimes(1);
  });
});

describe('a right-click', () => {
  /** @type {Element} */
  let surface;
  beforeEach(() => {
    stage.build(['hasBinding', 'readsSignal', 'calibrates']);
    stage.render(scene());
    surface = /** @type {Element} */ (host.querySelector('svg.okf-graph-svg'));
  });

  test('reports the node, the relation or the empty stage it landed on, and where', () => {
    fire(node(2), 'contextmenu', { button: 2, clientX: 30, clientY: 40 });
    expect(onMenu).toHaveBeenLastCalledWith({ kind: 'node', key: 2 }, { x: 30, y: 40 });
    fire(node('g:signal'), 'contextmenu', { button: 2 });
    expect(onMenu).toHaveBeenLastCalledWith({ kind: 'node', key: 'g:signal' }, { x: 0, y: 0 });

    const hits = all('.okf-graph-edge-hit');
    expect(hits).toHaveLength(3);
    fire(hits[1], 'contextmenu', { button: 2 });
    expect(onMenu.mock.lastCall?.[0]).toEqual({
      kind: 'edge',
      edge: { a: 2, b: 1, relation: 'calibrates', label: 'calibrates' },
    });

    fire(surface, 'contextmenu', { button: 2 });
    expect(onMenu.mock.lastCall?.[0]).toEqual({ kind: 'stage' });
  });

  test('the wide line under a relation follows its path', () => {
    const lines = all('path.okf-graph-edge[data-a]');
    const hits = all('.okf-graph-edge-hit');
    expect(hits.map((hit) => hit.getAttribute('d'))).toEqual(
      lines.map((line) => line.getAttribute('d'))
    );
  });

  test('takes the browser’s menu away, but not with Shift held', () => {
    const plain = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, button: 2 });
    surface.dispatchEvent(plain);
    expect(plain.defaultPrevented).toBe(true);

    onMenu.mockClear();
    const shifted = new MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      button: 2,
      shiftKey: true,
    });
    surface.dispatchEvent(shifted);
    expect(shifted.defaultPrevented).toBe(false);
    expect(onMenu).not.toHaveBeenCalled();
  });

  test('a right-click picks nothing', () => {
    fire(node(2), 'pointerdown', { button: 2 });
    fire(node(2), 'contextmenu', { button: 2 });
    expect(onSelect).not.toHaveBeenCalled();
  });
});

describe('relation labels', () => {
  test('are switched from outside, and the strip’s button follows', () => {
    stage.build(['hasBinding', 'readsSignal', 'calibrates']);
    stage.render(scene());
    const toggle = /** @type {HTMLElement} */ (host.querySelector('[aria-pressed]'));
    expect(stage.labels()).toBe(true);
    expect(toggle.getAttribute('aria-pressed')).toBe('true');
    stage.setLabels(false);
    expect(stage.labels()).toBe(false);
    expect(toggle.getAttribute('aria-pressed')).toBe('false');
  });
});

describe('shape', () => {
  test('a place is a square, a signal a diamond, the rest a circle', () => {
    expect(shape('place', 5).tagName).toBe('rect');
    expect(shape('signal', 5).tagName).toBe('path');
    expect(shape('device', 5).tagName).toBe('circle');
    expect(shape('class', 5).tagName).toBe('circle');
  });
});
