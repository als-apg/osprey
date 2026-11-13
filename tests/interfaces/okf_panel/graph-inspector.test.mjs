// @ts-check
/**
 * The Graph view's inspector (graph-inspector.js): what is shown of the
 * picked node or group:
 *   npx vitest run tests/interfaces/okf_panel/graph-inspector.test.mjs
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';

import {
  nodeRow,
  renderInspector,
} from '../../../src/osprey/interfaces/okf_panel/static/js/graph-inspector.js';
import {
  indexGraph,
  neighbourhood,
  overview,
} from '../../../src/osprey/interfaces/okf_panel/static/js/graph-model.js';
import { payload } from './graph-fixture.mjs';

/** @typedef {import('../../../src/osprey/interfaces/okf_panel/static/js/graph-model.js').NodeKey} NodeKey */

/** @type {HTMLElement} */
let host;
const handles = { onFocus: vi.fn(), onExpand: vi.fn(), onPage: vi.fn() };

/** @param {ReturnType<typeof payload>} [data] */
function hood(data = payload(), focus = 2) {
  const graph = indexGraph(data);
  const relations = new Set(graph.relationCounts.keys());
  return {
    graph,
    drawn: neighbourhood(graph, { focus, hops: 2, relations }),
    grouped: false,
    focus,
  };
}

function groups() {
  const graph = indexGraph(payload());
  const drawn = overview(graph, new Set(graph.relationCounts.keys()));
  return { graph, drawn, grouped: true, focus: null };
}

/** @param {string} selector */
const texts = (selector) => [...host.querySelectorAll(selector)].map((node) => node.textContent);

/** @param {string} text */
function button(text) {
  const found = [...host.querySelectorAll('button')].find((b) => b.textContent === text);
  if (!found) throw new Error('no button ' + text);
  return found;
}

beforeEach(() => {
  vi.clearAllMocks();
  document.body.innerHTML = '<aside id="inspector"></aside>';
  host = /** @type {HTMLElement} */ (document.getElementById('inspector'));
});

describe('nothing picked', () => {
  test('the inspector is closed, and opens again with a pick', () => {
    renderInspector(host, { ...hood(), selected: 2 }, handles);
    expect(host.hidden).toBe(false);
    renderInspector(host, { ...hood(), selected: null }, handles);
    expect(host.hidden).toBe(true);
    expect(host.childElementCount).toBe(0);
    renderInspector(host, { ...groups(), selected: null }, handles);
    expect(host.hidden).toBe(true);
    renderInspector(host, { ...hood(), selected: 2 }, handles);
    expect(host.hidden).toBe(false);
  });

  test('a group that is no longer drawn counts as nothing picked', () => {
    renderInspector(host, { ...groups(), selected: 'c:404' }, handles);
    expect(host.hidden).toBe(true);
  });
});

describe('a node', () => {
  test('shows its kind and class, description, properties and relations', () => {
    renderInspector(host, { ...hood(), selected: 2 }, handles);
    expect(texts('.osprey-chip-label')).toEqual(['Device · BeamPositionMonitor']);
    expect(texts('.okf-graph-title')).toEqual(['M/BPM1']);
    expect(texts('.okf-graph-description')).toEqual([
      'A monitor.',
      'Also called BPM 1, Orbit monitor 1',
    ]);
    expect(texts('dt')).toEqual(['placePath']);
    expect(texts('dd')).toEqual(['M/S1']);
    expect(texts('.osprey-section-label')).toEqual([
      '→ hasBinding · 1',
      '→ type · 1',
      '→ locatedIn · 1',
    ]);
  });

  test("the head names the record's own names, not its class's synonyms", () => {
    renderInspector(host, { ...hood(), selected: 2 }, handles);
    const lines = texts('.okf-graph-description');
    expect(lines).toContain('Also called BPM 1, Orbit monitor 1');
    expect(lines).not.toContain('Also called BPM');
  });

  test('altLabel is not repeated as a property row', () => {
    renderInspector(host, { ...hood(), selected: 2 }, handles);
    expect(texts('dt')).not.toContain('altLabel');
    expect(texts('dd')).not.toContain('BPM 1, Orbit monitor 1');
  });

  test('a record without names shows no Also called line', () => {
    renderInspector(host, { ...hood(), selected: 3 }, handles);
    expect(texts('.okf-graph-title')).toEqual(['M/Q1']);
    expect(texts('.okf-graph-description').some((line) => line?.startsWith('Also called'))).toBe(
      false
    );
  });

  test('a node without a class or properties shows neither', () => {
    renderInspector(host, { ...hood(), selected: 6 }, handles);
    expect(texts('.osprey-chip-label')).toEqual(['Signal']);
    expect(host.querySelector('dl')).toBeNull();
    expect(texts('.osprey-section-label')).toEqual(['← readsSignal · 1']);
  });

  test('the focus offers no Focus here; another node does', () => {
    renderInspector(host, { ...hood(), selected: 2 }, handles);
    expect(texts('.osprey-btn')).toEqual(['Knowledge page']);
    renderInspector(host, { ...hood(), selected: 4 }, handles);
    expect(texts('.osprey-btn')).toEqual(['Focus here']);
    button('Focus here').click();
    expect(handles.onFocus).toHaveBeenCalledWith(4);
  });

  test('one knowledge page is a button, several are a list', () => {
    renderInspector(host, { ...hood(), selected: 2 }, handles);
    button('Knowledge page').click();
    expect(handles.onPage).toHaveBeenCalledWith('devices/bpm');

    const data = payload();
    data.nodes[2].pages = ['devices/bpm', 'procedures/orbit'];
    renderInspector(host, { ...hood(data), selected: 2 }, handles);
    expect(texts('.osprey-btn')).toEqual([]);
    expect(texts('.osprey-section-label')[0]).toBe('Knowledge pages');
    button('procedures/orbit').click();
    expect(handles.onPage).toHaveBeenLastCalledWith('procedures/orbit');
  });

  test('a hub that was not walked offers the rest', () => {
    const view = hood();
    const drawn = { ...view.drawn, nodes: view.drawn.nodes.map((n) => ({ ...n })) };
    /** @type {any} */ (drawn.nodes.find((n) => n.key === 4)).badge = 1234;
    renderInspector(host, { ...view, drawn, selected: 4 }, handles);
    button('Show 1,234 more').click();
    expect(handles.onExpand).toHaveBeenCalledWith(4);
  });

  test('a long relation list is cut, and says by how much', () => {
    const data = payload();
    for (let i = 0; i < 40; i++) {
      data.nodes.push({ id: 'c:x' + i, kind: 'channel', label: 'x' + i, type: null, props: {} });
      data.edges.push([2, data.nodes.length - 1, 'hasBinding']);
    }
    renderInspector(host, { ...hood(data), selected: 2 }, handles);
    const list = /** @type {Element} */ (host.querySelector('.okf-graph-list'));
    expect(list.querySelector('.osprey-section-label')?.textContent).toBe('→ hasBinding · 41');
    expect(list.querySelectorAll('.okf-graph-row')).toHaveLength(30);
    expect(list.querySelector('.okf-graph-more')?.textContent).toBe('11 more');
  });
});

describe('a group', () => {
  test('a kind lists its members', () => {
    renderInspector(host, { ...groups(), selected: 'g:channel' }, handles);
    expect(texts('.osprey-chip-label')).toEqual(['Group']);
    expect(texts('.okf-graph-title')).toEqual(['Channels']);
    expect(texts('.okf-graph-description')).toEqual(['2 in the graph']);
    expect(texts('.okf-graph-row-name')).toEqual(['M:BPM1:X', 'M:Q1:SP']);
    expect(texts('.osprey-btn')).toEqual([]);
  });

  test('a class names itself, what it is also called, and can be focused', () => {
    renderInspector(host, { ...groups(), selected: 'c:8' }, handles);
    expect(texts('.osprey-chip-label')).toEqual(['Class']);
    expect(texts('.okf-graph-title')).toEqual(['BeamPositionMonitor']);
    expect(texts('.okf-graph-description')).toEqual(['1 in the graph', 'Also called BPM']);
    button('Focus here').click();
    expect(handles.onFocus).toHaveBeenCalledWith(8);
  });
});

describe('nodeRow', () => {
  test('names the node and its kind, and focuses it when picked', () => {
    const graph = indexGraph(payload());
    const onFocus = vi.fn();
    const row = nodeRow(graph, 3, onFocus);
    expect(row.textContent).toBe('M/Q1Device');
    expect(row.title).toBe('M/Q1');
    row.click();
    expect(onFocus).toHaveBeenCalledWith(3);
  });
});
