// @ts-check
/**
 * The Graph view's right-click menus (graph-menus.js): which rows a node, a
 * relation and the empty stage offer, and what each runs:
 *   npx vitest run tests/interfaces/okf_panel/graph-menus.test.mjs
 */

import { describe, expect, test, vi } from 'vitest';

import {
  edgeMenu,
  nodeMenu,
  stageMenu,
} from '../../../src/osprey/interfaces/okf_panel/static/js/graph-menus.js';
import {
  indexGraph,
  neighbourhood,
  overview,
} from '../../../src/osprey/interfaces/okf_panel/static/js/graph-model.js';
import { payload } from './graph-fixture.mjs';

/** @typedef {import('../../../src/osprey/interfaces/okf_panel/static/js/graph-menus.js').MenuView} MenuView */
/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

const BPM = 2;
const CHANNEL = 4;
const BPM_CLASS = 8;

function actions() {
  return {
    select: vi.fn(),
    focus: vi.fn(),
    expand: vi.fn(),
    page: vi.fn(),
    copy: vi.fn(),
    setRelations: vi.fn(),
    fit: vi.fn(),
    relayout: vi.fn(),
    setLabels: vi.fn(),
    reset: vi.fn(),
  };
}

/**
 * The neighbourhood of the BPM, or the overview.
 * @param {Partial<MenuView> & {payload?: ReturnType<typeof payload>}} [over]
 * @returns {MenuView}
 */
function view({ payload: served = payload(), ...over } = {}) {
  const graph = indexGraph(served);
  const relations = over.relations ?? new Set(graph.relationCounts.keys());
  const grouped = over.grouped ?? false;
  return {
    graph,
    drawn: grouped
      ? overview(graph, relations)
      : neighbourhood(graph, { focus: BPM, hops: 2, relations }),
    grouped,
    focus: grouped ? null : BPM,
    relations,
    labels: true,
    ...over,
  };
}

/** @param {MenuItem[]} items */
const labels = (items) =>
  items.map((item) =>
    'divider' in item ? '-' : 'heading' in item ? `# ${item.heading}` : item.label
  );

/**
 * @param {MenuItem[]} items
 * @param {string} label
 */
function row(items, label) {
  const found = items.find((item) => 'label' in item && item.label === label);
  if (!found || !('label' in found)) throw new Error('no row ' + label);
  return found;
}

describe('a node', () => {
  test('is named, and offers focus, its page and its name and id', () => {
    const act = actions();
    const items = nodeMenu(view(), CHANNEL, act);
    expect(labels(items)).toEqual(['# M:BPM1:X', 'Focus here', '-', 'Copy name', 'Copy id']);
    row(items, 'Focus here').run?.();
    expect(act.focus).toHaveBeenCalledWith(CHANNEL);
    row(items, 'Copy name').run?.();
    expect(act.copy).toHaveBeenLastCalledWith('M:BPM1:X', 'M:BPM1:X');
    row(items, 'Copy id').run?.();
    expect(act.copy).toHaveBeenLastCalledWith('c:M:BPM1:X', 'c:M:BPM1:X');
  });

  test('the focus cannot be focused again; its page opens', () => {
    const act = actions();
    const items = nodeMenu(view(), BPM, act);
    expect(row(items, 'Focus here').disabled).toBe(true);
    row(items, 'Open knowledge page').run?.();
    expect(act.page).toHaveBeenCalledWith('devices/bpm');
  });

  test('several pages are a submenu', () => {
    const served = payload();
    served.nodes[BPM].pages = ['devices/bpm', 'procedures/orbit'];
    const act = actions();
    const pages = row(nodeMenu(view({ payload: served }), BPM, act), 'Open knowledge page');
    expect(pages.run).toBeUndefined();
    const submenu = pages.submenu ?? [];
    expect(labels(submenu)).toEqual(['devices/bpm', 'procedures/orbit']);
    row(submenu, 'procedures/orbit').run?.();
    expect(act.page).toHaveBeenCalledWith('procedures/orbit');
  });

  test('a hub with neighbours left out offers them', () => {
    const shown = view();
    const hub = shown.drawn.nodes.find((node) => node.key === CHANNEL);
    if (!hub) throw new Error('fixture');
    hub.badge = 14;
    const act = actions();
    const items = nodeMenu(shown, CHANNEL, act);
    row(items, 'Show 14 more').run?.();
    expect(act.expand).toHaveBeenCalledWith(CHANNEL);
  });

  test('a name that is the id is copied once', () => {
    const items = nodeMenu(view(), BPM_CLASS, actions());
    expect(labels(items)).toEqual(['# BeamPositionMonitor', 'Focus here', '-', 'Copy name']);
  });

  test('a group of the overview lists its members; a class group can be focused', () => {
    const act = actions();
    const shown = view({ grouped: true });
    const group = nodeMenu(shown, 'g:channel', act);
    expect(labels(group)).toEqual(['# Channels · 2', 'List members']);
    row(group, 'List members').run?.();
    expect(act.select).toHaveBeenCalledWith('g:channel');

    const cls = nodeMenu(shown, `c:${BPM_CLASS}`, act);
    expect(labels(cls)).toEqual([
      '# BeamPositionMonitor · 1',
      'List members',
      'Focus here',
      '-',
      'Copy name',
    ]);
    row(cls, 'Focus here').run?.();
    expect(act.focus).toHaveBeenCalledWith(BPM_CLASS);
    row(cls, 'Copy name').run?.();
    expect(act.copy).toHaveBeenCalledWith('BeamPositionMonitor', 'BeamPositionMonitor');
  });

  test('a key that is not drawn has no menu', () => {
    expect(nodeMenu(view({ grouped: true }), 'g:nothing', actions())).toEqual([]);
    expect(nodeMenu(view(), 99, actions())).toEqual([]);
  });
});

describe('a relation', () => {
  const edge = { a: BPM, b: CHANNEL, relation: 'hasBinding', label: 'hasBinding' };

  test('can be hidden, shown alone, followed to either end, and copied', () => {
    const act = actions();
    const shown = view();
    const items = edgeMenu(shown, edge, act);
    expect(labels(items)).toEqual([
      '# hasBinding',
      'Hide',
      'Show only',
      '-',
      'Select source',
      'Select target',
      '-',
      'Copy as text',
    ]);
    expect(row(items, 'Hide').code).toBe('hasBinding');
    expect(row(items, 'Select source').code).toBe('M/BPM1');
    expect(row(items, 'Select target').code).toBe('M:BPM1:X');

    row(items, 'Hide').run?.();
    const hidden = act.setRelations.mock.calls[0][0];
    expect(hidden.has('hasBinding')).toBe(false);
    expect(hidden.size).toBe(shown.relations.size - 1);
    // The view's own set is not the one changed.
    expect(shown.relations.has('hasBinding')).toBe(true);

    row(items, 'Show only').run?.();
    expect([...act.setRelations.mock.calls[1][0]]).toEqual(['hasBinding']);
    row(items, 'Select target').run?.();
    expect(act.select).toHaveBeenCalledWith(CHANNEL);
    row(items, 'Copy as text').run?.();
    expect(act.copy).toHaveBeenCalledWith(
      'M/BPM1 hasBinding M:BPM1:X',
      'M/BPM1 hasBinding M:BPM1:X'
    );
  });

  test('the only relation drawn cannot be shown alone', () => {
    const items = edgeMenu(view({ relations: new Set(['hasBinding']) }), edge, actions());
    expect(row(items, 'Show only').disabled).toBe(true);
  });
});

describe('the empty stage', () => {
  test('offers the drawing’s verbs and a reset', () => {
    const act = actions();
    const items = stageMenu(view(), act);
    expect(labels(items)).toEqual([
      'Fit to view',
      'Lay out again',
      'Relation labels',
      '-',
      'Show all relations',
      'Reset view',
    ]);
    expect(row(items, 'Relation labels').checked).toBe(true);
    expect(row(items, 'Show all relations').disabled).toBe(true);
    for (const label of ['Fit to view', 'Lay out again', 'Relation labels', 'Reset view']) {
      row(items, label).run?.();
    }
    expect(act.fit).toHaveBeenCalledTimes(1);
    expect(act.relayout).toHaveBeenCalledTimes(1);
    expect(act.setLabels).toHaveBeenCalledWith(false);
    expect(act.reset).toHaveBeenCalledTimes(1);
  });

  test('with a relation hidden, all can be shown again', () => {
    const act = actions();
    const shown = view({ relations: new Set(['hasBinding']), labels: false });
    const items = stageMenu(shown, act);
    expect(row(items, 'Relation labels').checked).toBe(false);
    const all = row(items, 'Show all relations');
    expect(all.disabled).toBe(false);
    all.run?.();
    expect(act.setRelations.mock.calls[0][0].size).toBe(shown.graph.relationCounts.size);
  });
});
