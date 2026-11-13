// @ts-check
/*
 * OKF Knowledge Panel — the Graph view's right-click menus.
 *
 * One builder per surface of the drawing: a node (or a group of the
 * overview), a relation, and the empty stage. Each takes what the view shows
 * plus an {@link Actions} object of bound verbs, and returns the rows the
 * shared context menu (/design-system/js/context-menu.js) draws. Nothing here
 * touches the document, so which verb a surface offers, and when it is
 * disabled, is tested without a browser.
 *
 * The verbs are the ones the view already has: "Focus here" is a
 * double-click, "Hide relation" is the legend's checkbox, "Fit to view" is
 * the strip's button. A menu adds reach, never a second code path.
 */

import { count } from './graph-model.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */
/** @typedef {import('./graph-model.js').GraphIndex} GraphIndex */
/** @typedef {import('./graph-model.js').DrawnGraph} DrawnGraph */
/** @typedef {import('./graph-model.js').DrawnEdge} DrawnEdge */
/** @typedef {import('./graph-model.js').NodeKey} NodeKey */

/**
 * What a menu needs to know about the view at the moment it opens.
 * @typedef {object} MenuView
 * @property {GraphIndex} graph
 * @property {DrawnGraph} drawn what the stage shows
 * @property {boolean} grouped the stage shows the overview
 * @property {NodeKey | null} focus the node a neighbourhood is drawn around
 * @property {Set<string>} relations the relations drawn
 * @property {boolean} labels relation labels are drawn
 */

/**
 * The verbs a menu row can run, bound by the view.
 * @typedef {object} Actions
 * @property {(key: NodeKey) => void} select pick a node or a group
 * @property {(i: number) => void} focus draw a node's neighbourhood
 * @property {(i: number) => void} expand draw the neighbours a hub left out
 * @property {(conceptId: string) => void} page open a knowledge page
 * @property {(text: string, what: string) => void} copy copy text, naming it in the status line
 * @property {(relations: Set<string>) => void} setRelations draw these relations only
 * @property {() => void} fit
 * @property {() => void} relayout
 * @property {(on: boolean) => void} setLabels
 * @property {() => void} reset
 */

/**
 * A node of a neighbourhood, or a group of the overview.
 * @param {MenuView} view
 * @param {NodeKey} key
 * @param {Actions} actions
 * @returns {MenuItem[]}
 */
export function nodeMenu(view, key, actions) {
  const { graph, drawn, grouped, focus } = view;
  const picked = drawn.nodes.find((node) => node.key === key);

  if (typeof key === 'string') {
    if (!picked) return [];
    const classNode = picked.classNode;
    /** @type {MenuItem[]} */
    const items = [
      { heading: picked.label },
      { label: 'List members', run: () => actions.select(key) },
    ];
    if (classNode !== undefined) {
      const name = graph.nodes[classNode].label;
      items.push(
        { label: 'Focus here', run: () => actions.focus(classNode) },
        { divider: true },
        { label: 'Copy name', glyph: '⎘', run: () => actions.copy(name, name) }
      );
    }
    return items;
  }

  const node = graph.nodes[key];
  if (!node) return [];
  const pages = node.pages || [];
  /** @type {MenuItem[]} */
  const items = [
    { heading: node.label },
    {
      label: 'Focus here',
      hint: 'Double-click',
      disabled: !grouped && focus === key,
      tip: !grouped && focus === key ? 'The drawing is already focused on this node' : undefined,
      run: () => actions.focus(key),
    },
  ];
  if (picked && picked.badge) {
    items.push({ label: `Show ${count(picked.badge)} more`, run: () => actions.expand(key) });
  }
  if (pages.length === 1) {
    items.push({ label: 'Open knowledge page', run: () => actions.page(pages[0]) });
  } else if (pages.length > 1) {
    items.push({
      label: 'Open knowledge page',
      submenu: pages.map((page) => ({ label: page, run: () => actions.page(page) })),
    });
  }
  items.push(
    { divider: true },
    { label: 'Copy name', glyph: '⎘', run: () => actions.copy(node.label, node.label) }
  );
  if (node.id !== node.label) {
    items.push({ label: 'Copy id', run: () => actions.copy(node.id, node.id) });
  }
  return items;
}

/**
 * A relation line.
 * @param {MenuView} view
 * @param {DrawnEdge} edge
 * @param {Actions} actions
 * @returns {MenuItem[]}
 */
export function edgeMenu(view, edge, actions) {
  const { drawn, relations } = view;
  const nameOf = (/** @type {NodeKey} */ key) =>
    drawn.nodes.find((node) => node.key === key)?.label ?? String(key);
  const source = nameOf(edge.a);
  const target = nameOf(edge.b);
  const text = `${source} ${edge.relation} ${target}`;
  const without = new Set(relations);
  without.delete(edge.relation);
  return [
    { heading: edge.label },
    { label: 'Hide', code: edge.relation, run: () => actions.setRelations(without) },
    {
      label: 'Show only',
      code: edge.relation,
      disabled: relations.size <= 1,
      run: () => actions.setRelations(new Set([edge.relation])),
    },
    { divider: true },
    { label: 'Select source', code: source, run: () => actions.select(edge.a) },
    { label: 'Select target', code: target, run: () => actions.select(edge.b) },
    { divider: true },
    { label: 'Copy as text', glyph: '⎘', run: () => actions.copy(text, text) },
  ];
}

/**
 * The empty stage: the drawing as a whole.
 * @param {MenuView} view
 * @param {Actions} actions
 * @returns {MenuItem[]}
 */
export function stageMenu(view, actions) {
  const { graph, relations, labels } = view;
  const all = new Set(graph.relationCounts.keys());
  return [
    { label: 'Fit to view', run: () => actions.fit() },
    { label: 'Lay out again', run: () => actions.relayout() },
    { label: 'Relation labels', checked: labels, run: () => actions.setLabels(!labels) },
    { divider: true },
    {
      label: 'Show all relations',
      disabled: relations.size >= all.size,
      run: () => actions.setRelations(all),
    },
    { label: 'Reset view', run: () => actions.reset() },
  ];
}
