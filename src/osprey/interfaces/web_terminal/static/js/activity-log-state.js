// @ts-check
/**
 * What the operator has chosen on the Activity page, and what a press does to
 * it. The page reads a pressed element's `data-*` attributes and hands them to
 * `applyAction`; what is not a change of this state (a session pick, a refresh,
 * a copy) comes back `false` and stays with the page.
 *
 * @module activity-log-state
 */

import { VIEWS } from './activity-log-link.js';
import { callKey, findCallPath, walk } from './activity-log-model.js';

/** @typedef {import('./activity-log-link.js').View} View */
/** @typedef {import('./activity-log-model.js').ActivityNode} ActivityNode */
/** @typedef {import('./activity-log-rows.js').UiState} UiState */

/**
 * @typedef {Object} ActionContext  What a press is read against.
 * @property {ActivityNode[]} nodes  The session shown.
 * @property {string[]} toggles  The rows on the page that open, by their `data-toggle`.
 */

/**
 * @param {string | undefined} name
 * @returns {View | undefined}
 */
function viewNamed(name) {
  return VIEWS.find((view) => view === name);
}

/**
 * @param {Set<string>} set
 * @param {string} key
 */
function flip(set, key) {
  if (!set.delete(key)) set.add(key);
}

/**
 * The state of a page just opened.
 * @param {View} [view]
 * @returns {UiState}
 */
export function newUiState(view = 'timeline') {
  return {
    view,
    filter: 'all',
    scope: 'all',
    q: '',
    sort: { k: 'ts', dir: 1 },
    open: new Set(),
    closed: new Set(),
    full: new Map(),
  };
}

/**
 * Drop what belonged to the session left behind. The view, and how the
 * tool-call table is scoped and sorted, are the operator's and stay.
 * @param {UiState} ui
 */
export function resetForSession(ui) {
  ui.filter = 'all';
  ui.q = '';
  ui.open.clear();
  ui.closed.clear();
  ui.full.clear();
}

/**
 * Put the call `key` on show: the whole timeline, every subagent group around
 * the call open, the call itself expanded.
 * @param {UiState} ui
 * @param {ActivityNode[]} nodes
 * @param {string} key  a call's `callKey`
 */
export function revealCall(ui, nodes, key) {
  ui.view = 'timeline';
  ui.filter = 'all';
  ui.q = '';
  for (const parent of findCallPath(nodes, key) ?? []) ui.closed.delete(parent);
  ui.open.add(key);
}

/**
 * Apply one press to the state.
 * @param {UiState} ui
 * @param {Record<string, string | undefined>} data  the pressed element's dataset
 * @param {ActionContext} ctx
 * @returns {boolean} whether the state changed, so the page renders again
 */
export function applyAction(ui, data, ctx) {
  if (data.view !== undefined) {
    const view = viewNamed(data.view);
    if (!view) return false;
    ui.view = view;
    return true;
  }
  if (data.go !== undefined) {
    const [name, filter] = data.go.split(':');
    const view = viewNamed(name);
    if (!view) return false;
    ui.view = view;
    if (view === 'timeline' && filter) ui.filter = filter;
    return true;
  }
  if (data.jump !== undefined) {
    revealCall(ui, ctx.nodes, data.jump);
    return true;
  }
  if (data.filter !== undefined) {
    ui.filter = data.filter;
    return true;
  }
  if (data.scope !== undefined) {
    ui.scope = data.scope;
    return true;
  }
  if (data.sort !== undefined) {
    ui.sort = { k: data.sort, dir: ui.sort.k === data.sort ? -ui.sort.dir : 1 };
    return true;
  }
  if (data.toggle !== undefined) {
    flip(ui.open, data.toggle);
    return true;
  }
  if (data.collapse !== undefined) {
    flip(ui.closed, data.collapse);
    return true;
  }
  if (data.act === 'expand') {
    ui.closed.clear();
    for (const id of ctx.toggles) ui.open.add(id);
    walk(ctx.nodes, (node) => {
      // A subagent is a group, open unless collapsed; only calls expand.
      if (node.t === 'tool' && !node.sub) ui.open.add(callKey(node));
    });
    return true;
  }
  if (data.act === 'collapse') {
    ui.open.clear();
    return true;
  }
  return false;
}
