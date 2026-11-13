// @ts-check
/*
 * OKF Knowledge Panel — the Graph view's inspector: what is known of the
 * picked node.
 *
 * For a node: its kind and class, its description, its properties, its
 * relations in each direction, and the knowledge pages that name it. For a
 * group of the overview: what is in it. Each listed node is a row that
 * draws that node's neighbourhood.
 */

import { el } from './helpers.js';
import { count, KIND_NAMES, relationsOf } from './graph-model.js';

/** @typedef {import('./graph-model.js').GraphIndex} GraphIndex */
/** @typedef {import('./graph-model.js').DrawnGraph} DrawnGraph */
/** @typedef {import('./graph-model.js').NodeKey} NodeKey */

/**
 * @typedef {object} InspectorView
 * @property {GraphIndex} graph
 * @property {DrawnGraph} drawn what the stage shows
 * @property {boolean} grouped the stage shows the overview
 * @property {NodeKey | null} focus the node a neighbourhood is drawn around
 * @property {NodeKey | null} selected the picked node or group
 */

/**
 * @typedef {object} InspectorHandles
 * @property {(i: number) => void} onFocus draw a node's neighbourhood
 * @property {(i: number) => void} onExpand draw the neighbours a hub left out
 * @property {(conceptId: string) => void} onPage open a knowledge page
 */

const MEMBER_LIMIT = 60;
const RELATED_LIMIT = 30;

/**
 * A row that names a node and draws its neighbourhood when picked.
 * @param {GraphIndex} graph
 * @param {number} i
 * @param {(i: number) => void} onFocus
 * @returns {HTMLElement}
 */
export function nodeRow(graph, i, onFocus) {
  const node = graph.nodes[i];
  const row = el(
    'button',
    { class: 'osprey-list-row okf-graph-row', type: 'button', title: node.label },
    [
      el('span', { class: 'osprey-list-title okf-graph-row-name', text: node.label }),
      el('span', { class: 'okf-graph-row-kind', text: KIND_NAMES[node.kind] }),
    ]
  );
  row.addEventListener('click', () => onFocus(i));
  return row;
}

/**
 * @param {string} label
 * @param {() => void} onClick
 * @returns {HTMLElement}
 */
function action(label, onClick) {
  const button = el('button', { class: 'osprey-btn', type: 'button', text: label });
  button.addEventListener('click', onClick);
  return button;
}

/**
 * @param {string} chip
 * @param {string} title
 * @param {(string | null | undefined)[]} lines
 * @returns {HTMLElement}
 */
function head(chip, title, lines) {
  return el('div', { class: 'okf-graph-head' }, [
    el('span', { class: 'osprey-chip' }, [el('span', { class: 'osprey-chip-label', text: chip })]),
    el('h2', { class: 'okf-graph-title', text: title }),
    ...lines.map((line) => (line ? el('p', { class: 'okf-graph-description', text: line }) : null)),
  ]);
}

/**
 * Show the picked node or group in `host`, which is hidden with nothing picked.
 * @param {HTMLElement} host
 * @param {InspectorView} view
 * @param {InspectorHandles} handles
 */
export function renderInspector(host, view, { onFocus, onExpand, onPage }) {
  const { graph, drawn, grouped, focus, selected } = view;
  const picked = drawn.nodes.find((node) => node.key === selected);

  /**
   * A titled list of node rows, cut at `limit` with a line that says so.
   * @param {string} title
   * @param {number[]} nodes
   * @param {number} limit
   * @param {string} [more] what the cut line adds after the count
   */
  const nodeList = (title, nodes, limit, more = '') =>
    el('section', { class: 'okf-graph-list' }, [
      el('div', { class: 'osprey-section-label', text: title }),
      ...nodes.slice(0, limit).map((i) => nodeRow(graph, i, onFocus)),
      nodes.length > limit
        ? el('p', { class: 'okf-graph-more', text: `${count(nodes.length - limit)} more${more}` })
        : null,
    ]);

  /** @param {(HTMLElement | null)[]} parts */
  const show = (parts) => {
    host.replaceChildren(...parts.flatMap((part) => part ?? []));
    host.scrollTop = 0;
  };

  // With nothing picked the inspector is closed: the drawing has the room.
  host.hidden = selected === null || (typeof selected === 'string' && !picked);
  if (host.hidden) {
    host.replaceChildren();
    return;
  }

  if (typeof selected === 'string' && picked) {
    const classNode = picked.classNode;
    const cls = classNode === undefined ? null : graph.nodes[classNode];
    const members = picked.members || [];
    const also = cls && cls.props.altLabel;
    show([
      head(cls ? 'Class' : 'Group', cls ? cls.label : KIND_NAMES[picked.kind] + 's', [
        `${count(members.length)} in the graph`,
        also ? 'Also called ' + also : null,
      ]),
      classNode === undefined
        ? null
        : el('div', { class: 'okf-graph-actions' }, [
            action('Focus here', () => onFocus(classNode)),
          ]),
      members.length ? nodeList('Members', members, MEMBER_LIMIT, ': use the search field') : null,
    ]);
    return;
  }

  const i = /** @type {number} */ (selected);
  const node = graph.nodes[i];
  const cls = node.type ? graph.classIndex.get(node.type) : undefined;
  const pages = node.pages || [];
  const props = Object.entries(node.props).filter(
    ([key]) => key !== 'description' && key !== 'familyDescription'
  );
  show([
    head(
      KIND_NAMES[node.kind] + (cls === undefined ? '' : ' · ' + graph.nodes[cls].label),
      node.label,
      [node.props.description || node.props.familyDescription]
    ),
    el('div', { class: 'okf-graph-actions' }, [
      !grouped && focus === i ? null : action('Focus here', () => onFocus(i)),
      picked && picked.badge ? action(`Show ${count(picked.badge)} more`, () => onExpand(i)) : null,
      pages.length === 1 ? action('Knowledge page', () => onPage(pages[0])) : null,
    ]),
    pages.length > 1
      ? el('section', { class: 'okf-graph-list' }, [
          el('div', { class: 'osprey-section-label', text: 'Knowledge pages' }),
          ...pages.map((page) => {
            const row = el(
              'button',
              { class: 'osprey-list-row okf-graph-row', type: 'button', title: page },
              [el('span', { class: 'osprey-list-title okf-graph-row-name', text: page })]
            );
            row.addEventListener('click', () => onPage(page));
            return row;
          }),
        ])
      : null,
    props.length
      ? el(
          'dl',
          { class: 'okf-graph-props' },
          props.flatMap(([key, value]) => [el('dt', { text: key }), el('dd', { text: value })])
        )
      : null,
    ...relationsOf(graph, i).map((group) =>
      nodeList(
        `${group.outgoing ? '→' : '←'} ${group.relation} · ${count(group.nodes.length)}`,
        group.nodes,
        RELATED_LIMIT
      )
    ),
  ]);
}
