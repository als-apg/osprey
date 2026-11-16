// @ts-check
/**
 * The evaluations of a run as a table, newest first: the evaluation number,
 * the objective, the best so far and the value of each knob. The row of the
 * best point is marked. It is the chart's companion: the same records, read
 * as numbers.
 *
 * {@link updateEvaluationsTable} redraws the rows of a drawn table only when
 * the records changed, and keeps its scroll position.
 *
 * @module evaluations-table
 */

import { fmt, h } from './hyperscript.js';
import { finite, runningBest } from './run-model.js';

/**
 * `bests` is the best so far at each record, as the run's figures give it
 * (they follow the constraints); without it the table finds it from the
 * direction alone. `bestIndex` marks the row of the best point.
 *
 * @typedef {{
 *   records: Record<string, unknown>[]|null|undefined,
 *   objective: string|null,
 *   direction?: string|null,
 *   variables: string[],
 *   bests?: (number|null)[]|null,
 *   bestIndex?: number|null,
 *   label?: string|null,
 *   labelOf?: (name: string) => string,
 * }} TableSpec
 */

/**
 * @param {TableSpec} spec
 * @returns {Node[]}
 */
function tableParts(spec) {
  const { objective, variables } = spec;
  const records = spec.records || [];
  const labelOf = spec.labelOf || ((/** @type {string} */ name) => name);
  const values = objective ? records.map((r) => finite(r?.[objective])) : [];
  const direction = spec.direction ? String(spec.direction).toUpperCase() : null;
  const bests = spec.bests !== undefined ? spec.bests : runningBest(values, direction);
  const head = h(
    'tr',
    null,
    h('th', { class: 'num', scope: 'col', text: '#' }),
    h('th', { class: 'num', scope: 'col', text: spec.label || objective || 'Objective' }),
    bests ? h('th', { class: 'num', scope: 'col', text: 'Best so far' }) : null,
    ...variables.map((name) =>
      h('th', { class: 'num', scope: 'col', title: name, text: labelOf(name) })
    )
  );
  const rows = [];
  for (let i = records.length - 1; i >= 0; i -= 1) {
    const record = records[i];
    rows.push(
      h(
        'tr',
        { class: i === spec.bestIndex ? 'is-best' : null, 'data-index': i },
        h('td', { class: 'num', text: String(i + 1) }),
        h('td', { class: 'num opt-evals-obj', text: fmt(values[i]) }),
        bests ? h('td', { class: 'num', text: fmt(bests[i]) }) : null,
        ...variables.map((name) => h('td', { class: 'num', text: fmt(record?.[name]) }))
      )
    );
  }
  const cols = variables.length + (bests ? 3 : 2);
  const body = rows.length
    ? rows
    : [
        h(
          'tr',
          null,
          h('td', { class: 'opt-evals-none', colspan: cols, text: 'No evaluation yet.' })
        ),
      ];
  return [h('thead', null, head), h('tbody', null, ...body)];
}

/**
 * @param {TableSpec} spec
 */
function keyOf(spec) {
  return JSON.stringify([
    spec.records?.length ?? 0,
    spec.records?.[spec.records.length - 1] ?? null,
    spec.objective,
    spec.direction,
    spec.variables,
    spec.bestIndex,
    spec.bests === undefined ? 'derived' : spec.bests,
    spec.label,
    spec.variables.map((name) => spec.labelOf?.(name) ?? name),
  ]);
}

/**
 * A scrolling table of the evaluations.
 *
 * @param {TableSpec} spec
 * @returns {HTMLElement}
 */
export function evaluationsTable(spec) {
  const table = h('table', { class: 'opt-evals-table' }, ...tableParts(spec));
  const wrap = h('div', { class: 'opt-evals', tabindex: 0, role: 'region' }, table);
  wrap.setAttribute('aria-label', 'Evaluations, newest first');
  wrap.dataset.key = keyOf(spec);
  return wrap;
}

/**
 * Show `spec` in a drawn table; the rows are redrawn only when it changed.
 *
 * @param {HTMLElement} wrap  an {@link evaluationsTable}
 * @param {TableSpec} spec
 */
export function updateEvaluationsTable(wrap, spec) {
  const key = keyOf(spec);
  if (wrap.dataset.key === key) return;
  wrap.dataset.key = key;
  const table = wrap.querySelector('table');
  if (table) table.replaceChildren(...tableParts(spec));
}
