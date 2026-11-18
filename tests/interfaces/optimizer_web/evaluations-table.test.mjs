/**
 * Tests for the table of the evaluations (evaluations-table.js): the rows
 * newest first, the best row, the best-so-far column as it is given or as
 * the direction finds it, and an update that redraws the rows only when
 * they changed.
 */

import { describe, test, expect } from 'vitest';

import { runningJob, storedConfig } from './_fixtures.mjs';
import {
  evaluationsTable,
  updateEvaluationsTable,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/evaluations-table.js';
import { jobFigures } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-model.js';

const records = () => runningJob().data;

/** Records of a run with the constraint `loss < 1`: the highest charge (evaluation 2) fails it. */
const constrained = () => [
  { phase_a: 0, sol_b: 3, charge: 1.0, loss: 0.5 },
  { phase_a: 9, sol_b: 2, charge: 2.5, loss: 3 },
  { phase_a: -9, sol_b: 4, charge: 2.0, loss: 0.2 },
];

/** @param {Record<string, unknown>} [extra] */
const spec = (extra = {}) => ({
  records: records(),
  objective: 'charge',
  direction: 'MAXIMIZE',
  variables: ['phase_a', 'sol_b'],
  bestIndex: 1,
  ...extra,
});

/**
 * The cells of each row, by record index.
 * @param {HTMLElement} wrap
 * @returns {Record<string, (string|null)[]>}
 */
const cellsByIndex = (wrap) =>
  Object.fromEntries(
    Array.from(wrap.querySelectorAll('tbody tr[data-index]')).map((r) => [
      r.getAttribute('data-index'),
      Array.from(r.children).map((c) => c.textContent),
    ])
  );

describe('the evaluations table', () => {
  test('lists the evaluations newest first and marks the best', () => {
    const wrap = evaluationsTable(spec());
    const rows = Array.from(wrap.querySelectorAll('tbody tr'));
    expect(rows.map((r) => r.getAttribute('data-index'))).toEqual(['2', '1', '0']);
    expect(wrap.querySelector('tr.is-best')?.getAttribute('data-index')).toBe('1');
    expect(wrap.querySelectorAll('thead th')).toHaveLength(5);
  });

  test('without a direction it has no best-so-far column', () => {
    const wrap = evaluationsTable(spec({ direction: null }));
    expect(wrap.querySelectorAll('thead th')).toHaveLength(4);
  });

  test('an update redraws the rows only when the records changed', () => {
    const wrap = evaluationsTable(spec());
    const body = wrap.querySelector('tbody');
    updateEvaluationsTable(wrap, spec());
    expect(wrap.querySelector('tbody')).toBe(body);
    updateEvaluationsTable(wrap, spec({ records: [...records(), { charge: 3 }] }));
    expect(wrap.querySelectorAll('tbody tr')).toHaveLength(4);
  });

  test('the best-so-far column follows the bests it is given', () => {
    const config = storedConfig({
      objective_direction: 'MAXIMIZE',
      constraints: { loss: { type: 'LESS_THAN', value: 1 } },
    });
    const f = jobFigures(runningJob({ data: constrained() }), config);
    const wrap = evaluationsTable(
      spec({ records: constrained(), bests: f.bests, bestIndex: f.bestIndex })
    );
    // The best-so-far cell of each row, by record index.
    const cells = Object.fromEntries(
      Array.from(wrap.querySelectorAll('tbody tr')).map((r) => [
        r.getAttribute('data-index'),
        r.children[2].textContent,
      ])
    );
    expect(cells).toEqual({ 0: '1', 1: '1', 2: '2' });
    expect(wrap.querySelector('tr.is-best')?.getAttribute('data-index')).toBe('2');
  });

  test('without the bests given, the best so far follows the direction', () => {
    const up = cellsByIndex(evaluationsTable(spec()));
    const down = cellsByIndex(evaluationsTable(spec({ direction: 'minimize' })));
    // Charge 1, 2.5, 2: the highest so far, and the lowest so far.
    expect([0, 1, 2].map((i) => up[i][2])).toEqual(['1', '2.5', '2.5']);
    expect([0, 1, 2].map((i) => down[i][2])).toEqual(['1', '1', '1']);
  });

  test('each row holds the evaluation number, the objective and each knob; a value the record lacks shows no number', () => {
    const wrap = evaluationsTable(spec({ records: [...records(), { charge: 3 }] }));
    const rows = cellsByIndex(wrap);
    expect(rows[1]).toEqual(['2', '2.5', '2.5', '9', '2']);
    expect(rows[3].slice(0, 3)).toEqual(['4', '3', '3']);
    expect(rows[3].slice(3).some((text) => /\d/.test(text || ''))).toBe(false);
  });

  test('with no evaluation there is no evaluation row, and an update adds the first', () => {
    const wrap = evaluationsTable(spec({ records: [], bestIndex: null }));
    expect(wrap.querySelectorAll('tbody tr[data-index]')).toHaveLength(0);
    expect(wrap.querySelectorAll('tbody tr')).toHaveLength(1);
    updateEvaluationsTable(wrap, spec({ records: records().slice(0, 1), bestIndex: 0 }));
    expect(Object.keys(cellsByIndex(wrap))).toEqual(['0']);
  });

  test('an update keeps the scrolling box, so its scroll position stays', () => {
    const wrap = evaluationsTable(spec());
    const table = wrap.querySelector('table');
    wrap.scrollTop = 40;
    updateEvaluationsTable(wrap, spec({ records: [...records(), { charge: 3 }] }));
    expect(wrap.querySelector('table')).toBe(table);
    expect(wrap.scrollTop).toBe(40);
  });
});
