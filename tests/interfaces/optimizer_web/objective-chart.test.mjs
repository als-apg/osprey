/**
 * Tests for the objective chart (objective-chart.js): what the chart draws
 * for its input, that a field it does not get is not drawn, its text
 * alternative, and that a new evaluation moves the drawing in place.
 */

import { describe, test, expect } from 'vitest';

import {
  chartInput,
  chartModel,
  mountObjectiveChart,
  niceTicks,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/objective-chart.js';
import { jobFigures } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-model.js';
import { runningJob, storedConfig } from './_fixtures.mjs';

const records = () => runningJob().data;

/** Records of a run with the constraint `loss < 1`: the highest charge (evaluation 2) fails it. */
const constrained = () => [
  { phase_a: 0, sol_b: 3, charge: 1.0, loss: 0.5 },
  { phase_a: 9, sol_b: 2, charge: 2.5, loss: 3 },
  { phase_a: -9, sol_b: 4, charge: 2.0, loss: 0.2 },
];

/** @param {Record<string, unknown>} [extra] */
const input = (extra = {}) => ({
  records: records(),
  objective: 'charge',
  direction: 'MAXIMIZE',
  ...extra,
});

function mount() {
  const host = document.createElement('div');
  document.body.append(host);
  const chart = mountObjectiveChart(host);
  const svg = /** @type {SVGElement} */ (host.querySelector('svg'));
  /** @param {string} sel */
  const shown = (sel) => {
    const node = svg.querySelector(sel);
    return !!node && node.getAttribute('display') !== 'none';
  };
  return { host, chart, svg, shown };
}

describe('the chart model', () => {
  test('finds the best so far from the records and the direction', () => {
    expect(chartModel(input()).best).toEqual({ index: 1, value: 2.5 });
    expect(chartModel(input({ direction: 'MINIMIZE' })).best).toEqual({ index: 0, value: 1 });
  });

  test('an unknown direction gives no best; a given best is kept', () => {
    expect(chartModel(input({ direction: null })).best).toBeNull();
    expect(chartModel(input({ direction: null, best: null })).bests).toBeNull();
    expect(chartModel(input({ best: { index: 2, value: 2 } })).best).toEqual({
      index: 2,
      value: 2,
    });
  });

  test('the x-range is fixed by xMax and never shorter than the records', () => {
    expect(chartModel(input({ xMax: 10 })).xMax).toBe(10);
    expect(chartModel(input({ xMax: 2 })).xMax).toBe(3);
    expect(chartModel(input()).xMax).toBe(3);
  });

  test('the y-range holds every value and the value before the run', () => {
    const m = chartModel(input({ before: 4 }));
    expect(m.lo).toBeLessThan(1);
    expect(m.hi).toBeGreaterThan(4);
  });

  test('chartInput takes the best of the run figures', () => {
    const f = jobFigures(runningJob({ snapshots: { initial: { charge: 0.5 } } }), storedConfig());
    expect(chartInput(f, records())).toMatchObject({
      objective: 'charge',
      direction: 'MAXIMIZE',
      before: 0.5,
      best: { index: 1, value: 2.5 },
    });
  });

  test('with constraints the best line and the best are the figures’ own, as the service picks them', () => {
    // The service fills the direction in; evaluation 2 is best but fails `loss < 1`.
    const config = storedConfig({
      objective_direction: 'MAXIMIZE',
      constraints: { loss: { type: 'LESS_THAN', value: 1 } },
    });
    const f = jobFigures(runningJob({ data: constrained() }), config);
    const model = chartModel(chartInput(f, constrained()));
    expect(model.bests).toEqual(f.bests);
    expect(model.best?.index).toBe(f.bestIndex);
    expect(model.best?.index).toBe(2);
  });

  test('ticks are round and cover the range', () => {
    const { ticks, step } = niceTicks(0.13, 0.91, 4);
    expect(step).toBe(0.2);
    expect(ticks).toEqual([0.2, 0.4, 0.6, 0.8]);
  });
});

describe('the drawing', () => {
  test('draws each evaluation, the best line and ring, and the value before the run', () => {
    const { chart, svg, shown } = mount();
    chart.update(input({ before: 0.5, xMax: 10 }));
    expect(svg.querySelectorAll('.opt-ch-pt')).toHaveLength(3);
    expect(shown('.opt-ch-bestline')).toBe(true);
    expect(shown('.opt-ch-bestpt')).toBe(true);
    expect(shown('.opt-ch-before')).toBe(true);
    expect(svg.querySelector('.opt-ch-future')).not.toBeNull();
    chart.destroy();
  });

  test('leaves out what it is not given', () => {
    const { chart, svg, shown } = mount();
    chart.update(input({ direction: null }));
    expect(svg.querySelectorAll('.opt-ch-pt')).toHaveLength(3);
    expect(shown('.opt-ch-bestline')).toBe(false);
    expect(shown('.opt-ch-bestpt')).toBe(false);
    expect(shown('.opt-ch-before')).toBe(false);
    expect(shown('.opt-ch-cur')).toBe(false);
    expect(svg.querySelector('.opt-ch-end')).toBeNull();
    expect(svg.querySelector('.opt-ch-future')).toBeNull();
    chart.destroy();
  });

  test('a record without a value for the objective draws no point', () => {
    const { chart, svg } = mount();
    chart.update(input({ records: [...records(), { phase_a: 1, sol_b: 2 }] }));
    expect(svg.querySelectorAll('.opt-ch-pt')).toHaveLength(3);
    chart.destroy();
  });

  test('marks the end and the last point when asked', () => {
    const { chart, svg, shown } = mount();
    chart.update(input({ end: { index: 2, label: 'paused', tone: 'warning' }, current: 'still' }));
    expect(svg.querySelector('.opt-ch-end')?.getAttribute('class')).toContain('is-warning');
    expect(svg.querySelector('.opt-ch-endlabel')?.textContent).toBe('paused');
    expect(shown('.opt-ch-cur')).toBe(true);
    expect(svg.querySelector('.opt-ch-cur')?.getAttribute('class')).not.toContain('is-live');
    chart.update(input({ current: 'live' }));
    expect(svg.querySelector('.opt-ch-cur')?.getAttribute('class')).toContain('is-live');
    chart.destroy();
  });

  test('has a text alternative with the count, the best and the value before', () => {
    const { chart, svg } = mount();
    chart.update(input({ label: 'Charge', units: 'nC', before: 0.5 }));
    expect(svg.getAttribute('role')).toBe('img');
    const said = svg.getAttribute('aria-label') || '';
    expect(said).toContain('3 evaluations');
    expect(said).toContain('2.5 nC at evaluation 2');
    expect(said).toContain('0.5 nC');
    chart.destroy();
  });

  test('a new evaluation moves the drawn points in place and adds one', () => {
    const { chart, svg } = mount();
    chart.update(input({ xMax: 10 }));
    const box = svg.parentElement;
    const first = svg.querySelector('.opt-ch-pt');
    const line = svg.querySelector('.opt-ch-bestline');
    chart.update(input({ xMax: 10, records: [...records(), { charge: 3 }] }));
    expect(svg.parentElement).toBe(box);
    expect(svg.querySelector('.opt-ch-pt')).toBe(first);
    expect(svg.querySelector('.opt-ch-bestline')).toBe(line);
    expect(svg.querySelectorAll('.opt-ch-pt')).toHaveLength(4);
    chart.destroy();
  });

  test('an evaluation that fails a constraint is drawn as such', () => {
    const { chart, svg } = mount();
    const config = storedConfig({ constraints: { loss: { type: 'LESS_THAN', value: 1 } } });
    const f = jobFigures(runningJob({ data: constrained() }), config);
    chart.update(chartInput(f, constrained()));
    const bad = Array.from(svg.querySelectorAll('.opt-ch-pt.is-bad'));
    expect(bad.map((n) => n.getAttribute('data-index'))).toEqual(['1']);
    chart.destroy();
  });

  test('records with no finite value are not "no evaluation yet"', () => {
    const none = mount();
    none.chart.update(input({ records: [], xMax: 10 }));
    const unread = mount();
    unread.chart.update(input({ records: [{ phase_a: 1, sol_b: 3, charge: null }] }));
    const said = (/** @type {SVGElement} */ svg) => svg.querySelector('.opt-ch-empty')?.textContent;
    expect(said(unread.svg)).toBeTruthy();
    expect(said(unread.svg)).not.toBe(said(none.svg));
    none.chart.destroy();
    unread.chart.destroy();
  });

  test('with no evaluation yet it says so', () => {
    const { chart, svg } = mount();
    chart.update(input({ records: [], xMax: 10 }));
    expect(svg.querySelectorAll('.opt-ch-pt')).toHaveLength(0);
    expect(svg.querySelector('.opt-ch-empty')).not.toBeNull();
    chart.destroy();
  });
});
