/**
 * Unit tests for the BLUESKY panel's array-cell helpers
 * (bluesky_web/panels/bluesky/array-cell.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/bluesky_web/array-cell.test.mjs
 *
 * Pure helpers against literal arrays, plus the one detached SVG builder.
 */

import { test, expect, describe } from 'vitest';

import {
  SPARK_HEIGHT,
  SPARK_WIDTH,
  cellPlotPanel,
  isPlottableArray,
  sparklineElement,
  sparklinePoints,
  stepRow,
} from '../../../src/osprey/interfaces/bluesky_web/panels/bluesky/array-cell.js';

/** @param {string} points */
function parsePoints(points) {
  return points.split(' ').map((pair) => pair.split(',').map(Number));
}

describe('isPlottableArray', () => {
  test('a flat list of numbers and nulls is plottable', () => {
    expect(isPlottableArray([0, 1.5, null, -2])).toBe(true);
    expect(isPlottableArray([])).toBe(true);
  });

  test('a nested list, a string list and a scalar are not', () => {
    expect(
      isPlottableArray([
        [1, 2],
        [3, 4],
      ])
    ).toBe(false);
    expect(isPlottableArray(['a', 'b'])).toBe(false);
    expect(isPlottableArray(3)).toBe(false);
    expect(isPlottableArray(null)).toBe(false);
    expect(isPlottableArray({ refused: 'x' })).toBe(false);
  });
});

describe('sparklinePoints', () => {
  test('keeps a one-sample peak of a 1024-element array', () => {
    const values = new Array(1024).fill(0);
    values[517] = 900;
    const points = parsePoints(/** @type {string} */ (sparklinePoints(values)));
    const ys = points.map(([, y]) => y);
    // The peak is the top of the sketch; SVG y grows downwards.
    expect(Math.min(...ys)).toBe(0);
    expect(Math.max(...ys)).toBe(SPARK_HEIGHT);
  });

  test('a flat array is a level line through the middle', () => {
    const points = parsePoints(/** @type {string} */ (sparklinePoints([5, 5, 5, 5])));
    expect(new Set(points.map(([, y]) => y))).toEqual(new Set([SPARK_HEIGHT / 2]));
  });

  test('an array with no finite element has no sketch', () => {
    expect(sparklinePoints([])).toBeNull();
    expect(sparklinePoints([null, null])).toBeNull();
    expect(sparklinePoints([NaN, Infinity])).toBeNull();
  });

  test('never draws more than two points per pixel column', () => {
    const values = Array.from({ length: 1024 }, (_, i) => Math.sin(i / 7));
    const points = parsePoints(/** @type {string} */ (sparklinePoints(values)));
    expect(points.length).toBeLessThanOrEqual(2 * SPARK_WIDTH);
    /** @type {Map<number, number>} */
    const perColumn = new Map();
    for (const [x] of points) perColumn.set(Math.floor(x), (perColumn.get(Math.floor(x)) ?? 0) + 1);
    for (const n of perColumn.values()) expect(n).toBeLessThanOrEqual(2);
  });
});

describe('sparklineElement', () => {
  test('builds an svg whose polyline uses currentColor', () => {
    const svg = /** @type {SVGSVGElement} */ (sparklineElement([0, 3, 1]));
    expect(svg.namespaceURI).toBe('http://www.w3.org/2000/svg');
    expect(svg.getAttribute('class')).toBe('array-spark');
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    const lines = svg.querySelectorAll('polyline');
    expect(lines.length).toBe(1);
    expect(lines[0].getAttribute('stroke')).toBe('currentColor');
    expect(lines[0].getAttribute('points')).toBe(sparklinePoints([0, 3, 1]));
    expect(sparklineElement([null])).toBeNull();
  });
});

describe('cellPlotPanel', () => {
  test('states the column, its unit and the row', () => {
    const panel = cellPlotPanel({
      column: 'det:spectrum',
      unit: 'counts',
      values: [1, 2, 3],
      row: 1,
      rows: 10,
    });
    expect(panel.title).toBe('det:spectrum · row 2 of 10');
    expect(panel.x_label).toBe('index');
    expect(panel.y_label).toBe('det:spectrum');
    expect(panel.y_units).toBe('counts');
    expect(panel.x_units).toBeNull();
    expect(panel.x_columns).toBeNull();
    expect(panel.y_columns).toBeNull();
    expect(panel.annotations).toEqual([]);
    expect(panel.mark.kind).toBe('lines');
    const mark =
      /** @type {import('../../../src/osprey/interfaces/bluesky_web/panels/bluesky/figure-renderer.js').LinesMark} */ (
        panel.mark
      );
    expect(mark.series.length).toBe(1);
    expect(mark.series[0].label).toBe('det:spectrum');
    expect(mark.series[0].points).toEqual([
      { x: 0, y: 1 },
      { x: 1, y: 2 },
      { x: 2, y: 3 },
    ]);
    expect(mark.decimated).toBe(false);
    expect(mark.source_points).toBe(3);
  });

  test('a null element is a gap, never a zero', () => {
    const panel = cellPlotPanel({
      column: 'w',
      unit: null,
      values: [1, null, NaN],
      row: 0,
      rows: 1,
    });
    const mark = /** @type {any} */ (panel.mark);
    expect(mark.series[0].points.map((/** @type {{y: unknown}} */ p) => p.y)).toEqual([
      1,
      null,
      null,
    ]);
  });
});

describe('stepRow', () => {
  test('stops at either end', () => {
    expect(stepRow(0, -1, 10)).toBe(0);
    expect(stepRow(0, 1, 10)).toBe(1);
    expect(stepRow(9, 1, 10)).toBe(9);
    expect(stepRow(9, -1, 10)).toBe(8);
  });
});
