/**
 * Unit tests for the BLUESKY panel's figure renderer
 * (bluesky_panels/panels/bluesky/figure-renderer.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/bluesky_panels/figure-renderer.test.mjs
 *
 * Two layers:
 *  - The layout half against plain figure objects in the shape
 *    `services/bluesky_bridge/figure.py` dumps — scales, ticks, the gap split,
 *    heatmap/bar geometry, and the annotation text. No DOM involved.
 *  - `renderFigure` against a real element bag, driving the *real*
 *    theme-manager bridges through a `:root` custom-property fixture. Flipping
 *    that fixture between renders is how the "colors are re-read, never
 *    cached" rule is actually checked rather than asserted.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  FIGURE_HEIGHT,
  FIGURE_MARGIN,
  FIGURE_WIDTH,
  axisTicks,
  axisTitle,
  barDomain,
  barGeometry,
  categoryTicks,
  decimationAnnotations,
  divergingScale,
  figureNoteText,
  formatTickValue,
  gapAnnotations,
  gapCount,
  heatmapCells,
  heatmapExtent,
  heatmapShape,
  labelStride,
  linearScale,
  padRange,
  panelAnnotations,
  plotBox,
  polylinePoints,
  renderFigure,
  seriesColor,
  seriesExtent,
  shownPointCount,
  splitSegments,
} from '../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js';
import {
  groupPanels,
  resolveSectionSelection,
  resolveSeriesSelection,
} from '../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-controls.js';

/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').FigurePoint} FigurePoint */
/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').FigureSeries} FigureSeries */
/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').LinesMark} LinesMark */
/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').HeatmapMark} HeatmapMark */
/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').BarsMark} BarsMark */
/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').FigureMark} FigureMark */
/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').FigurePanel} FigurePanel */
/** @typedef {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').Figure} Figure */

// ---------------------------------------------------------------------------
// Wire-shaped builders — every field the bridge dumps, so a test object is
// exactly what the renderer sees at runtime.
// ---------------------------------------------------------------------------

/**
 * @param {Array<[number, number|null]>} pairs
 * @returns {FigurePoint[]}
 */
const points = (pairs) => pairs.map(([x, y]) => ({ x, y }));

/**
 * @param {string} label
 * @param {Array<[number, number|null]>} pairs
 * @param {{decimated?: boolean, source_points?: number}} [extra]
 * @returns {FigureSeries}
 */
function series(label, pairs, extra = {}) {
  const pts = points(pairs);
  return {
    label,
    points: pts,
    decimated: extra.decimated ?? false,
    source_points: extra.source_points ?? pts.length,
  };
}

/**
 * @param {FigureSeries[]} lines
 * @param {{decimated?: boolean, source_points?: number}} [extra]
 * @returns {LinesMark}
 */
function linesMark(lines, extra = {}) {
  const total = lines.reduce((sum, line) => sum + line.points.length, 0);
  return {
    kind: 'lines',
    series: lines,
    decimated: extra.decimated ?? false,
    source_points: extra.source_points ?? total,
  };
}

/**
 * @param {string[]} xLabels
 * @param {string[]} yLabels
 * @param {Array<Array<number|null>>} values
 * @param {{value_label?: string, value_units?: string|null, decimated?: boolean, source_points?: number}} [extra]
 * @returns {HeatmapMark}
 */
function heatmapMark(xLabels, yLabels, values, extra = {}) {
  const cells = values.reduce((sum, row) => sum + row.length, 0);
  return {
    kind: 'heatmap',
    x_labels: xLabels,
    y_labels: yLabels,
    values,
    value_label: extra.value_label ?? 'response',
    value_units: extra.value_units ?? null,
    decimated: extra.decimated ?? false,
    source_points: extra.source_points ?? cells,
  };
}

/**
 * @param {string[]} categories
 * @param {Array<number|null>} values
 * @param {{label?: string, decimated?: boolean, source_points?: number}} [extra]
 * @returns {BarsMark}
 */
function barsMark(categories, values, extra = {}) {
  return {
    kind: 'bars',
    label: extra.label ?? 'residual',
    categories,
    values,
    decimated: extra.decimated ?? false,
    source_points: extra.source_points ?? values.length,
  };
}

/**
 * @param {FigureMark} mark
 * @param {{title?: string, x_label?: string, y_label?: string, x_units?: string|null, y_units?: string|null, annotations?: string[], section?: string|null, series_picker?: boolean}} [extra]
 * @returns {FigurePanel}
 */
function panel(mark, extra = {}) {
  return {
    title: extra.title ?? 'Panel',
    x_label: extra.x_label ?? 'corrector',
    y_label: extra.y_label ?? 'orbit',
    x_units: extra.x_units ?? null,
    y_units: extra.y_units ?? null,
    annotations: extra.annotations ?? [],
    mark,
    section: extra.section ?? null,
    series_picker: extra.series_picker ?? false,
  };
}

/**
 * @param {FigurePanel[]} panels
 * @param {{partial?: boolean, source?: 'live'|'tiled', reason?: string|null}} [extra]
 * @returns {Figure}
 */
function figure(panels, extra = {}) {
  return {
    panels,
    partial: extra.partial ?? false,
    source: extra.source ?? 'tiled',
    reason: extra.reason ?? null,
  };
}

// ---------------------------------------------------------------------------
// Layout — pure functions, no DOM
// ---------------------------------------------------------------------------

describe('plot geometry', () => {
  test('the plot box is the viewBox minus the margins', () => {
    const box = plotBox();
    expect(box.left).toBe(FIGURE_MARGIN.left);
    expect(box.top).toBe(FIGURE_MARGIN.top);
    expect(box.width).toBe(FIGURE_WIDTH - FIGURE_MARGIN.left - FIGURE_MARGIN.right);
    expect(box.height).toBe(FIGURE_HEIGHT - FIGURE_MARGIN.top - FIGURE_MARGIN.bottom);
    expect(box.right).toBe(box.left + box.width);
    expect(box.bottom).toBe(box.top + box.height);
  });

  test('margins wider than the canvas still leave a positive box', () => {
    const box = plotBox(40, 40);
    expect(box.width).toBeGreaterThan(0);
    expect(box.height).toBeGreaterThan(0);
  });
});

describe('linearScale', () => {
  test('maps the domain onto the range, inverted or not', () => {
    const scale = linearScale(0, 10, 100, 200);
    expect(scale(0)).toBe(100);
    expect(scale(10)).toBe(200);
    expect(scale(5)).toBe(150);
    const inverted = linearScale(0, 10, 200, 100);
    expect(inverted(0)).toBe(200);
    expect(inverted(10)).toBe(100);
  });

  test('a flat domain lands in the middle of the range, never on an edge or NaN', () => {
    const scale = linearScale(7, 7, 0, 100);
    expect(scale(7)).toBe(50);
    expect(scale(999)).toBe(50);
  });
});

describe('padRange', () => {
  test('widens a real span by the fraction', () => {
    expect(padRange(0, 10)).toEqual([-0.5, 10.5]);
  });

  test('widens a flat domain symmetrically', () => {
    const [low, high] = padRange(4, 4);
    expect(low).toBeLessThan(4);
    expect(high).toBeGreaterThan(4);
    expect(4 - low).toBeCloseTo(high - 4);
  });

  test('a flat zero domain gets an absolute pad rather than nothing', () => {
    expect(padRange(0, 0)).toEqual([-0.5, 0.5]);
  });
});

describe('seriesExtent', () => {
  test('spans every series', () => {
    const extent = seriesExtent([
      series('a', [
        [0, 1],
        [5, 3],
      ]),
      series('b', [
        [2, -4],
        [9, 0],
      ]),
    ]);
    expect(extent).toEqual({ xMin: 0, xMax: 9, yMin: -4, yMax: 3 });
  });

  test('a gap contributes its x but not a y — the run sampled there', () => {
    const extent = seriesExtent([
      series('a', [
        [0, 1],
        [50, null],
      ]),
    ]);
    expect(extent).toEqual({ xMin: 0, xMax: 50, yMin: 1, yMax: 1 });
  });

  test('no finite y anywhere means there is nothing to plot', () => {
    expect(
      seriesExtent([
        series('a', [
          [0, null],
          [1, null],
        ]),
      ])
    ).toBeNull();
    expect(seriesExtent([])).toBeNull();
    expect(seriesExtent([series('a', [])])).toBeNull();
  });
});

describe('splitSegments', () => {
  test('a gap breaks the run instead of being interpolated across', () => {
    const segments = splitSegments(
      points([
        [0, 1],
        [1, 2],
        [2, null],
        [3, 4],
        [4, 5],
      ])
    );
    expect(segments).toHaveLength(2);
    expect(segments[0].map((p) => p.x)).toEqual([0, 1]);
    expect(segments[1].map((p) => p.x)).toEqual([3, 4]);
    // The gap's own x appears in neither run: nothing is drawn at an x where
    // the run recorded no reading.
    expect(segments.flat().map((p) => p.x)).not.toContain(2);
  });

  test('leading, trailing and consecutive gaps produce no empty runs', () => {
    const segments = splitSegments(
      points([
        [0, null],
        [1, null],
        [2, 1],
        [3, null],
        [4, null],
      ])
    );
    expect(segments).toHaveLength(1);
    expect(segments[0]).toHaveLength(1);
  });

  test('an all-gap series has no runs at all', () => {
    expect(
      splitSegments(
        points([
          [0, null],
          [1, null],
        ])
      )
    ).toEqual([]);
  });

  test('gapCount counts the samples with no reading', () => {
    expect(
      gapCount([
        series('a', [
          [0, 1],
          [1, null],
        ]),
        series('b', [
          [0, null],
          [1, null],
        ]),
      ])
    ).toBe(3);
  });

  test('polylinePoints emits scaled "x,y" pairs', () => {
    const identity = linearScale(0, 10, 0, 10);
    expect(polylinePoints(points([[0, 1], [10, 9]]), identity, identity)).toBe('0.0,1.0 10.0,9.0');
  });
});

describe('tick formatting and placement', () => {
  test('formats readable values and falls back to exponential at the extremes', () => {
    expect(formatTickValue(0)).toBe('0');
    expect(formatTickValue(1.23456)).toBe('1.235');
    expect(formatTickValue(-42)).toBe('-42');
    expect(formatTickValue(1.5e6)).toBe('1.5e+6');
    expect(formatTickValue(2e-5)).toBe('2.0e-5');
    expect(formatTickValue(null)).toBe('—');
    expect(formatTickValue(NaN)).toBe('—');
  });

  test('axisTicks spans the domain at the requested count', () => {
    const scale = linearScale(0, 100, 0, 500);
    const ticks = axisTicks(0, 100, scale, 5);
    expect(ticks).toHaveLength(5);
    expect(ticks[0]).toEqual({ pos: 0, label: '0' });
    expect(ticks[4]).toEqual({ pos: 500, label: '100' });
  });

  test('a flat domain gets a single tick', () => {
    const ticks = axisTicks(3, 3, linearScale(3, 3, 0, 10));
    expect(ticks).toEqual([{ pos: 5, label: '3' }]);
  });

  test('labelStride thins only once the axis is crowded', () => {
    expect(labelStride(5)).toBe(1);
    expect(labelStride(12)).toBe(1);
    expect(labelStride(24)).toBe(2);
    expect(labelStride(100)).toBe(9);
  });

  test('categoryTicks centers each label in its slot', () => {
    const ticks = categoryTicks(['a', 'b', 'c', 'd'], 0, 400);
    expect(ticks.map((tick) => tick.label)).toEqual(['a', 'b', 'c', 'd']);
    expect(ticks.map((tick) => tick.pos)).toEqual([50, 150, 250, 350]);
    expect(categoryTicks([], 0, 100)).toEqual([]);
  });

  test('a crowded category axis is thinned by the stride', () => {
    const labels = Array.from({ length: 30 }, (_, index) => `c${index}`);
    const ticks = categoryTicks(labels, 0, 300);
    expect(ticks.length).toBeLessThanOrEqual(12);
    expect(ticks[0].label).toBe('c0');
    expect(ticks[1].label).toBe('c3');
  });
});

describe('panel grouping and selection', () => {
  /**
   * @param {string} title
   * @param {string|null} [section]
   */
  const p = (title, section = null) =>
    panel(linesMark([series('a', [[0, 1]])]), { title, section });

  test('sections gather in first-appearance order, inline panels keep theirs', () => {
    const grouped = groupPanels([
      p('Matrix'),
      p('Response by BPM'),
      p('Spectrum', 'Singular values'),
      p('c1 sweep', 'Sweeps'),
      p('c2 sweep', 'Sweeps'),
    ]);

    expect(grouped.inline.map((panel) => panel.title)).toEqual(['Matrix', 'Response by BPM']);
    expect(grouped.sections.map((group) => group.name)).toEqual(['Singular values', 'Sweeps']);
    expect(grouped.sections[1].panels.map((panel) => panel.title)).toEqual([
      'c1 sweep',
      'c2 sweep',
    ]);
  });

  test('a figure with no sections groups exactly as it always did', () => {
    const grouped = groupPanels([p('one'), p('two')]);
    expect(grouped.inline).toHaveLength(2);
    expect(grouped.sections).toEqual([]);
  });

  test('a stored series selection is kept, minus anything no longer present', () => {
    // A live run drops correctors past the payload cap; the selection must
    // survive with what remains rather than resetting wholesale.
    expect(resolveSeriesSelection(['c1', 'c9'], ['c1', 'c2', 'c3'])).toEqual(['c1']);
  });

  test('a selection with nothing left falls back to the default, never to empty', () => {
    // An empty plot reads as "no data", which would be a lie about the run.
    expect(resolveSeriesSelection(['gone'], ['c1', 'c2', 'c3', 'c4'])).toEqual([
      'c1',
      'c2',
      'c3',
    ]);
    expect(resolveSeriesSelection(undefined, ['c1', 'c2'])).toEqual(['c1', 'c2']);
    expect(resolveSeriesSelection(undefined, [])).toEqual([]);
  });

  test('a stored section choice survives only while its panel does', () => {
    expect(resolveSectionSelection('c2 sweep', ['c1 sweep', 'c2 sweep'])).toBe('c2 sweep');
    expect(resolveSectionSelection('gone', ['c1 sweep', 'c2 sweep'])).toBe('c1 sweep');
    expect(resolveSectionSelection(undefined, [])).toBe('');
  });
});

/**
 * `divergingScale` for values known to hold a reading, typed as non-null.
 * @param {Array<Array<number|null>>} values
 */
function scaleOf(values) {
  const scale = divergingScale(values);
  if (scale === null) throw new Error('expected a scale');
  return scale;
}

describe('heatmap geometry', () => {
  const mark = heatmapMark(
    ['x0', 'x1', 'x2'],
    ['y0', 'y1'],
    [
      [1, 2, 3],
      [4, null, 6],
    ]
  );

  test('shape comes from the labels', () => {
    expect(heatmapShape(mark)).toEqual({ rows: 2, cols: 3 });
  });

  test('shape falls back to the values when labels are absent', () => {
    expect(heatmapShape(heatmapMark([], [], [[1, 2], [3, 4], [5, 6]]))).toEqual({
      rows: 3,
      cols: 2,
    });
  });

  test('the diverging scale anchors zero and takes a symmetric limit', () => {
    // A signed matrix: an ORM's cells are responses either side of zero.
    const scale = scaleOf([
      [-4, 2],
      [1, 3],
    ]);

    // The limit is max|value|, NOT the raw extent — so equal magnitudes of
    // opposite sign are equally salient, and two runs stay comparable.
    expect(scale.limit).toBe(4);

    // Zero is the plot background, so an unresponsive BPM reads as dead
    // rather than as the mid-tone a min→max ramp would give it.
    expect(scale.opacityFor(0)).toBe(0);

    // Equal magnitudes, opposite signs: same weight, different hue.
    expect(scale.opacityFor(-4)).toBe(scale.opacityFor(4));
    expect(scale.isNegative(-4)).toBe(true);
    expect(scale.isNegative(4)).toBe(false);

    // Magnitude scales linearly against the limit.
    expect(scale.opacityFor(2)).toBeCloseTo(0.5, 10);
    expect(scale.opacityFor(-2)).toBeCloseTo(0.5, 10);
  });

  test('an all-zero matrix paints nothing rather than inventing contrast', () => {
    const scale = scaleOf([[0, 0]]);
    expect(scale.limit).toBe(0);
    expect(scale.opacityFor(0)).toBe(0);
  });

  test('the diverging scale is null when no cell holds a reading', () => {
    expect(divergingScale([[null, null]])).toBeNull();
    expect(divergingScale([])).toBeNull();
  });

  test('the extent ignores cells with no reading', () => {
    expect(heatmapExtent(mark.values)).toEqual({ min: 1, max: 6 });
    expect(heatmapExtent([[null, null]])).toBeNull();
    expect(heatmapExtent([])).toBeNull();
  });

  test('values are row-major: values[j][i] is y_labels[j] x x_labels[i]', () => {
    const box = plotBox();
    const cells = heatmapCells(mark, box);
    expect(cells).toHaveLength(6);
    const cell = cells.find((candidate) => candidate.row === 1 && candidate.col === 2);
    expect(cell?.value).toBe(6);
    // Row 1 sits below row 0: rows read top-down in the same order as y_labels.
    const rowZero = cells.find((candidate) => candidate.row === 0 && candidate.col === 2);
    expect(cell && rowZero && cell.y > rowZero.y).toBe(true);
    // Column 2 sits right of column 0.
    const colZero = cells.find((candidate) => candidate.row === 1 && candidate.col === 0);
    expect(cell && colZero && cell.x > colZero.x).toBe(true);
  });

  test('the grid fills the plot box exactly', () => {
    const box = plotBox();
    const cells = heatmapCells(mark, box);
    expect(cells[0].x).toBe(box.left);
    expect(cells[0].y).toBe(box.top);
    const last = cells[cells.length - 1];
    expect(last.x + last.width).toBeCloseTo(box.right);
    expect(last.y + last.height).toBeCloseTo(box.bottom);
  });

  test('a cell with no reading keeps its slot and carries a null value', () => {
    const missing = heatmapCells(mark, plotBox()).find(
      (cell) => cell.row === 1 && cell.col === 1
    );
    expect(missing?.value).toBeNull();
  });

  test('an empty heatmap has no cells', () => {
    expect(heatmapCells(heatmapMark([], [], []), plotBox())).toEqual([]);
  });
});

describe('bar geometry', () => {
  test('the domain always contains zero, so bars are read against a baseline', () => {
    expect(barDomain([10, 12, 11])).toEqual({ min: 0, max: 12 });
    expect(barDomain([-3, -1])).toEqual({ min: -3, max: 0 });
    expect(barDomain([])).toEqual({ min: 0, max: 1 });
    expect(barDomain([null, null])).toEqual({ min: 0, max: 1 });
  });

  test('bars grow from the baseline in both directions', () => {
    const box = plotBox();
    const bars = barGeometry(barsMark(['up', 'down'], [4, -4]), box);
    const scaleY = linearScale(-4, 4, box.bottom, box.top);
    const baseline = scaleY(0);
    expect(bars[0].y).toBeCloseTo(scaleY(4));
    expect(bars[0].y + bars[0].height).toBeCloseTo(baseline);
    expect(bars[1].y).toBeCloseTo(baseline);
    expect(bars[1].y + bars[1].height).toBeCloseTo(scaleY(-4));
  });

  test('a category with no value keeps its slot and draws nothing', () => {
    const bars = barGeometry(barsMark(['a', 'b', 'c'], [1, null, 3]), plotBox());
    expect(bars).toHaveLength(3);
    expect(bars[1].value).toBeNull();
    expect(bars[1].height).toBe(0);
    expect(bars[1].category).toBe('b');
  });

  test('bars stay inside the plot box and do not overlap', () => {
    const box = plotBox();
    const bars = barGeometry(barsMark(['a', 'b', 'c'], [1, 2, 3]), box);
    expect(bars[0].x).toBeGreaterThanOrEqual(box.left);
    expect(bars[2].x + bars[2].width).toBeLessThanOrEqual(box.right);
    expect(bars[0].x + bars[0].width).toBeLessThan(bars[1].x);
  });

  test('an empty bars mark has no geometry', () => {
    expect(barGeometry(barsMark([], []), plotBox())).toEqual([]);
  });
});

describe('annotations', () => {
  test('shownPointCount counts what each mark kind actually draws', () => {
    expect(shownPointCount(linesMark([series('a', [[0, 1]]), series('b', [[0, 1], [1, 2]])]))).toBe(
      3
    );
    expect(shownPointCount(barsMark(['a', 'b'], [1, 2]))).toBe(2);
    expect(shownPointCount(heatmapMark(['x0', 'x1'], ['y0'], [[1, 2]]))).toBe(2);
  });

  test('an undecimated mark says nothing', () => {
    expect(decimationAnnotations(linesMark([series('a', [[0, 1]])]))).toEqual([]);
  });

  test('a thinned mark names how many of its source points are shown', () => {
    const mark = linesMark([series('bpm3', [[0, 1], [1, 2]], { decimated: true, source_points: 500 })], {
      decimated: true,
      source_points: 500,
    });
    expect(decimationAnnotations(mark)).toEqual(['showing 2 of 500 points']);
  });

  test('a multi-series mark also names each thinned series', () => {
    const thinned = series('bpm3', [[0, 1], [1, 2]], { decimated: true, source_points: 400 });
    const whole = series('bpm4', [[0, 1]]);
    const mark = linesMark([thinned, whole], { decimated: true, source_points: 401 });
    expect(decimationAnnotations(mark)).toEqual([
      'showing 3 of 401 points',
      'bpm3: showing 2 of 400 points',
    ]);
  });

  test('a heatmap and a bars mark report their own totals', () => {
    expect(
      decimationAnnotations(
        heatmapMark(['x0', 'x1'], ['y0'], [[1, 2]], { decimated: true, source_points: 90 })
      )
    ).toEqual(['showing 2 of 90 points']);
    expect(
      decimationAnnotations(barsMark(['a'], [1], { decimated: true, source_points: 12 }))
    ).toEqual(['showing 1 of 12 points']);
  });

  test('gaps are disclosed in words, not only as a break in the line', () => {
    expect(gapAnnotations(linesMark([series('a', [[0, 1], [1, null]])]))).toEqual([
      '1 sample had no reading (drawn as gaps)',
    ]);
    expect(
      gapAnnotations(linesMark([series('a', [[0, null], [1, null]])]))
    ).toEqual(['2 samples had no reading (drawn as gaps)']);
    expect(gapAnnotations(linesMark([series('a', [[0, 1]])]))).toEqual([]);
    expect(gapAnnotations(barsMark(['a'], [null]))).toEqual([]);
  });

  test("the panel's own annotations come first, then what the drawing discloses", () => {
    const mark = linesMark([series('a', [[0, 1], [1, null]], { decimated: true, source_points: 9 })], {
      decimated: true,
      source_points: 9,
    });
    expect(panelAnnotations(panel(mark, { annotations: ['fit failed on 2 correctors'] }))).toEqual([
      'fit failed on 2 correctors',
      'showing 2 of 9 points',
      '1 sample had no reading (drawn as gaps)',
    ]);
  });
});

describe('text helpers', () => {
  test('axisTitle joins a label with its units and tolerates either missing', () => {
    expect(axisTitle('orbit', 'mm')).toBe('orbit (mm)');
    expect(axisTitle('orbit', null)).toBe('orbit');
    expect(axisTitle('', 'mm')).toBe('mm');
    expect(axisTitle('', null)).toBe('');
  });

  test('the figure note reports source, partiality and reason', () => {
    expect(figureNoteText(figure([]))).toBe('stored data');
    expect(figureNoteText(figure([], { source: 'live', partial: true }))).toBe(
      'live data · still filling in'
    );
    expect(figureNoteText(figure([], { reason: 'render_failed' }))).toBe(
      'stored data · render failed'
    );
    // The bridge deliberately leaves the reason vocabulary open, so an
    // unforeseen code still has to read as words.
    expect(figureNoteText(figure([], { reason: 'some_new_reason' }))).toBe(
      'stored data · some new reason'
    );
  });

  test('seriesColor wraps the palette and falls back when it is empty', () => {
    expect(seriesColor(['a', 'b'], 0, 'fb')).toBe('a');
    expect(seriesColor(['a', 'b'], 3, 'fb')).toBe('b');
    expect(seriesColor([], 0, 'fb')).toBe('fb');
  });
});

// ---------------------------------------------------------------------------
// Drawing — against the real theme bridges
// ---------------------------------------------------------------------------

const LIGHT = {
  '--bg-primary': '#ffffff',
  '--chart-paper-bg': '#ffffff',
  '--chart-plot-bg': '#fafafa',
  '--chart-grid': '#dddddd',
  '--chart-axis-text': '#222222',
  '--chart-series-1': '#1f77b4',
  '--chart-series-2': '#ff7f0e',
};

const DARK = {
  '--bg-primary': '#101014',
  '--chart-paper-bg': '#101014',
  '--chart-plot-bg': '#18181c',
  '--chart-grid': '#3a3a42',
  '--chart-axis-text': '#e6e6ea',
  '--chart-series-1': '#7fc7ff',
  '--chart-series-2': '#ffb570',
};

const THEME_STYLE_ID = 'figure-renderer-theme-fixture';

/**
 * Drive the real theme-manager computed-style bridges by defining the
 * `--chart-*` custom properties the way a theme stylesheet does. Swapping the
 * rule is a theme flip as far as `chartTheme()`/`chartSeries()` can tell.
 *
 * @param {Record<string, string>} tokens
 */
function applyTheme(tokens) {
  const existing = document.getElementById(THEME_STYLE_ID);
  const style = existing ?? document.createElement('style');
  style.id = THEME_STYLE_ID;
  style.textContent = `:root { ${Object.entries(tokens)
    .map(([name, value]) => `${name}: ${value};`)
    .join(' ')} }`;
  if (!existing) document.head.appendChild(style);
}

function mount() {
  document.body.replaceChildren();
  const panels = document.createElement('div');
  const note = document.createElement('p');
  note.hidden = true;
  document.body.append(note, panels);
  return { panels, note };
}

/**
 * @param {Element} root
 * @param {string} selector
 * @returns {string[]}
 */
const textsOf = (root, selector) =>
  Array.from(root.querySelectorAll(selector)).map((node) => node.textContent ?? '');

/**
 * The figure's one section disclosure, typed as one.
 * @param {Element} root
 * @returns {HTMLDetailsElement}
 */
function disclosure(root) {
  const found = root.querySelector('details.figure-section');
  if (found === null) throw new Error('no section disclosure rendered');
  return /** @type {HTMLDetailsElement} */ (found);
}

/** @param {Element|undefined} target */
function click(target) {
  if (!target) throw new Error('nothing to click');
  target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

describe('renderFigure', () => {
  beforeEach(() => {
    applyTheme(LIGHT);
  });

  afterEach(() => {
    document.body.replaceChildren();
    document.getElementById(THEME_STYLE_ID)?.remove();
    vi.restoreAllMocks();
  });

  test('a lines panel draws one polyline per series in the theme palette', () => {
    const elements = mount();
    renderFigure(
      elements,
      figure([
        panel(
          linesMark([
            series('bpm3', [[0, 1], [1, 2], [2, 3]]),
            series('bpm4', [[0, 3], [1, 2], [2, 1]]),
          ]),
          { title: 'Orbit response', x_label: 'corrector', x_units: 'A', y_label: 'orbit', y_units: 'mm' }
        ),
      ])
    );

    const section = elements.panels.querySelector('section.figure-panel');
    expect(section?.getAttribute('data-mark')).toBe('lines');
    expect(section?.querySelector('.figure-panel-title')?.textContent).toBe('Orbit response');

    const lines = elements.panels.querySelectorAll('polyline.figure-line');
    expect(lines).toHaveLength(2);
    expect(lines[0].getAttribute('stroke')).toBe(LIGHT['--chart-series-1']);
    expect(lines[1].getAttribute('stroke')).toBe(LIGHT['--chart-series-2']);
    expect(elements.panels.querySelector('.figure-bg')?.getAttribute('fill')).toBe(
      LIGHT['--chart-plot-bg']
    );

    // Axis text comes from the figure's own label/units fields.
    expect(textsOf(elements.panels, 'text.figure-axis-title')).toEqual([
      'corrector (A)',
      'orbit (mm)',
    ]);
    expect(textsOf(elements.panels, 'ul.figure-legend li')).toEqual(['bpm3', 'bpm4']);
  });

  test('the same lines panel re-renders in the dark palette after a theme flip', () => {
    const elements = mount();
    const fig = figure([panel(linesMark([series('bpm3', [[0, 1], [1, 2]])]))]);

    renderFigure(elements, fig);
    expect(elements.panels.querySelector('polyline.figure-line')?.getAttribute('stroke')).toBe(
      LIGHT['--chart-series-1']
    );

    applyTheme(DARK);
    renderFigure(elements, fig);

    const line = elements.panels.querySelector('polyline.figure-line');
    expect(line?.getAttribute('stroke')).toBe(DARK['--chart-series-1']);
    expect(line?.getAttribute('stroke')).not.toBe(LIGHT['--chart-series-1']);
    expect(elements.panels.querySelector('.figure-bg')?.getAttribute('fill')).toBe(
      DARK['--chart-plot-bg']
    );
    expect(elements.panels.querySelector('text.figure-tick')?.getAttribute('fill')).toBe(
      DARK['--chart-axis-text']
    );
    // One render replaces the previous one rather than stacking on it.
    expect(elements.panels.querySelectorAll('polyline.figure-line')).toHaveLength(1);
  });

  test('a heatmap paints one rect per reading and labels both axes, in either theme', () => {
    for (const [name, tokens] of /** @type {Array<[string, Record<string, string>]>} */ ([
      ['light', LIGHT],
      ['dark', DARK],
    ])) {
      applyTheme(tokens);
      const elements = mount();
      renderFigure(
        elements,
        figure([
          panel(
            heatmapMark(
              ['ch1', 'ch2', 'ch3'],
              ['bpm1', 'bpm2'],
              [
                [1, 2, 3],
                [4, null, 6],
              ],
              { value_label: 'response', value_units: 'mm/A' }
            ),
            { title: 'Response matrix', x_label: 'corrector', y_label: 'monitor' }
          ),
        ])
      );

      // Six slots, six rects: five readings plus the one with no reading,
      // which is HATCHED rather than skipped — left unpainted it would be
      // plot background, and background is what a zero reading now looks like.
      const cells = elements.panels.querySelectorAll('rect.figure-cell');
      expect(cells, name).toHaveLength(6);
      const missing = elements.panels.querySelectorAll('rect.figure-cell-missing');
      expect(missing, name).toHaveLength(1);
      expect(missing[0].getAttribute('fill'), name).toMatch(/^url\(#figure-hatch-/);

      // All readings here are positive, so all take the positive hue.
      expect(cells[0].getAttribute('fill'), name).toBe(tokens['--chart-series-1']);

      // Opacity carries the magnitude: the smallest reading is the faintest.
      const opacities = Array.from(cells)
        .filter((cell) => cell.hasAttribute('fill-opacity'))
        .map((cell) => Number(cell.getAttribute('fill-opacity')));
      expect(opacities[0], name).toBeLessThan(opacities[opacities.length - 1]);

      // Both label lists are rendered, and only from the mark's own labels.
      const ticks = textsOf(elements.panels, 'text.figure-tick');
      expect(ticks, name).toEqual(expect.arrayContaining(['ch1', 'ch2', 'ch3', 'bpm1', 'bpm2']));

      // A heatmap gets a colorbar carrying its numbers, not a flat legend chip.
      expect(elements.panels.querySelectorAll('ul.figure-legend'), name).toHaveLength(0);
      expect(
        elements.panels.querySelector('.figure-colorbar-label')?.textContent,
        name
      ).toBe('response (mm/A)');
      expect(textsOf(elements.panels, '.figure-colorbar-tick'), name).toEqual(['-6', '6', '0']);
      expect(
        elements.panels.querySelector('section.figure-panel')?.getAttribute('data-mark'),
        name
      ).toBe('heatmap');
    }
  });

  test('a negative reading takes the other hue, and zero is left as background', () => {
    applyTheme(LIGHT);
    const elements = mount();
    renderFigure(
      elements,
      figure([
        panel(
          heatmapMark(
            ['c1', 'c2'],
            ['b1'],
            [[-5, 0]],
            { value_label: 'slope', value_units: 'mm/A' }
          )
        ),
      ])
    );

    const cells = elements.panels.querySelectorAll('rect.figure-cell');
    // Negative → the second palette slot, at full weight (it IS the limit).
    expect(cells[0].getAttribute('fill')).toBe(LIGHT['--chart-series-2']);
    expect(Number(cells[0].getAttribute('fill-opacity'))).toBe(1);
    // Zero → painted at opacity 0, which is the plot background.
    expect(Number(cells[1].getAttribute('fill-opacity'))).toBe(0);
  });

  test('a cell with no reading says so on hover, rather than reading as a zero', () => {
    const elements = mount();
    renderFigure(elements, figure([panel(heatmapMark(['x0'], ['y0'], [[null]]))]));
    // Nothing is plottable, so the panel says that rather than drawing a grid.
    expect(elements.panels.querySelector('.figure-empty')).not.toBeNull();

    const mixed = mount();
    renderFigure(mixed, figure([panel(heatmapMark(['x0', 'x1'], ['y0'], [[1, null]]))]));
    expect(
      mixed.panels.querySelector('rect.figure-cell-missing title')?.textContent
    ).toBe('No reading');
  });

  test('a series picker filters what is drawn and remembers the choice across a redraw', () => {
    const elements = mount();
    const selection = {};
    const fig = figure([
      panel(
        linesMark([
          series('c1', [[0, 1]]),
          series('c2', [[0, 2]]),
          series('c3', [[0, 3]]),
          series('c4', [[0, 4]]),
        ]),
        { title: 'Response by BPM', series_picker: true }
      ),
    ]);

    renderFigure(elements, fig, { selection });

    // Default: the first three of four, so the panel opens on something.
    const chips = elements.panels.querySelectorAll('button.figure-chip');
    expect(Array.from(chips).map((chip) => chip.textContent)).toEqual(['c1', 'c2', 'c3', 'c4']);
    expect(textsOf(elements.panels, 'ul.figure-legend li')).toEqual(['c1', 'c2', 'c3']);

    // Turning c4 on redraws with four lines...
    chips[3].dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(textsOf(elements.panels, 'ul.figure-legend li')).toEqual(['c1', 'c2', 'c3', 'c4']);

    // ...and the choice survives the ~1s poll redrawing the whole subtree,
    // which is the bug this state exists to prevent.
    renderFigure(elements, fig, { selection });
    expect(textsOf(elements.panels, 'ul.figure-legend li')).toEqual(['c1', 'c2', 'c3', 'c4']);
  });

  test('a series picker refuses to empty the plot', () => {
    const elements = mount();
    const selection = { series: { P: ['c1'] } };
    const fig = figure([
      panel(linesMark([series('c1', [[0, 1]]), series('c2', [[0, 2]])]), {
        title: 'P',
        series_picker: true,
      }),
    ]);
    renderFigure(elements, fig, { selection });
    expect(textsOf(elements.panels, 'ul.figure-legend li')).toEqual(['c1']);

    // Clicking the only selected chip would leave nothing drawn.
    click(elements.panels.querySelectorAll('button.figure-chip')[0]);
    expect(textsOf(elements.panels, 'ul.figure-legend li')).toEqual(['c1']);
  });

  test('sectioned panels render closed, below the inline ones, one at a time', () => {
    const elements = mount();
    const selection = {};
    const fig = figure([
      panel(heatmapMark(['c1'], ['b1'], [[1]]), { title: 'Response matrix' }),
      panel(linesMark([series('b1', [[0, 1]])]), { title: 'c1 sweep', section: 'Sweeps' }),
      panel(linesMark([series('b1', [[0, 2]])]), { title: 'c2 sweep', section: 'Sweeps' }),
    ]);

    renderFigure(elements, fig, { selection });

    // The inline finding comes first; the evidence is a closed disclosure.
    const details = disclosure(elements.panels);
    expect(details.open).toBe(false);
    const inline = /** @type {Element} */ (
      elements.panels.querySelector('section.figure-panel')
    );
    expect(details.compareDocumentPosition(inline)).toBe(Node.DOCUMENT_POSITION_PRECEDING);
    expect(elements.panels.querySelector('.figure-section-count')?.textContent).toBe('2 panels');

    // One panel at a time, chosen by the picker.
    expect(textsOf(details, 'h4.figure-panel-title')).toEqual(['c1 sweep']);

    click(details.querySelectorAll('button.figure-chip')[1]);
    expect(textsOf(elements.panels, 'h4.figure-panel-title')).toContain('c2 sweep');
    expect(textsOf(elements.panels, 'h4.figure-panel-title')).not.toContain('c1 sweep');
  });

  test('an opened section stays open across a redraw', () => {
    const elements = mount();
    const selection = {};
    const fig = figure([
      panel(heatmapMark(['c1'], ['b1'], [[1]]), { title: 'Response matrix' }),
      panel(linesMark([series('b1', [[0, 1]])]), { title: 'c1 sweep', section: 'Sweeps' }),
    ]);

    renderFigure(elements, fig, { selection });
    const details = disclosure(elements.panels);
    expect(details.open).toBe(false);

    // happy-dom does not fire `toggle` for a property set, so drive the event
    // the way a real disclosure click does.
    details.open = true;
    details.dispatchEvent(new Event('toggle'));

    renderFigure(elements, fig, { selection });
    expect(disclosure(elements.panels).open).toBe(true);
  });

  test('a figure with no sections renders exactly as it did before sections existed', () => {
    const elements = mount();
    renderFigure(
      elements,
      figure([
        panel(linesMark([series('a', [[0, 1]])]), { title: 'one' }),
        panel(linesMark([series('b', [[0, 2]])]), { title: 'two' }),
      ])
    );

    expect(elements.panels.querySelectorAll('details.figure-section')).toHaveLength(0);
    expect(elements.panels.querySelectorAll('.figure-picker')).toHaveLength(0);
    expect(textsOf(elements.panels, 'h4.figure-panel-title')).toEqual(['one', 'two']);
  });

  test('heatmap cells carry their value as hover text', () => {
    const elements = mount();
    renderFigure(elements, figure([panel(heatmapMark(['x0'], ['y0'], [[2.5]]))]));
    expect(elements.panels.querySelector('rect.figure-cell title')?.textContent).toBe('2.5');
  });

  test('bars draw one rect per value and label the categories, in either theme', () => {
    for (const [name, tokens] of /** @type {Array<[string, Record<string, string>]>} */ ([
      ['light', LIGHT],
      ['dark', DARK],
    ])) {
      applyTheme(tokens);
      const elements = mount();
      renderFigure(
        elements,
        figure([
          panel(barsMark(['ch1', 'ch2', 'ch3'], [1.5, null, -0.5], { label: 'residual' }), {
            title: 'Fit residual',
            y_label: 'residual',
            y_units: 'mm',
          }),
        ])
      );

      const bars = elements.panels.querySelectorAll('rect.figure-bar');
      expect(bars, name).toHaveLength(2); // the category with no value draws nothing
      expect(bars[0].getAttribute('fill'), name).toBe(tokens['--chart-series-1']);
      const ticks = textsOf(elements.panels, 'text.figure-tick');
      expect(ticks, name).toEqual(expect.arrayContaining(['ch1', 'ch2', 'ch3']));
      expect(textsOf(elements.panels, 'ul.figure-legend li'), name).toEqual(['residual']);
    }
  });

  test('a gapped series is drawn as separate runs and says so', () => {
    const elements = mount();
    renderFigure(
      elements,
      figure([
        panel(
          linesMark([
            series('bpm3', [
              [0, 1],
              [1, 2],
              [2, null],
              [3, 4],
            ]),
          ])
        ),
      ])
    );

    const lines = elements.panels.querySelectorAll('polyline.figure-line');
    expect(lines).toHaveLength(1);
    expect(lines[0].getAttribute('points')?.split(' ')).toHaveLength(2);
    // The reading after the gap is a lone point, so it is drawn as a dot
    // rather than vanishing.
    expect(elements.panels.querySelectorAll('circle.figure-point')).toHaveLength(1);
    expect(textsOf(elements.panels, 'ul.figure-annotations li')).toEqual([
      '1 sample had no reading (drawn as gaps)',
    ]);
  });

  test("a decimated panel renders both the figure's annotations and the decimation note", () => {
    const elements = mount();
    renderFigure(
      elements,
      figure([
        panel(
          linesMark([series('bpm3', [[0, 1], [1, 2]], { decimated: true, source_points: 20000 })], {
            decimated: true,
            source_points: 20000,
          }),
          { annotations: ['fit converged in 3 iterations'] }
        ),
      ])
    );
    expect(textsOf(elements.panels, 'ul.figure-annotations li')).toEqual([
      'fit converged in 3 iterations',
      'showing 2 of 20000 points',
    ]);
  });

  test('a panel with no plottable values says so and still carries its annotations', () => {
    const elements = mount();
    renderFigure(
      elements,
      figure([
        panel(linesMark([series('bpm3', [[0, null], [1, null]])]), {
          annotations: ['readback offline'],
        }),
      ])
    );
    expect(elements.panels.querySelector('svg.figure-plot')).toBeNull();
    expect(elements.panels.querySelector('.figure-empty')?.textContent).toBe(
      'No plottable values in this panel yet.'
    );
    expect(textsOf(elements.panels, 'ul.figure-annotations li')).toEqual([
      'readback offline',
      '2 samples had no reading (drawn as gaps)',
    ]);
  });

  test('the figure note goes to the caller’s note element', () => {
    const elements = mount();
    renderFigure(elements, figure([], { source: 'live', partial: true }));
    expect(elements.note.textContent).toBe('live data · still filling in');
    expect(elements.note.hidden).toBe(false);
    expect(elements.panels.querySelector('.figure-empty')?.textContent).toBe(
      'No figure for this run.'
    );
  });

  test('without a note element the note is written into the panel container instead', () => {
    const elements = mount();
    renderFigure({ panels: elements.panels }, figure([], { reason: 'no_render' }));
    expect(elements.panels.querySelector('.figure-note')?.textContent).toBe(
      'stored data · no render'
    );
  });

  test('multiple panels each get their own section, in figure order', () => {
    const elements = mount();
    renderFigure(
      elements,
      figure([
        panel(linesMark([series('a', [[0, 1], [1, 2]])]), { title: 'first' }),
        panel(barsMark(['x'], [1]), { title: 'second' }),
      ])
    );
    expect(textsOf(elements.panels, 'section.figure-panel .figure-panel-title')).toEqual([
      'first',
      'second',
    ]);
    expect(
      Array.from(elements.panels.querySelectorAll('section.figure-panel')).map((section) =>
        section.getAttribute('data-mark')
      )
    ).toEqual(['lines', 'bars']);
  });

  test('figure text is set as text, never parsed as markup', () => {
    const elements = mount();
    const hostile = '<img src=x onerror="alert(1)">';
    renderFigure(
      elements,
      figure([
        panel(linesMark([series(hostile, [[0, 1], [1, 2]])]), {
          title: hostile,
          annotations: [hostile],
        }),
      ])
    );
    expect(elements.panels.querySelector('img')).toBeNull();
    expect(elements.panels.querySelector('.figure-panel-title')?.textContent).toBe(hostile);
    expect(textsOf(elements.panels, 'ul.figure-legend li')).toEqual([hostile]);
    expect(textsOf(elements.panels, 'ul.figure-annotations li')).toEqual([hostile]);
  });

  test('rendering reads real theme tokens — the bridge never logs an untrusted read', () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const elements = mount();
    renderFigure(elements, figure([panel(linesMark([series('a', [[0, 1], [1, 2]])]))]));
    expect(errors).not.toHaveBeenCalled();
  });
});
