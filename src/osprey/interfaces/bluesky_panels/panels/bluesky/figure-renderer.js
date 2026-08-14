// @ts-check
/**
 * BLUESKY panel — the figure renderer.
 *
 * Draws a bridge `Figure` (see `services/bluesky_bridge/figure.py`) as themed
 * inline SVG. The module is split in two halves:
 *
 *  - **Layout** — pure functions (`plotBox`, `linearScale`, `axisTicks`,
 *    `splitSegments`, `heatmapCells`, `barGeometry`, …). No DOM, no theme, no
 *    globals; given plain figure objects they return numbers and strings, and
 *    they are what the unit tests exercise directly.
 *  - **Drawing** — `renderFigure(elements, figure, options)`, which turns that
 *    geometry into nodes. Every node is built with
 *    `createElement`/`createElementNS` and every string lands via `textContent`
 *    or `setAttribute`; nothing in this file assigns `innerHTML`.
 *
 * The reader's own choices about a figure — which series a panel draws, which
 * panel a section shows, whether a section is open — live in
 * `figure-controls.js`, along with the chrome that changes them. This file
 * draws what the bridge sent; that one owns what the reader did with it. Pass
 * `options.selection` to give those choices memory across a redraw.
 *
 * Colors are read from the theme-manager computed-style bridges *inside*
 * `renderFigure`, never at module or closure scope, so a caller that re-renders
 * from `subscribe()` gets the new theme's colors. Nothing here caches a
 * resolved token value.
 *
 * Honesty rules the drawing obeys, because the figure spec is built around
 * them:
 *  - A `null` y is a **gap** — a sample the run took without a reading. The
 *    line breaks there; it is never interpolated across and never quietly
 *    dropped (the panel annotates how many gaps it drew).
 *  - A decimated mark annotates how many of its source points are shown, so a
 *    thinned trace never passes for the full record.
 *  - Labels, units and annotations are rendered from the figure's own fields.
 *    This file invents no axis text.
 */

import { chartSeries, chartTheme } from '/design-system/js/theme-manager.js';

import {
  chipPicker,
  colorbarElement,
  groupPanels,
  resolveSeriesSelection,
  sectionElement,
} from './figure-controls.js';

/** @typedef {import('./figure-controls.js').FigureSelection} FigureSelection */

/** @typedef {{x: number, y: number|null}} FigurePoint */
/** @typedef {{label: string, points: FigurePoint[], decimated: boolean, source_points: number}} FigureSeries */
/** @typedef {{kind: 'lines', series: FigureSeries[], decimated: boolean, source_points: number}} LinesMark */
/** @typedef {{kind: 'heatmap', x_labels: string[], y_labels: string[], values: Array<Array<number|null>>, value_label: string, value_units: string|null, decimated: boolean, source_points: number}} HeatmapMark */
/** @typedef {{kind: 'bars', label: string, categories: string[], values: Array<number|null>, decimated: boolean, source_points: number}} BarsMark */
/** @typedef {LinesMark|HeatmapMark|BarsMark} FigureMark */
/** @typedef {{title: string, x_label: string, y_label: string, x_units: string|null, y_units: string|null, annotations: string[], mark: FigureMark, section?: string|null, series_picker?: boolean}} FigurePanel */
/** @typedef {{panels: FigurePanel[], partial: boolean, source: 'live'|'tiled', reason: string|null}} Figure */


/** @typedef {{left: number, top: number, width: number, height: number, right: number, bottom: number}} PlotBox */
/** @typedef {(value: number) => number} Scale */
/** @typedef {{pos: number, label: string}} AxisTick */

const SVG_NS = 'http://www.w3.org/2000/svg';

export const FIGURE_WIDTH = 640;
export const FIGURE_HEIGHT = 240;

/** Room for the y tick labels on the left and the x tick labels + title below. */
export const FIGURE_MARGIN = Object.freeze({ top: 14, right: 18, bottom: 44, left: 60 });

/** Most category labels drawn on one axis before `labelStride` starts thinning. */
const MAX_AXIS_LABELS = 12;

// ---------------------------------------------------------------------------
// Layout — pure, DOM-free
// ---------------------------------------------------------------------------

/**
 * The inner plot rectangle, in the figure's own viewBox units.
 *
 * @param {number} [width]
 * @param {number} [height]
 * @param {{top: number, right: number, bottom: number, left: number}} [margin]
 * @returns {PlotBox}
 */
export function plotBox(width = FIGURE_WIDTH, height = FIGURE_HEIGHT, margin = FIGURE_MARGIN) {
  const innerWidth = Math.max(1, width - margin.left - margin.right);
  const innerHeight = Math.max(1, height - margin.top - margin.bottom);
  return {
    left: margin.left,
    top: margin.top,
    width: innerWidth,
    height: innerHeight,
    right: margin.left + innerWidth,
    bottom: margin.top + innerHeight,
  };
}

/**
 * A linear domain -> range mapping.
 *
 * A degenerate domain (a flat series, a single sample) maps everything to the
 * middle of the range rather than to an edge or a NaN — a constant reading is
 * drawn as a level line through the middle of the plot, which is what it is.
 *
 * @param {number} domainMin
 * @param {number} domainMax
 * @param {number} rangeMin
 * @param {number} rangeMax
 * @returns {Scale}
 */
export function linearScale(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin;
  if (!Number.isFinite(span) || span === 0) {
    const middle = (rangeMin + rangeMax) / 2;
    return () => middle;
  }
  return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

/**
 * Widen a domain so the extreme samples are not drawn on the frame.
 *
 * A flat domain is widened symmetrically — by 5% of the value, or by 0.5 when
 * the value is zero and there is no scale to take a fraction of.
 *
 * @param {number} min
 * @param {number} max
 * @param {number} [fraction]
 * @returns {[number, number]}
 */
export function padRange(min, max, fraction = 0.05) {
  if (min === max) {
    const pad = Math.abs(min) * fraction || 0.5;
    return [min - pad, max + pad];
  }
  const pad = (max - min) * fraction;
  return [min - pad, max + pad];
}

/**
 * The drawable extent of a lines mark's series, or `null` when there is
 * nothing to plot.
 *
 * Gap points count toward the **x** extent and not the y extent: the run
 * sampled that x, it just has no reading there, so the axis still covers it.
 * `null` comes back when no series holds a single finite y — an all-gap panel
 * has no y domain to invent, and the caller says so in words instead.
 *
 * @param {FigureSeries[]} series
 * @returns {{xMin: number, xMax: number, yMin: number, yMax: number}|null}
 */
export function seriesExtent(series) {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const line of series) {
    for (const point of line.points) {
      if (!Number.isFinite(point.x)) continue;
      if (point.x < xMin) xMin = point.x;
      if (point.x > xMax) xMax = point.x;
      const y = point.y;
      if (y === null || y === undefined || !Number.isFinite(y)) continue;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  }
  if (yMin === Infinity || xMin === Infinity) return null;
  return { xMin, xMax, yMin, yMax };
}

/**
 * Split a series' points into gap-free runs.
 *
 * Every `null` (or non-finite) y ends the run in progress; the next reading
 * starts a new one. Drawing one polyline per run is what makes a dropout a
 * visible break instead of a straight line across data the run never took.
 *
 * @param {FigurePoint[]} points
 * @returns {FigurePoint[][]}
 */
export function splitSegments(points) {
  /** @type {FigurePoint[][]} */
  const segments = [];
  /** @type {FigurePoint[]} */
  let current = [];
  for (const point of points) {
    const y = point.y;
    if (y === null || y === undefined || !Number.isFinite(y) || !Number.isFinite(point.x)) {
      if (current.length > 0) segments.push(current);
      current = [];
      continue;
    }
    current.push(point);
  }
  if (current.length > 0) segments.push(current);
  return segments;
}

/**
 * How many samples in a lines mark have no reading.
 *
 * @param {FigureSeries[]} series
 * @returns {number}
 */
export function gapCount(series) {
  let gaps = 0;
  for (const line of series) {
    for (const point of line.points) {
      const y = point.y;
      if (y === null || y === undefined || !Number.isFinite(y)) gaps += 1;
    }
  }
  return gaps;
}

/**
 * An SVG `points` attribute for one gap-free run.
 *
 * @param {FigurePoint[]} segment
 * @param {Scale} scaleX
 * @param {Scale} scaleY
 * @returns {string}
 */
export function polylinePoints(segment, scaleX, scaleY) {
  return segment
    .map((point) => `${scaleX(point.x).toFixed(1)},${scaleY(Number(point.y)).toFixed(1)}`)
    .join(' ');
}

/**
 * A tick value formatted for an axis label. Wide-magnitude values go to
 * exponential rather than to a run of zeros or a long decimal tail.
 *
 * @param {number|null} value
 * @returns {string}
 */
export function formatTickValue(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || magnitude < 1e-3) return value.toExponential(1);
  return String(Number(value.toPrecision(4)));
}

/**
 * Evenly spaced ticks across a numeric domain, positioned in range units.
 *
 * @param {number} domainMin
 * @param {number} domainMax
 * @param {Scale} scale Maps a domain value to its coordinate.
 * @param {number} [count]
 * @returns {AxisTick[]}
 */
export function axisTicks(domainMin, domainMax, scale, count = 5) {
  const steps = Math.max(1, count - 1);
  if (domainMin === domainMax) {
    return [{ pos: scale(domainMin), label: formatTickValue(domainMin) }];
  }
  /** @type {AxisTick[]} */
  const ticks = [];
  for (let index = 0; index <= steps; index++) {
    const value = domainMin + ((domainMax - domainMin) * index) / steps;
    ticks.push({ pos: scale(value), label: formatTickValue(value) });
  }
  return ticks;
}

/**
 * Draw every n-th category label so a long axis stays readable. Always at
 * least 1, so a caller can index with it unconditionally.
 *
 * @param {number} count
 * @param {number} [max]
 * @returns {number}
 */
export function labelStride(count, max = MAX_AXIS_LABELS) {
  if (count <= max) return 1;
  return Math.ceil(count / max);
}

/**
 * Category tick positions at the center of each slot.
 *
 * @param {string[]} labels
 * @param {number} start Coordinate of the first slot's leading edge.
 * @param {number} extent Total length covered by all slots.
 * @returns {AxisTick[]}
 */
export function categoryTicks(labels, start, extent) {
  const count = labels.length;
  if (count === 0) return [];
  const slot = extent / count;
  const stride = labelStride(count);
  /** @type {AxisTick[]} */
  const ticks = [];
  for (let index = 0; index < count; index += stride) {
    ticks.push({ pos: start + slot * (index + 0.5), label: labels[index] });
  }
  return ticks;
}

/**
 * The grid shape of a heatmap. `values` is row-major (`values[j][i]` is
 * `y_labels[j]` x `x_labels[i]`), and the labels are the authority on the
 * shape — but a mark that ships values without labels still gets drawn.
 *
 * @param {HeatmapMark} mark
 * @returns {{rows: number, cols: number}}
 */
export function heatmapShape(mark) {
  const rows = Math.max(mark.y_labels.length, mark.values.length);
  let cols = mark.x_labels.length;
  for (const row of mark.values) cols = Math.max(cols, row.length);
  return { rows, cols };
}

/**
 * The value extent of a heatmap, ignoring cells with no reading. `null` when
 * no cell holds one.
 *
 * @param {Array<Array<number|null>>} values
 * @returns {{min: number, max: number}|null}
 */
export function heatmapExtent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const row of values) {
    for (const value of row) {
      if (value === null || value === undefined || !Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  if (min === Infinity) return null;
  return { min, max };
}

/**
 * The signed colour scale for a heatmap, anchored at zero.
 *
 * A response matrix is signed and roughly zero-centred: a corrector kicks the
 * beam and a BPM downstream reads positive or negative by phase advance. On a
 * sequential ramp anchored to `[min, max]` that structure is destroyed twice
 * over -- zero lands mid-ramp, so an unresponsive BPM paints like a real
 * response, and the most negative cell paints faintest, so the strongest
 * negative response is indistinguishable from no measurement at all.
 *
 * So: one hue each side of zero, and a magnitude scaled against a SYMMETRIC
 * `limit` of `max|value|` rather than the raw extent. Zero maps to opacity 0 --
 * the plot background, painted by nothing -- which is what makes a dead row
 * read as dead. The symmetric limit is also what makes two runs comparable: a
 * ramp fitted to each run's own extent means a different thing every time.
 *
 * `null` when no cell holds a reading.
 *
 * @param {Array<Array<number|null>>} values
 * @returns {{limit: number, opacityFor: (value: number) => number, isNegative: (value: number) => boolean}|null}
 */
export function divergingScale(values) {
  const extent = heatmapExtent(values);
  if (extent === null) return null;

  const limit = Math.max(Math.abs(extent.min), Math.abs(extent.max));
  return {
    limit,
    // A flat matrix (limit 0) is all zero, and all zero is all background:
    // inventing contrast for it would draw structure that is not there.
    opacityFor: (value) => (limit > 0 ? Math.min(1, Math.abs(value) / limit) : 0),
    isNegative: (value) => value < 0,
  };
}

/**
 * One rectangle per heatmap cell, row 0 at the top so the drawn grid reads in
 * the same order as `values` and `y_labels`. Cells with no reading are
 * returned with `value: null` — the caller leaves them unpainted rather than
 * coloring them as if a measurement existed.
 *
 * @param {HeatmapMark} mark
 * @param {PlotBox} box
 * @returns {Array<{x: number, y: number, width: number, height: number, value: number|null, row: number, col: number}>}
 */
export function heatmapCells(mark, box) {
  const { rows, cols } = heatmapShape(mark);
  if (rows === 0 || cols === 0) return [];
  const cellWidth = box.width / cols;
  const cellHeight = box.height / rows;
  /** @type {Array<{x: number, y: number, width: number, height: number, value: number|null, row: number, col: number}>} */
  const cells = [];
  for (let row = 0; row < rows; row++) {
    const values = mark.values[row] || [];
    for (let col = 0; col < cols; col++) {
      const raw = values[col];
      const value = raw === null || raw === undefined || !Number.isFinite(raw) ? null : raw;
      cells.push({
        x: box.left + col * cellWidth,
        y: box.top + row * cellHeight,
        width: cellWidth,
        height: cellHeight,
        value,
        row,
        col,
      });
    }
  }
  return cells;
}

/**
 * The y domain of a bars mark. Zero is always inside it, because a bar is read
 * against a baseline — clipping the domain to the values would make a set of
 * large, nearly equal values look like it starts from nothing.
 *
 * @param {Array<number|null>} values
 * @returns {{min: number, max: number}}
 */
export function barDomain(values) {
  let min = 0;
  let max = 0;
  for (const value of values) {
    if (value === null || value === undefined || !Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (min === 0 && max === 0) return { min: 0, max: 1 };
  return { min, max };
}

/**
 * One rectangle per bar, measured from the zero baseline. A category with no
 * value is returned with `value: null` and no height — the slot is kept (its
 * label still belongs on the axis) but nothing is drawn in it.
 *
 * @param {BarsMark} mark
 * @param {PlotBox} box
 * @returns {Array<{x: number, y: number, width: number, height: number, value: number|null, category: string, index: number}>}
 */
export function barGeometry(mark, box) {
  const count = Math.max(mark.values.length, mark.categories.length);
  if (count === 0) return [];
  const domain = barDomain(mark.values);
  const scaleY = linearScale(domain.min, domain.max, box.bottom, box.top);
  const baseline = scaleY(0);
  const slot = box.width / count;
  const width = slot * 0.7;
  /** @type {Array<{x: number, y: number, width: number, height: number, value: number|null, category: string, index: number}>} */
  const bars = [];
  for (let index = 0; index < count; index++) {
    const raw = mark.values[index];
    const value = raw === null || raw === undefined || !Number.isFinite(raw) ? null : raw;
    const x = box.left + slot * index + (slot - width) / 2;
    const category = mark.categories[index] ?? '';
    if (value === null) {
      bars.push({ x, y: baseline, width, height: 0, value: null, category, index });
      continue;
    }
    const top = scaleY(value);
    bars.push({
      x,
      y: Math.min(top, baseline),
      width,
      height: Math.abs(baseline - top),
      value,
      category,
      index,
    });
  }
  return bars;
}

/**
 * How many points a mark actually draws — the numerator of its decimation
 * annotation.
 *
 * @param {FigureMark} mark
 * @returns {number}
 */
export function shownPointCount(mark) {
  if (mark.kind === 'lines') {
    return mark.series.reduce((total, line) => total + line.points.length, 0);
  }
  if (mark.kind === 'bars') return mark.values.length;
  const { rows, cols } = heatmapShape(mark);
  return rows * cols;
}

/**
 * Annotations a thinned mark owes the reader.
 *
 * The aggregate line always comes first. A multi-series lines mark also names
 * each thinned series, since one busy detector can be decimated while its
 * neighbours are shown whole; a single-series mark would only repeat its own
 * numbers, so it does not.
 *
 * @param {FigureMark} mark
 * @returns {string[]}
 */
export function decimationAnnotations(mark) {
  /** @type {string[]} */
  const notes = [];
  if (mark.decimated) {
    notes.push(`showing ${shownPointCount(mark)} of ${mark.source_points} points`);
  }
  if (mark.kind === 'lines' && mark.series.length > 1) {
    for (const line of mark.series) {
      if (!line.decimated) continue;
      notes.push(`${line.label}: showing ${line.points.length} of ${line.source_points} points`);
    }
  }
  return notes;
}

/**
 * Annotations for the samples a mark could not draw. Only lines marks carry
 * per-sample gaps; a heatmap's missing cells are visible as holes in the grid.
 *
 * @param {FigureMark} mark
 * @returns {string[]}
 */
export function gapAnnotations(mark) {
  if (mark.kind !== 'lines') return [];
  const gaps = gapCount(mark.series);
  if (gaps === 0) return [];
  return [`${gaps} sample${gaps === 1 ? '' : 's'} had no reading (drawn as gaps)`];
}

/**
 * Every line of text under a panel: the plan's own annotations first, then
 * what the rendering itself has to disclose.
 *
 * @param {FigurePanel} panel
 * @returns {string[]}
 */
export function panelAnnotations(panel) {
  return [
    ...panel.annotations,
    ...decimationAnnotations(panel.mark),
    ...gapAnnotations(panel.mark),
  ];
}

/**
 * An axis title from a label and its units, either of which may be absent.
 *
 * @param {string} label
 * @param {string|null} units
 * @returns {string}
 */
export function axisTitle(label, units) {
  if (!label) return units || '';
  return units ? `${label} (${units})` : label;
}

/**
 * The figure-level note: where the data came from, whether it is still
 * arriving, and why it is the figure it is. Reason codes are the bridge's
 * (deliberately open) vocabulary, so they are spaced out rather than mapped —
 * an unforeseen reason still reads as words.
 *
 * @param {Figure} figure
 * @returns {string}
 */
export function figureNoteText(figure) {
  const parts = [figure.source === 'tiled' ? 'stored data' : 'live data'];
  if (figure.partial) parts.push('still filling in');
  if (figure.reason) parts.push(figure.reason.replace(/_/g, ' '));
  return parts.join(' · ');
}

/**
 * Pick a categorical color, wrapping around a palette of any length.
 *
 * @param {string[]} palette
 * @param {number} index
 * @param {string} fallback
 * @returns {string}
 */
export function seriesColor(palette, index, fallback) {
  if (palette.length === 0) return fallback;
  return palette[index % palette.length] || fallback;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

/** @typedef {{text: string, grid: string, plotBg: string, palette: string[]}} RenderColors */
/** @typedef {{label: string, color: string}} LegendEntry */
/** @typedef {{label: string, limit: number, negative: string, positive: string}} Colorbar */
/** @typedef {{plotted: boolean, legend: LegendEntry[], colorbar?: Colorbar}} MarkResult */

/** Distinguishes one render's `<pattern>` ids from another's on the same page. */
let patternSequence = 0;

/**
 * Define a diagonal hatch in `svg` and return the `fill` value referencing it.
 *
 * Ids must be unique per document, not per SVG: several figures share a page,
 * and a duplicate id would silently hand every one of them the first figure's
 * pattern -- drawn in the first figure's palette, which a theme flip would then
 * fail to update.
 *
 * @param {SVGElement} svg
 * @param {string} color
 * @returns {string}
 */
function hatchFill(svg, color) {
  const id = `figure-hatch-${(patternSequence += 1)}`;
  const pattern = svgNode('pattern', {
    id,
    width: 5,
    height: 5,
    patternUnits: 'userSpaceOnUse',
    patternTransform: 'rotate(45)',
  });
  pattern.appendChild(
    svgNode('line', {
      x1: 0,
      y1: 0,
      x2: 0,
      y2: 5,
      stroke: color,
      'stroke-width': 1.5,
      'stroke-opacity': 0.4,
    })
  );
  const defs = svgNode('defs', {});
  defs.appendChild(pattern);
  svg.appendChild(defs);
  return `url(#${id})`;
}

/**
 * @param {string} name
 * @param {Record<string, string|number>} attributes
 * @returns {SVGElement}
 */
function svgNode(name, attributes) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

/**
 * @param {number} x
 * @param {number} y
 * @param {string} text
 * @param {Record<string, string|number>} attributes
 * @returns {SVGElement}
 */
function svgText(x, y, text, attributes) {
  const node = svgNode('text', { x, y, ...attributes });
  node.textContent = text;
  return node;
}

/**
 * Horizontal gridlines with their y-axis labels, and the x-axis labels below
 * the plot. Vertical gridlines are deliberately absent — a category axis has
 * no continuum to rule, and the horizontal rules are what a reader traces a
 * value along.
 *
 * @param {SVGElement} svg
 * @param {PlotBox} box
 * @param {{x: AxisTick[], y: AxisTick[], grid?: boolean}} ticks
 * @param {RenderColors} colors
 */
function drawAxes(svg, box, ticks, colors) {
  const drawGrid = ticks.grid !== false;
  for (const tick of ticks.y) {
    if (drawGrid) {
      svg.appendChild(
        svgNode('line', {
          x1: box.left,
          x2: box.right,
          y1: tick.pos.toFixed(1),
          y2: tick.pos.toFixed(1),
          stroke: colors.grid,
          'stroke-width': 1,
          opacity: 0.5,
        })
      );
    }
    svg.appendChild(
      svgText(box.left - 8, tick.pos + 4, tick.label, {
        class: 'figure-tick',
        'text-anchor': 'end',
        fill: colors.text,
        'font-size': 10,
      })
    );
  }
  for (const tick of ticks.x) {
    svg.appendChild(
      svgText(tick.pos, box.bottom + 16, tick.label, {
        class: 'figure-tick',
        'text-anchor': 'middle',
        fill: colors.text,
        'font-size': 10,
      })
    );
  }
  svg.appendChild(
    svgNode('line', {
      x1: box.left,
      x2: box.right,
      y1: box.bottom,
      y2: box.bottom,
      stroke: colors.grid,
      'stroke-width': 1,
    })
  );
  svg.appendChild(
    svgNode('line', {
      x1: box.left,
      x2: box.left,
      y1: box.top,
      y2: box.bottom,
      stroke: colors.grid,
      'stroke-width': 1,
    })
  );
}

/**
 * @param {SVGElement} svg
 * @param {PlotBox} box
 * @param {string} xTitle
 * @param {string} yTitle
 * @param {RenderColors} colors
 */
function drawAxisTitles(svg, box, xTitle, yTitle, colors) {
  if (xTitle) {
    svg.appendChild(
      svgText(box.left + box.width / 2, FIGURE_HEIGHT - 8, xTitle, {
        class: 'figure-axis-title',
        'text-anchor': 'middle',
        fill: colors.text,
        'font-size': 11,
      })
    );
  }
  if (yTitle) {
    const x = 12;
    const y = box.top + box.height / 2;
    svg.appendChild(
      svgText(x, y, yTitle, {
        class: 'figure-axis-title',
        'text-anchor': 'middle',
        fill: colors.text,
        'font-size': 11,
        transform: `rotate(-90 ${x} ${y})`,
      })
    );
  }
}

/**
 * @param {SVGElement} svg
 * @param {LinesMark} mark
 * @param {FigurePanel} panel
 * @param {PlotBox} box
 * @param {RenderColors} colors
 * @returns {MarkResult}
 */
function drawLines(svg, mark, panel, box, colors) {
  const extent = seriesExtent(mark.series);
  if (extent === null) return { plotted: false, legend: [] };

  const [xMin, xMax] = padRange(extent.xMin, extent.xMax, 0);
  const [yMin, yMax] = padRange(extent.yMin, extent.yMax);
  const scaleX = linearScale(xMin, xMax, box.left, box.right);
  const scaleY = linearScale(yMin, yMax, box.bottom, box.top);

  drawAxes(svg, box, { x: axisTicks(xMin, xMax, scaleX), y: axisTicks(yMin, yMax, scaleY) }, colors);
  drawAxisTitles(
    svg,
    box,
    axisTitle(panel.x_label, panel.x_units),
    axisTitle(panel.y_label, panel.y_units),
    colors
  );

  /** @type {LegendEntry[]} */
  const legend = [];
  mark.series.forEach((line, index) => {
    const color = seriesColor(colors.palette, index, colors.text);
    for (const segment of splitSegments(line.points)) {
      if (segment.length === 1) {
        svg.appendChild(
          svgNode('circle', {
            class: 'figure-point',
            cx: scaleX(segment[0].x).toFixed(1),
            cy: scaleY(Number(segment[0].y)).toFixed(1),
            r: 2,
            fill: color,
          })
        );
        continue;
      }
      svg.appendChild(
        svgNode('polyline', {
          class: 'figure-line',
          points: polylinePoints(segment, scaleX, scaleY),
          fill: 'none',
          stroke: color,
          'stroke-width': 2,
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        })
      );
    }
    legend.push({ label: line.label, color });
  });
  return { plotted: true, legend };
}

/**
 * A heatmap is painted in one series color at varying opacity rather than by
 * interpolating between two token colors: the tokens are arbitrary CSS color
 * strings, and a ramp built by parsing them would break on the first theme
 * that ships a color space this file does not know how to take apart.
 *
 * @param {SVGElement} svg
 * @param {HeatmapMark} mark
 * @param {FigurePanel} panel
 * @param {PlotBox} box
 * @param {RenderColors} colors
 * @returns {MarkResult}
 */
function drawHeatmap(svg, mark, panel, box, colors) {
  const scale = divergingScale(mark.values);
  if (scale === null) return { plotted: false, legend: [] };

  const positive = seriesColor(colors.palette, 0, colors.text);
  // The second palette slot, not a parsed complement: the tokens are arbitrary
  // CSS colour strings and any ramp built by taking them apart would break on
  // the first theme shipping a colour space this file cannot parse.
  const negative = seriesColor(colors.palette, 1, colors.text);
  const missingFill = hatchFill(svg, colors.text);

  for (const cell of heatmapCells(mark, box)) {
    const attributes = {
      x: cell.x.toFixed(1),
      y: cell.y.toFixed(1),
      width: Math.max(0, cell.width).toFixed(1),
      height: Math.max(0, cell.height).toFixed(1),
    };
    // A missing cell is HATCHED, not skipped. Left unpainted it would be plot
    // background -- which is exactly what a zero reading now looks like, so
    // "never measured" and "measured, no response" would be the same picture.
    const rect =
      cell.value === null
        ? svgNode('rect', { ...attributes, class: 'figure-cell figure-cell-missing', fill: missingFill })
        : svgNode('rect', {
            ...attributes,
            class: 'figure-cell',
            fill: scale.isNegative(cell.value) ? negative : positive,
            'fill-opacity': scale.opacityFor(cell.value).toFixed(3),
          });
    const hover = svgNode('title', {});
    hover.textContent = cell.value === null ? 'No reading' : formatTickValue(cell.value);
    rect.appendChild(hover);
    svg.appendChild(rect);
  }

  const { rows, cols } = heatmapShape(mark);
  drawAxes(
    svg,
    box,
    {
      x: categoryTicks(mark.x_labels.slice(0, cols), box.left, box.width),
      y: categoryTicks(mark.y_labels.slice(0, rows), box.top, box.height),
      grid: false,
    },
    colors
  );
  drawAxisTitles(
    svg,
    box,
    axisTitle(panel.x_label, panel.x_units),
    axisTitle(panel.y_label, panel.y_units),
    colors
  );

  // A colorbar, not a legend swatch: a single flat chip labelled "Response
  // slope" says which quantity is painted but leaves every shade in the grid
  // unreadable as a number.
  return {
    plotted: true,
    legend: [],
    colorbar: {
      label: axisTitle(mark.value_label, mark.value_units),
      limit: scale.limit,
      negative,
      positive,
    },
  };
}

/**
 * @param {SVGElement} svg
 * @param {BarsMark} mark
 * @param {FigurePanel} panel
 * @param {PlotBox} box
 * @param {RenderColors} colors
 * @returns {MarkResult}
 */
function drawBars(svg, mark, panel, box, colors) {
  const bars = barGeometry(mark, box);
  if (bars.length === 0) return { plotted: false, legend: [] };

  const domain = barDomain(mark.values);
  const scaleY = linearScale(domain.min, domain.max, box.bottom, box.top);
  drawAxes(
    svg,
    box,
    {
      x: categoryTicks(bars.map((bar) => bar.category), box.left, box.width),
      y: axisTicks(domain.min, domain.max, scaleY),
    },
    colors
  );
  drawAxisTitles(
    svg,
    box,
    axisTitle(panel.x_label, panel.x_units),
    axisTitle(panel.y_label, panel.y_units),
    colors
  );

  for (const bar of bars) {
    if (bar.value === null) continue;
    const color = seriesColor(colors.palette, bar.index, colors.text);
    const rect = svgNode('rect', {
      class: 'figure-bar',
      x: bar.x.toFixed(1),
      y: bar.y.toFixed(1),
      width: bar.width.toFixed(1),
      height: bar.height.toFixed(1),
      fill: color,
    });
    const hover = svgNode('title', {});
    hover.textContent = formatTickValue(bar.value);
    rect.appendChild(hover);
    svg.appendChild(rect);
  }

  const legend = mark.label ? [{ label: mark.label, color: seriesColor(colors.palette, 0, colors.text) }] : [];
  return { plotted: true, legend };
}

/**
 * @param {LegendEntry[]} entries
 * @returns {HTMLElement}
 */
function legendList(entries) {
  const list = document.createElement('ul');
  list.className = 'figure-legend';
  for (const entry of entries) {
    const item = document.createElement('li');
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.backgroundColor = entry.color;
    const label = document.createElement('span');
    label.textContent = entry.label;
    item.append(swatch, label);
    list.appendChild(item);
  }
  return list;
}

/**
 * @param {string[]} notes
 * @returns {HTMLElement}
 */
function annotationList(notes) {
  const list = document.createElement('ul');
  list.className = 'figure-annotations';
  for (const note of notes) {
    const item = document.createElement('li');
    item.textContent = note;
    list.appendChild(item);
  }
  return list;
}

/**
 * @param {FigurePanel} panel
 * @param {RenderColors} colors
 * @param {{selection: Required<FigureSelection>, redraw: () => void}} [view]
 * @returns {HTMLElement}
 */
function renderPanel(panel, colors, view) {
  const section = document.createElement('section');
  section.className = 'figure-panel';
  section.dataset.mark = panel.mark.kind;

  if (panel.title) {
    const heading = document.createElement('h4');
    heading.className = 'figure-panel-title';
    heading.textContent = panel.title;
    section.appendChild(heading);
  }

  // A series picker filters what is DRAWN from series already on the wire, so
  // changing the comparison costs no fetch. The panel keeps every series in
  // `mark` either way -- narrowing that here would make the choice one-way.
  let mark = panel.mark;
  if (view && panel.series_picker && mark.kind === 'lines') {
    const available = mark.series.map((line) => line.label);
    const stored = view.selection.series[panel.title];
    const picked = resolveSeriesSelection(stored, available);
    section.appendChild(
      chipPicker({
        label: 'Show',
        options: available,
        selected: picked,
        multi: true,
        note: `${picked.length} of ${available.length}`,
        onPick: (value) => {
          const next = picked.includes(value)
            ? picked.filter((label) => label !== value)
            : [...picked, value];
          // Never let the last chip off: an empty plot reads as "no data",
          // which would be a lie about a run that has plenty.
          if (next.length > 0) view.selection.series[panel.title] = next;
          view.redraw();
        },
      })
    );
    mark = { ...mark, series: mark.series.filter((line) => picked.includes(line.label)) };
  }

  const box = plotBox();
  const svg = svgNode('svg', {
    class: 'figure-plot',
    viewBox: `0 0 ${FIGURE_WIDTH} ${FIGURE_HEIGHT}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
    'aria-label': panel.title || panel.mark.kind,
  });
  svg.appendChild(
    svgNode('rect', {
      class: 'figure-bg',
      x: 0,
      y: 0,
      width: FIGURE_WIDTH,
      height: FIGURE_HEIGHT,
      fill: colors.plotBg,
    })
  );

  let result;
  if (mark.kind === 'lines') result = drawLines(svg, mark, panel, box, colors);
  else if (mark.kind === 'heatmap') result = drawHeatmap(svg, mark, panel, box, colors);
  else result = drawBars(svg, mark, panel, box, colors);

  if (result.plotted) {
    section.appendChild(svg);
    if (result.colorbar) {
      section.appendChild(
        colorbarElement({
          label: result.colorbar.label,
          low: formatTickValue(-result.colorbar.limit),
          high: formatTickValue(result.colorbar.limit),
          negative: result.colorbar.negative,
          positive: result.colorbar.positive,
        })
      );
    }
    if (result.legend.length > 0) section.appendChild(legendList(result.legend));
  } else {
    const empty = document.createElement('p');
    empty.className = 'figure-empty';
    empty.textContent = 'No plottable values in this panel yet.';
    section.appendChild(empty);
  }

  // Annotations are appended whether or not anything was drawn: a panel with
  // no readings still owes the reader its decimation and gap notes.
  const notes = panelAnnotations(panel);
  if (notes.length > 0) section.appendChild(annotationList(notes));
  return section;
}

/** @typedef {{panels: HTMLElement, note?: HTMLElement|null}} FigureElements */

/**
 * Draw `figure` into the caller's containers.
 *
 * `elements.panels` is emptied and refilled with one `<section
 * class="figure-panel">` per panel. `elements.note`, when given, receives the
 * figure-level note (source, partial, reason) and is un-hidden; without it the
 * note is written as the first child of `elements.panels` instead, so the
 * provenance line is never silently dropped.
 *
 * Theme colors are read here, once per render, so a caller re-rendering from a
 * theme change gets the new palette.
 *
 * Pass a stable `options.selection` object to give the figure's pickers and
 * disclosures memory: this function replaces its container's children on every
 * call, so any state left in that DOM dies with it. The object is the caller's
 * -- reset it when the RUN changes, or one run's choices would be re-applied to
 * the next run's panels.
 *
 * @param {FigureElements} elements
 * @param {Figure} figure
 * @param {{selection?: FigureSelection}} [options]
 */
export function renderFigure(elements, figure, options = {}) {
  const theme = chartTheme();
  /** @type {RenderColors} */
  const colors = {
    text: theme.font.color || 'currentColor',
    grid: theme.xaxis.gridcolor || theme.yaxis.gridcolor || 'currentColor',
    plotBg: theme.plot_bgcolor || 'transparent',
    palette: chartSeries(),
  };

  const root = elements.panels;
  root.replaceChildren();

  const noteText = figureNoteText(figure);
  if (elements.note) {
    elements.note.textContent = noteText;
    elements.note.hidden = false;
  } else {
    const note = document.createElement('p');
    note.className = 'figure-note';
    note.textContent = noteText;
    root.appendChild(note);
  }

  if (figure.panels.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'figure-empty';
    empty.textContent = 'No figure for this run.';
    root.appendChild(empty);
    return;
  }

  const store = options.selection ?? {};
  store.series ??= {};
  store.section ??= {};
  store.open ??= {};
  const view = {
    selection: /** @type {Required<FigureSelection>} */ (store),
    redraw: () => renderFigure(elements, figure, { ...options, selection: store }),
  };

  const { inline, sections } = groupPanels(figure.panels);
  for (const panel of inline) root.appendChild(renderPanel(panel, colors, view));
  for (const group of sections) {
    root.appendChild(sectionElement(group, view, (panel) => renderPanel(panel, colors, view)));
  }
}
