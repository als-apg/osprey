// @ts-check
/**
 * BLUESKY panel — an array cell of the Results table, as a sketch and a plot.
 *
 * A run that records an array per row (a detector's spectrum) serves each cell
 * of that column as a JSON list in the same data window the table holds. This
 * module turns one such cell into the two views the table offers: a small
 * sketch that sits beside the cell's summary, and a `FigurePanel` the renderer
 * draws when the cell is opened. The plot is exactly the cell on screen — no
 * fetch, no route of its own.
 *
 * Everything here is pure or builds one detached element. No fetching, no
 * drawing beyond the sketch, no theme: the sketch strokes in `currentColor`
 * and takes its colour from the stylesheet. Every node is built with
 * `createElementNS`/`setAttribute`; nothing here assigns `innerHTML`.
 */

import { linearScale } from './figure-renderer.js';

/** @typedef {import('./figure-renderer.js').FigurePanel} FigurePanel */

const SVG_NS = 'http://www.w3.org/2000/svg';

/** The sketch's size in CSS pixels; its viewBox uses the same units. */
export const SPARK_WIDTH = 64;
export const SPARK_HEIGHT = 14;

/**
 * Whether `value` is an array the cell plot can draw: every element a number
 * or `null`. A nested list is not — flattening a 2-D reading into one line
 * would draw a picture of nothing — and neither is a list of text.
 *
 * @param {unknown} value
 * @returns {value is Array<number|null>}
 */
export function isPlottableArray(value) {
  return Array.isArray(value) && value.every((v) => v === null || typeof v === 'number');
}

/** @param {number} value */
function coord(value) {
  return String(Number(value.toFixed(2)));
}

/**
 * The sketch's polyline `points`, or `null` when no element is finite.
 *
 * Thinned by a min/max envelope: each pixel column keeps the lowest and the
 * highest finite value of the elements it covers, in the order they occur, so
 * a one-sample peak survives the thinning. An array no wider than the sketch
 * keeps every finite element. A flat array is a level line through the middle.
 *
 * @param {Array<number|null>} values
 * @param {number} [width]
 * @param {number} [height]
 * @returns {string|null}
 */
export function sparklinePoints(values, width = SPARK_WIDTH, height = SPARK_HEIGHT) {
  const finite = values.filter(
    /** @returns {v is number} */ (v) => typeof v === 'number' && Number.isFinite(v)
  );
  if (finite.length === 0) return null;
  const low = finite.reduce((a, b) => Math.min(a, b));
  const high = finite.reduce((a, b) => Math.max(a, b));
  const y = linearScale(low, high, height, 0);
  /** @type {string[]} */
  const points = [];
  /** @param {number} x @param {number} value */
  const push = (x, value) => points.push(`${coord(x)},${coord(y(value))}`);

  if (values.length <= width) {
    const x = linearScale(0, values.length - 1, 0, width);
    values.forEach((value, index) => {
      if (typeof value === 'number' && Number.isFinite(value)) push(x(index), value);
    });
    return points.join(' ');
  }

  for (let column = 0; column < width; column += 1) {
    const start = Math.floor((column * values.length) / width);
    const end = Math.floor(((column + 1) * values.length) / width);
    let minIndex = -1;
    let maxIndex = -1;
    for (let index = start; index < end; index += 1) {
      const value = values[index];
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      if (minIndex < 0 || value < /** @type {number} */ (values[minIndex])) minIndex = index;
      if (maxIndex < 0 || value > /** @type {number} */ (values[maxIndex])) maxIndex = index;
    }
    if (minIndex < 0) continue;
    const x = column + 0.5;
    const first = Math.min(minIndex, maxIndex);
    const second = Math.max(minIndex, maxIndex);
    push(x, /** @type {number} */ (values[first]));
    if (second !== first) push(x, /** @type {number} */ (values[second]));
  }
  return points.join(' ');
}

/**
 * The sketch of one array cell as a detached `<svg class="array-spark">`, or
 * `null` when the array has no finite element to draw.
 *
 * @param {Array<number|null>} values
 * @returns {SVGSVGElement|null}
 */
export function sparklineElement(values) {
  const points = sparklinePoints(values);
  if (points === null) return null;
  const svg = /** @type {SVGSVGElement} */ (document.createElementNS(SVG_NS, 'svg'));
  svg.setAttribute('class', 'array-spark');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('viewBox', `0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  const line = document.createElementNS(SVG_NS, 'polyline');
  line.setAttribute('points', points);
  line.setAttribute('fill', 'none');
  line.setAttribute('stroke', 'currentColor');
  line.setAttribute('stroke-width', '1');
  line.setAttribute('vector-effect', 'non-scaling-stroke');
  svg.appendChild(line);
  return svg;
}

/**
 * One array cell as a figure panel: its elements against their index.
 *
 * A `null` or non-finite element is a gap, which the renderer breaks the line
 * at and annotates; it is never drawn as zero. Every element is drawn, so the
 * panel is never decimated.
 *
 * @param {{column: string, unit: string|null, values: Array<number|null>, row: number, rows: number}} cell
 *   `row` is the zero-based index of the cell's row; `rows` the window's row count.
 * @returns {FigurePanel}
 */
export function cellPlotPanel({ column, unit, values, row, rows }) {
  const points = values.map((value, index) => ({
    x: index,
    y: typeof value === 'number' && Number.isFinite(value) ? value : null,
  }));
  return {
    title: `${column} · row ${row + 1} of ${rows}`,
    x_label: 'index',
    y_label: column,
    x_units: null,
    y_units: unit,
    x_columns: null,
    y_columns: null,
    annotations: [],
    mark: {
      kind: 'lines',
      series: [{ label: column, points, decimated: false, source_points: values.length }],
      decimated: false,
      source_points: values.length,
    },
  };
}

/**
 * The row one step from `row` in `direction`, held within `0..rows-1`: the same
 * row at either end.
 *
 * @param {number} row
 * @param {-1|1} direction
 * @param {number} rows
 * @returns {number}
 */
export function stepRow(row, direction, rows) {
  return Math.min(Math.max(row + direction, 0), Math.max(rows - 1, 0));
}
