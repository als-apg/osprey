// The lattice panel's charts: pure functions from numbers to an SVG string. Colours are
// the design system's chart tokens, so a chart follows the theme without a redraw.

/**
 * @typedef {object} Series
 * @property {string} label
 * @property {string} color a CSS colour, e.g. `var(--chart-series-1)`
 * @property {(number | null)[]} x
 * @property {(number | null)[]} y
 * @property {'line' | 'points' | 'both'} [mode] `line` when omitted
 * @property {boolean} [close] join the last point to the first
 * @property {string[]} [colors] one colour per point, for `points`
 */

/**
 * @typedef {object} Segment
 * @property {number} x1
 * @property {number} y1
 * @property {number} x2
 * @property {number} y2
 * @property {number} order
 */

/**
 * @typedef {object} PlotSpec
 * @property {string} xLabel
 * @property {string} yLabel
 * @property {Series[]} series
 * @property {Segment[]} [segments] drawn under the series, e.g. resonance lines
 * @property {[number, number]} [xRange] the data's extent when omitted
 * @property {[number, number]} [yRange] the data's extent when omitted
 * @property {number} [width]
 * @property {number} [height]
 */

const MARGIN = { left: 58, right: 14, top: 22, bottom: 38 };

/** @param {unknown} text */
export function esc(text) {
  return String(text).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c
  );
}

/**
 * A number as the panel prints it: six significant digits, no trailing zeros.
 * @param {number | string | null | undefined} value
 */
export function fmt(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6 || magnitude < 1e-4) return value.toExponential(3);
  return String(Number(value.toPrecision(6)));
}

/**
 * The extent of the finite values, widened by `pad` of its span on each side.
 * @param {(number | null)[][]} columns
 * @param {number} [pad]
 * @returns {[number, number]}
 */
export function extent(columns, pad = 0.05) {
  let low = Infinity;
  let high = -Infinity;
  for (const column of columns) {
    for (const value of column) {
      if (value === null || !Number.isFinite(value)) continue;
      if (value < low) low = value;
      if (value > high) high = value;
    }
  }
  if (low > high) return [0, 1];
  const span = high - low || Math.abs(high) || 1;
  return [low - span * pad, high + span * pad];
}

/**
 * Round tick values covering `[min, max]`.
 * @param {number} min
 * @param {number} max
 * @param {number} [count]
 * @returns {number[]}
 */
export function ticks(min, max, count = 6) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw;
  const out = [];
  // Each tick from its index, not a running sum, so 0.6 does not come out as 0.6000000000000001.
  for (let k = Math.ceil(min / step - 1e-9); k * step <= max + step * 1e-9; k += 1) {
    out.push(Number((k * step).toPrecision(12)));
  }
  return out;
}

/**
 * The resonance lines `m·νx + n·νy = p` up to `maxOrder`, clipped to the window.
 * @param {[number, number]} xRange
 * @param {[number, number]} yRange
 * @param {number} [maxOrder]
 * @returns {Segment[]}
 */
export function resonanceSegments(xRange, yRange, maxOrder = 4) {
  const [x0, x1] = xRange;
  const [y0, y1] = yRange;
  /** @type {Segment[]} */
  const out = [];
  for (let m = 0; m <= maxOrder; m += 1) {
    for (let n = -maxOrder; n <= maxOrder; n += 1) {
      const order = m + Math.abs(n);
      if (order === 0 || order > maxOrder || (m === 0 && n < 0)) continue;
      const corners = [m * x0 + n * y0, m * x0 + n * y1, m * x1 + n * y0, m * x1 + n * y1];
      const first = Math.ceil(Math.min(...corners));
      const last = Math.floor(Math.max(...corners));
      for (let p = first; p <= last; p += 1) {
        if (n === 0) {
          out.push({ x1: p / m, y1: y0, x2: p / m, y2: y1, order });
          continue;
        }
        // The line as y(x), cut where it leaves the window's y range.
        let xa = x0;
        let xb = x1;
        if (m !== 0) {
          const atY0 = (p - n * y0) / m;
          const atY1 = (p - n * y1) / m;
          xa = Math.max(x0, Math.min(atY0, atY1));
          xb = Math.min(x1, Math.max(atY0, atY1));
          if (xa >= xb) continue;
        }
        out.push({ x1: xa, y1: (p - m * xa) / n, x2: xb, y2: (p - m * xb) / n, order });
      }
    }
  }
  return out;
}

/**
 * A colour for a diffusion rate (log10 of the tune drift): blue stable, red chaotic.
 * @param {number | null} value
 */
export function diffusionColor(value) {
  if (value === null || !Number.isFinite(value)) return 'var(--chart-axis-text)';
  const share = Math.min(1, Math.max(0, (value + 10) / 8));
  return `hsl(${Math.round(240 - 240 * share)} 75% 50%)`;
}

/**
 * One chart as an SVG string.
 * @param {PlotSpec} spec
 */
export function plot(spec) {
  const width = spec.width ?? 640;
  const height = spec.height ?? 280;
  const [x0, x1] =
    spec.xRange ??
    extent(
      spec.series.map((s) => s.x),
      0.02
    );
  const [y0, y1] = spec.yRange ?? extent(spec.series.map((s) => s.y));
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;
  /** @param {number} x */
  const px = (x) => MARGIN.left + ((x - x0) / (x1 - x0 || 1)) * innerW;
  /** @param {number} y */
  const py = (y) => MARGIN.top + innerH - ((y - y0) / (y1 - y0 || 1)) * innerH;
  /** @param {number} v */
  const r = (v) => v.toFixed(1);

  const parts = [
    `<svg class="lt-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(spec.yLabel)} against ${esc(spec.xLabel)}">`,
    `<rect class="lt-plot-bg" x="${MARGIN.left}" y="${MARGIN.top}" width="${innerW}" height="${innerH}"/>`,
  ];
  for (const tick of ticks(x0, x1)) {
    const x = r(px(tick));
    parts.push(
      `<line class="lt-grid" x1="${x}" y1="${MARGIN.top}" x2="${x}" y2="${MARGIN.top + innerH}"/>`,
      `<text class="lt-tick" x="${x}" y="${MARGIN.top + innerH + 14}" text-anchor="middle">${fmt(tick)}</text>`
    );
  }
  for (const tick of ticks(y0, y1, 5)) {
    const y = r(py(tick));
    parts.push(
      `<line class="lt-grid" x1="${MARGIN.left}" y1="${y}" x2="${MARGIN.left + innerW}" y2="${y}"/>`,
      `<text class="lt-tick" x="${MARGIN.left - 6}" y="${y}" text-anchor="end" dominant-baseline="middle">${fmt(tick)}</text>`
    );
  }
  for (const segment of spec.segments ?? []) {
    parts.push(
      `<line class="lt-res" data-order="${segment.order}" x1="${r(px(segment.x1))}" y1="${r(py(segment.y1))}" x2="${r(px(segment.x2))}" y2="${r(py(segment.y2))}"/>`
    );
  }
  let legendX = MARGIN.left;
  for (const series of spec.series) {
    const mode = series.mode ?? 'line';
    /** @type {[number, number][][]} */
    const runs = [[]];
    series.x.forEach((x, index) => {
      const y = series.y[index];
      if (
        x === null ||
        y === null ||
        y === undefined ||
        !Number.isFinite(x) ||
        !Number.isFinite(y)
      ) {
        if (runs[runs.length - 1].length) runs.push([]); // a gap: the line breaks here
        return;
      }
      runs[runs.length - 1].push([px(x), py(y)]);
    });
    if (mode !== 'points') {
      for (const run of runs) {
        if (run.length < 2) continue;
        const path = run.map(([x, y], i) => `${i ? 'L' : 'M'}${r(x)} ${r(y)}`).join('');
        parts.push(
          `<path class="lt-line" style="stroke:${series.color}" d="${path}${series.close ? 'Z' : ''}"/>`
        );
      }
    }
    if (mode !== 'line') {
      series.x.forEach((x, index) => {
        const y = series.y[index];
        if (
          x === null ||
          y === null ||
          y === undefined ||
          !Number.isFinite(x) ||
          !Number.isFinite(y)
        )
          return;
        const color = series.colors?.[index] ?? series.color;
        parts.push(
          `<circle class="lt-pt" style="fill:${color}" cx="${r(px(x))}" cy="${r(py(y))}" r="2.6"/>`
        );
      });
    }
    parts.push(
      `<text class="lt-legend" style="fill:${series.color}" x="${legendX}" y="${MARGIN.top - 8}">${esc(series.label)}</text>`
    );
    legendX += series.label.length * 7 + 18;
  }
  parts.push(
    `<text class="lt-label" x="${MARGIN.left + innerW / 2}" y="${height - 6}" text-anchor="middle">${esc(spec.xLabel)}</text>`,
    `<text class="lt-label" x="12" y="${MARGIN.top + innerH / 2}" text-anchor="middle" transform="rotate(-90 12 ${MARGIN.top + innerH / 2})">${esc(spec.yLabel)}</text>`,
    '</svg>'
  );
  return parts.join('');
}
