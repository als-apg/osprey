// @ts-check
/**
 * The objective chart: one drawing of a run's objective for the run page and
 * for an ended or stored run. Plain SVG, built node by node (no markup
 * strings), coloured by design-system tokens in css/objective-chart.css, so
 * it follows the theme by itself.
 *
 * It draws each evaluation (a dot; the first tries in a shaded band), the
 * best so far (a step line, its point ringed), the value before the run (a
 * dashed line), the evaluations not run yet (a quiet band up to a fixed
 * x-range), an optional end marker (where a run paused or stopped), and the
 * point of the last evaluation (ringed; pulsing while the run is live). An
 * input it does not get is not drawn. The SVG has a text alternative that
 * names the count, the best and the value before the run.
 *
 * `update` is cheap to call on every poll: it redraws only when what it
 * shows changed, and it moves the drawn points in place instead of rebuilding
 * the drawing, so a live run does not flicker. The box takes its size from
 * CSS; the drawing follows the box.
 *
 * @module objective-chart
 */

import { fmt, h } from './hyperscript.js';
import { finite, runningBest } from './run-model.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** The size drawn while the box has none yet (hidden, or not laid out). */
const FALLBACK = Object.freeze({ width: 640, height: 320 });

/** Roughly one monospace glyph of the axis text, for the margins. */
const GLYPH = 7;

/**
 * @typedef {Record<string, unknown>} DataRecord
 * @typedef {'success'|'warning'|'error'|'muted'} EndTone
 *
 * What the chart shows. `records` are the data records as the job (`data`) or
 * the stored run (`data_records`) give them; indices are 0-based positions in
 * it (the chart labels them from 1).
 * @typedef {{
 *   records: DataRecord[]|null|undefined,
 *   objective: string|null,
 *   direction: string|null,
 *   label?: string|null,
 *   units?: string|null,
 *   before?: number|null,
 *   best?: { index: number, value: number }|null,
 *   bests?: (number|null)[]|null,
 *   feasible?: (boolean|null)[]|null,
 *   end?: { index: number, label: string, tone?: EndTone }|null,
 *   xMax?: number|null,
 *   firstTries?: number|null,
 *   current?: 'live'|'still'|null,
 * }} ChartInput
 *
 * @typedef {{
 *   values: (number|null)[],
 *   bests: (number|null)[]|null,
 *   best: { index: number, value: number }|null,
 *   before: number|null,
 *   xMax: number,
 *   lo: number,
 *   hi: number,
 * }} ChartModel
 */

/**
 * The chart's input from a run's figures (run-model.js): the records, the
 * objective, its direction, the value before the run, the best, the best so
 * far at each record and whether each record meets the constraints, so the
 * chart names the same best as the figures (the point `apply_best` writes).
 *
 * @param {import('./run-model.js').RunFigures} figures
 * @param {DataRecord[]|null|undefined} records
 * @returns {ChartInput}
 */
export function chartInput(figures, records) {
  const { objective, direction, before, best, bestIndex, bests, feasible } = figures;
  return {
    records,
    objective,
    direction,
    before,
    best: bestIndex === null || best === null ? null : { index: bestIndex, value: best },
    bests,
    feasible,
  };
}

/**
 * What the chart draws, without the drawing: the value of each evaluation,
 * the best so far, the best point, the x-range and the y-range.
 *
 * @param {ChartInput} input
 * @returns {ChartModel}
 */
export function chartModel(input) {
  const objective = input.objective;
  const values = objective ? (input.records || []).map((r) => finite(r?.[objective])) : [];
  const direction = input.direction ? String(input.direction).toUpperCase() : null;
  const bests = input.bests !== undefined ? input.bests : runningBest(values, direction);
  /** @type {{ index: number, value: number }|null} */
  let best = input.best === undefined ? null : input.best;
  if (input.best === undefined && bests) {
    for (let i = 0; i < bests.length; i += 1) {
      const v = bests[i];
      if (v !== null && (i === 0 || v !== bests[i - 1])) best = { index: i, value: v };
    }
  }
  const before = finite(input.before);
  const ys = /** @type {number[]} */ ([...values, before].filter((v) => v !== null));
  let lo = ys.length ? Math.min(...ys) : 0;
  let hi = ys.length ? Math.max(...ys) : 1;
  const span = hi - lo || Math.abs(hi) * 0.1 || 1;
  lo -= span * 0.08;
  hi += span * 0.1;
  const xMax = Math.max(values.length, finite(input.xMax) ?? 0, 1);
  return { values, bests, best, before, xMax, lo, hi };
}

/**
 * Round tick values from `lo` to `hi`, about `count` of them.
 *
 * @param {number} lo
 * @param {number} hi
 * @param {number} count
 * @returns {{ ticks: number[], step: number }}
 */
export function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    ticks.push(Number(v.toPrecision(12)));
  }
  return { ticks, step };
}

/**
 * A tick's text, with as many decimals as its step needs.
 *
 * @param {number} value
 * @param {number} step
 */
function tickText(value, step) {
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  if (decimals > 4) return value === 0 ? '0' : value.toExponential(1);
  return value.toFixed(decimals);
}

/**
 * The text alternative of the drawing.
 *
 * @param {ChartInput} input
 * @param {ChartModel} model
 */
function chartSummary(input, model) {
  const name = input.label || input.objective || 'The objective';
  const unit = input.units ? ` ${input.units}` : '';
  const n = model.values.length;
  const parts = [`${name} at each of ${n} evaluation${n === 1 ? '' : 's'}`];
  if (model.best) {
    parts.push(`best so far ${fmt(model.best.value)}${unit} at evaluation ${model.best.index + 1}`);
  }
  if (model.before !== null) parts.push(`before the run ${fmt(model.before)}${unit}`);
  if (input.end) parts.push(`${input.end.label} after evaluation ${input.end.index + 1}`);
  return `${parts.join('; ')}.`;
}

/**
 * An SVG element, with the prop rules of `h`: `class`, `text`, `null`/`false`
 * skipped, the rest attributes.
 *
 * @param {string} tag
 * @param {Record<string, unknown>|null} [props]
 * @param {...(Node|null|false|undefined)} children
 * @returns {SVGElement}
 */
function sv(tag, props, ...children) {
  const node = /** @type {SVGElement} */ (document.createElementNS(SVG_NS, tag));
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'text') node.textContent = String(value);
    else node.setAttribute(key, String(value));
  }
  for (const child of children) if (child) node.appendChild(child);
  return node;
}

/**
 * Set attributes of a drawn node; `null` removes one.
 *
 * @param {Element} node
 * @param {Record<string, string|number|null>} values
 */
function attrs(node, values) {
  for (const [key, value] of Object.entries(values)) {
    if (value === null) node.removeAttribute(key);
    else if (node.getAttribute(key) !== String(value)) node.setAttribute(key, String(value));
  }
}

/** @param {number} v */
const px = (v) => v.toFixed(1);

/**
 * The key to the drawing, for a caption beside the chart.
 *
 * @param {{ before?: boolean }} [options]  `before`: the run has a value before it
 * @returns {HTMLElement}
 */
export function chartLegend({ before = true } = {}) {
  /** @param {string} key @param {string} text */
  const item = (key, text) =>
    h(
      'span',
      { class: 'opt-legend-item', 'data-key': key },
      h('i', { class: `opt-ch-swatch opt-ch-swatch-${key}` }),
      text
    );
  return h(
    'span',
    { class: 'opt-legend' },
    item('point', 'each evaluation'),
    item('best', 'best so far'),
    before ? item('before', 'before the run') : null
  );
}

/**
 * Mount a chart in `host`. It appends its own box and redraws on `update`
 * and when the box changes size.
 *
 * @param {HTMLElement} host
 * @returns {{ update: (input: ChartInput) => void, destroy: () => void, box: HTMLElement }}
 */
export function mountObjectiveChart(host) {
  const layer = (/** @type {string} */ name) => sv('g', { 'data-layer': name });
  const bands = layer('bands');
  const grid = layer('grid');
  const beforeLine = sv('line', { class: 'opt-ch-before' });
  const bestLine = sv('path', { class: 'opt-ch-bestline' });
  const points = layer('points');
  const bestRing = sv('circle', { class: 'opt-ch-bestpt', r: 5 });
  const current = sv('circle', { class: 'opt-ch-cur', r: 6.5 });
  const end = layer('end');
  const labels = layer('labels');
  const svg = sv(
    'svg',
    { class: 'opt-chart-svg', role: 'img', focusable: 'false' },
    bands,
    grid,
    beforeLine,
    bestLine,
    points,
    bestRing,
    current,
    end,
    labels
  );
  const box = h('figure', { class: 'opt-chart' }, svg);
  host.append(box);

  /** @type {ChartInput|null} */
  let input = null;
  let drawn = '';

  function draw() {
    if (!input) return;
    const width = Math.floor(box.clientWidth) || FALLBACK.width;
    const height = Math.floor(box.clientHeight) || FALLBACK.height;
    const model = chartModel(input);
    const key = JSON.stringify([width, height, input]);
    if (key === drawn) return;
    drawn = key;
    render(input, model, width, height);
  }

  /**
   * @param {ChartInput} input
   * @param {ChartModel} model
   * @param {number} W
   * @param {number} H
   */
  function render(input, model, W, H) {
    const { values, bests, best, before, xMax, lo, hi } = model;
    const yTicks = niceTicks(lo, hi, Math.max(3, Math.round((H - 44) / 70)));
    const yTexts = yTicks.ticks.map((t) => tickText(t, yTicks.step));
    const sideTexts = [
      'evaluation',
      before === null ? '' : `before ${fmt(before, 4)}`,
      best ? `best ${fmt(best.value, 4)}` : '',
    ];
    const pad = {
      l: Math.min(84, 14 + GLYPH * Math.max(...yTexts.map((t) => t.length), 2)),
      r: Math.min(132, 16 + GLYPH * Math.max(...sideTexts.map((t) => t.length))),
      t: 14,
      b: 26,
    };
    const pw = Math.max(1, W - pad.l - pad.r);
    const ph = Math.max(1, H - pad.t - pad.b);
    const X = (/** @type {number} */ i) => pad.l + ((i + 0.5) / xMax) * pw;
    const Y = (/** @type {number} */ v) => pad.t + ((hi - v) / (hi - lo)) * ph;
    attrs(svg, { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    svg.setAttribute('aria-label', chartSummary(input, model));

    // Bands: the first tries, and the evaluations not run yet.
    const firstTries = Math.min(finite(input.firstTries) ?? 0, xMax);
    const bandNodes = [];
    if (firstTries > 0) {
      const w = X(firstTries - 0.5) - pad.l;
      bandNodes.push(
        sv('rect', { class: 'opt-ch-init', x: pad.l, y: pad.t, width: px(w), height: ph }),
        w > 70
          ? sv('text', {
              class: 'opt-ch-txt opt-ch-band',
              x: pad.l + 6,
              y: pad.t + 14,
              text: 'first tries',
            })
          : null
      );
    }
    if (values.length < xMax) {
      const x0 = X(values.length - 0.5);
      bandNodes.push(
        sv('rect', {
          class: 'opt-ch-future',
          x: px(x0),
          y: pad.t,
          width: px(pad.l + pw - x0),
          height: ph,
        })
      );
    }
    bands.replaceChildren(...bandNodes.filter((n) => n !== null));

    // Grid and axes.
    const gridNodes = [];
    yTicks.ticks.forEach((t, k) => {
      const y = px(Y(t));
      gridNodes.push(
        sv('line', { class: 'opt-ch-grid', x1: pad.l, x2: pad.l + pw, y1: y, y2: y }),
        sv('text', {
          class: 'opt-ch-txt',
          x: pad.l - 6,
          y: Number(y) + 4,
          'text-anchor': 'end',
          text: yTexts[k],
        })
      );
    });
    const xTicks = new Set([1, xMax]);
    const every = niceTicks(0, xMax, Math.max(2, Math.floor(pw / 80))).step;
    for (let i = Math.max(1, Math.ceil(every)); i < xMax; i += Math.max(1, Math.ceil(every))) {
      if (X(xMax - 1) - X(i - 1) > 28) xTicks.add(i);
    }
    for (const i of xTicks) {
      gridNodes.push(
        sv('text', {
          class: 'opt-ch-txt',
          x: px(X(i - 1)),
          y: H - 8,
          'text-anchor': 'middle',
          text: String(i),
        })
      );
    }
    gridNodes.push(
      sv('text', { class: 'opt-ch-txt', x: pad.l + pw + 10, y: H - 8, text: 'evaluation' })
    );
    // Nothing recorded yet, or records with no finite value of the objective.
    if (!values.some((v) => v !== null)) {
      gridNodes.push(
        sv('text', {
          class: 'opt-ch-txt opt-ch-empty',
          x: px(pad.l + pw / 2),
          y: px(pad.t + ph / 2),
          'text-anchor': 'middle',
          text: input.records?.length ? 'No finite value read yet' : 'No evaluation yet',
        })
      );
    }
    grid.replaceChildren(...gridNodes);

    // Before the run.
    if (before === null) attrs(beforeLine, { display: 'none' });
    else {
      const y = px(Y(before));
      attrs(beforeLine, { display: null, x1: pad.l, x2: pad.l + pw, y1: y, y2: y });
    }

    // The best so far, as a step line.
    let path = '';
    if (bests) {
      bests.forEach((v, i) => {
        if (v === null) return;
        const y = px(Y(v));
        path += path ? `H${px(X(i))}V${y}` : `M${px(X(i) - 4)},${y}`;
      });
      if (path) path += `H${px(X(bests.length - 1) + 4)}`;
    }
    if (path) attrs(bestLine, { display: null, d: path });
    else attrs(bestLine, { display: 'none', d: null });

    // Each evaluation: the drawn dots move in place, new ones are added.
    const drawnDots = /** @type {Element[]} */ (Array.from(points.children));
    let used = 0;
    values.forEach((v, i) => {
      if (v === null) return;
      const dot = drawnDots[used] || points.appendChild(sv('circle', { r: 2.8 }));
      used += 1;
      const bad = input.feasible?.[i] === false;
      const cls = `opt-ch-pt${i < firstTries ? ' is-init' : ''}${bad ? ' is-bad' : ''}`;
      attrs(dot, { class: cls, cx: px(X(i)), cy: px(Y(v)), 'data-index': i });
    });
    for (const extra of drawnDots.slice(used)) extra.remove();

    // The best point and the last point.
    if (best && finite(best.value) !== null) {
      attrs(bestRing, { display: null, cx: px(X(best.index)), cy: px(Y(best.value)) });
    } else attrs(bestRing, { display: 'none' });
    const last = values.length - 1;
    if (input.current && last >= 0 && values[last] !== null) {
      attrs(current, {
        display: null,
        class: `opt-ch-cur${input.current === 'live' ? ' is-live' : ''}`,
        cx: px(X(last)),
        cy: px(Y(/** @type {number} */ (values[last]))),
      });
    } else attrs(current, { display: 'none' });

    // The end marker, between the evaluation it follows and the next.
    if (input.end && input.end.index >= 0) {
      const x = px(X(input.end.index + 0.5));
      const tone = input.end.tone || 'muted';
      end.replaceChildren(
        sv('line', { class: `opt-ch-end is-${tone}`, x1: x, x2: x, y1: pad.t, y2: pad.t + ph }),
        sv('text', {
          class: `opt-ch-txt opt-ch-endlabel is-${tone}`,
          x: Number(x) + 4,
          y: pad.t + ph - 6,
          text: input.end.label,
        })
      );
    } else end.replaceChildren();

    // End labels in the right margin: before the run, and the best so far.
    const marks = [];
    if (before !== null)
      marks.push({ y: Y(before), text: sideTexts[1], cls: 'opt-ch-lbl-before', sub: '' });
    if (best)
      marks.push({
        y: Y(best.value),
        text: sideTexts[2],
        cls: 'opt-ch-lbl-best',
        sub: `evaluation ${best.index + 1}`,
      });
    if (marks.length === 2 && Math.abs(marks[0].y - marks[1].y) < 28) {
      marks[0].y = marks[1].y + (marks[0].y >= marks[1].y ? 28 : -28);
    }
    labels.replaceChildren(
      ...marks
        .flatMap((m) => [
          sv('text', {
            class: `opt-ch-txt ${m.cls}`,
            x: pad.l + pw + 8,
            y: px(m.y + 4),
            text: m.text,
          }),
          m.sub
            ? sv('text', { class: 'opt-ch-txt', x: pad.l + pw + 8, y: px(m.y + 17), text: m.sub })
            : null,
        ])
        .filter((n) => n !== null)
    );
    box.dataset.empty = values.some((v) => v !== null) ? 'false' : 'true';
  }

  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(() => draw()) : null;
  observer?.observe(box);

  return {
    box,
    update(next) {
      input = next;
      draw();
    },
    destroy() {
      observer?.disconnect();
      box.remove();
    },
  };
}
