// @ts-check
/**
 * The knobs of an ended run: the Knobs view of the outcome's chart area and
 * the inspector's list of what changed on the machine.
 *
 * Each knob is drawn on its safe range (`VariableMetadata.min_value`/
 * `max_value`), or on its search window when the environment has not been
 * read, with the run's marks: where it started (`initial_point`/
 * `snapshots.initial`), its best point, and the value on the machine when the
 * outcome proves one (see outcome-model.js). Under the rows, each knob at
 * each evaluation as a line in its search window. A value the run does not
 * hold is not drawn.
 *
 * @module outcome-knobs
 */

import { fmt, h } from './hyperscript.js';
import { knobBar, knobLabel, knobLegend, knobName } from './knob-bar.js';
import { signed } from './words.js';

/**
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./run-model.js').RunFigures} RunFigures
 * @typedef {import('./outcome-model.js').MachineState} MachineState
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * @param {string} tag
 * @param {Record<string, string|number>} attrs
 * @returns {SVGElement}
 */
function svg(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

/** @param {string|undefined} units */
const unit = (units) => (units ? `\u00a0${units}` : '');

/**
 * A knob's search window: the one the run used (`variable_bounds` of its job,
 * clamped and snapped by the service), else the stored config's; null
 * without one.
 *
 * @param {Record<string, number[]>|null|undefined} bounds
 * @param {OptimizationConfig|null} config
 * @param {string} name
 * @returns {{ min: number, max: number }|null}
 */
function windowOf(bounds, config, name) {
  const used = bounds?.[name];
  if (Array.isArray(used) && Number.isFinite(used[0]) && Number.isFinite(used[1])) {
    return { min: used[0], max: used[1] };
  }
  const o = config?.variable_overrides?.[name];
  return typeof o?.min === 'number' && typeof o?.max === 'number'
    ? { min: o.min, max: o.max }
    : null;
}

/**
 * @param {Record<string, unknown>|null|undefined} record
 * @param {string} name
 * @returns {number|null}
 */
function valueOf(record, name) {
  const v = record?.[name];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/**
 * @typedef {{
 *   variables: string[],
 *   config: OptimizationConfig|null,
 *   bounds?: Record<string, number[]>|null,
 *   env: EnvironmentDetails|null,
 *   figures: RunFigures,
 *   machine: MachineState,
 *   records: Record<string, unknown>[],
 * }} KnobsSource
 */

/**
 * The knob rows of the Knobs view.
 *
 * @param {KnobsSource} source
 * @returns {HTMLElement}
 */
function knobRows({ variables, config, bounds, env, figures, machine }) {
  const onMachine = machine.values;
  const endWord = onMachine ? 'now' : 'last';
  return h(
    'div',
    { class: 'opt-out-knobs' },
    h(
      'div',
      { class: 'opt-out-knob head' },
      h('span', { class: 'osprey-kicker', text: 'Knob' }),
      h('span', {
        class: 'osprey-kicker',
        text: env ? 'Safe range and search window' : 'Search window',
      }),
      h('span', { class: 'osprey-kicker opt-out-right', text: `Start · best · ${endWord}` })
    ),
    ...variables.map((name) => {
      const meta = env?.variables?.[name] || null;
      const win = windowOf(bounds, config, name);
      const lo = meta ? meta.min_value : win?.min;
      const hi = meta ? meta.max_value : win?.max;
      const start = valueOf(figures.initial, name);
      const best = valueOf(figures.bestRecord, name);
      const now = onMachine ? valueOf(onMachine, name) : null;
      const last = valueOf(figures.lastRecord, name);
      const units = meta?.units || '';
      const { title } = knobLabel(name, meta);
      const values = [
        start === null ? null : ['start', start],
        best === null ? null : ['best', best],
        now !== null ? ['now', now] : last === null ? null : ['last', last],
      ].filter(Boolean);
      return h(
        'div',
        { class: 'opt-out-knob', 'data-knob': name },
        knobName(name, meta),
        typeof lo === 'number' && typeof hi === 'number'
          ? knobBar({
              lo,
              hi,
              window: meta ? win : null,
              marks: { start, best, current: now },
              units,
              label: `${title}: ${values.map((v) => `${v?.[0]} ${fmt(v?.[1])}`).join(', ')}`,
            })
          : h('span'),
        h(
          'div',
          { class: 'opt-out-knob-vals' },
          ...values.map((v) =>
            h('span', {
              class: v?.[0] === 'now' ? 'opt-out-strong' : null,
              text: `${v?.[0]} ${fmt(v?.[1])}${unit(units)}`,
            })
          )
        )
      );
    })
  );
}

/**
 * Each knob at each evaluation: a line in its search window (the band), the
 * start dashed, the best evaluation a vertical line. A knob with neither a
 * window nor a value read gets the band alone.
 *
 * @param {KnobsSource} source
 * @returns {HTMLElement|null}
 */
function traces({ variables, config, bounds, env, figures, records }) {
  const n = Math.max(records.length, figures.total ?? 0);
  if (records.length < 2) return null;
  const W = 100;
  const H = 24;
  const rows = variables.map((name) => {
    const values = records.map((r) => valueOf(r, name));
    const seen = /** @type {number[]} */ (values.filter((v) => v !== null));
    const range =
      windowOf(bounds, config, name) ||
      (seen.length ? { min: Math.min(...seen), max: Math.max(...seen) } : null);
    const lo = range?.min ?? 0;
    const span = range ? range.max - range.min || 1 : 1;
    const y = (/** @type {number} */ v) => Math.max(0, Math.min(H, H - ((v - lo) / span) * H));
    const x = (/** @type {number} */ i) => ((i + 0.5) / n) * W;
    const points = values
      .map((v, i) => (v === null ? null : `${x(i).toFixed(2)},${y(v).toFixed(2)}`))
      .filter(Boolean);
    const box = svg('svg', {
      class: 'opt-out-spark',
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: 'none',
      'aria-hidden': 'true',
    });
    box.append(svg('rect', { class: 'opt-out-spark-band', x: 0, y: 0, width: W, height: H }));
    const start = range ? valueOf(figures.initial, name) : null;
    if (start !== null) {
      box.append(
        svg('line', { class: 'opt-out-spark-start', x1: 0, x2: W, y1: y(start), y2: y(start) })
      );
    }
    if (figures.bestIndex !== null) {
      const bx = x(figures.bestIndex);
      box.append(svg('line', { class: 'opt-out-spark-best', x1: bx, x2: bx, y1: 0, y2: H }));
    }
    if (points.length > 1)
      box.append(svg('path', { class: 'opt-out-spark-line', d: `M${points.join('L')}` }));
    const meta = env?.variables?.[name] || null;
    const { title } = knobLabel(name, meta);
    const last = valueOf(figures.lastRecord, name);
    return h(
      'div',
      { class: 'opt-out-trace' },
      h('span', { class: 'opt-out-trace-name', title: name, text: title }),
      box,
      h('span', {
        class: 'opt-out-trace-val',
        text: last === null ? '' : `${fmt(last)}${unit(meta?.units)}`,
      })
    );
  });
  return h(
    'div',
    { class: 'opt-out-traces' },
    h(
      'div',
      { class: 'osprey-section-label opt-flush' },
      'Knobs at each evaluation',
      h('span', {
        class: 'osprey-aside',
        text: 'band = search window, dashed = start, line = best',
      })
    ),
    ...rows
  );
}

/**
 * The Knobs view: the rows and, under them, the knobs at each evaluation.
 *
 * @param {KnobsSource} source
 * @returns {HTMLElement}
 */
export function knobsView(source) {
  if (!source.variables.length) {
    return h('p', { class: 'opt-note', text: 'This run names no knob.' });
  }
  return h('div', { class: 'opt-out-scroll' }, knobRows(source), traces(source));
}

/**
 * The key to the Knobs view's drawings.
 *
 * @param {MachineState} machine
 * @param {boolean} withSafeRange
 * @returns {HTMLElement}
 */
export function knobsKey(machine, withSafeRange) {
  /** @type {import('./knob-bar.js').KnobKey[]} */
  const keys = withSafeRange ? ['track', 'window', 'start', 'best'] : ['start', 'best'];
  if (machine.values) keys.push('current');
  return knobLegend(keys);
}

/**
 * Rows of "before → after" per knob, with the change.
 *
 * @param {{ name: string, from: number, to: number, delta: number }[]} changes
 * @param {EnvironmentDetails|null} env
 * @returns {HTMLElement}
 */
export function changeRows(changes, env) {
  return h(
    'div',
    { class: 'opt-out-changes' },
    ...changes.map((c) => {
      const meta = env?.variables?.[c.name] || null;
      const { title, channel } = knobLabel(c.name, meta);
      return h(
        'div',
        {
          class: 'opt-out-change',
          'data-knob': c.name,
          'data-from': String(c.from),
          'data-to': String(c.to),
        },
        h('span', { class: 'opt-out-change-label', title, text: title }),
        h('span', {
          class: 'opt-out-change-range',
          text: `${fmt(c.from)} → ${fmt(c.to)}${unit(meta?.units)}`,
        }),
        h('span', { class: 'opt-out-change-raw', title: channel || '', text: channel || '' }),
        h('span', { class: 'opt-out-change-delta', text: signed(c.delta) })
      );
    })
  );
}
