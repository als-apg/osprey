// @ts-check
/**
 * figures — pure builders from the service's contract data plus a theme
 * snapshot to Plotly figures (`{data, layout}`).
 *
 * The render adapter is the only caller: it takes the theme snapshot
 * (`{...chartTheme(), colorway: chartSeries(), font}`) once per update, hands
 * it to a builder here and renders the result with one `Plotly.react`. Because
 * react replaces the layout, every layout built here carries the theme itself,
 * in nested form (no dotted keys), plus a `uirevision` naming its plot so the
 * operator's zoom survives updates.
 *
 * Conventions every builder keeps:
 * - traces are `scatter` lines (or bars for corrector steps); trace colours are
 *   never set, so the theme's `colorway` picks them;
 * - an index axis is the device's 1-based position in its `Snapshot.catalog`
 *   family (ring order), so a device absent from the data is a gap (`null`),
 *   never a shifted point;
 * - `customdata` names the device each point belongs to, which is how a click
 *   resolves to a device;
 * - units come from the raw buffer's header.
 *
 * This module performs no I/O and never touches Plotly.
 *
 * @module figures
 */

/**
 * @typedef {{x?: string|null, y?: string|null, sum?: string|null}} Units
 * @typedef {{shape: number[], trigger_sample: number, units: Units}} RawHeader
 * @typedef {{header: RawHeader, samples: ArrayLike<number>}} RawData
 * @typedef {[number, number]} Range
 * @typedef {{turns?: Range|null, sum?: Range|null, x?: Range|null, y?: Range|null}} Ranges
 * @typedef {{name: string, device?: string, label?: string, places?: {id: string, level: string|null}[], s?: number|null, plane?: string|null, groups?: string[]}} CatalogEntry
 * @typedef {{bpms: CatalogEntry[], hcms: CatalogEntry[], vcms: CatalogEntry[], groups?: any[]}} Catalog
 * @typedef {Record<string, any>} Theme
 * @typedef {{data: Record<string, any>[], layout: Record<string, any>}} Figure
 */

/**
 * The matrix names of one catalog family, in ring order; empty without a catalog.
 *
 * @param {Catalog|null|undefined} catalog
 * @param {'bpms'|'hcms'|'vcms'} family
 * @returns {string[]}
 */
export function catalogNames(catalog, family) {
  const entries = catalog?.[family];
  return Array.isArray(entries) ? entries.map((entry) => entry.name) : [];
}

/** The sample axis title of the Raw tab's plots. */
export const TURN_AXIS_TITLE = 'Index of turn';

/** Half-width of the default horizontal/vertical position range, in mm. */
export const POSITION_HALF_RANGE_MM = 10;

/** Millimetres per one of each unit a position plane may be stated in. */
const MM_PER_UNIT = Object.freeze({
  m: 1000,
  cm: 10,
  mm: 1,
  um: 1e-3,
  'µm': 1e-3,
  'μm': 1e-3,
  nm: 1e-6,
});

/** Trace name of a pseudo-orbit's averaged trajectory, which makes no step. */
const NO_STEP_NAME = 'Mean (no step)';

const PLANE_TITLES = Object.freeze({ sum: 'Raw Sum', x: 'Horizontal', y: 'Vertical' });

/**
 * The default position range, ±10 mm expressed in `unit`; `null` (autoscale)
 * when the unit is absent or not a length this module knows.
 *
 * @param {string|null|undefined} unit
 * @returns {Range|null}
 */
export function positionRange(unit) {
  if (typeof unit !== 'string') return null;
  const mmPer = /** @type {Record<string, number>} */ (MM_PER_UNIT)[unit.trim()];
  if (mmPer === undefined) return null;
  const half = Number((POSITION_HALF_RANGE_MM / mmPer).toPrecision(12));
  return [-half, half];
}

/**
 * @param {string} title
 * @param {string|null|undefined} unit
 */
function axisTitle(title, unit) {
  return { text: unit ? `${title} (${unit})` : title };
}

/**
 * An axis fragment: the theme's axis styling, a title and a range — a fixed
 * range when one is given, autoscale otherwise.
 *
 * @param {Theme} theme
 * @param {'xaxis'|'yaxis'} key
 * @param {{text: string}} title
 * @param {Range|null|undefined} range
 * @param {Record<string, any>} [extra]
 */
function axis(theme, key, title, range, extra = {}) {
  const fragment = { ...(theme[key] || {}), title, automargin: true, ...extra };
  return range ? { ...fragment, range: [range[0], range[1]], autorange: false } : { ...fragment, autorange: true };
}

/**
 * A layout carrying the theme in nested form.
 *
 * @param {Theme} theme
 * @param {string} uirevision
 * @param {Record<string, any>} xaxis
 * @param {Record<string, any>} yaxis
 * @param {Record<string, any>} [extra]
 */
function layout(theme, uirevision, xaxis, yaxis, extra = {}) {
  const rest = { ...(theme || {}) };
  delete rest.xaxis;
  delete rest.yaxis;
  const font = rest.font;
  return {
    ...rest,
    font: { ...(font || {}) },
    uirevision,
    showlegend: true,
    legend: { orientation: 'h' },
    margin: { l: 56, r: 16, t: 24, b: 40 },
    xaxis,
    yaxis,
    ...extra,
  };
}

/**
 * A finite number, or `null` for anything else (NaN, ±∞, missing).
 *
 * @param {unknown} value
 * @returns {number|null}
 */
function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The BPM axis: the catalog's BPMs in ring order, or the shot's own order
 * when the snapshot has no catalog yet.
 *
 * @param {any} shot
 * @param {Catalog|null|undefined} catalog
 * @returns {string[]}
 */
function bpmAxis(shot, catalog) {
  return catalog && Array.isArray(catalog.bpms) ? catalogNames(catalog, 'bpms') : [...(shot?.bpm_order || [])];
}

/**
 * Each BPM's `{x, y}` reading of a shot. The readings are keyed by RM row
 * name, in `bpm_order` order, so the BPM is the one at the same position.
 *
 * @param {any} shot
 * @returns {Map<string, {x: number|null, y: number|null}>}
 */
function shotReadings(shot) {
  /** @type {Map<string, {x: number|null, y: number|null}>} */
  const readings = new Map();
  if (!shot) return readings;
  const xs = Object.values(shot.x || {});
  const ys = Object.values(shot.y || {});
  (shot.bpm_order || []).forEach((/** @type {string} */ bpm, /** @type {number} */ i) => {
    readings.set(bpm, { x: finite(xs[i]), y: finite(ys[i]) });
  });
  return readings;
}

/**
 * The BPMs past the earliest per-injection loss in `bpm_order` — the first
 * lost BPM of any included injection and every BPM after it.
 *
 * @param {any} shot
 * @returns {Set<string>}
 */
function maskedBpms(shot) {
  const order = /** @type {string[]} */ (shot?.bpm_order || []);
  let earliest = order.length;
  for (const lostAt of shot?.per_shot_lost_at || []) {
    const at = lostAt == null ? -1 : order.indexOf(lostAt);
    if (at >= 0 && at < earliest) earliest = at;
  }
  return new Set(order.slice(earliest));
}

/**
 * One line trace over the BPM index axis.
 *
 * @param {string} name
 * @param {string[]} names
 * @param {(bpm: string) => number|null} value
 */
function bpmTrace(name, names, value) {
  return {
    type: 'scatter',
    mode: 'lines+markers',
    name,
    x: names.map((_, i) => i + 1),
    y: names.map(value),
    customdata: [...names],
    hovertemplate: '%{customdata} (%{x}): %{y}<extra></extra>',
  };
}

/**
 * The Raw tab's three figures — BPM Raw Sum, Horizontal and Vertical against
 * the sample index, the trigger sample marked. Each plot draws one trace per
 * BPM, the action's injections concatenated and joined by `null` separators.
 *
 * @param {RawData} raw - the decoded raw buffer of the last action
 * @param {any} shot - the shot it belongs to (its `bpm_order` names the rows)
 * @param {Theme} theme
 * @param {{ranges?: Ranges}} [options]
 * @returns {{sum: Figure, x: Figure, y: Figure}}
 */
export function rawFigures(raw, shot, theme, options = {}) {
  const ranges = options.ranges || {};
  const [nInj, , nBpm, nSamples] = raw.header.shape;
  const units = raw.header.units || {};
  const order = /** @type {string[]} */ (shot?.bpm_order || []);
  const turns = Array.from({ length: nSamples }, (_, t) => t);
  const trigger = raw.header.trigger_sample;
  const turnRange = ranges.turns || (nSamples > 0 ? /** @type {Range} */ ([0, nSamples - 1]) : null);

  /** @param {number} plane @param {'sum'|'x'|'y'} key @param {Range|null|undefined} range */
  const build = (plane, key, range) => {
    const data = [];
    for (let b = 0; b < nBpm; b++) {
      const name = order[b] ?? `BPM ${b + 1}`;
      /** @type {(number|null)[]} */ const x = [];
      /** @type {(number|null)[]} */ const y = [];
      /** @type {(string|null)[]} */ const customdata = [];
      for (let inj = 0; inj < nInj; inj++) {
        if (inj > 0) {
          x.push(null);
          y.push(null);
          customdata.push(null);
        }
        const base = ((inj * 3 + plane) * nBpm + b) * nSamples;
        for (let t = 0; t < nSamples; t++) {
          x.push(turns[t]);
          y.push(finite(raw.samples[base + t]));
          customdata.push(name);
        }
      }
      data.push({
        type: 'scatter',
        mode: 'lines',
        name,
        x,
        y,
        customdata,
        hovertemplate: '%{customdata} turn %{x}: %{y}<extra></extra>',
      });
    }
    const shapes = [{ type: 'line', xref: 'x', yref: 'paper', x0: trigger, x1: trigger, y0: 0, y1: 1, line: { dash: 'dot', width: 1 } }];
    return {
      data,
      layout: layout(
        theme,
        `raw-${key}`,
        axis(theme, 'xaxis', { text: TURN_AXIS_TITLE }, turnRange),
        axis(theme, 'yaxis', axisTitle(PLANE_TITLES[key], units[key]), range),
        { shapes, showlegend: false },
      ),
    };
  };

  return {
    sum: build(2, 'sum', ranges.sum),
    x: build(0, 'x', ranges.x || positionRange(units.x)),
    y: build(1, 'y', ranges.y || positionRange(units.y)),
  };
}

/**
 * @typedef {{units?: Units, ranges?: Ranges}} PlaneOptions
 */

/**
 * The Trajectories figures — x and y against the BPM index: the action's
 * average (masked from the earliest per-injection loss on) and the selected
 * reference's readings. A pseudo-orbit's average makes no step, and says so.
 *
 * @param {any} shot
 * @param {{catalog?: Catalog|null, reference?: any}} snapshot
 * @param {Theme} theme
 * @param {PlaneOptions} [options]
 * @returns {{x: Figure, y: Figure}}
 */
export function trajectoryFigures(shot, snapshot, theme, options = {}) {
  const names = bpmAxis(shot, snapshot?.catalog);
  const readings = shotReadings(shot);
  const masked = maskedBpms(shot);
  const reference = snapshot?.reference;
  const refReadings = reference?.readings || null;
  const measured = shot?.readout === 'pseudo_orbit' ? NO_STEP_NAME : 'Measurement';
  const units = options.units || {};
  const ranges = options.ranges || {};

  /** @param {'x'|'y'} plane */
  const build = (plane) => {
    const data = [
      bpmTrace(measured, names, (bpm) => (masked.has(bpm) ? null : readings.get(bpm)?.[plane] ?? null)),
    ];
    if (refReadings) {
      data.push(bpmTrace(`Reference ${reference.name}`, names, (bpm) => finite(refReadings[bpm]?.[plane])));
    }
    return {
      data,
      layout: layout(
        theme,
        `traj-${plane}`,
        axis(theme, 'xaxis', { text: 'BPM index' }, null),
        axis(theme, 'yaxis', axisTitle(plane, units[plane]), ranges[plane] || positionRange(units[plane])),
      ),
    };
  };
  return { x: build('x'), y: build('y') };
}

/**
 * The Difference to Reference figures — dx and dy against the BPM index,
 * measurement minus reference, masked like the trajectories. Both share one
 * symmetric y range so the two planes compare at a glance.
 *
 * @param {any} shot
 * @param {{catalog?: Catalog|null, reference?: any}} snapshot
 * @param {Theme} theme
 * @param {PlaneOptions} [options]
 * @returns {{x: Figure, y: Figure}}
 */
export function differenceFigures(shot, snapshot, theme, options = {}) {
  const names = bpmAxis(shot, snapshot?.catalog);
  const readings = shotReadings(shot);
  const masked = maskedBpms(shot);
  const refReadings = snapshot?.reference?.readings || {};
  const units = options.units || {};

  /** @param {'x'|'y'} plane @param {string} bpm */
  const delta = (plane, bpm) => {
    const measured = masked.has(bpm) ? null : readings.get(bpm)?.[plane] ?? null;
    const ref = finite(refReadings[bpm]?.[plane]);
    return measured === null || ref === null ? null : measured - ref;
  };
  const traces = {
    x: bpmTrace('dx', names, (bpm) => delta('x', bpm)),
    y: bpmTrace('dy', names, (bpm) => delta('y', bpm)),
  };
  let peak = 0;
  for (const trace of [traces.x, traces.y]) {
    for (const v of trace.y) if (v !== null) peak = Math.max(peak, Math.abs(v));
  }
  /** @type {Range|null} */
  const shared = peak > 0 ? [-peak * 1.1, peak * 1.1] : null;

  /** @param {'x'|'y'} plane */
  const build = (plane) => ({
    data: [traces[plane]],
    layout: layout(
      theme,
      `diff-${plane}`,
      axis(theme, 'xaxis', { text: 'BPM index' }, null),
      axis(theme, 'yaxis', axisTitle(`d${plane}`, units[plane]), shared || positionRange(units[plane])),
      { showlegend: false },
    ),
  });
  return { x: build('x'), y: build('y') };
}

/**
 * The SV Spectrum — the solve's singular values on a log axis, each point
 * carrying its spectrum kind.
 *
 * @param {any} solve - `Snapshot.solve`
 * @param {Theme} theme
 * @returns {Figure}
 */
export function svFigure(solve, theme) {
  const values = /** @type {unknown[]} */ (solve?.singular_values || []);
  const kinds = /** @type {string[]} */ (solve?.kind || []);
  return {
    data: [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Singular values',
        x: values.map((_, i) => i + 1),
        y: values.map(finite),
        customdata: values.map((_, i) => kinds[i] ?? null),
        hovertemplate: 'SV %{x}: %{y} %{customdata}<extra></extra>',
      },
    ],
    layout: layout(
      theme,
      'sv',
      axis(theme, 'xaxis', { text: 'Singular value index' }, null),
      axis(theme, 'yaxis', { text: 'Singular value' }, null, { type: 'log' }),
      { showlegend: false },
    ),
  };
}

/**
 * The CM step — HCM and VCM step bars against each family's corrector index.
 *
 * @param {any} step - `Snapshot.step` (its `deltas` by corrector name)
 * @param {Catalog|null|undefined} catalog
 * @param {Theme} theme
 * @returns {Figure}
 */
export function cmStepFigure(step, catalog, theme) {
  const deltas = step?.deltas || {};
  /** @param {string} name @param {string[]} correctors */
  const bars = (name, correctors) => ({
    type: 'bar',
    name,
    x: correctors.map((_, i) => i + 1),
    y: correctors.map((c) => finite(deltas[c])),
    customdata: [...correctors],
    hovertemplate: '%{customdata} (%{x}): %{y}<extra></extra>',
  });
  return {
    data: [bars('HCM', catalogNames(catalog, 'hcms')), bars('VCM', catalogNames(catalog, 'vcms'))],
    layout: layout(
      theme,
      'cm-step',
      axis(theme, 'xaxis', { text: 'Corrector index' }, null),
      axis(theme, 'yaxis', { text: 'Step' }, null),
      { barmode: 'group' },
    ),
  };
}

/**
 * A per-BPM setting readback against the BPM index. Values may be numbers or
 * a setting's option labels; unread devices are gaps.
 *
 * @param {any} readback - one `SettingReadback`
 * @param {Catalog|null|undefined} catalog
 * @param {Theme} theme
 * @param {string} title
 * @param {string} uirevision
 * @returns {Figure}
 */
function readbackFigure(readback, catalog, theme, title, uirevision) {
  const values = readback?.values || {};
  const names = catalog?.bpms ? catalogNames(catalog, 'bpms') : Object.keys(values);
  /** @param {string} bpm */
  const value = (bpm) => {
    const v = values[bpm];
    return typeof v === 'string' || typeof v === 'boolean' ? String(v) : finite(v);
  };
  return {
    data: [{ ...bpmTrace(title, names, /** @type {any} */ (value)), mode: 'markers' }],
    layout: layout(
      theme,
      uirevision,
      axis(theme, 'xaxis', { text: 'BPM index' }, null),
      axis(theme, 'yaxis', { text: title }, null),
      { showlegend: false },
    ),
  };
}

/**
 * The drawer's attenuation readbacks per BPM.
 *
 * @param {any} readback - the attenuation `SettingReadback`
 * @param {Catalog|null|undefined} catalog
 * @param {Theme} theme
 * @returns {Figure}
 */
export function attenuationFigure(readback, catalog, theme) {
  return readbackFigure(readback, catalog, theme, 'Attenuation', 'drawer-attenuation');
}

/**
 * The drawer's pilot-tone readbacks per BPM.
 *
 * @param {any} readback - the pilot-tone `SettingReadback`
 * @param {Catalog|null|undefined} catalog
 * @param {Theme} theme
 * @returns {Figure}
 */
export function pilotToneFigure(readback, catalog, theme) {
  return readbackFigure(readback, catalog, theme, 'Pilot tone', 'drawer-pilot-tone');
}
