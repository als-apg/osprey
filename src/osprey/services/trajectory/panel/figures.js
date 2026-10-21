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
 * - a device axis is the device's 1-based position in ring order along its
 *   `Snapshot.catalog` family, so a device absent from the data is a gap
 *   (`null`), never a shifted point; its ticks are the devices' labels,
 *   thinned to fit, and every other run of devices sharing a leaf place is
 *   shaded by a band behind the traces;
 * - `customdata` names the device each point belongs to, which is how a click
 *   resolves to a device; the hover shows its label, place and `s`;
 * - an excluded BPM is a gap in every trace over the BPM axis;
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

/** Most device labels a device axis shows as ticks; more are thinned to every n-th. */
export const MAX_DEVICE_TICKS = 24;

/** The place bands' fill: a neutral grey at low opacity, readable on every theme. */
const BAND_FILL = Object.freeze({ fillcolor: 'gray', opacity: 0.12 });

/**
 * Every catalog entry by matrix name.
 *
 * @param {Catalog|null|undefined} catalog
 * @returns {Map<string, CatalogEntry>}
 */
function entriesByName(catalog) {
  /** @type {Map<string, CatalogEntry>} */
  const out = new Map();
  for (const family of /** @type {const} */ (['bpms', 'hcms', 'vcms'])) {
    const entries = catalog?.[family];
    if (Array.isArray(entries)) for (const entry of entries) out.set(entry.name, entry);
  }
  return out;
}

/**
 * A device's leaf place id — its own place — or null when it has none.
 *
 * @param {CatalogEntry|undefined} entry
 * @returns {string|null}
 */
export function leafOf(entry) {
  const places = entry?.places;
  return Array.isArray(places) && places.length ? places[places.length - 1].id : null;
}

/** The row label of devices without a place when no model is named. */
export const NO_PLACE = 'no place';

/**
 * Devices grouped by leaf place: one row per place in ring order (by the
 * first device of each along the ring), devices without a place on one row
 * labelled *model*. Rows keep the devices' given order.
 *
 * @template {CatalogEntry} T
 * @param {readonly T[]} entries
 * @param {string|null|undefined} model
 * @returns {{id: string, label: string, entries: T[]}[]}
 */
export function placeRows(entries, model) {
  /** @type {Map<string, {row: {id: string, label: string, entries: T[]}, s: number, at: number}>} */
  const rows = new Map();
  entries.forEach((entry, at) => {
    const place = leafOf(entry);
    const s = typeof entry.s === 'number' ? entry.s : Infinity;
    const found = rows.get(place ?? '');
    if (!found) rows.set(place ?? '', { row: { id: place ?? '', label: place ?? (model || NO_PLACE), entries: [entry] }, s, at });
    else {
      found.row.entries.push(entry);
      if (s < found.s) Object.assign(found, { s, at });
    }
  });
  return [...rows.values()].sort((a, b) => a.s - b.s || a.at - b.at).map(({ row }) => row);
}

/**
 * A device's hover text: label, leaf place and `s`, as far as the catalog knows them.
 *
 * @param {string} name
 * @param {CatalogEntry|undefined} entry
 * @returns {string}
 */
export function deviceHover(name, entry) {
  const parts = [entry?.label || name];
  const place = leafOf(entry);
  if (place) parts.push(place);
  if (typeof entry?.s === 'number') parts.push(`s = ${entry.s.toFixed(1)} m`);
  return parts.join(' · ');
}

/**
 * A device axis over *names* (positions 1..n): its label ticks thinned to at
 * most {@link MAX_DEVICE_TICKS}, a hover text per device, and a band behind
 * every other run of devices sharing a leaf place.
 *
 * @param {string[]} names
 * @param {Catalog|null|undefined} catalog
 * @returns {{tickvals: number[], ticktext: string[], hover: string[], shapes: Record<string, any>[]}}
 */
export function deviceAxis(names, catalog) {
  const byName = entriesByName(catalog);
  const step = Math.max(1, Math.ceil(names.length / MAX_DEVICE_TICKS));
  /** @type {number[]} */ const tickvals = [];
  /** @type {string[]} */ const ticktext = [];
  names.forEach((name, i) => {
    if (i % step !== 0) return;
    tickvals.push(i + 1);
    ticktext.push(byName.get(name)?.label || name);
  });
  /** @type {Record<string, any>[]} */
  const shapes = [];
  let run = 0;
  for (let start = 0; start < names.length; ) {
    const place = leafOf(byName.get(names[start]));
    let end = start;
    while (end + 1 < names.length && leafOf(byName.get(names[end + 1])) === place) end++;
    if (place != null && run % 2 === 1) {
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper', x0: start + 0.5, x1: end + 1.5, y0: 0, y1: 1,
        layer: 'below', line: { width: 0 }, ...BAND_FILL, name: place,
      });
    }
    if (place != null) run++;
    start = end + 1;
  }
  return { tickvals, ticktext, hover: names.map((name) => deviceHover(name, byName.get(name))), shapes };
}

/**
 * The x-axis fragment of a device axis.
 *
 * @param {Theme} theme
 * @param {string} title
 * @param {{tickvals: number[], ticktext: string[]}} ticks
 */
function deviceXAxis(theme, title, ticks) {
  return axis(theme, 'xaxis', { text: title }, null, { tickmode: 'array', tickvals: ticks.tickvals, ticktext: ticks.ticktext });
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
 * One line trace over the BPM axis, excluded BPMs as gaps.
 *
 * @param {string} name
 * @param {string[]} names
 * @param {(bpm: string) => number|null} value
 * @param {string[]} hover
 * @param {Set<string>} excluded
 */
function bpmTrace(name, names, value, hover, excluded) {
  return {
    type: 'scatter',
    mode: 'lines+markers',
    name,
    x: names.map((_, i) => i + 1),
    y: names.map((bpm) => (excluded.has(bpm) ? null : value(bpm))),
    customdata: [...names],
    text: [...hover],
    hovertemplate: '%{text}: %{y}<extra></extra>',
  };
}

/**
 * The BPMs out of a plot: those the shot left out and those excluded in force.
 *
 * @param {any} shot
 * @param {{excluded?: string[]|null}} snapshot
 * @returns {Set<string>}
 */
function excludedBpms(shot, snapshot) {
  return new Set([
    ...(Array.isArray(shot?.excluded_bpms) ? shot.excluded_bpms : []),
    ...(Array.isArray(snapshot?.excluded) ? snapshot.excluded : []),
  ]);
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
 * The Trajectories figures — x and y against the BPMs in ring order: the
 * action's average (masked from the earliest per-injection loss on) and the
 * selected reference's readings, excluded BPMs as gaps in both. A
 * pseudo-orbit's average makes no step, and says so.
 *
 * @param {any} shot
 * @param {{catalog?: Catalog|null, reference?: any, excluded?: string[]|null}} snapshot
 * @param {Theme} theme
 * @param {PlaneOptions} [options]
 * @returns {{x: Figure, y: Figure}}
 */
export function trajectoryFigures(shot, snapshot, theme, options = {}) {
  const names = bpmAxis(shot, snapshot?.catalog);
  const ticks = deviceAxis(names, snapshot?.catalog);
  const excluded = excludedBpms(shot, snapshot);
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
      bpmTrace(measured, names, (bpm) => (masked.has(bpm) ? null : readings.get(bpm)?.[plane] ?? null),
        ticks.hover, excluded),
    ];
    if (refReadings) {
      data.push(bpmTrace(`Reference ${reference.name}`, names, (bpm) => finite(refReadings[bpm]?.[plane]),
        ticks.hover, excluded));
    }
    return {
      data,
      layout: layout(
        theme,
        `traj-${plane}`,
        deviceXAxis(theme, 'BPM', ticks),
        axis(theme, 'yaxis', axisTitle(plane, units[plane]), ranges[plane] || positionRange(units[plane])),
        { shapes: ticks.shapes },
      ),
    };
  };
  return { x: build('x'), y: build('y') };
}

/**
 * The Difference to Reference figures — dx and dy against the BPMs in ring
 * order, measurement minus reference, masked and gapped like the
 * trajectories. Both share one symmetric y range so the two planes compare at
 * a glance.
 *
 * @param {any} shot
 * @param {{catalog?: Catalog|null, reference?: any, excluded?: string[]|null}} snapshot
 * @param {Theme} theme
 * @param {PlaneOptions} [options]
 * @returns {{x: Figure, y: Figure}}
 */
export function differenceFigures(shot, snapshot, theme, options = {}) {
  const names = bpmAxis(shot, snapshot?.catalog);
  const ticks = deviceAxis(names, snapshot?.catalog);
  const excluded = excludedBpms(shot, snapshot);
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
    x: bpmTrace('dx', names, (bpm) => delta('x', bpm), ticks.hover, excluded),
    y: bpmTrace('dy', names, (bpm) => delta('y', bpm), ticks.hover, excluded),
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
      deviceXAxis(theme, 'BPM', ticks),
      axis(theme, 'yaxis', axisTitle(`d${plane}`, units[plane]), shared || positionRange(units[plane])),
      { showlegend: false, shapes: ticks.shapes },
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
 * The correctors of both planes in ring order: by `s`, horizontal before
 * vertical at one position, a corrector without `s` after the placed ones.
 *
 * @param {Catalog|null|undefined} catalog
 * @returns {{name: string, family: 'hcms'|'vcms'}[]}
 */
function correctorAxis(catalog) {
  const all = /** @type {const} */ (['hcms', 'vcms']).flatMap((family, rank) => {
    const entries = catalog?.[family];
    return Array.isArray(entries)
      ? entries.map((entry, index) => ({ name: entry.name, family, s: typeof entry.s === 'number' ? entry.s : null, rank, index }))
      : [];
  });
  all.sort((a, b) => (a.s === null ? 1 : 0) - (b.s === null ? 1 : 0)
    || (a.s ?? 0) - (b.s ?? 0) || a.rank - b.rank || a.index - b.index);
  return all.map(({ name, family }) => ({ name, family }));
}

/**
 * The CM step — HCM and VCM step bars against the correctors in ring order,
 * each bar at its own corrector.
 *
 * @param {any} step - `Snapshot.step` (its `deltas` by corrector name)
 * @param {Catalog|null|undefined} catalog
 * @param {Theme} theme
 * @returns {Figure}
 */
export function cmStepFigure(step, catalog, theme) {
  const deltas = step?.deltas || {};
  const order = correctorAxis(catalog);
  const names = order.map((corrector) => corrector.name);
  const ticks = deviceAxis(names, catalog);
  /** @param {string} name @param {'hcms'|'vcms'} family */
  const bars = (name, family) => {
    const at = order.flatMap((corrector, i) => (corrector.family === family ? [i] : []));
    return {
      type: 'bar',
      name,
      x: at.map((i) => i + 1),
      y: at.map((i) => finite(deltas[names[i]])),
      customdata: at.map((i) => names[i]),
      text: at.map((i) => ticks.hover[i]),
      textposition: 'none',
      hovertemplate: '%{text}: %{y}<extra></extra>',
    };
  };
  return {
    data: [bars('HCM', 'hcms'), bars('VCM', 'vcms')],
    layout: layout(
      theme,
      'cm-step',
      deviceXAxis(theme, 'Corrector', ticks),
      axis(theme, 'yaxis', { text: 'Step' }, null),
      { barmode: 'overlay', shapes: ticks.shapes },
    ),
  };
}

/**
 * A per-BPM setting readback against the BPMs in ring order. Values may be numbers or
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
  const ticks = deviceAxis(names, catalog);
  return {
    data: [{ ...bpmTrace(title, names, /** @type {any} */ (value), ticks.hover, new Set()), mode: 'markers' }],
    layout: layout(
      theme,
      uirevision,
      deviceXAxis(theme, 'BPM', ticks),
      axis(theme, 'yaxis', { text: title }, null),
      { showlegend: false, shapes: ticks.shapes },
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
