// @ts-check
import { describe, expect, it } from 'vitest';

import {
  POSITION_HALF_RANGE_MM,
  TURN_AXIS_TITLE,
  attenuationFigure,
  MAX_DEVICE_TICKS,
  catalogNames,
  cmStepFigure,
  deviceAxis,
  differenceFigures,
  pilotToneFigure,
  positionRange,
  rawFigures,
  svFigure,
  trajectoryFigures,
} from './figures.js';

/** A theme snapshot as the render adapter hands it over (no hex literals). */
const THEME = Object.freeze({
  paper_bgcolor: 'rgb(10, 10, 10)',
  plot_bgcolor: 'rgb(20, 20, 20)',
  font: { color: 'rgb(200, 200, 200)', family: 'monospace' },
  xaxis: { gridcolor: 'rgb(60, 60, 60)' },
  yaxis: { gridcolor: 'rgb(60, 60, 60)' },
  colorway: ['red', 'blue', 'green'],
});

/** @param {string[]} names */
const entries = (names) => names.map((name, i) => ({ name, device: name, label: name, places: [], s: i, plane: null, groups: [] }));
const CATALOG = {
  bpms: entries(['B1', 'B2', 'B3', 'B4']),
  hcms: entries(['H1', 'H2', 'H3']),
  vcms: entries(['V1', 'V2']),
  groups: [],
};

/**
 * A raw buffer of `nInj` injections over BPMs B1, B2, B4 (B3 not in the shot),
 * `samples` turns each: value = inj*1000 + plane*100 + bpm*10 + t.
 */
/**
 * @param {number} [nInj]
 * @param {number} [samples]
 * @param {import('./figures.js').Units} [units]
 */
function raw(nInj = 2, samples = 4, units = { x: 'mm', y: 'mm', sum: null }) {
  const nBpm = 3;
  const data = new Float32Array(nInj * 3 * nBpm * samples);
  for (let inj = 0; inj < nInj; inj++) {
    for (let plane = 0; plane < 3; plane++) {
      for (let b = 0; b < nBpm; b++) {
        for (let t = 0; t < samples; t++) {
          data[((inj * 3 + plane) * nBpm + b) * samples + t] = inj * 1000 + plane * 100 + b * 10 + t;
        }
      }
    }
  }
  return { header: { shape: [nInj, 3, nBpm, samples], trigger_sample: 1, units }, samples: data };
}

/** A shot over B1, B2, B4 keyed by row names that differ from the BPM names. */
function shot(over = {}) {
  return {
    shot_id: 's1',
    model: 'm',
    x: { 'B1:x': 1, 'B2:x': 2, 'B4:x': 4 },
    y: { 'B1:y': -1, 'B2:y': -2, 'B4:y': -4 },
    sum: { 'B1:s': 1, 'B2:s': 1, 'B4:s': 1 },
    bpm_order: ['B1', 'B2', 'B4'],
    lost: [],
    last_bpm: 'B4',
    i0: 0,
    window: [0, 4],
    per_shot_lost_at: [null, null],
    shots_included: 2,
    shots_failed: [],
    commands_fired: [],
    readout: 'tbt',
    average: 'mean',
    raw_samples: 4,
    ...over,
  };
}

const SNAPSHOT = {
  catalog: CATALOG,
  reference: {
    name: 'golden',
    sha256: 'abc',
    matched: [],
    missing_in_reference: [],
    unused_in_reference: [],
    nan_count: 0,
    readout: 'tbt',
    readings: { B1: { x: 0.5, y: 0.5 }, B2: { x: 1, y: null }, B3: { x: 3, y: 3 }, B4: { x: 1, y: -1 } },
  },
};

/**
 * @param {any} value
 * @returns {boolean}
 */
function hasDottedKey(value) {
  if (value === null || typeof value !== 'object') return false;
  return Object.entries(value).some(([k, v]) /** @returns {boolean} */ => k.includes('.') || hasDottedKey(v));
}

/** @param {any} trace */
function noColor(trace) {
  return trace.line?.color === undefined && trace.marker?.color === undefined;
}

describe('positionRange', () => {
  it('is ±10 mm expressed in the header unit', () => {
    expect(POSITION_HALF_RANGE_MM).toBe(10);
    expect(positionRange('mm')).toEqual([-10, 10]);
    expect(positionRange('m')).toEqual([-0.01, 0.01]);
    expect(positionRange('um')).toEqual([-10000, 10000]);
    expect(positionRange('µm')).toEqual([-10000, 10000]);
  });

  it('autoscales when the unit is unknown or absent', () => {
    expect(positionRange(null)).toBeNull();
    expect(positionRange('furlong')).toBeNull();
  });
});

describe('rawFigures', () => {
  it('draws one trace per BPM per plot, injections joined by null separators', () => {
    const figs = rawFigures(raw(), shot(), THEME);
    for (const fig of [figs.sum, figs.x, figs.y]) {
      expect(fig.data).toHaveLength(3);
      expect(fig.data.map((t) => t.name)).toEqual(['B1', 'B2', 'B4']);
      for (const trace of fig.data) {
        expect(trace.type).toBe('scatter');
        expect(trace.mode).toBe('lines');
        expect(noColor(trace)).toBe(true);
      }
    }
    const b2x = figs.x.data[1];
    expect(b2x.x).toEqual([0, 1, 2, 3, null, 0, 1, 2, 3]);
    expect(b2x.y).toEqual([10, 11, 12, 13, null, 1010, 1011, 1012, 1013]);
    expect(b2x.customdata).toEqual(['B2', 'B2', 'B2', 'B2', null, 'B2', 'B2', 'B2', 'B2']);
    expect(figs.sum.data[0].y.slice(0, 4)).toEqual([200, 201, 202, 203]);
    expect(figs.y.data[2].y.slice(0, 4)).toEqual([120, 121, 122, 123]);
  });

  it('labels the sample axis and marks the trigger', () => {
    const fig = rawFigures(raw(), shot(), THEME).x;
    expect(fig.layout.xaxis.title.text).toBe(TURN_AXIS_TITLE);
    expect(TURN_AXIS_TITLE).toBe('Index of turn');
    expect(fig.layout.xaxis.range).toEqual([0, 3]);
    expect(fig.layout.shapes).toHaveLength(1);
    expect(fig.layout.shapes[0]).toMatchObject({ type: 'line', x0: 1, x1: 1, xref: 'x', yref: 'paper' });
  });

  it('defaults x/y to ±10 mm from the header units and autoscales the sum', () => {
    const figs = rawFigures(raw(2, 4, { x: 'um', y: 'mm', sum: 'counts' }), shot(), THEME);
    expect(figs.x.layout.yaxis.range).toEqual([-10000, 10000]);
    expect(figs.x.layout.yaxis.title.text).toBe('Horizontal (um)');
    expect(figs.y.layout.yaxis.range).toEqual([-10, 10]);
    expect(figs.sum.layout.yaxis.autorange).toBe(true);
    expect(figs.sum.layout.yaxis.range).toBeUndefined();
    expect(figs.sum.layout.yaxis.title.text).toBe('Raw Sum (counts)');
  });

  it('takes the operator ranges when given', () => {
    /** @type {import('./figures.js').Ranges} */
    const ranges = { turns: [1, 2], sum: [0, 5], x: [-1, 1], y: [-2, 2] };
    const figs = rawFigures(raw(), shot(), THEME, { ranges });
    expect(figs.sum.layout.yaxis.range).toEqual([0, 5]);
    expect(figs.sum.layout.yaxis.autorange).toBe(false);
    expect(figs.x.layout.yaxis.range).toEqual([-1, 1]);
    expect(figs.y.layout.yaxis.range).toEqual([-2, 2]);
    expect(figs.y.layout.xaxis.range).toEqual([1, 2]);
  });

  it('turns NaN samples into gaps', () => {
    const buffer = raw(1, 4);
    buffer.samples[1] = NaN;
    expect(rawFigures(buffer, shot(), THEME).x.data[0].y).toEqual([0, null, 2, 3]);
  });

  it('carries the theme in nested form, a uirevision, and no dotted key', () => {
    const fig = rawFigures(raw(), shot(), THEME).sum;
    expect(fig.layout.paper_bgcolor).toBe(THEME.paper_bgcolor);
    expect(fig.layout.font.color).toBe(THEME.font.color);
    expect(fig.layout.xaxis.gridcolor).toBe(THEME.xaxis.gridcolor);
    expect(fig.layout.colorway).toEqual(THEME.colorway);
    expect(fig.layout.uirevision).toBe('raw-sum');
    expect(hasDottedKey(fig.layout)).toBe(false);
    expect(THEME.xaxis).toEqual({ gridcolor: 'rgb(60, 60, 60)' });
  });
});

describe('trajectoryFigures', () => {
  it('plots measurement and reference against the family index with gaps', () => {
    const figs = trajectoryFigures(shot(), SNAPSHOT, THEME, { units: { x: 'mm', y: 'mm' } });
    const [meas, ref] = figs.x.data;
    expect(meas.name).toBe('Measurement');
    expect(meas.x).toEqual([1, 2, 3, 4]);
    expect(meas.y).toEqual([1, 2, null, 4]);
    expect(meas.customdata).toEqual(['B1', 'B2', 'B3', 'B4']);
    expect(ref.name).toBe('Reference golden');
    expect(ref.y).toEqual([0.5, 1, 3, 1]);
    expect(figs.y.data[0].y).toEqual([-1, -2, null, -4]);
    expect(figs.y.data[1].y).toEqual([0.5, null, 3, -1]);
    expect(figs.x.layout.yaxis.range).toEqual([-10, 10]);
    expect(figs.x.layout.xaxis.title.text).toBe('BPM');
    expect(figs.x.layout.xaxis.tickvals).toEqual([1, 2, 3, 4]);
    expect(figs.x.layout.xaxis.ticktext).toEqual(['B1', 'B2', 'B3', 'B4']);
    expect(figs.x.layout.uirevision).toBe('traj-x');
    expect(noColor(meas) && noColor(ref)).toBe(true);
  });

  it('masks the measurement from the earliest per-injection loss on', () => {
    const lossy = shot({ per_shot_lost_at: [null, 'B4', 'B2'] });
    const meas = trajectoryFigures(lossy, SNAPSHOT, THEME).x.data[0];
    expect(meas.y).toEqual([1, null, null, null]);
  });

  it('draws only the measurement without a reference', () => {
    const figs = trajectoryFigures(shot(), { catalog: CATALOG, reference: null }, THEME);
    expect(figs.x.data).toHaveLength(1);
  });

  it('labels a pseudo-orbit mean "no step"', () => {
    const figs = trajectoryFigures(shot({ readout: 'pseudo_orbit' }), SNAPSHOT, THEME);
    expect(figs.x.data[0].name).toBe('Mean (no step)');
    expect(figs.y.data[0].name).toBe('Mean (no step)');
  });

  it('puts the BPMs on the axis in the catalog order (position order)', () => {
    const ordered = { ...CATALOG, bpms: entries(['B4', 'B1', 'B3', 'B2']) };
    const meas = trajectoryFigures(shot(), { catalog: ordered, reference: null }, THEME).x.data[0];
    expect(meas.customdata).toEqual(['B4', 'B1', 'B3', 'B2']);
  });

  it('gaps the BPMs excluded in force and those the shot left out, in every trace', () => {
    const figs = trajectoryFigures(shot({ excluded_bpms: ['B4'] }), { ...SNAPSHOT, excluded: ['B1'] }, THEME);
    expect(figs.x.data[0].y).toEqual([null, 2, null, null]);
    expect(figs.x.data[1].y).toEqual([null, 1, 3, null]);
    const diff = differenceFigures(shot(), { ...SNAPSHOT, excluded: ['B2'] }, THEME);
    expect(diff.x.data[0].y).toEqual([0.5, null, null, 3]);
  });

  it('labels the ticks and hovers with label, place and s, and bands every other place', () => {
    const placed = (/** @type {string} */ name, /** @type {string} */ label, /** @type {string} */ place, /** @type {number} */ s) =>
      ({ name, device: name, label, places: [{ id: 'MACHINE', level: 'machine' }, { id: place, level: 'cell' }], s, plane: null, groups: [] });
    const catalog = {
      ...CATALOG,
      bpms: [placed('B1', 'Pick 1', 'C1', 1), placed('B2', 'Pick 2', 'C1', 2), placed('B3', 'Pick 3', 'C2', 3),
        placed('B4', 'Pick 4', 'C3', 4.25)],
    };
    const fig = trajectoryFigures(shot(), { catalog, reference: null }, THEME).x;
    expect(fig.layout.xaxis.ticktext).toEqual(['Pick 1', 'Pick 2', 'Pick 3', 'Pick 4']);
    expect(fig.data[0].text[3]).toBe('Pick 4 · C3 · s = 4.3 m');
    expect(fig.data[0].customdata).toEqual(['B1', 'B2', 'B3', 'B4']);
    expect(fig.layout.shapes).toEqual([expect.objectContaining({ type: 'rect', x0: 2.5, x1: 3.5, layer: 'below', name: 'C2' })]);
  });

  it('thins the tick labels to fit', () => {
    const names = Array.from({ length: 100 }, (_, i) => `N${i}`);
    const { tickvals, ticktext, hover } = deviceAxis(names, null);
    expect(tickvals.length).toBeLessThanOrEqual(MAX_DEVICE_TICKS);
    expect(tickvals.slice(0, 2)).toEqual([1, 6]);
    expect(ticktext[1]).toBe('N5');
    expect(hover).toHaveLength(100);
  });

  it('falls back to the shot order without a catalog', () => {
    const meas = trajectoryFigures(shot(), { catalog: null, reference: null }, THEME).x.data[0];
    expect(meas.x).toEqual([1, 2, 3]);
    expect(meas.customdata).toEqual(['B1', 'B2', 'B4']);
  });
});

describe('catalogNames', () => {
  it('lists a family by matrix name, empty without a catalog', () => {
    expect(catalogNames(CATALOG, 'hcms')).toEqual(['H1', 'H2', 'H3']);
    expect(catalogNames(null, 'bpms')).toEqual([]);
  });
});

describe('differenceFigures', () => {
  it('computes measurement minus reference on a shared symmetric axis', () => {
    const figs = differenceFigures(shot(), SNAPSHOT, THEME, { units: { x: 'mm', y: 'mm' } });
    expect(figs.x.data[0].y).toEqual([0.5, 1, null, 3]);
    expect(figs.y.data[0].y).toEqual([-1.5, null, null, -3]);
    expect(figs.x.data[0].customdata).toEqual(['B1', 'B2', 'B3', 'B4']);
    expect(figs.x.layout.yaxis.range).toEqual(figs.y.layout.yaxis.range);
    const [lo, hi] = figs.x.layout.yaxis.range;
    expect(lo).toBe(-hi);
    expect(hi).toBeGreaterThanOrEqual(3);
    expect(figs.x.layout.yaxis.title.text).toBe('dx (mm)');
    expect(figs.y.layout.yaxis.title.text).toBe('dy (mm)');
  });

  it('masks past the earliest loss and is empty without a reference', () => {
    const lossy = shot({ per_shot_lost_at: ['B4'] });
    expect(differenceFigures(lossy, SNAPSHOT, THEME).x.data[0].y).toEqual([0.5, 1, null, null]);
    const none = differenceFigures(shot(), { catalog: CATALOG, reference: null }, THEME);
    expect(none.x.data[0].y).toEqual([null, null, null, null]);
  });
});

describe('svFigure', () => {
  it('plots the singular values on a log axis', () => {
    const fig = svFigure({ singular_values: [10, 1, 0.1], kind: ['LIVE', 'LIVE', 'DISCARDED'] }, THEME);
    expect(fig.data[0].x).toEqual([1, 2, 3]);
    expect(fig.data[0].y).toEqual([10, 1, 0.1]);
    expect(fig.data[0].customdata).toEqual(['LIVE', 'LIVE', 'DISCARDED']);
    expect(fig.layout.yaxis.type).toBe('log');
    expect(noColor(fig.data[0])).toBe(true);
  });

  it('is empty without a solve', () => {
    expect(svFigure(null, THEME).data[0].y).toEqual([]);
  });
});

describe('cmStepFigure', () => {
  it('draws HCM and VCM bars at their correctors in position order with device names', () => {
    const fig = cmStepFigure({ deltas: { H1: 0.1, H3: -0.2, V2: 0.3, X9: 1 } }, CATALOG, THEME);
    const [h, v] = fig.data;
    expect(h).toMatchObject({ type: 'bar', name: 'HCM' });
    // s: H1 0, V1 0, H2 1, V2 1, H3 2 — horizontal first at one position.
    expect(h.x).toEqual([1, 3, 5]);
    expect(h.y).toEqual([0.1, null, -0.2]);
    expect(h.customdata).toEqual(['H1', 'H2', 'H3']);
    expect(v.x).toEqual([2, 4]);
    expect(v.y).toEqual([null, 0.3]);
    expect(v.customdata).toEqual(['V1', 'V2']);
    expect(fig.layout.xaxis.title.text).toBe('Corrector');
    expect(fig.layout.xaxis.ticktext).toEqual(['H1', 'V1', 'H2', 'V2', 'H3']);
    expect(noColor(h) && noColor(v)).toBe(true);
  });

  it('is empty without a step', () => {
    const fig = cmStepFigure(null, CATALOG, THEME);
    expect(fig.data[0].y).toEqual([null, null, null]);
  });
});

describe('drawer builders', () => {
  const att = { id: 'attenuation', scope: 'device', value: null, values: { B1: 12, B2: null, B4: 6 } };
  const pt = { id: 'pilot_tone', scope: 'device', value: null, values: { B1: 'Off', B3: 'On' } };

  it('plots the attenuation readback per BPM index with gaps', () => {
    const fig = attenuationFigure(att, CATALOG, THEME);
    expect(fig.data[0].x).toEqual([1, 2, 3, 4]);
    expect(fig.data[0].y).toEqual([12, null, null, 6]);
    expect(fig.data[0].customdata).toEqual(['B1', 'B2', 'B3', 'B4']);
    expect(fig.layout.uirevision).toBe('drawer-attenuation');
  });

  it('plots the pilot-tone readback, categorical values kept', () => {
    const fig = pilotToneFigure(pt, CATALOG, THEME);
    expect(fig.data[0].y).toEqual(['Off', null, 'On', null]);
    expect(fig.layout.uirevision).toBe('drawer-pilot-tone');
  });

  it('is empty without a readback', () => {
    expect(attenuationFigure(null, CATALOG, THEME).data[0].y).toEqual([null, null, null, null]);
  });
});
