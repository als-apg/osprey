// @ts-check
import { describe, expect, it, vi } from 'vitest';

import { RANGE_INPUTS, RAW_PLOTS, TRAJECTORY_PLOTS, mount, parseRange, trajectoryKey } from './plots.js';
import { createView } from './views.js';

const RAW_IDS = Object.values(RAW_PLOTS);
const TRAJ_IDS = Object.values(TRAJECTORY_PLOTS);

/** @param {string} id @param {Record<string, unknown>} [over] */
function shot(id, over = {}) {
  return {
    shot_id: id,
    parent_id: null,
    readout: 'tbt',
    fresh: true,
    bpm_order: ['B1', 'B2'],
    x: { B1: 0.1, B2: 0.2 },
    y: { B1: -0.1, B2: -0.2 },
    ...over,
  };
}

function raw() {
  return {
    header: { shape: [1, 3, 2, 4], trigger_sample: 1, units: { x: 'mm', y: 'mm', sum: 'counts' } },
    samples: new Float32Array(24).map((_, i) => i),
  };
}

/** @param {Record<string, unknown>} [over] */
function snapshot(over = {}) {
  return {
    holder: null,
    catalog: { bpms: [{ name: 'B1' }, { name: 'B2' }], hcms: [], vcms: [], groups: [] },
    reference: null,
    solve: { sha: 's1', config: { excluded_bpms: [], excluded_cms: [], alpha: 0.1 } },
    ...over,
  };
}

/** Resolve every pending promise chain. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * @param {{shots?: any[], rawResult?: any}} [options]
 */
function harness({ shots = [], rawResult } = {}) {
  /** @type {Map<string, Set<(p: any) => void>>} */
  const events = new Map();
  /** @type {Set<(s: any) => void>} */
  const snapshots = new Set();
  let current = /** @type {any} */ (null);
  const api = {
    shots: vi.fn(async () => ({ kind: 'ok', status: 200, body: { model: 'SR', shots } })),
    shotRaw: vi.fn(async () => rawResult ?? { kind: 'ok', status: 200, body: raw() }),
  };
  /** @type {string[]} */
  const rendered = [];
  /** @type {any[]} */
  const builds = [];
  const adapter = {
    renderGroup: vi.fn((/** @type {Record<string, string>} */ ids, /** @type {any} */ build) => {
      const figures = build({}, { ranges: {} });
      builds.push(figures);
      for (const [key, id] of Object.entries(ids)) if (figures[key]) rendered.push(id);
      return true;
    }),
    render: vi.fn(),
    setRanges: vi.fn(),
  };
  const view = createView();
  const notify = vi.fn();
  const host = document.createElement('div');
  const ctx = /** @type {any} */ ({
    api,
    adapter,
    view,
    notify,
    snapshot: () => current,
    onSnapshot: (/** @type {any} */ fn) => {
      snapshots.add(fn);
      return () => snapshots.delete(fn);
    },
    onEvent: (/** @type {string} */ type, /** @type {any} */ fn) => {
      const set = events.get(type) ?? new Set();
      events.set(type, set);
      set.add(fn);
      return () => set.delete(fn);
    },
    holding: () => true,
    area: () => null,
  });
  mount(host, ctx);
  return {
    api, adapter, view, notify, host, rendered, builds,
    /** @param {string} type @param {any} payload */
    emit(type, payload) {
      for (const fn of events.get(type) ?? []) fn(payload);
    },
    /** @param {any} snap */
    push(snap) {
      current = snap;
      for (const fn of snapshots) fn(snap);
    },
    clear() {
      rendered.length = 0;
      builds.length = 0;
      api.shots.mockClear();
      api.shotRaw.mockClear();
      adapter.renderGroup.mockClear();
    },
  };
}

/** A harness past its first snapshot, with one shot drawn. */
async function primed(over = {}) {
  const t = harness({ shots: [shot('s0')], ...over });
  t.push(snapshot());
  await flush();
  t.clear();
  return t;
}

describe('a shot event', () => {
  it('fetches its raw buffer once and draws the three raw plots and the four', async () => {
    const t = await primed();
    t.emit('shot', { type: 'shot', shot: shot('s1') });
    await flush();
    expect(t.api.shotRaw).toHaveBeenCalledTimes(1);
    expect(t.api.shotRaw).toHaveBeenCalledWith('s1');
    expect(t.api.shots).not.toHaveBeenCalled();
    expect(t.rendered.sort()).toEqual([...RAW_IDS, ...TRAJ_IDS].sort());
  });

  it('draws nothing for a shot_id already drawn', async () => {
    const t = await primed();
    t.emit('shot', { type: 'shot', shot: shot('s1') });
    await flush();
    t.clear();
    t.emit('shot', { type: 'shot', shot: shot('s1') });
    await flush();
    expect(t.api.shotRaw).not.toHaveBeenCalled();
    expect(t.adapter.renderGroup).not.toHaveBeenCalled();
  });

  it('draws nothing for the shot the list head already drew', async () => {
    const t = await primed();
    t.emit('shot', { type: 'shot', shot: shot('s0') });
    await flush();
    expect(t.api.shotRaw).not.toHaveBeenCalled();
    expect(t.adapter.renderGroup).not.toHaveBeenCalled();
  });

  it('draws only the four while the Raw tab plot is off', async () => {
    const t = await primed();
    t.view.set('plotRaw', false);
    t.emit('shot', { type: 'shot', shot: shot('s1') });
    await flush();
    expect(t.api.shotRaw).toHaveBeenCalledTimes(1);
    expect(t.rendered.sort()).toEqual([...TRAJ_IDS].sort());
  });

  it('draws the four from the buffer units and the shot readings', async () => {
    const t = await primed();
    t.emit('shot', { type: 'shot', shot: shot('s1') });
    await flush();
    const four = t.builds.find((f) => f.trajX);
    expect(four.trajX.data[0].y).toEqual([0.1, 0.2]);
    expect(four.trajX.layout.yaxis.title.text).toContain('mm');
  });

  it('keeps only the newest shot when buffers answer out of order', async () => {
    const t = await primed();
    /** @type {((v: any) => void)[]} */
    const answers = [];
    t.api.shotRaw.mockImplementation(() => new Promise((resolve) => answers.push(resolve)));
    t.emit('shot', { type: 'shot', shot: shot('s1') });
    t.emit('shot', { type: 'shot', shot: shot('s2', { x: { B1: 9, B2: 9 } }) });
    answers[1]({ kind: 'ok', status: 200, body: raw() });
    await flush();
    answers[0]({ kind: 'ok', status: 200, body: raw() });
    await flush();
    const fours = t.builds.filter((f) => f.trajX);
    expect(fours).toHaveLength(1);
    expect(fours[0].trajX.data[0].y).toEqual([9, 9]);
  });

  it('still draws the four and reports the refusal when the buffer is gone', async () => {
    const refused = { kind: 'refused', status: 410, code: 'raw_evicted', detail: 'gone', moved: null, extras: {} };
    const t = await primed({ rawResult: refused });
    t.emit('shot', { type: 'shot', shot: shot('s1') });
    await flush();
    expect(t.notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'refused', code: 'raw_evicted' }));
    expect(t.rendered.sort()).toEqual([...TRAJ_IDS].sort());
  });
});

describe('snapshots', () => {
  it('the first fetches one list and the head raw buffer, and draws the head', async () => {
    const t = harness({ shots: [shot('new'), shot('old')] });
    t.push(snapshot());
    await flush();
    expect(t.api.shots).toHaveBeenCalledTimes(1);
    expect(t.api.shotRaw).toHaveBeenCalledTimes(1);
    expect(t.api.shotRaw).toHaveBeenCalledWith('new');
    expect(t.rendered.sort()).toEqual([...RAW_IDS, ...TRAJ_IDS].sort());
  });

  it('the first with no stored shot draws nothing', async () => {
    const t = harness({ shots: [] });
    t.push(snapshot());
    await flush();
    expect(t.api.shots).toHaveBeenCalledTimes(1);
    expect(t.api.shotRaw).not.toHaveBeenCalled();
    expect(t.adapter.renderGroup).not.toHaveBeenCalled();
  });

  it('a holder change fetches the head again', async () => {
    const t = await primed();
    t.push(snapshot({ holder: { owner: 'alice', kind: 'panel' } }));
    await flush();
    expect(t.api.shots).toHaveBeenCalledTimes(1);
    expect(t.api.shotRaw).toHaveBeenCalledTimes(1);
  });

  it('a reference-only change draws the four without a fetch', async () => {
    const t = await primed();
    t.push(snapshot({ reference: { name: 'golden', sha256: 'abc', readings: { B1: { x: 0, y: 0 } } } }));
    await flush();
    expect(t.api.shots).not.toHaveBeenCalled();
    expect(t.api.shotRaw).not.toHaveBeenCalled();
    expect(t.rendered.sort()).toEqual([...TRAJ_IDS].sort());
    expect(t.builds[0].trajX.data.map((/** @type {any} */ d) => d.name)).toContain('Reference golden');
  });

  it('a reference sha change alone draws the four', async () => {
    const t = await primed();
    t.push(snapshot({ reference: { name: 'golden', sha256: 'abc', readings: {} } }));
    t.clear();
    t.push(snapshot({ reference: { name: 'golden', sha256: 'def', readings: {} } }));
    await flush();
    expect(t.rendered.sort()).toEqual([...TRAJ_IDS].sort());
  });

  it('an exclusion change draws the four without a fetch', async () => {
    const t = await primed();
    t.push(snapshot({ solve: { sha: 's2', config: { excluded_bpms: ['B1'], excluded_cms: [], alpha: 0.1 } } }));
    t.push(snapshot({ solve: { sha: 's3', config: { excluded_bpms: ['B1'], excluded_cms: ['H1'], alpha: 0.1 } } }));
    await flush();
    expect(t.api.shots).not.toHaveBeenCalled();
    expect(t.api.shotRaw).not.toHaveBeenCalled();
    expect(t.rendered).toHaveLength(2 * TRAJ_IDS.length);
  });

  it('a solve-only change draws nothing', async () => {
    const t = await primed();
    t.push(snapshot({ solve: { sha: 's9', config: { excluded_bpms: [], excluded_cms: [], alpha: 0.5 } } }));
    await flush();
    expect(t.api.shots).not.toHaveBeenCalled();
    expect(t.api.shotRaw).not.toHaveBeenCalled();
    expect(t.adapter.renderGroup).not.toHaveBeenCalled();
  });
});

describe('redraw and ranges', () => {
  it('a redraw request redraws from the last buffer without a fetch', async () => {
    const t = await primed();
    t.view.requestRedraw();
    await flush();
    expect(t.api.shotRaw).not.toHaveBeenCalled();
    expect(t.rendered.sort()).toEqual([...RAW_IDS, ...TRAJ_IDS].sort());
  });

  it('a redraw request before any shot draws nothing', () => {
    const t = harness();
    t.view.requestRedraw();
    expect(t.adapter.renderGroup).not.toHaveBeenCalled();
  });

  it('renders the four range inputs into its host', () => {
    const t = harness();
    for (const { name } of RANGE_INPUTS) {
      expect(t.host.querySelector(`input[data-control="${name}"]`)).not.toBeNull();
    }
  });

  it('a range edit sets the adapter ranges without a fetch', async () => {
    const t = await primed();
    for (const [name, text, key, range] of /** @type {const} */ ([
      ['range-turns', '0, 100', 'turns', [0, 100]],
      ['range-sum', '5 1', 'sum', [1, 5]],
      ['range-x', '-2:2', 'x', [-2, 2]],
      ['range-y', '', 'y', null],
    ])) {
      const input = /** @type {HTMLInputElement} */ (t.host.querySelector(`[data-control="${name}"]`));
      input.value = text;
      input.dispatchEvent(new Event('change'));
      expect(t.adapter.setRanges).toHaveBeenLastCalledWith({ [key]: range });
    }
    await flush();
    expect(t.api.shots).not.toHaveBeenCalled();
    expect(t.api.shotRaw).not.toHaveBeenCalled();
  });

  it('an invalid range is marked and not applied', () => {
    const t = harness();
    const input = /** @type {HTMLInputElement} */ (t.host.querySelector('[data-control="range-x"]'));
    input.value = 'wide';
    input.dispatchEvent(new Event('change'));
    expect(t.adapter.setRanges).not.toHaveBeenCalled();
    expect(input.getAttribute('aria-invalid')).toBe('true');
  });
});

describe('parseRange', () => {
  it.each([
    ['', null],
    ['  ', null],
    ['1,2', [1, 2]],
    ['2 , 1', [1, 2]],
    ['-1e-3 1e-3', [-0.001, 0.001]],
    ['1', undefined],
    ['1,1', undefined],
    ['1,2,3', undefined],
    ['a,b', undefined],
  ])('%j -> %j', (text, expected) => {
    expect(parseRange(text)).toEqual(expected);
  });
});

describe('trajectoryKey', () => {
  it('ignores solve fields other than the exclusions', () => {
    const a = snapshot();
    const b = snapshot({ solve: { sha: 'x', config: { excluded_bpms: [], excluded_cms: [], alpha: 9 } } });
    expect(trajectoryKey(a)).toBe(trajectoryKey(b));
  });
});
