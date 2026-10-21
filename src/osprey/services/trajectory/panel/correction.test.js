// @ts-check
import { describe, expect, it, vi } from 'vitest';

import {
  BPM_SET_CHANGED,
  BUSY,
  MACHINE_MOVED,
  NO_SESSION,
  NO_STEP,
  STEP_CHANGED,
  ZERO_STEP,
  applyReason,
  applyResultView,
  emptyKnobDrafts,
  mount,
  putBackReason,
  rmStatus,
  sameNames,
  seededKnobs,
  solvePartial,
} from './correction.js';
import { createView } from './views.js';

/** @param {string[]} names */
const entries = (names) => names.map((name, i) => ({ name, device: name, label: name, places: [], s: i, plane: null, groups: [] }));
const CATALOG = { bpms: entries(['B1', 'B2', 'B3']), hcms: entries(['H1', 'H2']), vcms: entries(['V1']), groups: [] };

const CONFIG = {
  alpha: 0.01, keep_live: 3, n_discard: 0, gain: 0.5,
  excluded_bpms: [], excluded_cms: [], loss_cutoff: 0.2, overridden: [],
};

/** @param {Record<string, unknown>} [over] */
function solveOf(over = {}) {
  return {
    sha: 's1', singular_values: [3, 2, 1], inverse_weights: [1, 1, 1], kind: ['LIVE', 'LIVE', 'LIVE'],
    L: 3, live_rank: 3, rm: { source: 'model', rows: 6, h_cols: 2, v_cols: 1, computed_at: null },
    config: CONFIG, ...over,
  };
}

/** @param {Record<string, unknown>} [over] */
function stepOf(over = {}) {
  return {
    digest: 'd1', deltas: { H1: 0.1, H2: 0, V1: -0.2 }, factor: 1, reason: 'OK', status: 'ok',
    all_zero: false, zeroed_h: [], zeroed_v: [], no_response_names: [], unsolved: [],
    shot_id: 'shot-1', reference: null, ...over,
  };
}

/** @param {Record<string, unknown>} [over] */
function snapshotOf(over = {}) {
  return {
    mode: 'idle', model: 'SR', catalog: CATALOG, solve: solveOf(), step: stepOf(), last_action: null,
    ...over,
  };
}

/** @param {any} body */
const ok = (body) => ({ kind: 'ok', status: 200, body });
/** @param {string} code @param {string} detail @param {any[]} [moved] */
const refused = (code, detail, moved = []) => ({ kind: 'refused', status: 409, code, detail, moved, extras: {} });
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

const MOVED = [
  { name: 'H1', before: 1, after: 1.1, outcome: 'ok' },
  { name: 'V1', before: 2, after: 1.8, outcome: 'ok' },
];

/** @param {{holding?: boolean, snapshot?: any, shots?: any[], withResults?: boolean}} [options] */
function harness({ holding = true, snapshot = snapshotOf(), shots = [], withResults = true } = {}) {
  /** @type {Set<(s: any) => void>} */
  const snapshotFns = new Set();
  /** @type {Map<string, Set<(p: any) => void>>} */
  const events = new Map();
  let current = snapshot;
  /** @type {{id: string, figure: any}[]} */
  const drawn = [];
  const api = {
    solve: vi.fn(async () => ok(solveOf())),
    step: vi.fn(async () => ok(stepOf())),
    apply: vi.fn(async () => ok({ step_digest: 'd1', shot_id: 'shot-1', moved: MOVED, stopped: false })),
    putBack: vi.fn(async () => ok({ step_digest: null, shot_id: null, moved: MOVED, stopped: false })),
    shots: vi.fn(async () => ok({ model: 'SR', shots })),
  };
  const notify = vi.fn();
  const view = createView();
  const adapter = {
    render: vi.fn((/** @type {string} */ id, /** @type {(t: any, o: any) => any} */ build) => {
      drawn.push({ id, figure: build({}, { ranges: {} }) });
      return true;
    }),
  };
  document.body.replaceChildren();
  const host = document.createElement('div');
  const results = document.createElement('div');
  document.body.append(host, results);
  const ctx = /** @type {any} */ ({
    api,
    notify,
    view,
    adapter,
    holding: () => holding,
    snapshot: () => current,
    area: (/** @type {string} */ name) => (withResults && name === 'correction-results' ? results : null),
    onSnapshot(/** @type {(s: any) => void} */ fn) {
      snapshotFns.add(fn);
      return () => void snapshotFns.delete(fn);
    },
    onEvent(/** @type {string} */ type, /** @type {(p: any) => void} */ fn) {
      const set = events.get(type) ?? new Set();
      events.set(type, set);
      set.add(fn);
      return () => void set.delete(fn);
    },
  });
  mount(host, ctx);
  /** @param {string} name */
  const el = (name) => /** @type {any} */ (document.querySelector(`[data-control="${name}"]`));
  return {
    api, notify, view, adapter, host, results, drawn, el,
    /** @param {any} next */
    push(next) {
      current = next;
      for (const fn of snapshotFns) fn(next);
    },
    /** @param {string} type @param {any} payload */
    emit(type, payload) {
      for (const fn of events.get(type) ?? []) fn(payload);
    },
    /** @param {string} name @param {string} value */
    edit(name, value) {
      const input = el(name);
      input.value = value;
      input.dispatchEvent(new Event('input'));
      input.dispatchEvent(new Event('change'));
    },
  };
}

describe('pure views', () => {
  it('seeds the knobs from the solve config in force', () => {
    expect(seededKnobs(snapshotOf())).toEqual({ loss_cutoff: '0.2', alpha: '0.01', n_discard: '0', gain: '0.5' });
    expect(seededKnobs(null)).toEqual({ loss_cutoff: '', alpha: '', n_discard: '', gain: '' });
  });

  it('sends only edited knobs and the cutoff as keep_live = L - n_discard', () => {
    const drafts = { ...emptyKnobDrafts(), alpha: '0.1', n_discard: '1' };
    expect(solvePartial(drafts, solveOf())).toEqual({ body: { alpha: 0.1, keep_live: 2 } });
    expect(solvePartial(emptyKnobDrafts(), solveOf())).toEqual({ body: {} });
    expect(solvePartial({ ...emptyKnobDrafts(), gain: '0.3', loss_cutoff: '0.5' }, solveOf()))
      .toEqual({ body: { gain: 0.3, loss_cutoff: 0.5 } });
  });

  it('refuses a cutoff that is not a whole number in [0, L]', () => {
    for (const text of ['4', '-1', '1.5']) {
      const parsed = solvePartial({ ...emptyKnobDrafts(), n_discard: text }, solveOf());
      expect(parsed).toEqual({ error: 'Singular Value Cutoff: enter a whole number from 0 to 3' });
    }
    expect(solvePartial({ ...emptyKnobDrafts(), n_discard: '1' }, null))
      .toEqual({ error: 'Singular Value Cutoff: Calculate Response Matrix first' });
    expect(solvePartial({ ...emptyKnobDrafts(), alpha: '' }, solveOf())).toEqual({ error: 'Tikhonov Regularization: enter a number' });
  });

  it('describes the response matrix by source and dimensions', () => {
    expect(rmStatus(null)).toBe('No response matrix yet');
    expect(rmStatus(solveOf())).toBe('RM from model: 6 × 3 (2 HCM, 1 VCM)');
    expect(rmStatus(solveOf({ rm: { source: 'model', rows: 6, h_cols: 2, v_cols: 1, computed_at: 0 } })))
      .toMatch(/^RM from model: 6 × 3 \(2 HCM, 1 VCM\), computed /);
  });

  it('compares BPM sets as sets', () => {
    expect(sameNames(['B1', 'B2'], ['B2', 'B1'])).toBe(true);
    expect(sameNames(['B1'], ['B1', 'B2'])).toBe(false);
  });

  describe('applyReason', () => {
    const base = { snapshot: snapshotOf(), kept: stepOf(), keptExcluded: [], machineMoved: false, holding: true };

    it('offers Apply when the kept digest is the pending one and the mode is idle', () => {
      expect(applyReason(base)).toBeNull();
    });

    it('names each reason it is withheld', () => {
      expect(applyReason({ ...base, holding: false })).toBe(NO_SESSION);
      expect(applyReason({ ...base, kept: null })).toBe(NO_STEP);
      expect(applyReason({ ...base, snapshot: snapshotOf({ mode: 'action' }) })).toBe(BUSY.action);
      expect(applyReason({ ...base, snapshot: snapshotOf({ mode: 'continuous' }) })).toBe(BUSY.continuous);
      expect(applyReason({ ...base, snapshot: snapshotOf({ step: stepOf({ digest: 'd2' }) }) })).toBe(STEP_CHANGED);
      expect(applyReason({ ...base, machineMoved: true })).toBe(MACHINE_MOVED);
      expect(applyReason({ ...base, keptExcluded: ['B2'] })).toBe(BPM_SET_CHANGED);
      expect(applyReason({ ...base, kept: stepOf({ all_zero: true }), snapshot: snapshotOf({ step: stepOf({ all_zero: true }) }) }))
        .toBe(ZERO_STEP);
    });

    it('compares the kept shot BPM set with the one in force', () => {
      const config = { ...CONFIG, excluded_bpms: ['B2'] };
      const snapshot = snapshotOf({ solve: solveOf({ config }) });
      expect(applyReason({ ...base, snapshot, keptExcluded: ['B2'] })).toBeNull();
      expect(applyReason({ ...base, snapshot, keptExcluded: [] })).toBe(BPM_SET_CHANGED);
      expect(applyReason({ ...base, snapshot, keptExcluded: null })).toBeNull();
    });

    it('asks for a new step once a newer shot has its own pending step', () => {
      const snapshot = snapshotOf({ step: stepOf({ digest: 'd2', shot_id: 'shot-2' }) });
      expect(applyReason({ ...base, snapshot, machineMoved: true, keptExcluded: ['B3'] })).toBe(STEP_CHANGED);
    });
  });

  it('offers Put back only after an apply or setup that moved something', () => {
    expect(putBackReason(snapshotOf(), true)).toBe('Nothing to put back');
    const moved = snapshotOf({ last_action: { name: 'apply', stopped: false, moved: MOVED } });
    expect(putBackReason(moved, true)).toBeNull();
    expect(putBackReason(moved, false)).toBe(NO_SESSION);
    expect(putBackReason({ ...moved, model: null }, true)).toBe('No model loaded');
    expect(putBackReason({ ...moved, mode: 'continuous' }, true)).toBe(BUSY.continuous);
  });

  it('lists moved settings before -> after as served', () => {
    const table = applyResultView([...MOVED, { name: 'H2', before: 3, outcome: 'limits' }]);
    expect(table.getAttribute('data-control')).toBe('apply-result');
    const rows = [...table.querySelectorAll('tbody tr')].map((tr) => [...tr.querySelectorAll('td')].map((td) => td.textContent));
    expect(rows).toEqual([['H1', '1 →', '1.1', 'ok'], ['V1', '2 →', '1.8', 'ok'], ['H2', '3 →', '—', 'limits']]);
  });
});

describe('mount', () => {
  it('renders every inventory control with the Singular Value Cutoff tooltip', () => {
    const t = harness();
    for (const name of ['solve-loss-cutoff', 'rm-calculate', 'rm-status', 'solve-alpha', 'solve-n-discard',
      'solve-gain', 'solve-calculate', 'step-calculate', 'step-plot-bars', 'step-apply', 'put-back', 'plot-cm-step']) {
      expect(t.el(name), name).not.toBeNull();
    }
    expect(t.el('solve-n-discard').title).toBe('Singular Value Cutoff (n_discard)');
    expect(t.el('rm-status').textContent).toBe('RM from model: 6 × 3 (2 HCM, 1 VCM)');
    expect(t.results.contains(t.el('plot-cm-step'))).toBe(true);
  });

  it('draws plot-sv from the solve and only again when its sha changes', () => {
    const t = harness();
    expect(t.drawn.filter((d) => d.id === 'plot-sv')).toHaveLength(1);
    expect(t.drawn[0].figure.data[0].y).toEqual([3, 2, 1]);
    t.push(snapshotOf());
    expect(t.drawn.filter((d) => d.id === 'plot-sv')).toHaveLength(1);
    t.push(snapshotOf({ solve: solveOf({ sha: 's2', singular_values: [5, 4, 3] }) }));
    const sv = t.drawn.filter((d) => d.id === 'plot-sv');
    expect(sv).toHaveLength(2);
    expect(sv[1].figure.data[0].y).toEqual([5, 4, 3]);
  });

  it('Calculate Response Matrix sends an empty partial', async () => {
    const t = harness();
    t.el('rm-calculate').click();
    await flush();
    expect(t.api.solve).toHaveBeenCalledWith({});
  });

  it('Calculate Pseudo-Inverse sends the edited knobs, n_discard as keep_live, then re-seeds', async () => {
    const t = harness();
    t.edit('solve-n-discard', '1');
    t.edit('solve-gain', '0.8');
    t.el('solve-calculate').click();
    await flush();
    expect(t.api.solve).toHaveBeenCalledWith({ gain: 0.8, keep_live: 2 });
    t.push(snapshotOf({ solve: solveOf({ config: { ...CONFIG, gain: 0.8, n_discard: 1, keep_live: 2 } }) }));
    expect(t.el('solve-n-discard').value).toBe('1');
    expect(t.el('solve-n-discard').getAttribute('data-pending')).toBeNull();
  });

  it('keeps an edited knob while snapshots arrive, re-seeding once reverted', () => {
    const t = harness();
    t.edit('solve-alpha', '0.5');
    t.push(snapshotOf({ solve: solveOf({ config: { ...CONFIG, alpha: 0.02 } }) }));
    expect(t.el('solve-alpha').value).toBe('0.5');
    t.el('solve-revert').click();
    expect(t.el('solve-alpha').value).toBe('0.02');
  });

  it('disables the solve buttons with the reason for an invalid cutoff or no session', () => {
    const t = harness();
    t.edit('solve-n-discard', '9');
    expect(t.el('solve-calculate').disabled).toBe(true);
    expect(t.el('solve-calculate').dataset.reason).toBe('Singular Value Cutoff: enter a whole number from 0 to 3');
    const away = harness({ holding: false });
    expect(away.el('rm-calculate').dataset.reason).toBe(NO_SESSION);
    expect(away.el('step-apply').dataset.reason).toBe(NO_SESSION);
  });

  it('Calculate CM step keeps the step, draws its bars and enables Apply', async () => {
    const t = harness();
    expect(t.el('step-apply').dataset.reason).toBe(NO_STEP);
    expect(t.el('plot-cm-step').hidden).toBe(true);
    t.el('step-calculate').click();
    await flush();
    await flush();
    expect(t.el('step-apply').disabled).toBe(false);
    expect(t.el('plot-cm-step').hidden).toBe(false);
    const bars = t.drawn.filter((d) => d.id === 'plot-cm-step');
    expect(bars).toHaveLength(1);
    expect(bars[0].figure.data[0].y).toEqual([0.1, 0]);
    expect(bars[0].figure.data[1].y).toEqual([-0.2]);
  });

  it('the plot tick hides the CM-step bars and draws them again when set', async () => {
    const t = harness();
    t.el('step-calculate').click();
    await flush();
    const tick = t.el('step-plot-bars');
    tick.checked = false;
    tick.dispatchEvent(new Event('change'));
    expect(t.view.get('plotCmStep')).toBe(false);
    expect(t.el('plot-cm-step').hidden).toBe(true);
    const before = t.drawn.filter((d) => d.id === 'plot-cm-step').length;
    t.el('step-plot-bars').checked = true;
    t.el('step-plot-bars').dispatchEvent(new Event('change'));
    expect(t.el('plot-cm-step').hidden).toBe(false);
    expect(t.drawn.filter((d) => d.id === 'plot-cm-step').length).toBe(before + 1);
  });

  it('withholds Apply once the pending step changed', async () => {
    const t = harness();
    t.el('step-calculate').click();
    await flush();
    t.push(snapshotOf({ step: stepOf({ digest: 'd2' }) }));
    expect(t.el('step-apply').dataset.reason).toBe(STEP_CHANGED);
  });

  it('withholds Apply while an action runs', async () => {
    const t = harness();
    t.el('step-calculate').click();
    await flush();
    t.push(snapshotOf({ mode: 'action' }));
    expect(t.el('step-apply').dataset.reason).toBe(BUSY.action);
  });

  it('withholds Apply when the BPM set changed since the kept step\'s shot', async () => {
    const t = harness({ shots: [{ shot_id: 'shot-1', excluded_bpms: [] }] });
    t.el('step-calculate').click();
    await flush();
    await flush();
    expect(t.api.shots).toHaveBeenCalled();
    t.push(snapshotOf({ solve: solveOf({ config: { ...CONFIG, excluded_bpms: ['B2'] } }) }));
    expect(t.el('step-apply').dataset.reason).toBe(BPM_SET_CHANGED);
  });

  it('offers Apply for a step whose shot left out the BPMs committed as excluded', async () => {
    const config = { ...CONFIG, excluded_bpms: ['B2'] };
    const t = harness({
      snapshot: snapshotOf({ solve: solveOf({ config }) }),
      shots: [{ shot_id: 'shot-1', excluded_bpms: ['B2'] }],
    });
    t.el('step-calculate').click();
    await flush();
    await flush();
    expect(t.el('step-apply').disabled).toBe(false);
    t.push(snapshotOf({ solve: solveOf({ config: { ...CONFIG, excluded_bpms: ['B2', 'B3'] } }) }));
    expect(t.el('step-apply').dataset.reason).toBe(BPM_SET_CHANGED);
  });

  it('takes the kept shot\'s BPM set from the shot event without a lookup', async () => {
    const t = harness();
    t.emit('shot', { shot: { shot_id: 'shot-1', excluded_bpms: ['B3'] } });
    t.el('step-calculate').click();
    await flush();
    await flush();
    expect(t.api.shots).not.toHaveBeenCalled();
    expect(t.el('step-apply').dataset.reason).toBe(BPM_SET_CHANGED);
  });

  it('Apply sends the kept digest and lists the served moves under the bars', async () => {
    const t = harness();
    t.el('step-calculate').click();
    await flush();
    await flush();
    t.el('step-apply').click();
    await flush();
    expect(t.api.apply).toHaveBeenCalledWith('d1');
    const table = t.el('apply-result');
    expect(t.results.contains(table)).toBe(true);
    expect(table.compareDocumentPosition(t.el('plot-cm-step')) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy();
    const names = [...table.querySelectorAll('tbody tr')].map((tr) => tr.getAttribute('data-moved'));
    expect(names).toEqual(['H1', 'V1']);
    expect(table.textContent).toContain('1.1');
  });

  it('a 409 machine_moved shows its reason and withholds Apply until a new step', async () => {
    const t = harness();
    t.api.apply.mockResolvedValueOnce(/** @type {any} */ (refused('machine_moved', 'H1 moved since shot-1')));
    t.el('step-calculate').click();
    await flush();
    await flush();
    t.el('step-apply').click();
    await flush();
    expect(t.notify).toHaveBeenCalledWith(expect.objectContaining({ code: 'machine_moved' }));
    expect(t.host.querySelector('[data-view="correction-message"]')?.textContent).toContain('H1 moved since shot-1');
    expect(t.el('step-apply').dataset.reason).toBe(MACHINE_MOVED);
    t.el('step-calculate').click();
    await flush();
    await flush();
    expect(t.el('step-apply').disabled).toBe(false);
  });

  it('an apply event marks the machine moved and shows the served moves', async () => {
    const t = harness();
    t.el('step-calculate').click();
    await flush();
    await flush();
    t.emit('apply', { step_digest: 'dx', shot_id: 'shot-1', moved: MOVED });
    expect(t.el('step-apply').dataset.reason).toBe(MACHINE_MOVED);
    expect(t.el('apply-result').querySelectorAll('tbody tr')).toHaveLength(2);
  });

  it('Put back posts the put-back body of the last apply', async () => {
    const snapshot = snapshotOf({ last_action: { name: 'apply', stopped: false, moved: MOVED } });
    const t = harness({ snapshot });
    expect(t.el('put-back').disabled).toBe(false);
    t.el('put-back').click();
    await flush();
    expect(t.api.putBack).toHaveBeenCalledWith({ model: 'SR', values: [{ name: 'H1', before: 1 }, { name: 'V1', before: 2 }] });
    expect(t.el('apply-result').textContent).toContain('Put back');
  });

  it('a refused step fetch keeps nothing and notifies', async () => {
    const t = harness();
    t.api.step.mockResolvedValueOnce(/** @type {any} */ (refused('no_step', 'no step is pending; take a shot first')));
    t.el('step-calculate').click();
    await flush();
    expect(t.el('step-apply').dataset.reason).toBe(NO_STEP);
    expect(t.notify).toHaveBeenCalledWith(expect.objectContaining({ code: 'no_step' }));
  });

  it('falls back to its own results host when the shell has none', () => {
    const t = harness({ withResults: false });
    expect(t.host.contains(t.el('plot-cm-step'))).toBe(true);
  });
});
