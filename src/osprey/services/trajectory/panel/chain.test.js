// @ts-check
import { describe, expect, it, vi } from 'vitest';

import {
  NO_SESSION,
  PLOT_STEP,
  SETUP_FIRST,
  TBT_TURNS_TIP,
  chainBody,
  chainRows,
  clampCount,
  mount,
  runReason,
  turnsBound,
} from './chain.js';
import { createView } from './views.js';

const CHAIN = [
  { id: 'readout', label: 'Readout BPMs', description: 'Reads every BPM.', default: true, position: 3,
    command: null, timeout_s: null },
  { id: 'arm', label: 'Arm BPMs', description: null, default: true, position: 0, command: null, timeout_s: null },
  { id: 'clean', label: 'Bunch Cleaning', description: 'Cleans the bunch.', default: false, position: 2,
    command: { address: 'SR:CLEAN', value: 1 }, timeout_s: 5 },
  { id: 'inject', label: 'Inject Beam', description: 'Fires one injected shot.', default: true, position: 1,
    command: null, timeout_s: null },
];

const SETUP = { applied_at: 1, by: 'alice', settings_sha: 'abc', turns_tracked: 260, void: false };

/** @param {Record<string, any>} [over] */
function snapshotWith(over = {}) {
  return { chain: CHAIN, n_shots_max: 8, setup: SETUP, continuous: null, mode: 'idle', ...over };
}

/** @param {any} body */
const ok = (body = {}) => ({ kind: 'ok', status: 200, body });
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** @param {{holding?: boolean, snapshot?: any}} [options] */
function harness({ holding = true, snapshot = snapshotWith() } = {}) {
  /** @type {Set<(s: any) => void>} */
  const fns = new Set();
  let current = snapshot;
  const api = { shot: vi.fn(async () => ok()) };
  const view = createView();
  const ctx = /** @type {any} */ ({
    api,
    view,
    notify: vi.fn(),
    holding: () => holding,
    snapshot: () => current,
    onSnapshot(/** @type {(s: any) => void} */ fn) {
      fns.add(fn);
      return () => void fns.delete(fn);
    },
    startContinuous: vi.fn(async () => ok()),
    stopContinuous: vi.fn(async () => ok()),
  });
  const host = document.createElement('div');
  document.body.replaceChildren(host);
  mount(host, ctx);
  /** @param {string} name */
  const el = (name) => /** @type {any} */ (host.querySelector(`[data-control="${name}"]`));
  return {
    host,
    ctx,
    api,
    view,
    el,
    /** @param {any} next */
    emit(next) {
      current = next;
      for (const fn of [...fns]) fn(next);
    },
    /** @param {string} name @param {string} value */
    type(name, value) {
      const input = el(name);
      input.value = value;
      input.dispatchEvent(new Event('change'));
    },
  };
}

describe('pure views', () => {
  it('orders the declared chain by position and ends with the panel-only plot row', () => {
    expect(chainRows(CHAIN).map((s) => s.id)).toEqual(['arm', 'inject', 'clean', 'readout', PLOT_STEP.id]);
    expect(chainRows([]).map((s) => s.id)).toEqual([PLOT_STEP.id]);
  });

  it('gives the run reason: no session, then no or a void setup, else none', () => {
    expect(runReason(snapshotWith(), false)).toBe(NO_SESSION);
    expect(runReason(snapshotWith({ setup: null }), true)).toBe(SETUP_FIRST);
    expect(runReason(snapshotWith({ setup: { ...SETUP, void: true } }), true)).toBe(SETUP_FIRST);
    expect(runReason(null, true)).toBe(SETUP_FIRST);
    expect(runReason(snapshotWith(), true)).toBeNull();
  });

  it('clamps a count into [1, max] and treats a missing max as unbounded', () => {
    expect(clampCount('5', 8)).toBe(5);
    expect(clampCount('99', 8)).toBe(8);
    expect(clampCount('0', 8)).toBe(1);
    expect(clampCount('x', 8)).toBe(1);
    expect(clampCount('2.7', 8)).toBe(2);
    expect(clampCount('500', null)).toBe(500);
  });

  it('bounds the turns by the setup turns tracked', () => {
    expect(turnsBound(snapshotWith())).toBe(260);
    expect(turnsBound(snapshotWith({ setup: null }))).toBeNull();
  });

  it('builds a chain body: TbT always one turn; pseudo-orbit keeps its turns', () => {
    expect(chainBody(['arm'], { readout: 'tbt', nShots: 3, nTurns: 40 }))
      .toEqual({ steps: ['arm'], readout: 'tbt', n_shots: 3, n_turns: 1 });
    expect(chainBody(['arm', 'readout'], { readout: 'pseudo_orbit', nShots: 1, nTurns: 40 }))
      .toEqual({ steps: ['arm', 'readout'], readout: 'pseudo_orbit', n_shots: 1, n_turns: 40 });
  });
});

describe('mount', () => {
  it('renders one row per chain step in position order with its declared default', () => {
    const t = harness();
    const rows = [...t.host.querySelectorAll('[data-step]')].map((r) => r.getAttribute('data-step'));
    expect(rows).toEqual(['arm', 'inject', 'clean', 'readout', 'plot']);
    expect(t.el('step-arm').checked).toBe(true);
    expect(t.el('step-clean').checked).toBe(false);
    expect(t.el('step-readout').checked).toBe(true);
    expect(t.host.querySelector('[data-step="inject"]')?.textContent).toContain('Inject Beam');
    for (const id of ['arm', 'inject', 'clean', 'readout', 'plot']) {
      expect(t.el(`step-${id}-run`)).not.toBeNull();
    }
    expect(t.el('step-arm-options')).toBeNull();
    expect(t.el('step-clean-options').textContent).toContain('Cleans the bunch.');
  });

  it('opens a step\'s Options to show its description', () => {
    const t = harness();
    const options = t.el('step-inject-options');
    expect(options.tagName).toBe('DETAILS');
    expect(options.open).toBe(false);
    options.querySelector('summary').click();
    expect(options.open).toBe(true);
    expect(options.textContent).toContain('Fires one injected shot.');
  });

  it('Run Selected posts the ticked step ids in chain order as one chain body', async () => {
    const t = harness();
    t.el('step-clean').click();
    t.el('step-inject').click();
    t.el('run-selected').click();
    await flush();
    expect(t.api.shot).toHaveBeenCalledTimes(1);
    expect(t.api.shot).toHaveBeenCalledWith({ steps: ['arm', 'clean', 'readout'], readout: 'tbt', n_shots: 1, n_turns: 1 });
  });

  it('a per-step Run posts that step alone', async () => {
    const t = harness();
    t.el('step-clean-run').click();
    await flush();
    expect(t.api.shot).toHaveBeenCalledWith({ steps: ['clean'], readout: 'tbt', n_shots: 1, n_turns: 1 });
  });

  it('shows a refusal as a notice with its code', async () => {
    const t = harness();
    t.api.shot.mockResolvedValueOnce(/** @type {any} */ ({
      kind: 'refused', status: 409, code: 'setup_required', detail: 'no setup', extras: {}, moved: null,
    }));
    t.el('run-selected').click();
    await flush();
    expect(t.ctx.notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'refused', code: 'setup_required' }));
  });

  it('keeps ticks across snapshots and picks up a newly declared step', () => {
    const t = harness();
    t.el('step-arm').click();
    t.emit(snapshotWith({ chain: [...CHAIN, { id: 'extra', label: 'Extra', description: null, default: true,
      position: 4, command: null, timeout_s: null }] }));
    expect(t.el('step-arm').checked).toBe(false);
    expect(t.el('step-extra').checked).toBe(true);
  });

  it('the plot row writes view.plotRaw and its Run asks for a redraw, never a route', async () => {
    const t = harness();
    expect(t.el('step-plot').checked).toBe(true);
    t.el('step-plot').click();
    expect(t.view.get('plotRaw')).toBe(false);
    t.el('step-plot').click();
    expect(t.view.get('plotRaw')).toBe(true);
    const before = t.view.get('redraw');
    t.el('step-plot-run').click();
    await flush();
    expect(t.view.get('redraw')).toBe(before + 1);
    expect(t.api.shot).not.toHaveBeenCalled();
    expect(t.el('step-plot-options').textContent).toContain(PLOT_STEP.description);
  });

  it('a plotRaw change elsewhere is reflected in the plot checkbox', () => {
    const t = harness();
    t.view.set('plotRaw', false);
    expect(t.el('step-plot').checked).toBe(false);
  });

  it.each([
    ['no setup', { setup: null }],
    ['a void setup', { setup: { ...SETUP, void: true } }],
  ])('disables Run Selected, every Run and Continuous with "Setup BPMs first" under %s', (_, over) => {
    const t = harness({ snapshot: snapshotWith(over) });
    for (const name of ['run-selected', 'chain-continuous', 'step-arm-run', 'step-inject-run', 'step-clean-run',
      'step-readout-run']) {
      expect(t.el(name).disabled).toBe(true);
      expect(t.el(name).getAttribute('data-reason')).toBe(SETUP_FIRST);
      expect(t.el(name).title).toBe(SETUP_FIRST);
    }
    expect(t.el('step-plot-run').disabled).toBe(false);
    t.emit(snapshotWith());
    expect(t.el('run-selected').disabled).toBe(false);
    expect(t.el('step-arm-run').disabled).toBe(false);
    expect(t.el('chain-continuous').disabled).toBe(false);
  });

  it('disables Run Selected when nothing is ticked', () => {
    const t = harness();
    for (const id of ['arm', 'inject', 'readout']) t.el(`step-${id}`).click();
    expect(t.el('run-selected').disabled).toBe(true);
  });

  it('bounds the number of shots by Snapshot.n_shots_max', async () => {
    const t = harness();
    expect(t.el('chain-n-shots').max).toBe('8');
    t.type('chain-n-shots', '50');
    expect(t.el('chain-n-shots').value).toBe('8');
    t.emit(snapshotWith({ n_shots_max: 4 }));
    expect(t.el('chain-n-shots').value).toBe('4');
    t.el('run-selected').click();
    await flush();
    expect(t.api.shot).toHaveBeenCalledWith(expect.objectContaining({ n_shots: 4 }));
  });

  it('in TbT the turns field is a fixed read-only 1 with its tooltip', async () => {
    const t = harness();
    const turns = t.el('chain-n-turns');
    expect(t.el('chain-readout-mode').value).toBe('tbt');
    expect(turns.readOnly).toBe(true);
    expect(turns.value).toBe('1');
    expect(turns.title).toBe(TBT_TURNS_TIP);
    t.type('chain-n-turns', '30');
    expect(t.el('chain-n-turns').value).toBe('1');
    t.el('run-selected').click();
    await flush();
    expect(t.api.shot).toHaveBeenCalledWith(expect.objectContaining({ readout: 'tbt', n_turns: 1 }));
  });

  it('in pseudo-orbit the turns field is editable, bounded by the setup turns tracked', async () => {
    const t = harness();
    t.type('chain-readout-mode', 'pseudo_orbit');
    const turns = t.el('chain-n-turns');
    expect(turns.readOnly).toBe(false);
    expect(turns.max).toBe('260');
    t.type('chain-n-turns', '30');
    expect(t.el('chain-n-turns').value).toBe('30');
    t.type('chain-n-turns', '999');
    expect(t.el('chain-n-turns').value).toBe('260');
    t.type('chain-n-turns', '30');
    t.el('run-selected').click();
    await flush();
    expect(t.api.shot).toHaveBeenCalledWith(expect.objectContaining({ readout: 'pseudo_orbit', n_turns: 30 }));
    t.type('chain-readout-mode', 'tbt');
    expect(t.el('chain-n-turns').readOnly).toBe(true);
    expect(t.el('chain-n-turns').value).toBe('1');
  });

  it('the Manual/Continuous toggle starts continuous with the chain body and stops it when running', async () => {
    const t = harness();
    const toggle = t.el('chain-continuous');
    expect(toggle.getAttribute('aria-pressed')).toBe('false');
    toggle.click();
    await flush();
    expect(t.ctx.startContinuous).toHaveBeenCalledWith({ steps: ['arm', 'inject', 'readout'], readout: 'tbt',
      n_shots: 1, n_turns: 1 });
    t.emit(snapshotWith({ mode: 'continuous', continuous: { shots: 2, lease_expires_in_s: 10 } }));
    expect(t.el('chain-continuous').getAttribute('aria-pressed')).toBe('true');
    expect(t.el('run-selected').disabled).toBe(true);
    t.el('chain-continuous').click();
    await flush();
    expect(t.ctx.stopContinuous).toHaveBeenCalledTimes(1);
    expect(t.ctx.startContinuous).toHaveBeenCalledTimes(1);
  });

  it('the stop half of the toggle stays usable when the setup goes void mid-run', () => {
    const t = harness({ snapshot: snapshotWith({ mode: 'continuous', continuous: { shots: 1 },
      setup: { ...SETUP, void: true } }) });
    expect(t.el('chain-continuous').disabled).toBe(false);
  });

  it('without a session every run control says to open one', () => {
    const t = harness({ holding: false });
    expect(t.el('run-selected').getAttribute('data-reason')).toBe(NO_SESSION);
    expect(t.el('step-arm-run').getAttribute('data-reason')).toBe(NO_SESSION);
  });
});
