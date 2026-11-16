/**
 * Tests for the run page (run-page.js, run-header.js, run-knobs.js,
 * run-page-model.js): what it shows while a run waits for its write
 * approval, runs, is paused or cancels, and which action each control calls.
 *
 * The safety rules it holds: a run that waits for approval looks different
 * from a running one (warning colour, no chart, "Nothing has been written");
 * Pause and Cancel are in the run header and never in the primary slot;
 * Cancel asks first and says the machine keeps the last point written and
 * nothing is restored; a refusal shows its reason; the page never approves a
 * write request; a pause is not said to hold while the run still writes.
 *
 * The fixtures follow what the service sends: a stored draft always carries
 * the objective direction (the service fills it in), a started run has no
 * write authorization until it asks for one, and a pause shows at once while
 * the runner only checks for it before each guided step.
 */

import { afterEach, describe, test, expect, vi } from 'vitest';

import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import { KNOB_KEYS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/knob-bar.js';
import { mountRunPage } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-page.js';
import {
  cancelWords,
  methodLine,
  nowSentence,
  pausePhase,
  pauseWords,
  runFacts,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-page-model.js';
import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';
import { STOP_WORDS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent} StoreEvent */

const ACTIONS = [
  'pause',
  'resume',
  'requestCancel',
  'confirmCancel',
  'dismissCancel',
  'loadEnvironment',
  'start',
  'markReviewed',
  'newRun',
  'editDraft',
];

function context() {
  const stage = document.createElement('section');
  const inspector = document.createElement('aside');
  document.body.append(stage, inspector);
  /** @type {Record<string, import('vitest').Mock>} */
  const actions = Object.fromEntries(ACTIONS.map((name) => [name, vi.fn()]));
  const setPrimary = vi.fn();
  const ctx = {
    stage,
    inspector,
    actions: /** @type {any} */ (actions),
    shell: {
      setPrimary,
      fillInspector: (/** @type {any} */ parts) =>
        inspector.replaceChildren(...inspectorFrame(parts, () => {})),
      setDrawer: vi.fn(),
      setDrawerTop: vi.fn(),
    },
  };
  return { ctx, stage, inspector, actions, setPrimary };
}

/** The record of a request that waits for a person. */
const awaiting = () => ({
  state: 'awaiting',
  target: 'va',
  channels: { 'A:SP': [-18, 18], 'B:SP': [2, 4] },
  steps: 11,
  completion: 'apply_best',
  id: 'req-0123456789',
  owner: 'carol',
  approver: null,
});

const granted = () => ({
  ...awaiting(),
  state: 'granted',
  approver: 'chip',
  expires_at_ns: Date.now() * 1e6 + 3.6e12,
});

const initial = { phase_a: 1, sol_b: 3, charge: 0.5 };

/**
 * The state and screen after the poll read `job`, its stored config (with
 * `config` over it) and its environment, then `extra`.
 *
 * @param {Record<string, unknown>} job
 * @param {StoreEvent[]} [extra]
 * @param {Record<string, unknown>} [config]
 */
function at(job, extra = [], config = {}) {
  const full = runningJob({ snapshots: { initial }, ...job });
  /** @type {StoreEvent[]} */
  const events = [
    { type: 'poll/job', job: full, at: 1000 },
    {
      type: 'poll/config',
      jobId: full.job_id,
      revision: 'rev-abcdef12345',
      // The service writes the resolved direction into every stored draft.
      config: storedConfig({ objective_direction: 'MAXIMIZE', ...config }),
    },
    { type: 'env/loaded', name: 'injector', details: envDetails(), presets: null, error: null },
    { type: 'session', owner: 'carol', ownerKnown: true },
    ...extra,
  ];
  const state = events.reduce(reduce, initialState());
  return { state, screen: deriveScreen(state) };
}

const waiting = () => at({ data: [], write_authorization: awaiting() });
const running = () => at({ write_authorization: granted() });
const paused = () => at({ status: 'paused', write_authorization: granted() });
/** Started a moment ago: no write authorization asked yet, nothing evaluated. */
const starting = () => at({ data: [], write_authorization: null });

/**
 * A run of 3 first tries (UCB on two knobs) and 30 guided steps, paused
 * with `done` evaluations recorded.
 *
 * @param {number} done
 */
function pausedAfter(done) {
  const data = Array.from({ length: done }, (_, i) => ({ phase_a: i, sol_b: 3, charge: i }));
  return at({ status: 'paused', write_authorization: granted(), total_evaluations: 33, data });
}

/** @param {HTMLElement} root @param {string} control */
const button = (root, control) =>
  /** @type {HTMLButtonElement} */ (root.querySelector(`[data-control="${control}"]`));

const confirmPop = () => /** @type {HTMLElement} */ (document.querySelector('.opt-confirm'));

afterEach(() => {
  document.body.replaceChildren();
});

describe('the run page', () => {
  test('a run that waits for approval: warning, no chart, nothing written, who approves', () => {
    const { ctx, stage, inspector } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = waiting();
    expect(screen.id).toBe('waiting-approval');
    view.update(state, screen);
    expect(/** @type {HTMLElement} */ (stage.querySelector('.opt-rp-head')).dataset.kind).toBe(
      'waiting'
    );
    expect(/** @type {HTMLElement} */ (stage.querySelector('.opt-rp-chart')).hidden).toBe(true);
    expect(stage.querySelector('.opt-wait-card')).not.toBeNull();
    expect(stage.textContent).toContain('Nothing has been written');
    expect(stage.querySelector('.opt-wait-card')?.textContent).toContain('carol');
    // The inspector lists what the request covers, named by the knobs' descriptions.
    const knobs = Array.from(inspector.querySelectorAll('[data-knob]'));
    expect(knobs.map((k) => k.getAttribute('data-knob'))).toEqual(['A:SP', 'B:SP']);
    expect(knobs[0].textContent).toContain('Buncher phase');
    view.unmount?.();
  });

  test('a run started a moment ago, before it asks for approval, waits; it is not running', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = starting();
    expect(screen.id).toBe('waiting-approval');
    view.update(state, screen);
    expect(/** @type {HTMLElement} */ (stage.querySelector('.opt-rp-chart')).hidden).toBe(true);
    expect(stage.querySelector('.opt-wait-card')).not.toBeNull();
    expect(stage.textContent).toContain('Nothing has been written');
    // No request exists yet: its cancel is not the withdrawal of a request.
    expect(cancelWords(screen).title).not.toBe(cancelWords(waiting().screen).title);
    view.unmount?.();
  });

  test('the cards of a waiting run show the value at the start, not a value of now', () => {
    const { ctx, inspector } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = waiting();
    view.update(state, screen);
    expect(inspector.querySelector('[data-mark="now"]')).toBeNull();
    expect(inspector.querySelector('[data-key="now"]')).toBeNull();
    expect(inspector.querySelector('[data-knob="A:SP"] [data-mark="start"]')).not.toBeNull();
    view.unmount?.();
  });

  test('a running run: the evaluation under way, the chart, a card per knob; no approval card', () => {
    const { ctx, stage, inspector } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = running();
    expect(screen.id).toBe('running');
    view.update(state, screen);
    expect(stage.querySelector('.opt-rp-sentence')?.textContent).toMatch(/\b4\b.*\b10\b/);
    expect(/** @type {HTMLElement} */ (stage.querySelector('.opt-rp-chart')).hidden).toBe(false);
    expect(/** @type {HTMLElement} */ (stage.querySelector('.opt-wait')).hidden).toBe(true);
    expect(stage.querySelectorAll('.opt-ch-pt')).toHaveLength(3);
    expect(stage.textContent).not.toContain('Nothing has been written');
    const knobs = Array.from(inspector.querySelectorAll('[data-knob]'));
    expect(knobs.map((k) => k.getAttribute('data-knob'))).toEqual(['phase_a', 'sol_b']);
    // The last evaluation recorded: phase_a -9, from 1 at the start.
    expect(knobs[0].querySelector('.opt-kn-val')?.textContent).toContain('-9');
    expect(knobs[0].querySelector('[data-mark="best"]')).not.toBeNull();
    view.unmount?.();
  });

  test('the knob cards do not call the last evaluation recorded what is on the machine', () => {
    const { ctx, inspector } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = running();
    view.update(state, screen);
    expect(inspector.textContent).not.toContain(KNOB_KEYS.current);
    expect(inspector.textContent).toContain(KNOB_KEYS.recorded);
    view.unmount?.();
  });

  test('the best on the page is the one the service applies: an evaluation that fails a constraint is not it', () => {
    const { ctx, stage, inspector } = context();
    const view = mountRunPage(ctx);
    const data = [
      { phase_a: 0, sol_b: 3, charge: 1.0, loss: 0.5 },
      { phase_a: 9, sol_b: 2, charge: 2.5, loss: 3 },
      { phase_a: -9, sol_b: 4, charge: 2.0, loss: 0.2 },
    ];
    const constraints = { loss: { type: 'LESS_THAN', value: 1 } };
    const { state, screen } = at({ write_authorization: granted(), data }, [], { constraints });
    expect(screen.figures?.bestIndex).toBe(2);
    view.update(state, screen);
    expect(stage.querySelectorAll('.opt-ch-pt.is-bad')).toHaveLength(1);
    // The best mark of a knob sits at the counted best (phase_a -9), not at 9.
    const best = inspector.querySelector('[data-knob="phase_a"] [data-mark="best"]');
    const recorded = inspector.querySelector('[data-knob="phase_a"] [data-mark="current"]');
    expect(best?.getAttribute('style')).toBe(recorded?.getAttribute('style'));
    /** @type {HTMLElement} */ (stage.querySelector('[data-value="table"]')).click();
    expect(stage.querySelector('.opt-evals tr.is-best')?.getAttribute('data-index')).toBe('2');
    view.unmount?.();
  });

  test('the evaluations view lists the records as a table', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = running();
    view.update(state, screen);
    /** @type {HTMLElement} */ (stage.querySelector('[data-value="table"]')).click();
    expect(stage.querySelectorAll('.opt-evals tbody tr')).toHaveLength(3);
    view.update(state, screen);
    expect(stage.querySelectorAll('.opt-evals tbody tr')).toHaveLength(3);
    view.unmount?.();
  });

  test('Pause and Cancel are in the run header; the primary slot stays empty', () => {
    const { ctx, stage, setPrimary } = context();
    const view = mountRunPage(ctx);
    for (const make of [waiting, starting, running, paused]) {
      const { state, screen } = make();
      view.update(state, screen);
      const head = /** @type {HTMLElement} */ (stage.querySelector('.opt-rp-head'));
      expect(head.contains(button(stage, 'pause'))).toBe(true);
      expect(head.contains(button(stage, 'cancel'))).toBe(true);
    }
    expect(setPrimary.mock.calls.every(([spec]) => spec === null)).toBe(true);
    view.unmount?.();
  });

  test('Cancel asks first; only the confirmation sends it, and it says nothing is restored', () => {
    const { ctx, stage, actions } = context();
    const view = mountRunPage(ctx);
    const first = running();
    view.update(first.state, first.screen);
    button(stage, 'cancel').click();
    expect(actions.requestCancel).toHaveBeenCalledTimes(1);
    expect(actions.confirmCancel).not.toHaveBeenCalled();
    expect(confirmPop().hidden).toBe(true);

    const armed = running();
    const state = reduce(armed.state, { type: 'control/arm', armed: true });
    view.update(state, deriveScreen(state));
    expect(confirmPop().hidden).toBe(false);
    expect(confirmPop().textContent).toContain(STOP_WORDS.now);
    /** @type {HTMLElement} */ (confirmPop().querySelector('[data-confirm="keep"]')).click();
    expect(actions.dismissCancel).toHaveBeenCalledTimes(1);
    /** @type {HTMLElement} */ (confirmPop().querySelector('[data-confirm="cancel"]')).click();
    expect(actions.confirmCancel).toHaveBeenCalledTimes(1);
    view.unmount?.();
  });

  test('each control names the job the page shows', () => {
    const { ctx, stage, actions } = context();
    const view = mountRunPage(ctx);
    const run = running();
    const id = run.screen.job.job_id;
    view.update(run.state, run.screen);
    button(stage, 'pause').click();
    expect(actions.pause).toHaveBeenCalledWith(id);
    const held = paused();
    view.update(held.state, held.screen);
    button(stage, 'pause').click();
    expect(actions.resume).toHaveBeenCalledWith(id);
    const state = reduce(run.state, { type: 'control/arm', armed: true });
    view.update(state, deriveScreen(state));
    /** @type {HTMLElement} */ (confirmPop().querySelector('[data-confirm="cancel"]')).click();
    expect(actions.confirmCancel).toHaveBeenCalledWith(id);
    view.unmount?.();
  });

  test('a confirmation closed by the page (Escape, a click elsewhere) disarms the cancel', async () => {
    const { ctx, actions } = context();
    const view = mountRunPage(ctx);
    const armed = running();
    const state = reduce(armed.state, { type: 'control/arm', armed: true });
    view.update(state, deriveScreen(state));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(confirmPop().hidden).toBe(true);
    expect(actions.dismissCancel).toHaveBeenCalled();
    expect(actions.confirmCancel).not.toHaveBeenCalled();
    view.unmount?.();
  });

  test('while the run cancels no control is enabled and the header says so', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = at({ status: 'cancelling', write_authorization: granted() });
    expect(screen.cancelling).toBe(true);
    view.update(state, screen);
    expect(button(stage, 'pause').disabled).toBe(true);
    expect(button(stage, 'cancel').disabled).toBe(true);
    expect(stage.querySelector('.opt-rp-sentence')?.textContent).toBe(nowSentence(screen));
    view.unmount?.();
  });

  test('a refused control shows its reason in a live region', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = at({ write_authorization: granted() }, [
      { type: 'control/sent', action: 'pause' },
      { type: 'control/answered', error: 'the job is not running' },
    ]);
    view.update(state, screen);
    const region = /** @type {HTMLElement} */ (stage.querySelector('.opt-rp-refusal'));
    expect(region.getAttribute('aria-live')).toBeTruthy();
    expect(region.textContent).toContain('the job is not running');
    view.unmount?.();
  });

  test('a paused run offers Resume in the same place; it sends resume', () => {
    const { ctx, stage, actions } = context();
    const view = mountRunPage(ctx);
    const run = running();
    view.update(run.state, run.screen);
    const slot = button(stage, 'pause');
    const { state, screen } = paused();
    view.update(state, screen);
    expect(button(stage, 'pause')).toBe(slot);
    expect(slot.disabled).toBe(false);
    slot.click();
    expect(actions.resume).toHaveBeenCalledTimes(1);
    expect(actions.pause).not.toHaveBeenCalled();
    view.unmount?.();
  });

  test('a pause asked while first tries remain does not show as holding; after them it does', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const seeding = pausedAfter(2);
    view.update(seeding.state, seeding.screen);
    // The first tries still write: no pause marker, the last point stays live.
    expect(stage.querySelector('.opt-ch-end')).toBeNull();
    expect(stage.querySelector('.opt-ch-cur')?.getAttribute('class')).toContain('is-live');
    const guided = pausedAfter(5);
    view.update(guided.state, guided.screen);
    expect(stage.querySelector('.opt-ch-end')).not.toBeNull();
    expect(stage.querySelector('.opt-ch-cur')?.getAttribute('class')).not.toContain('is-live');
    view.unmount?.();
  });

  test('a poll keeps the controls and the chart; a new evaluation updates them in place', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const one = running();
    view.update(one.state, one.screen);
    const pause = button(stage, 'pause');
    const cancel = button(stage, 'cancel');
    const svg = stage.querySelector('svg');
    const sentence = stage.querySelector('.opt-rp-sentence')?.textContent;
    const data = [...runningJob().data, { phase_a: 3, sol_b: 2.5, charge: 3.1 }];
    const two = at({ write_authorization: granted(), data });
    view.update(two.state, two.screen);
    expect(button(stage, 'pause')).toBe(pause);
    expect(button(stage, 'cancel')).toBe(cancel);
    expect(stage.querySelector('svg')).toBe(svg);
    expect(stage.querySelectorAll('.opt-ch-pt')).toHaveLength(4);
    expect(stage.querySelector('.opt-rp-sentence')?.textContent).not.toBe(sentence);
    view.unmount?.();
  });

  test('nothing on the waiting page approves a request or starts a run', () => {
    const { ctx, stage, inspector, actions } = context();
    const view = mountRunPage(ctx);
    const { state, screen } = waiting();
    view.update(state, screen);
    for (const node of [
      ...stage.querySelectorAll('button'),
      ...inspector.querySelectorAll('button'),
    ]) {
      /** @type {HTMLButtonElement} */ (node).click();
    }
    const called = ACTIONS.filter((name) => actions[name].mock.calls.length > 0);
    expect(called.every((name) => ['pause', 'requestCancel', 'dismissCancel'].includes(name))).toBe(
      true
    );
    expect(actions.start).not.toHaveBeenCalled();
    view.unmount?.();
  });
});

describe('the run page words', () => {
  test('a pause takes effect where the runner checks for it', () => {
    // Three first tries: no check between them; one before each guided step and the final write.
    expect(pausePhase(pausedAfter(0).screen)).toBe('first');
    expect(pausePhase(pausedAfter(2).screen)).toBe('first');
    expect(pausePhase(pausedAfter(3).screen)).toBe('step');
    expect(pausePhase(pausedAfter(33).screen)).toBe('end');
    // Without the plan the page does not claim either.
    const unplanned = { ...pausedAfter(2).screen, config: null };
    expect(pausePhase(unplanned)).toBeNull();
  });

  test('while first tries remain, the pause is described as not yet holding', () => {
    const seeding = pausedAfter(2).screen;
    const guided = pausedAfter(5).screen;
    expect(pauseWords(seeding).held).not.toBe(pauseWords(guided).held);
    expect(pauseWords(seeding).ask).not.toBe(pauseWords(guided).ask);
    expect(methodLine(seeding)).not.toBe(methodLine(guided));
    expect(nowSentence(seeding)).not.toBe(nowSentence(guided));
  });

  test('the header names the evaluation, never a step inside it', () => {
    expect(nowSentence(running().screen)).toMatch(/\b4\b.*\b10\b/);
    expect(nowSentence(waiting().screen)).toContain('carol');
    const done = at({
      write_authorization: granted(),
      data: Array.from({ length: 10 }, () => ({ phase_a: 0, sol_b: 3, charge: 1 })),
    }).screen;
    expect(nowSentence(done)).not.toBe(nowSentence(running().screen));
  });

  test('the line under it follows the first tries, then the method', () => {
    // UCB with two variables: three first tries, then 30 iterations planned.
    const lines = [0, 1, 3].map((n) =>
      methodLine(
        at({
          write_authorization: granted(),
          total_evaluations: 33,
          data: runningJob().data.slice(0, n),
        }).screen
      )
    );
    expect(new Set(lines).size).toBe(3);
    expect(lines[1]).toContain('2 of 3');
    expect(lines[2]).toContain('1 of 30');
  });

  test('time left shows only when the figures have it', () => {
    const { screen } = running();
    expect(screen.figures?.minutesLeft).toBeNull();
    const keys = runFacts(screen, 'nC')
      .filter((f) => f.value != null)
      .map((f) => f.key);
    expect(keys).not.toContain('left');
    expect(keys).toContain('best');
  });
});
