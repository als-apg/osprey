/**
 * Tests for the run page (run-page.js): what its stage shows while a run
 * waits for its write approval, runs or is paused, the chart and the table
 * of the evaluations, and where its controls are. The header
 * (run-header.test.mjs), the inspector (run-knobs.test.mjs) and the words
 * (run-page-model.test.mjs) have their own tests.
 *
 * The safety rules it holds: a run that waits for approval looks different
 * from a running one (warning colour, no chart, "Nothing has been written");
 * Pause and Cancel are in the run header and never in the primary slot; the
 * page never approves a write request; a pause is not said to hold while the
 * run still writes.
 */

import { afterEach, describe, test, expect, vi } from 'vitest';

import { runningJob } from './_fixtures.mjs';
import { granted, paused, pausedAfter, runAt, running, starting, waiting } from './_run.mjs';
import { mountRunPage } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-page.js';
import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';

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

/** @param {HTMLElement} root @param {string} control */
const button = (root, control) =>
  /** @type {HTMLButtonElement} */ (root.querySelector(`[data-control="${control}"]`));

afterEach(() => {
  document.body.replaceChildren();
});

describe('the run page', () => {
  test('a run that waits for approval: warning, no chart, nothing written, who approves', () => {
    const { ctx, stage } = context();
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
    expect(inspector.querySelectorAll('[data-knob]')).toHaveLength(2);
    view.unmount?.();
  });

  test('the best on the page is the one the service applies: an evaluation that fails a constraint is not it', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const data = [
      { phase_a: 0, sol_b: 3, charge: 1.0, loss: 0.5 },
      { phase_a: 9, sol_b: 2, charge: 2.5, loss: 3 },
      { phase_a: -9, sol_b: 4, charge: 2.0, loss: 0.2 },
    ];
    const constraints = { loss: { type: 'LESS_THAN', value: 1 } };
    const { state, screen } = runAt({ write_authorization: granted(), data }, [], { constraints });
    expect(screen.figures?.bestIndex).toBe(2);
    view.update(state, screen);
    expect(stage.querySelectorAll('.opt-ch-pt.is-bad')).toHaveLength(1);
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

  test('a poll keeps the chart; a new evaluation updates it in place', () => {
    const { ctx, stage } = context();
    const view = mountRunPage(ctx);
    const one = running();
    view.update(one.state, one.screen);
    const svg = stage.querySelector('svg');
    const data = [...runningJob().data, { phase_a: 3, sol_b: 2.5, charge: 3.1 }];
    const two = runAt({ write_authorization: granted(), data });
    view.update(two.state, two.screen);
    expect(stage.querySelector('svg')).toBe(svg);
    expect(stage.querySelectorAll('.opt-ch-pt')).toHaveLength(4);
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
