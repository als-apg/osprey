/**
 * Tests for the run page's inspector (run-knobs.js): one card per knob, in
 * the window the run used; what a waiting run's request covers; the start,
 * the best so far and the last evaluation recorded, never called what is on
 * the machine; the write approval's rows; the cards and the settings fold
 * kept across polls.
 */

import { afterEach, describe, test, expect, vi } from 'vitest';

import { envDetails, runningJob } from './_fixtures.mjs';
import { granted, runAt, running, starting, waiting } from './_run.mjs';
import { authorizationScope } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';
import {
  KNOB_KEYS,
  position,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/knob-bar.js';
import { approvalRows } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-page-model.js';
import { mountRunInspector } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-knobs.js';
import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').State} State
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').Screen} Screen
 */

const NOW = Date.parse('2026-09-19T12:00:00Z');

function mount() {
  const inspector = document.createElement('aside');
  document.body.append(inspector);
  const ctx = {
    stage: document.createElement('section'),
    inspector,
    actions: /** @type {any} */ ({}),
    shell: {
      setPrimary: vi.fn(),
      fillInspector: (/** @type {any} */ parts) =>
        inspector.replaceChildren(...inspectorFrame(parts, () => {})),
      setDrawer: vi.fn(),
      setDrawerTop: vi.fn(),
    },
  };
  const view = mountRunInspector(ctx);
  return {
    inspector,
    /** @param {{ state: State, screen: Screen }} at */
    show: ({ state, screen }) =>
      view.update(state, screen, {
        env: envDetails(),
        revision: null,
        objective: 'Charge',
        environment: 'Injector',
        now: NOW,
      }),
  };
}

/** @param {HTMLElement} root */
const cards = (root) => Array.from(root.querySelectorAll('[data-knob]'));

/** @param {HTMLElement} root @param {string} knob @param {string} mark */
const markAt = (root, knob, mark) =>
  /** @type {HTMLElement|null} */ (
    root.querySelector(`[data-knob="${knob}"] [data-mark="${mark}"]`)
  );

afterEach(() => {
  document.body.replaceChildren();
});

describe('the knob cards', () => {
  test("a run that waits lists what its request covers, named by the knobs' descriptions", () => {
    const { inspector, show } = mount();
    show(waiting());
    const knobs = cards(inspector);
    expect(knobs.map((k) => k.getAttribute('data-knob'))).toEqual(['A:SP', 'B:SP']);
    expect(knobs[0].textContent).toContain('Buncher phase');
  });

  test('the cards of a waiting run show the value at the start, not a value of now', () => {
    const { inspector, show } = mount();
    show(waiting());
    expect(inspector.querySelector('[data-mark="now"]')).toBeNull();
    expect(inspector.querySelector('[data-key="now"]')).toBeNull();
    expect(markAt(inspector, 'A:SP', 'start')).not.toBeNull();
  });

  test('a running run has a card per knob at its last evaluation recorded, with the best so far', () => {
    const { inspector, show } = mount();
    show(running());
    const knobs = cards(inspector);
    expect(knobs.map((k) => k.getAttribute('data-knob'))).toEqual(['phase_a', 'sol_b']);
    // The last evaluation recorded: phase_a -9, from 1 at the start.
    expect(knobs[0].querySelector('.opt-kn-val')?.textContent).toContain('-9');
    expect(markAt(inspector, 'phase_a', 'best')).not.toBeNull();
  });

  test('the knob cards do not call the last evaluation recorded what is on the machine', () => {
    const { inspector, show } = mount();
    show(running());
    expect(inspector.textContent).not.toContain(KNOB_KEYS.current);
    expect(inspector.textContent).toContain(KNOB_KEYS.recorded);
  });

  test('the best mark is the best the service applies: an evaluation that fails a constraint is not it', () => {
    const { inspector, show } = mount();
    const data = [
      { phase_a: 0, sol_b: 3, charge: 1.0, loss: 0.5 },
      { phase_a: 9, sol_b: 2, charge: 2.5, loss: 3 },
      { phase_a: -9, sol_b: 4, charge: 2.0, loss: 0.2 },
    ];
    const constraints = { loss: { type: 'LESS_THAN', value: 1 } };
    show(runAt({ write_authorization: granted(), data }, [], { constraints }));
    // The best mark of a knob sits at the counted best (phase_a -9), not at 9.
    const best = markAt(inspector, 'phase_a', 'best');
    const recorded = markAt(inspector, 'phase_a', 'current');
    expect(best?.getAttribute('style')).toBe(recorded?.getAttribute('style'));
  });

  test('a card is drawn in the window the run used, else the stored window; a knob with neither nor a safe range is left out', () => {
    const { inspector, show } = mount();
    const startAt = () => markAt(inspector, 'phase_a', 'start')?.style.left;
    // The start of phase_a is 1.
    show(runAt({ write_authorization: granted(), variable_bounds: { phase_a: [-9, 9] } }));
    expect(startAt()).toBe(`${position(1, -9, 9)?.toFixed(2)}%`);
    show(
      runAt({ write_authorization: granted(), variable_bounds: null }, [], {
        variables: ['phase_a', 'sol_b', 'ghost'],
      })
    );
    // The stored window of phase_a is -18 to 18.
    expect(startAt()).toBe(`${position(1, -18, 18)?.toFixed(2)}%`);
    expect(cards(inspector).map((k) => k.getAttribute('data-knob'))).toEqual(['phase_a', 'sol_b']);
  });

  test('with no knob to show, the note shows in place of the legend', () => {
    const { inspector, show } = mount();
    const run = running();
    show({
      ...run,
      screen: { ...run.screen, config: null, job: { ...run.screen.job, variable_names: [] } },
    });
    expect(cards(inspector)).toHaveLength(0);
    expect(/** @type {HTMLElement} */ (inspector.querySelector('.opt-note')).hidden).toBe(false);
    expect(/** @type {HTMLElement} */ (inspector.querySelector('.opt-kn-legend')).hidden).toBe(
      true
    );
    show(running());
    expect(/** @type {HTMLElement} */ (inspector.querySelector('.opt-note')).hidden).toBe(true);
  });

  test('a new evaluation moves the values in place: the cards and an open settings fold stay', () => {
    const { inspector, show } = mount();
    show(running());
    const card = /** @type {HTMLElement} */ (inspector.querySelector('[data-knob="phase_a"]'));
    const fold = /** @type {HTMLDetailsElement} */ (inspector.querySelector('details'));
    fold.open = true;
    const data = [...runningJob().data, { phase_a: 3, sol_b: 2.5, charge: 3.1 }];
    show(runAt({ write_authorization: granted(), data }));
    expect(inspector.querySelector('[data-knob="phase_a"]')).toBe(card);
    expect(card.querySelector('.opt-kn-val')?.textContent).toContain('3');
    expect(card.querySelector('.opt-kn-val')?.textContent).not.toContain('-9');
    expect(/** @type {HTMLDetailsElement} */ (inspector.querySelector('details')).open).toBe(true);
  });
});

describe('the run details', () => {
  /** @param {HTMLElement} root  the rows of the write approval, as label and value */
  const approvalShown = (root) => {
    const dl = /** @type {HTMLElement} */ (root.querySelector('dl'));
    const cells = Array.from(dl.children).map((n) => n.textContent);
    return {
      hidden: /** @type {HTMLElement} */ (dl.parentElement).hidden,
      rows: cells.flatMap((text, i) => (i % 2 ? [] : [[text, cells[i + 1]]])),
    };
  };

  test('the write approval shows the rows of the job’s authorization, for this session', () => {
    const { inspector, show } = mount();
    const run = running();
    show(run);
    const scope = authorizationScope(run.screen.job.write_authorization);
    expect(approvalShown(inspector)).toEqual({
      hidden: false,
      rows: approvalRows(scope, { sessionOwner: 'carol', now: NOW }),
    });
  });

  test('a run with no write authorization shows no approval section', () => {
    const { inspector, show } = mount();
    show(starting());
    expect(approvalShown(inspector)).toEqual({ hidden: true, rows: [] });
  });
});
