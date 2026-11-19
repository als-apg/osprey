/**
 * Tests for the stage of a draft (draft-stage.js): the sentence of what a run
 * does names every knob it writes, the fixed knobs of a model file included,
 * and the stage says when a run may write in terms that hold also when the
 * owner's write switch allows writes by itself.
 */

import { describe, test, expect } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import {
  draftStage,
  leadParts,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-stage.js';
import { BEFORE_A_WRITE } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

/** @param {(Node|string)[]} parts */
const text = (parts) =>
  parts.map((p) => (typeof p === 'string' ? p : p.textContent || '')).join('');

describe('what a run does to the machine', () => {
  const env = /** @type {any} */ (envDetails());
  const model = { label: 'm', path: '/results/models/m.joblib', variables: ['phase_a', 'kick_c'] };

  test('a model file that fixes more knobs makes the sentence name them', () => {
    // The draft tunes phase_a and sol_b; the model also holds kick_c, which
    // the run writes at its value at the start.
    const plain = text(leadParts(storedConfig({ objective_direction: 'MAXIMIZE' }), env, [model]));
    const fixed = storedConfig({ objective_direction: 'MAXIMIZE', model_file: model.path });
    const withModel = text(leadParts(fixed, env, [model]));
    expect(withModel).not.toBe(plain);
    expect(withModel).toContain('1');
    // A model the service does not list: the sentence says more knobs may be written.
    const unknown = text(leadParts(fixed, env, []));
    expect(unknown).not.toBe(plain);
    expect(unknown).not.toBe(withModel);
  });

  test('a model that fixes no knob beyond the tuned ones changes nothing', () => {
    // A method that is not Bayesian, so the model does not change the evaluations.
    const config = storedConfig({ algorithm: 'random', model_file: '/results/models/n.joblib' });
    const none = { label: 'n', path: '/results/models/n.joblib', variables: ['phase_a'] };
    const plain = storedConfig({ algorithm: 'random' });
    expect(text(leadParts(config, env, [none]))).toBe(text(leadParts(plain, env, [])));
  });

  test('when a run may write is said for a grant by the switch as well as by a person', () => {
    // A run writes its own grant while the owner's switch stands at "allowed".
    expect(BEFORE_A_WRITE).toMatch(/switch/);
    const stage = draftStage({
      config: storedConfig(),
      env,
      models: [],
      values: {},
      objectiveNow: null,
    });
    const root = document.createElement('div');
    root.append(...stage);
    expect(root.textContent).toContain(BEFORE_A_WRITE);
  });
});
