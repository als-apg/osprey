/**
 * Tests for the set-up's pure helpers (setup-model.js): the config the stage
 * draws while the form is edited, where a refusal reason belongs, a preset
 * applied over the form, the direction when the objective changes, what a
 * set-up seeds from, a draft that changed on the service, the knob filter.
 */

import { describe, test, expect } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { formFromEnv } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';
import {
  directionAfter,
  draftMoved,
  groupIssues,
  issueFields,
  knobMatches,
  missingKnobs,
  previewConfig,
  seedSource,
  withPreset,
  withUnlisted,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-model.js';
import {
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

const KNOBS = ['phase_a', 'sol_b', 'SR:MAG:VCM:01:CURRENT:SP', 'SR:MAG:VCM:01:CURRENT:SP2'];

describe('previewConfig', () => {
  test('draws the tuned knobs and leaves out a constraint not yet complete', () => {
    const form = formFromEnv(/** @type {any} */ (envDetails()));
    form.vars[1].include = false;
    form.vars[0].max = null;
    form.constraints.push({ observable: 'loss', type: 'LESS_THAN', value: null });
    const config = previewConfig(form, { environment: 'injector', target: '' });
    expect(config.variables).toEqual(['phase_a']);
    expect(config.variable_overrides).toEqual({
      phase_a: { min: -180, max: null, step_size: 0.5 },
    });
    expect(config.constraints).toEqual({});
    expect(config.control_target).toBeNull();
  });
});

describe('where a refusal reason belongs', () => {
  test.each([
    ['phase_a: min exceeds max.', ['knob:phase_a']],
    ['SR:MAG:VCM:01:CURRENT:SP2: step must be positive.', ['knob:SR:MAG:VCM:01:CURRENT:SP2']],
    [
      'variable_overrides.SR:MAG:VCM:01:CURRENT:SP: window outside',
      ['knob:SR:MAG:VCM:01:CURRENT:SP'],
    ],
    ['Pick a control target.', ['target']],
    ['Pick an environment.', ['environment']],
    ["Environment 'injector' has no preset 'x'.", ['preset']],
    ['charge is the objective; it cannot also be a constraint.', ['constraints']],
    ['Pick an objective.', ['objective']],
    ['Iterations must be 0 or more.', ['iterations']],
    ['Initial samples must be 1 or more.', ['initial']],
    ['Algorithm params must be a JSON object.', ['params']],
    ['Select at least one variable.', ['knobs']],
    ['the draft was refused', []],
  ])('%s', (text, fields) => {
    expect(issueFields(text, KNOBS)).toEqual(fields);
  });

  test('a knob name inside a longer identifier is not that knob', () => {
    expect(issueFields('phase_ab: min exceeds max.', ['phase_a'])).toEqual([]);
  });

  test('a reason that names several knobs is shown at each of them', () => {
    // A 422 list arrives as one reason that names every knob it refuses.
    const text =
      'variable_overrides.phase_a: outside the range; variable_overrides.sol_b: outside the range';
    expect(issueFields(text, KNOBS).sort()).toEqual(['knob:phase_a', 'knob:sol_b']);
    expect(groupIssues([text], KNOBS)).toEqual({ 'knob:phase_a': [text], 'knob:sol_b': [text] });
  });

  test('a reason that names no field is in no field', () => {
    expect(groupIssues(['Pick a control target.', 'service down'], KNOBS)).toEqual({
      target: ['Pick a control target.'],
    });
  });
});

describe('the direction when the objective changes', () => {
  const env = /** @type {any} */ (envDetails());

  test("the old objective's own direction does not follow the operator to the new one", () => {
    expect(directionAfter(env, 'charge', 'MAXIMIZE', 'loss')).toBe('');
  });

  test('a direction set against the old objective stays, and none stays none', () => {
    expect(directionAfter(env, 'charge', 'MINIMIZE', 'plain')).toBe('MINIMIZE');
    expect(directionAfter(env, 'charge', '', 'loss')).toBe('');
  });
});

test('a preset takes the knobs and what it names, and leaves the rest of the form', () => {
  const env = /** @type {any} */ (envDetails());
  const form = formFromEnv(env);
  form.direction = 'MINIMIZE';
  form.onComplete = 'hold';
  form.constraints.push({ observable: 'loss', type: 'LESS_THAN', value: 2 });
  const preset = { variables: ['sol_b'], algorithm: 'neldermead', n_iterations: null };
  const seeded = formFromEnv(env, { name: 'sols', preset, bounds: { sol_b: { min: 2, max: 3 } } });
  const next = withPreset(form, seeded, preset, env);
  expect(next.preset).toBe('sols');
  expect(next.vars.filter((v) => v.include).map((v) => v.name)).toEqual(['sol_b']);
  expect(next.algorithm).toBe('neldermead');
  expect(next.nIterations).toBe(form.nIterations);
  expect(next.objective).toBe(form.objective);
  expect(next).toMatchObject({ direction: 'MINIMIZE', onComplete: 'hold' });
  expect(next.constraints).toHaveLength(1);
});

test('a preset that names another objective drops the old objective’s own direction', () => {
  const env = /** @type {any} */ (envDetails());
  const form = formFromEnv(env);
  form.direction = 'MAXIMIZE';
  const preset = { variables: ['sol_b'], objective_name: 'loss' };
  const seeded = formFromEnv(env, { name: 'losses', preset });
  expect(withPreset(form, seeded, preset, env)).toMatchObject({ objective: 'loss', direction: '' });
});

describe('what a set-up seeds from', () => {
  /** @param {any[]} events */
  const at = (events) => events.reduce(reduce, initialState());
  const A = storedConfig({ n_iterations: 30 });
  const B = storedConfig({ n_iterations: 5 });
  const draftA = [
    { type: 'poll/job', job: { job_id: 'A', status: 'created' }, at: 0 },
    { type: 'poll/config', jobId: 'A', revision: 'rA', config: A },
  ];

  test('a draft edit seeds from the config it was opened with, also when another draft is current', () => {
    const state = at([
      ...draftA,
      {
        type: 'setup/open',
        jobId: 'A',
        environment: 'injector',
        target: 'va',
        from: A,
        revision: 'rA',
      },
      { type: 'poll/job', job: { job_id: 'B', status: 'created' }, at: 1 },
      { type: 'poll/config', jobId: 'B', revision: 'rB', config: B },
    ]);
    expect(seedSource(state)).toEqual({ config: A, revision: 'rA' });
  });

  test('a draft edit with no config read for that draft has nothing to seed from', () => {
    const state = at([
      ...draftA,
      { type: 'setup/open', jobId: 'A', environment: 'injector', target: 'va' },
      { type: 'poll/job', job: { job_id: 'B', status: 'created' }, at: 1 },
      { type: 'poll/config', jobId: 'B', revision: 'rB', config: B },
    ]);
    expect(seedSource(state)).toEqual({ config: null, revision: null });
  });

  test('a draft that changed on the service is noticed, but not while the set-up saves it', () => {
    const open = at([
      ...draftA,
      {
        type: 'setup/open',
        jobId: 'A',
        environment: 'injector',
        target: 'va',
        from: A,
        revision: 'rA',
      },
    ]);
    expect(draftMoved(open, 'rA')).toBe(false);
    const changed = reduce(open, { type: 'poll/config', jobId: 'A', revision: 'rA2', config: B });
    expect(draftMoved(changed, 'rA')).toBe(true);
    expect(draftMoved(changed, 'rA2')).toBe(false);
    expect(draftMoved(reduce(changed, { type: 'setup/saving' }), 'rA')).toBe(false);
  });
});

test('knobs a stored draft names that its environment no longer has are listed', () => {
  const config = storedConfig({ variables: ['phase_a', 'gone_c'] });
  expect(missingKnobs(config, /** @type {any} */ (envDetails()))).toEqual(['gone_c']);
});

test('a picker value no option holds gets an option of its own', () => {
  /** @type {[string, string][]} */
  const options = [
    ['', 'None'],
    ['a.joblib', 'a'],
  ];
  expect(withUnlisted(options, 'b.joblib').map(([v]) => v)).toEqual(['', 'a.joblib', 'b.joblib']);
  expect(withUnlisted(options, 'a.joblib')).toBe(options);
  expect(withUnlisted(options, '')).toBe(options);
});

test('the knob filter matches every word in the identifier or the description', () => {
  expect(knobMatches('phase_a', { description: 'Buncher phase' }, 'bunch')).toBe(true);
  expect(knobMatches('phase_a', { description: 'Buncher phase' }, 'phase buncher')).toBe(true);
  expect(knobMatches('sol_b', { description: '' }, 'phase')).toBe(false);
  expect(knobMatches('sol_b', null, '  ')).toBe(true);
});
