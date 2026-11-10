/**
 * Unit tests for the draft form's state (draft-model.js): seeding from an
 * environment, a preset and resolved bounds; loading a stored draft; the
 * config body the relay is sent.
 */

import { test, expect, describe } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import {
  buildConfig,
  formFromConfig,
  formFromEnv,
  parseNumber,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';

const WHERE = { environment: 'injector', target: 'va' };

describe('formFromEnv', () => {
  test('no preset: every variable at its safety range and default step', () => {
    const form = formFromEnv(envDetails());
    expect(form.vars).toEqual([
      { name: 'phase_a', include: true, min: -180, max: 180, step: 0.5 },
      { name: 'sol_b', include: true, min: 0, max: 10, step: 0.01 },
    ]);
    expect(form.objective).toBe('charge');
  });

  test('a preset selects its variables and takes the resolved bounds and its steps', () => {
    const preset = {
      variables: ['phase_a'],
      variable_overrides: { phase_a: { step_size: 1 } },
      algorithm: 'upper_confidence_bound',
      n_iterations: 40,
      objective_name: 'loss',
    };
    const bounds = { phase_a: { min: -10, max: 10, was_clamped: false } };
    const form = formFromEnv(envDetails(), { name: 'phases', preset, bounds });
    expect(form.vars[0]).toEqual({ name: 'phase_a', include: true, min: -10, max: 10, step: 1 });
    expect(form.vars[1].include).toBe(false);
    expect(form).toMatchObject({
      preset: 'phases',
      algorithm: 'upper_confidence_bound',
      nIterations: 40,
      objective: 'loss',
    });
  });
});

describe('formFromConfig → buildConfig', () => {
  test('a stored draft round-trips, keeping fields the form does not edit', () => {
    const config = storedConfig({
      surrogate_model: 'xopt_gp',
      execution_mode: 'readonly',
      constraints: { loss: { type: 'LESS_THAN', value: 3 } },
      algorithm_params: { beta: 2 },
    });
    const form = formFromConfig(config, envDetails());
    const { config: out, errors } = buildConfig(form, WHERE);
    expect(errors).toEqual([]);
    expect(out).toMatchObject({
      surrogate_model: 'xopt_gp',
      execution_mode: 'readonly',
      control_target: 'va',
      variables: ['phase_a', 'sol_b'],
      variable_overrides: {
        phase_a: { min: -18, max: 18, step_size: 0.5 },
        sol_b: { min: 2, max: 4, step_size: 0.01 },
      },
      constraints: { loss: { type: 'LESS_THAN', value: 3 } },
      algorithm_params: { beta: 2 },
      n_initial_samples: null,
      objective_direction: null,
    });
  });

  test('the target and environment come from the pickers, never a default', () => {
    const form = formFromEnv(envDetails());
    expect(buildConfig(form, { environment: 'injector', target: '' }).errors).toContain(
      'Pick a control target.'
    );
  });

  test('invalid input is refused with reasons, not sent', () => {
    const form = formFromEnv(envDetails());
    form.vars[0].min = 5;
    form.vars[0].max = 1;
    form.vars[1].step = 0;
    form.algorithmParams = '[1]';
    form.constraints.push({ observable: 'loss', type: 'LESS_THAN', value: null });
    const { config, errors } = buildConfig(form, WHERE);
    expect(config).toBeNull();
    expect(errors).toEqual([
      'phase_a: min exceeds max.',
      'sol_b: step must be positive.',
      'Each constraint needs an observable, a relation and a threshold.',
      'Algorithm params must be a JSON object.',
    ]);
  });

  test('no selected variable is an error, never "all variables"', () => {
    const form = formFromEnv(envDetails());
    for (const v of form.vars) v.include = false;
    expect(buildConfig(form, WHERE).errors).toContain('Select at least one variable.');
  });
});

test('parseNumber: blank is null, junk is null', () => {
  expect(parseNumber(' ')).toBeNull();
  expect(parseNumber('2.5')).toBe(2.5);
  expect(parseNumber('abc')).toBeNull();
});
