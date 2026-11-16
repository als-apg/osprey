/**
 * Tests for the settings of a stored config (config-sections.js): every
 * stored field is on screen, and the flags of approval-model.js mark the
 * fields that change what a run does to the machine.
 */

import { describe, test, expect } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import {
  MOVED_WINDOW,
  configSections,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/config-sections.js';

/** @param {HTMLElement[]} sections */
function mount(sections) {
  const root = document.createElement('div');
  root.append(...sections);
  return root;
}

/** @param {HTMLElement} root @param {string} key */
function field(root, key) {
  const dd = root.querySelector(`[data-field="${key}"]`);
  return {
    text: dd?.textContent || '',
    flagged: !!dd?.classList.contains('is-flagged'),
    tone: dd?.classList.contains('is-err')
      ? 'err'
      : dd?.classList.contains('is-warn')
        ? 'warn'
        : null,
  };
}

describe('the settings of a stored config', () => {
  test('every stored field is shown, with what the service derives from it', () => {
    const root = mount(
      configSections(
        storedConfig({
          constraints: { loss: { type: 'LESS_THAN', value: 3 } },
          algorithm_params: { beta: 2 },
          model_file: 'm.joblib',
          surrogate_model: 'heteroskedastic',
          environment_config: { noise: 0.1 },
          on_complete: 'hold',
          preset: 'phases',
        }),
        /** @type {any} */ (envDetails()),
        [{ label: 'm', path: 'm.joblib', variables: ['phase_a', 'sol_b', 'kick_c'] }],
        { environmentLabel: 'Injector' }
      )
    );
    expect(
      [...root.querySelectorAll('[data-section]')].map((s) => s.getAttribute('data-section'))
    ).toEqual(['where', 'goal', 'knobs', 'search', 'end', 'advanced']);
    expect(field(root, 'target').text).toContain('va');
    expect(field(root, 'environment').text).toContain('injector');
    expect(field(root, 'preset').text).toBe('phases');
    expect(field(root, 'objective').text).toContain('Charge');
    expect(field(root, 'constraints').text).toContain('3');
    expect(field(root, 'iterations').text).toBe('30');
    // A pre-trained model on a Bayesian method: one initial sample.
    expect(field(root, 'initial').text).toContain('1');
    expect(field(root, 'evaluations').text).toBe('31');
    expect(field(root, 'on-complete').text).toContain('hold');
    expect(field(root, 'algorithm-params').text).toMatch(/beta\W+2/);
    expect(field(root, 'surrogate').text).toBe('heteroskedastic');
    expect(field(root, 'model-file').text).toContain('kick_c');
    expect(field(root, 'environment-config').text).toContain('"noise": 0.1');
    const knobs = [...root.querySelectorAll('[data-knob]')].map((k) => k.getAttribute('data-knob'));
    expect(knobs).toEqual(['phase_a', 'sol_b']);
    expect(root.querySelector('[data-knob="phase_a"]')?.textContent).toContain('-18 to 18');
  });

  test('the real machine, a direction against the observable and a read-only run are flagged', () => {
    const root = mount(
      configSections(
        storedConfig({
          control_target: 'live',
          objective_direction: 'MINIMIZE',
          execution_mode: 'readonly',
        }),
        /** @type {any} */ (envDetails()),
        []
      )
    );
    expect(field(root, 'target')).toMatchObject({ flagged: true, tone: 'err' });
    expect(field(root, 'direction')).toMatchObject({ flagged: true, tone: 'warn' });
    expect(field(root, 'direction').text).toContain('MAXIMIZE');
    expect(field(root, 'execution-mode').flagged).toBe(true);
  });

  test('before the environment is read the direction is not verified; a missing target is flagged', () => {
    const root = mount(configSections(storedConfig({ control_target: null }), null, null));
    expect(field(root, 'direction').flagged).toBe(true);
    expect(field(root, 'target')).toMatchObject({ flagged: true, tone: 'err' });
    expect(field(root, 'surrogate').text).toBe('default');
  });

  test('a window the service moved and an unresolved one mark their knob', () => {
    const config = storedConfig({
      variable_overrides: {
        phase_a: { min: -18, max: 18, step_size: 0.5 },
        sol_b: { min: null, max: 4, step_size: 0.01 },
      },
    });
    const root = mount(
      configSections(config, /** @type {any} */ (envDetails()), [], { moved: ['phase_a'] })
    );
    expect(root.querySelector('[data-knob="phase_a"]')?.classList.contains('is-warn')).toBe(true);
    expect(root.querySelector('[data-knob="sol_b"]')?.classList.contains('is-err')).toBe(true);
  });

  test('a model the service does not list leaves its fixed channels unknown, and says so', () => {
    const root = mount(
      configSections(
        storedConfig({ model_file: 'gone.joblib' }),
        /** @type {any} */ (envDetails()),
        []
      )
    );
    expect(field(root, 'model-file').flagged).toBe(true);
  });

  test("a direction the service filled in reads as the objective's own, not as set in the draft", () => {
    const env = /** @type {any} */ (envDetails());
    // The service writes charge's own MAXIMIZE into every stored draft.
    const filled = mount(
      configSections(storedConfig({ objective_direction: 'MAXIMIZE' }), env, [])
    );
    const unset = mount(configSections(storedConfig({ objective_direction: null }), env, []));
    expect(field(filled, 'direction')).toEqual(field(unset, 'direction'));
    const against = mount(
      configSections(storedConfig({ objective_direction: 'MINIMIZE' }), env, [])
    );
    expect(field(against, 'direction').text).not.toBe(field(filled, 'direction').text);
  });

  test('a window the service moved is not said to be clamped only: a snap to the step grid counts', () => {
    // movedKnobs counts a snap to the step grid as well as a clamp to the range.
    expect(MOVED_WINDOW).toMatch(/step/);
  });
});
