/**
 * DOM tests for the approval surface (approval-view.js): every approved field
 * is on screen, the flags draw the eye, and Start follows the gate.
 */

import { test, expect, vi } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { renderApproval } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-view.js';
import {
  approvalInit,
  reduceApproval,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';

/** @param {any} config */
function approved(config) {
  return reduceApproval(approvalInit(), {
    type: 'draft',
    jobId: 'j1',
    revision: 'rev-abcdef123456',
    config,
  });
}

/** @param {HTMLElement} root @param {string} label */
function rowText(root, label) {
  const row = [...root.querySelectorAll('.appr-row')].find(
    (r) => r.querySelector('dt')?.textContent === label
  );
  return {
    text: row?.querySelector('dd')?.textContent || '',
    flagged: !!row?.classList.contains('flagged'),
  };
}

/**
 * @param {any} [config]
 * @param {{ enabled: boolean, reason: string|null }} [gate]
 * @param {any} [approval]
 */
function render(
  config = storedConfig(),
  gate = { enabled: true, reason: null },
  approval = approved(config),
  env = /** @type {any} */ (envDetails())
) {
  const root = document.createElement('div');
  const handlers = { onStart: vi.fn(), onReviewed: vi.fn() };
  renderApproval(
    root,
    {
      approval,
      env,
      models: [{ label: 'm', path: 'm.joblib', variables: ['phase_a', 'sol_b', 'kick_c'] }],
      readbacks: { phase_a: 0.5 },
      gate,
    },
    handlers
  );
  return { root, handlers };
}

test('shows every field the operator approves', () => {
  const { root } = render(
    storedConfig({
      control_target: 'live',
      objective_direction: 'MINIMIZE',
      on_complete: 'hold',
      constraints: { loss: { type: 'LESS_THAN', value: 3 } },
      algorithm_params: { beta: 2 },
      model_file: 'm.joblib',
      surrogate_model: 'heteroskedastic',
      environment_config: { noise: 0.1 },
    })
  );
  expect(rowText(root, 'Control target')).toEqual({ text: 'livereal machine', flagged: true });
  expect(rowText(root, 'Objective').text).toBe('charge');
  expect(rowText(root, 'Direction').flagged).toBe(true);
  expect(rowText(root, 'Direction').text).toContain("differs from the observable's MAXIMIZE");
  expect(rowText(root, 'Iterations').text).toBe('30');
  expect(rowText(root, 'Initial samples').text).toBe('1 (auto: pre-trained model: 1)');
  expect(rowText(root, 'On complete').text).toBe('hold — hold the last point tried');
  expect(rowText(root, 'Constraints').text).toBe('loss < 3');
  expect(rowText(root, 'Algorithm params').text).toBe('{ "beta": 2 }');
  expect(rowText(root, 'Model file').text).toBe('m.joblib — fixed at current values: kick_c');
  expect(rowText(root, 'Surrogate model').text).toBe('heteroskedastic');
  expect(rowText(root, 'Environment config').text).toBe('{ "noise": 0.1 }');
  expect(root.textContent).toContain('revision rev-abcdef');
});

test('the derived initial-sample count is shown with its rule', () => {
  const { root } = render();
  expect(rowText(root, 'Initial samples').text).toBe('3 (auto: max(2, 2 + 1))');
  expect(rowText(root, 'Direction')).toEqual({
    text: 'MAXIMIZE (observable default)',
    flagged: false,
  });
});

test("before the draft's environment loads, the direction reads unverified", () => {
  const config = storedConfig({ objective_direction: 'MINIMIZE' });
  const { root } = render(config, { enabled: false, reason: 'x' }, approved(config), null);
  expect(rowText(root, 'Direction')).toEqual({
    text: 'MINIMIZE (set in the draft)direction unverified',
    flagged: true,
  });
  expect(rowText(root, 'Environment config').text).toBe('none');
  expect(rowText(root, 'Surrogate model').text).toBe('default');
});

test('the variables table carries resolved bounds, step, max step, readback and range %', () => {
  const { root } = render();
  const cells = [...root.querySelectorAll('.appr-vars tbody tr')[0].querySelectorAll('td')].map(
    (c) => c.textContent
  );
  expect(cells).toEqual(['phase_a [deg]', '-18', '18', '0.5', '20', '0.5', '10.0%']);
});

test('Start follows the gate; a changed draft shows the notice and the review button', () => {
  const { root, handlers } = render();
  const start = /** @type {HTMLButtonElement} */ (root.querySelector('#start-btn'));
  expect(start.disabled).toBe(false);
  start.click();
  expect(handlers.onStart).toHaveBeenCalledOnce();

  const changed = reduceApproval(approved(storedConfig()), {
    type: 'draft',
    jobId: 'j1',
    revision: 'rev-2',
    config: storedConfig({ n_iterations: 99 }),
  });
  const again = render(
    storedConfig(),
    { enabled: false, reason: 'Review the changed draft first.' },
    changed
  );
  expect(/** @type {HTMLButtonElement} */ (again.root.querySelector('#start-btn')).disabled).toBe(
    true
  );
  expect(again.root.querySelector('.banner-warn')?.textContent).toContain('draft changed');
  expect(rowText(again.root, 'Iterations').text).toBe('99');
  /** @type {HTMLButtonElement} */ (again.root.querySelector('#reviewed-btn')).click();
  expect(again.handlers.onReviewed).toHaveBeenCalledOnce();
});

test('no draft: a plain empty state, no Start', () => {
  const root = document.createElement('div');
  renderApproval(
    root,
    { approval: approvalInit(), env: null, models: null, gate: { enabled: false, reason: null } },
    { onStart: vi.fn(), onReviewed: vi.fn() }
  );
  expect(root.querySelector('#start-btn')).toBeNull();
});
