/**
 * DOM tests for the draft form (draft-form.js): inputs mutate the form state
 * in place, range % follows the bounds, and the submit buttons follow the
 * context's create/update/locked flags.
 */

import { test, expect, vi } from 'vitest';

import { envDetails } from './_fixtures.mjs';
import { renderDraftForm } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-form.js';
import { formFromEnv } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';

function setup(ctx = {}) {
  const root = document.createElement('div');
  const env = envDetails();
  const form = formFromEnv(env);
  const handlers = {
    onPreset: vi.fn(),
    onRefreshReadbacks: vi.fn(),
    onLoadDraft: null,
    onSubmit: vi.fn(),
  };
  renderDraftForm(
    root,
    form,
    {
      env,
      presets: { phases: { display_name: 'Phase tuning' } },
      models: [],
      readbacks: { phase_a: 1.5 },
      canCreate: true,
      canUpdate: false,
      lockedReason: null,
      message: null,
      ...ctx,
    },
    handlers
  );
  return { root, form, handlers };
}

/**
 * @param {Element|null} el
 * @param {string} value
 */
function type(el, value) {
  const input = /** @type {HTMLInputElement} */ (el);
  input.value = value;
  input.dispatchEvent(new Event('input'));
}

test('editing a bound updates the state and its range % cell', () => {
  const { root, form } = setup();
  const row = /** @type {HTMLTableRowElement} */ (root.querySelector('.form-vars tbody tr'));
  expect(row.cells[7].textContent).toBe('100.0%');
  type(root.querySelector('[aria-label="phase_a min"]'), '-18');
  type(root.querySelector('[aria-label="phase_a max"]'), '18');
  expect(form.vars[0]).toMatchObject({ min: -18, max: 18 });
  expect(row.cells[7].textContent).toBe('10.0%');
  expect(row.cells[6].textContent).toBe('1.5');
  expect(row.cells[5].textContent).toBe('20');
});

test('the preset picker and the buttons call their handlers', () => {
  const { root, handlers } = setup();
  const preset = /** @type {HTMLSelectElement} */ (root.querySelector('#preset-select'));
  preset.value = 'phases';
  preset.dispatchEvent(new Event('change'));
  expect(handlers.onPreset).toHaveBeenCalledWith('phases');
  /** @type {HTMLButtonElement} */ (root.querySelector('#create-draft-btn')).click();
  expect(handlers.onSubmit).toHaveBeenCalledWith('create');
  expect(/** @type {HTMLButtonElement} */ (root.querySelector('#update-draft-btn')).disabled).toBe(
    true
  );
});

test('a locked form disables both submits and says why', () => {
  const { root } = setup({
    lockedReason: 'A run is running; drafts wait until it ends.',
    canUpdate: true,
  });
  expect(/** @type {HTMLButtonElement} */ (root.querySelector('#create-draft-btn')).disabled).toBe(
    true
  );
  expect(/** @type {HTMLButtonElement} */ (root.querySelector('#update-draft-btn')).disabled).toBe(
    true
  );
  expect(root.textContent).toContain('A run is running');
});

test('the constraint picker does not offer the objective, and follows a new one', () => {
  const { root } = setup();
  [...root.querySelectorAll('button')].find((b) => b.textContent === 'Add constraint')?.click();
  const options = () =>
    [...root.querySelectorAll('[aria-label="Constraint observable"] option')].map(
      (o) => /** @type {HTMLOptionElement} */ (o).value
    );
  expect(options()).toEqual(['', 'loss', 'plain']);
  const objective = /** @type {HTMLSelectElement} */ (root.querySelector('#objective-select'));
  objective.value = 'loss';
  objective.dispatchEvent(new Event('change'));
  expect(options()).toEqual(['', 'charge', 'plain']);
});

test('constraints are added and edited in the state', () => {
  const { root, form } = setup();
  const add = [...root.querySelectorAll('button')].find((b) => b.textContent === 'Add constraint');
  add?.click();
  expect(form.constraints).toEqual([{ observable: '', type: 'GREATER_THAN', value: null }]);
  type(root.querySelector('[aria-label="Constraint threshold"]'), '4');
  expect(form.constraints[0].value).toBe(4);
});
