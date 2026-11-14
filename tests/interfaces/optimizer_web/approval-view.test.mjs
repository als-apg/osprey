/**
 * DOM tests for the approval surface (approval-view.js): every approved field
 * is on screen, the flags draw the eye, and Start follows the gate.
 */

import { test, expect, vi } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { renderApproval } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-view.js';
import {
  OWNER_UNKNOWN_NOTICE,
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
 * @param {{ enabled: boolean, reason: string|null, warnings?: string[] }} [gate]
 * @param {any} [approval]
 * @param {any} [env]
 * @param {{ status?: string|null, owner?: string|null }} [extra]
 */
function render(
  config = storedConfig(),
  gate = { enabled: true, reason: null },
  approval = approved(config),
  env = /** @type {any} */ (envDetails()),
  { status = 'created', owner = null } = {}
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
      gate: { warnings: [], ...gate },
      status,
      owner,
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

/** A draft changed since it was shown: the review notice and button are due. */
function changedDraft() {
  return reduceApproval(approved(storedConfig()), {
    type: 'draft',
    jobId: 'j1',
    revision: 'rev-2',
    config: storedConfig({ n_iterations: 99 }),
  });
}

test('an ended job shows neither the review notice nor the review button', () => {
  const { root } = render(
    storedConfig(),
    { enabled: false, reason: 'This job has ended (error). Create a new draft.' },
    changedDraft(),
    undefined,
    { status: 'error' }
  );
  expect(root.querySelector('.banner-warn')).toBeNull();
  expect(root.textContent).not.toContain('I have reviewed this draft');
  expect(root.textContent).toContain('This job has ended (error). Create a new draft.');
});

test("Start keeps its place: the review button's slot is held while unneeded", () => {
  const position = (/** @type {HTMLElement} */ root) => {
    const foot = /** @type {HTMLElement} */ (root.querySelector('.foot-row'));
    return (
      [...root.children].indexOf(foot) * 100 +
      [...foot.children].findIndex((c) => c.id === 'start-btn')
    );
  };
  const before = render(
    storedConfig(),
    { enabled: false, reason: 'Review the changed draft first.' },
    changedDraft()
  );
  const reviewed = reduceApproval(changedDraft(), { type: 'reviewed' });
  const after = render(storedConfig(), { enabled: true, reason: null }, reviewed);
  expect(position(after.root)).toBe(position(before.root));
  expect(after.root.querySelector('#reviewed-btn')).toBeNull();
  const slot = /** @type {HTMLButtonElement} */ (after.root.querySelector('.slot-reserved'));
  expect(slot.disabled).toBe(true);
  expect(slot.getAttribute('aria-hidden')).toBe('true');
  expect(after.root.querySelector('.banner-ok')?.textContent).toContain('Reviewed');
});

test('the owner: "Starts as" when named, the warning before the click when not', () => {
  const named = render(undefined, undefined, undefined, undefined, { owner: 'op1' });
  expect(named.root.querySelector('.start-owner')?.textContent).toBe('Starts as op1.');
  const unnamed = render(storedConfig(), {
    enabled: true,
    reason: null,
    warnings: [OWNER_UNKNOWN_NOTICE],
  });
  expect(unnamed.root.querySelector('.start-warn')?.textContent).toBe(OWNER_UNKNOWN_NOTICE);
  expect(unnamed.root.textContent).not.toContain('Starts as');
  expect(/** @type {HTMLButtonElement} */ (unnamed.root.querySelector('#start-btn')).disabled).toBe(
    false
  );
});

test("the session's late answer changes nothing above Start: the owner line is in its row", () => {
  /** Every node above the foot row, as tag.class, and Start's index in its row. */
  const layout = (/** @type {HTMLElement} */ root) => {
    const foot = /** @type {HTMLElement} */ (root.querySelector('.foot-row'));
    const children = [...root.children];
    return {
      above: children.slice(0, children.indexOf(foot)).map((c) => `${c.tagName}.${c.className}`),
      start: [...foot.children].findIndex((c) => c.id === 'start-btn'),
    };
  };
  const pending = render();
  const named = render(undefined, undefined, undefined, undefined, { owner: 'op1' });
  const unnamed = render(storedConfig(), {
    enabled: true,
    reason: null,
    warnings: [OWNER_UNKNOWN_NOTICE],
  });
  expect(layout(named.root)).toEqual(layout(pending.root));
  expect(layout(unnamed.root)).toEqual(layout(pending.root));
  const note = (/** @type {HTMLElement} */ root) => root.querySelector('.foot-row > .start-note');
  expect(note(named.root)?.querySelector('.start-owner')?.textContent).toBe('Starts as op1.');
  expect(note(unnamed.root)?.querySelector('.start-warn')).not.toBeNull();
});

test("a draft that may not move: its notice sits beside Start with the owner's, Start stays live", () => {
  const notice = 'Nelder-Mead may not move from the current setpoints.';
  const { root } = render(
    storedConfig(),
    { enabled: true, reason: null, warnings: [OWNER_UNKNOWN_NOTICE, notice] },
    undefined,
    undefined,
    { owner: null }
  );
  const note = /** @type {HTMLElement} */ (root.querySelector('.foot-row > .start-note'));
  expect([...note.querySelectorAll('.start-warn')].map((w) => w.textContent)).toEqual([
    OWNER_UNKNOWN_NOTICE,
    notice,
  ]);
  expect(/** @type {HTMLButtonElement} */ (root.querySelector('#start-btn')).disabled).toBe(false);

  const named = render(
    storedConfig(),
    { enabled: true, reason: null, warnings: [notice] },
    undefined,
    undefined,
    { owner: 'op1' }
  );
  expect(named.root.querySelector('.start-warn')?.textContent).toBe(notice);
  expect(named.root.querySelector('.start-owner')?.textContent).toBe('Starts as op1.');
});

test('no draft: a plain empty state, no Start', () => {
  const root = document.createElement('div');
  renderApproval(
    root,
    {
      approval: approvalInit(),
      env: null,
      models: null,
      gate: { enabled: false, reason: null, warnings: [] },
      status: null,
    },
    { onStart: vi.fn(), onReviewed: vi.fn() }
  );
  expect(root.querySelector('#start-btn')).toBeNull();
});
