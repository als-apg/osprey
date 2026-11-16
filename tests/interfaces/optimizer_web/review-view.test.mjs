/**
 * Tests for the review screens (review-view.js) and the rules of Start that
 * hold on them: Start starts only on a click, and only the job and revision
 * the view drew; a draft that changed while on screen holds Start until it is
 * acknowledged; a session with no account warns and leaves Start enabled; a
 * refused Start shows its reason and is not sent again; a reading of the
 * machine never rebuilds Start.
 */

import { describe, test, expect } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { mountWith } from './_views.mjs';
import {
  OWNER_UNKNOWN_NOTICE,
  approvalEnvKey,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';
import { mountReviewView } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/review-view.js';

const KEY = approvalEnvKey(storedConfig());

/** @param {any[]} more */
function draft(more = /** @type {any[]} */ ([])) {
  return [
    {
      type: 'poll/job',
      job: { job_id: 'j1', status: 'created', environment_name: 'injector' },
      at: 0,
    },
    { type: 'poll/config', jobId: 'j1', revision: 'rev-1', config: storedConfig() },
    { type: 'approvalEnv/reset', key: KEY },
    {
      type: 'approvalEnv/loaded',
      env: { key: KEY, details: envDetails(), readbacks: { phase_a: 1 }, models: [], error: null },
    },
    { type: 'session', owner: 'op1', ownerKnown: true },
    ...more,
  ];
}

/** @param {HTMLElement} root */
const startBtn = (root) => /** @type {HTMLButtonElement} */ (root.querySelector('#start-btn'));

describe('Start on the review', () => {
  test('starts only on a click, and only the job and revision that were drawn', () => {
    const t = mountWith(mountReviewView, draft());
    expect(t.screen().id).toBe('review');
    t.apply({
      type: 'poll/job',
      job: { job_id: 'j1', status: 'created', environment_name: 'injector' },
      at: 1,
    });
    expect(t.actions.start).not.toHaveBeenCalled();
    expect(startBtn(t.inspector).disabled).toBe(false);
    startBtn(t.inspector).click();
    expect(t.actions.start).toHaveBeenLastCalledWith({ jobId: 'j1', revision: 'rev-1' });
    const primary = t.primary();
    expect(primary).toMatchObject({ disabled: false, key: expect.stringContaining('rev-1') });
    primary.onClick();
    expect(t.actions.start).toHaveBeenLastCalledWith({ jobId: 'j1', revision: 'rev-1' });
  });

  test('a draft that changed on screen holds Start until it is acknowledged', () => {
    const t = mountWith(mountReviewView, draft());
    t.apply({
      type: 'poll/config',
      jobId: 'j1',
      revision: 'rev-2',
      config: storedConfig({ n_iterations: 99 }),
    });
    expect(t.screen().id).toBe('review-changed');
    expect(startBtn(t.inspector).disabled).toBe(true);
    expect(t.primary()).toMatchObject({ disabled: true });
    expect(t.inspector.querySelector('[data-notice="changed"]')).not.toBeNull();
    // The new values are the ones on screen.
    expect(t.inspector.querySelector('[data-field="iterations"]')?.textContent).toBe('99');

    const ack = /** @type {HTMLInputElement} */ (t.inspector.querySelector('#reviewed-check'));
    ack.checked = true;
    ack.dispatchEvent(new Event('change'));
    expect(t.actions.markReviewed).toHaveBeenCalledTimes(1);
    expect(t.actions.start).not.toHaveBeenCalled();

    t.apply({ type: 'approval/reviewed' });
    expect(startBtn(t.inspector).disabled).toBe(false);
    startBtn(t.inspector).click();
    expect(t.actions.start).toHaveBeenLastCalledWith({ jobId: 'j1', revision: 'rev-2' });
  });

  test('with no account the review warns beside Start, and Start stays enabled', () => {
    const t = mountWith(
      mountReviewView,
      draft([{ type: 'session', owner: null, ownerKnown: false }])
    );
    expect(t.screen().id).toBe('review-no-owner');
    const warning = t.inspector.querySelector('.opt-insp-foot [data-notice="start-warning"]');
    expect(warning?.textContent).toBe(OWNER_UNKNOWN_NOTICE);
    expect(startBtn(t.inspector).disabled).toBe(false);
    expect(t.primary()).toMatchObject({ disabled: false });
    expect(t.inspector.querySelector('.opt-foot-note')?.textContent).toContain('no account');
  });

  test('a refused Start shows the reason and is not sent again', () => {
    const t = mountWith(mountReviewView, draft());
    t.apply(
      { type: 'start/sent' },
      { type: 'start/answered', error: 'write authorization refused: owner_unknown' }
    );
    const refusal = t.inspector.querySelector('[data-notice="refused"]');
    expect(refusal?.getAttribute('role')).toBe('alert');
    expect(refusal?.textContent).toContain('owner_unknown');
    t.apply({
      type: 'poll/job',
      job: { job_id: 'j1', status: 'created', environment_name: 'injector' },
      at: 2,
    });
    expect(t.actions.start).not.toHaveBeenCalled();
  });

  test('a new reading of the machine moves the stage, never Start', () => {
    const t = mountWith(mountReviewView, draft());
    const before = startBtn(t.inspector);
    const primaryKey = t.primary().key;
    t.apply({ type: 'approvalEnv/readbacks', key: KEY, readbacks: { phase_a: 5, sol_b: 3 } });
    expect(startBtn(t.inspector)).toBe(before);
    expect(t.primary().key).toBe(primaryKey);
    expect(t.stage.querySelector('[data-knob="sol_b"] [data-mark="now"]')).not.toBeNull();
  });

  test('Start is held while the draft is still loading', () => {
    const t = mountWith(mountReviewView, [
      { type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 0 },
    ]);
    expect(startBtn(t.inspector).disabled).toBe(true);
    expect(t.primary()).toMatchObject({ disabled: true });
  });
});

describe('what the review shows', () => {
  test('windows the service moved on save are marked, on the stage and in the settings', () => {
    const t = mountWith(
      mountReviewView,
      draft([
        {
          type: 'setup/saved',
          note: { jobId: 'j1', tone: 'warn', text: 'Draft created — moved.', moved: ['phase_a'] },
        },
      ])
    );
    expect(t.inspector.querySelector('[data-notice="saved"]')?.textContent).toContain('moved');
    expect(
      t.stage.querySelector('.opt-knob-row[data-knob="phase_a"]')?.classList.contains('is-warn')
    ).toBe(true);
    expect(
      t.inspector.querySelector('.opt-cfg-knob[data-knob="phase_a"]')?.classList.contains('is-warn')
    ).toBe(true);
    expect(
      t.stage.querySelector('.opt-knob-row[data-knob="sol_b"]')?.classList.contains('is-warn')
    ).toBe(false);
  });

  test('the objective is read on the draft target, and Edit opens the set-up', () => {
    const t = mountWith(mountReviewView, draft());
    expect(t.actions.read).toHaveBeenCalledWith('observables', 'injector', 'va', ['charge']);
    /** @type {HTMLElement} */ (t.inspector.querySelector('#edit-draft-btn')).click();
    expect(t.actions.editDraft).toHaveBeenCalledTimes(1);
    expect(t.inspector.querySelector('.opt-foot-note')?.textContent).toContain('op1');
  });
});
