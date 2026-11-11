/** DOM tests for the run card (run-view.js). */

import { test, expect, vi } from 'vitest';

import { runningJob } from './_fixtures.mjs';
import {
  CANCEL_NOTE,
  renderRunStatus,
  statusTone,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-view.js';

test('a running job shows Pause and Cancel, progress and best so far', () => {
  const root = document.createElement('div');
  const onControl = vi.fn();
  renderRunStatus(root, runningJob(), { cancelArmed: false, runError: null }, { onControl });
  expect(root.querySelector('#pause-btn')).not.toBeNull();
  expect(root.querySelector('#resume-btn')).toBeNull();
  expect(root.querySelector('#progress-label')?.textContent).toBe(
    '3 of 10 evaluations · best charge 2.5'
  );
  /** @type {HTMLButtonElement} */ (root.querySelector('#cancel-btn')).click();
  expect(onControl).toHaveBeenCalledWith('cancel');
});

test('an armed cancel says the machine holds, and no restore control exists', () => {
  const root = document.createElement('div');
  renderRunStatus(
    root,
    runningJob({ status: 'paused' }),
    { cancelArmed: true, runError: null },
    { onControl: vi.fn() }
  );
  expect(root.querySelector('#cancel-btn')?.textContent).toBe('Confirm cancel');
  expect(root.textContent).toContain(CANCEL_NOTE);
  // A cancel can land after a candidate is written but before it is read back,
  // so the note never claims the held point was evaluated.
  expect(CANCEL_NOTE).toContain('last point written, which may not have been evaluated yet');
  expect(CANCEL_NOTE).toContain('Restoring the starting setpoints is a separate, manual step.');
  expect(root.querySelector('#resume-btn')).not.toBeNull();
  expect(root.textContent?.toLowerCase()).not.toContain('restore now');
});

test('a run waiting on write approval says so', () => {
  const root = document.createElement('div');
  renderRunStatus(
    root,
    runningJob({ status: 'submitted', write_authorization: { state: 'awaiting', id: 'r1' } }),
    { cancelArmed: false, runError: null },
    { onControl: vi.fn() }
  );
  expect(root.querySelector('.banner-warn')?.textContent).toContain(
    'approve the request in the web terminal'
  );
});

test('statusTone', () => {
  expect(statusTone('error')).toBe('err');
  expect(statusTone('running')).toBe('ok');
  expect(statusTone('created')).toBe('info');
});

test('before the job reports a total, the planned count from the stored config shows', () => {
  const root = document.createElement('div');
  const job = runningJob({ status: 'submitted', total_evaluations: null, data: [] });
  renderRunStatus(
    root,
    job,
    { cancelArmed: false, runError: null, plannedTotal: 33 },
    { onControl: vi.fn() }
  );
  expect(root.querySelector('#progress-label')?.textContent).toContain('0 of 33 evaluations');

  // No config to plan from: the total stays unknown.
  renderRunStatus(root, job, { cancelArmed: false, runError: null }, { onControl: vi.fn() });
  expect(root.querySelector('#progress-label')?.textContent).toContain('0 of ? evaluations');

  // The job's own total wins once it reports one.
  renderRunStatus(
    root,
    runningJob(),
    { cancelArmed: false, runError: null, plannedTotal: 33 },
    { onControl: vi.fn() }
  );
  expect(root.querySelector('#progress-label')?.textContent).toContain('3 of 10 evaluations');
});
