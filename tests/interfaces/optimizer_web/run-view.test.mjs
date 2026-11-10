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
