/** Unit tests for the approval card's environment read (approval-env.js). */

import { test, expect } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import {
  READBACK_REFRESH_MS,
  readbacksDue,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-env.js';
import { approvalEnvKey } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';

/** @param {string} status */
function waiting(status = 'created') {
  const config = storedConfig();
  return {
    job: { job_id: 'j1', status },
    approval: { jobId: 'j1', revision: 'rev-1', config, needsReview: false, notice: null },
    approvalEnv: {
      key: approvalEnvKey(config),
      details: /** @type {any} */ (envDetails()),
      readbacks: {},
      models: [],
      error: null,
    },
  };
}

test('readbacks are due on a created draft once the refresh interval has passed', () => {
  expect(readbacksDue(waiting(), 0, READBACK_REFRESH_MS - 1)).toBe(false);
  expect(readbacksDue(waiting(), 0, READBACK_REFRESH_MS)).toBe(true);
});

test('readbacks are never due for a job that is not a waiting draft, or before the env loads', () => {
  expect(readbacksDue(waiting('running'), 0, 60000)).toBe(false);
  const loading = waiting();
  loading.approvalEnv.details = null;
  expect(readbacksDue(loading, 0, 60000)).toBe(false);
  const other = waiting();
  other.approval.jobId = 'j0';
  expect(readbacksDue(other, 0, 60000)).toBe(false);
});
