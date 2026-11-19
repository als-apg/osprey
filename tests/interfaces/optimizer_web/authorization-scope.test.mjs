/**
 * Tests for `authorizationScope` (approval-model.js): what a run's write
 * authorization covers, read from the job's `write_authorization` record as
 * it grows from `requesting` to `granted`. A field the record does not carry
 * stays null; nothing is guessed.
 */

import { describe, test, expect } from 'vitest';

import { authorizationScope } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';

/** The record while the run waits for a person (the service merges each state into it). */
const awaiting = () => ({
  state: 'awaiting',
  target: 'va',
  channels: { 'A:SP': [-18, 18], 'B:SP': [2, 4] },
  steps: 11,
  completion: 'apply_best',
  id: 'req-0123456789',
  owner: 'carol',
  approver: null,
  reason: null,
  detail: null,
});

describe('authorizationScope', () => {
  test('no record gives no scope', () => {
    expect(authorizationScope(null)).toBeNull();
    expect(authorizationScope(undefined)).toBeNull();
  });

  test('a request names its owner, its writes, its channels and its id; not yet where or until when', () => {
    expect(authorizationScope(awaiting())).toEqual({
      state: 'awaiting',
      owner: 'carol',
      id: 'req-0123456789',
      steps: 11,
      approver: null,
      approvedFrom: null,
      expiresAt: null,
      channels: [
        { channel: 'A:SP', min: -18, max: 18 },
        { channel: 'B:SP', min: 2, max: 4 },
      ],
      completion: 'apply_best',
    });
  });

  test('a grant adds where it was approved and its expiry', () => {
    const ns = 1_800_000_000_000_000_000;
    const scope = authorizationScope({
      ...awaiting(),
      state: 'granted',
      approver: 'chip',
      expires_at_ns: ns,
    });
    expect(scope?.approver).toBe('chip');
    expect(scope?.approvedFrom).toContain('control-target switch');
    expect(scope?.expiresAt).toBe(ns / 1e6);
  });

  test('a self-issued grant says the write switch allowed it', () => {
    const scope = authorizationScope({ state: 'granted', approver: 'write_switch' });
    expect(scope?.approvedFrom).toContain('write switch');
  });

  test('malformed fields are left out, not repaired', () => {
    const scope = authorizationScope({
      state: 'requesting',
      steps: 'many',
      channels: { 'A:SP': [0], 'B:SP': 'x', 'C:SP': [1, 2] },
      expires_at_ns: 'soon',
    });
    expect(scope?.steps).toBeNull();
    expect(scope?.expiresAt).toBeNull();
    expect(scope?.channels).toEqual([{ channel: 'C:SP', min: 1, max: 2 }]);
    expect(scope?.owner).toBeNull();
  });
});
