/**
 * Unit tests for the plan panel's live draft client (draft-client.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/services/bluesky_panels/draft-client.test.mjs
 *
 * Two layers are exercised:
 *  - The pure reducers (`reduceFrame`, `reduceReset`, `computeDelta`,
 *    `shouldShowAffordance`, `resolvePinnedRevision`) with plain frame
 *    objects — no DOM, no network, no `EventSource` (happy-dom has none).
 *  - `createDraftClient(deps)`, driven with a fake `sseFactory` (captures the
 *    `onFrame` callback so a test can push frames as plain objects) and a
 *    real `renderSchemaForm` collector over a small ORM-shaped schema, so the
 *    pending-key/flash/apply logic runs against real DOM without ever
 *    constructing a real `EventSource`.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  createInitialState,
  reduceFrame,
  reduceReset,
  computeDelta,
  shouldShowAffordance,
  resolvePinnedRevision,
  createDraftClient,
  createSSEConnection,
  generateClientId,
  buildLaunchRequestBody,
  classifyLaunchResponse,
  resultsPanelUrl,
  buildLaunchBanner,
} from '../../../src/osprey/services/bluesky_panels/panels/plan/draft-client.js';
import { renderSchemaForm } from '../../../src/osprey/services/bluesky_panels/panels/plan/schema-form.js';

// A small ORM-shaped fixture: one channel-list (container widget, exercises
// whole-value-replacement flash) and two plain scalars (exercise the
// pending-key rule via a real native `input` event).
const ORM_SCHEMA = {
  properties: {
    correctors: {
      items: { type: 'string' },
      minItems: 1,
      title: 'Correctors',
      type: 'array',
      'x-widget': 'channel-list',
    },
    span_a: {
      exclusiveMinimum: 0,
      maximum: 10.0,
      title: 'Max kick (A)',
      type: 'number',
    },
    num: {
      minimum: 3,
      title: 'Number of steps',
      type: 'integer',
    },
  },
  required: ['correctors', 'span_a', 'num'],
  title: 'PARAMS',
  type: 'object',
};

// ORM_SCHEMA plus a segmented control — kept separate from ORM_SCHEMA so the
// many existing `.toEqual({num: ...})`-style collector assertions above don't
// have to also account for a segmented field's always-contributing value.
const SEGMENTED_SCHEMA = {
  properties: {
    ...ORM_SCHEMA.properties,
    sweep: {
      default: 'bidirectional',
      enum: ['bidirectional', 'monodirectional'],
      title: 'Sweep direction',
      type: 'string',
      'x-widget': 'segmented',
    },
  },
  required: ORM_SCHEMA.required,
  title: 'PARAMS',
  type: 'object',
};

// A minimal grid_scan-shaped fixture: one array-of-flat-objects field,
// rendered as schema-form.js's editable table (`.table-add` / `.row-x`).
const TABLE_SCHEMA = {
  $defs: {
    Axis: {
      properties: {
        setpoint: { title: 'Setpoint', type: 'string' },
        num_points: { minimum: 2, title: 'Num Points', type: 'integer' },
      },
      required: ['setpoint', 'num_points'],
      title: 'Axis',
      type: 'object',
    },
  },
  properties: {
    axes: { items: { $ref: '#/$defs/Axis' }, minItems: 1, title: 'Axes', type: 'array' },
  },
  required: ['axes'],
  title: 'PARAMS',
  type: 'object',
};

// ---------------------------------------------------------------------------
// Pure reducers
// ---------------------------------------------------------------------------

describe('createInitialState / reduceReset', () => {
  test('starts with a null baseline and no draft', () => {
    const state = createInitialState('tab-1');
    expect(state.lastAppliedRevision).toBeNull();
    expect(state.draftPlanName).toBeNull();
    expect(state.bound).toBe(false);
  });

  test('reduceReset sets baseline/draft from a snapshot, including a null draft', () => {
    const state = createInitialState('tab-1');
    const next = reduceReset(state, { draft: null, revision: 7 });
    expect(next.lastAppliedRevision).toBe(7);
    expect(next.draftPlanName).toBeNull();
    expect(next.draftArgs).toBeNull();
  });
});

describe('reduceFrame — revision rules', () => {
  test('revision <= last applied is dropped', () => {
    let state = createInitialState('tab-1');
    state = reduceReset(state, { draft: null, revision: 5 });
    const { state: next, action } = reduceFrame(state, {
      type: 'change',
      draft: { plan_name: 'orm', plan_args: {} },
      changed: [],
      revision: 5,
      origin: 'mcp-agent',
    });
    expect(action).toEqual({ type: 'drop' });
    expect(next.lastAppliedRevision).toBe(5);
  });

  test('revision === last applied + 1 applies', () => {
    let state = createInitialState('tab-1');
    state = reduceReset(state, { draft: null, revision: 5 });
    const draft = { plan_name: 'orm', plan_args: { num: 4 } };
    const { state: next, action } = reduceFrame(state, {
      type: 'change',
      draft,
      changed: ['num'],
      revision: 6,
      origin: 'mcp-agent',
    });
    expect(action.type).toBe('apply');
    expect(next.lastAppliedRevision).toBe(6);
    expect(next.draftArgs).toEqual({ num: 4 });
  });

  test('revision gap (> last + 1) triggers resync', () => {
    let state = createInitialState('tab-1');
    state = reduceReset(state, { draft: null, revision: 5 });
    const { state: next, action } = reduceFrame(state, {
      type: 'change',
      draft: { plan_name: 'orm', plan_args: {} },
      changed: [],
      revision: 8,
      origin: 'mcp-agent',
    });
    expect(action).toEqual({ type: 'resync' });
    // A resync doesn't mutate state itself — the caller must GET /draft and
    // feed the result through reduceReset.
    expect(next.lastAppliedRevision).toBe(5);
  });

  test('no baseline yet (pre-hello) is treated as a gap -> resync', () => {
    const state = createInitialState('tab-1');
    const { action } = reduceFrame(state, {
      type: 'change',
      draft: { plan_name: 'orm', plan_args: {} },
      changed: [],
      revision: 1,
      origin: 'mcp-agent',
    });
    expect(action).toEqual({ type: 'resync' });
  });

  test('a hello frame resets the baseline unconditionally, including backward', () => {
    let state = createInitialState('tab-1');
    state = reduceReset(state, { draft: null, revision: 42 });
    const { state: next, action } = reduceFrame(state, {
      type: 'hello',
      draft: { plan_name: 'orm', plan_args: { num: 1 } },
      revision: 2, // lower than the prior baseline — a bridge restart.
    });
    expect(action).toEqual({ type: 'reset' });
    expect(next.lastAppliedRevision).toBe(2);
    expect(next.draftPlanName).toBe('orm');
  });

  test('an own-origin frame advances the baseline but is reported as echo', () => {
    let state = createInitialState('tab-1');
    state = reduceReset(state, { draft: null, revision: 5 });
    const { state: next, action } = reduceFrame(state, {
      type: 'change',
      draft: { plan_name: 'orm', plan_args: { num: 9 } },
      changed: ['num'],
      revision: 6,
      origin: 'tab-1',
    });
    expect(action.type).toBe('echo');
    expect(next.lastAppliedRevision).toBe(6);
    expect(next.draftArgs).toEqual({ num: 9 });
  });

  test('a launched frame records the banner without touching the revision baseline', () => {
    let state = createInitialState('tab-1');
    state = reduceReset(state, { draft: { plan_name: 'orm', plan_args: { num: 3 } }, revision: 5 });
    const { state: next, action } = reduceFrame(state, {
      type: 'launched',
      draft: null,
      revision: 5,
      run_id: 'run-abc',
    });
    expect(action).toEqual({ type: 'launched', frame: expect.objectContaining({ run_id: 'run-abc' }) });
    expect(next.launchBanner).toEqual({ runId: 'run-abc', revision: 5 });
    // A launch is an event, not a draft mutation: baseline and draft are
    // untouched, so a later change frame still applies against revision 5.
    expect(next.lastAppliedRevision).toBe(5);
    expect(next.draftPlanName).toBe('orm');
    expect(next.draftArgs).toEqual({ num: 3 });
  });

  test('a launched frame with a missing/blank run_id is dropped (no banner, no crash)', () => {
    let state = createInitialState('tab-1');
    state = reduceReset(state, { draft: null, revision: 5 });
    for (const bad of [
      { type: 'launched', draft: null, revision: 6 },
      { type: 'launched', draft: null, revision: 6, run_id: '' },
      { type: 'launched', draft: null, revision: 6, run_id: 42 },
      { type: 'launched', draft: null },
    ]) {
      const { state: next, action } = reduceFrame(state, /** @type {any} */ (bad));
      expect(action).toEqual({ type: 'drop' });
      expect(next.launchBanner).toBeNull();
      expect(next.lastAppliedRevision).toBe(5);
    }
  });
});

describe('shouldShowAffordance', () => {
  test('false while bound', () => {
    const state = { ...createInitialState('t'), bound: true, draftPlanName: 'orm', selectedName: 'orm' };
    expect(shouldShowAffordance(state)).toBe(false);
  });

  test('false with no draft', () => {
    const state = { ...createInitialState('t'), draftPlanName: null, selectedName: 'orm' };
    expect(shouldShowAffordance(state)).toBe(false);
  });

  test('false when the draft names a different plan than the one selected', () => {
    const state = { ...createInitialState('t'), draftPlanName: 'grid_scan', selectedName: 'orm' };
    expect(shouldShowAffordance(state)).toBe(false);
  });

  test('true when unbound and the draft matches the selected plan', () => {
    const state = { ...createInitialState('t'), draftPlanName: 'orm', selectedName: 'orm' };
    expect(shouldShowAffordance(state)).toBe(true);
  });
});

describe('computeDelta', () => {
  test('a present key goes into plan_args_patch', () => {
    const { plan_args_patch, remove } = computeDelta({ num: 4, span_a: 1.5 }, ['num']);
    expect(plan_args_patch).toEqual({ num: 4 });
    expect(remove).toEqual([]);
  });

  test('a blanked (absent/OMIT-dropped) key goes into remove[]', () => {
    const { plan_args_patch, remove } = computeDelta({ num: 4 }, ['num', 'span_a']);
    expect(plan_args_patch).toEqual({ num: 4 });
    expect(remove).toEqual(['span_a']);
  });
});

describe('resolvePinnedRevision', () => {
  test('a successful flush pins its own response revision', () => {
    expect(resolvePinnedRevision({ patched: true, revision: 12 }, 3)).toBe(12);
  });

  test('nothing pending falls back to the last-applied baseline', () => {
    expect(resolvePinnedRevision({ patched: false, revision: null }, 3)).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// createDraftClient orchestrator — fake SSE + HTTP, real schema-form DOM.
// ---------------------------------------------------------------------------

/**
 * @param {any} [overrides]
 * @param {any} [schema]
 * @returns {any}
 */
function makeHarness(overrides = {}, schema = ORM_SCHEMA) {
  const formEl = /** @type {HTMLFormElement} */ (document.createElement('form'));
  document.body.appendChild(formEl);
  const collector = renderSchemaForm(formEl, schema, {});

  /** @type {(frame: any) => void} */
  let pushFrame = () => {
    throw new Error('sseFactory not yet wired');
  };
  const sseFactory = vi.fn((/** @type {any} */ _url, /** @type {any} */ { onFrame }) => {
    pushFrame = onFrame;
    return { close: vi.fn() };
  });

  /** @type {string[]} */
  let knownPlanNames = ['orm', 'grid_scan'];

  const deps = {
    formEl,
    api: (/** @type {string} */ path) => path,
    clientId: 'tab-1',
    getCollector: () => collector,
    getPlanNames: () => knownPlanNames,
    selectPlan: vi.fn(async () => {}),
    refetchPlans: vi.fn(async () => {}),
    getDraft: vi.fn(async () => /** @type {any} */ ({ draft: null, revision: 0 })),
    patchDraft: vi.fn(
      async () =>
        /** @type {any} */ ({ ok: true, status: 200, body: { revision: 1, changed: [], plan_name: 'orm' } })
    ),
    deleteDraft: vi.fn(async () => {}),
    onBoundChange: vi.fn(),
    onAffordance: vi.fn(),
    onUnknownPlanBanner: vi.fn(),
    onAgentEditNote: vi.fn(),
    onPatchRejected: vi.fn(),
    onLaunchBanner: vi.fn(),
    sseFactory,
    ...overrides,
  };

  const client = createDraftClient(deps);

  return {
    client,
    deps,
    formEl,
    getCollector: () => collector,
    setKnownPlanNames: (/** @type {string[]} */ names) => {
      knownPlanNames = names;
    },
    push: (/** @type {any} */ frame) => pushFrame(frame),
  };
}

describe('createDraftClient — binding transitions', () => {
  /** @type {any} */
  let harness;

  beforeEach(() => {
    harness = makeHarness();
  });

  afterEach(() => {
    harness.client.destroy();
    document.body.replaceChildren();
  });

  test('selecting a plan that matches the existing draft binds and seeds from GET /draft', async () => {
    // hello establishes the draft's plan_name/args before any selection.
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 5 } }, revision: 1 });

    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { num: 7, span_a: 2.5 } },
      revision: 1,
    });

    await harness.client.onPlanSelected('orm');

    expect(harness.client.isBound()).toBe(true);
    expect(harness.deps.onBoundChange).toHaveBeenCalledWith(true);
    // Seeded from the fresh GET /draft snapshot, not bare schema defaults.
    expect(harness.getCollector()()).toEqual({ num: 7, span_a: 2.5 });
  });

  test('selecting a plan that does not match the draft stays unbound, no affordance', async () => {
    harness.push({ type: 'hello', draft: { plan_name: 'grid_scan', plan_args: {} }, revision: 1 });

    await harness.client.onPlanSelected('orm');

    expect(harness.client.isBound()).toBe(false);
    expect(harness.deps.onAffordance).toHaveBeenLastCalledWith(null);
  });

  test('a plan-change frame landing on the locally-selected-but-unbound plan shows the affordance only', async () => {
    // Establish a baseline (no draft yet) before selecting -> unbound, no affordance.
    harness.push({ type: 'hello', draft: null, revision: 0 });
    await harness.client.onPlanSelected('orm');
    expect(harness.deps.onAffordance).toHaveBeenLastCalledWith(null);

    const applyValuesSpy = vi.spyOn(harness.getCollector(), 'applyValues');

    // Now a plan-change frame arrives naming the very plan the operator is
    // already looking at, while still unbound.
    harness.push({
      type: 'plan-change',
      draft: { plan_name: 'orm', plan_args: { num: 3 } },
      changed: ['num'],
      revision: 1,
      origin: 'mcp-agent',
    });
    await flushMicrotasks();

    expect(harness.client.isBound()).toBe(false);
    expect(harness.deps.onBoundChange).not.toHaveBeenCalledWith(true);
    expect(harness.deps.onAffordance).toHaveBeenLastCalledWith('orm');
    // Unbound: no value application, no PATCH-worthy side effect.
    expect(applyValuesSpy).not.toHaveBeenCalled();
  });

  test('clicking the affordance binds, seeds from GET /draft, and clears pending', async () => {
    harness.push({ type: 'hello', draft: null, revision: 0 });
    await harness.client.onPlanSelected('orm');
    harness.push({
      type: 'plan-change',
      draft: { plan_name: 'orm', plan_args: { num: 3 } },
      changed: ['num'],
      revision: 1,
      origin: 'mcp-agent',
    });
    await flushMicrotasks();
    expect(harness.deps.onAffordance).toHaveBeenLastCalledWith('orm');

    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { num: 9 } },
      revision: 1,
    });
    await harness.client.onAffordanceClick();

    expect(harness.client.isBound()).toBe(true);
    expect(harness.getCollector()()).toEqual({ num: 9 });
    // No pending edit survives a fresh bind.
    const flushResult = await harness.client.flushNow();
    expect(flushResult).toEqual({ patched: false, revision: null });
    expect(harness.deps.patchDraft).not.toHaveBeenCalled();
  });

  test('a frame while already bound applies changed[] and flashes only those fields', async () => {
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    const numEl = harness.getCollector().fields.num.el;
    expect(numEl.classList.contains('draft-flash')).toBe(false);

    harness.push({
      type: 'change',
      draft: { plan_name: 'orm', plan_args: { num: 4 } },
      changed: ['num'],
      revision: 2,
      origin: 'mcp-agent',
    });
    await flushMicrotasks();

    expect(harness.getCollector()()).toEqual({ num: 4 });
    expect(numEl.classList.contains('draft-flash')).toBe(true);
    expect(harness.deps.onAgentEditNote).toHaveBeenCalledWith(['num']);
  });

  test('own-origin (echo) frames advance the baseline but skip apply and flash', async () => {
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');

    const numEl = harness.getCollector().fields.num.el;
    const applyValuesSpy = vi.spyOn(harness.getCollector(), 'applyValues');

    harness.push({
      type: 'change',
      draft: { plan_name: 'orm', plan_args: { num: 99 } },
      changed: ['num'],
      revision: 2,
      origin: 'tab-1', // this tab's own PATCH echoing back.
    });
    await flushMicrotasks();

    expect(applyValuesSpy).not.toHaveBeenCalled();
    expect(numEl.classList.contains('draft-flash')).toBe(false);
    expect(harness.deps.onAgentEditNote).not.toHaveBeenCalled();
  });

  test('a clear frame while bound unbinds the form', async () => {
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    harness.push({ type: 'clear', draft: null, changed: ['num'], revision: 2, origin: 'mcp-agent' });
    await flushMicrotasks();

    expect(harness.client.isBound()).toBe(false);
    expect(harness.deps.onBoundChange).toHaveBeenLastCalledWith(false);
  });

  test('discard-draft unbinds and calls DELETE /draft', async () => {
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    await harness.client.onDiscardClick();

    expect(harness.deps.deleteDraft).toHaveBeenCalledTimes(1);
    expect(harness.client.isBound()).toBe(false);
  });

  test('a failed DELETE /draft (sidecar-unreachable) leaves bound state consistent and surfaces the failure', async () => {
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    harness.deps.deleteDraft.mockRejectedValueOnce(new Error('sidecar unreachable'));

    await harness.client.onDiscardClick();

    // Still bound: the discard never actually happened server-side, so the
    // panel must not optimistically show an unbound (manual) form.
    expect(harness.client.isBound()).toBe(true);
    expect(harness.deps.onPatchRejected).toHaveBeenCalled();
  });
});

describe('createDraftClient — unknown-plan banner', () => {
  test('an unknown draft plan_name triggers exactly one /plans refetch, then a banner if still missing', async () => {
    const harness = makeHarness();
    harness.setKnownPlanNames(['orm']); // 'phantom' is not (yet) known.

    harness.push({ type: 'hello', draft: { plan_name: 'phantom', plan_args: {} }, revision: 1 });
    await flushMicrotasks();

    expect(harness.deps.refetchPlans).toHaveBeenCalledTimes(1);
    expect(harness.deps.onUnknownPlanBanner).toHaveBeenLastCalledWith('phantom');

    harness.client.destroy();
  });

  test('the banner clears once the plan becomes known after a refetch', async () => {
    const harness = makeHarness();
    harness.setKnownPlanNames(['orm']);
    harness.deps.refetchPlans.mockImplementation(async () => {
      harness.setKnownPlanNames(['orm', 'phantom']);
    });

    harness.push({ type: 'hello', draft: { plan_name: 'phantom', plan_args: {} }, revision: 1 });
    await flushMicrotasks();

    expect(harness.deps.onUnknownPlanBanner).toHaveBeenLastCalledWith(null);

    harness.client.destroy();
  });
});

describe('createDraftClient — pending-key rule and PATCH-back', () => {
  /** @type {any} */
  let harness;

  beforeEach(async () => {
    vi.useFakeTimers();
    harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
  });

  afterEach(() => {
    harness.client.destroy();
    document.body.replaceChildren();
    vi.useRealTimers();
  });

  test('a real user input event marks the field pending and debounces a PATCH', async () => {
    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    numInput.value = '9';
    numInput.dispatchEvent(new Event('input', { bubbles: true }));

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(1);
    const body = harness.deps.patchDraft.mock.calls[0][0];
    expect(body.client_id).toBe('tab-1');
    expect(body.expected_plan_name).toBe('orm');
    expect(body.plan_args_patch).toEqual({ num: 9 });
    expect(body.remove).toEqual([]);
  });

  test('applyValues (programmatic) never marks a field pending', async () => {
    harness.getCollector().applyValues({ num: 42 });

    await vi.advanceTimersByTimeAsync(1000);

    expect(harness.deps.patchDraft).not.toHaveBeenCalled();
  });

  test('blanking a field marks it pending and PATCHes it into remove[]', async () => {
    const spanInput = /** @type {HTMLInputElement} */ (
      harness.getCollector().fields.span_a.el.querySelector('input')
    );
    spanInput.value = '';
    spanInput.dispatchEvent(new Event('input', { bubbles: true }));

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(1);
    const body = harness.deps.patchDraft.mock.calls[0][0];
    expect(body.remove).toEqual(['span_a']);
    expect(body.plan_args_patch).toEqual({});
  });

  test('rapid successive edits to the same field debounce into a single PATCH', async () => {
    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    for (const v of ['2', '3', '4']) {
      numInput.value = v;
      numInput.dispatchEvent(new Event('input', { bubbles: true }));
      await vi.advanceTimersByTimeAsync(100); // less than the debounce window
    }
    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(1);
    expect(harness.deps.patchDraft.mock.calls[0][0].plan_args_patch).toEqual({ num: 4 });
  });

  test('a 409 from the flush PATCH drops the edit and resyncs', async () => {
    harness.deps.patchDraft.mockResolvedValueOnce({
      ok: false,
      status: 409,
      body: { code: 'plan_name_mismatch', detail: 'stale' },
    });
    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { num: 1 } },
      revision: 2,
    });

    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    numInput.value = '5';
    numInput.dispatchEvent(new Event('input', { bubbles: true }));

    const result = await harness.client.flushNow();

    expect(result).toEqual({ patched: false, revision: null });
    // Resynced to the fresh snapshot's value, discarding the dropped edit.
    expect(harness.getCollector()()).toEqual({ num: 1 });
    expect(harness.client.getLastAppliedRevision()).toBe(2);
  });

  test('a remote change frame on a still-pending key clears its pending status (no redundant echo PATCH)', async () => {
    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    numInput.value = '2';
    numInput.dispatchEvent(new Event('input', { bubbles: true })); // marks 'num' pending; debounce scheduled.

    // Before the debounce fires, a remote edit lands on the SAME field.
    harness.push({
      type: 'change',
      draft: { plan_name: 'orm', plan_args: { num: 99 } },
      changed: ['num'],
      revision: 2,
      origin: 'mcp-agent',
    });
    await flushMicrotasks();
    expect(harness.getCollector()()).toEqual({ num: 99 }); // remote value wins.

    await vi.advanceTimersByTimeAsync(500); // let the (now-stale) debounce timer fire.

    expect(harness.deps.patchDraft).not.toHaveBeenCalled();
  });
});

describe('createDraftClient — launch pin resolution via flushNow', () => {
  test('a flush with pending edits pins the PATCH response revision', async () => {
    vi.useFakeTimers();
    const harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');

    harness.deps.patchDraft.mockResolvedValueOnce({
      ok: true,
      status: 200,
      body: { revision: 55, changed: ['num'], plan_name: 'orm' },
    });
    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    numInput.value = '2';
    numInput.dispatchEvent(new Event('input', { bubbles: true }));

    const flushResult = await harness.client.flushNow();
    expect(flushResult).toEqual({ patched: true, revision: 55 });
    expect(resolvePinnedRevision(flushResult, harness.client.getLastAppliedRevision())).toBe(55);

    harness.client.destroy();
    vi.useRealTimers();
  });

  test('a flush with nothing pending falls back to the last-applied baseline', async () => {
    const harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 7 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 7 });
    await harness.client.onPlanSelected('orm');

    const flushResult = await harness.client.flushNow();
    expect(flushResult).toEqual({ patched: false, revision: null });
    expect(resolvePinnedRevision(flushResult, harness.client.getLastAppliedRevision())).toBe(7);

    harness.client.destroy();
  });
});

describe('createDraftClient — structural edits (form-change) mark pending, applyValues does not', () => {
  test('removing a channel-list chip marks it pending and PATCHes the removal', async () => {
    vi.useFakeTimers();
    const harness = makeHarness();
    harness.push({
      type: 'hello',
      draft: { plan_name: 'orm', plan_args: { correctors: ['HCM1'], num: 5 } },
      revision: 1,
    });
    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { correctors: ['HCM1'], num: 5 } },
      revision: 1,
    });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    const listEl = harness.getCollector().fields.correctors.el.querySelector('.channel-list');
    listEl.querySelector('.chan-x').click();

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(1);
    const body = harness.deps.patchDraft.mock.calls[0][0];
    expect(body.remove).toEqual(['correctors']);

    harness.client.destroy();
    vi.useRealTimers();
  });

  test('clicking a segmented option marks it pending and PATCHes the new value', async () => {
    vi.useFakeTimers();
    const harness = makeHarness({}, SEGMENTED_SCHEMA);
    harness.push({
      type: 'hello',
      draft: { plan_name: 'orm', plan_args: { sweep: 'bidirectional' } },
      revision: 1,
    });
    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { sweep: 'bidirectional' } },
      revision: 1,
    });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    const options = harness.getCollector().fields.sweep.el.querySelectorAll('.segmented-option');
    options[1].click(); // monodirectional

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(1);
    const body = harness.deps.patchDraft.mock.calls[0][0];
    expect(body.plan_args_patch).toEqual({ sweep: 'monodirectional' });

    harness.client.destroy();
    vi.useRealTimers();
  });

  test('adding then removing a table row marks it pending and PATCHes each time', async () => {
    vi.useFakeTimers();
    const harness = makeHarness({}, TABLE_SCHEMA);
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: {} }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: {} }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    const axesEl = harness.getCollector().fields.axes.el;
    axesEl.querySelector('.table-add').click(); // second (now two) blank rows
    const rows = axesEl.querySelectorAll('.obj-table tbody tr');
    const inputs = rows[0].querySelectorAll('input');
    inputs[0].value = 'QF1';
    inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
    inputs[1].value = '5';
    inputs[1].dispatchEvent(new Event('input', { bubbles: true }));

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(1);
    expect(harness.deps.patchDraft.mock.calls[0][0].plan_args_patch).toEqual({
      axes: [{ setpoint: 'QF1', num_points: 5 }],
    });

    // Now remove that row -- another structural form-change, another PATCH.
    axesEl.querySelector('.row-x').click();
    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(2);
    expect(harness.deps.patchDraft.mock.calls[1][0].remove).toEqual(['axes']);

    harness.client.destroy();
    vi.useRealTimers();
  });

  test('a programmatic applyValues on a container field never marks it pending', async () => {
    vi.useFakeTimers();
    const harness = makeHarness({}, TABLE_SCHEMA);
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: {} }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: {} }, revision: 1 });
    await harness.client.onPlanSelected('orm');

    harness.getCollector().applyValues({ axes: [{ setpoint: 'QD1', num_points: 3 }] });

    await vi.advanceTimersByTimeAsync(1000);

    expect(harness.deps.patchDraft).not.toHaveBeenCalled();

    harness.client.destroy();
    vi.useRealTimers();
  });
});

describe('createDraftClient — removed keys clear on apply (not just added/changed)', () => {
  test('a change frame whose changed[] key is no longer in draftArgs blanks that field', async () => {
    const harness = makeHarness();
    harness.push({
      type: 'hello',
      draft: { plan_name: 'orm', plan_args: { num: 5, span_a: 2 } },
      revision: 1,
    });
    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { num: 5, span_a: 2 } },
      revision: 1,
    });
    await harness.client.onPlanSelected('orm');
    expect(harness.getCollector()()).toEqual({ num: 5, span_a: 2 });

    // The agent removed span_a: the new draft no longer carries it, but it's
    // still named in changed[].
    harness.push({
      type: 'change',
      draft: { plan_name: 'orm', plan_args: { num: 5 } },
      changed: ['span_a'],
      revision: 2,
      origin: 'mcp-agent',
    });
    await flushMicrotasks();

    expect(harness.getCollector()()).toEqual({ num: 5 });
  });

  test('a resync (e.g. after a 409) clears a field absent from the fresh snapshot', async () => {
    vi.useFakeTimers();
    const harness = makeHarness();
    harness.push({
      type: 'hello',
      draft: { plan_name: 'orm', plan_args: { num: 5, span_a: 2 } },
      revision: 1,
    });
    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { num: 5, span_a: 2 } },
      revision: 1,
    });
    await harness.client.onPlanSelected('orm');
    expect(harness.getCollector()()).toEqual({ num: 5, span_a: 2 });

    // 409 on the next flush -> resync to a snapshot that no longer has span_a.
    harness.deps.patchDraft.mockResolvedValueOnce({
      ok: false,
      status: 409,
      body: { code: 'plan_name_mismatch' },
    });
    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'orm', plan_args: { num: 5 } },
      revision: 2,
    });

    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    numInput.value = '6';
    numInput.dispatchEvent(new Event('input', { bubbles: true }));

    await harness.client.flushNow();

    // span_a is gone from the resynced snapshot -- it must be cleared, not
    // left showing its stale prior value.
    expect(harness.getCollector()()).toEqual({ num: 5 });

    harness.client.destroy();
    vi.useRealTimers();
  });
});

describe('createDraftClient — flush coalesces edits added mid-flight', () => {
  test('an edit landing while a PATCH is in flight is drained by the same flush, and pins its revision', async () => {
    const harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');

    /** @type {(value: any) => void} */
    let resolveFirstPatch = () => {};
    const firstPatch = new Promise((resolve) => {
      resolveFirstPatch = resolve;
    });
    harness.deps.patchDraft
      .mockReturnValueOnce(firstPatch)
      .mockResolvedValueOnce({ ok: true, status: 200, body: { revision: 20, changed: ['span_a'], plan_name: 'orm' } });

    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    numInput.value = '2';
    numInput.dispatchEvent(new Event('input', { bubbles: true }));

    const flushPromise = harness.client.flushNow();
    // A second edit lands on a DIFFERENT field while the first PATCH is
    // still in flight (its promise hasn't resolved yet).
    const spanInput = /** @type {HTMLInputElement} */ (
      harness.getCollector().fields.span_a.el.querySelector('input')
    );
    spanInput.value = '3';
    spanInput.dispatchEvent(new Event('input', { bubbles: true }));

    resolveFirstPatch({ ok: true, status: 200, body: { revision: 10, changed: ['num'], plan_name: 'orm' } });
    const result = await flushPromise;

    expect(harness.deps.patchDraft).toHaveBeenCalledTimes(2);
    expect(harness.deps.patchDraft.mock.calls[0][0].plan_args_patch).toEqual({ num: 2 });
    expect(harness.deps.patchDraft.mock.calls[1][0].plan_args_patch).toEqual({ span_a: 3 });
    // The mid-flight edit's own PATCH revision is what gets pinned.
    expect(result).toEqual({ patched: true, revision: 20 });

    harness.client.destroy();
  });
});

describe('createDraftClient — 422 keeps the value and surfaces the rejection, no resync', () => {
  test('a 422 does not resync and reports the field-scoped detail', async () => {
    vi.useFakeTimers();
    const harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');

    harness.deps.patchDraft.mockResolvedValueOnce({
      ok: false,
      status: 422,
      body: { detail: { field: 'num', error: 'value is not a valid integer' } },
    });

    const numInput = /** @type {HTMLInputElement} */ (harness.getCollector().fields.num.el.querySelector('input'));
    numInput.value = '999';
    numInput.dispatchEvent(new Event('input', { bubbles: true }));

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.deps.onPatchRejected).toHaveBeenCalledWith({ field: 'num', error: 'value is not a valid integer' });
    // No resync: getDraft was only ever called once, for the initial bind.
    expect(harness.deps.getDraft).toHaveBeenCalledTimes(1);
    // The operator's typed value is left exactly as-is.
    expect(numInput.value).toBe('999');

    harness.client.destroy();
    vi.useRealTimers();
  });
});

describe('createDraftClient — cross-plan rebind reuses selectPlan (reentrancy)', () => {
  test('a plan-change frame while bound calls selectPlan exactly once, which re-enters onPlanSelected and binds', async () => {
    let renderedName = /** @type {string|null} */ (null);
    // `ref` sidesteps the forward-reference: `selectPlan` is only ever
    // *invoked* after `makeHarness` returns and `ref.harness` is set, but it
    // must close over something already in scope at definition time.
    const ref = /** @type {{ harness?: any }} */ ({});
    const harness = makeHarness({
      selectPlan: vi.fn(async (/** @type {string} */ name) => {
        renderedName = name;
        await ref.harness.client.onPlanSelected(name);
      }),
    });
    ref.harness = harness;

    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    harness.deps.getDraft.mockResolvedValueOnce({
      draft: { plan_name: 'grid_scan', plan_args: { num: 2 } },
      revision: 2,
    });
    harness.push({
      type: 'plan-change',
      draft: { plan_name: 'grid_scan', plan_args: { num: 2 } },
      changed: ['num'],
      revision: 2,
      origin: 'mcp-agent',
    });
    await flushMicrotasks();

    expect(harness.deps.selectPlan).toHaveBeenCalledTimes(1);
    expect(harness.deps.selectPlan).toHaveBeenCalledWith('grid_scan');
    expect(renderedName).toBe('grid_scan');
    expect(harness.client.isBound()).toBe(true);
    expect(harness.getCollector()()).toEqual({ num: 2 });

    harness.client.destroy();
  });
});

describe('createDraftClient — orchestrator-level revision gap', () => {
  test('a frame arriving at last+3 triggers a GET /draft resync and resets the baseline to the snapshot', async () => {
    const harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.deps.getDraft).toHaveBeenCalledTimes(1);

    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 9 } }, revision: 4 });
    harness.push({
      type: 'change',
      draft: { plan_name: 'orm', plan_args: { num: 9 } },
      changed: ['num'],
      revision: 4, // last applied is 1; 4 > 1 + 1 -> gap.
      origin: 'mcp-agent',
    });
    await flushMicrotasks();

    expect(harness.deps.getDraft).toHaveBeenCalledTimes(2);
    expect(harness.client.getLastAppliedRevision()).toBe(4);
    expect(harness.getCollector()()).toEqual({ num: 9 });

    harness.client.destroy();
  });
});

describe('createDraftClient — resync epoch guards a late-resolving fetch', () => {
  test('a slow resync that resolves after a newer one does not clobber the newer state', async () => {
    const harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    /** @type {(value: any) => void} */
    let resolveFirst = () => {};
    const firstPromise = new Promise((resolve) => {
      resolveFirst = resolve;
    });
    harness.deps.getDraft
      .mockReturnValueOnce(firstPromise)
      .mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 2 } }, revision: 5 });

    const firstResyncPromise = harness.client.resync(); // epoch N, stays pending
    const secondResyncPromise = harness.client.resync(); // epoch N+1, resolves immediately

    await secondResyncPromise;
    expect(harness.client.getLastAppliedRevision()).toBe(5);
    expect(harness.getCollector()()).toEqual({ num: 2 });

    // The first (now-superseded) fetch finally resolves, with an OLDER
    // revision than the one already applied -- it must be discarded, not
    // reset the baseline backward.
    resolveFirst({ draft: { plan_name: 'orm', plan_args: { num: 99 } }, revision: 3 });
    await firstResyncPromise;

    expect(harness.client.getLastAppliedRevision()).toBe(5);
    expect(harness.getCollector()()).toEqual({ num: 2 });

    harness.client.destroy();
  });

  test('a hello frame arriving mid-fetch invalidates an in-flight fetch (bridge-restart-safe)', async () => {
    const harness = makeHarness();
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    harness.deps.getDraft.mockResolvedValueOnce({ draft: { plan_name: 'orm', plan_args: { num: 1 } }, revision: 1 });
    await harness.client.onPlanSelected('orm');
    expect(harness.client.isBound()).toBe(true);

    /** @type {(value: any) => void} */
    let resolveSlowFetch = () => {};
    const slowFetch = new Promise((resolve) => {
      resolveSlowFetch = resolve;
    });
    harness.deps.getDraft.mockReturnValueOnce(slowFetch);

    // Start a resync whose GET /draft never resolves yet.
    const resyncPromise = harness.client.resync();

    // While that fetch is still in flight, the bridge restarts: a hello
    // frame arrives with a LOW revision (the process-lifetime counter reset).
    // A hello is itself a baseline reset and must invalidate the earlier
    // in-flight fetch's epoch, even though that fetch was never routed
    // through reduceFrame at all.
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: { num: 7 } }, revision: 2 });
    await flushMicrotasks();

    expect(harness.client.getLastAppliedRevision()).toBe(2);
    expect(harness.getCollector()()).toEqual({ num: 7 });

    // The slow fetch that started BEFORE the hello finally resolves,
    // carrying a revision that would look "newer" than the post-restart
    // counter -- it must be discarded, not clobber the hello's baseline.
    resolveSlowFetch({ draft: { plan_name: 'orm', plan_args: { num: 999 } }, revision: 50 });
    await resyncPromise;

    expect(harness.client.getLastAppliedRevision()).toBe(2);
    expect(harness.getCollector()()).toEqual({ num: 7 });

    harness.client.destroy();
  });
});

describe('createSSEConnection', () => {
  /** @returns {{FakeES: any, instances: any[]}} */
  function makeFakeEventSource() {
    /** @type {any[]} */
    const instances = [];
    class FakeES {
      /** @param {string} url */
      constructor(url) {
        this.url = url;
        this.closed = false;
        instances.push(this);
      }
      close() {
        this.closed = true;
      }
    }
    return { FakeES, instances };
  }

  test('reconnects with capped exponential backoff on repeated errors', () => {
    vi.useFakeTimers();
    const { FakeES, instances } = makeFakeEventSource();
    const conn = createSSEConnection('/draft/events', { onFrame: vi.fn(), EventSourceCtor: FakeES });

    expect(instances).toHaveLength(1);
    instances[0].onerror();
    expect(instances[0].closed).toBe(true);

    vi.advanceTimersByTime(999);
    expect(instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(instances).toHaveLength(2); // first backoff: 1000ms

    instances[1].onerror();
    vi.advanceTimersByTime(1999);
    expect(instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(instances).toHaveLength(3); // second backoff: 2000ms

    instances[2].onerror();
    vi.advanceTimersByTime(3999);
    expect(instances).toHaveLength(3);
    vi.advanceTimersByTime(1);
    expect(instances).toHaveLength(4); // third backoff: 4000ms

    conn.close();
    vi.useRealTimers();
  });

  test('backoff caps at 15s', () => {
    vi.useFakeTimers();
    const { FakeES, instances } = makeFakeEventSource();
    const conn = createSSEConnection('/draft/events', { onFrame: vi.fn(), EventSourceCtor: FakeES });

    // Walk the uncapped doubling sequence (1s, 2s, 4s, 8s) — each iteration
    // errors the latest instance and advances exactly that step's delay so
    // the next `connect()` actually fires before erroring again.
    for (const delay of [1000, 2000, 4000, 8000]) {
      instances[instances.length - 1].onerror();
      vi.advanceTimersByTime(delay);
    }
    expect(instances).toHaveLength(5); // one initial connection + four reconnects

    // The next backoff would double 8000 -> 16000ms uncapped; it must cap at 15000.
    instances[4].onerror();
    vi.advanceTimersByTime(14999);
    expect(instances).toHaveLength(5);
    vi.advanceTimersByTime(1);
    expect(instances).toHaveLength(6);

    conn.close();
    vi.useRealTimers();
  });

  test('backoff resets after a successful frame', () => {
    vi.useFakeTimers();
    const { FakeES, instances } = makeFakeEventSource();
    const conn = createSSEConnection('/draft/events', { onFrame: vi.fn(), EventSourceCtor: FakeES });

    instances[0].onerror();
    vi.advanceTimersByTime(1000);
    expect(instances).toHaveLength(2); // reconnected after 1000ms

    instances[1].onmessage({ data: JSON.stringify({ type: 'hello', draft: null, revision: 0 }) });
    instances[1].onerror();
    vi.advanceTimersByTime(999);
    expect(instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(instances).toHaveLength(3); // reset to 1000ms again, not 2000ms

    conn.close();
    vi.useRealTimers();
  });

  test('close() during a pending retry cancels the reconnect', () => {
    vi.useFakeTimers();
    const { FakeES, instances } = makeFakeEventSource();
    const conn = createSSEConnection('/draft/events', { onFrame: vi.fn(), EventSourceCtor: FakeES });

    instances[0].onerror();
    conn.close();
    vi.advanceTimersByTime(20000);
    expect(instances).toHaveLength(1); // no reconnect after close()

    vi.useRealTimers();
  });

  test('a well-formed frame is forwarded to onFrame', () => {
    const { FakeES, instances } = makeFakeEventSource();
    const onFrame = vi.fn();
    const conn = createSSEConnection('/draft/events', { onFrame, EventSourceCtor: FakeES });

    const frame = { type: 'hello', draft: null, revision: 0 };
    instances[0].onmessage({ data: JSON.stringify(frame) });

    expect(onFrame).toHaveBeenCalledWith(frame);
    conn.close();
  });
});

describe('generateClientId', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test('uses crypto.randomUUID when available', () => {
    vi.stubGlobal('crypto', { randomUUID: () => '11111111-1111-1111-1111-111111111111' });
    expect(generateClientId()).toBe('11111111-1111-1111-1111-111111111111');
  });

  test('falls back to a Math.random-based id when crypto.randomUUID is unavailable', () => {
    vi.stubGlobal('crypto', {});
    const id = generateClientId();
    expect(id).toMatch(/^tab-/);
    expect(id.length).toBeGreaterThan(4);
  });
});

describe('buildLaunchRequestBody', () => {
  test('bound mode sends only draft_revision, never plan_name/plan_args', () => {
    const body = buildLaunchRequestBody({
      bound: true,
      pinnedRevision: 7,
      planName: 'orm',
      planArgs: { num: 1 },
    });
    expect(body).toEqual({ draft_revision: 7 });
  });

  test('unbound (manual) mode sends the collected plan_name/plan_args', () => {
    const body = buildLaunchRequestBody({
      bound: false,
      pinnedRevision: null,
      planName: 'orm',
      planArgs: { num: 1 },
    });
    expect(body).toEqual({ plan_name: 'orm', plan_args: { num: 1 } });
  });
});

describe('classifyLaunchResponse', () => {
  test('200 writes_not_armed', () => {
    expect(classifyLaunchResponse(200, { status: 'writes_not_armed' })).toEqual({ type: 'writes_not_armed' });
  });

  test('200 with a run_id', () => {
    expect(classifyLaunchResponse(200, { run_id: 'run-123' })).toEqual({ type: 'run_started', runId: 'run-123' });
  });

  test('409 with code stale_draft_revision is distinguished from a bare 409', () => {
    expect(classifyLaunchResponse(409, { code: 'stale_draft_revision' })).toEqual({
      type: 'stale_draft_revision',
    });
  });

  test('409 with code draft_revision_already_launched is its own outcome, distinct from stale', () => {
    expect(classifyLaunchResponse(409, { code: 'draft_revision_already_launched' })).toEqual({
      type: 'draft_revision_already_launched',
    });
    // The two 409 discriminators must never collapse into each other.
    expect(classifyLaunchResponse(409, { code: 'stale_draft_revision' })).toEqual({
      type: 'stale_draft_revision',
    });
  });

  test('409 without a code is a generic conflict', () => {
    expect(classifyLaunchResponse(409, { detail: 'already running' })).toEqual({
      type: 'conflict',
      detail: 'already running',
    });
  });

  test('502 is bridge_unreachable', () => {
    expect(classifyLaunchResponse(502, null)).toEqual({ type: 'bridge_unreachable' });
  });

  test('any other status is a generic error carrying the detail or status', () => {
    expect(classifyLaunchResponse(500, { detail: 'boom' })).toEqual({ type: 'error', detail: 'boom' });
    expect(classifyLaunchResponse(500, null)).toEqual({ type: 'error', detail: 'HTTP 500' });
  });
});

describe('createDraftClient — launched banner', () => {
  /** @type {any} */
  let harness;

  beforeEach(() => {
    harness = makeHarness();
  });

  afterEach(() => {
    harness.client.destroy();
    document.body.replaceChildren();
  });

  test('a launched frame surfaces the banner via onLaunchBanner', async () => {
    harness.push({ type: 'hello', draft: { plan_name: 'orm', plan_args: {} }, revision: 1 });
    await flushMicrotasks();
    harness.push({ type: 'launched', draft: null, revision: 1, run_id: 'run-xyz' });
    await flushMicrotasks();
    expect(harness.deps.onLaunchBanner).toHaveBeenCalledWith({ runId: 'run-xyz', revision: 1 });
  });

  test('a malformed launched frame (no run_id) is dropped — onLaunchBanner never fires', async () => {
    harness.push({ type: 'hello', draft: null, revision: 1 });
    await flushMicrotasks();
    harness.push({ type: 'launched', draft: null, revision: 2 });
    await flushMicrotasks();
    expect(harness.deps.onLaunchBanner).not.toHaveBeenCalled();
  });
});

describe('resultsPanelUrl', () => {
  test('swaps the plan panel segment for the results panel and deep-links the run', () => {
    expect(resultsPanelUrl('/panel/plan', 'run-1')).toBe('/panel/scan-results/?run_id=run-1');
  });

  test('falls back to an absolute results path when served without a proxy prefix', () => {
    expect(resultsPanelUrl('', 'run-1')).toBe('/panel/scan-results/?run_id=run-1');
  });

  test('URL-encodes the run id', () => {
    expect(resultsPanelUrl('/panel/plan', 'a b/c?d')).toBe('/panel/scan-results/?run_id=a%20b%2Fc%3Fd');
  });
});

describe('buildLaunchBanner', () => {
  test('renders the revision prefix and a results-panel link (run_id via textContent)', () => {
    const frag = buildLaunchBanner(
      document,
      { runId: 'run-77', revision: 9 },
      (id) => `/panel/scan-results/?run_id=${id}`
    );
    const host = document.createElement('div');
    host.appendChild(frag);
    expect(host.textContent).toBe('revision 9 launched → run run-77');
    const link = /** @type {HTMLAnchorElement|null} */ (host.querySelector('a.launch-run-link'));
    expect(link).not.toBeNull();
    if (!link) throw new Error('unreachable: link asserted non-null');
    expect(link.getAttribute('href')).toBe('/panel/scan-results/?run_id=run-77');
    expect(link.dataset.runId).toBe('run-77');
    expect(link.target).toBe('_blank');
  });

  test('an HTML-bearing run_id never becomes markup — only a text node and an encoded href', () => {
    const evil = '<img src=x onerror=alert(1)>';
    const frag = buildLaunchBanner(document, { runId: evil, revision: 1 }, (id) =>
      resultsPanelUrl('/panel/plan', id)
    );
    const host = document.createElement('div');
    host.appendChild(frag);
    // Nothing was parsed from the payload; it survives verbatim as text.
    expect(host.querySelector('img')).toBeNull();
    const link = /** @type {HTMLAnchorElement|null} */ (host.querySelector('a.launch-run-link'));
    expect(link).not.toBeNull();
    if (!link) throw new Error('unreachable: link asserted non-null');
    expect(link.textContent).toBe(`run ${evil}`);
    expect(link.getAttribute('href')).toBe(`/panel/scan-results/?run_id=${encodeURIComponent(evil)}`);
  });
});

/** Let queued microtasks (chained promises inside onFrame's async handling) settle. */
async function flushMicrotasks() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}
