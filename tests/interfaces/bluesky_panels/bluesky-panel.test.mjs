/**
 * Unit tests for the BLUESKY panel: its queue logic, its Results view, and
 * the shell that holds Plans | Queue | Results together.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/bluesky_panels/bluesky-panel.test.mjs
 *
 * Three layers are exercised:
 *  - The pure queue decisions (`reduceQueueFrame`, `queueControls`,
 *    `describeProgress`, `classifyQueueRefusal`, the move-body builders,
 *    `historyRecords`) against literal wire frames — no DOM, no network.
 *  - `createQueueStream` with a fake EventSource constructor, so reconnect
 *    and malformed-frame handling run without a real stream.
 *  - `createResultsView` against a real (happy-dom) element bag and a stubbed
 *    `fetch`, which is the only way to pin the record-404-but-data-present
 *    case the bridge documents.
 */

import { readFileSync } from 'node:fs';
import { cwd } from 'node:process';

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  ABORT_LABEL,
  CONFIRM_ABORT_LABEL,
  CONFIRM_WITHDRAW_STOP_LABEL,
  STOP_LABEL,
  WITHDRAW_STOP_LABEL,
  abortButtonClass,
  abortButtonLabel,
  abortControl,
  abortOutcomeTone,
  abortRefusalTone,
  abortSuccessMessage,
  classifyQueueRefusal,
  createInitialQueueState,
  createQueueStream,
  describeProgress,
  describeQueueStatus,
  describeStartRequest,
  historyChanged,
  historyEmptyState,
  historyRecords,
  itemParamSummary,
  itemRunId,
  moveDownBody,
  moveUpBody,
  queueControls,
  queueEmptyState,
  reduceQueueFrame,
  refusalTone,
  startRequest,
  stopButtonClass,
  stopButtonLabel,
  writeOutcomeTone,
} from '../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/queue-client.js';
import {
  badgeToneForStatus,
  createResultsView,
  formatCell,
  runMetaParts,
  shouldKeepPolling,
} from '../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/results-view.js';
// The absolute specifier the panel itself imports, so the `setTheme()` this
// file calls drives the very same subscriber set `createResultsView` joins.
import { setTheme } from '/design-system/js/theme-manager.js';

/** A manager summary in the shape `_status_summary` publishes. */
function summary(overrides = {}) {
  return {
    available: true,
    manager_state: 'idle',
    worker_environment_exists: true,
    items_in_queue: 0,
    items_in_history: 0,
    running_item_uid: null,
    plan_queue_uid: 'q1',
    plan_history_uid: 'h1',
    queue_stop_pending: false,
    queue_autostart_enabled: false,
    ...overrides,
  };
}

function item(overrides = {}) {
  return {
    item_uid: 'uid-1',
    name: 'grid_scan',
    item_type: 'plan',
    kwargs: {},
    meta: { osprey_run_id: 'run-1' },
    ...overrides,
  };
}

function frame(overrides = {}) {
  return { type: 'queue', status: summary(), items: [], running_item: null, ...overrides };
}

// ---------------------------------------------------------------------------
// Frame reduction
// ---------------------------------------------------------------------------

describe('reduceQueueFrame', () => {
  test('a hello frame seeds state and marks the stream connected', () => {
    const next = reduceQueueFrame(createInitialQueueState(), frame({ type: 'hello' }));
    expect(next.connected).toBe(true);
    expect(next.frames).toBe(1);
    expect(next.status?.manager_state).toBe('idle');
  });

  test('frames are full snapshots — a later frame replaces, never merges', () => {
    const first = reduceQueueFrame(
      createInitialQueueState(),
      frame({ items: [item(), item({ item_uid: 'uid-2' })] })
    );
    const second = reduceQueueFrame(first, frame({ items: [item({ item_uid: 'uid-9' })] }));
    expect(second.items.map((i) => i.item_uid)).toEqual(['uid-9']);
  });

  test('the running item is carried through with its progress record', () => {
    const running = item({ item_uid: 'uid-r', progress: { rows_seen: 4, fraction: 0.5 } });
    const next = reduceQueueFrame(createInitialQueueState(), frame({ running_item: running }));
    expect(next.runningItem?.progress.rows_seen).toBe(4);
  });

  test('an unrecognized or malformed frame leaves state untouched by identity', () => {
    const state = reduceQueueFrame(createInitialQueueState(), frame({ items: [item()] }));
    expect(reduceQueueFrame(state, { type: 'heartbeat' })).toBe(state);
    expect(reduceQueueFrame(state, null)).toBe(state);
    expect(reduceQueueFrame(state, 'nonsense')).toBe(state);
    // The point of the identity check: a bad frame must not blank a live queue.
    expect(state.items).toHaveLength(1);
  });

  test('a non-array items field degrades to an empty list rather than throwing', () => {
    const next = reduceQueueFrame(createInitialQueueState(), frame({ items: null }));
    expect(next.items).toEqual([]);
  });

  test('non-object entries in items are dropped', () => {
    const next = reduceQueueFrame(createInitialQueueState(), frame({ items: [item(), null, 7] }));
    expect(next.items).toHaveLength(1);
  });

  test('an empty running_item object reads as no running item', () => {
    const next = reduceQueueFrame(createInitialQueueState(), frame({ running_item: null }));
    expect(next.runningItem).toBeNull();
  });
});

describe('historyChanged', () => {
  test('a moved history uid asks for a re-fetch', () => {
    expect(historyChanged(summary(), summary({ plan_history_uid: 'h2' }))).toBe(true);
  });

  test('a moved item count asks for a re-fetch', () => {
    expect(historyChanged(summary(), summary({ items_in_history: 3 }))).toBe(true);
  });

  test('a run leaving the running slot asks for a re-fetch', () => {
    expect(historyChanged(summary({ running_item_uid: 'uid-1' }), summary())).toBe(true);
  });

  test('an unchanged summary does not', () => {
    expect(historyChanged(summary(), summary())).toBe(false);
  });

  test('the first summary always does, and an unavailable one never does', () => {
    expect(historyChanged(null, summary())).toBe(true);
    expect(historyChanged(summary(), null)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Status + controls
// ---------------------------------------------------------------------------

describe('describeQueueStatus', () => {
  test('an unreadable manager reads as an error with its reason', () => {
    const described = describeQueueStatus({ available: false, reason: 'manager_unreachable' });
    expect(described.tone).toBe('err');
    expect(described.label).toBe('unavailable');
    expect(described.note).toContain('manager_unreachable');
  });

  test('an idle manager is ok, a draining one is in-flight', () => {
    expect(describeQueueStatus(summary()).tone).toBe('ok');
    expect(describeQueueStatus(summary({ manager_state: 'executing_queue' })).tone).toBe('info');
  });

  test('autostart is surfaced — it is observed, never assumed off', () => {
    const described = describeQueueStatus(summary({ queue_autostart_enabled: true }));
    expect(described.note).toContain('Autostart');
  });

  test('a null status reads as unavailable rather than throwing', () => {
    expect(describeQueueStatus(null).label).toBe('unavailable');
  });
});

describe('start request (the tokenless agent asking this panel to arm)', () => {
  test('startRequest surfaces the summary record, and only a real one', () => {
    const record = {
      request_id: 'r1',
      requested_by: 'agent',
      requested_at: '2026-08-12T20:00:00+00:00',
      items_in_queue: 2,
    };
    const withRequest = reduceQueueFrame(
      createInitialQueueState(),
      frame({ status: summary({ start_request: record }) })
    );
    expect(startRequest(withRequest)).toEqual(record);

    const without = reduceQueueFrame(createInitialQueueState(), frame());
    expect(startRequest(without)).toBeNull();

    // A summary that never mentions the key, a null status, and a malformed
    // (non-object) record all read as "nothing to render", never a throw.
    expect(startRequest(createInitialQueueState())).toBeNull();
    const malformed = reduceQueueFrame(
      createInitialQueueState(),
      frame({ status: summary({ start_request: 'yes please' }) })
    );
    expect(startRequest(malformed)).toBeNull();
  });

  test('the request survives a manager outage frame — it is bridge state', () => {
    const record = { request_id: 'r1', requested_by: 'agent', items_in_queue: 1 };
    const outage = reduceQueueFrame(
      createInitialQueueState(),
      frame({
        status: {
          available: false,
          reason: 'manager_unreachable',
          start_request: record,
        },
      })
    );
    expect(startRequest(outage)).toEqual(record);
  });

  test('describeStartRequest names who asked, how much, and when', () => {
    const sentence = describeStartRequest({
      request_id: 'r1',
      requested_by: 'agent',
      requested_at: '2026-08-12T20:00:00+00:00',
      items_in_queue: 2,
    });
    expect(sentence).toContain('The agent asks to start the queue');
    expect(sentence).toContain('2 items at the time');
    expect(sentence).toContain('requested ');
    expect(sentence).toContain('exactly as listed below');
  });

  test('describeStartRequest omits what it cannot state truthfully', () => {
    const sentence = describeStartRequest({ request_id: 'r1' });
    expect(sentence).toContain('The agent asks to start the queue');
    expect(sentence).toContain('the queued items');
    expect(sentence).not.toContain('requested ');

    // A singular count reads as one item, not "1 items".
    expect(describeStartRequest({ items_in_queue: 1 })).toContain('1 item at the time');

    // An unparseable timestamp is dropped, never rendered as "Invalid Date".
    expect(describeStartRequest({ requested_at: 'not-a-date' })).not.toContain('Invalid');
  });
});

describe('queueControls', () => {
  /**
   * @param {any} status
   * @param {any[]} [items]
   * @param {any} [runningItem]
   */
  function stateWith(status, items = [], runningItem = null) {
    return { status, items, runningItem, connected: true, frames: 1 };
  }

  test('an unreadable manager disables start but never the halt', () => {
    const controls = queueControls(stateWith({ available: false, reason: 'x' }));
    expect(controls.start.disabled).toBe(true);
    expect(controls.start.reason).toBeTruthy();
    // An unreadable manager summary is exactly when an operator most needs to
    // send a halt; what this panel knows becomes advisory text, not a gate.
    expect(controls.stop).not.toHaveProperty('disabled');
    expect(controls.stop.note).toContain('still sent');
  });

  test('start is live once something is queued', () => {
    const controls = queueControls(stateWith(summary(), [item()]));
    expect(controls.start.disabled).toBe(false);
    expect(controls.start.reason).toBeNull();
  });

  test('start is off with an empty queue, and off again while it drains', () => {
    expect(queueControls(stateWith(summary())).start.reason).toContain('Nothing');
    const draining = queueControls(stateWith(summary({ manager_state: 'executing_queue' }), [item()]));
    expect(draining.start.disabled).toBe(true);
    expect(draining.start.reason).toContain('already running');
  });

  test('the halt has NO disabled state in any queue condition', () => {
    // The bridge leaves a plain stop completely ungated by design, so there is
    // no state in which this panel may refuse to send one. The shape carries
    // no `disabled` key at all — a key that is always false is an invitation
    // to start setting it.
    const conditions = [
      summary(),
      summary({ manager_state: 'executing_queue' }),
      summary({ manager_state: 'paused' }),
      summary({ queue_stop_pending: true }),
      { available: false, reason: 'manager_unreachable' },
      null,
    ];
    for (const status of conditions) {
      expect(queueControls(stateWith(status)).stop).not.toHaveProperty('disabled');
    }
  });

  test('a plain stop is never marked as arming, whatever the queue is doing', () => {
    const idle = queueControls(stateWith(summary())).stop;
    const active = queueControls(stateWith(summary({ manager_state: 'executing_queue' }))).stop;
    expect(idle.body).toEqual({ cancel: false });
    expect(active.body).toEqual({ cancel: false });
    expect(idle.arming).toBe(false);
    expect(active.arming).toBe(false);
    // An idle queue is advisory context, not a refusal.
    expect(idle.note).toContain('still accepted');
    expect(active.note).toBeNull();
  });

  test('a pending stop flips the control to a withdrawal, marked as arming', () => {
    const controls = queueControls(
      stateWith(summary({ manager_state: 'executing_queue', queue_stop_pending: true }))
    );
    expect(controls.stop.body).toEqual({ cancel: true });
    expect(controls.stop.arming).toBe(true);
    // The consequence, spelled out: this un-halts a queue a human halted.
    expect(controls.stop.note).toContain('keep draining');
  });
});

describe('stop button presentation', () => {
  const plain = { body: { cancel: false }, arming: false, note: null };
  const withdrawal = { body: { cancel: true }, arming: true, note: 'consequence' };

  test('a plain stop is one click and never wears the caution class', () => {
    expect(stopButtonLabel(plain, false)).toBe(STOP_LABEL);
    expect(stopButtonClass(plain, false)).toBe('btn');
    // Even if a stale armed flag survives, a plain stop must not be painted as
    // the consequential action.
    expect(stopButtonClass(plain, true)).toBe('btn');
  });

  test('the withdrawal is two-step, and only the armed step is caution-styled', () => {
    expect(stopButtonLabel(withdrawal, false)).toBe(WITHDRAW_STOP_LABEL);
    expect(stopButtonClass(withdrawal, false)).toBe('btn');
    expect(stopButtonLabel(withdrawal, true)).toBe(CONFIRM_WITHDRAW_STOP_LABEL);
    expect(stopButtonClass(withdrawal, true)).toBe('btn confirm');
  });

  test('the caution class means armed-and-consequential, not merely "stop"', () => {
    // This is the regression that made the review: `.confirm` on the safe halt
    // and no `.confirm` on the hardware-arming withdrawal is exactly inverted
    // from the PLAN panel's meaning of the class.
    expect(stopButtonClass(plain, false)).not.toContain('confirm');
    expect(stopButtonClass(withdrawal, true)).toContain('confirm');
  });

  test('success is only green when nothing was armed', () => {
    expect(writeOutcomeTone(false)).toBe('ok');
    // Green is unreachable for any outcome that can move hardware.
    expect(writeOutcomeTone(true)).toBe('warn');
    expect(writeOutcomeTone(true)).not.toBe('ok');
  });

  test('the resting withdrawal label points the same way as its effect', () => {
    // "Cancel pending stop" scanned as MORE halting — opposite to what the
    // button does. At rest, before any confirm, the label must already say the
    // queue keeps going.
    expect(WITHDRAW_STOP_LABEL).not.toMatch(/cancel/i);
    expect(WITHDRAW_STOP_LABEL.toLowerCase()).toContain('keeps draining');
    expect(STOP_LABEL.toLowerCase()).toContain('stop');
  });
});

describe('empty states assert knowledge, not counts', () => {
  /**
   * @param {any} status
   * @param {any[]} [items]
   */
  function stateWith(status, items = []) {
    return { status, items, runningItem: null, connected: true, frames: 1 };
  }

  test('an unreadable manager never reads as an empty queue', () => {
    // The bridge sends `items: []` alongside an unavailable summary, so a
    // count-based empty state renders an outage as "nothing queued" — and an
    // operator who believes it enqueues a duplicate of pending work.
    const outage = queueEmptyState(stateWith({ available: false, reason: 'manager_unreachable' }));
    expect(outage.hidden).toBe(false);
    expect(outage.message).toContain('unknown');
    expect(outage.message).not.toContain('Nothing queued');
  });

  test('a genuinely empty queue says so, once it has actually been read', () => {
    const known = queueEmptyState(stateWith(summary()));
    expect(known.message).toBe('Nothing queued.');
  });

  test('before the first frame the panel claims nothing', () => {
    const fresh = queueEmptyState(createInitialQueueState());
    expect(fresh.message).toContain('unknown');
  });

  test('a populated queue hides the empty state whatever the status says', () => {
    expect(queueEmptyState(stateWith(summary(), [item()])).hidden).toBe(true);
    expect(queueEmptyState(stateWith({ available: false }, [item()])).hidden).toBe(true);
  });

  test('history claims nothing until a fetch has actually landed', () => {
    const never = historyEmptyState([], false);
    expect(never.hidden).toBe(false);
    expect(never.message).toBe('Completed runs could not be loaded.');

    const loaded = historyEmptyState([], true);
    expect(loaded.message).toBe('No completed runs yet.');

    expect(historyEmptyState([{ id: 'r1' }], true).hidden).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Reordering
// ---------------------------------------------------------------------------

describe('move bodies', () => {
  const items = [item({ item_uid: 'a' }), item({ item_uid: 'b' }), item({ item_uid: 'c' })];

  test('placement is relative to the neighbour uid, not an index', () => {
    expect(moveUpBody(items, 2)).toEqual({ before_uid: 'b' });
    expect(moveDownBody(items, 0)).toEqual({ after_uid: 'b' });
  });

  test('the ends have nowhere to go', () => {
    expect(moveUpBody(items, 0)).toBeNull();
    expect(moveDownBody(items, 2)).toBeNull();
  });

  test('an out-of-range index (a click racing a re-render) yields no request', () => {
    expect(moveUpBody(items, 99)).toBeNull();
    expect(moveDownBody(items, 99)).toBeNull();
    expect(moveDownBody([], 0)).toBeNull();
  });

  test('a neighbour with no uid yields no request rather than a broken body', () => {
    expect(moveUpBody([item({ item_uid: undefined }), item()], 1)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Item identity and summaries
// ---------------------------------------------------------------------------

describe('itemRunId', () => {
  test('reads the OSPREY run id out of item metadata', () => {
    expect(itemRunId(item())).toBe('run-1');
  });

  test('an item enqueued out of band has none', () => {
    expect(itemRunId(item({ meta: {} }))).toBeNull();
    expect(itemRunId(item({ meta: undefined }))).toBeNull();
    expect(itemRunId(null)).toBeNull();
  });

  test('the bridge strips the plan stamp, and the run id still resolves', () => {
    // What the bridge actually puts on the wire: the enqueue path stamps both
    // `osprey_run_id` and `osprey_plan` into the item's metadata, and the
    // queue relays drop the latter. The join key survives that strip.
    expect(itemRunId(item({ meta: { osprey_run_id: 'run-1' } }))).toBe('run-1');
  });
});

describe('itemParamSummary', () => {
  test('kwargs ARE the plan params — no params envelope is opened', () => {
    expect(itemParamSummary(item({ kwargs: { detectors: ['a', 'b'], num: 5 } }))).toBe(
      'detectors=[2], num=5'
    );
    expect(itemParamSummary(item({ kwargs: { params: { num: 5 } } }))).toBe('params={…}');
  });

  test('long values are truncated and extra keys are counted, not dropped silently', () => {
    const summaryText = itemParamSummary(item({ kwargs: { a: 1, b: 2, c: 3, d: 4 } }));
    expect(summaryText).toContain('+1 more');
  });

  test('an item with no kwargs summarizes to nothing', () => {
    expect(itemParamSummary(item({ kwargs: {} }))).toBe('');
    expect(itemParamSummary(null)).toBe('');
  });
});

// ---------------------------------------------------------------------------
// Progress — the indeterminate rules
// ---------------------------------------------------------------------------

describe('describeProgress', () => {
  test('no progress record at all renders indeterminate, never 0%', () => {
    const view = describeProgress(null);
    expect(view.mode).toBe('indeterminate');
    expect(view.percent).toBeNull();
  });

  test('an unknown denominator renders indeterminate with the point count', () => {
    // The common case for an agent-authored session plan: the estimator only
    // recognizes the shipped plan shapes, so `fraction` is null while the scan
    // is running perfectly well.
    const view = describeProgress({ rows_seen: 7, expected_points: null, fraction: null, complete: false });
    expect(view.mode).toBe('indeterminate');
    expect(view.percent).toBeNull();
    expect(view.label).toBe('7 points so far');
  });

  test('a known denominator renders a percentage', () => {
    const view = describeProgress({ rows_seen: 5, expected_points: 20, fraction: 0.25, complete: false });
    expect(view.mode).toBe('determinate');
    expect(view.percent).toBe(25);
    expect(view.label).toBe('5 of 20 points');
  });

  test('a genuine zero is determinate 0%, distinct from unknown', () => {
    const view = describeProgress({ rows_seen: 0, expected_points: 20, fraction: 0, complete: false });
    expect(view.mode).toBe('determinate');
    expect(view.percent).toBe(0);
  });

  test('fraction 1.0 with complete false reads as finishing, not as stuck', () => {
    const view = describeProgress({ rows_seen: 20, expected_points: 20, fraction: 1, complete: false });
    expect(view.percent).toBe(100);
    expect(view.label).toContain('finishing');
    expect(view.complete).toBe(false);
  });

  test('rows_seen counts events and may exceed the stored rows', () => {
    const view = describeProgress({ rows_seen: 12000, expected_points: 12000, fraction: 1, complete: true });
    expect(view.label).toContain('12000 of 12000');
    expect(view.label).toContain('finished');
  });

  test('a completed run with an unknown denominator still reports honestly', () => {
    const view = describeProgress({ rows_seen: 3, expected_points: null, fraction: null, complete: true });
    expect(view.mode).toBe('indeterminate');
    expect(view.label).toBe('3 points — finished');
  });
});

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

describe('classifyQueueRefusal', () => {
  test('classifies on detail.code and surfaces detail.detail verbatim', () => {
    const sentence = 'Starting the queue requires an arming token; none is configured.';
    const refusal = classifyQueueRefusal(503, {
      detail: { code: 'launch_token_required', detail: sentence, manager_state: 'idle' },
    });
    expect(refusal.code).toBe('launch_token_required');
    expect(refusal.message).toBe(sentence);
    expect(refusal.managerState).toBe('idle');
  });

  test('a browse-only refusal keeps its capability record and its flip command', () => {
    const sentence =
      'This deployment uses the mock connector … run `osprey set ' +
      'connector=virtual_accelerator` and redeploy.';
    const refusal = classifyQueueRefusal(409, {
      detail: {
        code: 'browse_only_connector',
        detail: sentence,
        capability: { can_execute: false, reason: 'browse_only_connector', detail: sentence },
      },
    });
    expect(refusal.code).toBe('browse_only_connector');
    expect(refusal.message).toBe(sentence);
    expect(refusal.capability.can_execute).toBe(false);
  });

  test('a TOP-LEVEL code is ignored — the queue relay never unwraps the envelope', () => {
    // Negative control for the silent-failure mode: reading `body.code` on the
    // queue surface finds nothing, and a panel that did would mis-classify
    // every refusal without ever erroring.
    const refusal = classifyQueueRefusal(409, {
      code: 'stale_draft_revision',
      detail: { code: 'queue_request_rejected', detail: 'the manager rejected the request' },
    });
    expect(refusal.code).toBe('queue_request_rejected');
  });

  test('the sidecar 502 (a plain-string detail) still yields a shown sentence', () => {
    const refusal = classifyQueueRefusal(502, { detail: 'bluesky bridge unreachable' });
    expect(refusal.code).toBeNull();
    expect(refusal.message).toBe('bluesky bridge unreachable');
  });

  test('a body-less or unparsable refusal still names its status', () => {
    expect(classifyQueueRefusal(500, null).message).toContain('500');
    expect(classifyQueueRefusal(409, { detail: { code: 'x', detail: '  ' } }).message).toContain('409');
  });

  test('a coded refusal is a warning; an uncoded failure is an error', () => {
    expect(refusalTone('launch_token_required')).toBe('warn');
    expect(refusalTone(null)).toBe('err');
  });

  test('a FAILED HALT is an error everywhere, not routine amber', () => {
    // `abort_pause_timeout` means the Run Engine never paused, so nothing was
    // aborted and the plan may still be moving hardware. Paired with the
    // by-design refusal above so this cannot pass by making everything `err`.
    expect(refusalTone('abort_pause_timeout')).toBe('err');
    expect(refusalTone('launch_token_required')).toBe('warn');
  });

  test('on the abort path a refusal is an error — except nothing to stop', () => {
    // Every refusal of an abort means the machine was not stopped, including
    // the codes other routes mint as by-design degraded states.
    expect(abortRefusalTone('manager_unreachable')).toBe('err');
    expect(abortRefusalTone('queue_request_rejected')).toBe('err');
    expect(abortRefusalTone('abort_pause_timeout')).toBe('err');
    expect(abortRefusalTone(null)).toBe('err');
    // ...but nothing was running, so nothing is moving: that is reassurance.
    expect(abortRefusalTone('nothing_running')).toBe('warn');
    // The escalation is per-call, not a second copy of the code table: the same
    // codes stay amber on the routes where they genuinely are by design.
    expect(refusalTone('manager_unreachable')).toBe('warn');
    expect(refusalTone('queue_request_rejected')).toBe('warn');
  });
});

// ---------------------------------------------------------------------------
// History records
// ---------------------------------------------------------------------------

describe('historyRecords', () => {
  test('keeps terminal runs only — the queue half already shows the rest', () => {
    const records = historyRecords([
      { id: 'r1', status: 'running', plan_name: 'orm' },
      { id: 'r2', status: 'pending', plan_name: 'orm' },
      { id: 'r3', status: 'completed', plan_name: 'grid_scan' },
      { id: 'r4', status: 'stopped', plan_name: 'orm' },
      { id: 'r5', status: 'error', plan_name: 'orm', error: 'boom' },
    ]);
    expect(records.map((r) => r.id)).toEqual(['r3', 'r4', 'r5']);
    expect(records[2].error).toBe('boom');
  });

  test('reads defensively: only `id` is required, everything else may be absent', () => {
    const records = historyRecords([{ id: 'r1', status: 'completed' }]);
    expect(records[0].planName).toBeNull();
    expect(records[0].error).toBeNull();
  });

  test('records with no usable id are skipped, not rendered as blanks', () => {
    expect(historyRecords([{ status: 'completed' }, { id: '', status: 'completed' }, null])).toEqual([]);
  });

  test('a non-list response yields an empty list rather than throwing', () => {
    expect(historyRecords(null)).toEqual([]);
    expect(historyRecords({ detail: 'bluesky bridge unreachable' })).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// SSE transport
// ---------------------------------------------------------------------------

describe('createQueueStream', () => {
  /** Minimal EventSource stand-in: happy-dom has none. */
  class FakeEventSource {
    /** @type {FakeEventSource[]} */
    static instances = [];
    /** @param {string} url */
    constructor(url) {
      this.url = url;
      this.closed = false;
      /** @type {((event: {data: string}) => void)|null} */
      this.onmessage = null;
      /** @type {(() => void)|null} */
      this.onerror = null;
      FakeEventSource.instances.push(this);
    }
    close() {
      this.closed = true;
    }
  }

  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('parsed frames reach onFrame and mark the stream connected', () => {
    /** @type {any[]} */
    const frames = [];
    /** @type {boolean[]} */
    /** @type {boolean[]} */
    const connections = [];
    createQueueStream('/queue/events', {
      onFrame: (f) => frames.push(f),
      onConnectionChange: (c) => connections.push(c),
      EventSourceCtor: /** @type {any} */ (FakeEventSource),
    });
    FakeEventSource.instances[0].onmessage?.({ data: JSON.stringify(frame({ type: 'hello' })) });
    expect(frames[0].type).toBe('hello');
    expect(connections).toEqual([true]);
  });

  test('a malformed frame is ignored rather than killing the stream', () => {
    /** @type {any[]} */
    const frames = [];
    createQueueStream('/queue/events', {
      onFrame: (f) => frames.push(f),
      EventSourceCtor: /** @type {any} */ (FakeEventSource),
    });
    const source = FakeEventSource.instances[0];
    expect(() => source.onmessage?.({ data: 'not json' })).not.toThrow();
    source.onmessage?.({ data: JSON.stringify(frame()) });
    expect(frames).toHaveLength(1);
  });

  test('a dropped connection reports disconnected and reconnects with backoff', () => {
    /** @type {boolean[]} */
    const connections = [];
    createQueueStream('/queue/events', {
      onFrame: () => {},
      onConnectionChange: (c) => connections.push(c),
      EventSourceCtor: /** @type {any} */ (FakeEventSource),
    });
    FakeEventSource.instances[0].onerror?.();
    expect(connections).toEqual([false]);
    expect(FakeEventSource.instances).toHaveLength(1);
    vi.advanceTimersByTime(1000);
    expect(FakeEventSource.instances).toHaveLength(2);

    // Second failure backs off further rather than hammering the sidecar.
    FakeEventSource.instances[1].onerror?.();
    vi.advanceTimersByTime(1000);
    expect(FakeEventSource.instances).toHaveLength(2);
    vi.advanceTimersByTime(1000);
    expect(FakeEventSource.instances).toHaveLength(3);
  });

  test('close() stops both the stream and any pending reconnect', () => {
    const stream = createQueueStream('/queue/events', {
      onFrame: () => {},
      EventSourceCtor: /** @type {any} */ (FakeEventSource),
    });
    FakeEventSource.instances[0].onerror?.();
    stream.close();
    vi.advanceTimersByTime(60000);
    expect(FakeEventSource.instances).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Results half — pure helpers
// ---------------------------------------------------------------------------

describe('results helpers', () => {
  test('a stopped run is a warning, not an error — an operator halted it', () => {
    expect(badgeToneForStatus('completed')).toBe('ok');
    expect(badgeToneForStatus('running')).toBe('info');
    expect(badgeToneForStatus('stopped')).toBe('warn');
    expect(badgeToneForStatus('error')).toBe('err');
  });

  test('run meta omits absent fields instead of fabricating them', () => {
    expect(runMetaParts({ id: 'r1', status: 'pending' })).toEqual([['status', 'pending']]);
    const parts = runMetaParts({
      id: 'r1',
      status: 'running',
      plan_name: 'orm',
      run_uid: 'uid-9',
      progress: { rows_seen: 2, expected_points: null, fraction: null, complete: false },
    });
    expect(parts).toContainEqual(['plan', 'orm']);
    expect(parts).toContainEqual(['progress', '2 points so far']);
    expect(parts).toContainEqual(['run uid', 'uid-9']);
  });

  test('cells render missing values as a dash and trim float noise', () => {
    expect(formatCell(null)).toBe('—');
    expect(formatCell(undefined)).toBe('—');
    expect(formatCell(0.1 + 0.2)).toBe('0.3');
    expect(formatCell('ok')).toBe('ok');
  });

  test('partial data outranks a terminal status when deciding to keep polling', () => {
    expect(shouldKeepPolling('running', false)).toBe(true);
    expect(shouldKeepPolling('completed', false)).toBe(false);
    // Including for a run the manager has rotated away: the record is gone but
    // the durable store is still filling in, and that is what the table shows.
    expect(shouldKeepPolling('completed', true)).toBe(true);
    expect(shouldKeepPolling('stopped', true)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Results half — the view, against a real element bag
// ---------------------------------------------------------------------------

describe('createResultsView', () => {
  function mountElements() {
    document.body.innerHTML = `
      <span id="run-status-badge" hidden></span>
      <p id="run-meta" hidden></p>
      <p id="run-note" hidden></p>
      <div id="results-empty"></div>
      <div id="table-card" hidden><p id="table-note"></p>
        <table><thead><tr id="table-head-row"></tr></thead><tbody id="table-body"></tbody></table>
      </div>
      <div id="figure-card" hidden>
        <p id="figure-note" hidden></p><div id="figure-panels"></div>
      </div>`;
    /** @param {string} id */
    const byId = (id) => /** @type {any} */ (document.getElementById(id));
    return {
      statusBadge: byId('run-status-badge'),
      meta: byId('run-meta'),
      note: byId('run-note'),
      emptyState: byId('results-empty'),
      tableCard: byId('table-card'),
      tableNote: byId('table-note'),
      tableHeadRow: byId('table-head-row'),
      tableBody: byId('table-body'),
      figureCard: byId('figure-card'),
      figurePanels: byId('figure-panels'),
      figureNote: byId('figure-note'),
    };
  }

  /**
   * `routes` is read on every call rather than captured, so a test can swap a
   * route between poll ticks — which is the only way to drive "this tick
   * failed, the last one did not".
   *
   * @param {Record<string, {status: number, body: any}>} routes
   */
  function stubFetch(routes) {
    return vi.fn(async (/** @type {string} */ url) => {
      const route = routes[url];
      if (!route) throw new Error(`unstubbed ${url}`);
      return {
        ok: route.status >= 200 && route.status < 300,
        status: route.status,
        json: async () => route.body,
      };
    });
  }

  const DATA = {
    run_uid: null,
    columns: ['time', 'signal'],
    rows: [
      [1, 10],
      [2, 20],
    ],
    row_count: 2,
    truncated: false,
  };

  /**
   * What `GET /runs/{id}/figure` serves: a bare `Figure`, no wrapper. Typed
   * against the renderer's own wire typedef so a fixture that drifts from the
   * bridge's shape fails typecheck rather than quietly testing a fiction.
   *
   * @type {import('../../../src/osprey/interfaces/bluesky_panels/panels/bluesky/figure-renderer.js').Figure}
   */
  const FIGURE = {
    panels: [
      {
        title: 'signal',
        x_label: 'time',
        y_label: 'signal',
        x_units: 's',
        y_units: 'counts',
        annotations: [],
        mark: {
          kind: 'lines',
          series: [
            {
              label: 'signal',
              points: [
                { x: 1, y: 10 },
                { x: 2, y: 20 },
              ],
              decimated: false,
              source_points: 2,
            },
          ],
          decimated: false,
          source_points: 2,
        },
      },
    ],
    partial: false,
    source: 'live',
    reason: null,
  };

  /** @type {ReturnType<typeof createResultsView>|undefined} */
  let view;

  afterEach(() => {
    if (view) view.destroy();
    view = undefined;
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  test('following nothing shows the empty state and issues no request', async () => {
    const fetchMock = stubFetch({});
    vi.stubGlobal('fetch', fetchMock);
    view = createResultsView({ api: (p) => p, elements: mountElements() });
    view.follow(null);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(view.currentRunId()).toBeNull();
  });

  test('a completed run renders its record, its raw table, and its figure', async () => {
    vi.stubGlobal(
      'fetch',
      stubFetch({
        '/runs/r1': { status: 200, body: { id: 'r1', status: 'completed', plan_name: 'orm' } },
        '/runs/r1/data': { status: 200, body: DATA },
        '/runs/r1/figure': { status: 200, body: FIGURE },
      })
    );
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');
    await vi.waitFor(() => expect(elements.tableCard.hidden).toBe(false));

    expect(elements.statusBadge.textContent).toBe('completed');
    expect(elements.meta.textContent).toContain('orm');
    expect(elements.tableBody.querySelectorAll('tr')).toHaveLength(2);
    expect(elements.tableNote.textContent).toBe('2 rows');

    // The figure is drawn by the renderer, into the container this view hands
    // it — one section per panel, plus the provenance line.
    expect(elements.figureCard.hidden).toBe(false);
    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);
    expect(elements.figureNote.hidden).toBe(false);
    expect(elements.figureNote.textContent).toContain('live data');
  });

  test('a default figure carries its reason and is drawn like any other', async () => {
    // A `reason` means the bridge could not run the plan's own render() and
    // plotted the raw columns instead. That is a fallback VIEW, not an error
    // state: nothing about it may be styled or hidden as a failure.
    vi.stubGlobal(
      'fetch',
      stubFetch({
        '/runs/r1': { status: 200, body: { id: 'r1', status: 'completed' } },
        '/runs/r1/data': { status: 200, body: DATA },
        '/runs/r1/figure': { status: 200, body: { ...FIGURE, reason: 'no_render' } },
      })
    );
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');
    await vi.waitFor(() => expect(elements.figureCard.hidden).toBe(false));

    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);
    // The renderer humanizes the wire vocabulary — `no_render` reads as
    // "no render" — but it never drops the reason.
    expect(elements.figureNote.textContent).toContain('no render');
    expect(elements.emptyState.classList.contains('error')).toBe(false);
  });

  test('a settled run re-draws its figure on a theme change, with no poll tick', async () => {
    // The reason `lastFigure` is retained at all: a settled run has STOPPED
    // polling, so nothing else will ever redraw it. Without this, flipping to
    // light mode would leave a finished scan in the dark palette until the
    // operator picked another run.
    const fetchMock = stubFetch({
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'completed' } },
      '/runs/r1/data': { status: 200, body: DATA },
      '/runs/r1/figure': { status: 200, body: FIGURE },
    });
    vi.stubGlobal('fetch', fetchMock);
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');
    await vi.waitFor(() => expect(elements.figureCard.hidden).toBe(false));

    const drawn = elements.figurePanels.querySelector('.figure-panel');
    const callsBefore = fetchMock.mock.calls.length;

    setTheme('light');

    // The renderer replaces the container's children wholesale, so a new node
    // in place of the old one IS the re-render — and no request was made to
    // get it.
    const redrawn = elements.figurePanels.querySelector('.figure-panel');
    expect(redrawn).not.toBeNull();
    expect(redrawn).not.toBe(drawn);
    expect(fetchMock.mock.calls.length).toBe(callsBefore);
  });

  test('the theme subscription is released on destroy', async () => {
    vi.stubGlobal(
      'fetch',
      stubFetch({
        '/runs/r1': { status: 200, body: { id: 'r1', status: 'completed' } },
        '/runs/r1/data': { status: 200, body: DATA },
        '/runs/r1/figure': { status: 200, body: FIGURE },
      })
    );
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');
    await vi.waitFor(() => expect(elements.figureCard.hidden).toBe(false));

    view.destroy();
    view = undefined;
    const drawn = elements.figurePanels.querySelector('.figure-panel');
    setTheme('dark');
    // A destroyed view still holding a live subscription would keep drawing
    // into a container the panel has moved on from.
    expect(elements.figurePanels.querySelector('.figure-panel')).toBe(drawn);
  });

  test('a run the manager has rotated away still shows its stored data', async () => {
    // Documented and correct, not a gap: `GET /runs/{id}` 404s once the
    // manager forgets the item, while its data stays durable in Tiled.
    vi.stubGlobal(
      'fetch',
      stubFetch({
        '/runs/old': { status: 404, body: { detail: "unknown run 'old'" } },
        '/runs/old/data': { status: 200, body: DATA },
        '/runs/old/figure': { status: 200, body: { ...FIGURE, source: 'tiled' } },
      })
    );
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('old');
    await vi.waitFor(() => expect(elements.tableCard.hidden).toBe(false));

    expect(elements.figureCard.hidden).toBe(false);
    expect(elements.note.hidden).toBe(false);
    expect(elements.note.textContent).toContain('no longer in the queue');
    expect(elements.statusBadge.hidden).toBe(true);
  });

  test('a run with a record but no data yet is not reported as an error', async () => {
    // The bridge-restart case: a still-executing run has no live buffer and
    // nothing in Tiled yet, so 404 on data mid-run is correct.
    vi.stubGlobal(
      'fetch',
      stubFetch({
        '/runs/r2': { status: 200, body: { id: 'r2', status: 'running' } },
        '/runs/r2/data': { status: 404, body: { detail: "unknown run 'r2'" } },
        '/runs/r2/figure': { status: 404, body: { detail: "unknown run 'r2'" } },
      })
    );
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r2');
    await vi.waitFor(() => expect(elements.emptyState.textContent).toContain('No data recorded'));
    expect(elements.emptyState.classList.contains('error')).toBe(false);
    expect(elements.statusBadge.textContent).toBe('running');
    // A figure 404 is the same "neither source knows this run" as a data 404;
    // there is nothing to draw, so the card stays away.
    expect(elements.figureCard.hidden).toBe(true);
  });

  test('a transient figure failure leaves the last good figure on screen', async () => {
    // Mirrors what the table already does with its last good rows. A 502 is
    // the bridge restarting mid-scan — the figure it served a second ago is
    // still the best thing the operator can be shown.
    vi.useFakeTimers();
    /** @type {Record<string, {status: number, body: any}>} */
    const routes = {
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'running' } },
      '/runs/r1/data': { status: 200, body: { ...DATA, partial: true } },
      '/runs/r1/figure': { status: 200, body: { ...FIGURE, partial: true } },
    };
    vi.stubGlobal('fetch', stubFetch(routes));
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');

    await vi.advanceTimersByTimeAsync(0);
    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);

    routes['/runs/r1/figure'] = { status: 502, body: { detail: 'bluesky bridge unreachable' } };
    await vi.advanceTimersByTimeAsync(1100);

    expect(elements.figureCard.hidden).toBe(false);
    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);
  });

  test('an empty figure never replaces a good one', async () => {
    // `source_unavailable` is served as a 200 with no panels: the store could
    // not be read THIS tick, which says nothing about the figure drawn last
    // tick. Blanking here would flicker a settled scan away and back.
    vi.useFakeTimers();
    /** @type {Record<string, {status: number, body: any}>} */
    const routes = {
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'running' } },
      '/runs/r1/data': { status: 200, body: { ...DATA, partial: true } },
      '/runs/r1/figure': { status: 200, body: { ...FIGURE, partial: true } },
    };
    vi.stubGlobal('fetch', stubFetch(routes));
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');

    await vi.advanceTimersByTimeAsync(0);
    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);

    routes['/runs/r1/figure'] = {
      status: 200,
      body: { panels: [], partial: true, source: 'tiled', reason: 'source_unavailable' },
    };
    await vi.advanceTimersByTimeAsync(1100);

    expect(elements.figureCard.hidden).toBe(false);
    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);
  });

  test('a 200 that is not a figure is transient, and never kills the poll tick', async () => {
    // What a misconfigured proxy answering for the bridge looks like. The
    // figure read shares its tick with the record and table reads, so a body
    // this view cannot parse must not throw: that would take the whole tick
    // down and freeze the panel with no visible reason.
    vi.useFakeTimers();
    /** @type {Record<string, {status: number, body: any}>} */
    const routes = {
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'running' } },
      '/runs/r1/data': { status: 200, body: { ...DATA, partial: true } },
      '/runs/r1/figure': { status: 200, body: { ...FIGURE, partial: true } },
    };
    const fetchMock = stubFetch(routes);
    vi.stubGlobal('fetch', fetchMock);
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');

    await vi.advanceTimersByTimeAsync(0);
    routes['/runs/r1/figure'] = { status: 200, body: [] };
    await vi.advanceTimersByTimeAsync(1100);

    // A tick beyond the bad one, because the bad tick's own requests go out
    // before its body is read — counting them would prove nothing.
    const afterBadPoll = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1100);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterBadPoll);

    // Table still updating, last good figure still drawn.
    expect(elements.tableCard.hidden).toBe(false);
    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);
  });

  test('a null record body mid-poll keeps the tick alive and holds the table', async () => {
    // The read proxy's `safe_json` relays a null body on a 200, so every read
    // in this tick can arrive as 200 + null. Unchecked, `body.status` would
    // throw and abandon the tick — no reschedule, and the activity marker
    // stuck on with nothing on screen to explain it.
    vi.useFakeTimers();
    /** @type {Record<string, {status: number, body: any}>} */
    const routes = {
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'running' } },
      '/runs/r1/data': { status: 200, body: { ...DATA, partial: true } },
      '/runs/r1/figure': { status: 200, body: { ...FIGURE, partial: true } },
    };
    const fetchMock = stubFetch(routes);
    vi.stubGlobal('fetch', fetchMock);
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');

    await vi.advanceTimersByTimeAsync(0);
    routes['/runs/r1'] = { status: 200, body: null };
    await vi.advanceTimersByTimeAsync(1100);

    // The tick that SAW the null body has to have rescheduled itself. Counting
    // requests right after it would prove nothing — its own three requests go
    // out before the body is ever read, so that count rises even when the tick
    // then dies. Only a further tick proves it survived.
    const afterBadPoll = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1100);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterBadPoll);

    expect(elements.note.textContent).toContain('Could not load the run record');
    // And the halves that DID answer are still on screen.
    expect(elements.tableCard.hidden).toBe(false);
    expect(elements.tableBody.querySelectorAll('tr')).toHaveLength(2);
    expect(elements.figurePanels.querySelectorAll('.figure-panel')).toHaveLength(1);
  });

  test('a null data body mid-poll keeps the tick alive and holds the last rows', async () => {
    // Same hazard on the table's read: `renderTable` walks `columns` and
    // `rows`, so a body with neither is transient, and transient means the
    // last good window stays where the operator can still read it.
    vi.useFakeTimers();
    /** @type {Record<string, {status: number, body: any}>} */
    const routes = {
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'running' } },
      '/runs/r1/data': { status: 200, body: { ...DATA, partial: true } },
      '/runs/r1/figure': { status: 200, body: { ...FIGURE, partial: true } },
    };
    const fetchMock = stubFetch(routes);
    vi.stubGlobal('fetch', fetchMock);
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');

    await vi.advanceTimersByTimeAsync(0);
    routes['/runs/r1/data'] = { status: 200, body: null };
    await vi.advanceTimersByTimeAsync(1100);

    // Same reasoning as the record case: only a tick AFTER the bad one proves
    // the bad one rescheduled rather than died.
    const afterBadPoll = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1100);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterBadPoll);

    expect(elements.tableCard.hidden).toBe(false);
    expect(elements.tableBody.querySelectorAll('tr')).toHaveLength(2);
    expect(elements.statusBadge.textContent).toBe('running');
    // Not an error state: a read that failed this tick is not a claim that
    // the run went wrong.
    expect(elements.emptyState.hidden).toBe(true);
  });

  test('a run that exists in neither source clears the figure and goes quiet', async () => {
    // The one genuinely terminal case. The banner must not sit over a plot of
    // a run that no longer exists, so whatever was drawn is cleared first —
    // and with nothing left to wait for, the poll stops.
    vi.useFakeTimers();
    const fetchMock = stubFetch({
      '/runs/gone': { status: 404, body: { detail: "unknown run 'gone'" } },
      '/runs/gone/data': { status: 404, body: { detail: "unknown run 'gone'" } },
      '/runs/gone/figure': { status: 404, body: { detail: "unknown run 'gone'" } },
    });
    vi.stubGlobal('fetch', fetchMock);
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('gone');

    await vi.advanceTimersByTimeAsync(0);
    expect(elements.emptyState.textContent).toContain('No record and no data');
    expect(elements.emptyState.classList.contains('error')).toBe(true);
    expect(elements.figureCard.hidden).toBe(true);

    const afterFirstPoll = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1100);
    expect(fetchMock.mock.calls.length).toBe(afterFirstPoll);
  });

  test('a renderer that throws does not take the poll tick down with it', async () => {
    // A body whose `panels` array is well-formed but whose contents are not
    // gets past the wire check and reaches the renderer. Containing the throw
    // here is what keeps the record and table reads — which share this tick —
    // alive, and the tab's activity marker honest.
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    /** @type {Record<string, {status: number, body: any}>} */
    const routes = {
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'running' } },
      '/runs/r1/data': { status: 200, body: { ...DATA, partial: true } },
      '/runs/r1/figure': { status: 200, body: { ...FIGURE, panels: [{ nonsense: true }] } },
    };
    const fetchMock = stubFetch(routes);
    vi.stubGlobal('fetch', fetchMock);
    const elements = mountElements();
    view = createResultsView({ api: (p) => p, elements });
    view.follow('r1');

    await vi.advanceTimersByTimeAsync(0);
    const afterBadPoll = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1100);

    // The first tick already hit the throwing panel, so a further tick is what
    // proves the throw was contained rather than fatal.
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterBadPoll);
    expect(elements.tableCard.hidden).toBe(false);
    // Named specifically, because the theme bridges also log to console.error
    // in this environment (no stylesheet, so the sentinel token reads empty) —
    // a bare "was called" assertion here would pass without a throw at all.
    expect(consoleError).toHaveBeenCalledWith(
      'osprey bluesky results: the figure renderer threw',
      expect.any(Error)
    );
    // And the draw genuinely failed: the card was never revealed.
    expect(elements.figureCard.hidden).toBe(true);
    consoleError.mockRestore();
  });

  test('a run whose figure is still partial keeps polling after its data settles', async () => {
    // The Tiled-served case the bridge documents: a run with no stop document
    // reports `partial: true` on the figure even once its record reads
    // terminal and its data window looks settled. Either half still filling in
    // has to keep the poll alive.
    vi.useFakeTimers();
    const fetchMock = stubFetch({
      '/runs/r1': { status: 200, body: { id: 'r1', status: 'completed' } },
      '/runs/r1/data': { status: 200, body: DATA },
      '/runs/r1/figure': { status: 200, body: { ...FIGURE, partial: true, source: 'tiled' } },
    });
    vi.stubGlobal('fetch', fetchMock);
    view = createResultsView({ api: (p) => p, elements: mountElements() });
    view.follow('r1');

    await vi.advanceTimersByTimeAsync(0);
    const afterFirstPoll = fetchMock.mock.calls.length;
    expect(afterFirstPoll).toBe(3);

    await vi.advanceTimersByTimeAsync(1100);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterFirstPoll);
  });

  test('a rotated-away run whose data is still partial keeps polling', async () => {
    // The record 404s (the manager forgot it) while the durable store is still
    // filling in. The absent record says nothing about whether the data has
    // settled, so `partial: true` alone must keep the poll alive.
    vi.useFakeTimers();
    const fetchMock = stubFetch({
      '/runs/old': { status: 404, body: { detail: "unknown run 'old'" } },
      '/runs/old/data': { status: 200, body: { ...DATA, partial: true } },
      // Settled, so the data half alone is what keeps this alive.
      '/runs/old/figure': { status: 200, body: { ...FIGURE, source: 'tiled' } },
    });
    vi.stubGlobal('fetch', fetchMock);
    view = createResultsView({ api: (p) => p, elements: mountElements() });
    view.follow('old');

    await vi.advanceTimersByTimeAsync(0);
    const afterFirstPoll = fetchMock.mock.calls.length;
    expect(afterFirstPoll).toBe(3);

    await vi.advanceTimersByTimeAsync(1100);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterFirstPoll);
  });

  test('a settled rotated-away run stops polling', async () => {
    // The negative control for both tests above: with neither half reporting
    // `partial`, a 404 record is the end of the story and the panel must go
    // quiet.
    vi.useFakeTimers();
    const fetchMock = stubFetch({
      '/runs/old': { status: 404, body: { detail: "unknown run 'old'" } },
      '/runs/old/data': { status: 200, body: DATA },
      '/runs/old/figure': { status: 200, body: { ...FIGURE, source: 'tiled' } },
    });
    vi.stubGlobal('fetch', fetchMock);
    view = createResultsView({ api: (p) => p, elements: mountElements() });
    view.follow('old');

    await vi.advanceTimersByTimeAsync(0);
    const afterFirstPoll = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetchMock.mock.calls.length).toBe(afterFirstPoll);
  });
});

// ---------------------------------------------------------------------------
// The emergency abort control
// ---------------------------------------------------------------------------

describe('abortControl', () => {
  /** @param {any} overrides */
  function stateWith(overrides = {}) {
    return { ...createInitialQueueState(), status: summary(), ...overrides };
  }

  test('has no disabled state at all — the shape carries no such key', () => {
    // Same invariant as the stop control, and stronger: this is the ONLY
    // surface that halts a plan already moving hardware. A `disabled` key here
    // is the whole class of bug (C1 on the sibling control) reintroduced.
    const states = [
      createInitialQueueState(),
      stateWith({ runningItem: item() }),
      stateWith({ status: { available: false, reason: 'manager_unreachable' } }),
      stateWith({ status: summary({ manager_state: 'executing_queue' }), runningItem: item() }),
    ];
    for (const state of states) {
      expect(Object.keys(abortControl(state))).toEqual(['note', 'running']);
    }
  });

  test('names the consequence while a plan is running', () => {
    const control = abortControl(stateWith({ runningItem: item() }));
    expect(control.running).toBe(true);
    expect(control.note).toContain('Discards the rest of the running plan');
    expect(control.note).toContain('Hardware is left wherever the scan stopped');
  });

  test('an unreadable manager summary says the abort is still sent', () => {
    // The panel's knowledge is one poll stale at best. "Could not read" is
    // exactly when a scan may be under way, so the wording must not imply the
    // control is inert.
    const control = abortControl(
      stateWith({ status: { available: false, reason: 'manager_unreachable' } })
    );
    expect(control.note).toContain('still sent');
  });

  test('no running item still offers the abort, deferring to the bridge', () => {
    const control = abortControl(stateWith({ runningItem: null }));
    expect(control.running).toBe(false);
    expect(control.note).toContain('still sent');
    expect(control.note).toContain('the bridge answers');
  });
});

describe('abort button label and class', () => {
  test('two steps, and the second states what it does', () => {
    expect(abortButtonLabel(false)).toBe(ABORT_LABEL);
    expect(abortButtonLabel(true)).toBe(CONFIRM_ABORT_LABEL);
    expect(CONFIRM_ABORT_LABEL).toContain('hardware stays put');
  });

  test('the caution class is on the armed step only', () => {
    // At rest this is a halt, and painting a halt as the dangerous action is
    // the inversion the sibling stop control already had to fix.
    expect(abortButtonClass(false)).toBe('btn');
    expect(abortButtonClass(true)).toBe('btn confirm');
  });
});

describe('abortSuccessMessage', () => {
  test('leads with the consequence, not a congratulation', () => {
    const message = abortSuccessMessage({ aborted: true, abort_pending: false });
    expect(message).toContain('Hardware is left wherever the scan stopped');
    expect(abortOutcomeTone()).toBe('warn');
  });

  test('relays the manager sentence verbatim when the bridge sent one', () => {
    const message = abortSuccessMessage({ aborted: true, msg: 'RE aborted at point 12' });
    expect(message).toContain('RE aborted at point 12');
  });

  test('a pending abort reads as still unwinding, never as a failure', () => {
    const message = abortSuccessMessage({ aborted: true, abort_pending: true });
    expect(message).toContain('still unwinding');
    expect(message.toLowerCase()).not.toContain('failed');
  });

  test('a body with no msg adds no empty trailer', () => {
    expect(abortSuccessMessage({ aborted: true, msg: '   ' })).not.toContain('Manager:');
    expect(abortSuccessMessage(null)).toContain('Abort sent');
  });
});

// ---------------------------------------------------------------------------
// Bundle integrity
// ---------------------------------------------------------------------------

describe('bundle wiring', () => {
  // Resolved from the repo root rather than from `import.meta.url`: under
  // happy-dom the document's origin is `http://localhost`, so a URL relative
  // to this module is not a file path at all. Vitest always runs from the
  // repo root.
  const BUNDLE = `${cwd()}/src/osprey/interfaces/bluesky_panels/panels/bluesky/`;

  test('every element id the shell, queue view and figure renderer look up exists in index.html', () => {
    // The bundle has no build step and is served as authored, so a renamed id
    // fails only at runtime, in the operator's browser, as a silently dead
    // control. This is the drift-net for that. (plans-view.js gets the same
    // guard in draft-client.test.mjs.)
    //
    // figure-renderer.js contributes nothing today — it is handed its
    // containers and looks nothing up — and is scanned precisely so that stays
    // true: the first `byId` added there has to name a real element.
    const source = [`${BUNDLE}panel.js`, `${BUNDLE}queue-view.js`, `${BUNDLE}figure-renderer.js`]
      .map((path) => readFileSync(path, 'utf-8'))
      .join('\n');
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');

    const referenced = [...source.matchAll(/(?:byId|getElementById)\(\s*'([^']+)'/g)].map(
      (match) => match[1]
    );
    const declared = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));

    expect(referenced.length).toBeGreaterThan(10);
    expect(referenced.filter((id) => !declared.has(id))).toEqual([]);
  });

  test('the manifest names the panel id the registry registers', () => {
    const manifest = JSON.parse(readFileSync(`${BUNDLE}manifest.json`, 'utf-8'));
    expect(manifest.id).toBe('bluesky');
    expect(manifest.label).toBe('BLUESKY');
    expect(manifest.entry).toBe('index.html');
  });

  test('the panel chrome agrees with the manifest', () => {
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
    expect(html).toContain('<title>Bluesky</title>');
    expect(html).toContain('<h1>Bluesky</h1>');
    expect(html).toContain('OSPREY Bluesky Panels');
  });

  test('the two halts render OUTSIDE the three view containers', () => {
    // The whole point of the status strip: a halt must never be a tab switch
    // away. If either button ever migrates inside a view container, it becomes
    // invisible on the other two tabs.
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
    const firstView = html.indexOf('id="view-plans"');
    expect(firstView).toBeGreaterThan(-1);
    for (const id of ['stop-btn', 'abort-btn', 'queue-state-badge', 'queue-banner']) {
      expect(html.indexOf(`id="${id}"`), `${id} must precede the view containers`).toBeLessThan(
        firstView
      );
    }
    // Start is the counter-example — arming a queue stays a deliberate act
    // behind the Queue tab.
    expect(html.indexOf('id="start-btn"')).toBeGreaterThan(firstView);
  });

  test('the in-body view switcher and filter hide only when embedded AND Expert', () => {
    // The single most important correctness point of the merge, checked as
    // resolved style rather than as matching source text. `contributeHeader`
    // is a no-op standalone, and the hub collapses a service tile's bar in
    // Simple mode — in both of those cases the in-body strip is the ONLY way
    // to navigate the panel, so a plain `body.embedded` rule would strand the
    // operator on whichever view happened to be showing. Same rule, same
    // reason, for the plan filter.
    const style = document.createElement('style');
    style.textContent = readFileSync(`${BUNDLE}panel.css`, 'utf-8');
    document.head.appendChild(style);
    const tabs = document.createElement('nav');
    tabs.className = 'panel-tabs';
    const filter = document.createElement('div');
    filter.className = 'sidebar-head';
    const strip = document.createElement('div');
    strip.className = 'status-strip';
    document.body.append(tabs, filter, strip);

    /** @param {{embedded: boolean, simple: boolean}} context */
    function shown({ embedded, simple }) {
      document.body.classList.toggle('embedded', embedded);
      if (simple) document.documentElement.setAttribute('data-ui-mode', 'simple');
      else document.documentElement.removeAttribute('data-ui-mode');
      return {
        tabs: getComputedStyle(tabs).display !== 'none',
        filter: getComputedStyle(filter).display !== 'none',
        strip: getComputedStyle(strip).display !== 'none',
      };
    }

    try {
      expect(shown({ embedded: false, simple: false })).toEqual({
        tabs: true,
        filter: true,
        strip: true,
      });
      expect(shown({ embedded: false, simple: true })).toEqual({
        tabs: true,
        filter: true,
        strip: true,
      });
      // The one case the hub really is drawing these for us.
      expect(shown({ embedded: true, simple: false })).toEqual({
        tabs: false,
        filter: false,
        strip: true,
      });
      // Simple: the tile bar is collapsed to zero height, so the body keeps
      // both — and the safety strip is on screen in every single case.
      expect(shown({ embedded: true, simple: true })).toEqual({
        tabs: true,
        filter: true,
        strip: true,
      });
    } finally {
      style.remove();
      tabs.remove();
      filter.remove();
      strip.remove();
      document.body.classList.remove('embedded');
      document.documentElement.removeAttribute('data-ui-mode');
    }
  });

  test('the abort button never ships disabled either', () => {
    // Same pin as the halt below, on the control that matters most: this is
    // the only surface that stops a plan already moving hardware. panel.js
    // sets NO `disabled` property on #abort-btn, so an attribute here would be
    // permanent — `className` assignment replaces classes, not attributes.
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
    const abortButton = html.match(/<button[^>]*id="abort-btn"[^>]*>/);
    expect(abortButton).not.toBeNull();
    if (!abortButton) throw new Error('unreachable: #abort-btn asserted present');
    expect(abortButton[0]).not.toMatch(/\bdisabled\b/);
    // Resting class only: `confirm` is the armed second step's, and shipping
    // it as the default paints a halt as the consequential action.
    expect(abortButton[0]).not.toMatch(/\bconfirm\b/);
  });

  test('the halt button never ships disabled', () => {
    // Direct pin on the invariant. panel.js sets NO `disabled` property on
    // #stop-btn — by design, since the halt has no disabled state — so an
    // attribute here is not a default, it is permanent: it would survive every
    // render and leave the operator with a dead halt in every queue state.
    // `className` assignment replaces classes, not attributes, so nothing in
    // the render path can undo it.
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
    const stopButton = html.match(/<button[^>]*id="stop-btn"[^>]*>/);
    expect(stopButton).not.toBeNull();
    if (!stopButton) throw new Error('unreachable: #stop-btn asserted present');
    expect(stopButton[0]).not.toMatch(/\bdisabled\b/);
    // Resting class only — `confirm` is the armed withdrawal's, added by
    // stopButtonClass, and shipping it as the default paints the safe halt as
    // the consequential action before any render runs.
    expect(stopButton[0]).not.toMatch(/\bconfirm\b/);
  });

  test('the pre-script empty states claim nothing either', () => {
    // The static markup is what an operator sees for the instant before
    // panel.js runs (and for good, if the module fails to load). It must not
    // assert an empty queue any more than the rendered state does.
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
    expect(html).not.toContain('>Nothing queued.<');
    expect(html).not.toContain('>No completed runs yet.<');
    expect(html).toContain('Queue contents unknown');
  });
});

// ---------------------------------------------------------------------------
// Booting the real bundle
// ---------------------------------------------------------------------------

/**
 * The layer the pure tests cannot see: the actual index.html markup driven by
 * the actual panel.js, wired together.
 *
 * Two findings in a row lived exactly here — a control that ships `disabled`
 * with no code path to clear it, and empty states asserting contents the panel
 * never read. Both are invisible to a module-level test (every pure helper
 * returns the right answer; nothing applies it to the shipped button) and
 * invisible to the id drift-net (the id exists, it is just dead). Booting the
 * bundle is what closes that gap.
 */
describe('booting the shipped bundle', () => {
  const BUNDLE = `${cwd()}/src/osprey/interfaces/bluesky_panels/panels/bluesky/`;

  /** Captures the panel's EventSource so a test can push frames at it. */
  class BootEventSource {
    /** @type {BootEventSource[]} */
    static instances = [];
    /** @param {string} url */
    constructor(url) {
      this.url = url;
      /** @type {((event: {data: string}) => void)|null} */
      this.onmessage = null;
      /** @type {(() => void)|null} */
      this.onerror = null;
      BootEventSource.instances.push(this);
    }
    close() {}
  }

  /** @returns {{stopBtn: any, startBtn: any, abortBtn: any, pushFrame: (f: any) => void}} */
  function boot() {
    // Body only: the <head> carries module scripts for the design-system
    // assets, which have no server here. The body is the markup under test.
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
    const body = html.match(/<body>([\s\S]*)<\/body>/);
    if (!body) throw new Error('could not extract <body> from index.html');
    document.body.innerHTML = body[1].replace(/<script[\s\S]*?<\/script>/g, '');

    const byId = (/** @type {string} */ id) => /** @type {any} */ (document.getElementById(id));
    return {
      stopBtn: byId('stop-btn'),
      startBtn: byId('start-btn'),
      abortBtn: byId('abort-btn'),
      pushFrame(frameObject) {
        // Only the QUEUE stream: the merged bundle also opens the shared
        // plan-draft stream, and a queue frame pushed at that one would be
        // read as a draft frame with a revision gap.
        for (const source of BootEventSource.instances) {
          if (source.url.includes('/queue/events') && source.onmessage) {
            source.onmessage({ data: JSON.stringify(frameObject) });
          }
        }
      },
    };
  }

  beforeEach(() => {
    BootEventSource.instances = [];
    vi.resetModules();
    vi.stubGlobal('EventSource', BootEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, status: 200, json: async () => [] }))
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
  });

  test('the halt is live from first paint, before any frame arrives', async () => {
    const panel = boot();
    // Pre-import: the shipped markup itself must not disable it.
    expect(panel.stopBtn.disabled).toBe(false);
    await import(`${BUNDLE}panel.js`);
    // Post-boot, still before any queue frame — this is exactly the window in
    // which a queue may already be executing.
    expect(panel.stopBtn.disabled).toBe(false);
  });

  test('the halt stays live across every queue state', async () => {
    const panel = boot();
    await import(`${BUNDLE}panel.js`);

    const states = [
      { manager_state: 'idle', queue_stop_pending: false },
      { manager_state: 'executing_queue', queue_stop_pending: false },
      { manager_state: 'executing_queue', queue_stop_pending: true },
      { manager_state: 'paused', queue_stop_pending: false },
    ];
    for (const overrides of states) {
      panel.pushFrame({
        type: 'queue',
        status: summary(overrides),
        items: [item()],
        running_item: null,
      });
      expect(panel.stopBtn.disabled, `stop disabled while ${overrides.manager_state}`).toBe(false);
    }
  });

  test('start IS managed across the same frames — proving renders landed', async () => {
    // The control: without this, "stop is never disabled" could just mean the
    // render never ran. Start must actually flip as the queue changes.
    const panel = boot();
    await import(`${BUNDLE}panel.js`);

    panel.pushFrame({ type: 'queue', status: summary(), items: [], running_item: null });
    expect(panel.startBtn.disabled).toBe(true); // nothing queued

    panel.pushFrame({ type: 'queue', status: summary(), items: [item()], running_item: null });
    expect(panel.startBtn.disabled).toBe(false); // something to start

    panel.pushFrame({
      type: 'queue',
      status: summary({ manager_state: 'executing_queue' }),
      items: [item()],
      running_item: null,
    });
    expect(panel.startBtn.disabled).toBe(true); // already draining
  });

  test('the withdrawal takes two clicks and the plain stop takes one', async () => {
    const panel = boot();
    await import(`${BUNDLE}panel.js`);
    const fetchMock = /** @type {any} */ (globalThis.fetch);

    panel.pushFrame({
      type: 'queue',
      status: summary({ manager_state: 'executing_queue' }),
      items: [],
      running_item: null,
    });
    const before = fetchMock.mock.calls.length;
    panel.stopBtn.click();
    expect(fetchMock.mock.calls.length).toBe(before + 1);
    expect(fetchMock.mock.calls[before][1].body).toContain('"cancel":false');

    panel.pushFrame({
      type: 'queue',
      status: summary({ manager_state: 'executing_queue', queue_stop_pending: true }),
      items: [],
      running_item: null,
    });
    expect(panel.stopBtn.textContent).toContain('Withdraw');
    const armed = fetchMock.mock.calls.length;
    panel.stopBtn.click();
    // First click only arms — no request yet.
    expect(fetchMock.mock.calls.length).toBe(armed);
    expect(panel.stopBtn.className).toContain('confirm');
    panel.stopBtn.click();
    expect(fetchMock.mock.calls.length).toBe(armed + 1);
    expect(fetchMock.mock.calls[armed][1].body).toContain('"cancel":true');
  });

  test('the abort is live from first paint, before any frame arrives', async () => {
    const panel = boot();
    expect(panel.abortBtn.disabled).toBe(false);
    await import(`${BUNDLE}panel.js`);
    // The window before the first frame is exactly when a plan may already be
    // moving hardware and nobody here knows it yet.
    expect(panel.abortBtn.disabled).toBe(false);
  });

  test('the abort stays live across every queue state', async () => {
    const panel = boot();
    await import(`${BUNDLE}panel.js`);

    /** @type {Array<{label: string, status: any, running: any}>} */
    const frames = [
      { label: 'idle', status: summary(), running: null },
      {
        label: 'executing',
        status: summary({ manager_state: 'executing_queue' }),
        running: item(),
      },
      { label: 'paused', status: summary({ manager_state: 'paused' }), running: item() },
      {
        label: 'stop-pending',
        status: summary({ manager_state: 'executing_queue', queue_stop_pending: true }),
        running: item(),
      },
      // The manager could not be read at all — the state a panel is most
      // tempted to grey a control out in, and the one where it must not.
      {
        label: 'unavailable',
        status: { available: false, reason: 'manager_unreachable' },
        running: null,
      },
    ];
    for (const { label, status, running } of frames) {
      panel.pushFrame({ type: 'queue', status, items: [], running_item: running });
      expect(panel.abortBtn.disabled, `abort disabled while ${label}`).toBe(false);
    }
  });

  test('the abort takes two clicks and drives the real wire call', async () => {
    const panel = boot();
    await import(`${BUNDLE}panel.js`);
    const fetchMock = /** @type {any} */ (globalThis.fetch);

    panel.pushFrame({
      type: 'queue',
      status: summary({ manager_state: 'executing_queue' }),
      items: [],
      running_item: item(),
    });

    const before = fetchMock.mock.calls.length;
    panel.abortBtn.click();
    // First click only arms: no request, and the caution treatment appears.
    expect(fetchMock.mock.calls.length).toBe(before);
    expect(panel.abortBtn.className).toContain('confirm');
    expect(panel.abortBtn.textContent).toContain('Confirm');

    panel.abortBtn.click();
    expect(fetchMock.mock.calls.length).toBe(before + 1);
    const [url, init] = fetchMock.mock.calls[before];
    expect(String(url)).toContain('/queue/abort');
    expect(init.method).toBe('POST');
    // And it disarms, so a third stray click cannot fire a second abort.
    expect(panel.abortBtn.className).not.toContain('confirm');
  });

  test('a failed halt paints the banner RED, not the by-design amber', async () => {
    // The end of the wire, in the DOM: an `abort_pause_timeout` means the plan
    // may still be moving hardware, and it must not look like an unarmed
    // start. Driven through the real click path so a regression anywhere
    // between the button and `setBanner` is caught.
    const refusal = {
      detail: {
        code: 'abort_pause_timeout',
        detail: 'NOTHING WAS ABORTED and the plan may still be running.',
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) =>
        String(url).includes('/queue/abort')
          ? { ok: false, status: 503, json: async () => refusal }
          : { ok: true, status: 200, json: async () => [] }
      )
    );

    const panel = boot();
    await import(`${BUNDLE}panel.js`);
    const banner = /** @type {any} */ (document.getElementById('queue-banner'));

    panel.pushFrame({
      type: 'queue',
      status: summary({ manager_state: 'executing_queue' }),
      items: [],
      running_item: item(),
    });
    panel.abortBtn.click();
    panel.abortBtn.click();

    await vi.waitFor(() => expect(banner.hidden).toBe(false));
    expect(banner.className).toContain('banner-err');
    expect(banner.className).not.toContain('banner-warn');
    expect(banner.textContent).toContain('NOTHING WAS ABORTED');
  });

  test('a by-design refusal on an ORDINARY write still reads as a warning', async () => {
    // Negative control for the test above: if every refusal painted red, the
    // red on a failed halt would carry no information at all.
    const refusal = {
      detail: {
        code: 'launch_token_required',
        detail: 'Starting the queue requires an arming token; none is configured.',
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) =>
        String(url).includes('/queue/start')
          ? { ok: false, status: 503, json: async () => refusal }
          : { ok: true, status: 200, json: async () => [] }
      )
    );

    const panel = boot();
    await import(`${BUNDLE}panel.js`);
    const banner = /** @type {any} */ (document.getElementById('queue-banner'));

    panel.pushFrame({ type: 'queue', status: summary(), items: [item()], running_item: null });
    panel.startBtn.click();

    await vi.waitFor(() => expect(banner.hidden).toBe(false));
    expect(banner.className).toContain('banner-warn');
    expect(banner.className).not.toContain('banner-err');
  });

  test('an armed abort confirm disarms when the running item goes away', async () => {
    // The confirm refers to a specific run. Leaving it armed would let the
    // next click abort whatever the queue moved on to.
    const panel = boot();
    await import(`${BUNDLE}panel.js`);
    const fetchMock = /** @type {any} */ (globalThis.fetch);

    panel.pushFrame({
      type: 'queue',
      status: summary({ manager_state: 'executing_queue' }),
      items: [],
      running_item: item(),
    });
    panel.abortBtn.click();
    expect(panel.abortBtn.className).toContain('confirm');

    panel.pushFrame({ type: 'queue', status: summary(), items: [], running_item: null });
    expect(panel.abortBtn.className).not.toContain('confirm');

    const before = fetchMock.mock.calls.length;
    panel.abortBtn.click();
    expect(fetchMock.mock.calls.length).toBe(before); // re-arms instead of firing
  });
});

// ---------------------------------------------------------------------------
// The merged panel's shell: three views, one status strip
// ---------------------------------------------------------------------------

/**
 * PLAN and BLUESKY are one panel now, and the shell that holds them together
 * is the part with no pure module behind it: which view is showing, what the
 * tile bar is told about them, and whether picking a run takes the operator to
 * it. All of that only exists once the real markup and the real panel.js are
 * wired together, so these boot the bundle exactly as the browser does.
 */
describe('the merged panel shell', () => {
  const BUNDLE = `${cwd()}/src/osprey/interfaces/bluesky_panels/panels/bluesky/`;

  /** Captures the panel's EventSources so a test can push queue frames. */
  class ShellEventSource {
    /** @type {ShellEventSource[]} */
    static instances = [];
    /** @param {string} url */
    constructor(url) {
      this.url = url;
      /** @type {((event: {data: string}) => void)|null} */
      this.onmessage = null;
      /** @type {(() => void)|null} */
      this.onerror = null;
      ShellEventSource.instances.push(this);
    }
    close() {}
  }

  /**
   * Every contribution the panel has posted at the (faked) hub. Untyped on
   * purpose: these assert the WIRE shape, so the test must be able to look at
   * whatever was actually sent.
   *
   * @type {any[]}
   */
  let contributions = [];

  /**
   * Mount the shipped markup and pretend to be inside the web terminal.
   *
   * `embedded` decides whether `contributeHeader` does anything at all — it is
   * a strict no-op outside a frame — so a test that wants to see the
   * contribution has to claim both the body class and a distinct parent.
   *
   * @param {{embedded?: boolean, mode?: 'expert'|'simple'}} [options]
   */
  function mount({ embedded = true, mode = 'expert' } = {}) {
    const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
    const body = html.match(/<body>([\s\S]*)<\/body>/);
    if (!body) throw new Error('could not extract <body> from index.html');
    document.body.innerHTML = body[1].replace(/<script[\s\S]*?<\/script>/g, '');
    document.documentElement.setAttribute('data-ui-mode', mode);
    if (embedded) document.body.classList.add('embedded');
    Object.defineProperty(window, 'parent', {
      configurable: true,
      value: {
        /** @param {any} message */
        postMessage(message) {
          if (message && message.type === 'osprey-header-contribution') {
            contributions.push(message.items);
          }
        },
      },
    });
  }

  /**
   * The most recent contribution's items.
   *
   * @returns {any[]}
   */
  function lastContribution() {
    return contributions[contributions.length - 1];
  }

  /** @param {string} id */
  const byId = (id) => /** @type {any} */ (document.getElementById(id));

  /**
   * The nav item of the most recent contribution, with its entries typed as
   * the loose wire objects they are.
   *
   * @returns {{id: string, items: Array<{id: string, label: string, active?: boolean}>}}
   */
  function navContribution() {
    return lastContribution().find((entry) => entry.kind === 'nav');
  }

  /**
   * The Results nav entry of the most recent contribution. Asserted present
   * rather than optional-chained: an absent Results entry is itself the
   * regression these tests exist to catch.
   */
  function resultsNavEntry() {
    const entry = navContribution().items.find((candidate) => candidate.id === 'results');
    if (!entry) throw new Error('no Results entry in the contribution');
    return entry;
  }

  /** @param {any} frameObject */
  function pushQueueFrame(frameObject) {
    for (const source of ShellEventSource.instances) {
      if (source.url.includes('/queue/events') && source.onmessage) {
        source.onmessage({ data: JSON.stringify(frameObject) });
      }
    }
  }

  /**
   * Deliver a hub header action exactly as the postMessage bridge would.
   *
   * @param {string} id
   * @param {string} [value]
   */
  function sendHeaderAction(id, value) {
    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'osprey-header-action', id, value },
        origin: window.location.origin,
      })
    );
  }

  beforeEach(() => {
    ShellEventSource.instances = [];
    contributions = [];
    vi.resetModules();
    vi.stubGlobal('EventSource', ShellEventSource);
    // A settled world by default: every route answers in its documented shape,
    // and any run the panel follows is terminal, so the results poll runs once
    // and stops instead of outliving the test. Tests that need a live run
    // override this.
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) => {
        const path = String(url);
        if (path.endsWith('/data')) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              columns: [],
              rows: [],
              row_count: 0,
              truncated: false,
              partial: false,
            }),
          };
        }
        if (/\/runs\/[^/]+$/.test(path)) {
          return { ok: true, status: 200, json: async () => ({ id: 'run', status: 'completed' }) };
        }
        return { ok: true, status: 200, json: async () => [] };
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
    document.body.classList.remove('embedded');
    document.documentElement.removeAttribute('data-ui-mode');
  });

  /** Which of the three view containers is on screen. */
  function visibleViews() {
    return ['plans', 'queue', 'results'].filter((id) => !byId(`view-${id}`).hidden);
  }

  test('exactly one view shows at a time, and Plans is the one on boot', async () => {
    mount();
    await import(`${BUNDLE}panel.js`);
    expect(visibleViews()).toEqual(['plans']);
    expect(byId('view-tab-plans').getAttribute('aria-selected')).toBe('true');
  });

  test('the in-body strip switches views, and the tabs track it', async () => {
    mount();
    await import(`${BUNDLE}panel.js`);

    byId('view-tab-queue').click();
    expect(visibleViews()).toEqual(['queue']);
    expect(byId('view-tab-queue').getAttribute('aria-selected')).toBe('true');
    expect(byId('view-tab-plans').getAttribute('aria-selected')).toBe('false');

    byId('view-tab-results').click();
    expect(visibleViews()).toEqual(['results']);
  });

  test('the hub nav action drives the same switch as the in-body strip', async () => {
    mount();
    await import(`${BUNDLE}panel.js`);

    sendHeaderAction('view', 'results');
    expect(visibleViews()).toEqual(['results']);
    expect(byId('view-tab-results').classList.contains('active')).toBe(true);

    // An id the panel does not know must be ignored rather than blanking every
    // view — the value comes from another frame, not from this module.
    sendHeaderAction('view', 'nonsense');
    expect(visibleViews()).toEqual(['results']);
  });

  test('picking a run in the Queue view opens the Results view on it', async () => {
    mount();
    await import(`${BUNDLE}panel.js`);
    byId('view-tab-queue').click();

    pushQueueFrame({
      type: 'queue',
      status: summary({ items_in_queue: 1 }),
      items: [item()],
      running_item: null,
    });
    const row = /** @type {any} */ (byId('queue-items').querySelector('button.queue-label'));
    expect(row).not.toBeNull();
    row.click();

    expect(visibleViews()).toEqual(['results']);
    // And the selection is reflected back in the queue list it came from.
    expect(byId('queue-items').querySelector('.queue-row.selected')).not.toBeNull();
  });

  test('a ?run_id= deep link still selects the run and opens Results', async () => {
    mount();
    window.history.replaceState({}, '', '/bluesky/?run_id=run-deep');
    try {
      await import(`${BUNDLE}panel.js`);
      expect(visibleViews()).toEqual(['results']);
      const calls = /** @type {any} */ (globalThis.fetch).mock.calls;
      /** @type {string[]} */
      const asked = calls.map((/** @type {any[]} */ call) => String(call[0]));
      expect(asked.some((url) => url.includes('/runs/run-deep'))).toBe(true);
    } finally {
      window.history.replaceState({}, '', '/');
    }
  });

  test('the three views are contributed as one nav item, active flag and all', async () => {
    mount();
    await import(`${BUNDLE}panel.js`);

    const nav = navContribution();
    expect(nav.id).toBe('view');
    expect(nav.items.map((entry) => entry.id)).toEqual(['plans', 'queue', 'results']);
    expect(nav.items.map((entry) => entry.label)).toEqual(['Plans', 'Queue', 'Results']);
    expect(nav.items.filter((entry) => entry.active).map((entry) => entry.id)).toEqual(['plans']);

    byId('view-tab-queue').click();
    const after = navContribution();
    expect(after.items.filter((entry) => entry.active).map((entry) => entry.id)).toEqual(['queue']);
  });

  test('the plan filter is contributed only while Plans is showing', async () => {
    mount();
    await import(`${BUNDLE}panel.js`);
    expect(lastContribution().some((entry) => entry.kind === 'search')).toBe(true);

    byId('view-tab-queue').click();
    expect(lastContribution().some((entry) => entry.kind === 'search')).toBe(false);

    byId('view-tab-plans').click();
    const search = lastContribution().find((entry) => entry.kind === 'search');
    expect(search.id).toBe('plan-filter');
    expect(search.placeholder).toContain('Filter plans');
  });

  test('Simple mode contributes no search — the hub collapses the bar there', async () => {
    // Simple deliberately ENLARGES a search box; the tile bar it would live in
    // is zero-height. The in-body one is the one on screen.
    mount({ mode: 'simple' });
    await import(`${BUNDLE}panel.js`);
    expect(lastContribution().some((entry) => entry.kind === 'search')).toBe(false);
    // The nav still goes up — a collapsed bar renders nothing, it does not
    // reject anything.
    expect(lastContribution().some((entry) => entry.kind === 'nav')).toBe(true);
  });

  test('the hub search action filters the same plans the in-body box does', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) =>
        String(url).endsWith('/plans')
          ? {
              ok: true,
              status: 200,
              json: async () => [
                { name: 'grid_scan', provenance: 'shipped', description: '' },
                { name: 'orm', provenance: 'shipped', description: '' },
              ],
            }
          : { ok: true, status: 200, json: async () => ({}) }
      )
    );
    mount();
    await import(`${BUNDLE}panel.js`);
    await vi.waitFor(() => expect(byId('plan-tree').textContent).toContain('grid_scan'));

    sendHeaderAction('plan-filter', 'orm');
    expect(byId('plan-tree').textContent).not.toContain('grid_scan');
    expect(byId('plan-tree').textContent).toContain('orm');
    // The in-body box is kept in step, so flipping to Simple — where the
    // contributed copy disappears — does not lose the filter.
    expect(byId('plan-search').value).toBe('orm');
  });

  test('the Results tab carries a marker while data arrives for an unwatched run', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) => {
        const path = String(url);
        if (path.endsWith('/runs/run-1')) {
          return { ok: true, status: 200, json: async () => ({ id: 'run-1', status: 'running' }) };
        }
        if (path.endsWith('/runs/run-1/data')) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              columns: ['x'],
              rows: [[1]],
              row_count: 1,
              truncated: false,
              partial: true,
            }),
          };
        }
        return { ok: true, status: 200, json: async () => [] };
      })
    );
    mount();
    await import(`${BUNDLE}panel.js`);

    byId('view-tab-queue').click();
    pushQueueFrame({
      type: 'queue',
      status: summary({ items_in_queue: 1 }),
      items: [item()],
      running_item: null,
    });
    /** @type {any} */ (byId('queue-items').querySelector('button.queue-label')).click();
    // Selecting opened Results, so no marker: the operator IS watching. Wait
    // for the first poll to land, which is what starts the polling.
    await vi.waitFor(() => expect(byId('table-card').hidden).toBe(false));
    expect(resultsNavEntry().label).toBe('Results');

    // Walk away while it is still filling in.
    byId('view-tab-queue').click();
    expect(resultsNavEntry().label).toBe('Results •');

    // Coming back clears it.
    byId('view-tab-results').click();
    expect(resultsNavEntry().label).toBe('Results');
  });

  test('standalone, the panel contributes nothing and stays navigable', async () => {
    // The single most important correctness point of the merge: with no hub
    // there is no contributed strip, so the in-body one must still work.
    mount({ embedded: false });
    await import(`${BUNDLE}panel.js`);
    expect(contributions).toEqual([]);
    byId('view-tab-results').click();
    expect(visibleViews()).toEqual(['results']);
  });

  test('the two halts are live on every view, not just the Queue one', async () => {
    mount();
    await import(`${BUNDLE}panel.js`);
    for (const view of ['plans', 'queue', 'results']) {
      byId(`view-tab-${view}`).click();
      expect(byId('stop-btn').disabled, `stop dead on ${view}`).toBe(false);
      expect(byId('abort-btn').disabled, `abort dead on ${view}`).toBe(false);
      // Not merely enabled — actually on screen, i.e. outside the hidden views.
      expect(byId('stop-btn').closest('[hidden]'), `stop hidden on ${view}`).toBeNull();
      expect(byId('abort-btn').closest('[hidden]'), `abort hidden on ${view}`).toBeNull();
    }
  });
});
