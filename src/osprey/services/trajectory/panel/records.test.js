// @ts-check
import { describe, expect, it, vi } from 'vitest';

import {
  RAW_EVICTED,
  RECORDS_KIND,
  addRecord,
  createRecords,
  fillRecords,
  floatsToBase64,
  mount,
  recordsView,
} from './records.js';

/**
 * @param {string} id
 * @param {number} timestamp
 * @param {Record<string, any>} [extra]
 */
const shot = (id, timestamp, extra = {}) => ({ shot_id: id, timestamp, parent_id: null, x: {}, ...extra });

/** @param {ArrayLike<number>} samples */
const raw = (samples) => ({ header: { shape: [1, samples.length], trigger_sample: 0, units: {} },
  samples: Float32Array.from(samples) });

/** @param {string} text */
function decodeFloats(text) {
  const bytes = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  return Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
}

/**
 * A stubbed api: `listed` is the shots list, `raws` the raw route's answers.
 * @param {{listed?: any[], raws?: Record<string, any>}} [stub]
 */
function stubApi({ listed = [], raws = {} } = {}) {
  return {
    shots: vi.fn(async () => ({ kind: 'ok', status: 200, body: { model: 'm', shots: listed } })),
    shotRaw: vi.fn(async (/** @type {string} */ id) => raws[id]
      ?? { kind: 'refused', status: 404, code: 'unknown_shot', detail: id, moved: null, extras: {} }),
  };
}

/** @param {Partial<Parameters<typeof createRecords>[0]>} [options] */
async function recording(options = {}) {
  const download = vi.fn();
  const notify = vi.fn();
  const records = createRecords({
    api: /** @type {any} */ (stubApi()), retention: () => 3, download, notify,
    now: () => '2026-10-19T12:00:00.000Z', ...options,
  });
  await records.setRecording(true);
  return { records, download, notify };
}

describe('addRecord', () => {
  it('keeps the newest shots first up to the cap and names the ones pushed out', () => {
    let kept = { records: /** @type {any[]} */ ([]), evicted: /** @type {string[]} */ ([]) };
    const evicted = [];
    for (let i = 1; i <= 5; i += 1) {
      kept = addRecord(kept.records, shot(`s-${i}`, i), 3);
      evicted.push(...kept.evicted);
    }
    expect(kept.records.map((r) => r.id)).toEqual(['s-5', 's-4', 's-3']);
    expect(evicted).toEqual(['s-1', 's-2']);
  });

  it('never records a derived shot', () => {
    const kept = addRecord([], shot('s-2', 2, { parent_id: 's-1' }), 3);
    expect(kept.records).toEqual([]);
    expect(kept.evicted).toEqual([]);
  });

  it('holds each shot once', () => {
    const once = addRecord([], shot('s-1', 1), 3);
    expect(addRecord(once.records, shot('s-1', 1), 3).records).toHaveLength(1);
  });

  it('keeps every shot when no retention is published', () => {
    let records = /** @type {any[]} */ ([]);
    for (let i = 0; i < 10; i += 1) records = addRecord(records, shot(`s-${i}`, i), null).records;
    expect(records).toHaveLength(10);
  });
});

describe('fillRecords', () => {
  it('fills in the missed shots read after the floor, newest first, never a derived one', () => {
    const held = [{ id: 's-4', shot: shot('s-4', 4) }];
    const listed = [shot('s-5', 5), shot('s-4', 4), shot('s-3', 3, { parent_id: 's-2' }),
      shot('s-2', 2), shot('s-1', 1)];
    const kept = fillRecords(held, listed, 1, 10);
    expect(kept.records.map((r) => r.id)).toEqual(['s-5', 's-4', 's-2']);
  });

  it('caps the filled list at the retention', () => {
    const held = [{ id: 's-1', shot: shot('s-1', 1) }];
    const listed = [shot('s-4', 4), shot('s-3', 3), shot('s-2', 2), shot('s-1', 1)];
    const kept = fillRecords(held, listed, 0, 2);
    expect(kept.records.map((r) => r.id)).toEqual(['s-4', 's-3']);
    expect(kept.evicted).toEqual(['s-1']);
  });
});

describe('floatsToBase64', () => {
  it('writes little-endian float32 that decodes back', () => {
    expect(decodeFloats(floatsToBase64([1.5, -2, 0.25]))).toEqual([1.5, -2, 0.25]);
  });
});

describe('createRecords', () => {
  it('records only while Record data is ticked', async () => {
    const records = createRecords({ api: /** @type {any} */ (stubApi()), retention: () => 3, download: vi.fn() });
    records.add(shot('s-1', 1));
    expect(records.records()).toEqual([]);
    await records.setRecording(true);
    records.add(shot('s-2', 2));
    await records.setRecording(false);
    records.add(shot('s-3', 3));
    expect(records.records().map((r) => r.id)).toEqual(['s-2']);
  });

  it('honours the retention cap and notes each record that drops out', async () => {
    const { records, notify } = await recording();
    for (let i = 1; i <= 5; i += 1) records.add(shot(`s-${i}`, i));
    expect(records.records().map((r) => r.id)).toEqual(['s-5', 's-4', 's-3']);
    const dropped = notify.mock.calls.map(([notice]) => notice.title);
    expect(dropped).toEqual(['record s-1 dropped', 'record s-2 dropped']);
  });

  it('skips a derived shot', async () => {
    const { records } = await recording();
    records.add(shot('s-1', 1));
    records.add(shot('s-2', 2, { parent_id: 's-1' }));
    expect(records.records().map((r) => r.id)).toEqual(['s-1']);
  });

  it('fills in the shots a continuous run missed, never the ones stored before recording', async () => {
    const api = stubApi({ listed: [shot('s-old', 10)] });
    const records = createRecords({ api: /** @type {any} */ (api), retention: () => 10, download: vi.fn() });
    await records.setRecording(true);
    records.add(shot('s-11', 11));
    api.shots.mockResolvedValueOnce({ kind: 'ok', status: 200, body: { model: 'm', shots: [
      shot('s-14', 14), shot('s-13', 13, { parent_id: 's-12' }), shot('s-12', 12), shot('s-11', 11), shot('s-old', 10),
    ] } });
    await records.fill();
    expect(records.records().map((r) => r.id)).toEqual(['s-14', 's-12', 's-11']);
  });

  it('saves every record with all its turns and lists an evicted id instead of throwing', async () => {
    const api = stubApi({ raws: {
      's-2': { kind: 'ok', status: 200, body: raw([1, 2, 3, 4, 5, 6]) },
      's-1': { kind: 'refused', status: 410, code: RAW_EVICTED, detail: 'gone', moved: null, extras: {} },
    } });
    const { records, download, notify } = await recording({ api: /** @type {any} */ (api),
      model: () => 'sr' });
    records.add(shot('s-1', 1));
    records.add(shot('s-2', 2));

    const outcome = await records.save();

    expect(outcome.saved).toBe(true);
    expect(outcome.evicted).toEqual(['s-1']);
    expect(api.shotRaw.mock.calls.map(([id]) => id).sort()).toEqual(['s-1', 's-2']);
    expect(download).toHaveBeenCalledTimes(1);
    const [name, text] = download.mock.calls[0];
    expect(name).toMatch(/^trajectory-records-.*\.json$/);
    const file = JSON.parse(text);
    expect(file.kind).toBe(RECORDS_KIND);
    expect(file.model).toBe('sr');
    expect(file.evicted).toEqual(['s-1']);
    expect(file.records).toHaveLength(1);
    expect(file.records[0].id).toBe('s-2');
    expect(file.records[0].shot.shot_id).toBe('s-2');
    expect(file.records[0].raw.header.shape).toEqual([1, 6]);
    expect(decodeFloats(file.records[0].raw.samples)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(notify.mock.calls.some(([n]) => n.detail === 's-1')).toBe(true);
    expect(records.saving()).toBe(false);
  });

  it('writes no file while it holds no records', async () => {
    const { records, download } = await recording();
    expect((await records.save()).saved).toBe(false);
    expect(download).not.toHaveBeenCalled();
  });

  it('flush empties the list and the next fill brings nothing back', async () => {
    const api = stubApi();
    const { records } = await recording({ api: /** @type {any} */ (api) });
    records.add(shot('s-1', 1));
    records.add(shot('s-2', 2));
    records.flush();
    expect(records.records()).toEqual([]);
    api.shots.mockResolvedValueOnce({ kind: 'ok', status: 200, body: { model: 'm',
      shots: [shot('s-2', 2), shot('s-1', 1)] } });
    await records.fill();
    expect(records.records()).toEqual([]);
  });
});

describe('recordsView', () => {
  const on = () => ({ toggle: vi.fn(), save: vi.fn(), flush: vi.fn() });

  it('names its controls and disables Save and Flush with no records', () => {
    const node = recordsView({ recording: false, saving: false, records: [], retention: 3 }, on());
    expect(node.querySelector('[data-control="record-toggle"]')).not.toBeNull();
    expect(node.querySelector('[data-control="records-save"]')?.getAttribute('data-reason')).toBe('no records');
    expect(node.querySelector('[data-control="records-flush"]')?.hasAttribute('disabled')).toBe(true);
  });

  it('hands the toggle state to its handler', () => {
    const handlers = on();
    const node = recordsView({ recording: false, saving: false, records: [], retention: 3 }, handlers);
    const box = /** @type {HTMLInputElement} */ (node.querySelector('[data-control="record-toggle"]'));
    box.checked = true;
    box.dispatchEvent(new Event('change'));
    expect(handlers.toggle).toHaveBeenCalledWith(true);
  });
});

describe('mount', () => {
  const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

  function fakeCtx(api = stubApi()) {
    /** @type {Map<string, (payload: any) => void>} */
    const handlers = new Map();
    /** @type {((snapshot: any) => void)[]} */
    const snapshots = [];
    let snapshot = { retention: 2, model: 'sr', continuous: null };
    const ctx = {
      api,
      snapshot: () => snapshot,
      onSnapshot: (/** @type {any} */ fn) => { snapshots.push(fn); return () => {}; },
      onEvent: (/** @type {string} */ type, /** @type {any} */ fn) => { handlers.set(type, fn); return () => {}; },
      notify: vi.fn(),
    };
    return {
      ctx,
      emit: (/** @type {string} */ type, /** @type {any} */ payload) => handlers.get(type)?.(payload),
      push: (/** @type {any} */ next) => { snapshot = next; for (const fn of snapshots) fn(next); },
    };
  }

  it('records shots from the stream into the area and flush empties it', async () => {
    const host = document.createElement('div');
    const { ctx, emit } = fakeCtx();
    mount(host, /** @type {any} */ (ctx));
    const box = /** @type {HTMLInputElement} */ (host.querySelector('[data-control="record-toggle"]'));
    box.checked = true;
    box.dispatchEvent(new Event('change'));
    await flush();
    emit('shot', { shot: shot('s-1', 1) });
    emit('shot', { shot: shot('s-2', 2, { parent_id: 's-1' }) });
    emit('shot', { shot: shot('s-3', 3) });
    emit('shot', { shot: shot('s-4', 4) });
    expect([...host.querySelectorAll('[data-record]')].map((n) => n.getAttribute('data-record')))
      .toEqual(['s-4', 's-3']);
    expect(ctx.notify).toHaveBeenCalledWith(expect.objectContaining({ title: 'record s-1 dropped' }));
    /** @type {HTMLButtonElement} */ (host.querySelector('[data-control="records-flush"]')).click();
    expect(host.querySelectorAll('[data-record]')).toHaveLength(0);
  });

  it('fills in from the shots list when a continuous run ends', async () => {
    const api = stubApi();
    const host = document.createElement('div');
    const { ctx, emit, push } = fakeCtx(api);
    mount(host, /** @type {any} */ (ctx));
    const box = /** @type {HTMLInputElement} */ (host.querySelector('[data-control="record-toggle"]'));
    box.checked = true;
    box.dispatchEvent(new Event('change'));
    await flush();
    push({ retention: 5, model: 'sr', continuous: { shots: 0, lease_expires_in_s: 10 } });
    emit('shot', { shot: shot('s-1', 1) });
    api.shots.mockResolvedValue({ kind: 'ok', status: 200, body: { model: 'sr',
      shots: [shot('s-3', 3), shot('s-2', 2), shot('s-1', 1)] } });
    emit('continuous_ended', { end: 'stopped', shots: 3 });
    await flush();
    expect([...host.querySelectorAll('[data-record]')].map((n) => n.getAttribute('data-record')))
      .toEqual(['s-3', 's-2', 's-1']);
  });
});
