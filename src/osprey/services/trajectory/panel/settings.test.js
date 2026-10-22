// @ts-check
import { describe, expect, it, vi } from 'vitest';

import { createSettings, readValue, settingsFor } from './settings.js';

const READBACKS = {
  read_at: 1,
  setup: null,
  settings: [
    { id: 'pilot_tone', scope: 'global', value: 'Off', values: {} },
    { id: 'bpm_mode', scope: 'device', value: null, values: { B1: 'low', B2: 'low', B3: null } },
    { id: 'attenuation', scope: 'device', value: null, values: { B1: 12, B2: 6 } },
    { id: 'turns_tracked', scope: 'global', value: 260, values: {} },
  ],
};

/** @param {any} value */
const ok = (value) => ({ kind: 'ok', status: 200, body: value });
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * A fake area context with a setup route and a snapshot feed.
 * @param {{setup?: any, readbacks?: any[]}} [answers]
 */
function fakeCtx(answers = {}) {
  /** @type {Set<(s: any) => void>} */
  const fns = new Set();
  let snapshot = /** @type {any} */ (null);
  const readbacks = [...(answers.readbacks ?? [READBACKS])];
  const api = {
    setupReadbacks: vi.fn(async () => ok(readbacks.length > 1 ? readbacks.shift() : readbacks[0])),
    setup: vi.fn(async () => answers.setup ?? ok({ moved: [], unsettled: [], setup: null })),
  };
  const ctx = /** @type {any} */ ({
    api,
    snapshot: () => snapshot,
    onSnapshot(/** @type {(s: any) => void} */ fn) {
      fns.add(fn);
      return () => void fns.delete(fn);
    },
  });
  return {
    ctx,
    api,
    /** @param {any} next */
    push(next) {
      snapshot = next;
      for (const fn of [...fns]) fn(next);
    },
  };
}

describe('readValue', () => {
  it('answers a global reading, and the agreed reading of a device setting', () => {
    expect(readValue(READBACKS.settings[0])).toBe('Off');
    expect(readValue(READBACKS.settings[1])).toBe('low');
  });

  it('answers null when the BPMs disagree or none was read', () => {
    expect(readValue(READBACKS.settings[2])).toBeNull();
    expect(readValue({ id: 'x', scope: 'device', values: { B1: null } })).toBeNull();
    expect(readValue({ id: 'x', scope: 'global', value: null })).toBeNull();
  });
});

describe('createSettings', () => {
  it('lays pending values over the readbacks and drops them on revert', () => {
    const model = createSettings();
    const changes = vi.fn();
    model.on(changes);
    model.seed(READBACKS);
    expect(model.values()).toEqual({ pilot_tone: 'Off', bpm_mode: 'low', turns_tracked: 260 });
    model.set('pilot_tone', 'On');
    model.set('attenuation', 9);
    expect(model.pending()).toEqual({ pilot_tone: 'On', attenuation: 9 });
    expect(model.values()).toMatchObject({ pilot_tone: 'On', attenuation: 9 });
    model.revert();
    expect(model.pending()).toEqual({});
    expect(model.values().pilot_tone).toBe('Off');
    expect(changes).toHaveBeenCalledTimes(4);
  });

  it('keeps pending values across a re-seed', () => {
    const model = createSettings();
    model.seed(READBACKS);
    model.set('turns_tracked', 512);
    model.seed({ ...READBACKS, settings: READBACKS.settings.slice(0, 1) });
    expect(model.values()).toEqual({ pilot_tone: 'Off', turns_tracked: 512 });
  });
});

describe('settingsFor', () => {
  it('answers one model per context, seeded from the setup readbacks', async () => {
    const { ctx, api } = fakeCtx();
    const sync = settingsFor(ctx);
    expect(settingsFor(ctx)).toBe(sync);
    expect(settingsFor(fakeCtx().ctx)).not.toBe(sync);
    await flush();
    expect(api.setupReadbacks).toHaveBeenCalledTimes(1);
    expect(sync.model.readbacks()).toBe(READBACKS);
  });

  it('re-reads the readbacks on every settings_sha change, and only then', async () => {
    const { ctx, api, push } = fakeCtx();
    settingsFor(ctx);
    await flush();
    push({ setup: null });
    push({ setup: { settings_sha: 'a' } });
    push({ setup: { settings_sha: 'a', void: true } });
    push({ setup: { settings_sha: 'b' } });
    push({ setup: null });
    await flush();
    expect(api.setupReadbacks).toHaveBeenCalledTimes(4);
  });

  it('commits through the setup route, drops what it sent, and re-reads', async () => {
    const after = { ...READBACKS, settings: [{ id: 'pilot_tone', scope: 'global', value: 'On', values: {} }] };
    const { ctx, api } = fakeCtx({ readbacks: [READBACKS, after] });
    const sync = settingsFor(ctx);
    await flush();
    sync.model.set('pilot_tone', 'On');
    sync.model.set('attenuation', 3);
    const result = await sync.commit({ settings: { pilot_tone: 'On' }, excluded_bpms: ['B2'] });
    expect(result.kind).toBe('ok');
    expect(api.setup).toHaveBeenCalledWith({ settings: { pilot_tone: 'On' }, excluded_bpms: ['B2'] });
    expect(sync.model.pending()).toEqual({ attenuation: 3 });
    expect(api.setupReadbacks).toHaveBeenCalledTimes(2);
    expect(sync.model.readbacks()).toBe(after);
  });

  it('keeps the pending values when the setup is refused, and still re-reads', async () => {
    const refused = { kind: 'refused', status: 409, code: 'apply_in_progress', detail: 'busy', moved: null, extras: {} };
    const { ctx, api } = fakeCtx({ setup: refused });
    const sync = settingsFor(ctx);
    await flush();
    sync.model.set('pilot_tone', 'On');
    expect((await sync.commit({ settings: { pilot_tone: 'On' } })).kind).toBe('refused');
    expect(sync.model.pending()).toEqual({ pilot_tone: 'On' });
    expect(api.setupReadbacks).toHaveBeenCalledTimes(2);
  });
});
