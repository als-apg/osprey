// @ts-check
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createApi } from './api.js';
import {
  defaultName,
  fileName,
  importBody,
  mount,
  referenceFile,
  saveFileReason,
  selectCurrentReason,
} from './references.js';
import { createView } from './views.js';

/** A fresh TbT shot, as a `shot` event carries it; *over* replaces fields. */
function shot(over = {}) {
  return { shot_id: 's-2', fresh: true, readout: 'tbt', parent_id: null, ...over };
}

/** The readings route's answer for `golden`: exactly the import body. */
const GOLDEN = Object.freeze({
  name: 'golden',
  readings: { BPM1: { x: 0.125, y: -0.5 }, BPM2: { x: null, y: 1e-6 } },
  readout: 'tbt',
  select: false,
  source_target: 'va',
  source_model: 'SR',
});

/**
 * A fetch stub over a route table; each entry answers `[status, body]`.
 * @param {Record<string, (init: RequestInit) => [number, unknown]>} routes
 */
function fakeFetch(routes) {
  /** @type {{ key: string, body: any }[]} */
  const calls = [];
  const fetchImpl = vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init) => {
    const key = `${init.method} ${url}`;
    calls.push({ key, body: init.body === undefined ? undefined : JSON.parse(String(init.body)) });
    const route = routes[key];
    const [status, body] = route ? route(init) : [404, { detail: { code: 'not_found', detail: key } }];
    return /** @type {any} */ ({ ok: status < 400, status, json: async () => body });
  });
  return { fetchImpl, calls };
}

/**
 * Mount the area over a fake context.
 * @param {{ routes?: Record<string, (init: RequestInit) => [number, unknown]>,
 *   snapshot?: any, holding?: boolean }} [options]
 */
function mounted({ routes = {}, snapshot = { last_shot: null, reference: null }, holding = true } = {}) {
  const { fetchImpl, calls } = fakeFetch(routes);
  /** @type {Map<string, Set<(payload: any) => void>>} */
  const events = new Map();
  /** @type {Set<(snapshot: any) => void>} */
  const snapshots = new Set();
  let current = snapshot;
  const notices = /** @type {any[]} */ ([]);
  const host = document.createElement('div');
  document.body.replaceChildren(host);
  const ctx = /** @type {any} */ ({
    api: createApi({ getClientId: () => 'tab-1', fetchImpl: /** @type {any} */ (fetchImpl) }),
    snapshot: () => current,
    onSnapshot: (/** @type {any} */ fn) => {
      snapshots.add(fn);
      return () => snapshots.delete(fn);
    },
    onEvent: (/** @type {string} */ type, /** @type {any} */ fn) => {
      const set = events.get(type) ?? new Set();
      events.set(type, set);
      set.add(fn);
      return () => set.delete(fn);
    },
    adapter: {},
    view: createView(),
    notify: (/** @type {any} */ notice) => notice && notices.push(notice),
    holding: () => holding,
    area: () => null,
  });
  mount(host, ctx);
  return {
    host,
    calls,
    notices,
    /** @param {any} next */
    pushSnapshot(next) {
      current = next;
      for (const fn of snapshots) fn(next);
    },
    /** @param {any} payload */
    pushShot(payload) {
      for (const fn of events.get('shot') ?? []) fn({ type: 'shot', shot: payload });
    },
    /** @param {string} name */
    control: (name) => /** @type {HTMLButtonElement} */ (host.querySelector(`[data-control="${name}"]`)),
    refusal: () => host.querySelector('[data-role="reference-refusal"]'),
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

afterEach(() => {
  vi.restoreAllMocks();
});

describe('defaultName', () => {
  it('names a reference by the facility clock, to the second', () => {
    expect(defaultName(new Date())).toMatch(/^ref-\d{8}-\d{6}$/);
  });
});

describe('Select Current as Reference', () => {
  it('needs a fresh TbT shot this tab may select', () => {
    expect(selectCurrentReason(null, true)).toMatch(/shot/i);
    expect(selectCurrentReason(shot({ fresh: false }), true)).toMatch(/not fresh/);
    expect(selectCurrentReason(shot({ readout: 'pseudo_orbit' }), true)).toMatch(/pseudo_orbit/);
    expect(selectCurrentReason(shot(), false)).toMatch(/hold/i);
    expect(selectCurrentReason(shot(), true)).toBeNull();
  });

  it('a non-fresh latest shot disables the control with its reason', () => {
    const area = mounted();
    area.pushShot(shot({ shot_id: 's-1' }));
    expect(area.control('ref-select-current').disabled).toBe(false);

    area.pushShot(shot({ shot_id: 's-2', fresh: false }));

    const button = area.control('ref-select-current');
    expect(button.disabled).toBe(true);
    expect(button.dataset.reason).toMatch(/s-2.*not fresh/);
    expect(button.title).toBe(button.dataset.reason);
  });

  it('judges the snapshot\'s last shot after a reload, from the stored shots', async () => {
    const area = mounted({
      routes: { 'GET api/shots': () => [200, { model: 'SR', shots: [shot({ shot_id: 's-9', fresh: false })] }] },
    });
    area.pushSnapshot({ last_shot: { shot_id: 's-9' }, reference: null });
    await flush();

    expect(area.control('ref-select-current').dataset.reason).toMatch(/s-9.*not fresh/);
  });

  it('saves the latest fresh shot under the typed name and selects it', async () => {
    const area = mounted({
      routes: {
        'POST api/references': () => [200, { model: 'SR', name: 'flat', sha256: 'abc', shot_id: 's-2' }],
        'POST api/references/select': () => [200, { reference: { name: 'flat' }, step: null }],
      },
    });
    area.pushShot(shot());
    const input = /** @type {HTMLInputElement} */ (area.host.querySelector('[data-control="reference-name"]'));
    input.value = 'flat';

    area.control('ref-select-current').click();
    await flush();
    await flush();

    expect(area.calls.map((c) => c.key)).toEqual(['POST api/references', 'POST api/references/select']);
    expect(area.calls[0].body).toEqual({ name: 'flat', shot_id: 's-2' });
    expect(area.calls[1].body).toEqual({ name: 'flat' });
    expect(area.refusal()?.textContent ?? '').toBe('');
  });

  it('a save refusal is shown inline by its code and nothing is selected', async () => {
    const area = mounted({
      routes: {
        'POST api/references': () => [409, { detail: { code: 'shot_not_fresh', detail: 'shot s-2 is not fresh' } }],
      },
    });
    area.pushShot(shot());

    area.control('ref-select-current').click();
    await flush();

    expect(area.calls.map((c) => c.key)).toEqual(['POST api/references']);
    expect(area.refusal()?.getAttribute('data-code')).toBe('shot_not_fresh');
    expect(area.refusal()?.textContent).toContain('shot s-2 is not fresh');
  });
});

describe('Save Reference to File', () => {
  it('needs a selected reference', () => {
    expect(saveFileReason(null)).toMatch(/select/i);
    expect(saveFileReason({ reference: null })).toMatch(/select/i);
    expect(saveFileReason({ reference: { name: 'golden' } })).toBeNull();
  });

  it('writes the readings route\'s answer as the file, named after the reference', () => {
    expect(fileName('golden')).toBe('golden.json');
    expect(JSON.parse(referenceFile(GOLDEN))).toEqual(GOLDEN);
  });

  it('downloads the selected reference\'s readings', async () => {
    /** @type {Blob[]} */
    const blobs = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
      blobs.push(/** @type {Blob} */ (blob));
      return 'blob:golden';
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const clicked = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(
      /** @this {HTMLAnchorElement} */ function () {
        expect(this.download).toBe('golden.json');
      },
    );
    const area = mounted({
      routes: { 'GET api/references/golden/readings': () => [200, GOLDEN] },
      snapshot: { last_shot: null, reference: { name: 'golden' } },
    });

    area.control('ref-save-file').click();
    await flush();

    expect(clicked).toHaveBeenCalledOnce();
    expect(JSON.parse(await blobs[0].text())).toEqual(GOLDEN);
  });
});

describe('Load Reference from file', () => {
  it('reads a file as the import body, selecting it', () => {
    expect(importBody(referenceFile(GOLDEN))).toEqual({ body: { ...GOLDEN, select: true } });
    expect(importBody('not json').error).toMatch(/JSON/);
    expect(importBody('[1, 2]').error).toMatch(/object/);
  });

  /**
   * @param {ReturnType<typeof mounted>} area
   * @param {string} text
   */
  async function upload(area, text) {
    const input = /** @type {HTMLInputElement} */ (area.host.querySelector('[data-control="reference-file"]'));
    const file = new File([text], 'golden.json', { type: 'application/json' });
    Object.defineProperty(input, 'files', { value: [file], configurable: true });
    input.dispatchEvent(new Event('change'));
    await flush();
    await flush();
  }

  it('an import refusal renders inline by its code', async () => {
    const area = mounted({
      routes: {
        'POST api/references/import': () => [400, {
          detail: { code: 'unknown_bpm', detail: 'the reference names BPM9, which the tool does not read' },
        }],
      },
    });

    await upload(area, referenceFile({ ...GOLDEN, readings: { BPM9: { x: 1, y: 1 } } }));

    expect(area.refusal()?.getAttribute('data-code')).toBe('unknown_bpm');
    expect(area.refusal()?.textContent).toContain('unknown_bpm');
    expect(area.refusal()?.textContent).toContain('BPM9');
  });

  it('a file that is not a reference is refused before any request', async () => {
    const area = mounted();

    await upload(area, '{ broken');

    expect(area.calls).toEqual([]);
    expect(area.refusal()?.textContent).toMatch(/JSON/);
  });

  it('a saved file re-imports unchanged', async () => {
    /** @type {Blob[]} */
    const blobs = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
      blobs.push(/** @type {Blob} */ (blob));
      return 'blob:golden';
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const area = mounted({
      routes: {
        'GET api/references/golden/readings': () => [200, GOLDEN],
        'POST api/references/import': () => [200, { model: 'SR', name: 'golden', sha256: 'abc', reference: null, step: null }],
      },
      snapshot: { last_shot: null, reference: { name: 'golden' } },
    });

    area.control('ref-save-file').click();
    await flush();
    await upload(area, await blobs[0].text());

    const imported = area.calls.find((c) => c.key === 'POST api/references/import');
    expect(imported?.body).toEqual({ ...GOLDEN, select: true });
    expect(area.refusal()?.textContent ?? '').toBe('');
  });

  it('loading needs the session, since it selects', () => {
    const area = mounted({ holding: false });
    expect(area.control('ref-load-file').dataset.reason).toMatch(/hold/i);
  });
});
