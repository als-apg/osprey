import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  bindControls,
  draftOf,
  panelOf,
  setSelection,
  setShape,
  showState,
} from '../panel/panel.js';

// The panel's markup, as index.html carries it.
const MARKUP = `
  <span data-field="plane"><button data-plane="x"></button><button data-plane="y"></button></span>
  <select data-field="model" hidden></select>
  <span data-field="orm"><span data-field="orm-text"></span><button data-field="recompute"></button></span>
  <span data-field="no-model" hidden>no closed-orbit model</span>
  <p data-field="posture" hidden></p>
  <p data-field="notice" hidden></p>
  <p data-field="error" hidden></p>
  <div data-field="body"></div>
  <div data-field="confirm" hidden></div>`;

/** @returns {import('../panel/panel.js').BumpState} */
function state() {
  return {
    models: [
      {
        name: 'SR',
        length_m: 196.8,
        orm: { status: 'ready', age_s: 5, version: 1, error: null, unsolved: [] },
        members: { bpms: [], correctors: [] },
      },
    ],
    posture: { allowed: true, reason: null, phrase: null },
    orbit: { position: {}, golden: {} },
    plan: null,
  };
}

/**
 * A fetcher answering every request with `status`, recording the draft bodies it was sent.
 * @param {number} [status]
 */
function recorder(status = 204) {
  /** @type {any[]} */
  const drafts = [];
  const fetcher = vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init) => {
    if (url === 'api/draft') drafts.push(JSON.parse(String(init?.body)));
    return new Response(null, { status });
  });
  return { fetcher: /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)), drafts };
}

describe("the panel's draft", () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    vi.useFakeTimers();
    root = document.createElement('main');
    root.innerHTML = MARKUP;
    document.body.append(root);
  });

  afterEach(() => {
    vi.useRealTimers();
    root.remove();
  });

  it('carries everything the operator has set, computed or not', () => {
    showState(root, state());
    setSelection(root, { targets: { B2: 0.1 }, rule: 'manual', correctors: ['C1'] });
    expect(draftOf(panelOf(root))).toEqual({
      plane: 'x',
      model: 'SR',
      targets: { B2: 0.1 },
      rule: 'manual',
      correctors: ['C1'],
      priority: 'closure',
      outside: 'hold',
      shape: 'flat',
      amplitude: 0,
    });
  });

  it('publishes once the operator pauses, with the last edit only', async () => {
    const { fetcher, drafts } = recorder();
    bindControls(root, fetcher);
    showState(root, state());
    setShape(root, 'peak');
    setSelection(root, { targets: { B2: 0.1 } });
    setSelection(root, { targets: { B2: 0.2 }, priority: 'targets' });
    expect(drafts).toEqual([]);
    await vi.advanceTimersByTimeAsync(300);
    expect(drafts).toHaveLength(1);
    expect(drafts[0]).toMatchObject({ targets: { B2: 0.2 }, priority: 'targets', shape: 'peak' });
  });

  it('sends nothing again while the draft is unchanged, as polls render', async () => {
    const { fetcher, drafts } = recorder();
    bindControls(root, fetcher);
    showState(root, state());
    await vi.advanceTimersByTimeAsync(300);
    showState(root, state());
    showState(root, state());
    await vi.advanceTimersByTimeAsync(300);
    expect(drafts).toHaveLength(1);
  });

  it('retries a failed publish on the next render, but not after a 403', async () => {
    const failing = recorder(503);
    bindControls(root, failing.fetcher);
    showState(root, state());
    await vi.advanceTimersByTimeAsync(300);
    showState(root, state());
    await vi.advanceTimersByTimeAsync(300);
    expect(failing.drafts).toHaveLength(2);

    const refused = recorder(403);
    panelOf(root).fetcher = refused.fetcher;
    setSelection(root, { targets: { B3: 0.1 } });
    await vi.advanceTimersByTimeAsync(300);
    showState(root, state());
    await vi.advanceTimersByTimeAsync(300);
    expect(refused.drafts).toHaveLength(1);
  });

  it('publishes nothing from a read-only panel or one never wired', async () => {
    const { fetcher, drafts } = recorder();
    showState(root, state());
    panelOf(root).fetcher = fetcher;
    setSelection(root, { targets: { B2: 0.1 } });
    await vi.advanceTimersByTimeAsync(300);
    expect(drafts).toEqual([]);

    bindControls(root, fetcher);
    panelOf(root).readOnly = true;
    setSelection(root, { targets: { B2: 0.3 } });
    await vi.advanceTimersByTimeAsync(300);
    expect(drafts).toEqual([]);
  });
});
