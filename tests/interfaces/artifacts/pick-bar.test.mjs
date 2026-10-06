/**
 * Unit tests for the gallery's pick bar (pick-bar.js): the slim bar above the
 * sidebar list that deletes several picked artifacts after one confirm.
 *
 *   npx vitest run tests/interfaces/artifacts/pick-bar.test.mjs
 *
 * Every effect is injected — fetch, confirm, the state.js accessors and
 * onDeleted — so these tests hold their own picks and artifact list.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import { createPickBar } from '../../../src/osprey/interfaces/artifacts/static/js/pick-bar.js';

function mountBar() {
  document.body.innerHTML = `
    <div class="sidebar-pick-bar" id="pick-bar" hidden role="toolbar" aria-label="Selected artifacts">
      <span id="pick-count"></span>
      <button class="btn btn-danger" id="pick-delete">Delete</button>
      <button class="btn" id="pick-clear">Clear</button>
    </div>
  `;
  return {
    bar: /** @type {HTMLElement} */ (document.getElementById('pick-bar')),
    count: /** @type {HTMLElement} */ (document.getElementById('pick-count')),
    deleteBtn: /** @type {HTMLButtonElement} */ (document.getElementById('pick-delete')),
    clearBtn: /** @type {HTMLButtonElement} */ (document.getElementById('pick-clear')),
  };
}

/**
 * @param {{ids: string[], pinned?: string[], selected?: string|null, focused?: string|null}} spec
 */
function makeDeps({ ids, pinned = [], selected = null, focused = null }) {
  let artifacts = ids.map((id) => ({ id, pinned: pinned.includes(id) }));
  let picked = new Set(ids);
  /** @type {any} */
  let selectedArtifact = selected ? artifacts.find((a) => a.id === selected) : null;
  /** @type {any} */
  let focusedArtifact = focused ? artifacts.find((a) => a.id === focused) : null;
  const deps = {
    fetch: vi.fn(),
    confirm: vi.fn().mockReturnValue(true),
    log: vi.fn(),
    getPickedIds: () => new Set(picked),
    clearPicked: vi.fn(() => { picked = new Set(); }),
    getArtifacts: () => artifacts,
    removeArtifact: vi.fn((/** @type {string} */ id) => {
      artifacts = artifacts.filter((a) => a.id !== id);
      picked.delete(id);
    }),
    getSelectedArtifact: () => selectedArtifact,
    setSelectedArtifact: vi.fn((/** @type {any} */ a) => { selectedArtifact = a; }),
    getFocusedArtifact: () => focusedArtifact,
    setFocusedArtifact: vi.fn((/** @type {any} */ a) => { focusedArtifact = a; }),
    onDeleted: vi.fn(),
    onCleared: vi.fn(),
  };
  return {
    deps,
    picks: () => picked,
    artifacts: () => artifacts,
    selected: () => selectedArtifact,
    focused: () => focusedArtifact,
  };
}

/** @param {any} body */
function okResponse(body) {
  return { ok: true, json: () => Promise.resolve(body) };
}

/** Let the fetch promise chain settle. */
async function settle() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
}

/** @type {ReturnType<typeof mountBar>} */
let els;
beforeEach(() => {
  els = mountBar();
});

describe('render', () => {
  test('is hidden with fewer than two picks', () => {
    const { deps } = makeDeps({ ids: ['a'] });
    createPickBar(els, deps).render();
    expect(els.bar.hidden).toBe(true);
  });

  test('shows "3 selected" with three picks', () => {
    const { deps } = makeDeps({ ids: ['a', 'b', 'c'] });
    createPickBar(els, deps).render();
    expect(els.bar.hidden).toBe(false);
    expect(els.count.textContent).toBe('3 selected');
    expect(els.deleteBtn.disabled).toBe(false);
  });

  test('disables Delete with the title "Pinned artifacts are kept" when every pick is pinned', () => {
    const { deps } = makeDeps({ ids: ['a', 'b'], pinned: ['a', 'b'] });
    const bar = createPickBar(els, deps);
    bar.render();
    expect(els.deleteBtn.disabled).toBe(true);
    expect(els.deleteBtn.title).toBe('Pinned artifacts are kept');

    els.deleteBtn.click();
    expect(deps.confirm).not.toHaveBeenCalled();
    expect(deps.fetch).not.toHaveBeenCalled();
  });

  test('re-enables Delete once an unpinned row is picked', () => {
    const state = makeDeps({ ids: ['a', 'b', 'c'], pinned: ['a', 'b'] });
    state.deps.getPickedIds = () => new Set(['a', 'b']);
    const bar = createPickBar(els, state.deps);
    bar.render();
    expect(els.deleteBtn.disabled).toBe(true);

    state.deps.getPickedIds = () => new Set(['a', 'b', 'c']);
    bar.render();
    expect(els.deleteBtn.disabled).toBe(false);
    expect(els.deleteBtn.title).toBe('');
  });
});

describe('Delete', () => {
  test('confirms "Delete 3 artifacts? This cannot be undone." with no pinned rows', () => {
    const { deps } = makeDeps({ ids: ['a', 'b', 'c'] });
    deps.confirm.mockReturnValue(false);
    createPickBar(els, deps).render();
    els.deleteBtn.click();
    expect(deps.confirm).toHaveBeenCalledWith('Delete 3 artifacts? This cannot be undone.');
  });

  test('confirms "… 2 pinned stay." and sends only the unpinned ids when two of five are pinned', async () => {
    const { deps } = makeDeps({ ids: ['a', 'b', 'c', 'd', 'e'], pinned: ['b', 'd'] });
    deps.fetch.mockResolvedValue(okResponse({ deleted: ['a', 'c', 'e'], missing: [] }));
    createPickBar(els, deps).render();

    els.deleteBtn.click();
    await settle();

    expect(deps.confirm).toHaveBeenCalledWith('Delete 3 artifacts? 2 pinned stay. This cannot be undone.');
    expect(deps.fetch).toHaveBeenCalledTimes(1);
    const [url, init] = deps.fetch.mock.calls[0];
    expect(url).toBe('/api/artifacts/delete');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(init.body)).toEqual({ ids: ['a', 'c', 'e'] });
  });

  test('a cancelled confirm sends nothing', () => {
    const { deps } = makeDeps({ ids: ['a', 'b'] });
    deps.confirm.mockReturnValue(false);
    createPickBar(els, deps).render();
    els.deleteBtn.click();
    expect(deps.fetch).not.toHaveBeenCalled();
    expect(deps.clearPicked).not.toHaveBeenCalled();
  });

  test('on success removes every deleted id, clears a deleted selected or focused artifact, clears the picks and runs onDeleted once', async () => {
    const state = makeDeps({ ids: ['a', 'b', 'c'], selected: 'a', focused: 'b' });
    const { deps } = state;
    deps.fetch.mockResolvedValue(okResponse({ deleted: ['a', 'b'], missing: ['c'] }));
    createPickBar(els, deps).render();

    els.deleteBtn.click();
    await settle();

    expect(deps.removeArtifact.mock.calls.map((c) => c[0])).toEqual(['a', 'b']);
    expect(state.selected()).toBeNull();
    expect(state.focused()).toBeNull();
    expect(state.picks().size).toBe(0);
    expect(deps.onDeleted).toHaveBeenCalledTimes(1);
    expect(els.bar.hidden).toBe(true);
  });

  test('on success leaves a selected or focused artifact that was not deleted', async () => {
    const state = makeDeps({ ids: ['a', 'b', 'c'], selected: 'c', focused: 'c' });
    state.deps.getPickedIds = () => new Set(['a', 'b']);
    state.deps.fetch.mockResolvedValue(okResponse({ deleted: ['a', 'b'], missing: [] }));
    createPickBar(els, state.deps).render();

    els.deleteBtn.click();
    await settle();

    expect(state.selected()?.id).toBe('c');
    expect(state.focused()?.id).toBe('c');
    expect(state.deps.setSelectedArtifact).not.toHaveBeenCalled();
    expect(state.deps.setFocusedArtifact).not.toHaveBeenCalled();
  });

  test('a failed request leaves the picks and logs', async () => {
    const state = makeDeps({ ids: ['a', 'b'] });
    state.deps.fetch.mockResolvedValue({ ok: false, json: () => Promise.resolve({}) });
    createPickBar(els, state.deps).render();

    els.deleteBtn.click();
    await settle();

    expect(state.picks().size).toBe(2);
    expect(state.deps.removeArtifact).not.toHaveBeenCalled();
    expect(state.deps.onDeleted).not.toHaveBeenCalled();
    expect(state.deps.log).toHaveBeenCalledTimes(1);
    expect(state.deps.log.mock.calls[0][0]).toBe('Delete failed:');
  });

  test('a rejected request leaves the picks and logs', async () => {
    const state = makeDeps({ ids: ['a', 'b'] });
    state.deps.fetch.mockRejectedValue(new Error('offline'));
    createPickBar(els, state.deps).render();

    els.deleteBtn.click();
    await settle();

    expect(state.picks().size).toBe(2);
    expect(state.deps.log).toHaveBeenCalledTimes(1);
  });
});

describe('Clear', () => {
  test('empties the picks and hides the bar', () => {
    const state = makeDeps({ ids: ['a', 'b', 'c'] });
    createPickBar(els, state.deps).render();

    els.clearBtn.click();

    expect(state.picks().size).toBe(0);
    expect(els.bar.hidden).toBe(true);
    expect(state.deps.onCleared).toHaveBeenCalledTimes(1);
  });

  test('clear() empties the picks and says whether any were held', () => {
    const state = makeDeps({ ids: ['a', 'b'] });
    const bar = createPickBar(els, state.deps);
    bar.render();

    expect(bar.clear()).toBe(true);
    expect(state.picks().size).toBe(0);
    expect(bar.clear()).toBe(false);
    expect(state.deps.onCleared).toHaveBeenCalledTimes(1);
  });
});
