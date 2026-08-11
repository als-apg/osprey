/**
 * Contract tests for the rail→workspace drag bridge (rail-drag.js):
 *   npx vitest run tests/interfaces/web_terminal/rail-drag.test.mjs
 *
 * rail-drag.js wires HTML5 drags that START on a rail entry into dockview's
 * external-drag hooks: onUnhandledDragOver accepts our MIME on split targets
 * only (one panel per tile — tab/header/center geometries never accept), and
 * onDidDrop resolves the payload + drop position into the caller's
 * onDropPanel(id, position). It also owns the drag-time iframe shields
 * (`.dock-dragging` on .main-container + the adapter's inline pointer shield),
 * raised in railDragStart and lowered in railDragEnd.
 *
 * dockview and the collaborators are stubbed at the module boundary, exactly
 * as in dock-sync.test.mjs.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

const MOD = '../../../src/osprey/interfaces/web_terminal/static/js/rail-drag.js';

const { getDockApi, setIframePointerShield, state } = vi.hoisted(() => ({
  state: { api: /** @type {any} */ (null) },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
  setIframePointerShield: vi.fn(),
}));
getDockApi.mockImplementation(() => state.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  defaultServiceWidth: () => 600,
  setServiceRedock: () => {},
  onDragGesture: () => [],
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  setIframePointerShield,
  PLACEHOLDER_PREFIX: 'iframe:',
  PLACEHOLDER_COMPONENT: 'dock-iframe-placeholder',
}));

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.api = null;
  document.body.innerHTML = '';
  document.documentElement.removeAttribute('data-ui-mode');
});

/** Fake DockviewApi exposing only the external-dnd hooks rail-drag wires. */
function makeApi() {
  /** @type {any} */
  const api = {
    /** @type {null | ((e: any) => void)} */ _overCb: null,
    /** @type {null | ((e: any) => void)} */ _dropCb: null,
  };
  api.onUnhandledDragOver = vi.fn((/** @type {any} */ cb) => {
    api._overCb = cb;
    return { dispose() {} };
  });
  api.onDidDrop = vi.fn((/** @type {any} */ cb) => {
    api._dropCb = cb;
    return { dispose() {} };
  });
  return api;
}

/** An UnhandledDragOverEvent-shaped stub carrying our MIME (or not). */
function overEvent({ target = 'content', position = 'right', withMime = true } = {}) {
  return {
    nativeEvent: { dataTransfer: { types: withMime ? ['application/x-osprey-panel'] : ['text/plain'] } },
    target,
    position,
    accepted: false,
    accept() { this.accepted = true; },
  };
}

/** A DidDrop-shaped stub whose dataTransfer serves the payload on getData.
 *  @param {{id?: string, position?: string, group?: any, withMime?: boolean}} [opts] */
function dropEvent({ id = 'ariel', position = 'right', group = { id: 'g1' }, withMime = true } = {}) {
  return {
    nativeEvent: {
      dataTransfer: {
        getData: (/** @type {string} */ k) =>
          withMime && k === 'application/x-osprey-panel' ? id : '',
      },
    },
    position,
    group,
  };
}

/** @param {any} api */
async function wire(api, onDropPanel = vi.fn()) {
  state.api = api;
  const container = document.createElement('div');
  container.className = 'main-container';
  document.body.appendChild(container);
  const mod = await import(MOD);
  mod.initRailDrag({ onDropPanel });
  return { mod, container, onDropPanel };
}

describe('drag-over acceptance (paints dockview\'s drop overlay)', () => {
  test('accepts a rail drag on a split geometry', async () => {
    const api = makeApi();
    await wire(api);
    const e = overEvent({ target: 'content', position: 'right' });
    api._overCb(e);
    expect(e.accepted).toBe(true);
  });

  test('accepts edge-of-grid targets', async () => {
    const api = makeApi();
    await wire(api);
    const e = overEvent({ target: 'edge', position: 'left' });
    api._overCb(e);
    expect(e.accepted).toBe(true);
  });

  test('never accepts stacking geometries (tab, header space, center)', async () => {
    const api = makeApi();
    await wire(api);
    for (const e of [
      overEvent({ target: 'tab' }),
      overEvent({ target: 'header_space' }),
      overEvent({ target: 'content', position: 'center' }),
    ]) {
      api._overCb(e);
      expect(e.accepted).toBe(false);
    }
  });

  test('ignores foreign drags without the rail MIME (e.g. a file drag)', async () => {
    const api = makeApi();
    await wire(api);
    const e = overEvent({ withMime: false });
    api._overCb(e);
    expect(e.accepted).toBe(false);
  });
});

describe('drop routing', () => {
  test('resolves the payload and hands (id, {referenceGroup, direction}) to onDropPanel', async () => {
    const api = makeApi();
    const group = { id: 'g1' };
    const { onDropPanel } = await wire(api);

    api._dropCb(dropEvent({ id: 'okf', position: 'bottom', group }));

    // Overlay 'bottom' translates to addPanel's 'below' — the two vocabularies
    // differ on the vertical axis and the untranslated value throws.
    expect(onDropPanel).toHaveBeenCalledExactlyOnceWith('okf', {
      referenceGroup: group,
      direction: 'below',
    });
  });

  test('overlay "top" translates to direction "above"', async () => {
    const api = makeApi();
    const group = { id: 'g1' };
    const { onDropPanel } = await wire(api);

    api._dropCb(dropEvent({ id: 'okf', position: 'top', group }));

    expect(onDropPanel).toHaveBeenCalledExactlyOnceWith('okf', {
      referenceGroup: group,
      direction: 'above',
    });
  });

  test('a groupless (grid-edge) drop passes a direction-only position', async () => {
    const api = makeApi();
    const { onDropPanel } = await wire(api);

    api._dropCb(dropEvent({ id: 'okf', position: 'left', group: null }));

    expect(onDropPanel).toHaveBeenCalledExactlyOnceWith('okf', { direction: 'left' });
  });

  test('ignores drops without the rail MIME payload', async () => {
    const api = makeApi();
    const { onDropPanel } = await wire(api);
    api._dropCb(dropEvent({ withMime: false }));
    expect(onDropPanel).not.toHaveBeenCalled();
  });

  test('a center drop never places (stacking backstop)', async () => {
    const api = makeApi();
    const { onDropPanel } = await wire(api);
    api._dropCb(dropEvent({ position: 'center' }));
    expect(onDropPanel).not.toHaveBeenCalled();
  });
});

describe('railDragStart / railDragEnd — payload + shields', () => {
  /** @returns {any} minimal DataTransfer stand-in */
  function fakeDT() {
    return {
      data: /** @type {Record<string, string>} */ ({}),
      effectAllowed: '',
      setData(/** @type {string} */ k, /** @type {string} */ v) { this.data[k] = v; },
    };
  }

  test('stamps the MIME payload, raises both shields, and reports accepted', async () => {
    const api = makeApi();
    const { mod, container } = await wire(api);
    const dt = fakeDT();

    expect(mod.railDragStart('ariel', dt)).toBe(true);
    expect(dt.data['application/x-osprey-panel']).toBe('ariel');
    expect(dt.data['text/plain']).toBe('ariel');
    expect(container.classList.contains('dock-dragging')).toBe(true);
    expect(setIframePointerShield).toHaveBeenCalledExactlyOnceWith(true);
  });

  test('railDragEnd lowers both shields', async () => {
    const api = makeApi();
    const { mod, container } = await wire(api);
    mod.railDragStart('ariel', fakeDT());

    mod.railDragEnd();

    expect(container.classList.contains('dock-dragging')).toBe(false);
    expect(setIframePointerShield).toHaveBeenLastCalledWith(false);
  });

  test('a document-level terminator lowers the shields when the source\'s dragend is lost', async () => {
    // HTML5 delivers dragend only to the drag's SOURCE element; a rail entry
    // removed mid-drag (SSE hide_panel / arrange prune) can never fire it, and
    // during a native drag mouseup/pointerup are suppressed. Without a
    // document-level fallback the shields stay up forever and every panel
    // iframe is left pointer-events:none — the "frozen workspace" bug.
    const api = makeApi();
    const { mod, container } = await wire(api);
    mod.railDragStart('ariel', fakeDT());
    expect(container.classList.contains('dock-dragging')).toBe(true);

    // The source button was removed; the user's next gesture reaches document.
    document.dispatchEvent(new Event('mouseup'));

    expect(container.classList.contains('dock-dragging')).toBe(false);
    expect(setIframePointerShield).toHaveBeenLastCalledWith(false);
  });

  test('terminator listeners disarm after firing — a later gesture cannot re-lower mid-drag', async () => {
    const api = makeApi();
    const { mod, container } = await wire(api);

    // First drag ends via the failsafe.
    mod.railDragStart('ariel', fakeDT());
    document.dispatchEvent(new Event('mouseup'));
    expect(container.classList.contains('dock-dragging')).toBe(false);

    // Second drag: the stale listeners must be gone, so the shield stays up
    // until this drag's own terminator.
    mod.railDragStart('okf', fakeDT());
    expect(container.classList.contains('dock-dragging')).toBe(true);
    document.dispatchEvent(new Event('dragend'));
    expect(container.classList.contains('dock-dragging')).toBe(false);
  });

  test('railDragEnd also disarms the failsafe (normal completion path)', async () => {
    const api = makeApi();
    const { mod, container } = await wire(api);
    mod.railDragStart('ariel', fakeDT());
    mod.railDragEnd();
    expect(container.classList.contains('dock-dragging')).toBe(false);
    setIframePointerShield.mockClear();

    // No armed listener may fire later and fight a dockview-owned gesture.
    document.dispatchEvent(new Event('mouseup'));
    expect(setIframePointerShield).not.toHaveBeenCalled();
  });

  test('cancels in simple mode (locked layout)', async () => {
    const api = makeApi();
    const { mod, container } = await wire(api);
    document.documentElement.setAttribute('data-ui-mode', 'simple');

    expect(mod.railDragStart('ariel', fakeDT())).toBe(false);
    expect(container.classList.contains('dock-dragging')).toBe(false);
    expect(setIframePointerShield).not.toHaveBeenCalled();
  });

  test('cancels in fallback mode (no dock to drop into)', async () => {
    const { mod } = await wire(null);
    expect(mod.railDragStart('ariel', fakeDT())).toBe(false);
  });
});

describe('initRailDrag — wiring, idempotency, late arrival', () => {
  test('is idempotent — a second init does not double-wire', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    mod.initRailDrag({ onDropPanel: vi.fn() });
    expect(api.onUnhandledDragOver).toHaveBeenCalledTimes(1);
    expect(api.onDidDrop).toHaveBeenCalledTimes(1);
  });

  test('retries and wires once a late DockviewApi arrives', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import(MOD);
      mod.initRailDrag({ onDropPanel: vi.fn() });
      const api = makeApi();
      state.api = api;
      expect(api.onDidDrop).not.toHaveBeenCalled();
      vi.advanceTimersByTime(150);
      expect(api.onDidDrop).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
