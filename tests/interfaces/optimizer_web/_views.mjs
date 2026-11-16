/**
 * A harness for the view tests: a view mounted on a stage and an inspector,
 * with mocked actions and shell, driven by store events.
 */

import { vi } from 'vitest';

import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

/** Let pending promises settle. */
export const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * @param {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/view-host.js').MountView} mount
 * @param {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent[]} events
 * @param {Record<string, any>} [actionsOver]
 */
export function mountWith(mount, events, actionsOver = {}) {
  const stage = document.createElement('section');
  const inspector = document.createElement('aside');
  document.body.replaceChildren(stage, inspector);
  const setPrimary = vi.fn();
  /** @type {Record<string, import('vitest').Mock>} */
  const actions = {
    setSetupWhere: vi.fn(),
    loadEnvironment: vi.fn(async () => {}),
    initialBounds: vi.fn(async () => ({ bounds: null, error: null })),
    read: vi.fn(async () => {}),
    refuseSetup: vi.fn(),
    saveDraft: vi.fn(async () => true),
    closeSetup: vi.fn(),
    start: vi.fn(async () => true),
    markReviewed: vi.fn(),
    editDraft: vi.fn(),
    ...actionsOver,
  };
  const ctx = {
    stage,
    inspector,
    actions: /** @type {any} */ (actions),
    shell: {
      setPrimary,
      fillInspector: (/** @type {any} */ parts) =>
        inspector.replaceChildren(...inspectorFrame(parts, () => {})),
      setDrawer: vi.fn(),
      setDrawerTop: vi.fn(),
    },
  };
  let state = events.reduce(reduce, initialState());
  const view = mount(ctx);
  view.update(state, deriveScreen(state));
  return {
    stage,
    inspector,
    actions,
    setPrimary,
    view,
    get state() {
      return state;
    },
    screen: () => deriveScreen(state),
    /** The last spec the view gave the toolbar's primary slot. */
    primary: () => setPrimary.mock.calls.at(-1)?.[0],
    /** @param {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent[]} more */
    apply(...more) {
      state = more.reduce(reduce, state);
      view.update(state, deriveScreen(state));
    },
  };
}

/**
 * @param {Element|null} el
 * @param {string} value
 */
export function type(el, value) {
  const input = /** @type {HTMLInputElement} */ (el);
  input.value = value;
  input.dispatchEvent(new Event('input'));
}

/**
 * @param {Element|null} el
 * @param {string} value
 */
export function choose(el, value) {
  const select = /** @type {HTMLSelectElement} */ (el);
  select.value = value;
  select.dispatchEvent(new Event('change'));
}
