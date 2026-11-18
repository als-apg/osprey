/**
 * Tests for the empty screen (empty-view.js): what it shows before, after and
 * without a read of the service, and that it redraws only when that changes.
 */

import { describe, test, expect, vi } from 'vitest';

import { mountEmptyView } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/empty-view.js';
import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';
import { BEFORE_A_WRITE } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/view-host.js').ViewContext} ViewContext */

function context() {
  const stage = document.createElement('section');
  const inspector = document.createElement('aside');
  const setPrimary = vi.fn();
  /** @type {ViewContext} */
  const ctx = {
    stage,
    inspector,
    actions: /** @type {any} */ ({ newRun: vi.fn(), refresh: vi.fn(async () => {}) }),
    shell: {
      setPrimary,
      fillInspector: (parts) => inspector.replaceChildren(...inspectorFrame(parts, () => {})),
      setDrawer: vi.fn(),
      setDrawerTop: vi.fn(),
    },
  };
  return { ctx, stage, inspector, setPrimary };
}

/** @param {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent[]} events */
const at = (events) => {
  const state = events.reduce(reduce, initialState());
  return { state, screen: deriveScreen(state) };
};

describe('the empty screen', () => {
  test('before the first read it says it is reading, and offers no conclusion', () => {
    const { ctx, stage } = context();
    const view = mountEmptyView(ctx);
    const { state, screen } = at([]);
    view.update(state, screen);
    expect(stage.textContent).toContain('Reading the optimizer');
    expect(stage.textContent).not.toContain('No draft');
  });

  test('with nothing to show it offers a new run, in the stage and as the primary action', () => {
    const { ctx, stage, inspector, setPrimary } = context();
    const view = mountEmptyView(ctx);
    const { state, screen } = at([{ type: 'poll/job', job: null, at: 0 }]);
    view.update(state, screen);
    expect(stage.querySelector('.osprey-state-message')?.textContent).toBe('No draft and no run');
    /** @type {HTMLElement} */ (stage.querySelector('button')).click();
    expect(ctx.actions.newRun).toHaveBeenCalledTimes(1);
    expect(setPrimary).toHaveBeenCalledWith(expect.objectContaining({ label: 'New run' }));
    expect(inspector.querySelectorAll('.opt-how li')).toHaveLength(5);
    const drawn = stage.firstElementChild?.firstElementChild;
    view.update(state, screen);
    expect(stage.firstElementChild?.firstElementChild).toBe(drawn);
  });

  test('with nothing to show it says when a run may write, as the stage of a draft does', () => {
    const { ctx, stage } = context();
    const view = mountEmptyView(ctx);
    const { state, screen } = at([{ type: 'poll/job', job: null, at: 0 }]);
    view.update(state, screen);
    // A grant by the owner's switch, not only an approval by a person.
    expect(stage.textContent).toContain(BEFORE_A_WRITE);
  });

  test('an unreachable service says so and offers a retry', () => {
    const { ctx, stage } = context();
    const view = mountEmptyView(ctx);
    const { state, screen } = at([{ type: 'poll/failed', error: 'optimizer service unreachable' }]);
    view.update(state, screen);
    expect(stage.querySelector('.osprey-state-detail')?.textContent).toBe(
      'optimizer service unreachable'
    );
    /** @type {HTMLElement} */ (stage.querySelector('button')).click();
    expect(ctx.actions.refresh).toHaveBeenCalledTimes(1);
  });

  test('the primary action and the inspector foot both open a new run', () => {
    const { ctx, inspector, setPrimary } = context();
    const view = mountEmptyView(ctx);
    const { state, screen } = at([{ type: 'poll/job', job: null, at: 0 }]);
    view.update(state, screen);
    setPrimary.mock.calls.at(-1)?.[0].onClick();
    /** @type {HTMLElement} */ (inspector.querySelector('button.primary')).click();
    expect(ctx.actions.newRun).toHaveBeenCalledTimes(2);
  });

  test('another reason for a failed read replaces the one shown; a read that answers clears it', () => {
    const { ctx, stage } = context();
    const view = mountEmptyView(ctx);
    const first = at([{ type: 'poll/failed', error: 'unreachable' }]);
    view.update(first.state, first.screen);
    const next = at([{ type: 'poll/failed', error: 'timed out' }]);
    view.update(next.state, next.screen);
    expect(stage.querySelector('.osprey-state-detail')?.textContent).toBe('timed out');
    const back = at([
      { type: 'poll/failed', error: 'timed out' },
      { type: 'poll/job', job: null, at: 1 },
    ]);
    view.update(back.state, back.screen);
    expect(stage.querySelector('.osprey-state-detail')?.textContent).not.toBe('timed out');
    /** @type {HTMLElement} */ (stage.querySelector('button')).click();
    expect(ctx.actions.newRun).toHaveBeenCalledTimes(1);
    expect(ctx.actions.refresh).not.toHaveBeenCalled();
  });
});
