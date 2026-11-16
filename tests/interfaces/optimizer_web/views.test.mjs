/**
 * Tests for the view contract (view-host.js) and the empty screen
 * (empty-view.js): one view at a time, kept across its own screens, emptied
 * zones on a switch, a failing view contained; and what the empty screen
 * shows before, after and without a read of the service.
 */

import { describe, test, expect, vi } from 'vitest';

import { mountEmptyView } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/empty-view.js';
import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';
import { createViewHost } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/view-host.js';

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

describe('the view host', () => {
  /** @param {string} name */
  function fakeView(name) {
    const log = { mounted: 0, updates: /** @type {string[]} */ ([]), unmounted: 0 };
    /** @type {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/view-host.js').MountView} */
    const mount = (ctx) => {
      log.mounted += 1;
      ctx.stage.append(document.createElement(name));
      return {
        update: (_state, screen) => log.updates.push(screen.id),
        unmount: () => (log.unmounted += 1),
      };
    };
    return { mount, log };
  }

  test('one instance serves every screen of its view; another view replaces it on emptied zones', () => {
    const { ctx, stage, inspector, setPrimary } = context();
    const a = fakeView('a-view');
    const b = fakeView('b-view');
    const views = /** @type {any} */ ({ empty: a.mount, history: a.mount, result: b.mount });
    const host = createViewHost({ views, ctx });
    const empty = at([{ type: 'poll/job', job: null, at: 0 }]);
    host.update(empty.state, empty.screen);
    const history = at([{ type: 'select', selection: { kind: 'history' } }]);
    host.update(history.state, history.screen);
    expect(a.log).toEqual({ mounted: 1, updates: ['empty', 'history'], unmounted: 0 });

    inspector.append(document.createElement('p'));
    setPrimary.mockClear();
    const result = at([{ type: 'select', selection: { kind: 'run', runId: 'x/y' } }]);
    host.update(result.state, result.screen);
    expect(a.log.unmounted).toBe(1);
    expect(setPrimary).toHaveBeenCalledWith(null);
    expect(inspector.children).toHaveLength(0);
    expect(Array.from(stage.children).map((n) => n.tagName.toLowerCase())).toEqual(['b-view']);
    expect(b.log.updates).toEqual(['result']);
  });

  test('a view that throws is replaced by a centred error; the next view still mounts', () => {
    const { ctx, stage } = context();
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    const good = fakeView('good-view');
    const views = /** @type {any} */ ({
      empty: () => ({
        update: () => {
          throw new Error('broken');
        },
      }),
      history: good.mount,
    });
    const host = createViewHost({ views, ctx });
    const empty = at([{ type: 'poll/job', job: null, at: 0 }]);
    host.update(empty.state, empty.screen);
    expect(stage.querySelector('.osprey-state-detail')?.textContent).toBe('broken');
    expect(error).toHaveBeenCalled();
    const history = at([{ type: 'select', selection: { kind: 'history' } }]);
    host.update(history.state, history.screen);
    expect(stage.querySelector('good-view')).not.toBeNull();
    error.mockRestore();
  });

  test('a view that threw once leaves no primary action and is mounted again on the next update', () => {
    const { ctx, stage, setPrimary } = context();
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    let throws = true;
    let mounts = 0;
    const views = /** @type {any} */ ({
      empty: (/** @type {ViewContext} */ c) => {
        mounts += 1;
        return {
          update: () => {
            if (throws) throw new Error('broken');
            c.stage.replaceChildren(document.createElement('ok-view'));
          },
        };
      },
    });
    const host = createViewHost({ views, ctx });
    const empty = at([{ type: 'poll/job', job: null, at: 0 }]);
    host.update(empty.state, empty.screen);
    expect(setPrimary).toHaveBeenLastCalledWith(null);
    throws = false;
    host.update(empty.state, empty.screen);
    expect(mounts).toBe(2);
    expect(stage.querySelector('ok-view')).not.toBeNull();
    error.mockRestore();
  });
});

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
});
