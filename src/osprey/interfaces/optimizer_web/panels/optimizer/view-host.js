// @ts-check
/**
 * The view contract, and the host that mounts one view at a time.
 *
 * A view is one module that exports a mount function ({@link MountView}).
 * The host mounts the view the table names for the screen, calls its
 * `update(state, screen)` after every store change, and unmounts it when
 * another view's screen comes: it then empties the stage, the inspector and
 * the primary slot. Two screens served by the same mount function (running
 * and paused, say) keep one instance, so the view keeps its DOM, its scroll
 * and its focus between them.
 *
 * A view
 * - draws only into `ctx.stage` and `ctx.inspector` (with `ctx.shell.fillInspector`
 *   for the inspector's head, body and foot);
 * - reads only `state` and `screen`, and asks for data and acts through
 *   `ctx.actions` (it never calls the sidecar);
 * - redraws only what changed: `update` runs on every poll (about once a
 *   second during a run), so a view compares a key of what it shows and
 *   leaves a control under the cursor alone;
 * - sets the toolbar's primary action with `ctx.shell.setPrimary` (empty
 *   during a run, whatever it asks).
 *
 * A view that throws is replaced by a centred error, so one broken view does
 * not stop the poll or the other zones.
 *
 * @module view-host
 */

import { renderState } from '/design-system/js/panel-state.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {import('./store.js').ScreenId} ScreenId
 *
 * @typedef {{
 *   stage: HTMLElement,
 *   inspector: HTMLElement,
 *   actions: import('./actions.js').Actions,
 *   shell: {
 *     setPrimary: (spec: import('./shell.js').PrimarySpec|null) => void,
 *     fillInspector: (parts: import('./shell.js').InspectorParts) => void,
 *     setDrawer: (open: boolean) => void,
 *     setDrawerTop: (px: number) => void,
 *   },
 * }} ViewContext
 *
 * @typedef {{
 *   update: (state: State, screen: Screen) => void,
 *   unmount?: () => void,
 * }} ViewInstance
 *
 * @typedef {(ctx: ViewContext) => ViewInstance} MountView
 * @typedef {Record<ScreenId, MountView>} ViewTable
 */

/**
 * @param {{ views: ViewTable, ctx: ViewContext }} options
 */
export function createViewHost({ views, ctx }) {
  /** @type {MountView|null} */
  let mounted = null;
  /** @type {ViewInstance|null} */
  let instance = null;

  /** @param {unknown} err */
  function fail(err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error('optimizer panel: a view failed', err);
    // Mount again on the next update, and leave no stale primary action.
    instance = null;
    mounted = null;
    ctx.shell.setPrimary(null);
    renderState(ctx.stage, {
      icon: 'triangle-alert',
      message: 'This view could not be drawn',
      detail: message,
    });
  }

  return {
    /**
     * @param {State} state
     * @param {Screen} screen
     */
    update(state, screen) {
      const mount = views[screen.id];
      try {
        if (mount !== mounted) {
          instance?.unmount?.();
          ctx.stage.replaceChildren();
          ctx.inspector.replaceChildren();
          ctx.shell.setPrimary(null);
          mounted = mount;
          instance = mount(ctx);
        }
        instance?.update(state, screen);
      } catch (err) {
        fail(err);
      }
    },
  };
}
