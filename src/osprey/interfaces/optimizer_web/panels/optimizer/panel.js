// @ts-check
/**
 * OPTIMIZER panel entry: builds the store, the reads (sync.js), the actions,
 * the frame (shell.js), the runs list and the view host, wires them to each
 * other, and starts the poll. It draws nothing itself.
 *
 * After every store change the frame, the runs list and the view of the
 * screen are drawn from the same state and the same derived screen, so no
 * zone can show a different moment than another.
 *
 * Nothing restores by itself: a cancelled or failed run holds the machine
 * where it is, and the panel has no Restore.
 *
 * @module panel
 */

import { panelApiPrefix } from '/design-system/js/dom.js';
import { markPageReady } from '/design-system/js/page-ready.js';
import { initTooltips } from '/design-system/js/tooltip.js';
import { createActions } from './actions.js';
import { createApi } from './api-client.js';
import { mountEmptyView } from './empty-view.js';
import { mountHistoryView } from './history-view.js';
import { mountOutcomeView } from './outcome-view.js';
import { createPoller } from './poller.js';
import { mountReviewView } from './review-view.js';
import { mountRunPage } from './run-page.js';
import { createRunsList } from './runs-list.js';
import { mountSetupView } from './setup-view.js';
import { createShell } from './shell.js';
import { createStore, deriveScreen } from './store.js';
import { createSync } from './sync.js';
import { createViewHost } from './view-host.js';

/**
 * Which view draws each screen. One group per view module, each group on
 * lines of its own, so a view that replaces its group's entries touches no
 * other group's lines.
 *
 * @type {import('./view-host.js').ViewTable}
 */
export const VIEWS = {
  empty: mountEmptyView,

  // Set-up and review: setup-view.js, knob-picker.js, review-view.js.
  setup: mountSetupView,
  'setup-issues': mountSetupView,
  review: mountReviewView,
  'review-changed': mountReviewView,
  'review-no-owner': mountReviewView,

  // The run page: run-page.js, objective-chart.js.
  'waiting-approval': mountRunPage,
  running: mountRunPage,
  paused: mountRunPage,

  // Outcome, history and a past run: outcome-view.js, history-view.js.
  completed: mountOutcomeView,
  failed: mountOutcomeView,
  cancelled: mountOutcomeView,
  history: mountHistoryView,
  result: mountOutcomeView,
};

/**
 * @param {{
 *   api: import('./api-client.js').Api,
 *   doc?: Document,
 *   views?: import('./view-host.js').ViewTable,
 *   setTimer?: (fn: () => void, ms: number) => any,
 *   clearTimer?: (handle: any) => void,
 *   now?: () => number,
 * }} options
 */
export function createPanel({
  api,
  doc = document,
  views = VIEWS,
  setTimer,
  clearTimer,
  now = Date.now,
}) {
  const store = createStore();
  const sync = createSync({ api, store, now });
  const poller = createPoller({ tick: sync.tick, setTimer, clearTimer });
  const actions = createActions({ api, store, refresh: () => poller.now(), now });
  const shell = createShell({ actions, doc });
  const list = createRunsList({
    host: /** @type {HTMLElement} */ (doc.getElementById('runs-list')),
    actions,
    onPicked: () => shell.closeOverlay(),
    now,
  });
  const host = createViewHost({
    views,
    ctx: {
      ...shell.zones,
      actions,
      shell: {
        setPrimary: shell.setPrimary,
        fillInspector: shell.fillInspector,
        setDrawer: shell.setDrawer,
        setDrawerTop: shell.setDrawerTop,
      },
    },
  });

  let drawing = false;
  let again = false;
  /**
   * Draw every zone from the store's latest state. An action a zone calls
   * while it draws (a read it asks for) changes the state: the zones are then
   * drawn once more, in order, so none is left with an older state.
   */
  function draw() {
    if (drawing) {
      again = true;
      return;
    }
    drawing = true;
    try {
      do {
        again = false;
        const state = store.state;
        const screen = deriveScreen(state);
        shell.update(state, screen);
        list.update(state);
        host.update(state, screen);
      } while (again);
    } finally {
      drawing = false;
    }
  }
  store.subscribe(draw);

  return {
    store,
    actions,
    poller,
    async boot() {
      draw();
      // Which account this session's writes carry; a failed read leaves it unknown.
      void api.session().then((owner) => store.dispatch({ type: 'session', ...owner }));
      await sync.loadCatalog();
      poller.start();
    },
  };
}

if (typeof document !== 'undefined' && document.getElementById('optimizer-root')) {
  initTooltips(document.body);
  // Ready once boot() has drawn the first state (index.html declares the mark
  // pending until then).
  void createPanel({ api: createApi({ prefix: panelApiPrefix() }) })
    .boot()
    .finally(() => markPageReady());
}
