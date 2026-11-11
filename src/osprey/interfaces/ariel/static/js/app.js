// @ts-check
/**
 * ARIEL panel bootstrap.
 *
 * Boot order is load-bearing: frame kind and theme role first, capabilities
 * once, the facility time zone before anything renders a time, then the
 * panes and their hosts, and the router last so the initial hash reaches
 * every store subscriber. `initSettings` runs before `initLayout` because it
 * may remove `#settings-drawer`.
 *
 * The reader is store-free, so its wiring lives here: it reads `sel` and
 * `att`, writes `att` from the lightbox callbacks, `sel` from thread links and
 * `compose` from the Follow up action. Reader features join through the
 * reader seam in `registerReaderFeatures`, and their keys reach the shown
 * entry through `wireReaderKeys`, so a feature edits neither `reader.js` nor
 * `keyboard.js`.
 */

import { initTheme } from '/design-system/js/theme-manager.js';
import { markPageReady } from '/design-system/js/page-ready.js';
import { applyEmbedded } from '/design-system/js/frame-params.js';
import '/design-system/js/components/osprey-drawer.js';
import { capabilitiesApi } from './api.js';
import { applyFacilityTimezone, setEnhancementModules } from './components.js';
import { initSettings } from './settings.js';
import { initLayout } from './layout.js';
import { initRail } from './rail.js';
import { initFacets } from './facets.js';
import { initList } from './list.js';
import { initListActivity } from './list-activity.js';
import { showInReader, clearReader, closeLightbox, runReaderKey } from './reader.js';
import { initEntries } from './entries.js';
import { initDashboard } from './dashboard.js';
import { initKeyboard } from './keyboard.js';
import { initHeader } from './header.js';
import { initRouter } from './router.js';
import { registerAskAgent } from './ask-agent.js';
import { registerRelated } from './related.js';
import { registerInspector } from './inspector.js';
import { getState, setState, subscribe } from './store.js';

/** Text-entry controls, where keys type rather than act (as in `keyboard.js`). */
const TYPING_SELECTOR = 'input, textarea, select, [contenteditable]:not([contenteditable="false"])';

// The host tile bar is the one header.
applyEmbedded();

// Theming is the hub's; the panel only follows its broadcasts.
initTheme({ role: 'follower' });

/**
 * Map `sel` / `att` onto the reader pane.
 *
 * A thumbnail click opens the lightbox inside the reader, which reports it
 * through `onAttachmentOpen`; `shownAtt` records that so the resulting `att`
 * write is not answered by loading the entry again. `att:null` from outside
 * (Esc, the router) closes the lightbox without a callback, so nothing is
 * written back. A new `sel` takes `att` only from the same patch (a deep
 * link); a stale `att` is cleared, as is one naming no viewable picture.
 * @param {HTMLElement} container - `#ariel-reader-body`
 */
function wireReader(container) {
  /** @type {string|null} */
  let shownAtt = null;

  const callbacks = {
    onAttachmentOpen: (/** @type {string|null} */ id) => {
      shownAtt = id;
      if (getState().att !== id) setState({ att: id });
    },
    onAttachmentClose: () => {
      shownAtt = null;
      setState({ att: null });
    },
    onAttachmentMissing: () => {
      shownAtt = null;
      if (getState().att !== null) setState({ att: null });
    },
    onSelectEntry: (/** @type {string} */ id) => {
      if (id) setState({ sel: id, att: null, narrowView: 'reader' });
    },
    onFollowUp: (/** @type {string} */ id) => {
      if (id) setState({ compose: { draft: null, followsUp: id } });
    },
  };

  /** @param {string|null} sel @param {string|null} att */
  const show = (sel, att) => {
    closeLightbox(container);
    shownAtt = null;
    if (sel) showInReader(container, sel, { attachmentId: att, ...callbacks });
    else clearReader(container);
  };

  subscribe((next, _prev, changed) => {
    if (changed.has('sel')) {
      const att = changed.has('att') ? next.att : null;
      show(next.sel, att);
      // Deferred: a write from inside a subscriber would reach later
      // subscribers (the router's hash) before this one's `next`.
      if (att === null && next.att !== null) {
        const { sel, att: stale } = next;
        queueMicrotask(() => {
          const now = getState();
          if (now.sel === sel && now.att === stale) setState({ att: null });
        });
      }
    } else if (changed.has('att')) {
      if (next.att === null) {
        closeLightbox(container);
        shownAtt = null;
      } else if (next.att !== shownAtt && next.sel) {
        show(next.sel, next.att);
      }
    }
  });

  const { sel, att } = getState();
  if (sel) show(sel, att);
}

/**
 * Register the reader's seam features (`reader.js` `registerReader*`), once,
 * before the reader first renders. The thread section is the reader's own
 * first section; each feature module adds its registration call here.
 */
function registerReaderFeatures() {
  // One call per feature, in display order.
  registerAskAgent();
  registerRelated();
  registerInspector();
}

/**
 * Route keys registered through the reader seam to the shown entry. Runs
 * after `keyboard.js`, whose own keys are `defaultPrevented` by then, under
 * its guards: this document's keydowns only, no modifier, not while typing.
 * @param {Document} doc
 * @param {HTMLElement} container - `#ariel-reader-body`
 */
function wireReaderKeys(doc, container) {
  doc.defaultView?.addEventListener('keydown', (event) => {
    const el = /** @type {HTMLElement|null} */ (event.target);
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (el?.ownerDocument && el.ownerDocument !== doc) return;
    if (el?.isContentEditable || el?.closest?.(TYPING_SELECTOR)) return;
    if (runReaderKey(container, event.key)) event.preventDefault();
  });
}

/**
 * Boot the panel. Module init is non-fatal so the router still runs, and the
 * page still navigates, without a backend.
 */
async function init() {
  /** @type {any} */
  let capabilities = null;
  try {
    capabilities = await capabilitiesApi.get();
  } catch (e) {
    console.warn('Failed to fetch capabilities, using fallback:', e);
  }

  try {
    applyFacilityTimezone(capabilities);
    setEnhancementModules(capabilities?.enhancement_modules);
    initSettings(capabilities);
    initLayout(document);

    const rail = document.getElementById('ariel-rail');
    if (rail) initRail(rail);
    const facets = rail?.querySelector('.ariel-rail-facets');
    if (facets instanceof HTMLElement) initFacets(facets, { capabilities });
    const list = document.getElementById('ariel-list');
    if (list) initList(list, { capabilities });
    // After the list: its pill sits above the list body.
    if (list) initListActivity({ capabilities });
    registerReaderFeatures();
    const reader = document.getElementById('ariel-reader-body');
    if (reader) wireReader(reader);

    initEntries(document);
    initDashboard(document);
    initKeyboard(document);
    if (reader) wireReaderKeys(document, reader);
    initHeader({ capabilities });
  } catch (e) {
    console.error('Module initialization failed (non-fatal):', e);
  }

  initRouter({ dimensions: capabilities?.filter_dimensions ?? null });

  console.log('ARIEL Web Interface initialized');
}

// The page is ready once init() has settled, whatever it could draw
// (index.html declares the mark pending until then).
const start = () => void init().finally(() => markPageReady());
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', start);
} else {
  start();
}
