// @ts-check
/* OSPREY Design System — the page-ready mark
 *
 * Every page OSPREY serves says once, in one place, that it can be driven:
 * `<html data-page-ready="true">`. A browser test, a docs recording or a
 * facility's own script waits for that attribute and for nothing else.
 *
 * This module is the one writer of the mark. A page reaches it by one of three
 * routes:
 *
 *   - the shared start-up: `initTheme()` (theme-manager.js) calls
 *     `markPageReadyOnceParsed()`, which marks the page once the document's
 *     `DOMContentLoaded` handlers have run;
 *   - late dressing: a page whose start-up runs past that point (it fetches,
 *     draws a first state, builds a dock) carries
 *     `<html data-page-ready="pending">` in its markup, and calls
 *     `markPageReady()` itself when it is dressed, on its failure path too: a
 *     page that drew its error state is ready; a pending page that throws an
 *     uncaught error, or leaves a rejection unhandled, before that call is
 *     marked `failed` by this module, and stays failed;
 *   - a page with no script at all carries `data-page-ready="true"` literally.
 *
 * `pending` lives in the markup, never in a call, because only the markup is
 * early enough: a script that said "wait for me" could run after the shared
 * start-up had already marked the page.
 *
 * Ready means "can be driven", not "has data". A test that needs data waits for
 * the data. `failed` means the page cannot be driven: a waiter stops there.
 *
 * @module page-ready
 */

/** The `<html>` attribute the mark lives in. */
export const PAGE_READY_ATTRIBUTE = 'data-page-ready';

/** The value a late page declares in its markup until it marks itself. */
const PENDING = 'pending';

/** The value the writer sets when a pending page threw before it marked itself. */
const FAILED = 'failed';

/**
 * Mark the document ready to drive. Idempotent, and an end state stands: a
 * second call, or a call on a page already marked failed, changes nothing.
 * @param {Document} [doc]
 */
export function markPageReady(doc = document) {
  const root = doc.documentElement;
  const value = root.getAttribute(PAGE_READY_ATTRIBUTE);
  if (value === 'true' || value === FAILED) return;
  root.setAttribute(PAGE_READY_ATTRIBUTE, 'true');
}

/**
 * Mark a pending page failed when its running script throws. The attribute is
 * the only state: two loaded copies of this module read and write the same
 * mark and agree. A page that is not pending is never attested.
 */
function attestFailure() {
  const root = document.documentElement;
  if (root.getAttribute(PAGE_READY_ATTRIBUTE) !== PENDING) return;
  root.setAttribute(PAGE_READY_ATTRIBUTE, FAILED);
}

// Neither listener captures: a resource that fails to load (a script, a
// stylesheet, an image) reports on its own element and does not bubble, so
// only an error thrown by running script, or a rejection nobody handled,
// reaches them.
if (typeof window !== 'undefined') {
  window.addEventListener('error', attestFailure);
  window.addEventListener('unhandledrejection', attestFailure);
}

/**
 * The shared start-up's rule: mark the document once its `DOMContentLoaded`
 * handlers have run, unless its markup declared late dressing (`pending`), in
 * which case the page marks itself.
 * @param {Document} [doc]
 */
export function markPageReadyOnceParsed(doc = document) {
  if (doc.documentElement.getAttribute(PAGE_READY_ATTRIBUTE) === PENDING) return;
  // A task queued after the dispatch runs after every listener of it, whether
  // this is called before the document is parsed or from one of those listeners.
  const mark = () => setTimeout(() => markPageReady(doc), 0);
  if (doc.readyState === 'loading') {
    doc.addEventListener('DOMContentLoaded', mark, { once: true });
  } else {
    mark();
  }
}
