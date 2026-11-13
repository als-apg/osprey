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
 *     page that drew its error state is ready;
 *   - a page with no script at all carries `data-page-ready="true"` literally.
 *
 * `pending` lives in the markup, never in a call, because only the markup is
 * early enough: a script that said "wait for me" could run after the shared
 * start-up had already marked the page.
 *
 * Ready means "can be driven", not "has data". A test that needs data waits for
 * the data.
 *
 * @module page-ready
 */

/** The `<html>` attribute the mark lives in. */
export const PAGE_READY_ATTRIBUTE = 'data-page-ready';

/** The value a late page declares in its markup until it marks itself. */
const PENDING = 'pending';

/**
 * Mark the document ready to drive. Idempotent: a second call changes nothing.
 * @param {Document} [doc]
 */
export function markPageReady(doc = document) {
  const root = doc.documentElement;
  if (root.getAttribute(PAGE_READY_ATTRIBUTE) === 'true') return;
  root.setAttribute(PAGE_READY_ATTRIBUTE, 'true');
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
