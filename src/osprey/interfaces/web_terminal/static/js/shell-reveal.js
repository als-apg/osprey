// @ts-check
/* OSPREY Web Terminal — the loading shell's one reveal
 *
 * The hub loads in two visible states and one step between them. Until
 * start-up is over the page is `<html data-boot="loading">`: the sidebar is
 * drawn and inert, the workspace is under the loading veil (terminal.css,
 * "Loading shell"; shell-boot.js keeps the veil's words honest). Then
 * {@link revealShell} sets `data-boot="revealed"` and the finished page fades
 * in at once. Nothing in between is shown: no tile appears and then moves, no
 * row appears and then fills.
 *
 * The reveal waits for what would otherwise move in front of the operator
 * (the stored layout, the landing panel), within a budget. A backend that is
 * slow to answer costs the budget and no more; its tile then arrives the way
 * any later panel does.
 *
 * A page whose markup declares no `data-boot` (the session pop-out, a test
 * document) has no veil and every function here is a no-op for it.
 */

/** The `<html>` attribute the loading state lives in. */
export const BOOT_ATTRIBUTE = 'data-boot';

/** How long the reveal waits for the arrangement to settle. */
export const SETTLE_BUDGET_MS = 800;

/**
 * Take the result of a read shell-boot.js started before the modules arrived.
 *
 * The promise is the parsed JSON, shared by every taker until the reveal, so
 * each taker gets its own copy and none can change what the next one reads.
 * @param {string} path - the root-relative API path, unprefixed
 * @returns {Promise<any> | null} null when no such read was started, or the
 *   page is past its reveal
 */
export function takeEarlyRead(path) {
  const early = /** @type {any} */ (globalThis).__OSPREY_EARLY__;
  const read = early && Object.hasOwn(early, path) ? early[path] : null;
  if (!read) return null;
  return read.then((/** @type {any} */ body) => structuredClone(body));
}

/** Drop the early reads: from here on every read asks the server. */
function dropEarlyReads() {
  /** @type {any} */ (globalThis).__OSPREY_EARLY__ = null;
}

/**
 * A one-way latch: `opened` settles the first time `open` is called, and
 * stays settled. For a start-up condition several code paths can meet.
 * @returns {{ opened: Promise<void>, open: () => void }}
 */
export function latch() {
  /** @type {() => void} */
  let open = () => {};
  /** @type {Promise<void>} */
  const opened = new Promise((resolve) => {
    open = resolve;
  });
  return { opened, open: () => open() };
}

/**
 * Resolve once every promise has settled or the budget has run out, whichever
 * is first. Never rejects: a rejected promise counts as settled.
 * @param {Array<Promise<unknown>>} waits
 * @param {number} [budgetMs]
 * @returns {Promise<void>}
 */
export function settledWithin(waits, budgetMs = SETTLE_BUDGET_MS) {
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let timer;
  const budget = new Promise((resolve) => {
    timer = setTimeout(resolve, budgetMs);
  });
  return Promise.race([Promise.allSettled(waits), budget]).then(() => clearTimeout(timer));
}

/**
 * Reveal the page. Idempotent, and a no-op on a page that is not loading: one
 * that declared no `data-boot`, one already revealed (shell-boot.js's
 * fail-open), one that failed to load.
 * @param {Document} [doc]
 * @returns {boolean} whether this call revealed the page
 */
export function revealShell(doc = document) {
  dropEarlyReads();
  const root = doc.documentElement;
  if (root.getAttribute(BOOT_ATTRIBUTE) !== 'loading') return false;
  // The bar shell-boot.js filled as the files arrived ends full.
  const bar = /** @type {HTMLElement | null} */ (
    doc.querySelector('[data-boot-bar][data-progress]')
  );
  if (bar) {
    bar.style.setProperty('--boot-progress', '1');
    bar.setAttribute('aria-valuenow', '100');
  }
  root.setAttribute(BOOT_ATTRIBUTE, 'revealed');
  return true;
}
