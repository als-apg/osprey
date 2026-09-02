// @ts-check
/* OSPREY Web Terminal — bar layout sync.
 *
 * The client half of `/api/bar-items`. bar-host.js paints from what the server
 * rendered, synchronously and with no network in its import path; this module
 * is everything that happens AFTER that paint — reading the stored arrangement,
 * writing an operator's edit back, and keeping two tabs from silently
 * overwriting one another.
 *
 * FIRST PAINT NEVER WAITS. The boot `GET` is started at import time and
 * awaited by nobody. A fetch that hangs, 500s or never resolves leaves the
 * server-rendered bars exactly as they are — the reconcile it feeds is a
 * CORRECTION, not the render. That is also what makes the GET worth doing at
 * all: the server-side render drops `data-bar-options` (it emits only type,
 * adopted and follows), so a deployment that configured a UTC clock paints a
 * local one until this reconcile arrives with the options attached.
 *
 * A PUT ONLY EVER FOLLOWS AN EXPLICIT EDIT. Boot does not write. A visibility
 * re-GET does not write. A document this build repaired on the way in — a
 * missing `status_visible`, an option added in a later build — does not write:
 * a repair is this client's opinion, not the operator's, and writing it back
 * would let a stale tab publish its own defaults over a real preference.
 * `saveLayout()` refuses without `{edit: true}` for exactly that reason: the
 * flag is the caller stating a human did this.
 *
 * A RESET IS A DELETE. `resetLayout()` discards the operator's document and
 * takes back whatever the deployment says, rather than writing the deployment's
 * layout out under the operator's name — the two are different states from then
 * on, and only the first follows a later `web.bar_items` edit. It carries no
 * revision, so it has no conflict ladder, and it is the ONE verb the read-only
 * latch below does not block: it is the recovery that latch's own contract
 * points at.
 *
 * A DOCUMENT WE HAD TO DROP CONTENT FROM IS READ-ONLY (FR5). `normalize()`
 * reports `readonly` when an entry was lost — an unknown item type, a host this
 * build will not render it in, an unreadable schema version. Rendering what
 * survived is right; writing it back is not, because the write would DELETE
 * whatever this build could not understand. So `readonly` latches and every
 * later save is refused before it reaches the network. A document that
 * salvaged nothing at all is not even reconciled: blanking the bars over a
 * version we cannot read is strictly worse than leaving the server's paint up.
 *
 * TWO TABS. There is no broadcast — the protection is arithmetic. Every PUT
 * carries the revision this client is holding; the server answers 409 with the
 * document that actually won. The ladder is: adopt that document (so the
 * operator SEES the arrangement they are about to change), re-apply the edit on
 * top of it, retry exactly once, and if that also loses, stop and say so. One
 * retry, never a loop: a second conflict means another tab is actively editing,
 * and a client that kept retrying would be fighting a human. `visibilitychange`
 * closes the gap in the other direction — a tab returning to the foreground
 * re-GETs, so it stops holding a revision that has been dead for an hour.
 *
 * DEPLOYMENT CONTEXT IS SERVED, NOT INFERRED. `available()` in the catalog asks
 * whether this deployment offers an item, and a wrong answer costs something
 * either way: a false "unavailable" drops the item AND latches read-only for the
 * session, so saving dies silently, while a false "available" lets this client
 * put back an item the deployment cannot render. So the server evaluates all
 * three facts where it already knows them and stamps them on `<html>` as
 * `data-bar-context`; this module only parses them. The inference this replaced
 * read identity and the plan queue off the shells the server had rendered — so a
 * deployment that OFFERED an item but did not PLACE it read as unavailable — and
 * panel health off `PANELS`, the full shipped catalog, which says `ariel-status`
 * on every build including deployments serving no such panel.
 *
 * A page carrying no stamp at all is not this build's render, and it gets no
 * guess: every gated item is refused, which renders what survived and refuses to
 * write the document back. Assuming availability instead would write an
 * arrangement this deployment cannot show.
 *
 * NOT HERE: the edit UI (Phase 3), which is the only caller `saveLayout()` has,
 * and item disposal — bar-items.js watches the pool itself, so the shells a
 * reconcile parks are disposed without this module telling it to.
 */

import { withPrefix } from './api.js';
import { BAR_CATALOG } from './bar-catalog.js';
import { docOf, reconcile } from './bar-host.js';
import { normalize } from './bar-layout.js';
import { applyOverflow } from './bar-overflow.js';

/** @typedef {import('./bar-layout.js').BarLayout} BarLayout */
/** @typedef {import('./bar-layout.js').BarLayoutContext} BarLayoutContext */
/** @typedef {import('./bar-host.js').BarRoot} BarRoot */

/**
 * Why a save did not happen.
 * @typedef {'not-an-edit' | 'readonly' | 'conflict' | 'invalid' | 'unavailable' | 'network'} BarSyncReason
 */

/** The one endpoint this module speaks to. */
const ENDPOINT = '/api/bar-items';

/** What the operator is told when a save did not land. One action, one fact. */
const NOT_SAVED = 'Layout not saved';

/** The same, for a reset: the arrangement it would have discarded is still there. */
const NOT_RESET = 'Layout not reset';

/**
 * The other half of a reset going wrong, and the opposite fact: the removal
 * LANDED and the answer to it could not be read. Saying "not reset" here would
 * be backwards, and the arrangement still on screen is one that no longer
 * exists on the server.
 */
const RESET_UNREAD = 'Layout reset. Reload to see it.';

/** A failed save, with the reason a caller can branch on. */
export class BarSyncError extends Error {
  /**
   * @param {BarSyncReason} reason
   * @param {string} message
   */
  constructor(reason, message) {
    super(message);
    this.name = 'BarSyncError';
    /** @type {BarSyncReason} */
    this.reason = reason;
  }
}

/** The document this client is holding. @type {BarLayout | null} */
let current = null;

/** Latched by any document content was lost from. @type {boolean} */
let readonly = false;

/** Undo the visibility listener, if one is armed. @type {(() => void) | null} */
let stopVisibility = null;

/** Surfaces that render "Layout not saved". @type {Set<(text: string) => void>} */
const noticeListeners = new Set();

/* ---- state readers ---- */

/**
 * The layout document this client is holding, or null before the first GET
 * lands. The edit UI edits a copy of this.
 * @returns {BarLayout | null}
 */
export function currentLayout() {
  return current;
}

/**
 * Whether this client refuses to write. True once any document arrived that
 * this build had to drop content from — see FR5 in the module comment.
 * @returns {boolean}
 */
export function isLayoutReadonly() {
  return readonly;
}

/* ---- notices ---- */

/**
 * Render "Layout not saved" somewhere of your own. The edit UI registers here
 * so the message lands in its sheet, styled by that sheet's own stylesheet.
 *
 * This module paints nothing itself. It owns no stylesheet, and a surface it
 * invented would be one more thing to keep in the design system for a message
 * that already has a place to live. With no listener registered the message is
 * simply not shown.
 * @param {(text: string) => void} listener
 * @returns {() => void} unsubscribe
 */
export function onSyncNotice(listener) {
  noticeListeners.add(listener);
  return () => noticeListeners.delete(listener);
}

/**
 * Tell the operator something, through whatever surface is registered.
 * @param {string} text
 */
function notice(text) {
  for (const listener of Array.from(noticeListeners)) {
    try {
      listener(text);
    } catch (err) {
      console.error('[bar-sync] notice listener threw', err);
    }
  }
}

/* ---- deployment context ---- */

/** Where `root()` stamps what this deployment offers. */
const CONTEXT_ATTR = 'data-bar-context';

/** What a page with no stamp offers: nothing that has to be asked about. */
const NO_CONTEXT = Object.freeze(
  /** @type {BarLayoutContext} */ ({
    identityAvailable: false,
    blueskyAvailable: false,
    statusBarIds: [],
  })
);

/**
 * What this deployment offers, as the catalog's `available()` asks it —
 * `root()`'s own evaluation, read back off `<html>`. See the module comment for
 * why nothing here is inferred from the page.
 *
 * An absent or unreadable stamp answers "nothing is offered", which refuses
 * every gated item rather than claiming one this deployment may not have.
 *
 * Exported because it is THE source of deployment facts on this page: any
 * surface that asks `available()` — the edit sheet deciding which tiles to
 * offer, as much as the normalizer deciding what to keep — must ask the same
 * context, or the two disagree about one item and the operator's save dies as
 * `readonly` with no explanation.
 * @param {BarRoot} root
 * @returns {BarLayoutContext}
 */
export function deploymentContext(root) {
  const raw = docOf(root).documentElement?.getAttribute(CONTEXT_ATTR);
  if (!raw) {
    console.warn(`[bar-sync] no ${CONTEXT_ATTR} on this page; no item is treated as available`);
    return NO_CONTEXT;
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return NO_CONTEXT;
    return {
      identityAvailable: parsed.identityAvailable === true,
      blueskyAvailable: parsed.blueskyAvailable === true,
      statusBarIds: Array.isArray(parsed.statusBarIds)
        ? parsed.statusBarIds.filter(
            /** @param {unknown} id @returns {id is string} */
            (id) => typeof id === 'string' && id.length > 0
          )
        : [],
    };
  } catch {
    console.warn(`[bar-sync] ${CONTEXT_ATTR} is not readable JSON`, raw);
    return NO_CONTEXT;
  }
}

/* ---- adopting a document ---- */

/**
 * Take a served document as this client's own: normalize it, latch read-only if
 * anything was lost, render it, and re-run the overflow ladder — a fold decided
 * against the previous arrangement's widths is a stale fold.
 *
 * A document that salvaged NOTHING is latched but not rendered. The two cases
 * that produce it are an unreadable schema version and a document whose every
 * entry this build refused, and in both the server's own paint is the better
 * thing to leave on screen than an empty bar.
 * @param {unknown} raw
 * @param {BarRoot} root
 * @returns {BarLayout | null} the adopted document, or null when nothing was
 */
function adopt(raw, root) {
  const result = normalize(raw, BAR_CATALOG, deploymentContext(root));
  if (result.readonly) {
    readonly = true;
    console.warn('[bar-sync] the stored bar layout will not be written back', result.dropped);
  }
  const empty = result.layout.header.length === 0 && result.layout.status.length === 0;
  if (result.readonly && empty) return null;
  current = result.layout;
  reconcile(current, root);
  applyOverflow(root);
  return current;
}

/* ---- the wire ---- */

/**
 * One request to the layout endpoint.
 * @param {string} method
 * @param {unknown} [body]
 * @returns {Promise<Response>}
 */
function request(method, body) {
  /** @type {RequestInit} */
  const init = { method, headers: { Accept: 'application/json' } };
  if (body !== undefined) {
    init.headers = { ...init.headers, 'Content-Type': 'application/json' };
    init.body = JSON.stringify(body);
  }
  return fetch(withPrefix(ENDPOINT), init);
}

/**
 * The refusal a response carries. The route answers `{detail: {error, ...}}`,
 * which is FastAPI's shape for the ladder in `routes/bar_items.py`.
 * @param {number} status
 * @returns {BarSyncReason}
 */
function reasonFor(status) {
  if (status === 409) return 'conflict';
  if (status === 422) return 'invalid';
  if (status === 503) return 'unavailable';
  return 'network';
}

/**
 * Read a response body, tolerating one that is not JSON at all.
 * @param {Response} response
 * @returns {Promise<any>}
 */
async function bodyOf(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

/* ---- reading ---- */

/**
 * Ask the server what this operator's bars should be, and render the answer.
 *
 * Never throws and never rejects: this runs at boot and on every foregrounding,
 * and the correct outcome of a failed correction is the paint that is already
 * on screen.
 * @param {BarRoot} [root]
 * @returns {Promise<BarLayout | null>}
 */
export async function syncLayout(root = document) {
  try {
    const response = await request('GET');
    if (!response.ok) {
      console.warn(`[bar-sync] GET ${ENDPOINT} answered ${response.status}`);
      return null;
    }
    return adopt(await bodyOf(response), root);
  } catch (err) {
    console.warn('[bar-sync] could not read the bar layout', err);
    return null;
  }
}

/* ---- writing ---- */

/**
 * Save an arrangement the operator just made.
 *
 * `edit` is not a convenience flag — it is the whole gate. Every other caller
 * of a layout document in this module (boot, the visibility re-GET, the adopt
 * a 409 forces) reaches the same data and must never write, so the ONE thing
 * that distinguishes a human's arrangement from a client's own opinion is
 * stated at the call site.
 *
 * The revision sent is the one this client is HOLDING, not one the caller
 * supplies: the point of the conditional write is that a tab cannot save
 * against a revision it never saw.
 *
 * @param {unknown} next - the arrangement to store; normalized before it is sent
 * @param {{edit?: boolean, root?: BarRoot}} [options]
 * @returns {Promise<BarLayout>} the document as persisted
 * @throws {BarSyncError} `not-an-edit`, `readonly`, `conflict`, `invalid`,
 *   `unavailable` or `network` — nothing was saved in any of them
 */
export async function saveLayout(next, options = {}) {
  const root = options.root ?? document;
  if (options.edit !== true) {
    throw new BarSyncError('not-an-edit', 'A bar layout is only saved from an operator edit');
  }
  if (readonly) throw refuse('readonly');

  const wanted = normalize(next, BAR_CATALOG, deploymentContext(root));
  if (wanted.readonly) throw refuse('readonly');

  const first = await attempt(wanted.layout, root);
  if (first.saved) return first.saved;
  if (first.reason !== 'conflict') throw refuse(first.reason);

  // The conflict ladder: adopt what won, put the operator's edit back on top of
  // it, and try once more. `adopt()` has already moved `current.rev` on, so the
  // retry is conditional on the revision that beat us rather than on ours.
  if (readonly) throw refuse('readonly');
  const retry = await attempt(wanted.layout, root);
  if (retry.saved) return retry.saved;
  throw refuse(retry.reason);
}

/**
 * One conditional write. A 409 adopts the document it carries before returning,
 * so the caller's decision about retrying is made against what the server holds.
 * @param {BarLayout} layout
 * @param {BarRoot} root
 * @returns {Promise<{saved: BarLayout | null, reason: BarSyncReason}>}
 */
async function attempt(layout, root) {
  /** @type {Response} */
  let response;
  try {
    response = await request('PUT', { ...layout, rev: current ? current.rev : layout.rev });
  } catch (err) {
    console.warn('[bar-sync] could not save the bar layout', err);
    return { saved: null, reason: 'network' };
  }
  const body = await bodyOf(response);
  if (response.ok) return { saved: adopt(body, root), reason: 'network' };
  const reason = reasonFor(response.status);
  if (reason === 'conflict') adopt(body?.detail?.layout, root);
  else console.warn(`[bar-sync] PUT ${ENDPOINT} answered ${response.status}`, body?.detail);
  return { saved: null, reason };
}

/**
 * Say a save did not happen, once, and build the error for it. `not-an-edit`
 * never reaches here: a caller that forgot the flag is a bug in that caller,
 * not something to tell the operator about.
 * @param {BarSyncReason} reason
 * @returns {BarSyncError}
 */
function refuse(reason) {
  notice(NOT_SAVED);
  return new BarSyncError(reason, NOT_SAVED);
}

/* ---- resetting ---- */

/**
 * Discard this operator's arrangement and go back to the deployment's own.
 *
 * A DELETE, not a write of the deployment's layout under the operator's name:
 * "saved nothing" and "saved a document that happens to match the deployment"
 * behave differently from here on, and only the first follows `web.bar_items`
 * when a deployment next edits it. The route says the same thing from its end.
 *
 * THIS IS THE WAY OUT OF READ-ONLY, so it is the one verb the latch does not
 * block. The latch exists to stop this client WRITING BACK a document it had to
 * drop content from — that write would delete whatever this build could not
 * understand. A reset deletes that document deliberately and asks the server
 * what the deployment says instead, which is the recovery `bar-layout.js`'s own
 * contract names ("issue ZERO PUTs until the user explicitly resets"). Refusing
 * it here would leave an operator whose stored layout this build cannot fully
 * read with no way back at all. The latch is cleared only once the server has
 * confirmed the removal, and what comes back is judged on its own merits — a
 * deployment default this build cannot fully render latches again.
 *
 * There is no revision and so no conflict ladder: a reset is unconditional by
 * construction, which is why it is its own verb rather than a `rev`-less PUT.
 *
 * @param {{edit?: boolean, root?: BarRoot}} [options]
 * @returns {Promise<BarLayout>} the deployment default, as adopted
 * @throws {BarSyncError} `not-an-edit`, `unavailable` or `network`, in all of
 *   which the operator's arrangement is untouched — or `invalid`, which is the
 *   one case where the removal DID land and only the answer to it could not be
 *   read; the latch stays set there so the discarded arrangement cannot be
 *   written back
 */
export async function resetLayout(options = {}) {
  const root = options.root ?? document;
  if (options.edit !== true) {
    throw new BarSyncError('not-an-edit', 'A bar layout is only reset from an operator edit');
  }

  /** @type {Response} */
  let response;
  try {
    response = await request('DELETE');
  } catch (err) {
    console.warn('[bar-sync] could not reset the bar layout', err);
    throw refuseReset('network');
  }
  const body = await bodyOf(response);
  if (!response.ok) {
    console.warn(`[bar-sync] DELETE ${ENDPOINT} answered ${response.status}`, body?.detail);
    throw refuseReset(reasonFor(response.status));
  }

  // Cleared here so `adopt()` judges what came back on its own merits — it only
  // ever SETS the latch — and set again below if nothing readable arrived. The
  // latch must never be left down over the document that was just deleted: the
  // next edit would PUT it straight back and resurrect what the operator threw
  // away.
  readonly = false;
  // The route answers with the deployment default in the GET envelope; a body
  // that did not survive the parse is re-read rather than guessed at, because
  // the removal has already happened and the client must not keep rendering the
  // arrangement it just discarded.
  const adopted = body === null ? await syncLayout(root) : adopt(body, root);
  if (!adopted) {
    readonly = true;
    notice(RESET_UNREAD);
    throw new BarSyncError('invalid', RESET_UNREAD);
  }
  return adopted;
}

/**
 * Say a reset did not happen. Its own sentence: after a failed reset the
 * operator's arrangement is still in place, which "Layout not saved" would
 * describe backwards.
 * @param {BarSyncReason} reason
 * @returns {BarSyncError}
 */
function refuseReset(reason) {
  notice(NOT_RESET);
  return new BarSyncError(reason, NOT_RESET);
}

/* ---- boot ---- */

/**
 * Start the boot GET and keep the client's document fresh.
 *
 * The GET is deliberately not awaited by the caller — see the module comment.
 * The visibility listener is the whole of the freshness story: a tab that has
 * been in the background for an hour is holding a revision that may be long
 * dead, and re-reading on the way back is what turns its next save from a
 * guaranteed 409 into an ordinary write.
 * @param {BarRoot} [root]
 * @returns {() => void} stop
 */
export function initBarSync(root = document) {
  stopBarSync();
  const doc = docOf(root);
  const onVisible = () => {
    if (doc.visibilityState === 'visible') void syncLayout(root);
  };
  doc.addEventListener('visibilitychange', onVisible);
  stopVisibility = () => doc.removeEventListener('visibilitychange', onVisible);
  void syncLayout(root);
  return stopBarSync;
}

/** Stop keeping the document fresh. The teardown entry point. */
export function stopBarSync() {
  if (stopVisibility) stopVisibility();
  stopVisibility = null;
}

// The correction pass, started at import time and awaited by nobody.
if (typeof document !== 'undefined') initBarSync();
