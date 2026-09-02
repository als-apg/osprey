/* OSPREY Web Terminal — bar item hosts.
 *
 * The header and the status bar are ITEM HOSTS: each renders one ordered list
 * of `.bar-item` shells, and a hidden `#bar-item-pool` holds every shell the
 * current layout does not name. This module owns the movement between those
 * three containers and nothing else — it is the one place that decides where a
 * shell lives.
 *
 * Three properties make the difference between a host and a re-render, and all
 * three are why `reconcile()` is a keyed pass over element identities rather
 * than a `replaceChildren`:
 *
 *   Adopted nodes are MOVED, never rebuilt. Seven chrome nodes (`#activity-strip`,
 *   `#docs-link`, `#command-palette-btn`, the logo, the identity block, the
 *   display menu, the status dots) are rendered server-side and carry live
 *   state that no builder can reconstruct: `#activity-strip` is an `aria-live`
 *   region that `activity-history.js` mutates into its own trigger, and it
 *   self-boots from its own module before any layout arrives. Rebuilding it
 *   would leave a dead strip with no error anywhere. So a shell whose SSR body
 *   already exists is stamped `data-bar-adopted` at hydration and its subtree
 *   is never touched again — `document.getElementById('activity-strip')`
 *   returns the same node object across any number of pool round-trips.
 *
 *   Focus and selection survive a move. Re-parenting a node blurs it, so a
 *   reconcile that lands while the operator is typing in the search item would
 *   silently eat the caret. The pass brackets every move with a capture and a
 *   restore of `document.activeElement` plus, for text inputs, the selection
 *   range and direction.
 *
 *   Popovers close before their item moves. A popover is positioned against a
 *   button that is about to be somewhere else; the seam is a registry keyed by
 *   shell, so the popover implementations (which arrive with the items) hand
 *   this module a closer and never have to know about the reconcile.
 *
 * Two consequences of placement are owned here as well, because both are facts
 * about where an item currently sits and nothing else can know them:
 *
 *   An item that hides its own body collapses its shell. Items reveal and hide
 *   their bodies on their own schedule (the docs link waits for a configured
 *   URL, a health dot waits for its panel's config), and a shell left standing
 *   around a hidden body would spend the bar's gap on nothing. One
 *   `MutationObserver` per shell mirrors the body's `hidden` onto the shell,
 *   which is what `.bar-item[hidden] { display: none !important }` collapses.
 *
 *   Consecutive `align: 'baseline'` items share a baseline. They are wrapped in
 *   a generated `.bar-baseline-run` so the wordmark and the identity text sit on
 *   one line; the run is a fact about ADJACENCY, so it forms and dissolves as
 *   items move. The same goes for `data-follows`, which carries the preceding
 *   item's type and is what the identity middot keys off instead of a sibling
 *   combinator that would keep painting after either item moved away.
 *
 * First paint does not wait for the network. Hydration runs SYNCHRONOUSLY at
 * import time over whatever the server rendered: `GET /api/bar-items` is a
 * reconcile input, never a first-paint dependency, and there is deliberately
 * no fetch anywhere in this module's import path.
 *
 * Non-adopted bodies come from the type's single builder, registered here by
 * the item modules. One builder per type means the same code runs at first
 * paint and when the operator drags the item in — there is no second, simpler
 * "initial render" path to drift out of step.
 *
 * DOM contract (server-rendered):
 *   host      `[data-bar-host="header"]`, falling back to `.header-actions`;
 *             `[data-bar-host="status"]`, falling back to `.status-bar`
 *   shell     `.bar-item[data-bar-item="<type>"]`, a direct child of a host, of
 *             a `.bar-baseline-run` in a host, or of the pool; optional
 *             `data-bar-key` and `data-bar-options`
 *   pool      `#bar-item-pool`, hidden, holding parked SHELLS (not bare nodes)
 *
 * Deliberately NOT here, so the seams are named rather than assumed: the
 * overflow ladder and the concrete item builders. The ladder lives in
 * bar-overflow.js and folds by MOVING — `topLevelManaged()`/`managedShells()`
 * are what it reads, `parkShell()`/`restoreShell()` are what it moves with.
 */

import { BAR_HOSTS, barItemType, defaultOptions, densityForHost } from './bar-catalog.js';

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */
/** @typedef {import('./bar-catalog.js').BarDensity} BarDensity */
/** @typedef {import('./bar-catalog.js').BarItemOptions} BarItemOptions */
/** @typedef {import('./bar-layout.js').BarLayout} BarLayout */

/** Where hosts, shells and the pool are looked up. A document, or a subtree. */
/** @typedef {Document | Element} BarRoot */

/**
 * What a builder is handed. `shell` is the element the body is appended to —
 * passed so a builder can stamp its own attributes, never so it can move it.
 * @typedef {object} BarBuildContext
 * @property {string} type
 * @property {string} key - stable within one layout; survives host moves
 * @property {BarDensity} density - the density of the host it is being built for
 * @property {BarItemOptions} options - complete, merged over the type's defaults
 * @property {HTMLElement} shell
 */

/**
 * The single body builder for one type. Returning `null` is legal and means
 * the type renders as an empty shell (spacing items do exactly that).
 * @typedef {(ctx: BarBuildContext) => Node | null} BarItemBuilder
 */

/**
 * A captured focus position. `start` is null for anything that is not a text
 * input, which is also the case when the browser refuses selection access.
 * @typedef {object} BarFocusState
 * @property {HTMLElement} node
 * @property {number | null} start
 * @property {number | null} end
 * @property {'forward' | 'backward' | 'none' | null} direction
 */

const SHELL_CLASS = 'bar-item';
const RUN_CLASS = 'bar-baseline-run';
const POOL_ID = 'bar-item-pool';
const POOL_SELECTOR = '#bar-item-pool';
const SHELL_SELECTOR = '.bar-item[data-bar-item]';

/**
 * Host lookup, most specific first. The explicit `data-bar-host` attribute is
 * the contract; the class fallbacks keep a host resolvable while the templates
 * still spell it the old way.
 * @type {Readonly<Record<BarHost, readonly string[]>>}
 */
const HOST_SELECTORS = Object.freeze({
  header: Object.freeze(['[data-bar-host="header"]', '.header-actions']),
  status: Object.freeze(['[data-bar-host="status"]', '.status-bar']),
});

/** Input types that carry a selection range worth restoring. */
const SELECTABLE_INPUT_TYPES = new Set(['text', 'search', 'url', 'tel', 'password']);

/** Every known shell, by key. Rebuilt by `hydrate()`, extended by `reconcile()`. */
/** @type {Map<string, HTMLElement>} */
const shells = new Map();

/** One builder per type. Module-scoped: registered once, at item-module import. */
/** @type {Map<string, BarItemBuilder>} */
const builders = new Map();

/** Closers to run before a shell moves, by shell. */
/** @type {Map<HTMLElement, Set<() => void>>} */
const popovers = new Map();

/** The `hidden` mirror watching each shell's body. One per shell. */
/** @type {Map<HTMLElement, MutationObserver>} */
const mirrors = new Map();

/** The root the last `hydrate()`/`reconcile()` ran against. @type {BarRoot | null} */
let activeRoot = typeof document === 'undefined' ? null : document;

/* ---- lookup ---- */

/**
 * The owning document of a root, which may itself be the document.
 *
 * Exported because every module in the bar stack needs it: a root is either the
 * document or an element inside one, and each of them has to reach the owning
 * document to read `<html>` attributes, create nodes or bind listeners. One
 * copy here, beside the `BarRoot` type it narrows, keeps the coercion — and the
 * cast that expresses it — from being restated per module.
 * @param {BarRoot} root
 * @returns {Document}
 */
export function docOf(root) {
  const owner = /** @type {Element} */ (root).ownerDocument;
  return owner ?? /** @type {Document} */ (/** @type {unknown} */ (root));
}

/**
 * The container for one host, or null when this deployment does not render it.
 * @param {BarHost} host
 * @param {BarRoot} [root]
 * @returns {HTMLElement | null}
 */
export function hostElement(host, root = activeRoot ?? document) {
  for (const selector of HOST_SELECTORS[host] ?? []) {
    const found = root.querySelector(selector);
    if (found) return /** @type {HTMLElement} */ (found);
  }
  return null;
}

/**
 * The pool, if it has been rendered. Lookup only — see `ensurePool()` for the
 * parking path, which must never fail to find somewhere safe to put a node.
 * @param {BarRoot} [root]
 * @returns {HTMLElement | null}
 */
export function poolElement(root = activeRoot ?? document) {
  return /** @type {HTMLElement | null} */ (root.querySelector(POOL_SELECTOR));
}

/**
 * The pool, created if absent. Parking must never destroy adopted chrome, so a
 * deployment whose template forgot the pool still gets somewhere to park.
 * @param {BarRoot} root
 * @returns {HTMLElement}
 */
function ensurePool(root) {
  const existing = poolElement(root);
  if (existing) return existing;
  const doc = docOf(root);
  const pool = doc.createElement('div');
  pool.id = POOL_ID;
  pool.hidden = true;
  (doc.body ?? doc.documentElement).appendChild(pool);
  return pool;
}

/**
 * Whether a node is in a live bar rather than parked in the pool. The tour, the
 * ladder and every id-based lookup ask this before treating a node as visible:
 * pooled chrome still resolves by id, it is simply not on screen.
 * @param {Element | null | undefined} node
 * @returns {boolean}
 */
export function isLive(node) {
  if (!node || typeof node.closest !== 'function') return false;
  return !node.closest(POOL_SELECTOR);
}

/**
 * The shell for a layout key, if one exists.
 * @param {string} key
 * @returns {HTMLElement | null}
 */
export function shellForKey(key) {
  return shells.get(key) ?? null;
}

/* ---- builders ---- */

/**
 * Register the single body builder for a type. Registering fills any already
 * hydrated shell of that type whose body is still empty, so an item module
 * imported after this one still paints without waiting for a reconcile.
 * @param {string} type
 * @param {BarItemBuilder} builder
 * @returns {() => void} unregister
 */
export function registerItemBuilder(type, builder) {
  builders.set(type, builder);
  fillUnbuilt(type);
  return () => {
    if (builders.get(type) === builder) builders.delete(type);
  };
}

/**
 * Whether a type can currently build its own body.
 * @param {string} type
 * @returns {boolean}
 */
export function hasItemBuilder(type) {
  return builders.has(type);
}

/**
 * Build the bodies of placed shells of one type that have none yet.
 * @param {string} type
 */
function fillUnbuilt(type) {
  for (const shell of shells.values()) {
    if (shell.dataset.barItem !== type) continue;
    const density = shell.dataset.barDensity;
    if (!density) continue;
    buildBody(shell, type, readOptions(shell, type), /** @type {BarDensity} */ (density));
    armMirror(shell);
  }
}

/* ---- popovers ---- */

/**
 * Hand this module a way to close a popover an item owns, so the reconcile can
 * close it before the item moves out from under it. The popover implementations
 * arrive with their items; this is the only thing they owe the host.
 * @param {HTMLElement} shell
 * @param {() => void} close
 * @returns {() => void} unregister
 */
export function registerBarPopover(shell, close) {
  let closers = popovers.get(shell);
  if (!closers) {
    closers = new Set();
    popovers.set(shell, closers);
  }
  closers.add(close);
  return () => {
    const set = popovers.get(shell);
    if (!set) return;
    set.delete(close);
    if (set.size === 0) popovers.delete(shell);
  };
}

/**
 * Run every closer registered for one shell. A throwing closer must not stop
 * the reconcile: the move is the important half.
 * @param {HTMLElement} shell
 */
function closePopoversFor(shell) {
  const closers = popovers.get(shell);
  if (!closers) return;
  for (const close of Array.from(closers)) {
    try {
      close();
    } catch (err) {
      console.error('[bar-host] popover closer threw', err);
    }
  }
}

/** Close every registered bar popover (teardown, navigation, edit mode). */
export function closeBarPopovers() {
  for (const shell of Array.from(popovers.keys())) closePopoversFor(shell);
}

/* ---- the hidden mirror ---- */

/**
 * Copy the body's `hidden` onto its shell. Without this the shell keeps its box
 * — and the bar's gap on either side of it — around a body that has hidden
 * itself, which is how a deployment with no docs URL would still pay for a docs
 * item. `.bar-item[hidden]` in bars.css is the other half.
 *
 * A shell holding several bodies collapses only once every one of them is
 * hidden: one live body is reason enough to keep the box.
 * @param {HTMLElement} shell
 */
function syncHidden(shell) {
  const bodies = Array.from(shell.children);
  if (bodies.length === 0) return;
  shell.hidden = bodies.every((body) => body.hasAttribute('hidden'));
}

/**
 * Watch one shell's bodies for `hidden` and mirror it. An observer watches
 * NODES, not a position, so the shell's single observer is re-armed whenever a
 * body is rebuilt or the shell moves between a host and the pool — both can
 * hand the shell bodies the old observer is not watching. Re-arming applies the
 * current state first, so the records a `disconnect()` drops carry nothing that
 * is then lost.
 *
 * An empty shell (the spacing items, and any type whose builder has not arrived
 * yet) has nothing to mirror and keeps whatever visibility it has.
 * @param {HTMLElement} shell
 */
function armMirror(shell) {
  mirrors.get(shell)?.disconnect();
  mirrors.delete(shell);
  const bodies = Array.from(shell.children);
  if (bodies.length === 0) return;
  syncHidden(shell);
  const Observer = shell.ownerDocument.defaultView?.MutationObserver;
  if (!Observer) return;
  const observer = new Observer(() => syncHidden(shell));
  const watch = { attributes: true, attributeFilter: ['hidden'] };
  for (const body of bodies) observer.observe(body, watch);
  mirrors.set(shell, observer);
}

/** Drop every mirror, for a re-hydration against a fresh DOM. */
function disconnectMirrors() {
  for (const observer of mirrors.values()) observer.disconnect();
  mirrors.clear();
}

/* ---- shells ---- */

/**
 * @param {Element} node
 * @returns {boolean}
 */
function isShell(node) {
  return (
    node.classList.contains(SHELL_CLASS) && !!(/** @type {HTMLElement} */ (node).dataset.barItem)
  );
}

/**
 * The layout key for the n-th item of a type. First occurrence keys on the bare
 * type, so the common (singleton) case reads as `data-bar-key="activity"`.
 * @param {string} type
 * @param {Map<string, number>} counts
 * @returns {string}
 */
function nextKey(type, counts) {
  const n = counts.get(type) ?? 0;
  counts.set(type, n + 1);
  return n === 0 ? type : `${type}#${n}`;
}

/**
 * A shell's options: the type's defaults, with anything the server stamped in
 * `data-bar-options` merged over them.
 * @param {HTMLElement} shell
 * @param {string} type
 * @returns {Record<string, string | number | boolean>}
 */
function readOptions(shell, type) {
  const merged = defaultOptions(type);
  const raw = shell.dataset.barOptions;
  if (!raw) return merged;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') Object.assign(merged, parsed);
  } catch {
    // Unreadable stamp: the type's declared defaults render instead.
  }
  return merged;
}

/**
 * Stamp everything about a shell that follows from its type, its options and
 * where it currently lives. `density` is absent for a parked shell — the pool
 * has no density, and clearing it is what makes an unplaced shell inert.
 * @param {HTMLElement} shell
 * @param {string} type
 * @param {BarItemOptions} options
 * @param {BarDensity | null} density
 */
function stampShell(shell, type, options, density) {
  shell.classList.add(SHELL_CLASS);
  shell.dataset.barItem = type;
  const entry = barItemType(type);
  if (entry) shell.dataset.barAlign = entry.align;
  if (density) shell.dataset.barDensity = density;
  else delete shell.dataset.barDensity;
  if (Object.keys(options).length > 0) shell.dataset.barOptions = JSON.stringify(options);
  applyFlexHint(shell, type, options);
}

/**
 * Stamp the catalog's flex hint on the shell. This is what makes spacing
 * back-pressure CSS-owned: gaps and spaces shrink through the `flex-shrink`
 * declared here, continuously and ahead of every JS ladder rung, so a spacing
 * item can never clip the chrome next to it.
 * @param {HTMLElement} shell
 * @param {string} type
 * @param {BarItemOptions} options
 */
function applyFlexHint(shell, type, options) {
  const entry = barItemType(type);
  const hint = entry ? entry.flex(options) : null;
  shell.style.setProperty('flex', hint ? hint.flex : '');
  shell.style.setProperty('min-width', hint?.minWidth ?? '');
}

/**
 * Build (or rebuild) a non-adopted shell's body from its type's builder.
 * Adopted chrome is never rebuilt; a type with no builder yet is left empty and
 * marked, so registering the builder later fills it rather than needing a
 * reconcile to notice.
 * @param {HTMLElement} shell
 * @param {string} type
 * @param {BarItemOptions} options
 * @param {BarDensity} density
 */
function buildBody(shell, type, options, density) {
  if (shell.dataset.barAdopted === 'true') return;
  const builder = builders.get(type);
  if (!builder) {
    shell.dataset.barUnbuilt = 'true';
    return;
  }
  const signature = JSON.stringify([density, options]);
  if (shell.dataset.barBuilt === signature) return;
  const key = shell.dataset.barKey ?? type;
  /** @type {Node | null} */
  let body = null;
  try {
    body = builder({ type, key, density, options, shell });
  } catch (err) {
    console.error(`[bar-host] builder for "${type}" threw`, err);
    return;
  }
  closePopoversFor(shell);
  shell.replaceChildren();
  if (body) shell.appendChild(body);
  shell.dataset.barBuilt = signature;
  delete shell.dataset.barUnbuilt;
}

/* ---- hydration ---- */

/**
 * Index and stamp every server-rendered shell. Runs synchronously at import
 * time and is the whole of first paint: no fetch, no await, no layout document
 * needed. Re-running it re-reads the DOM from scratch, which is also what makes
 * it the per-test entry point.
 * @param {BarRoot | null} [root]
 */
export function hydrate(root = typeof document === 'undefined' ? null : document) {
  if (!root) return;
  activeRoot = root;
  shells.clear();
  popovers.clear();
  disconnectMirrors();
  /** @type {Map<string, number>} */
  const counts = new Map();
  for (const host of BAR_HOSTS) {
    const container = hostElement(host, root);
    if (container) adoptAll(container, densityForHost(host), counts);
  }
  const pool = poolElement(root);
  if (pool) adoptAll(pool, null, counts);
}

/**
 * Index every shell in one container.
 * @param {HTMLElement} container
 * @param {BarDensity | null} density
 * @param {Map<string, number>} counts
 */
function adoptAll(container, density, counts) {
  for (const node of Array.from(container.querySelectorAll(SHELL_SELECTOR))) {
    adoptShell(/** @type {HTMLElement} */ (node), density, counts);
  }
}

/**
 * Take one server-rendered shell under management. A shell that already has an
 * element body is ADOPTED CHROME: its subtree is off-limits from here on.
 * @param {HTMLElement} shell
 * @param {BarDensity | null} density
 * @param {Map<string, number>} counts
 */
function adoptShell(shell, density, counts) {
  const type = shell.dataset.barItem;
  if (!type) return;
  const fallbackKey = nextKey(type, counts);
  const key = shell.dataset.barKey || fallbackKey;
  shell.dataset.barKey = key;
  if (shell.firstElementChild) shell.dataset.barAdopted = 'true';
  shells.set(key, shell);
  const options = readOptions(shell, type);
  stampShell(shell, type, options, density);
  if (density) buildBody(shell, type, options, density);
  armMirror(shell);
}

/* ---- reconcile ---- */

/**
 * Place every item the layout names, in order, in its host; park every shell it
 * does not name in the pool. Nodes MOVE — nothing here rebuilds an adopted
 * body, and focus, selection and open popovers are all handled around the
 * moves rather than after them.
 * @param {BarLayout | null | undefined} layout
 * @param {BarRoot} [root]
 */
export function reconcile(layout, root = activeRoot ?? document) {
  if (!layout || !root) return;
  activeRoot = root;
  const focus = captureFocus(root);
  /** @type {Map<string, number>} */
  const counts = new Map();
  // Keys are assigned over the whole document, so a repeated type keeps a
  // stable identity when the operator moves one of them between hosts.
  const plans = BAR_HOSTS.map((host) => ({ host, items: planHost(layout, host, counts) }));
  for (const { host, items } of plans) {
    const container = hostElement(host, root);
    if (container) placeItems(container, host, items, root);
  }
  applyBarVisibility(layout, root);
  restoreFocus(focus);
}

/**
 * One host's placement plan. Entries this build cannot render are dropped
 * silently — `bar-layout.js` already normalizes, and a stale document from a
 * newer deployment is a normal thing to be handed, not an error.
 * @param {BarLayout} layout
 * @param {BarHost} host
 * @param {Map<string, number>} counts
 * @returns {{key: string, type: string, options: BarItemOptions}[]}
 */
function planHost(layout, host, counts) {
  const raw = /** @type {unknown} */ (layout[host]);
  const list = Array.isArray(raw) ? raw : [];
  /** @type {{key: string, type: string, options: BarItemOptions}[]} */
  const plan = [];
  for (const entry of list) {
    const type = entry && typeof entry.type === 'string' ? entry.type : '';
    if (!type || !barItemType(type)) continue;
    const options = defaultOptions(type);
    if (entry.options && typeof entry.options === 'object') Object.assign(options, entry.options);
    plan.push({ key: nextKey(type, counts), type, options });
  }
  return plan;
}

/**
 * Whether a node is a generated baseline run.
 * @param {Element | null} node
 * @returns {boolean}
 */
function isRun(node) {
  return !!node && node.classList.contains(RUN_CLASS);
}

/**
 * The children of a container this pass may move: item shells and the run
 * wrappers holding them. Anything else a template mounts beside the items is
 * skipped rather than adopted, so it keeps its place.
 * @param {Element} container
 * @returns {HTMLElement[]}
 */
export function topLevelManaged(container) {
  return Array.from(container.children)
    .filter((node) => isShell(node) || isRun(node))
    .map((node) => /** @type {HTMLElement} */ (node));
}

/**
 * Every shell a container currently holds, whether directly or inside one of
 * its runs. This is the set the placement pass parks from.
 * @param {Element} container
 * @returns {HTMLElement[]}
 */
export function managedShells(container) {
  /** @type {HTMLElement[]} */
  const found = [];
  for (const node of topLevelManaged(container)) {
    if (isShell(node)) found.push(node);
    else for (const child of topLevelManaged(node)) if (isShell(child)) found.push(child);
  }
  return found;
}

/**
 * Put `nodes` in this order inside `container`, moving ONLY what is out of
 * position — a shell that is already where the layout wants it keeps its
 * popover open and, more to the point, is never re-parented. Anything the
 * container holds beyond `nodes` is left where it is for the caller to park.
 * @param {HTMLElement} container
 * @param {readonly HTMLElement[]} nodes - the final order
 */
function placeSequence(container, nodes) {
  const existing = topLevelManaged(container);
  /** @type {Set<HTMLElement>} */
  const placed = new Set();
  let index = 0;
  for (const node of nodes) {
    // Skip what this pass already placed (it moved from further down the
    // container) and what it moved out entirely (into a run wrapper).
    while (
      index < existing.length &&
      (placed.has(existing[index]) || existing[index].parentElement !== container)
    ) {
      index += 1;
    }
    const at = index < existing.length ? existing[index] : null;
    placed.add(node);
    if (node === at) {
      index += 1;
      continue;
    }
    closePopoversWithin(node);
    container.insertBefore(node, at);
  }
}

/**
 * Close the popovers of everything a move takes with it — a run wrapper carries
 * its items along, and their popovers are anchored to buttons that are about to
 * be somewhere else.
 * @param {HTMLElement} node
 */
function closePopoversWithin(node) {
  if (!isRun(node)) {
    closePopoversFor(node);
    return;
  }
  for (const child of Array.from(node.children)) {
    if (isShell(child)) closePopoversFor(/** @type {HTMLElement} */ (child));
  }
}

/**
 * Keyed placement: resolve every planned item to its shell, group consecutive
 * baseline items into their run wrappers, place the resulting top-level
 * sequence, then park every shell this host no longer names.
 * @param {HTMLElement} container
 * @param {BarHost} host
 * @param {{key: string, type: string, options: BarItemOptions}[]} items
 * @param {BarRoot} root
 */
function placeItems(container, host, items, root) {
  const density = densityForHost(host);
  const present = managedShells(container);
  /** @type {HTMLElement[]} */
  const planned = [];
  for (const item of items) {
    const shell = ensureShell(item, root);
    stampShell(shell, item.type, item.options, density);
    buildBody(shell, item.type, item.options, density);
    armMirror(shell);
    planned.push(shell);
  }
  stampFollows(items, planned);
  placeSequence(container, topLevelNodes(container, items, planned, root));
  const keep = new Set(planned);
  for (const shell of present) if (!keep.has(shell)) parkShell(shell, root);
  pruneRuns(container);
}

/**
 * The top-level sequence for one host: a bare shell for every item that stands
 * alone, and one `.bar-baseline-run` wrapper per maximal run of two or more
 * consecutive `align: 'baseline'` items. A run of one is not a run — which is
 * how the wrapper dissolves the moment the operator moves identity away from
 * the logo, leaving `pruneRuns()` an empty wrapper to drop.
 * @param {HTMLElement} container
 * @param {{type: string}[]} items
 * @param {readonly HTMLElement[]} planned - one shell per item, in order
 * @param {BarRoot} root
 * @returns {HTMLElement[]}
 */
function topLevelNodes(container, items, planned, root) {
  /** @type {HTMLElement[]} */
  const nodes = [];
  /** @type {Set<HTMLElement>} */
  const used = new Set();
  let index = 0;
  while (index < items.length) {
    let end = index;
    while (end < items.length && barItemType(items[end].type)?.align === 'baseline') end += 1;
    if (end - index < 2) {
      nodes.push(planned[index]);
      index += 1;
      continue;
    }
    const group = planned.slice(index, end);
    const wrapper = runWrapper(group, container, used, root);
    placeSequence(wrapper, group);
    nodes.push(wrapper);
    index = end;
  }
  return nodes;
}

/**
 * The wrapper one run renders into: the one its items already sit in when the
 * run is unchanged (so nothing inside it moves), a fresh one otherwise.
 * @param {readonly HTMLElement[]} group
 * @param {HTMLElement} container
 * @param {Set<HTMLElement>} used - wrappers already claimed by an earlier run
 * @param {BarRoot} root
 * @returns {HTMLElement}
 */
function runWrapper(group, container, used, root) {
  for (const shell of group) {
    const parent = shell.parentElement;
    if (isRun(parent) && parent?.parentElement === container && !used.has(parent)) {
      used.add(parent);
      return parent;
    }
  }
  const wrapper = docOf(root).createElement('div');
  wrapper.className = RUN_CLASS;
  used.add(wrapper);
  return wrapper;
}

/** Drop the wrappers this pass emptied. @param {HTMLElement} container */
function pruneRuns(container) {
  for (const node of topLevelManaged(container)) {
    if (isRun(node) && !node.querySelector(SHELL_SELECTOR)) node.remove();
  }
}

/**
 * Stamp each shell with the type of the item before it in its host. Adjacency
 * is a fact about ORDER, so it is stamped rather than left to a CSS sibling
 * combinator that would keep painting after the operator moved either item —
 * `[data-follows="logo"]`, the identity middot, is what reads it today.
 * @param {{type: string}[]} items
 * @param {readonly HTMLElement[]} planned - one shell per item, in order
 */
function stampFollows(items, planned) {
  for (let index = 0; index < items.length; index += 1) {
    if (index === 0) delete planned[index].dataset.follows;
    else planned[index].dataset.follows = items[index - 1].type;
  }
}

/**
 * The shell for a planned item, created on first placement (a drag-in of a type
 * the server never rendered a shell for).
 * @param {{key: string, type: string}} item
 * @param {BarRoot} root
 * @returns {HTMLElement}
 */
function ensureShell(item, root) {
  const existing = shells.get(item.key);
  if (existing) return existing;
  const shell = docOf(root).createElement('div');
  shell.className = SHELL_CLASS;
  shell.dataset.barItem = item.type;
  shell.dataset.barKey = item.key;
  shells.set(item.key, shell);
  return shell;
}

/**
 * Move a shell out of the bars and into the pool. Parking is the ONLY removal
 * this module performs: destroying an adopted node here would stop it
 * resolving by id for every other module that looks it up. A parked shell has
 * no density and no neighbours, and its mirror is re-armed so the item can go
 * on hiding and revealing its body while it waits off screen.
 *
 * Exported because the overflow ladder (bar-overflow.js) folds by parking:
 * folding must never be `shell.hidden`, which is the mirror's output channel.
 * {@link restoreShell} is the way back.
 * @param {HTMLElement} shell
 * @param {BarRoot} [root]
 */
export function parkShell(shell, root = activeRoot ?? document) {
  closePopoversFor(shell);
  delete shell.dataset.barDensity;
  delete shell.dataset.follows;
  ensurePool(root).appendChild(shell);
  armMirror(shell);
}

/**
 * Take a shell back into a host without a reconcile — the overflow ladder
 * unfolding an item it had parked. Parking cleared the density and handed the
 * body's subscriptions to bar-items.js to dispose, so a returning shell needs
 * both stamped and built again before it is back on screen.
 * @param {HTMLElement} shell
 * @param {BarHost} host
 */
export function restoreShell(shell, host) {
  const type = shell.dataset.barItem;
  if (!type) return;
  const options = readOptions(shell, type);
  const density = densityForHost(host);
  stampShell(shell, type, options, density);
  buildBody(shell, type, options, density);
  armMirror(shell);
}

/**
 * Mirror `header_visible` and `status_visible` onto the root element, which is
 * where the server stamps them and where the shell-body height math reads them.
 * @param {BarLayout} layout
 * @param {BarRoot} root
 */
function applyBarVisibility(layout, root) {
  const html = docOf(root).documentElement;
  if (!html) return;
  if (typeof layout.header_visible === 'boolean') {
    html.dataset.headerBar = layout.header_visible ? 'visible' : 'hidden';
  }
  if (typeof layout.status_visible === 'boolean') {
    html.dataset.statusBar = layout.status_visible ? 'visible' : 'hidden';
  }
}

/* ---- focus ---- */

/**
 * Whether an element carries a selection range worth restoring.
 * @param {HTMLElement} node
 * @returns {boolean}
 */
function hasSelection(node) {
  if (node.tagName === 'TEXTAREA') return true;
  if (node.tagName !== 'INPUT') return false;
  const type = (/** @type {HTMLInputElement} */ (node).type || 'text').toLowerCase();
  return SELECTABLE_INPUT_TYPES.has(type);
}

/**
 * Remember where the caret is before anything moves.
 * @param {BarRoot} root
 * @returns {BarFocusState | null}
 */
function captureFocus(root) {
  const doc = docOf(root);
  const active = doc.activeElement;
  if (!active || active === doc.body || active === doc.documentElement) return null;
  const node = /** @type {HTMLElement} */ (active);
  if (typeof node.focus !== 'function') return null;
  /** @type {BarFocusState} */
  const state = { node, start: null, end: null, direction: null };
  if (hasSelection(node)) {
    const input = /** @type {HTMLInputElement} */ (/** @type {unknown} */ (node));
    try {
      state.start = input.selectionStart;
      state.end = input.selectionEnd;
      state.direction = input.selectionDirection;
    } catch {
      // Some input types refuse selection access; focus alone is restored.
    }
  }
  return state;
}

/**
 * Put the caret back. A node the reconcile parked is left alone: it is off
 * screen, and focusing pooled chrome would be worse than losing the caret.
 * @param {BarFocusState | null} state
 */
function restoreFocus(state) {
  if (!state) return;
  const { node } = state;
  if (!node.isConnected || !isLive(node)) return;
  if (node.ownerDocument.activeElement !== node) node.focus({ preventScroll: true });
  if (state.start === null) return;
  const input = /** @type {HTMLInputElement} */ (/** @type {unknown} */ (node));
  if (typeof input.setSelectionRange !== 'function') return;
  try {
    input.setSelectionRange(state.start, state.end ?? state.start, state.direction ?? undefined);
  } catch {
    // Selection could not be restored; focus already has been.
  }
}

// First paint. Synchronous, from the SSR DOM, with no network in the path.
hydrate();
