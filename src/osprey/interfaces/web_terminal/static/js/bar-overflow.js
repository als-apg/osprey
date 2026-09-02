/* OSPREY Web Terminal — the bar crowding ladder.
 *
 * A bar that runs out of room must give something up, and WHICH thing it gives
 * up is a policy decision, not a layout accident. This module owns that policy
 * for both item hosts. It is a separate module from bar-host.js on purpose:
 * the host owns where a shell lives, this owns when the host is asked to move
 * one, and the two answer to different inputs (a layout document vs. the width
 * of a box).
 *
 * ---- The ladder, in order ----
 *
 *   0. Spacing yields, continuously and BEFORE any rung here. Gaps and spaces
 *      shrink through the `flex-shrink` the catalog declares on their shells,
 *      so the bar is already as narrow as its spacing can make it by the time
 *      this module measures anything. The JS ladder therefore never touches a
 *      spacing item — not to fold it, not to resize it. If it did, it would be
 *      racing CSS for the same pixels.
 *
 *   1. Text ellipsizes. Also CSS (`min-width: 0` on the activity shell). The
 *      only thing this module does for rung 1 is stay out of the way: it does
 *      not fold anything while the bar is merely tight, because a probe that
 *      reports no overflow is a bar whose text has already absorbed it.
 *
 *   2. Search collapses to its magnifier. `data-bar-collapsed="true"` on the
 *      search shell is the hook; the width comes off in CSS. Search NEVER
 *      folds — a command palette you cannot see is a command palette nobody
 *      finds — so this rung is where search stops.
 *
 *   3. Foldable items fold, LOWEST PRIORITY FIRST. Folding parks the shell in
 *      `#bar-item-pool` through the host's own move machinery, which keeps the
 *      node alive (and lets bar-items.js dispose its subscriptions), and adds
 *      an `overflowLabel` row to the bar's overflow menu. Only the six types
 *      whose `overflowLabel(ctx)` is non-null fold; locked chrome never does.
 *
 * Folding is NEVER `shell.hidden`. That attribute is the hidden-mirror's
 * output channel in bar-host.js — it is rewritten on every placement pass and
 * on every body mutation — so a ladder that hid shells would spend its life
 * fighting the mirror. The ladder moves nodes; the mirror writes `hidden`.
 *
 * ---- Measuring, and why it is injectable ----
 *
 * Every rung is driven by one probe, {@link CrowdingProbe}, which answers two
 * questions about a host: by how many pixels does its content exceed its box,
 * and how wide is that box. Production reads both off the DOM. Tests inject a
 * fake through {@link mockCrowding}, because happy-dom has no layout engine —
 * every element there is 0 px wide, so a real measurement would report a bar
 * that is never crowded and the ladder would be untestable. The proof that the
 * declared flex hints actually prevent overflow is a browser assertion and
 * lives in the Playwright suite, not here.
 *
 * The width half of the report is what makes unfolding safe. An item folded at
 * width W is only offered its place back once the host is WIDER than W, so the
 * ladder cannot oscillate between two rungs at a single width.
 */

import { BAR_HOSTS, barItemType } from './bar-catalog.js';
import {
  docOf,
  hostElement,
  managedShells,
  parkShell,
  restoreShell,
  topLevelManaged,
} from './bar-host.js';
import { disposeItem, toggleOverflowMenu } from './tile-header-items.js';
import { svgIcon } from './svg-icons.js';

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */
/** @typedef {import('./bar-catalog.js').BarItemOptions} BarItemOptions */
/** @typedef {import('./bar-host.js').BarRoot} BarRoot */
/** @typedef {import('./tile-header-items.js').OverflowRow} OverflowRow */

/**
 * What the ladder knows about one host's width.
 * @typedef {object} CrowdingReport
 * @property {number} overflow - px the content exceeds the box by; <= 0 is roomy
 * @property {number} width - the host's current inner width in px
 */

/**
 * The measurement seam. Production measures the DOM; tests inject a fake via
 * {@link mockCrowding} and drive the ladder from it.
 * @typedef {(container: HTMLElement, host: BarHost) => CrowdingReport} CrowdingProbe
 */

/**
 * One rung this ladder has climbed down, and everything needed to climb back
 * up it: the shell it acted on, where that shell sat, and the host width the
 * step was taken at.
 * @typedef {object} LadderStep
 * @property {'search' | 'fold'} kind
 * @property {HTMLElement} shell
 * @property {HTMLElement | null} anchor - the top-level node the shell sat before
 * @property {number} width - the host width when the step was taken
 */

/** The trigger button that opens a host's overflow menu. */
const TRIGGER_CLASS = 'bar-overflow-trigger';

/** One rung per iteration, so a pass is bounded even if a probe misbehaves. */
const MAX_STEPS = 24;

/** The rungs each host has climbed down, oldest first. @type {Map<BarHost, LadderStep[]>} */
const ladders = new Map();

/**
 * Shells the operator picked out of the overflow menu. A promoted item folds
 * LAST among the foldables, so picking it in a bar that is still too narrow
 * folds the next item instead of undoing the pick.
 * @type {Set<HTMLElement>}
 */
const promoted = new Set();

/** @type {CrowdingProbe} */
function measureCrowding(container) {
  return { overflow: container.scrollWidth - container.clientWidth, width: container.clientWidth };
}

/** @type {CrowdingProbe} */
let probe = measureCrowding;

/**
 * Drive the ladder from a fake measurement instead of the DOM. The returned
 * function puts the real probe back; a suite that forgets to call it leaks the
 * fake into every later test in the file.
 * @param {CrowdingProbe} fake
 * @returns {() => void} restore
 */
export function mockCrowding(fake) {
  const previous = probe;
  probe = fake;
  return () => {
    probe = previous;
  };
}

/**
 * Climb the ladder for both hosts until each is as roomy as it can be. Run
 * this after a `reconcile()` and whenever a bar changes width.
 * @param {BarRoot} [root]
 */
export function applyOverflow(root = document) {
  for (const host of BAR_HOSTS) {
    const container = hostElement(host, root);
    if (container) runLadder(container, host);
  }
}

/**
 * Run the ladder now and keep running it as the bars change width. Kept as an
 * explicit call rather than an import-time side effect: the bars are hydrated
 * by the time anything wants a ladder, and a module that observes on import
 * would need a ResizeObserver in every test that touches it.
 * @param {BarRoot} [root]
 * @returns {() => void} stop watching
 */
export function watchCrowding(root = document) {
  const view = docOf(root).defaultView;
  /** @type {(() => void)[]} */
  const stops = [];
  for (const host of BAR_HOSTS) {
    const container = hostElement(host, root);
    if (!container || !view) continue;
    const run = () => runLadder(container, host);
    if (view.ResizeObserver) {
      const observer = new view.ResizeObserver(run);
      observer.observe(container);
      stops.push(() => observer.disconnect());
    } else {
      view.addEventListener('resize', run);
      stops.push(() => view.removeEventListener('resize', run));
    }
  }
  applyOverflow(root);
  return () => {
    for (const stop of stops) stop();
  };
}

/**
 * Forget every rung, without touching the DOM. The teardown entry point: a
 * page leaving, or a test starting from a fresh document, where restoring
 * shells into a container that is about to be replaced is pure waste.
 */
export function resetOverflow() {
  for (const steps of ladders.values()) {
    for (const step of steps) if (step.kind === 'search') delete step.shell.dataset.barCollapsed;
    steps.length = 0;
  }
  promoted.clear();
}

/* ---- the pass ---- */

/**
 * Fold or unfold one rung at a time until the host stops changing. Each
 * iteration re-measures, so a rung that did not buy enough room is followed by
 * the next one down rather than by a guess.
 * @param {HTMLElement} container
 * @param {BarHost} host
 */
function runLadder(container, host) {
  for (let step = 0; step < MAX_STEPS; step += 1) {
    const report = probe(container, host);
    const changed =
      report.overflow > 0 ? foldOne(container, host, report) : unfoldOne(container, host, report);
    if (!changed) break;
  }
  syncTrigger(container, host);
  stampFollows(container);
}

/**
 * Climb one rung down: collapse search if it is still expanded, otherwise fold
 * the lowest-priority foldable this host still shows.
 * @param {HTMLElement} container
 * @param {BarHost} host
 * @param {CrowdingReport} report
 * @returns {boolean} whether anything changed
 */
function foldOne(container, host, report) {
  const steps = stepsFor(host);
  const search = managedShells(container).find((shell) => shell.dataset.barItem === 'search');
  if (search && search.dataset.barCollapsed !== 'true') {
    search.dataset.barCollapsed = 'true';
    steps.push({ kind: 'search', shell: search, anchor: null, width: report.width });
    return true;
  }
  const shell = nextFoldable(container, host);
  if (!shell) return false;
  steps.push({ kind: 'fold', shell, anchor: after(container, shell), width: report.width });
  parkShell(shell);
  return true;
}

/**
 * Climb one rung back up, but only once the host is genuinely wider than it
 * was when the rung was taken. Steps are undone in the order they were taken,
 * reversed, so search un-collapses last — it was the first thing to give.
 * @param {HTMLElement} container
 * @param {BarHost} host
 * @param {CrowdingReport} report
 * @returns {boolean} whether anything changed
 */
function unfoldOne(container, host, report) {
  const steps = stepsFor(host);
  const step = steps[steps.length - 1];
  if (!step || report.width <= step.width) return false;
  steps.pop();
  undoStep(step, container, host);
  return true;
}

/**
 * Put back what one step took away. A folded shell goes back where it sat and
 * is rebuilt: parking cleared its density and handed its subscriptions to
 * bar-items.js to dispose, so the body that comes back is a fresh one showing
 * the CURRENT state rather than the state it froze at.
 * @param {LadderStep} step
 * @param {HTMLElement} container
 * @param {BarHost} host
 */
function undoStep(step, container, host) {
  if (step.kind === 'search') {
    delete step.shell.dataset.barCollapsed;
    return;
  }
  const anchor = step.anchor?.parentElement === container ? step.anchor : triggerOf(container);
  container.insertBefore(step.shell, anchor);
  delete step.shell.dataset.barBuilt;
  restoreShell(step.shell, host);
}

/**
 * The next item to fold: the lowest-priority foldable this host shows, with
 * anything the operator promoted out of the menu kept for last. Locked chrome,
 * search and the spacing items all declare a null `overflowLabel` and are
 * therefore never candidates — the catalog is the whole policy.
 * @param {HTMLElement} container
 * @param {BarHost} host
 * @returns {HTMLElement | null}
 */
function nextFoldable(container, host) {
  /** @type {HTMLElement | null} */
  let best = null;
  /** @type {[number, number] | null} */
  let bestRank = null;
  for (const shell of managedShells(container)) {
    const rank = foldRank(shell, host);
    if (!rank) continue;
    if (!bestRank || rank[0] < bestRank[0] || (rank[0] === bestRank[0] && rank[1] < bestRank[1])) {
      best = shell;
      bestRank = rank;
    }
  }
  return best;
}

/**
 * How eager a shell is to fold, lower folding first, or null when it never
 * folds at all.
 * @param {HTMLElement} shell
 * @param {BarHost} host
 * @returns {[number, number] | null} [promoted last, then priority]
 */
function foldRank(shell, host) {
  const entry = barItemType(shell.dataset.barItem ?? '');
  if (!entry || entry.locked) return null;
  if (!entry.overflowLabel(contextFor(shell, host))) return null;
  return [promoted.has(shell) ? 1 : 0, entry.priority];
}

/**
 * The top-level node a shell currently sits before, which is where it goes
 * back when it unfolds. Null means "at the end", where the trigger button is.
 * @param {HTMLElement} container
 * @param {HTMLElement} shell
 * @returns {HTMLElement | null}
 */
function after(container, shell) {
  const nodes = topLevelManaged(container);
  const index = nodes.indexOf(shell);
  return index < 0 ? null : (nodes[index + 1] ?? null);
}

/**
 * The catalog context for one placed shell: which bar it is in, and the
 * options the host stamped on it.
 * @param {HTMLElement} shell
 * @param {BarHost} host
 * @returns {{host: BarHost, options: BarItemOptions}}
 */
function contextFor(shell, host) {
  /** @type {Record<string, string | number | boolean>} */
  let options = {};
  const raw = shell.dataset.barOptions;
  if (raw) {
    try {
      options = JSON.parse(raw);
    } catch {
      options = {};
    }
  }
  return { host, options };
}

/**
 * One host's rungs, created on first use.
 * @param {BarHost} host
 * @returns {LadderStep[]}
 */
function stepsFor(host) {
  let steps = ladders.get(host);
  if (!steps) {
    steps = [];
    ladders.set(host, steps);
  }
  return steps;
}

/**
 * Re-stamp `data-follows` over what the host actually SHOWS. bar-host stamps
 * it from the layout, which still names the items this pass folded away; the
 * middot it drives is a claim about two items being next to each other on
 * screen, so folding one has to move the claim with it.
 * @param {HTMLElement} container
 */
function stampFollows(container) {
  /** @type {string | null} */
  let previous = null;
  for (const shell of managedShells(container)) {
    if (previous) shell.dataset.follows = previous;
    else delete shell.dataset.follows;
    previous = shell.dataset.barItem ?? null;
  }
}

/* ---- the overflow menu ---- */

/**
 * The host's overflow trigger, if it currently has one.
 * @param {HTMLElement} container
 * @returns {HTMLElement | null}
 */
function triggerOf(container) {
  return /** @type {HTMLElement | null} */ (container.querySelector(`.${TRIGGER_CLASS}`));
}

/**
 * Show the trigger exactly while this host has something folded, and keep it
 * last in the bar — a placement pass appends to the end of the container, so
 * the trigger is re-anchored after every run rather than trusted to stay put.
 * @param {HTMLElement} container
 * @param {BarHost} host
 */
function syncTrigger(container, host) {
  const existing = triggerOf(container);
  if (!stepsFor(host).some((step) => step.kind === 'fold')) {
    if (existing) {
      disposeItem(existing);
      existing.remove();
    }
    return;
  }
  container.appendChild(existing ?? buildTrigger(container, host));
}

/**
 * Build the ⋯ button. Rows are computed on click, not now: the fold set moves
 * with the width of the bar, and a menu built ahead of time would open on a
 * list that has since changed.
 * @param {HTMLElement} container
 * @param {BarHost} host
 * @returns {HTMLElement}
 */
function buildTrigger(container, host) {
  const btn = container.ownerDocument.createElement('button');
  btn.type = 'button';
  btn.className = `bar-item-btn ${TRIGGER_CLASS}`;
  btn.title = 'More items';
  btn.setAttribute('aria-label', 'More items');
  btn.setAttribute('aria-haspopup', 'menu');
  btn.setAttribute('aria-expanded', 'false');
  btn.appendChild(svgIcon('ellipsis'));
  btn.addEventListener('click', () => toggleOverflowMenu(btn, overflowRows(container, host)));
  return btn;
}

/**
 * One row per folded item, most recently folded first — which is the order
 * they would come back in, and so the order that puts the item closest to
 * returning at the top. Picking a row brings that item back into the bar and
 * marks it promoted, so the ladder folds something else in its place instead
 * of immediately undoing the pick.
 * @param {HTMLElement} container
 * @param {BarHost} host
 * @returns {OverflowRow[]}
 */
function overflowRows(container, host) {
  /** @type {OverflowRow[]} */
  const rows = [];
  for (const step of [...stepsFor(host)].reverse()) {
    if (step.kind !== 'fold') continue;
    const entry = barItemType(step.shell.dataset.barItem ?? '');
    const label = entry?.overflowLabel(contextFor(step.shell, host));
    if (!label) continue;
    const shell = step.shell;
    rows.push({ label, pick: () => promote(shell, container, host) });
  }
  return rows;
}

/**
 * Bring one folded item back because the operator asked for it by name.
 * @param {HTMLElement} shell
 * @param {HTMLElement} container
 * @param {BarHost} host
 */
function promote(shell, container, host) {
  promoted.add(shell);
  const steps = stepsFor(host);
  const index = steps.findIndex((step) => step.shell === shell);
  if (index >= 0) undoStep(steps.splice(index, 1)[0], container, host);
  runLadder(container, host);
}
