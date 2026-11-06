// @ts-check
/* OSPREY Web Terminal — the sidebar's session tree
 *
 * The Agent row wraps the tree: the operator's sessions (session-model.js),
 * each expanding to its artifacts grouped by day (Activity mode, the default)
 * or by form (Kind mode), then the Earlier row (artifacts no session owns),
 * "More sessions", and a notice when the session history is cut short or
 * unreadable.
 * The rows are drawn by sidebar-artifact-rows.js; this module owns the data,
 * the clicks and the viewer.
 *
 * DATA (lazy). Session rows load unconditionally, through the model. Groups
 * and counts load only while the artifacts panel is served (app.js runs
 * {@link initSidebarArtifacts} / {@link teardownSidebarArtifacts} under
 * `whilePanelServed`): one `session-summary` per loaded page of sessions, and
 * each open session's artifacts through `session-list` (`runs=grouped`, paged
 * by the cursor in the box, at whatever height; in Kind mode each open
 * group pages on its own).
 *
 * CLICKS. Only the session in use can show its artifacts; every other
 * session stays folded and has no disclosure. A click on another session's
 * name switches to it (`switchTo`, reopening the Agent pane first), which
 * unfolds it, and reveals the session's pick: fetched by id, inserted into its
 * group (its run opened), scrolled to and shown, the panels left to the
 * session's own layout (session-layout.js). A click (or double-click) on the
 * session in use folds or unfolds it. The fold is this tab's own state, reset
 * on every switch. An artifact click shows it without changing the session,
 * focusing an already docked Artifacts panel in place, and records it as its
 * owner's last-open artifact.
 *
 * PAGE LOAD. Once the listing has the active row, its last-open artifact is
 * revealed once, without the open rule: with the Artifacts panel docked it is
 * shown there; otherwise the layout stays as it is and the tree only opens its
 * session, group and run and scrolls to it. app.js runs it once
 * ({@link revealOnLoad}).
 *
 * THE HIGHLIGHT. At most one artifact row is selected (`aria-selected`): the
 * one the Artifacts panel shows, and none while that panel is not on view
 * (no tile, no pop-out window). It is derived on every render from the
 * session in use and the dock, so a switch, a close and a re-render all redraw
 * it from scratch; panel-manager's `osprey-dock-view-changed` re-renders the
 * tree when the panel comes on or goes off view.
 *
 * THE VIEWER. The artifacts iframe shows ONE artifact, told with
 * `osprey-artifact-select`; this module forwards the shown artifact's update
 * and delete frames to it. Which one, of the active session only:
 *   - the pick (session-model's, per session, cleared on every key change);
 *   - else, in chat, the agent's focus unless it is a run record;
 *   - else the newest of the session (not a shipped example), then any.
 * A non-automatic agent focus (or any focus on the terminal surface) takes the
 * pick and records last-artifact; an automatic one only records the focus. The
 * pane header's ‹ N of M › pages through the shown artifact's own session.
 *
 * LIVE UPDATES. An `artifact` frame counts once per run (`exec_id`, else id),
 * only when `operator_visible`, by its listed form; a counted run only takes
 * the new status. `artifact_updated` sets the status and re-places an entry
 * whose owner changed; `artifact_deleted` and a reconnect re-fetch the
 * summary. A frame whose owner is no known row refreshes the model.
 *
 * @module sidebar-artifacts
 */

import { isTerminalSurface, onSurfaceChange } from './agent-surface.js';
import { createEventSource } from './api.js';
import { contributeLocal } from './tile-header-contrib.js';
import * as iframeSync from './panel-iframe-sync.js';
import { activateTab } from './panel-manager.js';
import { hasDockedTile } from './panel-placement.js';
import { isPoppedOut } from './popout.js';
import { setServiceMenuExtras } from './panel-menu-policy.js';
import { isTerminalPanelOpen, openTerminalPanel } from './dock-workspace.js';
import * as store from './sidebar-artifact-listing.js';
import * as rows from './sidebar-artifact-rows.js';
import { createBoxPager } from './sidebar-artifact-paging.js';
import { headerItems, menuItems, openArtifact, saveArtifact } from './artifacts-pane-header.js';
import * as model from './session-model.js';

/**
 * @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry
 * @typedef {import('./sidebar-artifact-listing.js').Listing} Listing
 * @typedef {import('./session-model.js').SessionRow} SessionRow
 */

/** The panel id of the artifact gallery. */
const ARTIFACTS_PANEL_ID = 'artifacts';

/** The panel's event stream. */
const EVENTS_PATH = `${store.ARTIFACTS_BASE}/api/events`;

/** The tree's containers: the sidebar's and the collapsed sidebar's flyout. */
const TREE_SELECTOR = '[data-session-tree]';

/** @typedef {'kind' | 'activity'} TreeView */

function freshState() {
  return {
    /** @type {TreeView} */
    view: 'activity',
    /** The session in use folded by the operator (reset on a switch). */
    activeFolded: false,
    /** Whether the Earlier row is open. */
    earlierOpen: false,
    /** Open form groups per session. @type {Map<string, Set<string>>} */
    openGroups: new Map(),
    /** Open runs, by run key. @type {Set<string>} */
    openRuns: new Set(),
    /** @type {string | null} */
    focusId: null,
    /** @type {string | null} */
    shownId: null,
  };
}

let state = freshState();
/** The listings, the counts and every entry seen (sidebar-artifact-listing.js). */
const data = store.createTreeData({ onChange: () => render() });
/** @type {import('./sidebar-artifact-rows.js').TreeForm[]} */
let forms = rows.treeForms(null);
let served = false;

const pager = createBoxPager({
  isServed: () => served,
  loadMore: (treeId) => data.loadMoreOf(treeId, state.view, openGroupsOf(treeId)),
  boxes: () => document.querySelectorAll(`${TREE_SELECTOR} [data-session-box]`),
});
let treeStarted = false;
let abort = new AbortController();
/** The tree's own document listeners (removed only by a test reset). */
let treeAbort = new AbortController();
/** @type {{stop: () => void} | null} */
let stream = null;
let streamOpened = false;
let lastHeader = '';
/** Whether the Artifacts panel was on view at the last render. */
let panelWasOnView = false;

// ---- Ids ----

/**
 * A row's id in the tree: its transcript, or `key:<K>` for the synthesized row.
 * @param {SessionRow} row
 */
function treeIdOf(row) {
  return row.transcript_id ?? `key:${row.session_key}`;
}

/** @returns {string | null} the active row's tree id */
function activeTreeId() {
  const row = model.activeRow();
  return row ? treeIdOf(row) : null;
}

/**
 * The tree id an artifact belongs under: its owner's, or Earlier.
 * @param {ArtifactEntry} entry
 */
function treeIdOfEntry(entry) {
  return entry.owner ?? store.EARLIER;
}

// ---- Init / teardown ----

/**
 * Start the session tree: rows, clicks, the model. Runs whether or not the
 * artifacts panel is served (session rows load unconditionally). Idempotent.
 */
function initSessionTree() {
  if (treeStarted) return;
  treeStarted = true;
  state.view = store.readTreeView();
  const signal = treeAbort.signal;
  document.addEventListener('click', onClick, { signal });
  document.addEventListener('scroll', pager.onScroll, { capture: true, passive: true, signal });
  document.addEventListener('osprey-sidebar-rendered', render, { signal });
  document.addEventListener('osprey-artifact-opened', onArtifactOpened, { signal });
  // panel-manager.js's DOCK_VIEW_EVENT (not imported: tests mock that module).
  document.addEventListener('osprey-dock-view-changed', onDockView, { signal });
  pager.watch(document.querySelectorAll(TREE_SELECTOR), signal);
  model.subscribe(onModelChange);
  onSurfaceChange(() => render());
  void model.initSessionModel();
  render();
}

/**
 * Start the session tree, and with `served` (the default, as
 * `whilePanelServed` calls it) its artifact half: groups, counts, the viewer.
 * app.js calls it once with `served: false` at boot, so session rows load
 * whether or not the panel is served. Idempotent.
 * @param {{served?: boolean}} [options]
 */
export function initSidebarArtifacts({ served: withArtifacts = true } = {}) {
  initSessionTree();
  if (served || !withArtifacts) return;
  served = true;
  window.addEventListener('message', onMessage, { signal: abort.signal });
  stream = createEventSource(EVENTS_PATH, { onMessage: onFrame, onOpen: onStreamOpen });
  setServiceMenuExtras(ARTIFACTS_PANEL_ID, menuExtras);
  void store.fetchTypeRegistry().then((registry) => {
    if (!served) return;
    forms = rows.treeForms(registry);
    render();
  });
  void summarize(true);
  void restoreFocus();
  render();
}

/** Stop the artifact half and take its rows off the page. */
export function teardownSidebarArtifacts() {
  if (!served) return;
  served = false;
  abort.abort();
  abort = new AbortController();
  stream?.stop();
  stream = null;
  streamOpened = false;
  data.stop();
  setServiceMenuExtras(ARTIFACTS_PANEL_ID, null);
  state = { ...freshState(), view: state.view };
  lastHeader = '';
  contributeLocal(ARTIFACTS_PANEL_ID, headerItems(null, null), onHeaderAction);
  render();
}

// ---- Data ----

/**
 * Summarize the loaded sessions: the pages not summarized yet, or all of
 * them (`all`: a delete, a reconnect).
 * @param {boolean} [all]
 */
async function summarize(all = false) {
  if (!served) return;
  const ids = model
    .rows()
    .map((r) => r.transcript_id)
    .filter((id) => typeof id === 'string');
  if ((await data.summarize(/** @type {string[]} */ (ids), all)) && served) render();
}

/** @param {{reason: string}} change */
function onModelChange({ reason }) {
  // The session switched to always opens; the fold was the previous one's.
  if (reason === 'pointer' || reason === 'switched') state.activeFolded = false;
  if (reason === 'rows') void summarize();
  render();
}

/**
 * Read the agent's persisted focus once the active session's first page has
 * landed: a reload, a second tab or a reconnect learns it here, as automatic.
 */
async function restoreFocus() {
  const id = await store.fetchFocusedId();
  if (served && id && id !== state.focusId) await onFocus(id, true);
}

/** @param {string | null} id */
const held = (id) => data.held(id);

// ---- Which artifact is shown ----

/** @param {ArtifactEntry} entry */
function inActiveSession(entry) {
  const active = model.activeRow();
  if (!active) return false;
  if (active.transcript_id) return entry.owner === active.transcript_id;
  return entry.session_id === active.session_key;
}

/** @returns {ArtifactEntry | null} */
function resolveShown() {
  const pick = held(model.pick());
  if (pick) return pick;
  if (!isTerminalSurface()) {
    const focus = held(state.focusId);
    if (focus && focus.run_record !== true && focus.form !== 'runs' && inActiveSession(focus)) {
      return focus;
    }
  }
  const entries = activeListing()?.entries ?? [];
  return entries.find((a) => a.origin !== 'demo') ?? entries[0] ?? null;
}

function activeListing() {
  const row = model.activeRow();
  return row ? data.activeListing(row) : null;
}

/**
 * The listing the pager walks for an artifact: its own session's.
 * @param {ArtifactEntry} entry
 */
function pagerListing(entry) {
  if (inActiveSession(entry)) return /** @type {Listing} */ (activeListing());
  return data.listingOf(treeIdOfEntry(entry));
}

/** @param {1 | -1} delta */
async function step(delta) {
  const shown = held(state.shownId);
  const target = shown ? await data.stepFrom(pagerListing(shown), shown.id, delta) : null;
  if (!target || !served) return;
  model.setPick(target.id);
  model.recordLastArtifact(target);
}

/** @param {ArtifactEntry | null} shown */
function pagerPosition(shown) {
  return shown ? data.position(pagerListing(shown), shown.id) : null;
}

// ---- Recording ----

/**
 * A chat artifact link opened an artifact (app.js's `osprey-artifact-opened`):
 * record it as its owner's last-open artifact; the session does not change.
 * @param {Event} event
 */
function onArtifactOpened(event) {
  const id = /** @type {CustomEvent} */ (event).detail?.id;
  if (typeof id === 'string') void recordArtifactOpened(id);
}

/**
 * @param {string} id
 * @returns {Promise<void>}
 */
export async function recordArtifactOpened(id) {
  const entry = held(id) ?? (await store.fetchArtifact(id));
  if (entry) {
    data.hold(entry);
    model.recordLastArtifact(entry);
  }
}

// ---- The event stream ----

function onStreamOpen() {
  if (!served) return;
  if (streamOpened) {
    void summarize(true);
    void restoreFocus();
  }
  streamOpened = true;
}

/**
 * The extra listing a live entry also lands in: the active key's, when the
 * entry is the active session's.
 * @param {ArtifactEntry} entry
 */
function extraListings(entry) {
  const active = model.activeRow();
  return active && inActiveSession(entry) ? [`key:${active.session_key}|`] : [];
}

/** Whether an entry's owner is a loaded row (no owner = Earlier, always known). @param {ArtifactEntry} entry */
const knownOwner = (entry) => entry.owner == null || model.sessionOfArtifact(entry) !== null;

/** @param {any} frame */
function onFrame(frame) {
  if (!served || !frame || typeof frame !== 'object' || typeof frame.id !== 'string') return;
  const entry = /** @type {ArtifactEntry} */ ({ ...frame });
  delete (/** @type {any} */ (entry).type);
  if (frame.type === 'artifact') {
    if (!data.added(entry, extraListings(entry))) return;
    if (!knownOwner(entry)) model.refresh();
  } else if (frame.type === 'artifact_updated') {
    if (data.updated(entry, extraListings(entry)) && !knownOwner(entry)) model.refresh();
    if (frame.id === state.shownId) sendChanged('updated', entry);
  } else if (frame.type === 'artifact_deleted') {
    if (frame.id === state.shownId) sendChanged('deleted');
    data.deleted(frame.id);
    if (model.pick() === frame.id) model.setPick(null);
    if (state.focusId === frame.id) state.focusId = null;
    void summarize(true);
  } else {
    if (frame.type === 'focus') void onFocus(frame.id, frame.automatic === true);
    return;
  }
  render();
}

/**
 * The agent focused an artifact.
 * @param {string} id
 * @param {boolean} automatic
 */
async function onFocus(id, automatic) {
  if (!held(id)) {
    const entry = await store.fetchArtifact(id);
    if (!served || entry?.id !== id) return;
    data.hold(entry);
  }
  state.focusId = id;
  if (!automatic || isTerminalSurface()) {
    model.setPick(id);
    if (!automatic) model.recordLastArtifact(held(id));
  }
  render();
}

// ---- The viewer, the pane header and the tile menu ----

/** @returns {HTMLIFrameElement | null} */
function artifactsIframe() {
  return /** @type {HTMLIFrameElement | null} */ (
    document.querySelector(`iframe.panel-iframe[data-panel-id="${ARTIFACTS_PANEL_ID}"]`)
  );
}

function sendSelect() {
  const iframe = artifactsIframe();
  if (iframe) iframeSync.sendArtifactSelectToIframe(iframe, state.shownId);
}

/**
 * @param {'updated' | 'deleted'} change
 * @param {ArtifactEntry} [artifact]
 */
function sendChanged(change, artifact) {
  const iframe = artifactsIframe();
  if (iframe && state.shownId)
    iframeSync.sendArtifactChangedToIframe(iframe, state.shownId, change, artifact);
}

/** @param {MessageEvent} event */
function onMessage(event) {
  if (!served || event.origin !== window.location.origin) return;
  if (event.data?.type !== 'osprey-artifact-ready') return;
  const iframe = artifactsIframe();
  if (iframe && event.source === iframe.contentWindow) sendSelect();
}

/** @param {string} id */
function onHeaderAction(id) {
  if (!served) return;
  const shown = held(state.shownId);
  if (!shown) return;
  if (id === 'prev' || id === 'next') void step(id === 'prev' ? -1 : 1);
  else if (id === 'open') openArtifact(shown);
  else if (id === 'save') saveArtifact(shown);
}

function menuExtras() {
  const shown = served ? held(state.shownId) : null;
  if (!shown || !artifactsIframe()) return [];
  return menuItems(shown, (action) =>
    iframeSync.sendArtifactActionToIframe(artifactsIframe(), shown.id, action)
  );
}

/** Whether the Artifacts panel is on view: docked in a tile, or in its own window. */
const panelOnView = () => hasDockedTile(ARTIFACTS_PANEL_ID) || isPoppedOut(ARTIFACTS_PANEL_ID);

/** The dock changed: redraw the highlight when the panel came on or went off view. */
function onDockView() {
  if (served && panelOnView() !== panelWasOnView) render();
}

/** Show the Artifacts panel through the open rule: a docked one is focused in place. */
function showArtifactsPanel() {
  if (!hasDockedTile(ARTIFACTS_PANEL_ID)) activateTab(ARTIFACTS_PANEL_ID, { userInitiated: true });
}

// ---- Drawing ----

/**
 * The open form groups of a session, made on first use.
 * @param {string} treeId
 * @returns {Set<string>}
 */
function openGroupsOf(treeId) {
  if (!state.openGroups.has(treeId)) state.openGroups.set(treeId, new Set());
  return /** @type {Set<string>} */ (state.openGroups.get(treeId));
}

/**
 * The rows inside one open session's box.
 * @param {string} treeId
 * @returns {HTMLElement[]}
 */
function boxRows(treeId) {
  const common = {
    shownId: panelWasOnView ? state.shownId : null,
    openRuns: state.openRuns,
  };
  const open = openGroupsOf(treeId);
  const { entries, counts } = data.boxEntries(treeId, state.view, open);
  if (state.view === 'activity') return rows.activityRows(entries, common);
  return rows.kindRows(entries, { ...common, forms, sessionId: treeId, counts, openGroups: open });
}

/** @returns {Node[]} the whole tree */
function treeNodes() {
  /** @type {Node[]} */
  const nodes = [];
  const active = activeTreeId();
  let loadedTotal = 0;
  for (const row of model.rows()) {
    const treeId = treeIdOf(row);
    const count = data.totals.counts[treeId]?.total ?? 0;
    loadedTotal += row.transcript_id ? count : 0;
    const isActive = treeId === active;
    const open = served && isActive && !state.activeFolded;
    const node = rows.sessionRow(/** @type {any} */ (row), {
      active: isActive,
      open,
      count,
      summarized: treeId in data.totals.counts,
    });
    node.dataset.sessionId = treeId;
    nodes.push(node);
    if (open) nodes.push(rows.sessionBox(treeId, boxRows(treeId)));
  }
  if (served && data.totals.earlier.total > 0) {
    const open = state.earlierOpen;
    nodes.push(rows.earlierRow({ count: data.totals.earlier.total, open }));
    if (open) nodes.push(rows.sessionBox(store.EARLIER, boxRows(store.EARLIER)));
  }
  if (model.hasMore())
    nodes.push(rows.moreSessionsRow(Math.max(0, data.totals.ownedTotal - loadedTotal)));
  const record = model.recordState();
  if (record === 'incomplete' || record === 'unavailable') nodes.push(rows.noticeRow(record));
  return nodes;
}

/** Draw the tree and the header; tell the viewer when what it shows changed. */
function render() {
  if (!treeStarted) return;
  const shown = served ? resolveShown() : null;
  const changed = (shown?.id ?? null) !== state.shownId;
  state.shownId = shown?.id ?? null;
  panelWasOnView = served && panelOnView();
  for (const tree of document.querySelectorAll(TREE_SELECTOR)) {
    const el = document.activeElement;
    const focusedId = el instanceof HTMLElement && tree.contains(el) ? el.dataset.id : null;
    const scrolls = [...tree.querySelectorAll('[data-session-box]')].map((box) => [
      /** @type {HTMLElement} */ (box).dataset.sessionBox,
      box.scrollTop,
    ]);
    tree.replaceChildren(...treeNodes());
    for (const [id, top] of scrolls) {
      const box = tree.querySelector(`[data-session-box="${CSS.escape(String(id))}"]`);
      if (box) box.scrollTop = /** @type {number} */ (top);
    }
    if (focusedId)
      rows.rowIn(/** @type {HTMLElement} */ (tree), focusedId)?.focus({ preventScroll: true });
  }
  if (!served) return;
  pager.fill();
  const current = activeListing();
  if (current) data.ensureLoaded(current);
  const header = headerItems(shown, pagerPosition(shown));
  const key = JSON.stringify(header);
  if (key !== lastHeader) {
    lastHeader = key;
    contributeLocal(ARTIFACTS_PANEL_ID, header, onHeaderAction);
  }
  if (!changed) return;
  sendSelect();
  scrollToShown();
}

/** @param {string | null} [id] the row to scroll to, the shown one by default */
function scrollToShown(id = state.shownId) {
  if (!id) return;
  for (const tree of document.querySelectorAll(TREE_SELECTOR)) {
    rows.rowIn(/** @type {HTMLElement} */ (tree), id)?.scrollIntoView?.({ block: 'nearest' });
  }
}

// ---- Reveal ----

/**
 * Page load (OR-3): reveal the active row's last-open artifact without the
 * open rule, shown in a docked Artifacts panel or else scrolled to in the tree
 * with the layout untouched. app.js calls it once the listing has the row.
 * @returns {boolean} whether it ran (not before the row is listed and served)
 */
export function revealOnLoad() {
  const row = model.activeRow();
  if (!row || row.synthesized || !served) return false;
  if (row.last_artifact_id) void reveal(treeIdOf(row), row.last_artifact_id, true);
  return true;
}

/**
 * Reveal an artifact of a session: fetch it, put it into its group (its run
 * opened), scroll to it and show it. On page load (`onLoad`) the open rule is
 * not called, and without a docked Artifacts panel the artifact is only
 * scrolled to, the layout left as it is. `keepLayout` skips the open rule
 * too: after a switch the panels are the session's layout's to decide.
 * @param {string} treeId
 * @param {string | null} id  the pick, or null for the session's newest
 * @param {boolean} [onLoad]
 * @param {boolean} [keepLayout]
 */
async function reveal(treeId, id, onLoad = false, keepLayout = false) {
  if (!served) return;
  if (treeId === activeTreeId()) state.activeFolded = false;
  let entry = id ? (held(id) ?? (await store.fetchArtifact(id))) : null;
  if (!entry) {
    const listing = data.listingOf(treeId);
    if (!listing.loaded) await listing.loadFirst();
    entry = listing.entries.find((a) => a.origin !== 'demo') ?? listing.entries[0] ?? null;
    if (!entry) return;
  }
  data.hold(entry);
  if (state.view === 'kind') {
    const { form, runKey } = await data.revealIn(treeId, entry);
    if (form) openGroupsOf(treeId).add(form);
    if (runKey) state.openRuns.add(runKey);
  }
  const onlyScroll = onLoad && !hasDockedTile(ARTIFACTS_PANEL_ID);
  if (!onlyScroll && !model.pick()) model.setPick(entry.id);
  render();
  scrollToShown(onlyScroll ? entry.id : state.shownId);
  if (!onLoad && !keepLayout) showArtifactsPanel();
}

// ---- Input ----

/** @param {MouseEvent} event */
function onClick(event) {
  const target = event.target instanceof Element ? event.target : null;
  const tree = target?.closest(TREE_SELECTOR);
  if (!target || !tree) return;
  const session = target.closest('.osprey-tree-session[data-session-id]');
  const earlier = target.closest('[data-tree-earlier]');
  const disc = target.closest('[data-tree-disc]');
  if (session instanceof HTMLElement) {
    // A double-click's second click is the same gesture: it never undoes the
    // first (a fold, or the switch it started).
    if (event.detail > 1) return;
    const treeId = session.dataset.sessionId ?? '';
    // The session in use folds or unfolds; another is entered, never browsed.
    if (treeId === activeTreeId()) {
      state.activeFolded = !state.activeFolded;
      render();
    } else void openInAgent(treeId);
    return;
  }
  if (earlier) {
    state.earlierOpen = !state.earlierOpen;
    return render();
  }
  if (target.closest('[data-tree-more]')) return void model.loadMore();
  const head = target.closest('[data-group-head]');
  if (head instanceof HTMLElement) {
    const [treeId, form] = splitGroup(head.dataset.groupHead ?? '');
    flip(openGroupsOf(treeId), form);
    return render();
  }
  const row = target.closest('.osprey-list-row[data-id]');
  if (!(row instanceof HTMLElement)) return;
  const run = row.closest('[data-run]') ?? (row.matches('[data-run]') ? row : null);
  if (disc && run instanceof HTMLElement) {
    flip(state.openRuns, run.dataset.run ?? '');
    return render();
  }
  pickArtifact(/** @type {string} */ (row.dataset.id));
}

/**
 * @param {string} value  `sid:form`; the session id may itself hold colons
 * @returns {[string, string]}
 */
function splitGroup(value) {
  const at = value.lastIndexOf(':');
  return [value.slice(0, at), value.slice(at + 1)];
}

/** Toggle a key's membership. @param {Set<string>} set @param {string} key */
function flip(set, key) {
  if (!set.delete(key)) set.add(key);
}

/**
 * Open a session in the Agent (another session's name, the session menu's
 * "Open in Agent"): switch to it (the Agent pane reopened first), unfold it,
 * and reveal its pick. After a switch the session's own layout says whether
 * the Artifacts panel is open (session-layout.js); on the session in use the
 * panel is brought up.
 * @param {string} treeId
 */
export async function openInAgent(treeId) {
  const row = model.rows().find((r) => treeIdOf(r) === treeId);
  if (!row) return;
  if (!isTerminalPanelOpen()) openTerminalPanel();
  const wasActive = treeId === activeTreeId();
  const moved = wasActive ? false : await model.switchTo(row, { reason: 'gesture' });
  if (!wasActive && !moved) return;
  const now = activeTreeId() ?? treeId;
  state.activeFolded = false;
  render();
  await reveal(now, model.pick() ?? row.last_artifact_id ?? null, false, !wasActive);
}

/**
 * The operator picked an artifact: show it without changing the session and
 * record it as its owner's last-open artifact.
 * @param {string} id
 */
function pickArtifact(id) {
  model.setPick(id);
  model.recordLastArtifact(held(id));
  render();
  showArtifactsPanel();
}

/** @returns {TreeView} the tree mode in force */
export function treeView() {
  return state.view;
}

/**
 * Persist and apply the tree mode.
 * @param {TreeView} view
 */
export function setTreeView(view) {
  if (view !== 'kind' && view !== 'activity') return;
  state.view = view;
  store.writeTreeView(view);
  render();
}

/** Forget everything (tests). */
export function resetSidebarArtifacts() {
  teardownSidebarArtifacts();
  treeAbort.abort();
  treeAbort = new AbortController();
  treeStarted = false;
  forms = rows.treeForms(null);
  state = freshState();
  panelWasOnView = false;
  model.resetSessionModel();
}
