/* OSPREY Web Terminal — Dockview Workspace Shell */

import { fitTerminal } from './terminal.js';
import { fetchJSON } from './api.js';
import { reconcile } from './dock-reconcile.js';
import {
  createTileTab,
  OSPREY_TAB_COMPONENT,
  setTerminalConsoleBarSource,
  setTerminalHeaderSource,
} from './dock-tab.js';
import { registerContribHost, unregisterContribHost } from './tile-header-contrib.js';
import { subscribe } from '/design-system/js/theme-manager.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** @typedef {import('./dock-reconcile.js').PanelDescriptor} PanelDescriptor */

/**
 * @typedef {object} ContentRenderer
 * @property {HTMLElement} element  Node dockview appends into its content container.
 * @property {(params: any) => void} init  dockview lifecycle hook (no-op here).
 */

/**
 * Live DockviewApi once initialized, else null. Kept module-scoped so the
 * follow-on dock tasks (state-sync, persistence) can reach the
 * same instance through getDockApi().
 * @type {any}
 */
let dockApi = null;

const PANEL_TERMINAL = 'terminal';

/** Dock tab title for the Agent pane (the terminal/chat card). A real title
 *  makes the pane a first-class tile; the bar it adopts (dock-tab.js) is the
 *  visible label. Stored layouts carrying an older title (`SESSION`, or the
 *  empty string from before the pane was titled) are healed on restore by
 *  normalizeTerminalTitle. */
const TERMINAL_TITLE = 'Agent';

/** Attribute on <html> present exactly while the Agent pane is absent from
 *  the dock (`data-agent="closed"`); the sidebar's Agent row reads it. */
const AGENT_ATTRIBUTE = 'data-agent';

/** Whether the Agent pane was open at the last {@link syncAgentPane}, or null
 *  before the first sync. @type {boolean | null} */
let agentPaneOpen = null;

/** localStorage key prefix; the per-project key from GET /api/panels is appended
 *  so one browser origin persists a separate layout per project — and, through
 *  `layoutKey()`, per view. */
const LAYOUT_KEY_PREFIX = 'osprey-dock-layout-';

/** Per-project layout key BASE (prefix + project_key), or null until
 *  initPersistence has fetched the project key. No key ⇒ persistence is inert
 *  this session.
 *
 *  Never used bare: every read and write goes through `layoutKey()`, which
 *  suffixes it per render through `scopedStorageKey()`. Project key alone is not enough on a multi-user mount,
 *  where every render shares one origin and would otherwise inherit — and
 *  overwrite — whichever arrangement another operator left behind for the same
 *  project. Resolved at each use, never stored resolved, so the key always
 *  reflects the document actually being served (see storage-scope.js).
 *  @type {string|null} */
let layoutStorageKeyBase = null;

/**
 * The storage key for the arrangement, or null while persistence is inert.
 * @returns {string|null}
 */
function layoutKey() {
  return layoutStorageKeyBase ? scopedStorageKey(layoutStorageKeyBase) : null;
}

/** Pending debounce handle for a persist write (see schedulePersist).
 *  @type {ReturnType<typeof setTimeout>|null} */
let persistHandle = null;

/** Sidebar width used when `#sidebar` is not in the document (its expanded
 *  width, `--wt-sidebar-width`). */
const SIDEBAR_FALLBACK_WIDTH = 232;

/** Fraction of the workspace the terminal takes by default — mirrors the old
 *  fixed split (terminal `flex: 0 1 40%`, workspace `flex: 1 1 60%`). */
const TERMINAL_WIDTH_FRACTION = 0.4;

/**
 * Pixel width for the service-panel group when it first docks beside the
 * terminal — the service side of the old fixed split (terminal 40% / services
 * 60%). Consumed by dock-iframe.js's ensurePlaceholder.
 * The sidebar's current width (expanded or collapsed) is not ours.
 * @returns {number}
 */
export function defaultServiceWidth() {
  const sidebar = document.getElementById('sidebar')?.offsetWidth ?? SIDEBAR_FALLBACK_WIDTH;
  return Math.max(0, Math.round((window.innerWidth - sidebar) * (1 - TERMINAL_WIDTH_FRACTION)));
}

/**
 * Callback the iframe adapter registers (the managed-iframe set lives there),
 * invoked after EVERY layout apply — default rebuild, stored-layout restore.
 * It re-docks any visible service placeholder the applied
 * layout lacks and prunes placeholders whose service no longer exists, so no
 * apply can strand a visible panel or leave a dead tab.
 * @type {(() => void) | null}
 */
let redockServices = null;

/** @param {() => void} fn  Register the service-placeholder redock hook. */
export function setServiceRedock(fn) {
  redockServices = fn;
}

/**
 * Callback dock-sync registers to learn when the BOOT layout is final — i.e.
 * when initPersistence has finished deciding what this session starts with,
 * whether that is a restored layout or the default because nothing was stored
 * or no project key came back.
 *
 * It exists because the occupancy reporter must not describe a layout the
 * restore is about to replace, and the restore's timing is a network round trip,
 * not a delay anyone can budget for. Registration (rather than dock-sync being
 * imported here) keeps the dependency one-way: dock-sync already imports this
 * module, and the reverse import would close a cycle.
 * @type {(() => void) | null}
 */
let layoutRestored = null;

/** Whether the boot layout has already been announced. "Settled" is a LATCHED
 *  state, not a passing event: initPersistence's fetch routinely resolves before
 *  dock-sync has finished its own retry-poll wiring, so an announcement made to
 *  nobody must still be delivered to whoever registers next. Without this a fast
 *  server — the common case locally — silently costs the reporter its first
 *  report. */
let layoutRestoredAlready = false;

/**
 * Register the boot-layout-settled hook. Fires immediately when the boot layout
 * has already settled, so registration order does not matter.
 * @param {() => void} fn
 */
export function setLayoutRestoredHook(fn) {
  layoutRestored = fn;
  if (layoutRestoredAlready) fn();
}

/**
 * Whether the boot layout has settled: the arrangement on screen is the final
 * one and no restore is pending that would re-apply it. False until the boot
 * layout is announced, true from then on.
 * @returns {boolean}
 */
export function bootLayoutSettled() {
  return layoutRestoredAlready;
}

/** Announce the boot layout as final, latching it for any later subscriber. */
function announceLayoutRestored() {
  layoutRestoredAlready = true;
  layoutRestored?.();
}

/**
 * Wrap an already-rendered page subtree as a dockview content renderer. dockview
 * MOVES `element` into its content container (a DOM relocation, not a clone), so
 * every reference terminal.js / chat.js / panel-manager.js already hold into the
 * subtree stays live and its ids/classes stay stable (the CSS surface gate
 * keys off them). A neutral flex host makes the subtree fill its dockview panel
 * regardless of the subtree's own flex-basis: the old fixed split relied on the
 * subtree being a flex child of .main-container, but dockview sizes the group.
 * @param {HTMLElement} subtree
 * @returns {ContentRenderer}
 */
function adoptSubtree(subtree) {
  const host = document.createElement('div');
  host.className = 'dock-panel-host';
  host.appendChild(subtree);
  return {
    element: host,
    init() {
      /* content is pre-rendered page DOM */
    },
  };
}

/**
 * Initialize the dockview workspace in #dock-root and dock the terminal/chat
 * card as the shell's one native panel; service panels arrive as `iframe:`
 * placeholders (dock-iframe.js), the first of which docks to the terminal's
 * left at the classic 60/40 split. The legacy #right-panel column is NOT
 * docked — in the docked shell it is only the fallback iframe host, so it is
 * hidden here to keep its `display: contents` holder from laying it out
 * beside the grid. No-ops (leaving the rest of boot intact — fallback mode,
 * #right-panel visible) if the root or terminal subtree is missing; when only
 * the vendored dockview global failed to load, the fallback host gets its
 * tile bar ({@link initFallbackTileBar}) so embedded panels keep their one
 * header.
 */
export function initDockWorkspace() {
  // Idempotent: a second call is a no-op once the grid exists, so the follow-on
  // dock tasks (persistence) can import from this file without
  // risking a re-entry that would double-adopt the source subtrees.
  if (dockApi) return;

  const root = document.getElementById('dock-root');
  const workspaceSource = document.getElementById('right-panel');
  const terminalSource = document.querySelector('.terminal-panel');
  if (!root || !(terminalSource instanceof HTMLElement)) return;

  // Grab the live Agent bars NOW, while terminalSource is still attached to
  // the document — createComponent's adoptSubtree (below) moves this whole
  // subtree into a detached host div ahead of dockview's own insertion of
  // that host. dockview always builds a panel's content component before its
  // tab component, so a later `document.querySelector` in dock-tab.js is
  // guaranteed to miss the bars at that point, not merely liable to. See
  // dock-tab.js's setTerminalHeaderSource for the consuming side. The
  // console bar exists by now because app.js builds the chat (initChat)
  // before the dock; a console that failed to build leaves the header as
  // the Agent bar in both views.
  const terminalHeaderSource = terminalSource.querySelector('.terminal-header');
  if (terminalHeaderSource instanceof HTMLElement) {
    setTerminalHeaderSource(terminalHeaderSource);
  } else {
    console.error('.terminal-header not found; terminal tile will render without its header bar');
  }
  const consoleBarSource = terminalSource.querySelector('#operator-container > .op-session-bar');
  if (consoleBarSource instanceof HTMLElement) setTerminalConsoleBarSource(consoleBarSource);

  const dockview = /** @type {any} */ (window)['dockview-core'];
  if (!dockview || typeof dockview.createDockview !== 'function') {
    console.error('dockview-core global unavailable; workspace shell not initialized');
    if (workspaceSource instanceof HTMLElement) initFallbackTileBar(workspaceSource);
    return;
  }

  // Docked mode: the fallback host must not lay out beside the grid (its
  // .dock-panel-sources holder is display:contents). Inline so it outranks the
  // .files-panel display rules; fallback mode never reaches this line.
  if (workspaceSource instanceof HTMLElement) workspaceSource.style.display = 'none';

  dockApi = dockview.createDockview(root, {
    // Without an explicit theme dockview stamps its DEFAULT (abyss) class on
    // the .dv-shell it creates inside #dock-root — a closer ancestor than the
    // classes wireDockTheme manages on the root, so abyss's dark `--dv-*`
    // values would shadow the light theme for every descendant. Start on the
    // matching base theme; wireDockTheme keeps it live from then on.
    theme: schemeIsLight() ? dockview.themeLight : dockview.themeDark,
    createComponent: (/** @type {{ name: string }} */ options) => {
      if (options.name === PANEL_TERMINAL) return adoptSubtree(terminalSource);
      return { element: document.createElement('div'), init() {} };
    },
    // Every tile's tab renders as the host header bar (title/actions) — see
    // dock-tab.js. defaultTabComponent routes panels with no explicit
    // tabComponent (i.e. all of them, including restored layouts) through
    // createTabComponent.
    createTabComponent: (/** @type {{ id: string }} */ options) => createTileTab(options.id),
    defaultTabComponent: OSPREY_TAB_COMPONENT,
    // One panel per tile means every tab is a lone tab: stretch it across the
    // whole strip so the bar spans its tile, the close control anchors at the
    // tile's right edge, and the void container (dockview's empty-space drag
    // surface) collapses to zero. Without this the tab is sized to its
    // content and two thirds of the "bar" is void painted a different color.
    singleTabMode: 'fullwidth',
  });

  // Bind dockview's light/dark base theme (root classes + the api's own theme
  // option) to the active OSPREY theme and keep both in sync on every live
  // toggle. The concrete `--dv-*` values are remapped onto OSPREY tokens in
  // dockview-overrides.css (which follow html[data-theme] on their own); the
  // base-theme swap only ensures any var NOT remapped there still resolves to
  // a correct light/dark default.
  wireDockTheme(root);

  // One panel per tile: the sidebar is the workspace's tab system, so no drop may
  // create a tab stack. Veto every drop geometry that would stack — onto a
  // tile's tab strip, its empty header space, or its center — leaving edge
  // drops (splits) as the only accepted targets. The veto MUST be subscribed
  // on each GROUP MODEL: that is the emitter whose defaultPrevented the group's
  // handleDropEvent checks before applying a drop — this dockview build routes
  // group willDrop/willShowOverlay only to an internal dnd service, never to
  // the api-level events, so an api.onWillDrop veto would see nothing.
  // onWillDrop is the hard gate (checked on every dnd strategy immediately
  // before the move); onWillShowOverlay gets the same predicate so a doomed
  // target never even paints a drop highlight.
  wireStackingVeto(dockApi);

  arrangeDefaultLayout(dockApi);

  wireTerminalRefit(dockApi);
  wirePointerShield(dockApi, root);

  // Fire-and-forget: the default layout above is already usable, so a slow or
  // failed /api/panels fetch just means "no persistence this session". A stored
  // layout for the current view, once fetched, is reconciled and applied over
  // the default.
  void initPersistence(dockApi);
}

/**
 * A drop geometry that would stack panels as tabs in one tile: onto a tab, the
 * tab strip's empty header space, or a tile's center. Edge drops (splits) are
 * the only accepted geometry under the one-panel-per-tile model.
 * @param {any} e  a dockview WillShowOverlay / WillDrop event
 * @returns {boolean}
 */
function isStackingDrop(e) {
  return (
    e.kind === 'tab' ||
    e.kind === 'header_space' ||
    (e.kind === 'content' && e.position === 'center')
  );
}

/**
 * Subscribe the stacking veto on every group's model — current groups and every
 * group dockview creates later. Group models are where dockview checks
 * defaultPrevented before applying a drop (see the call-site comment in
 * initDockWorkspace). Subscriptions live as long as their group; dockview
 * disposes the model's emitters with the group, so no manual teardown.
 * @param {any} api
 */
function wireStackingVeto(api) {
  const veto = (/** @type {any} */ e) => {
    if (isStackingDrop(e)) e.preventDefault();
  };
  const wireGroup = (/** @type {any} */ group) => {
    const model = group?.model;
    if (typeof model?.onWillDrop !== 'function') return;
    model.onWillDrop(veto);
    if (typeof model.onWillShowOverlay === 'function') model.onWillShowOverlay(veto);
  };
  api.onDidAddGroup(wireGroup);
  for (const group of api.groups ?? []) wireGroup(group);
}

/**
 * Build the default arrangement into an EMPTY-or-cleared grid: the terminal
 * card alone (full width until a service placeholder docks to its left at the
 * 60/40 split — see dock-iframe.js's ensurePlaceholder). Shared by first boot
 * and resetDockLayout(); clears first so a reset restores the default
 * regardless of the current arrangement. Ends by re-docking every visible
 * service placeholder via the dock-iframe hook, so a reset never strands a
 * visible service panel with no placeholder to follow.
 * @param {any} api
 */
function arrangeDefaultLayout(api) {
  api.clear();
  api.addPanel({
    id: PANEL_TERMINAL,
    component: PANEL_TERMINAL,
    title: TERMINAL_TITLE,
  });
  redockServices?.();
  syncAgentPane();
}

/**
 * Re-fit xterm after any settled layout change. dockview fires onDidLayoutChange
 * once per resolved change (panel add / move / close and sash-END — it emits
 * nothing during a live sash drag; the continuous refit while a sash is dragged
 * comes from terminal.js's ResizeObserver on .terminal-body, whose observed
 * parent is preserved by this shell). onDidActivePanelChange covers the terminal
 * card being revealed after it was hidden behind another tab in its group.
 * @param {any} api
 */
function wireTerminalRefit(api) {
  const refit = () => fitTerminal();
  api.onDidLayoutChange(refit);
  api.onDidActivePanelChange(refit);
}

/**
 * Pointer-shield for dockview drags. During an HTML5 tab/group drag, an iframe
 * (a separate browsing context) swallows the pointer/drag events that pass over
 * it, which would break dockview's drop-target detection over any panel hosting
 * one. dockview neutralizes iframe pointer-capture internally for its own
 * draggables; this mirrors and widens that guarantee across the whole drag
 * lifecycle by toggling a `.dock-dragging` state class on .main-container, under
 * which panel iframes get `pointer-events: none` (see terminal.css).
 *
 * A covering pointer-events overlay is deliberately NOT used: dockview attaches
 * its dragenter/dragover/drop listeners on each GROUP element, so a layer opaque
 * to pointers sitting above the groups would intercept those events and break
 * docking. Neutralizing the iframes — not the dockview chrome — is the shield.
 * @param {any} api
 * @param {HTMLElement} root
 */
function wirePointerShield(api, root) {
  const container = root.closest('.main-container');
  if (!(container instanceof HTMLElement)) return;
  onDragGesture(api, {
    onStart: () => container.classList.add('dock-dragging'),
    onEnd: () => container.classList.remove('dock-dragging'),
  });
}

// The gesture's natural terminators: dockview exposes no explicit drag-END api
// event, so both shields watch the document for these (covers completed AND
// cancelled drags, and both the HTML5 and pointer dnd strategies).
const DRAG_TERMINATORS = ['dragend', 'drop', 'mouseup', 'pointerup'];

/**
 * Run onStart when a dockview drag gesture begins (onWillDrag{Panel,Group}) and
 * onEnd when it ends. The end is inferred from DRAG_TERMINATORS on the document
 * since dockview emits no drag-END event; the terminator listeners are capture-
 * phase and self-removing. Re-entrancy-guarded so a gesture is raised at most
 * once. The single drag-boundary both drag shields share — this module's
 * `.dock-dragging` class toggle and dock-iframe.js's inline iframe pointer-events
 * (which the class rule can't reach past their inline style).
 * @param {any} api
 * @param {{ onStart: () => void, onEnd: () => void }} handlers
 * @returns {any[]} the onWillDrag disposables
 */
export function onDragGesture(api, { onStart, onEnd }) {
  let active = false;
  const raise = () => {
    if (active) return;
    active = true;
    onStart();
    const lower = () => {
      active = false;
      onEnd();
      for (const type of DRAG_TERMINATORS) document.removeEventListener(type, lower, true);
    };
    for (const type of DRAG_TERMINATORS) document.addEventListener(type, lower, true);
  };
  return [api.onWillDragPanel(raise), api.onWillDragGroup(raise)];
}

/**
 * Whether the active OSPREY theme is light, read from the computed
 * `color-scheme` (tokens.css sets it per html[data-theme]) rather than parsed
 * from the theme id, so it stays correct across every theme family.
 * @returns {boolean}
 */
function schemeIsLight() {
  return getComputedStyle(document.documentElement).colorScheme.trim() === 'light';
}

/**
 * Bind dockview's own light/dark base theme to the active OSPREY theme and keep
 * it live on every toggle — BOTH halves: the theme class on #dock-root (the
 * anchor for dockview-overrides.css's token remap) and the DockviewApi's own
 * `theme` option, which controls the class dockview stamps on its .dv-shell.
 * The .dv-shell element sits between the root and every panel, so leaving it on
 * dockview's default theme would shadow the root-level vars (see initDockWorkspace).
 * dockview.css defines a full `--dv-*` baseline per theme class, so this swap
 * keeps any var NOT remapped in dockview-overrides.css on a correct light/dark
 * default. The subscription is page-lifetime, matching the shell's other dock
 * wiring — the theme hub outlives the workspace.
 * @param {HTMLElement} root
 */
function wireDockTheme(root) {
  const apply = () => {
    const light = schemeIsLight();
    root.classList.toggle('dockview-theme-light', light);
    root.classList.toggle('dockview-theme-dark', !light);
    const dockview = /** @type {any} */ (window)['dockview-core'];
    if (dockApi && dockview) {
      dockApi.updateOptions({ theme: light ? dockview.themeLight : dockview.themeDark });
    }
  };
  apply();
  subscribe(apply);
}

/* ---- Agent pane presence (first-class open/close/reopen) ---- */

/**
 * Whether the Agent pane (the terminal/chat tile) is in the dock. Without a
 * dock (fallback mode, or before init) the card is always on the page, so it
 * counts as open.
 * @returns {boolean}
 */
export function isTerminalPanelOpen() {
  return dockApi ? Boolean(dockApi.getPanel(PANEL_TERMINAL)) : true;
}

/**
 * Publish the Agent pane's presence: `html[data-agent="closed"]` is present
 * exactly while the pane is absent from the dock, and `osprey-agent-pane`
 * `{detail: {open}}` is dispatched on `document` whenever that changes (and
 * once at boot). Called at the end of every path that adds or removes the
 * pane — open, close, the default arrangements, and a stored-layout restore.
 */
function syncAgentPane() {
  const open = isTerminalPanelOpen();
  if (open === agentPaneOpen) return;
  agentPaneOpen = open;
  if (open) document.documentElement.removeAttribute(AGENT_ATTRIBUTE);
  else document.documentElement.setAttribute(AGENT_ATTRIBUTE, 'closed');
  document.dispatchEvent(new CustomEvent('osprey-agent-pane', { detail: { open } }));
}

/**
 * Surface the Agent pane: focus it when present, re-create it when closed
 * (the sidebar Agent row's activate path). Re-adding by the same component
 * re-adopts the live terminal subtree — createComponent still holds the
 * source element, so the running xterm/chat state survives close → reopen
 * (the same mechanism resetDockLayout relies on). A reopened tile docks at
 * the grid's right edge, the pane's home side.
 */
export function openTerminalPanel() {
  if (!dockApi) return;
  const existing = dockApi.getPanel(PANEL_TERMINAL);
  if (existing) {
    existing.api?.setActive?.();
    return;
  }
  try {
    dockApi.addPanel({
      id: PANEL_TERMINAL,
      component: PANEL_TERMINAL,
      title: TERMINAL_TITLE,
      position: { direction: 'right' },
    });
  } catch (err) {
    console.error('dock: terminal reopen failed', err);
  }
  syncAgentPane();
}

/**
 * Close the Agent pane: the tile bar's × (dock-tab.js) and the sidebar row's
 * Delete-key path. The sidebar's Agent row is unaffected — it is the
 * permanent reopen affordance. Resolves whether a tile was closed, so a
 * caller offering an Undo offers one only for a close that happened.
 * @returns {boolean}
 */
export function closeTerminalPanel() {
  if (!dockApi) return false;
  const panel = dockApi.getPanel(PANEL_TERMINAL);
  if (!panel) return false;
  try {
    dockApi.removePanel(panel);
    return true;
  } catch (err) {
    console.error('dock: terminal close failed', err);
    return false;
  } finally {
    syncAgentPane();
  }
}

/* ---- Layout persistence (reconcile itself lives in dock-reconcile.js) ---- */

/**
 * The panels the dock can currently (re)create, derived from the live api so it
 * stays generic across whatever is registered (native terminal/workspace plus
 * any iframe placeholders adopted by the iframe-adapter task). Each value of the
 * serialized panels map is already a `{id, contentComponent, ...}` descriptor.
 * @param {any} api
 * @returns {PanelDescriptor[]}
 */
function currentRegisteredPanels(api) {
  try {
    return Object.values(api.toJSON().panels || {});
  } catch {
    return [];
  }
}

/**
 * Serialize the CURRENT layout to localStorage. No key ⇒ nothing is stored.
 */
function persistCurrentLayout() {
  const key = layoutKey();
  if (!dockApi || !key) return;
  try {
    localStorage.setItem(key, JSON.stringify(dockApi.toJSON()));
  } catch {
    /* storage blocked/full — the layout still lives for this session */
  }
}

/**
 * Debounce persist writes: onDidLayoutChange fires once per settled change, but a
 * single user gesture (or a fromJSON restore) can emit a burst; coalesce them
 * into one write on the trailing edge.
 */
function schedulePersist() {
  if (persistHandle != null) return;
  persistHandle = setTimeout(() => {
    persistHandle = null;
    persistCurrentLayout();
  }, 150);
}

/**
 * Load, reconcile, and apply the persisted layout. Reconciled against the
 * currently-registered panels, so panels added or removed since it was stored
 * are folded in. Returns whether a stored layout was applied: no stored value,
 * a corrupt one, or a failed apply all return false and leave the default grid
 * standing. A layout that parsed but would not apply is dropped so the next reload
 * starts clean.
 * @param {any} api
 * @returns {boolean}
 */
function applyStoredLayout(api) {
  const key = layoutKey();
  if (!key) return false;
  let raw;
  try {
    raw = localStorage.getItem(key);
  } catch {
    return false;
  }
  if (!raw) return false;

  const next = reconcile(raw, currentRegisteredPanels(api));
  if (!next) return false;
  try {
    api.fromJSON(next);
    ensureTerminalPanel(api);
    normalizeTerminalTitle(api);
    // Let the adapter re-sync: ensure visible service placeholders exist and
    // prune any restored placeholder whose service no longer exists.
    redockServices?.();
    syncAgentPane();
    return true;
  } catch (err) {
    console.error('dock: stored layout failed to apply; using the default', err);
    try {
      localStorage.removeItem(key);
    } catch {
      /* non-fatal */
    }
    return false;
  }
}

/**
 * Resolve the per-project storage key, apply any stored layout over the
 * default, then wire settled-change persistence.
 * onDidLayoutChange is subscribed AFTER the stored layout is applied so the
 * restore itself doesn't trigger a redundant write-back.
 *
 * Every exit announces the boot layout as final through the layoutRestored hook
 * — including the no-project-key early return, where the default arrangement IS
 * the final one. Missing an exit would leave the occupancy reporter waiting on
 * its fallback deadline, so the announcement is unconditional rather than tied
 * to a restore having happened.
 * @param {any} api
 */
async function initPersistence(api) {
  let projectKey = null;
  try {
    const info = await fetchJSON('/api/panels');
    if (info && typeof info.project_key === 'string' && info.project_key) {
      projectKey = info.project_key;
    }
  } catch {
    /* offline / endpoint error — no persistence this session */
  }
  if (!projectKey) {
    announceLayoutRestored();
    return;
  }
  layoutStorageKeyBase = LAYOUT_KEY_PREFIX + projectKey;

  // Restore the stored arrangement over the default. With nothing stored, the
  // default is the grid initDockWorkspace already built. Panels registered AFTER
  // this point (late iframe placeholders / agent SSE) are folded in by the iframe
  // adapter and dock-sync, not here.
  applyStoredLayout(api);
  announceLayoutRestored();
  api.onDidLayoutChange(schedulePersist);
}

/**
 * Clear the stored layout and restore the default arrangement live. Bound to the command palette's "Reset layout" row and the terminal
 * tile's context menu.
 */
export function resetDockLayout() {
  const key = layoutKey();
  if (key) {
    try {
      localStorage.removeItem(key);
    } catch {
      /* non-fatal */
    }
  }
  if (dockApi) arrangeDefaultLayout(dockApi);
}

/**
 * Re-add the terminal card when a restored layout lacks it, docked at the
 * grid's right edge (its home side). reconcile() no longer appends missing
 * panels into the serialized layout — the live api sizes a fresh tile
 * correctly, a JSON-appended node would not — so the restore paths call this
 * right after fromJSON. Keeps the pre-existing behavior that a reload always
 * comes back with the terminal present.
 * @param {any} api
 */
function ensureTerminalPanel(api) {
  if (api.getPanel(PANEL_TERMINAL)) return;
  try {
    api.addPanel({
      id: PANEL_TERMINAL,
      component: PANEL_TERMINAL,
      title: TERMINAL_TITLE,
      position: { direction: 'right' },
    });
  } catch (err) {
    console.error('dock: terminal re-add after restore failed', err);
  }
}

/**
 * Stamp the Agent tab's current title onto a restored layout. fromJSON
 * restores titles verbatim, so a layout persisted under an older title (the
 * empty string from before the tab was titled, or `SESSION`) would keep
 * resurrecting it from storage without this.
 * @param {any} api
 */
function normalizeTerminalTitle(api) {
  const panel = api.getPanel(PANEL_TERMINAL);
  if (!panel || panel.title === TERMINAL_TITLE) return;
  if (typeof panel.setTitle === 'function') panel.setTitle(TERMINAL_TITLE);
  else panel.api?.setTitle?.(TERMINAL_TITLE);
}

/* ---- Fallback tile bar (no dockview) ---- */

/**
 * Give the fallback iframe host the tile bar a docked tile has. Embed contract
 * v2 promises an embedded panel exactly one header — the hub's tile bar — and
 * panels on it hide their in-body copies of whatever they contribute, so a
 * host without a bar would leave those controls nowhere. The fallback shows
 * one panel at a time (#right-panel's `data-active-panel`), so it carries one
 * bar: the active panel's name and its contribution region, registered with
 * tile-header-contrib.js exactly as a docked TileTab registers its own. The
 * bar follows the active panel and hides only when none is active.
 *
 * The Agent card needs nothing here: without a dock it is never adopted, so
 * its bar stays at the top of the card — the console bar in chat (the card's
 * `.terminal-header` is hidden there, operator.css), the header on the terminal.
 *
 * The bar reuses the docked bar's `.tile-tab` markup so it renders with the
 * same styles; the inline layout only adapts that full-height tab to a row
 * above #panel-content. Idempotent per host.
 * @param {HTMLElement} host  the fallback iframe host (#right-panel)
 * @returns {HTMLElement | null} the bar, or null when the host has no #panel-content
 */
export function initFallbackTileBar(host) {
  const existing = host.querySelector(':scope > .fallback-tile-bar');
  if (existing instanceof HTMLElement) return existing;
  const content = host.querySelector('#panel-content, .panel-content');
  if (!(content instanceof HTMLElement)) return null;

  host.style.flexDirection = 'column';
  const bar = document.createElement('div');
  bar.className = 'tile-tab fallback-tile-bar';
  bar.style.flex = '0 0 var(--wt-pane-head-height)';
  bar.style.height = 'var(--wt-pane-head-height)';
  bar.style.cursor = 'default';
  const title = document.createElement('span');
  title.className = 'tile-tab-title';
  bar.appendChild(title);
  host.insertBefore(bar, content);

  /** @type {{ panelId: string, el: HTMLElement } | null} */
  let registered = null;

  const sync = () => {
    const panelId = host.dataset.activePanel || null;
    if (registered && registered.panelId !== panelId) {
      unregisterContribHost(registered.panelId, registered.el);
      registered.el.remove();
      registered = null;
    }
    if (panelId && !registered) {
      // A fresh region per panel: registerContribHost wires its listeners on
      // the element, so reusing one across panels would stack them.
      const el = document.createElement('div');
      el.className = 'tile-tab-contrib';
      bar.appendChild(el);
      registerContribHost(panelId, el);
      registered = { panelId, el };
    }
    title.textContent = panelId ? panelLabel(panelId) : '';
    // Inline display, not `hidden`: the .tile-tab rule's display:flex would
    // outrank the attribute's user-agent display:none.
    bar.style.display = panelId ? '' : 'none';
  };

  sync();
  new MutationObserver(sync).observe(host, {
    attributes: true,
    attributeFilter: ['data-active-panel'],
  });
  return bar;
}

/**
 * A panel's display name, read from its sidebar row (the one place the hub
 * renders it outside a tile); the id when the sidebar does not carry the panel.
 * @param {string} panelId
 * @returns {string}
 */
function panelLabel(panelId) {
  const btn = document.querySelector(`.panel-rail-button[data-panel-id="${CSS.escape(panelId)}"]`);
  return btn?.getAttribute('aria-label') || panelId;
}

/**
 * The live DockviewApi, or null before initDockWorkspace() has run. Exposed for
 * the follow-on dock tasks (state-sync / persistence).
 * @returns {any}
 */
export function getDockApi() {
  return dockApi;
}
