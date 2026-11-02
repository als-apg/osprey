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
import {
  TERMINAL_KEY,
  fromSerialized,
  getLayoutHolder,
  initLayoutHolder,
  keyOfPanelId,
  panelIdOf,
} from './dock-apply.js';
import { closeKey, defaultLayout, reopenTerminal } from './dock-open.js';
import { withEchoSuppressed } from './dock-sync.js';
import { onPopoutChange } from './popout.js';
import { registerContribHost, unregisterContribHost } from './tile-header-contrib.js';
import { keysOf } from '/static/vendor/dock-engine.mjs';
import { subscribe } from '/design-system/js/theme-manager.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('./dock-apply.js').LayoutHolder} LayoutHolder */
/** @typedef {import('./dock-apply.js').PanelRecord} PanelRecord */

/**
 * The persisted dock: the holder's dock layout and its panel records by bare
 * key. Any other `version` is not read (the page opens its default), except a
 * dockview serialization stored by the dockview-era shell, which the repair
 * reads into this form.
 * @typedef {object} StoredDock
 * @property {2} version
 * @property {Layout|null} layout
 * @property {Record<string, PanelRecord>} panels
 */

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

/** The page's layout holder once initDockWorkspace has created it.
 *  @type {LayoutHolder|null} */
let layoutHolder = null;

/** The stored-dock format this module reads and writes. */
const STORE_VERSION = 2;

/** Panel records the last restore brought in, by bare key: the title and
 *  params a restored tab is built with until dockview holds its own entry.
 *  @type {Record<string, PanelRecord>} */
let restoredPanels = {};

/** Panel ids GET /api/panels lists as buildable, or null while unknown.
 *  @type {string[]|null} */
let serverPanelIds = null;

const PANEL_TERMINAL = TERMINAL_KEY;

/** Attribute on <html> present exactly while the Agent pane is absent from
 *  the dock (`data-agent="closed"`); the sidebar's Agent row reads it. */
const AGENT_ATTRIBUTE = 'data-agent';

/** Whether the Agent pane was open at the last {@link syncAgentPane}, or null
 *  before the first sync. @type {boolean | null} */
let agentPaneOpen = null;

/** localStorage key prefix; the per-project key from GET /api/panels is appended
 *  so one browser origin persists a separate layout per project. */
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

/**
 * Callback the iframe adapter registers (the managed-iframe set lives there),
 * invoked after EVERY whole-layout apply — default rebuild, stored-layout
 * restore. It re-docks any visible service placeholder the applied
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
 * The host is built once: every apply rebuilds the dockview panel, and each
 * rebuild re-adopts the same host, so the subtree is moved, never rebuilt.
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
 * left at the classic 60/40 split. Every change to the dock goes through the
 * layout holder (dock-apply.js), the one writer of dockview. The legacy
 * #right-panel column is NOT docked — in the docked shell it is only the
 * fallback iframe host, so it is hidden here to keep its `display: contents`
 * holder from laying it out beside the grid. No-ops (leaving the rest of boot
 * intact — fallback mode, #right-panel visible) if the root or terminal subtree
 * is missing; when only the vendored dockview global failed to load, the
 * fallback host gets its tile bar ({@link initFallbackTileBar}) so embedded
 * panels keep their one header.
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

  const terminalRenderer = adoptSubtree(terminalSource);
  dockApi = dockview.createDockview(root, {
    // Without an explicit theme dockview stamps its DEFAULT (abyss) class on
    // the .dv-shell it creates inside #dock-root — a closer ancestor than the
    // classes wireDockTheme manages on the root, so abyss's dark `--dv-*`
    // values would shadow the light theme for every descendant. Start on the
    // matching base theme; wireDockTheme keeps it live from then on.
    theme: schemeIsLight() ? dockview.themeLight : dockview.themeDark,
    createComponent: (/** @type {{ name: string }} */ options) => {
      if (options.name === PANEL_TERMINAL) return terminalRenderer;
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

  // The holder applies every layout inside dock-sync's echo guard, so the
  // rebuild's own focus changes are never reported as a person's click. A
  // restored tab is built with the record its store carried.
  layoutHolder = initLayoutHolder({
    api: dockApi,
    guard: withEchoSuppressed,
    newPanel: (key) => restoredPanels[key],
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

  wireHolderSync(dockApi, layoutHolder);

  arrangeDefaultLayout();

  wireTerminalRefit(dockApi);
  wirePointerShield(dockApi, root);

  // Fire-and-forget: the default layout above is already usable, so a slow or
  // failed /api/panels fetch just means "no persistence this session". A stored
  // layout, once fetched, is repaired and applied over the default.
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
 * Keep the holder in step with what dockview does on its own.
 *
 * dockview still owns two gestures the holder does not compute: its
 * drag-and-drop of a tile (a split elsewhere in the grid) and a sash drag. Both
 * settle in onDidLayoutChange, where the grid is read back: a tree of the held
 * shape carries new sizes, any other tree is the one dockview built, and the
 * holder follows it without re-applying. An apply of the holder's own lands
 * here too, as the same tree it just wrote, and changes nothing.
 *
 * The focused tile follows dockview's active panel (a tab click, the focus of
 * a rebuilt group); applies are held while a drag is live; and a panel moving
 * to or from its own pop-out window is recorded on the holder, which is the
 * layout the agent reads.
 * @param {any} api
 * @param {LayoutHolder} holder
 */
function wireHolderSync(api, holder) {
  api.onDidActivePanelChange(() => {
    const active = api.activePanel;
    const key = keyOfPanelId(active?.id);
    const tile = active?.group?.id;
    if (key !== null && typeof tile === 'string') holder.show(tile, key);
  });
  api.onDidLayoutChange(() => {
    let next;
    try {
      next = fromSerialized(api.toJSON());
    } catch {
      return;
    }
    if (next === null && holder.layout !== null) return; // a grid mid-rebuild says nothing
    holder.adopt(next);
  });
  onDragGesture(api, {
    onStart: () => holder.setDragLive(true),
    onEnd: () => holder.setDragLive(false),
  });
  onPopoutChange((panelId, poppedOut) => holder.setPoppedOut(panelId, poppedOut));
}

/**
 * Hand a whole layout to the holder, which validates it and applies it with
 * one `fromJSON`, then let the iframe adapter re-sync: it re-docks any visible
 * service placeholder the layout lacks and prunes placeholders whose service
 * no longer exists, so no apply strands a visible panel or leaves a dead tab.
 * False (nothing changes) for an invalid layout.
 * @param {Layout|null} layout
 * @returns {boolean}
 */
function applyHeld(layout) {
  if (!layoutHolder || !layoutHolder.restore(layout)) return false;
  redockServices?.();
  syncAgentPane();
  return true;
}

/**
 * Build the default arrangement: the terminal card alone (full width until a
 * service placeholder docks to its left at the 60/40 split — see
 * dock-open.js's firstTile), replacing whatever the dock holds. Shared by
 * first boot and resetDockLayout().
 */
function arrangeDefaultLayout() {
  applyHeld(defaultLayout());
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
  return layoutHolder ? keysOf(layoutHolder.layout).includes(PANEL_TERMINAL) : true;
}

/**
 * Publish the Agent pane's presence: `html[data-agent="closed"]` is present
 * exactly while the pane is absent from the dock, and `osprey-agent-pane`
 * `{detail: {open}}` is dispatched on `document` whenever that changes (and
 * once at boot). Called at the end of every path that adds or removes the
 * pane — open, close, the default arrangement, and a stored-layout restore.
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
  const holder = layoutHolder;
  if (!dockApi || !holder) return;
  const result = holder.run((layout, ctx) => reopenTerminal(layout, ctx.dock));
  if (!result.changed) {
    // Already in its tile: dockview only has to bring the tile forward.
    dockApi.getPanel(PANEL_TERMINAL)?.api?.setActive?.();
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
  const holder = layoutHolder;
  if (!dockApi || !holder) return false;
  const closed = holder.run((layout) => closeKey(layout, PANEL_TERMINAL)).changed;
  syncAgentPane();
  return closed;
}

/**
 * Close a service panel's tile — the tile bar's × on a service tile. Local
 * only: the panel keeps its sidebar row (dock-sync routes the gesture to
 * panel-manager's vacate first, in the capture phase). A key not in the dock
 * changes nothing.
 * @param {string} serviceId  the panel's bare id
 * @returns {boolean} whether a tab was closed
 */
export function closeDockPanel(serviceId) {
  const holder = layoutHolder;
  if (!dockApi || !holder) return false;
  return holder.run((layout) => closeKey(layout, serviceId)).changed;
}

/* ---- Layout persistence (the repair itself lives in dock-reconcile.js) ---- */

/**
 * The per-panel record reopening a stored layout in a new session needs, by
 * bare key: the title and params each service tab carries (the terminal is
 * the page's own and needs none). Read off the live dock, falling back to the
 * record a restore brought in for a tab dockview has not built yet.
 * @param {Layout|null} layout
 * @returns {Record<string, PanelRecord>}
 */
function panelRecords(layout) {
  /** @type {Record<string, any>} */
  let live = {};
  try {
    live = dockApi?.toJSON?.().panels ?? {};
  } catch {
    /* an unreadable dock records nothing beyond what was restored */
  }
  /** @type {Record<string, PanelRecord>} */
  const out = {};
  if (layout === null) return out;
  for (const key of keysOf(layout)) {
    if (key === PANEL_TERMINAL) continue;
    const entry = live[panelIdOf(key)];
    /** @type {PanelRecord|undefined} */
    const record = entry
      ? {
          ...(typeof entry.title === 'string' ? { title: entry.title } : {}),
          ...(entry.params != null ? { params: entry.params } : {}),
        }
      : restoredPanels[key];
    if (record) out[key] = record;
  }
  return out;
}

/**
 * The value persisted: the holder's layout and its panel records.
 * @returns {StoredDock|null}  null before the holder exists
 */
function storedValue() {
  if (!layoutHolder) return null;
  const layout = layoutHolder.layout;
  return { version: STORE_VERSION, layout, panels: panelRecords(layout) };
}

/**
 * Read the stored value. A `{version: 2, layout, panels}` object is read as
 * is; a dockview serialization stored by the dockview-era shell is handed to
 * the repair as the layout, with its panel entries as the records. Anything
 * else — no value, unparseable JSON, another version — is null: the caller
 * opens the default, with no warning.
 * @returns {{ layout: unknown, panels: Record<string, PanelRecord> } | null}
 */
function readStored() {
  const key = layoutKey();
  if (!key) return null;
  let value;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!value || typeof value !== 'object') return null;
  if (value.version === STORE_VERSION && 'layout' in value) {
    const panels = value.panels && typeof value.panels === 'object' ? value.panels : {};
    return { layout: value.layout, panels };
  }
  if (!('version' in value) && value.grid && value.panels && typeof value.panels === 'object') {
    /** @type {Record<string, PanelRecord>} */
    const panels = {};
    for (const [id, entry] of Object.entries(value.panels)) {
      const panelKey = keyOfPanelId(id);
      if (panelKey === null || panelKey === PANEL_TERMINAL || !entry || typeof entry !== 'object') {
        continue;
      }
      const e = /** @type {any} */ (entry);
      panels[panelKey] = {
        ...(typeof e.title === 'string' ? { title: e.title } : {}),
        ...(e.params != null ? { params: e.params } : {}),
      };
    }
    return { layout: value, panels };
  }
  return null;
}

/**
 * The keys the dock can build now: the panels the server lists (enabled
 * built-ins and custom panels) plus every key already in the dock. Before the
 * server's list is known (or when it is absent) the stored layout's own keys
 * are trusted, so nothing is pruned on a guess.
 * @param {unknown} stored
 * @returns {Set<string>}
 */
function buildableKeys(stored) {
  const keys = new Set(layoutHolder?.layout ? keysOf(layoutHolder.layout) : []);
  const known = serverPanelIds ?? safeKeys(stored);
  for (const key of known) keys.add(key);
  return keys;
}

/**
 * `keysOf` for an unvalidated value: [] when it is not a readable layout.
 * @param {any} value
 * @returns {string[]}
 */
function safeKeys(value) {
  try {
    const layout = value && typeof value === 'object' && value.grid ? fromSerialized(value) : value;
    return layout ? keysOf(layout) : [];
  } catch {
    return [];
  }
}

/**
 * Load, repair and apply the persisted layout through the holder. The repair
 * (dock-reconcile.js) closes every key the dock cannot build any more; a
 * layout that lost the terminal gets it back at the dock's right edge
 * (dock-open.js's reopenTerminal), so a restore always shows the terminal.
 * Returns whether a stored layout was applied: no value, another version, an
 * invalid layout or one with nothing left all return false and leave the
 * default grid standing (the next change overwrites the value). A layout that
 * would not apply is dropped so the next reload starts clean.
 * @returns {boolean}
 */
function applyStoredLayout() {
  const holder = layoutHolder;
  const stored = readStored();
  if (!holder || !stored) return false;
  let layout = reconcile(stored.layout, buildableKeys(stored.layout));
  if (!layout) return false;
  if (!keysOf(layout).includes(PANEL_TERMINAL)) {
    layout = reopenTerminal(layout, holder.dock()).layout;
  }
  const previous = restoredPanels;
  restoredPanels = { ...stored.panels };
  if (applyHeld(layout)) return true;
  restoredPanels = previous;
  forgetStored();
  return false;
}

/** Drop the stored value, so the next reload starts from the default. */
function forgetStored() {
  const key = layoutKey();
  if (!key) return;
  try {
    localStorage.removeItem(key);
  } catch {
    /* non-fatal */
  }
}

/**
 * Write the holder's CURRENT layout to the persisted value. No key ⇒ nothing
 * is stored.
 */
function persistCurrentLayout() {
  const key = layoutKey();
  const value = storedValue();
  if (!key || !value) return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage blocked/full — the layout still lives for this session */
  }
}

/**
 * Debounce persist writes: a single gesture (or an apply) can emit a burst of
 * dockview changes; coalesce them into one write on the trailing edge.
 */
function schedulePersist() {
  if (persistHandle != null) return;
  persistHandle = setTimeout(() => {
    persistHandle = null;
    persistCurrentLayout();
  }, 150);
}

/**
 * Resolve the per-project storage key and the panels the dock can build, apply
 * any stored layout over the default, then wire settled-change persistence
 * (layout changes and tab switches). The listeners are subscribed AFTER the
 * stored layout is applied so the restore itself doesn't trigger a redundant
 * write-back.
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
    serverPanelIds = panelIdsOf(info);
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
  applyStoredLayout();
  announceLayoutRestored();
  api.onDidLayoutChange(schedulePersist);
  api.onDidActivePanelChange(schedulePersist);
}

/**
 * The panel ids GET /api/panels lists as buildable — enabled built-ins and
 * custom panels — or null when the payload carries neither list.
 * @param {any} info
 * @returns {string[]|null}
 */
function panelIdsOf(info) {
  const enabled = Array.isArray(info?.enabled) ? info.enabled : null;
  const custom = Array.isArray(info?.custom) ? info.custom : null;
  if (!enabled && !custom) return null;
  /** @type {string[]} */
  const ids = [];
  for (const item of [...(enabled ?? []), ...(custom ?? [])]) {
    const id = typeof item === 'string' ? item : item?.id;
    if (typeof id === 'string' && id) ids.push(id);
  }
  return ids;
}

/**
 * Clear the stored layout and restore the default arrangement live, through
 * the holder. Bound to the command palette's "Reset layout" row and the
 * terminal tile's context menu.
 */
export function resetDockLayout() {
  forgetStored();
  if (dockApi) arrangeDefaultLayout();
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
 * the follow-on dock tasks (state-sync / persistence), which READ it; every
 * change goes through the holder ({@link getLayoutHolder}).
 * @returns {any}
 */
export function getDockApi() {
  return dockApi;
}

export { getLayoutHolder };
