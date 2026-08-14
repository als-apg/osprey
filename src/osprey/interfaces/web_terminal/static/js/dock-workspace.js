/* OSPREY Web Terminal — Dockview Workspace Shell */

import { fitTerminal } from './terminal.js';
import { fetchJSON } from './api.js';
import { reconcile, PLACEHOLDER_PREFIX } from './dock-reconcile.js';
import { createTileTab, OSPREY_TAB_COMPONENT, setTerminalHeaderSource } from './dock-tab.js';
import { subscribe } from '/design-system/js/theme-manager.js';

/** @typedef {import('./dock-reconcile.js').PanelDescriptor} PanelDescriptor */

/**
 * @typedef {object} ContentRenderer
 * @property {HTMLElement} element  Node dockview appends into its content container.
 * @property {(params: any) => void} init  dockview lifecycle hook (no-op here).
 */

/**
 * Live DockviewApi once initialized, else null. Kept module-scoped so the
 * follow-on dock tasks (state-sync, persistence, mode-transition) can reach the
 * same instance through getDockApi().
 * @type {any}
 */
let dockApi = null;

const PANEL_TERMINAL = 'terminal';

/** Dock tab title for the terminal/chat card. A real title (rather than the
 *  earlier empty-string de-labeling) makes the terminal a first-class tile:
 *  alone in its tile the single-tab strip is collapsed anyway (see
 *  dockview-overrides.css), and stacked with service tabs it must be
 *  identifiable. The card's own `.terminal-header` remains the in-panel label. */
const TERMINAL_TITLE = 'SESSION';

/** localStorage key prefix; the per-project key from GET /api/panels is appended
 *  so one browser origin persists a separate expert layout per project. */
const LAYOUT_KEY_PREFIX = 'osprey-dock-layout-';

/** Resolved localStorage key (prefix + project_key), or null until initPersistence
 *  has fetched the project key. No key ⇒ persistence is inert this session.
 *  @type {string|null} */
let layoutStorageKey = null;

/** Pending debounce handle for a persist write (see schedulePersist).
 *  @type {ReturnType<typeof setTimeout>|null} */
let persistHandle = null;

/** Width of the left icon rail (see .panel-rail-region in terminal.css). */
const RAIL_WIDTH = 74;

/** Fraction of the workspace the terminal takes by default — mirrors the old
 *  fixed split (terminal `flex: 0 1 40%`, workspace `flex: 1 1 60%`). */
const TERMINAL_WIDTH_FRACTION = 0.4;

/**
 * Pixel width for the service-panel group when it first docks beside the
 * terminal — the service side of the old fixed split (terminal 40% / services
 * 60%). Consumed by dock-iframe.js's ensurePlaceholder and the locked simple
 * layout below, so both modes derive the same split from one definition.
 * @returns {number}
 */
export function defaultServiceWidth() {
  // The rail consumes horizontal space only as a left column; in top
  // position it's a strip above the workspace, so the full width is ours.
  const rail =
    document.documentElement.getAttribute('data-rail-position') === 'top' ? 0 : RAIL_WIDTH;
  return Math.max(0, Math.round((window.innerWidth - rail) * (1 - TERMINAL_WIDTH_FRACTION)));
}

/**
 * Callback the iframe adapter registers (the managed-iframe set lives there),
 * invoked after EVERY layout apply — default rebuild, stored-layout restore,
 * mode-flip restore. It re-docks any visible service placeholder the applied
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
 * whether that is a restored expert layout, the synthesized simple layout, or
 * the plain default because no project key came back.
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

/** Announce the boot layout as final, latching it for any later subscriber. */
function announceLayoutRestored() {
  layoutRestoredAlready = true;
  layoutRestored?.();
}

/**
 * Wrap an already-rendered page subtree as a dockview content renderer. dockview
 * MOVES `element` into its content container (a DOM relocation, not a clone), so
 * every reference terminal.js / chat.js / panel-manager.js already hold into the
 * subtree stays live and its ids/classes stay stable (the Phase-A CSS mode gate
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
  return { element: host, init() { /* content is pre-rendered page DOM */ } };
}

/**
 * Initialize the dockview workspace in #dock-root and dock the terminal/chat
 * card as the shell's one native panel; service panels arrive as `iframe:`
 * placeholders (dock-iframe.js), the first of which docks to the terminal's
 * left at the classic 60/40 split. The legacy #right-panel column is NOT
 * docked — in the docked shell it is only the fallback iframe host, so it is
 * hidden here to keep its `display: contents` holder from laying it out
 * beside the grid. No-ops (leaving the rest of boot intact — fallback mode,
 * #right-panel visible) if the root or terminal subtree is missing, or if the
 * vendored dockview global failed to load.
 */
export function initDockWorkspace() {
  // Idempotent: a second call is a no-op once the grid exists, so the follow-on
  // dock tasks (persistence, mode-transition) can import from this file without
  // risking a re-entry that would double-adopt the source subtrees.
  if (dockApi) return;

  const root = document.getElementById('dock-root');
  const workspaceSource = document.getElementById('right-panel');
  const terminalSource = document.querySelector('.terminal-panel');
  if (!root || !(terminalSource instanceof HTMLElement)) return;

  // Grab the live .terminal-header reference NOW, while terminalSource is
  // still attached to the document — createComponent's adoptSubtree (below)
  // moves this whole subtree into a detached host div ahead of dockview's own
  // insertion of that host. dockview always builds a panel's content
  // component before its tab component, so a later `document.querySelector`
  // in dock-tab.js is guaranteed to miss the header at that point, not merely
  // liable to. See dock-tab.js's setTerminalHeaderSource for the consuming
  // side.
  const terminalHeaderSource = terminalSource.querySelector('.terminal-header');
  if (terminalHeaderSource instanceof HTMLElement) {
    setTerminalHeaderSource(terminalHeaderSource);
  } else {
    console.error('.terminal-header not found; terminal tile will render without its header bar');
  }

  const dockview = /** @type {any} */ (window)['dockview-core'];
  if (!dockview || typeof dockview.createDockview !== 'function') {
    console.error('dockview-core global unavailable; workspace shell not initialized');
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

  // One panel per tile: the rail is the workspace's tab system, so no drop may
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
  // expert layout, once fetched, is reconciled and applied over the default.
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
  return e.kind === 'tab' || e.kind === 'header_space'
    || (e.kind === 'content' && e.position === 'center');
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

/* ---- Terminal tile presence (first-class open/close/reopen) ---- */

/** @returns {boolean} whether the terminal tile currently exists in the grid. */
export function isTerminalOpen() {
  return !!dockApi?.getPanel(PANEL_TERMINAL);
}

/**
 * Surface the terminal tile: focus it when present, re-create it when closed
 * (the rail entry's activate path). Re-adding by the same component re-adopts
 * the live terminal subtree — createComponent still holds the source element,
 * so the running xterm/chat state survives close → reopen (the same mechanism
 * resetDockLayout relies on). A reopened tile docks at the grid's right edge,
 * the terminal's home side. Reopening is expert-only: the locked simple layout
 * always contains the terminal, so there is nothing to reopen there.
 */
export function openTerminalPanel() {
  if (!dockApi) return;
  const existing = dockApi.getPanel(PANEL_TERMINAL);
  if (existing) {
    existing.api?.setActive?.();
    return;
  }
  if (currentUiMode() !== 'expert') return;
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
}

/**
 * Close the terminal tile (the rail entry's Delete-key path). The dock tab's
 * own "×" goes through dockview directly; both end at the same removePanel.
 * The rail's SESSION entry is unaffected — it is the permanent reopen
 * affordance. No-op in simple mode — the locked layout is not user-editable.
 */
export function closeTerminalPanel() {
  if (!dockApi || currentUiMode() !== 'expert') return;
  const panel = dockApi.getPanel(PANEL_TERMINAL);
  if (!panel) return;
  try {
    dockApi.removePanel(panel);
  } catch (err) {
    console.error('dock: terminal close failed', err);
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

/** @returns {'expert'|'simple'} the UI mode from the authoritative <html> attribute. */
function currentUiMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple' ? 'simple' : 'expert';
}

/**
 * Serialize the CURRENT layout to localStorage — EXPERT mode only. The locked
 * simple layout is synthesized, never persisted (the mode-transition task owns
 * the synthesizer): while simple is active the stored value stays the stashed
 * expert layout, so a reload in simple mode restores the arrangement intact.
 */
function persistCurrentLayout() {
  if (!dockApi || !layoutStorageKey || currentUiMode() !== 'expert') return;
  try {
    localStorage.setItem(layoutStorageKey, JSON.stringify(dockApi.toJSON()));
  } catch { /* storage blocked/full — the layout still lives for this session */ }
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
 * Load, reconcile, and apply a persisted expert layout over the freshly-built
 * default. No stored value, a corrupt one, or a failed apply all leave the
 * default in place (full fallback only on parse/apply failure). A layout that
 * parsed but would not apply is dropped so the next reload starts clean.
 * @param {any} api
 */
function applyStoredLayout(api) {
  if (!layoutStorageKey) return;
  let raw;
  try {
    raw = localStorage.getItem(layoutStorageKey);
  } catch {
    return;
  }
  if (!raw) return;

  const next = reconcile(raw, currentRegisteredPanels(api));
  if (!next) return;
  try {
    api.fromJSON(next);
    ensureTerminalPanel(api);
    normalizeTerminalTitle(api);
    // Let the adapter re-sync: ensure visible service placeholders exist and
    // prune any restored placeholder whose service no longer exists.
    redockServices?.();
  } catch (err) {
    console.error('dock: stored layout failed to apply; keeping default', err);
    try {
      localStorage.removeItem(layoutStorageKey);
    } catch { /* non-fatal */ }
  }
}

/**
 * Resolve the per-project storage key, apply any stored expert layout over the
 * default, then wire settled-change persistence. onDidLayoutChange is subscribed
 * AFTER the stored layout is applied so the restore itself doesn't trigger a
 * redundant write-back.
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
  } catch { /* offline / endpoint error — no persistence this session */ }
  if (!projectKey) {
    announceLayoutRestored();
    return;
  }
  layoutStorageKey = LAYOUT_KEY_PREFIX + projectKey;

  // Boot into the mode the server rendered. In simple mode the locked layout is
  // synthesized fresh (leaving the stored EXPERT value untouched for a later flip
  // back); in expert mode the stored arrangement is restored over the default.
  // Panels registered AFTER this point (late iframe placeholders / agent SSE) are
  // folded in by the iframe adapter and dock-sync, not here.
  if (currentUiMode() === 'simple') {
    applySimpleLayout(api);
  } else {
    applyStoredLayout(api);
  }
  announceLayoutRestored();
  api.onDidLayoutChange(schedulePersist);
}

/**
 * Clear the stored layout and restore the default arrangement live. Exposed for
 * a "Reset layout" control (not yet bound to any chrome); the browser suite
 * exercises it as the canonical reset path.
 */
export function resetDockLayout() {
  if (layoutStorageKey) {
    try {
      localStorage.removeItem(layoutStorageKey);
    } catch { /* non-fatal */ }
  }
  if (dockApi) arrangeDefaultLayout(dockApi);
}

/* ---- Mode transition (expert ⇄ simple) ---- */

/**
 * Apply the dock half of a UI-mode flip. Called by app.js's initModeToggle AFTER
 * the html[data-ui-mode] swap, so currentUiMode() already reflects the target.
 * No-ops before the workspace exists. Both directions are live — no reload.
 *  - expert→simple: stash the current expert layout as the persisted value, then
 *    apply the synthesized LOCKED layout (one tab-stack of every service panel |
 *    terminal/chat right; drag/float/close/resize off).
 *  - simple→expert: unlock, then reconcile + restore the stashed expert layout
 *    (falling back to the default arrangement if none is usable).
 * @param {'expert'|'simple'} mode
 */
export function applyDockMode(mode) {
  if (!dockApi) return;
  if (mode === 'simple') {
    stashExpertLayout(dockApi);
    applySimpleLayout(dockApi);
  } else {
    restoreExpertLayout(dockApi);
    unlockLayout(dockApi);
  }
}

/**
 * Write the CURRENT (expert) layout straight to the persisted value, bypassing
 * the expert-only guard on schedulePersist. Called at the instant of the
 * expert→simple flip — the arrangement is still the expert one (only the mode
 * attribute has changed) — so the stash IS the stored layout: a reload while in
 * simple mode restores it intact. No key ⇒ simple→expert falls back to default.
 * @param {any} api
 */
function stashExpertLayout(api) {
  if (!layoutStorageKey) return;
  try {
    localStorage.setItem(layoutStorageKey, JSON.stringify(api.toJSON()));
  } catch { /* storage blocked — simple→expert will fall back to the default */ }
}

/**
 * Synthesize and apply the LOCKED simple layout: ONE service tile on the left,
 * the terminal/chat card on the right (the chat keeps the right-hand column in
 * both modes, mirroring the expert default). One panel per tile is a workspace
 * invariant and the rail is the tab system, so the tile carries exactly the
 * service placeholder that was showing (preferring the focused one) — the
 * other panels stay one rail click from taking the tile over. The placeholder
 * is re-created by its own id, which is what lets the iframe adapter re-find
 * and re-overlay it (it resolves placeholders by id on every settle). Locks
 * last so every group and sash the rebuild created is covered.
 * @param {any} api
 */
function applySimpleLayout(api) {
  const registered = currentRegisteredPanels(api);
  const terminal = registered.find((p) => p.id === PANEL_TERMINAL)
    ?? { id: PANEL_TERMINAL, contentComponent: PANEL_TERMINAL };
  const placeholders = registered.filter((p) => p.id.startsWith(PLACEHOLDER_PREFIX));
  const activeId = api.activePanel?.id;
  const service = placeholders.find((p) => p.id === activeId) ?? placeholders[0] ?? null;

  api.clear();
  addPanelFromDescriptor(api, { ...terminal, title: TERMINAL_TITLE });
  if (service) {
    addPanelFromDescriptor(
      api,
      service,
      { referencePanel: PANEL_TERMINAL, direction: 'left' },
      defaultServiceWidth(),
    );
  }

  lockLayout(api);
}

/**
 * Restore the stored expert layout: reconcile it against the currently-registered
 * panels (so panels added/removed while simple was active are folded in) and
 * apply it; fall back to the default arrangement when there is no usable stored
 * layout or the apply fails.
 * @param {any} api
 */
function restoreExpertLayout(api) {
  let raw = null;
  if (layoutStorageKey) {
    try {
      raw = localStorage.getItem(layoutStorageKey);
    } catch { /* storage blocked — fall through to default */ }
  }
  const next = raw ? reconcile(raw, currentRegisteredPanels(api)) : null;
  if (next) {
    try {
      api.fromJSON(next);
      ensureTerminalPanel(api);
      normalizeTerminalTitle(api);
      redockServices?.();
      return;
    } catch (err) {
      console.error('dock: expert layout restore failed; using default', err);
    }
  }
  arrangeDefaultLayout(api);
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
 * Stamp the terminal tab's current title onto a restored layout. Layouts
 * persisted before the terminal tab was titled carry the old empty-string
 * title, and fromJSON restores titles verbatim — without this, an upgraded
 * client would keep resurrecting an unlabeled terminal tab from storage.
 * @param {any} api
 */
function normalizeTerminalTitle(api) {
  const panel = api.getPanel(PANEL_TERMINAL);
  if (!panel || panel.title === TERMINAL_TITLE) return;
  if (typeof panel.setTitle === 'function') panel.setTitle(TERMINAL_TITLE);
  else panel.api?.setTitle?.(TERMINAL_TITLE);
}

/**
 * Add a panel from a serialized panel descriptor (as read off toJSON().panels),
 * optionally at a given position and initial width. contentComponent maps to
 * dockview's `component` create key; title/params/tabComponent pass through
 * when present.
 * @param {any} api
 * @param {PanelDescriptor} descriptor
 * @param {any} [position]
 * @param {number} [initialWidth]
 */
function addPanelFromDescriptor(api, descriptor, position, initialWidth) {
  /** @type {any} */
  const opts = {
    id: descriptor.id,
    component: descriptor.contentComponent ?? descriptor.id,
    title: descriptor.title ?? '',
  };
  if (descriptor.tabComponent) opts.tabComponent = descriptor.tabComponent;
  if (descriptor.params != null) opts.params = descriptor.params;
  if (position) opts.position = position;
  if (initialWidth != null && initialWidth > 0) opts.initialWidth = initialWidth;
  api.addPanel(opts);
}

/**
 * Lock the layout against user editing: disable all drag/float (runtime
 * disableDnd, which dockview merges and re-wires) and all sash resizing. Tab
 * clicks still activate panels; close controls are hidden via CSS under
 * html[data-ui-mode="simple"] (dockview-overrides.css).
 * @param {any} api
 */
function lockLayout(api) {
  api.updateOptions({ disableDnd: true });
  api.locked = true;
}

/**
 * Release the simple-mode lock. disableDnd is an accessor option checked per
 * gesture; `locked` is cleared here and again is idempotent if a later restore
 * rebuilds the grid.
 * @param {any} api
 */
function unlockLayout(api) {
  api.updateOptions({ disableDnd: false });
  api.locked = false;
}

/**
 * The live DockviewApi, or null before initDockWorkspace() has run. Exposed for
 * the follow-on dock tasks (state-sync / persistence / mode-transition).
 * @returns {any}
 */
export function getDockApi() {
  return dockApi;
}
