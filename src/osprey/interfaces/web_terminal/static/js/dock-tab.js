// @ts-check
/* OSPREY Web Terminal — Tile Header Bar (custom dockview tab renderer)
 *
 * Every tile carries one header bar: the tile's identity on the left, its
 * actions right-anchored, so naming and closing is one gesture learned once.
 * The bar itself is the drag handle (cursor: grab). A tile holds ONE panel
 * (dock-policy.js), so the bar is that panel's and there is no tab strip:
 * the sidebar is where the operator switches panels.
 *
 *   service tiles — the identity is the panel's name (`.tile-tab-title`);
 *     then the panel's icons (at most three, tile-header-contrib.js) and the
 *     fixed ⋯ / Open in new window / × cluster. The ⋯ opens the tile's one menu
 *     (panel-menu-policy.js), the same menu a right-click on the bar opens.
 *     The × closes the tile LOCALLY (the panel keeps its sidebar row;
 *     dock-sync routes the click to a vacate handler).
 *   the Agent tile — an adopted live bar (agent-bar.js: the session label ▾
 *     and +), then ⋯ and close; Open in new window is a ⋯ row. In chat it
 *     adopts the console's `.op-session-bar`; on the terminal surface the
 *     `.terminal-header`, the same shape. The bar is chosen when the
 *     tab is built and swapped on a change of surface. Its close runs
 *     dock-workspace's closeTerminalPanel, which keeps `html[data-agent]` and
 *     the `osprey-agent-pane` event in step.
 *
 * Open in new window, on every tile, moves the tile's content into a window the hub
 * drives and closes the tile (pane-window.js).
 *
 * A pane in its own window keeps this same bar, built in `mode: 'detached'` and
 * moved into the window (pane-window.js): the name, the contributed controls or
 * the adopted Agent bar as above, the ⋯, and Reattach in the far-right slot; its
 * menu carries none of the tile's layout verbs. dockview plays no part there: the caller runs
 * `init` with the title, moves `element` into the window, then calls `attached`,
 * which registers what belongs to the window's document.
 *
 * dockview builds one tab component per panel, so a tile has exactly one bar,
 * stretched over the tile by dockview's single-tab fullwidth mode
 * (dock-workspace.js). Registered via dockview's createTabComponent /
 * defaultTabComponent.
 *
 * Drags: every drag the dock starts itself records its engine Source here at
 * `dragstart` (dockDragStart) for the drop bridge (dock-drop.js) to read, since
 * `dataTransfer.getData` is empty during dragover. The bar — dockview's own
 * tab drag — is `{tile}` (dock-drop maps it); a sidebar row is `{new}` or
 * `{tab}` (rail-drag.js).
 *
 * Both bar kinds are also a right-click surface: a right-click on the bar opens
 * the same menu as its ⋯ button (panel-context-menu.js), routed through the
 * setTileContextMenuHandler registration seam below.
 */

import { TERMINAL_RAIL_ID, titleCase } from './panel-catalog.js';
import { registerContribHost, unregisterContribHost } from './tile-header-contrib.js';
import {
  PLACEHOLDER_PREFIX,
  TAB_KEY_ATTRIBUTE,
  TERMINAL_KEY,
  isDetached,
  keyOfPanelId,
} from './dock-apply.js';
import { withEchoSuppressed } from './dock-sync.js';
import { closeDockPanel } from './dock-workspace.js';
import { closeTerminalTile } from './panel-menu-policy.js';
import { detach, detachBlocker } from './pane-window.js';
import { isTerminalSurface, onSurfaceChange } from './agent-surface.js';
import { closeContextMenu } from '/design-system/js/context-menu.js';
import { icon } from '/design-system/js/icons.js';
import { closeAllPopovers } from '/design-system/js/popover.js';
import { initTooltips } from '/design-system/js/tooltip.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Source} Source */
/** @typedef {import('/static/vendor/dock-engine.mjs').TabRects} TabRects */

/** defaultTabComponent name registered on the dockview instance. */
export const OSPREY_TAB_COMPONENT = 'osprey-tile-tab';

/** The transfer type a drag of the dock's own carries (JSON of its Source). */
export const DOCK_SOURCE_MIME = 'application/x-osprey-dock-source';

/**
 * The page's two live Agent bars, cached across tab rebuilds: after a close
 * the adopted node is detached, and the reopen path must re-adopt the SAME
 * live subtree so every reference held by terminal.js, sessions.js and the
 * chat stays valid — the header-level counterpart of dock-workspace's
 * adoptSubtree mechanism.
 *
 * Registered explicitly via setTerminalHeaderSource /
 * setTerminalConsoleBarSource — there is deliberately NO lazy
 * `document.querySelector` fallback here. dockview always constructs a
 * panel's content component before its tab component, so by the time the
 * terminal's TileTab is constructed, dock-workspace's own content adoption
 * (adoptSubtree) has already moved the whole `.terminal-panel` subtree into a
 * detached host div, ahead of dockview inserting that host into the live
 * document. A document-wide query at TileTab-construction time is therefore
 * GUARANTEED to miss both bars. Callers MUST register the references up
 * front, before adoption can detach them (see dock-workspace.js's
 * initDockWorkspace, which does this before its first addPanel for the
 * terminal).
 * @type {HTMLElement | null}
 */
let terminalHeaderEl = null;

/** The chat console's `.op-session-bar`. @type {HTMLElement | null} */
let terminalConsoleBarEl = null;

/**
 * Registers the live `.terminal-header` node, the terminal surface's Agent bar.
 * Called once by dock-workspace.js before the terminal panel is added to
 * dockview (see initDockWorkspace).
 * @param {HTMLElement} el
 */
export function setTerminalHeaderSource(el) {
  terminalHeaderEl = el;
  barHomes.set(el, el.parentElement);
  wireTerminalBar(el);
}

/**
 * Registers the chat console's `.op-session-bar`, the chat's Agent bar. Called once by dock-workspace.js beside setTerminalHeaderSource; the
 * console is built before the dock (app.js runs initChat first).
 * @param {HTMLElement} el
 */
export function setTerminalConsoleBarSource(el) {
  terminalConsoleBarEl = el;
  barHomes.set(el, el.parentElement);
  wireTerminalBar(el);
}

/**
 * Wire the interactive-children pointerdown guard and the bar's contextmenu
 * route directly on a registered bar node, here rather than in the TileTab
 * constructor: the node is long-lived across close/reopen/surface change, but a
 * TileTab is rebuilt on every one of those, so attaching in the constructor
 * would stack a new listener on the same node each time. Registration happens
 * exactly once per node.
 * @param {HTMLElement} el
 */
function wireTerminalBar(el) {
  // Interactive children (selector, + New, the console's icon buttons) must
  // act, not drag: stop pointerdown from reaching dockview's tab drag
  // tracking. Plain bar surface still bubbles, so the bar remains the drag
  // handle.
  el.addEventListener('pointerdown', (e) => {
    if (e.target instanceof Element && e.target.closest(INTERACTIVE)) {
      e.stopPropagation();
    }
  });
  // The Agent bar's right-click surface is the adopted bar itself, not the
  // tab root: the bar fills the tile bar, and listening on the root as well
  // would run the handler twice for one press.
  wireTileContextMenu(el, () => TERMINAL_RAIL_ID);
}

/**
 * The bar the Agent tile adopts for the surface showing: the console bar in chat
 * (the header when the console never built), the header on the terminal.
 * @returns {HTMLElement | null}
 */
function terminalBar() {
  return (isTerminalSurface() ? null : terminalConsoleBarEl) ?? terminalHeaderEl;
}

/**
 * Where each Agent bar lives while the tile has not adopted it, so a bar the
 * tile lets go of keeps its ids and listeners on the page.
 * @type {Map<HTMLElement, HTMLElement | null>}
 */
const barHomes = new Map();

/** The Agent tile's tab root while the tile exists. @type {HTMLElement | null} */
let agentTabRoot = null;

/** Put the current surface's bar on the Agent tile, sending the other one home. */
function syncAgentBar() {
  const bar = terminalBar();
  if (!agentTabRoot || !bar || bar.parentElement === agentTabRoot) return;
  for (const other of [terminalHeaderEl, terminalConsoleBarEl]) {
    if (other && other !== bar && other.parentElement === agentTabRoot) {
      const home = barHomes.get(other);
      if (home) home.prepend(other);
      else other.remove();
    }
  }
  agentTabRoot.prepend(bar);
}

onSurfaceChange(syncAgentBar);

/**
 * A bar in a window is going away: an Agent bar it still holds goes back where it
 * lives, out of a document that may be closing. A tile that has already adopted
 * the bar keeps it, and one built later adopts it from there.
 * @param {HTMLElement} root  the tab root letting go
 */
function sendAgentBarsHome(root) {
  for (const bar of [terminalHeaderEl, terminalConsoleBarEl]) {
    if (bar?.parentElement === root) barHomes.get(bar)?.prepend(bar);
  }
}

/** Matches the interactive children of a bar: they act, they never drag. */
const INTERACTIVE = 'button, select, input, a, [role="button"]';

/**
 * Handler a tile-bar menu request (the ⋯ button, a right-click) is routed to,
 * registered by panel-manager.js at init — it owns the whole menu policy
 * (which verbs, and whether the surface has a menu at all), exactly as it does
 * for the rail.
 *
 * A registration seam rather than an import, mirroring dock-sync's
 * setTileCloseHandler: panel-manager imports dock-workspace, which imports
 * this module, so importing panel-manager back would close a cycle.
 * @type {((id: string, opts: TileMenuRequest) => boolean) | null}
 */
let tileContextMenuHandler = null;

/**
 * Where the menu should appear and what it belongs to. `anchorEl` is the
 * header bar: the menu scopes its scroll dismissal to it, so a panel
 * scrolling its own content underneath leaves the menu standing. `mode` says
 * whether the bar is on a tile or in the pane's own window, read when the menu is
 * asked for: the Agent bar is wired once and travels between the two. With
 * `triggerEl` (the bar's ⋯ button) the menu hangs under that button instead of
 * opening at x/y, and a second press closes it.
 * @typedef {{ x: number, y: number, anchorEl: HTMLElement, mode: TileMode,
 *   triggerEl?: HTMLElement }} TileMenuRequest
 */

/**
 * On a tile in the dock, or in the pane's own window.
 * @typedef {'tile' | 'detached'} TileMode
 */

/**
 * Registers the tile-header context-menu handler. Called once by
 * panel-manager.js at init. The handler returns true when it opened a menu.
 * @param {((id: string, opts: TileMenuRequest) => boolean) | null} fn
 */
export function setTileContextMenuHandler(fn) {
  tileContextMenuHandler = fn;
}

/**
 * Route right-clicks on a header bar to the registered handler.
 *
 * Declining is doing NOTHING — no menu and no preventDefault — so the browser
 * shows its own menu instead. That happens on interactive header children
 * (right-clicking inside a contributed search input must still offer
 * copy/paste) and whenever the handler itself declines because it has no
 * rows to offer for that surface.
 * @param {HTMLElement} el  the header surface, and the menu's anchor
 * @param {() => string | null} idOf  the panel id the menu's verbs act on
 */
function wireTileContextMenu(el, idOf) {
  el.addEventListener('contextmenu', (e) => {
    if (e.target instanceof Element && e.target.closest(INTERACTIVE)) return;
    const id = idOf();
    if (id === null) return;
    const opened = tileContextMenuHandler?.(id, {
      x: e.clientX,
      y: e.clientY,
      anchorEl: el,
      mode: isDetached(id) ? 'detached' : 'tile',
    });
    if (opened) e.preventDefault();
  });
}

/**
 * The bar's ⋯: the tile's one menu button. A click asks the registered handler
 * for the same menu a right-click on the bar opens, hung under the button.
 * Shared with the fallback bar (dock-fallback-bar.js).
 * @param {() => string | null} idOf  the panel id the menu's verbs act on
 * @param {HTMLElement} anchorEl  the header surface, for the menu's scroll scoping
 * @returns {HTMLButtonElement}
 */
export function buildTileMenuButton(idOf, anchorEl) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'tile-tab-more osprey-icon-btn';
  btn.appendChild(icon('more'));
  btn.title = 'Panel menu';
  btn.setAttribute('aria-label', 'Panel menu');
  btn.setAttribute('aria-haspopup', 'menu');
  btn.setAttribute('aria-expanded', 'false');
  btn.addEventListener('click', (e) => {
    if (e.defaultPrevented) return;
    e.preventDefault();
    const id = idOf();
    if (id === null) return;
    const r = btn.getBoundingClientRect();
    tileContextMenuHandler?.(id, {
      x: r.left,
      y: r.bottom,
      anchorEl,
      triggerEl: btn,
      mode: isDetached(id) ? 'detached' : 'tile',
    });
  });
  return btn;
}

// --- drag sources -----------------------------------------------------------------

/** @type {Source | null} */
let currentSource = null;
/** @type {(() => void) | null} */
let disarmEnd = null;
/** @type {Set<(source: Source) => void>} */
const startListeners = new Set();

/** @param {Source} source */
function sourceText(source) {
  if ('tab' in source) return source.tab;
  if ('tile' in source) return source.tile;
  return source.new;
}

/**
 * Starts a drag of the dock's own: records `source` for the drop bridge,
 * writes the transfer and notifies onDockDragStart listeners (the bridge
 * raises the drag shield there). The gesture's `dragend` clears it; a `drop`
 * clears it one turn later, after the bridge's own drop handler has read it.
 * @param {Source} source
 * @param {DataTransfer | null} dataTransfer
 */
export function dockDragStart(source, dataTransfer) {
  disarmEnd?.();
  currentSource = source;
  if (dataTransfer) {
    dataTransfer.setData(DOCK_SOURCE_MIME, JSON.stringify(source));
    dataTransfer.setData('text/plain', sourceText(source));
    dataTransfer.effectAllowed = 'move';
  }
  const onEnd = () => clearDockDrag();
  const onDrop = () => setTimeout(() => currentSource === source && clearDockDrag(), 0);
  document.addEventListener('dragend', onEnd, true);
  document.addEventListener('drop', onDrop, true);
  disarmEnd = () => {
    document.removeEventListener('dragend', onEnd, true);
    document.removeEventListener('drop', onDrop, true);
    disarmEnd = null;
  };
  for (const listener of [...startListeners]) listener(source);
}

/** @returns {Source | null} the source of the dock's own drag in progress */
export function dockDragSource() {
  return currentSource;
}

/** Forgets the drag in progress. */
export function clearDockDrag() {
  currentSource = null;
  disarmEnd?.();
}

/**
 * Subscribes to every dockDragStart; returns the unsubscribe.
 * @param {(source: Source) => void} listener
 * @returns {() => void}
 */
export function onDockDragStart(listener) {
  startListeners.add(listener);
  return () => {
    startListeners.delete(listener);
  };
}

// --- hit-test geometry ------------------------------------------------------------

/** @param {Element} el */
function rectOf(el) {
  const r = el.getBoundingClientRect();
  return { x: r.left, y: r.top, width: r.width, height: r.height };
}

/**
 * The bar rect of every tile under `scope`, keyed by tile id, in client
 * coordinates — the TabRects input of the engine's hitTest. The bar is the
 * tile's whole header row (its identity, the free space and its actions), and
 * it carries no tab rects: a tile holds one panel, so the engine's bar zone
 * is what the drop bridge refuses, and the tile's content below the bar is
 * what its edge bands are measured against.
 * @param {ParentNode} [scope]
 * @returns {TabRects}
 */
export function measureTabRects(scope = document) {
  /** @type {TabRects} */
  const rects = {};
  for (const bar of scope.querySelectorAll('.tile-tab[data-tile]')) {
    if (!(bar instanceof HTMLElement) || !bar.isConnected) continue;
    const tile = bar.dataset.tile;
    if (tile === undefined || tile in rects) continue;
    rects[tile] = {
      barRect: rectOf(bar.closest('.dv-tabs-and-actions-container') ?? bar),
      tabRects: [],
    };
  }
  return rects;
}

// --- the renderer -----------------------------------------------------------------

/** The tile button's name, and its tooltip while nothing blocks it. */
const WINDOW_BUTTON_LABEL = 'Open in new window';

/**
 * A tile's Open in new window button: moves the tile's content into a window of its own
 * (pane-window.js). It runs the pop-out inside the click, the gesture the
 * browser needs to open a window at all. Content that cannot go (a panel not
 * loaded yet, an opaque-origin panel) gets the button disabled, with the
 * reason as its title. The gate cannot be read when the tile is built: the
 * panel's iframe and its frame kind arrive later, and a button disabled then
 * would never receive the click that re-reads it. So the button starts
 * enabled and reads the gate when the pointer or focus arrives and again at
 * click time.
 * @param {string} panelId  service panel id (the placeholder id minus its
 *   prefix), or TERMINAL_RAIL_ID
 * @returns {HTMLButtonElement}
 */
function buildWindowButton(panelId) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'tile-tab-window osprey-icon-btn';
  btn.appendChild(icon('popout'));
  btn.setAttribute('aria-label', WINDOW_BUTTON_LABEL);
  btn.title = WINDOW_BUTTON_LABEL;
  const paint = () => {
    const blocker = detachBlocker(panelId);
    btn.disabled = blocker !== null;
    btn.title = blocker ?? WINDOW_BUTTON_LABEL;
  };
  btn.addEventListener('pointerenter', paint);
  btn.addEventListener('focus', paint);
  btn.addEventListener('click', (e) => {
    if (e.defaultPrevented) return;
    e.preventDefault();
    paint();
    if (!btn.disabled) detach(panelId);
  });
  return btn;
}

/**
 * The far-right slot of a bar in the pane's own window.
 * @param {(() => void) | undefined} onReattach
 * @returns {HTMLButtonElement}
 */
function buildReattachButton(onReattach) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'tile-tab-reattach osprey-btn';
  btn.textContent = 'Reattach';
  btn.addEventListener('click', (e) => {
    if (e.defaultPrevented) return;
    e.preventDefault();
    onReattach?.();
  });
  return btn;
}

/**
 * @typedef {object} TileTabOptions
 * @property {TileMode} [mode] - default 'tile'
 * @property {() => void} [onReattach] - the Reattach of a 'detached' bar
 */

/**
 * The tile-tab renderer (dockview ITabRenderer: element / init / dispose).
 */
class TileTab {
  /**
   * @param {string} id  dockview panel id ('terminal' or 'iframe:<panelId>'), or
   *   the bare panel id of a pane in its own window
   * @param {TileTabOptions} [options]
   */
  constructor(id, { mode = 'tile', onReattach } = {}) {
    this._panelId = id;
    this._mode = mode;
    /** @type {{ dispose: () => void }[]} */
    this._disposables = [];
    /** @type {HTMLElement | null} Reattach (a bar in the pane's own window) */
    this._reattachEl = null;
    /** @type {(() => void) | null} stops the window's tooltips */
    this._stopTooltips = null;

    const root = document.createElement('div');
    root.className = 'tile-tab';
    // dockview does not stamp its panel ids anywhere on the tab DOM. Doing it
    // here gives the tile a stable handle for the browser suite and for any
    // future per-tile styling, independent of a title that may be renamed.
    root.dataset.panelId = id;

    /** @type {HTMLElement | null} visible name element (service tiles only) */
    this._titleEl = null;
    /** @type {HTMLElement | null} contribution region (service tiles only) */
    this._contribEl = null;
    /** @type {string | null} service panel id the region is keyed by */
    this._servicePanelId = null;

    const terminal = id === TERMINAL_RAIL_ID;
    const key = terminal ? TERMINAL_KEY : keyOfPanelId(id);
    root.setAttribute(TAB_KEY_ATTRIBUTE, key ?? id);
    if (terminal) {
      root.classList.add('tile-tab-terminal');
      const bar = terminalBar();
      // DOM relocation — live references survive. The pointerdown guard and
      // the context menu are wired once, on the node itself, at registration.
      if (bar) root.appendChild(bar);
      agentTabRoot = root;
    } else {
      const title = document.createElement('span');
      title.className = 'tile-tab-title';
      root.appendChild(title);
      this._titleEl = title;

      // Contribution region: the panel-controls slot between the tile's name
      // and its close button (tile-header-contrib.js renders into it from the
      // panel's last `canopy-host-header-contribution`). Keyed by the service
      // panel id, i.e. the dockview placeholder id minus its prefix.
      const contrib = document.createElement('div');
      contrib.className = 'tile-tab-contrib';
      // Interactive children act instead of dragging — same guard as the
      // adopted Agent bar; empty region surface still bubbles so the
      // bar remains the drag handle.
      contrib.addEventListener('pointerdown', (e) => {
        if (e.target instanceof Element && e.target.closest(INTERACTIVE)) {
          e.stopPropagation();
        }
      });
      root.appendChild(contrib);
      this._contribEl = contrib;
      this._servicePanelId = id.startsWith(PLACEHOLDER_PREFIX)
        ? id.slice(PLACEHOLDER_PREFIX.length)
        : id;
      // A bar on its way to a window registers from there (attached).
      if (mode === 'tile') registerContribHost(this._servicePanelId, contrib);
      // The bar is the panel's second home for its verbs, alongside the
      // sidebar row. Wired per construction because a service tab's root IS
      // rebuilt with the tab — unlike the long-lived adopted Agent bar.
      wireTileContextMenu(root, () => this._servicePanelId);
    }

    const actions = document.createElement('div');
    actions.className = 'tile-tab-actions';
    // The DefaultTab idiom: preventDefault on pointerdown keeps an action
    // press from starting a drag/focus dance; the click still fires.
    actions.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    root.appendChild(actions);
    this._element = root;
    // Fixed at the right edge, never folded: ⋯ on every bar, then Open in new
    // window (a service tile's; the Agent's is a ⋯ row) and × on a tile,
    // Reattach in the pane's own window.
    actions.appendChild(
      buildTileMenuButton(() => (terminal ? TERMINAL_RAIL_ID : this._servicePanelId), root)
    );

    if (mode === 'detached') {
      this._reattachEl = buildReattachButton(onReattach);
      actions.appendChild(this._reattachEl);
      return;
    }

    if (!terminal) actions.appendChild(buildWindowButton(this._servicePanelId ?? id));
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'tile-tab-close osprey-icon-btn';
    close.appendChild(icon('x'));
    close.title = terminal ? 'Close Agent pane (reopen from the sidebar)' : 'Close tile';
    close.setAttribute('aria-label', terminal ? 'Close Agent pane' : 'Close tile');
    close.addEventListener('click', (e) => {
      if (e.defaultPrevented) return;
      e.preventDefault();
      // Both closes go through dock-workspace, i.e. the layout holder: the
      // rebuild makes dockview activate a surviving tile, and that is a side
      // effect of the close, not a human focus gesture, so it must not reach
      // the focus reporter — the same suppression retireTile applies to its
      // own removal. The Agent tile's close also puts the closed state on
      // `html[data-agent]` for the sidebar, and raises the Undo notice.
      withEchoSuppressed(() =>
        terminal ? closeTerminalTile() : closeDockPanel(this._servicePanelId ?? this._panelId)
      );
    });
    actions.appendChild(close);
  }

  get element() {
    return this._element;
  }

  /** @param {any} params  dockview TabPartInitParameters (title, params, api, …) */
  init(params) {
    if (this._reattachEl) {
      this._reattachEl.title = `Put ${titleCase(params.title ?? '')} back in the workspace`;
    }
    // The tile id the bar belongs to, for the drop bridge's hit test and the
    // holder's focus restore; it follows the panel across group changes.
    const stamp = () => {
      const group = params.api?.group ?? null;
      if (group) this._element.dataset.tile = String(group.id);
      else delete this._element.dataset.tile;
    };
    stamp();
    const groupChange = params.api?.onDidGroupChange?.(stamp);
    if (groupChange?.dispose) this._disposables.push(groupChange);
    // The panel title is a service bar's visible label AND its accessible
    // name, kept current so a renamed panel never shows or announces a stale
    // one. The Agent bar is exempt: the bar it adopted names it already.
    if (this._panelId === TERMINAL_RAIL_ID) return;
    /** @param {string} title */
    // The registry's labels are the panel ids' shouting form (ARIEL, EVENTS);
    // the chrome shows them in title case, as the sidebar rows do.
    const applyName = (title) => {
      const name = titleCase(title);
      if (this._titleEl) this._titleEl.textContent = name;
      this._element.setAttribute('aria-label', name);
    };
    applyName(params.title ?? '');
    const sub = params.api?.onDidTitleChange?.((/** @type {any} */ e) => applyName(e?.title ?? ''));
    if (sub?.dispose) this._disposables.push(sub);
  }

  /**
   * A bar built for a window is in that window's document now: what is bound to a
   * document binds to this one. Called after every move, so after a reload of the
   * window as well, with a new tab and a new document.
   */
  attached() {
    if (this._contribEl && this._servicePanelId) {
      registerContribHost(this._servicePanelId, this._contribEl);
    }
    this._stopTooltips?.();
    this._stopTooltips = initTooltips(this._element.ownerDocument.body);
  }

  dispose() {
    for (const d of this._disposables.splice(0)) d.dispose();
    // A window's document may be on its way out (R10): nothing stays open in it,
    // or the opener's next click would only close what can no longer be seen.
    if (this._mode === 'detached') {
      closeAllPopovers();
      closeContextMenu();
    }
    this._stopTooltips?.();
    this._stopTooltips = null;
    if (agentTabRoot === this._element) agentTabRoot = null;
    if (this._contribEl && this._servicePanelId) {
      unregisterContribHost(this._servicePanelId, this._contribEl);
    }
    if (this._mode === 'detached') sendAgentBarsHome(this._element);
  }
}

/**
 * Factory for dockview's createTabComponent option, and for the bar of a pane in
 * its own window (`mode: 'detached'`).
 * @param {string} id
 * @param {TileTabOptions} [options]
 * @returns {TileTab}
 */
export function createTileTab(id, options) {
  return new TileTab(id, options);
}
