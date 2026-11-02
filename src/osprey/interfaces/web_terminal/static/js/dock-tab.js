// @ts-check
/* OSPREY Web Terminal — Tile Header Bar (custom dockview tab renderer)
 *
 * Under the one-panel-per-tile invariant a tile's "tab" can never have
 * siblings, so instead of a tab it renders the tile's HOST HEADER BAR. Every
 * tile gets the same bar grammar — the tile's identity on the left, its
 * actions right-anchored — so naming and closing a tile is one gesture
 * learned once. There is no drag badge: the bar itself is the drag handle
 * (cursor: grab), a convention the workspace teaches by use:
 *
 *   service tiles — a visible `.tile-tab-title`, pop out and close. Pop out
 *     moves the panel into its own browser window (popout.js), which retires
 *     the tile until that window closes. The close is
 *     a LOCAL tile close (the panel keeps its sidebar row; dock-sync routes
 *     the click to a vacate handler).
 *   the Agent tile — an adopted live bar + close. In chat it adopts the
 *     console's `.op-session-bar` (status dot, `Agent`, New conversation,
 *     notifications, Export, ⋯); on the terminal surface it adopts the
 *     `.terminal-header` (session label, session selector, "+ New", ⋯). The bar
 *     is chosen when the tab is built and swapped on a change of surface.
 *     The close runs dock-workspace's closeTerminalPanel, which keeps
 *     `html[data-agent]` and the `osprey-agent-pane` event in step.
 *
 * Registered via dockview's createTabComponent/defaultTabComponent
 * (dock-workspace.js), which keeps dockview's native tab DnD, drop
 * hit-testing and close plumbing intact — the bar IS the drag handle.
 *
 * Both bar kinds are also a right-click surface: the tile header opens the
 * same verb menu as the panel's sidebar row (panel-context-menu.js), routed
 * through the setTileContextMenuHandler registration seam below.
 */

import { TERMINAL_RAIL_ID, titleCase } from './panel-catalog.js';
import { registerContribHost, unregisterContribHost } from './tile-header-contrib.js';
import { PLACEHOLDER_PREFIX } from './dock-reconcile.js';
import { withEchoSuppressed } from './dock-sync.js';
import { closeTerminalPanel } from './dock-workspace.js';
import { popOutPanel, canPopOut } from './popout.js';
import { isTerminalSurface, onSurfaceChange } from './agent-surface.js';
import { icon } from '/design-system/js/icons.js';

/** What the disabled Pop out button says about an opaque-origin panel. */
const CANNOT_POP_OUT = 'This panel runs at an opaque origin and cannot be popped out.';

/** defaultTabComponent name registered on the dockview instance. */
export const OSPREY_TAB_COMPONENT = 'osprey-tile-tab';

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
  wireTileContextMenu(el, TERMINAL_RAIL_ID);
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

/** Matches the interactive children of an adopted Agent bar. */
const INTERACTIVE = 'button, select, input, a, [role="button"]';

/**
 * Handler a tile-header right-click is routed to, registered by
 * panel-manager.js at init — it owns the whole menu policy (which verbs, and
 * whether the surface has a menu at all), exactly as it does for the rail.
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
 * scrolling its own content underneath leaves the menu standing.
 * @typedef {{ x: number, y: number, anchorEl: HTMLElement }} TileMenuRequest
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
 * @param {string} id  the panel id the menu's verbs act on
 */
function wireTileContextMenu(el, id) {
  el.addEventListener('contextmenu', (e) => {
    if (e.target instanceof Element && e.target.closest(INTERACTIVE)) return;
    const opened = tileContextMenuHandler?.(id, {
      x: e.clientX,
      y: e.clientY,
      anchorEl: el,
    });
    if (opened) e.preventDefault();
  });
}

/**
 * A service tile's Pop out button: moves the panel into its own window. It
 * runs the pop-out inside the click, the gesture the browser needs to open a
 * window at all. A panel with no first-party page to open (an opaque-origin
 * panel) gets the button disabled, with the reason as its title. The gate
 * cannot be read when the tile is built: the panel's URL and its frame kind
 * resolve later, and a button disabled then would never receive the click
 * that re-reads it. So the button starts enabled and reads the gate when the
 * pointer or focus arrives and again at click time.
 * @param {string} panelId  service panel id (the placeholder id minus its prefix)
 * @returns {HTMLButtonElement}
 */
function buildPopoutButton(panelId) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'tile-tab-popout osprey-icon-btn';
  btn.appendChild(icon('popout'));
  btn.setAttribute('aria-label', 'Pop out');
  btn.title = 'Pop out';
  const paint = () => {
    const allowed = canPopOut(panelId);
    btn.disabled = !allowed;
    btn.title = allowed ? 'Pop out' : CANNOT_POP_OUT;
  };
  btn.addEventListener('pointerenter', paint);
  btn.addEventListener('focus', paint);
  btn.addEventListener('click', (e) => {
    if (e.defaultPrevented) return;
    e.preventDefault();
    paint();
    if (!btn.disabled) popOutPanel(panelId);
  });
  return btn;
}

/**
 * The tile-tab renderer (dockview ITabRenderer: element / init / dispose).
 */
class TileTab {
  /** @param {string} id  dockview panel id ('terminal' or 'iframe:<panelId>') */
  constructor(id) {
    this._panelId = id;
    /** @type {{ dispose: () => void }[]} */
    this._disposables = [];
    /** @type {any} */
    this._api = null;

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
      // panel's last `osprey-header-contribution`). Keyed by the service
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
      registerContribHost(this._servicePanelId, contrib);
      // The bar is the panel's second home for its verbs, alongside the
      // sidebar row. Wired per construction because a service tab's root IS
      // rebuilt with the tab — unlike the long-lived adopted Agent bar.
      wireTileContextMenu(root, this._servicePanelId);
    }

    const actions = document.createElement('div');
    actions.className = 'tile-tab-actions';
    // The DefaultTab idiom: preventDefault on pointerdown keeps an action
    // press from starting a drag/focus dance; the click still fires.
    actions.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    if (!terminal) actions.appendChild(buildPopoutButton(this._servicePanelId ?? id));
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'tile-tab-close osprey-icon-btn';
    close.appendChild(icon('x'));
    close.title = terminal ? 'Close Agent pane (reopen from the sidebar)' : 'Close tile';
    close.setAttribute('aria-label', terminal ? 'Close Agent pane' : 'Close tile');
    close.addEventListener('click', (e) => {
      if (e.defaultPrevented) return;
      e.preventDefault();
      // The removal makes dockview auto-activate a surviving tile; that is a
      // side effect of the close, not a human focus gesture, so it must not
      // reach the focus reporter — the same suppression retireTile applies to
      // its own removal. The Agent tile closes through dock-workspace so the
      // closed state reaches `html[data-agent]` and the sidebar.
      withEchoSuppressed(() => (terminal ? closeTerminalPanel() : this._api?.close?.()));
    });
    actions.appendChild(close);
    root.appendChild(actions);

    this._element = root;
  }

  get element() {
    return this._element;
  }

  /** @param {any} params  dockview TabPartInitParameters (title, params, api, …) */
  init(params) {
    this._api = params.api;
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

  dispose() {
    for (const d of this._disposables.splice(0)) d.dispose();
    if (agentTabRoot === this._element) agentTabRoot = null;
    if (this._contribEl && this._servicePanelId) {
      unregisterContribHost(this._servicePanelId, this._contribEl);
    }
  }
}

/**
 * Factory for dockview's createTabComponent option.
 * @param {string} id
 * @returns {TileTab}
 */
export function createTileTab(id) {
  return new TileTab(id);
}
