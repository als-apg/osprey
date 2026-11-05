// @ts-check
/* OSPREY Web Terminal — Tile Header Bar (custom dockview tab renderer)
 *
 * Every tile carries one header bar: the tile's identity on the left, its
 * actions right-anchored, so naming, switching and closing is one gesture
 * learned once. The bar itself is the drag handle (cursor: grab).
 *
 *   service tiles — the identity is the showing tab's name (`.tile-tab-title`)
 *     while the tile holds one tab, and a TAB STRIP (`.tile-tab-tabs`, one
 *     `.tile-tab-item` per tab, the showing one marked) once it holds more.
 *     The contributed controls, Pop out and × beside it belong to the SHOWING
 *     tab and follow it: a switch swaps the whole bar. Pop out moves the panel
 *     into its own browser window (popout.js), which retires its tab until
 *     that window closes. The × closes the showing tab — the tile when that was
 *     its last tab — LOCALLY (the panel keeps its sidebar row; dock-sync routes
 *     the click to a vacate handler).
 *   the Agent tile — an adopted live bar + close. In chat it adopts the
 *     console's `.op-session-bar` (status dot, `Agent`, New conversation,
 *     notifications, Export, ⋯); on the terminal surface it adopts the
 *     `.terminal-header` (session label, session selector, "+ New", ⋯). The bar
 *     is chosen when the tab is built and swapped on a change of surface. The
 *     Agent pane is always its own tile (no strip, never a tab of another), and
 *     its close runs dock-workspace's closeTerminalPanel, which keeps
 *     `html[data-agent]` and the `osprey-agent-pane` event in step.
 *
 * dockview builds one tab component per panel. A tile with several tabs
 * therefore has several bars, and only the showing panel's is visible
 * (dockview-overrides.css hides `.dv-tab.dv-inactive-tab`); each bar renders
 * the whole tile's strip from the group it sits in, so whichever is showing
 * tells the same story. Registered via dockview's createTabComponent /
 * defaultTabComponent (dock-workspace.js).
 *
 * Drags: every drag the dock starts itself records its engine Source here at
 * `dragstart` (dockDragStart) for the drop bridge (dock-drop.js) to read, since
 * `dataTransfer.getData` is empty during dragover. A strip item starts `{tab}`;
 * the bar elsewhere — dockview's own tab drag — is `{tile}` (dock-drop maps it).
 *
 * Both bar kinds are also a right-click surface: the tile header opens the
 * same verb menu as the panel's sidebar row (panel-context-menu.js), routed
 * through the setTileContextMenuHandler registration seam below.
 */

import { TERMINAL_RAIL_ID, titleCase } from './panel-catalog.js';
import { registerContribHost, unregisterContribHost } from './tile-header-contrib.js';
import {
  PLACEHOLDER_PREFIX,
  TAB_KEY_ATTRIBUTE,
  TERMINAL_KEY,
  getLayoutHolder,
  keyOfPanelId,
} from './dock-apply.js';
import { withEchoSuppressed } from './dock-sync.js';
import { closeDockPanel, closeTerminalPanel } from './dock-workspace.js';
import { popOutPanel, canPopOut } from './popout.js';
import { isTerminalSurface, onSurfaceChange } from './agent-surface.js';
import { icon } from '/design-system/js/icons.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Source} Source */
/** @typedef {import('/static/vendor/dock-engine.mjs').TabRects} TabRects */

/** What the disabled Pop out button says about an opaque-origin panel. */
const CANNOT_POP_OUT = 'This panel runs at an opaque origin and cannot be popped out.';

/** defaultTabComponent name registered on the dockview instance. */
export const OSPREY_TAB_COMPONENT = 'osprey-tile-tab';

/** The strip's items (one per tab of the tile); each carries `data-panel-key`. */
export const TAB_ITEM_CLASS = 'tile-tab-item';
/** The transfer type a drag of the dock's own carries (JSON of its Source). */
export const DOCK_SOURCE_MIME = 'application/x-osprey-dock-source';

/** Keys that move the showing tab along the strip, wrapping at the ends. */
const MOVE_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'Home', 'End']);
/** Keys that close the showing tab from the strip. */
const CLOSE_KEYS = new Set(['Delete', 'Backspace']);

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

/** Matches the interactive children of a bar: they act, they never drag. */
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
 * @param {(e: MouseEvent) => string | null} idOf  the panel id the menu's verbs
 *   act on — a strip item's own tab, else the showing one
 */
function wireTileContextMenu(el, idOf) {
  el.addEventListener('contextmenu', (e) => {
    if (e.target instanceof Element && e.target.closest(INTERACTIVE)) return;
    const id = idOf(e);
    if (id === null) return;
    const opened = tileContextMenuHandler?.(id, {
      x: e.clientX,
      y: e.clientY,
      anchorEl: el,
    });
    if (opened) e.preventDefault();
  });
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
 * The bar and tab rects of every tile under `scope`, keyed by tile id, in
 * client coordinates — the TabRects input of the engine's hitTest. The bar is
 * the tile's whole header row (its identity, the free space and the showing
 * tab's actions), so a drop anywhere on it adds a tab: the header has no dead
 * pixels. The tabs are the strip's items, or the lone title when the tile
 * holds one tab; the Agent tile has none (nothing tabs into it).
 *
 * Only the showing panel's bar is on screen; a hidden sibling's is skipped.
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
    if (bar.parentElement?.classList.contains('dv-inactive-tab')) continue;
    const items = [...bar.querySelectorAll(`.${TAB_ITEM_CLASS}`)];
    const title = bar.querySelector('.tile-tab-title');
    const strip = bar.querySelector('.tile-tab-tabs');
    const tabs =
      strip instanceof HTMLElement && !strip.hidden
        ? items
        : title instanceof HTMLElement && !title.hidden
          ? [title]
          : [];
    rects[tile] = {
      barRect: rectOf(bar.closest('.dv-tabs-and-actions-container') ?? bar),
      tabRects: tabs.map(rectOf),
    };
  }
  return rects;
}

// --- the renderer -----------------------------------------------------------------

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
 * @param {string[]} keys
 * @param {string | null} showing
 * @param {string} key  the pressed key
 * @returns {string | null}
 */
function nextKey(keys, showing, key) {
  if (keys.length === 0) return null;
  if (key === 'Home') return keys[0];
  if (key === 'End') return keys[keys.length - 1];
  const at = showing === null ? -1 : keys.indexOf(showing);
  const step = key === 'ArrowLeft' ? -1 : 1;
  return keys[(at + step + keys.length) % keys.length];
}

/**
 * A dockview panel's layout key. Every panel this dock builds is the terminal
 * or a service placeholder; any other id is its own key.
 * @param {any} panel
 * @returns {string}
 */
function keyOf(panel) {
  return keyOfPanelId(panel?.id) ?? String(panel?.id ?? '');
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
    /** @type {{ dispose: () => void }[]} the group's own subscriptions */
    this._groupSubscriptions = [];
    /** @type {{ dispose: () => void }[]} the group's panels' title subscriptions */
    this._titleSubscriptions = [];
    /** @type {any} */
    this._api = null;
    /** @type {any} the dockview group this tab's panel sits in */
    this._group = null;
    /** @type {string} this panel's own name, as rendered */
    this._name = '';

    const root = document.createElement('div');
    root.className = 'tile-tab';
    // dockview does not stamp its panel ids anywhere on the tab DOM. Doing it
    // here gives the tile a stable handle for the browser suite and for any
    // future per-tile styling, independent of a title that may be renamed.
    root.dataset.panelId = id;

    /** @type {HTMLElement | null} visible name element (service tiles only) */
    this._titleEl = null;
    /** @type {HTMLElement | null} the tab strip (service tiles only) */
    this._stripEl = null;
    /** @type {HTMLElement | null} contribution region (service tiles only) */
    this._contribEl = null;
    /** @type {HTMLButtonElement | null} */
    this._closeEl = null;
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

      // The strip: the tile's tabs once it holds more than one. Hidden (and
      // the title shown) while the tile holds this one tab.
      const strip = document.createElement('div');
      strip.className = 'tile-tab-tabs';
      strip.setAttribute('role', 'tablist');
      strip.setAttribute('aria-orientation', 'horizontal');
      strip.setAttribute('aria-label', 'Tile tabs');
      strip.tabIndex = 0;
      strip.hidden = true;
      strip.addEventListener('click', (e) => this._onStripClick(e));
      strip.addEventListener('keydown', (e) => this._onStripKeyDown(e));
      strip.addEventListener('dragstart', (e) => this._onStripDragStart(e));
      root.appendChild(strip);
      this._stripEl = strip;

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
      registerContribHost(this._servicePanelId, contrib);
      // The bar is the panel's second home for its verbs, alongside the
      // sidebar row. Wired per construction because a service tab's root IS
      // rebuilt with the tab — unlike the long-lived adopted Agent bar. A
      // right-click on a strip item is about that tab.
      wireTileContextMenu(root, (e) => {
        const item = e.target instanceof Element ? e.target.closest(`.${TAB_ITEM_CLASS}`) : null;
        const itemKey = item instanceof HTMLElement ? item.getAttribute(TAB_KEY_ATTRIBUTE) : null;
        return itemKey ?? this._servicePanelId;
      });
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
      // Both closes go through dock-workspace, i.e. the layout holder: the
      // rebuild makes dockview activate a surviving tile, and that is a side
      // effect of the close, not a human focus gesture, so it must not reach
      // the focus reporter — the same suppression retireTile applies to its
      // own removal. The Agent tile's close also puts the closed state on
      // `html[data-agent]` for the sidebar.
      withEchoSuppressed(() =>
        terminal ? closeTerminalPanel() : closeDockPanel(this._servicePanelId ?? this._panelId)
      );
    });
    actions.appendChild(close);
    this._closeEl = close;
    root.appendChild(actions);

    this._element = root;
  }

  get element() {
    return this._element;
  }

  /** @param {any} params  dockview TabPartInitParameters (title, params, api, …) */
  init(params) {
    this._api = params.api;
    this._watchGroup(params.api?.group ?? null);
    const groupChange = params.api?.onDidGroupChange?.(() =>
      this._watchGroup(params.api?.group ?? null)
    );
    if (groupChange?.dispose) this._disposables.push(groupChange);
    // The panel title is a service bar's visible label AND its accessible
    // name, kept current so a renamed panel never shows or announces a stale
    // one. The Agent bar is exempt: the bar it adopted names it already.
    if (this._panelId === TERMINAL_RAIL_ID) return;
    /** @param {string} title */
    // The registry's labels are the panel ids' shouting form (ARIEL, EVENTS);
    // the chrome shows them in title case, as the sidebar rows do.
    const applyName = (title) => {
      this._name = titleCase(title);
      if (this._titleEl) this._titleEl.textContent = this._name;
      this._element.setAttribute('aria-label', this._name);
      this._renderStrip();
    };
    applyName(params.title ?? '');
    const sub = params.api?.onDidTitleChange?.((/** @type {any} */ e) => applyName(e?.title ?? ''));
    if (sub?.dispose) this._disposables.push(sub);
  }

  /**
   * Follow the group this panel sits in: its tile id, and every change to
   * its tabs or their names re-renders the strip.
   * @param {any} group
   */
  _watchGroup(group) {
    for (const s of this._groupSubscriptions.splice(0)) s.dispose?.();
    this._group = group;
    if (!group) {
      delete this._element.dataset.tile;
      this._watchTitles();
      return;
    }
    this._element.dataset.tile = String(group.id);
    const model = group.model;
    // A tab joining or leaving re-makes the title subscriptions (never the
    // group's own, which may be the emitter firing), so a sibling that joined
    // after this bar was built still renames its item.
    const membership = () => this._watchTitles();
    const refresh = () => this._renderStrip();
    /** @type {Array<[string, () => void]>} */
    const wiring = [
      ['onDidAddPanel', membership],
      ['onDidRemovePanel', membership],
      ['onDidActivePanelChange', refresh],
    ];
    for (const [name, fn] of wiring) {
      if (typeof model?.[name] === 'function') this._groupSubscriptions.push(model[name](fn));
    }
    this._watchTitles();
  }

  /** Follow every sibling's title, then render the strip. */
  _watchTitles() {
    for (const s of this._titleSubscriptions.splice(0)) s.dispose?.();
    const refresh = () => this._renderStrip();
    for (const panel of this._group?.panels ?? []) {
      const sub = panel?.api?.onDidTitleChange?.(refresh);
      if (sub?.dispose) this._titleSubscriptions.push(sub);
    }
    this._renderStrip();
  }

  /**
   * The tile's tabs in bar order, named as the sidebar names them.
   * @returns {{ key: string, title: string }[]}
   */
  _tabs() {
    const panels = this._group?.panels ?? [];
    return panels.map((/** @type {any} */ panel) => ({
      key: keyOf(panel),
      title: panel?.id === this._panelId ? this._name : titleCase(String(panel?.title ?? '')),
    }));
  }

  /** The identity area: the title alone, or the strip once the tile holds tabs. */
  _renderStrip() {
    const strip = this._stripEl;
    const title = this._titleEl;
    if (!strip || !title) return;
    const tabs = this._tabs();
    const showing = this._group?.activePanel ? keyOf(this._group.activePanel) : null;
    const several = tabs.length > 1;
    strip.hidden = !several;
    title.hidden = several;
    if (this._closeEl) {
      this._closeEl.title = several ? 'Close tab' : 'Close tile';
      this._closeEl.setAttribute('aria-label', several ? 'Close tab' : 'Close tile');
    }
    if (!several) {
      strip.replaceChildren();
      strip.removeAttribute('aria-activedescendant');
      return;
    }
    const tile = this._element.dataset.tile ?? '';
    strip.replaceChildren(
      ...tabs.map(({ key, title: name }) => {
        const item = document.createElement('div');
        const isShowing = key === showing;
        item.className = isShowing
          ? `${TAB_ITEM_CLASS} ${TAB_ITEM_CLASS}--showing`
          : TAB_ITEM_CLASS;
        item.id = `tile-tab-item-${tile}-${key}`;
        item.setAttribute('role', 'tab');
        item.setAttribute('aria-selected', String(isShowing));
        item.setAttribute(TAB_KEY_ATTRIBUTE, key);
        item.tabIndex = -1;
        item.draggable = true;
        item.title = name;
        const label = document.createElement('span');
        label.className = `${TAB_ITEM_CLASS}-label`;
        label.textContent = name;
        item.appendChild(label);
        return item;
      })
    );
    if (showing === null) strip.removeAttribute('aria-activedescendant');
    else strip.setAttribute('aria-activedescendant', `tile-tab-item-${tile}-${showing}`);
  }

  /**
   * Show `key` in this tile: dockview shows it, the holder records it (the
   * active-panel listener in dock-workspace records it too; both are the same
   * fact, so the second is a no-op).
   * @param {string} key
   */
  _show(key) {
    const panel = (this._group?.panels ?? []).find((/** @type {any} */ p) => keyOf(p) === key);
    if (!panel) return;
    const hadFocus = !!this._stripEl && this._stripEl.contains(document.activeElement);
    if (this._group.activePanel !== panel) panel.api?.setActive?.();
    const tile = this._element.dataset.tile;
    if (tile !== undefined) getLayoutHolder()?.show(tile, key);
    // The keyboard stays on the strip: the bar now showing is another panel's,
    // so a strip that had focus hands it to the one on screen.
    if (hadFocus) {
      const next = this._group.element?.querySelector(
        '.dv-tab:not(.dv-inactive-tab) .tile-tab-tabs'
      );
      if (next instanceof HTMLElement && next !== this._stripEl)
        next.focus({ preventScroll: true });
    }
  }

  /** @param {MouseEvent} e */
  _onStripClick(e) {
    if (!(e.target instanceof Element)) return;
    const item = e.target.closest(`.${TAB_ITEM_CLASS}`);
    const key = item instanceof HTMLElement ? item.getAttribute(TAB_KEY_ATTRIBUTE) : null;
    if (key !== null) this._show(key);
  }

  /** @param {KeyboardEvent} e */
  _onStripKeyDown(e) {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    const keys = this._tabs().map((t) => t.key);
    const showing = this._group?.activePanel ? keyOf(this._group.activePanel) : null;
    if (CLOSE_KEYS.has(e.key)) {
      e.preventDefault();
      if (showing !== null && showing !== TERMINAL_KEY) {
        withEchoSuppressed(() => closeDockPanel(showing));
      }
      return;
    }
    if (!MOVE_KEYS.has(e.key)) return;
    e.preventDefault();
    const next = nextKey(keys, showing, e.key);
    if (next !== null && next !== showing) this._show(next);
  }

  /** @param {DragEvent} e */
  _onStripDragStart(e) {
    if (!(e.target instanceof Element)) return;
    const item = e.target.closest(`.${TAB_ITEM_CLASS}`);
    const key = item instanceof HTMLElement ? item.getAttribute(TAB_KEY_ATTRIBUTE) : null;
    if (key !== null) dockDragStart({ tab: key }, e.dataTransfer);
  }

  dispose() {
    for (const d of this._disposables.splice(0)) d.dispose();
    for (const s of this._groupSubscriptions.splice(0)) s.dispose?.();
    for (const s of this._titleSubscriptions.splice(0)) s.dispose?.();
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
