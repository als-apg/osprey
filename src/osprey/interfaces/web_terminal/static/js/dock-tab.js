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
 *   service tiles — a visible `.tile-tab-title` + close. The close is
 *     a LOCAL tile close (the panel keeps its rail entry; dock-sync routes
 *     the click to a vacate handler); membership removal stays the rail
 *     entry's "×". Popout remains rail-only.
 *   the terminal tile — the adopted live `.terminal-header` + close.
 *     The header supplies the identity and more: session LED, session id,
 *     the session selector and "+ New" are frequently-used, stateful
 *     controls with nowhere better to live.
 *
 * Registered via dockview's createTabComponent/defaultTabComponent
 * (dock-workspace.js), which keeps dockview's native tab DnD, drop
 * hit-testing and close plumbing intact — the bar IS the drag handle.
 */

import { TERMINAL_RAIL_ID } from './panel-catalog.js';
import { registerContribHost, unregisterContribHost } from './tile-header-contrib.js';
import { PLACEHOLDER_PREFIX } from './dock-reconcile.js';
import { withEchoSuppressed } from './dock-sync.js';
import { svgIcon } from './svg-icons.js';

/** defaultTabComponent name registered on the dockview instance. */
export const OSPREY_TAB_COMPONENT = 'osprey-tile-tab';

/**
 * The page's one `.terminal-header` node, cached across tab rebuilds: after a
 * terminal close the node is detached, and the reopen path must re-adopt the
 * SAME live subtree so every reference held by terminal.js / session code
 * stays valid — the header-level counterpart of dock-workspace's adoptSubtree
 * mechanism.
 *
 * Registered explicitly via setTerminalHeaderSource — there is deliberately NO
 * lazy `document.querySelector` fallback here. dockview always constructs a
 * panel's content component before its tab component, so by the time the
 * terminal's TileTab is constructed, dock-workspace's own content adoption
 * (adoptSubtree) has already moved the whole `.terminal-panel` subtree into a
 * detached host div, ahead of dockview inserting that host into the live
 * document. A document-wide query at TileTab-construction time is therefore
 * GUARANTEED to miss the header (it exists, just not reachable from
 * `document` at that instant), not merely liable to — a fallback here would
 * just silently re-introduce that bug. Callers MUST register the reference up
 * front, before adoption can detach it (see dock-workspace.js's
 * initDockWorkspace, which does this before its first addPanel for the
 * terminal).
 * @type {HTMLElement | null}
 */
let terminalHeaderEl = null;

/**
 * Registers the live `.terminal-header` node. Called once by dock-workspace.js
 * before the terminal panel is added to dockview (see initDockWorkspace).
 * Also wires the interactive-children pointerdown guard directly on the node,
 * here rather than in the TileTab constructor: the header node is long-lived
 * across close/reopen/mode-switch, but a TileTab is rebuilt on every one of
 * those, so attaching in the constructor would stack a new listener on the
 * same node each time. Registration happens exactly once per node.
 * @param {HTMLElement} el
 */
export function setTerminalHeaderSource(el) {
  terminalHeaderEl = el;
  // Interactive children (session selector, + New) must act, not drag: stop
  // pointerdown from reaching dockview's tab drag tracking. Plain header
  // surface still bubbles, so the bar remains the drag handle.
  el.addEventListener('pointerdown', (e) => {
    if (e.target instanceof Element && e.target.closest(INTERACTIVE)) {
      e.stopPropagation();
    }
  });
}

/** @returns {HTMLElement | null} */
function terminalHeader() {
  return terminalHeaderEl;
}

/** Matches the interactive children of the adopted terminal header. */
const INTERACTIVE = 'button, select, input, a, [role="button"]';

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

    if (id === TERMINAL_RAIL_ID) {
      root.classList.add('tile-tab-terminal');
      const header = terminalHeader();
      if (header) {
        root.appendChild(header); // DOM relocation — live references survive
        // The interactive-children pointerdown guard is wired once, on the
        // node itself, by setTerminalHeaderSource — not here (see that
        // function's docstring for why attaching per-construction would
        // accumulate listeners).
      }
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
      // adopted terminal header; empty region surface still bubbles so the
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
    }

    const actions = document.createElement('div');
    actions.className = 'tile-tab-actions';
    // The DefaultTab idiom: preventDefault on pointerdown keeps an action
    // press from starting a drag/focus dance; the click still fires.
    actions.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'tile-tab-close';
    close.appendChild(svgIcon('close'));
    close.title = 'Close tile';
    close.setAttribute('aria-label', 'Close tile');
    close.addEventListener('click', (e) => {
      if (e.defaultPrevented) return;
      e.preventDefault();
      // The removal makes dockview auto-activate a surviving tile; that is a
      // side effect of the close, not a human focus gesture, so it must not
      // reach the focus reporter — the same suppression retireTile applies to
      // its own removal.
      withEchoSuppressed(() => this._api?.close?.());
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
    // one. The terminal bar is exempt: the header it adopted names it already.
    if (this._panelId === TERMINAL_RAIL_ID) return;
    /** @param {string} title */
    const applyName = (title) => {
      if (this._titleEl) this._titleEl.textContent = title;
      this._element.setAttribute('aria-label', title);
    };
    applyName(params.title ?? '');
    const sub = params.api?.onDidTitleChange?.((/** @type {any} */ e) => applyName(e?.title ?? ''));
    if (sub?.dispose) this._disposables.push(sub);
  }

  dispose() {
    for (const d of this._disposables.splice(0)) d.dispose();
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
