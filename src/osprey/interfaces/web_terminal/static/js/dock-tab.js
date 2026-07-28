// @ts-check
/* OSPREY Web Terminal — Tile Header Bar (custom dockview tab renderer)
 *
 * Under the one-panel-per-tile invariant a tile's "tab" can never have
 * siblings, so instead of a tab it renders the tile's HOST HEADER BAR:
 * grip + title + popout + close for service placeholders, and grip +
 * the adopted live `.terminal-header` + close for the terminal card.
 * Registered via dockview's createTabComponent/defaultTabComponent
 * (dock-workspace.js), which keeps dockview's native tab DnD, drop
 * hit-testing and close plumbing intact — the bar IS the drag handle.
 */

import { TERMINAL_RAIL_ID } from './panel-catalog.js';
import { PLACEHOLDER_PREFIX } from './dock-reconcile.js';

/** defaultTabComponent name registered on the dockview instance. */
export const OSPREY_TAB_COMPONENT = 'osprey-tile-tab';

/**
 * Resolves a service panel id to its standalone (non-embedded) URL, or null
 * when unknown. Registered by panel-manager.js (which owns the runtime URL
 * state) via setStandaloneUrlResolver — same late-binding idiom as
 * dock-workspace's setServiceRedock, avoiding an import cycle.
 * @type {((panelId: string) => string | null) | null}
 */
let standaloneUrlResolver = null;

/** @param {(panelId: string) => string | null} fn */
export function setStandaloneUrlResolver(fn) {
  standaloneUrlResolver = fn;
}

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

/**
 * @param {string} className
 * @param {string} glyph
 * @param {string} label
 * @returns {HTMLButtonElement}
 */
function actionButton(className, glyph, label) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = className;
  btn.textContent = glyph;
  btn.title = label;
  btn.setAttribute('aria-label', label);
  return btn;
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
    /** @type {HTMLSpanElement | null} */
    this._title = null;
    /** @type {any} */
    this._api = null;

    const root = document.createElement('div');
    root.className = 'tile-tab';

    const grip = document.createElement('span');
    grip.className = 'tile-tab-grip';
    grip.setAttribute('aria-hidden', 'true');
    root.appendChild(grip);

    const isTerminal = id === TERMINAL_RAIL_ID;
    if (isTerminal) {
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
      this._title = document.createElement('span');
      this._title.className = 'tile-tab-title';
      root.appendChild(this._title);
    }

    const actions = document.createElement('div');
    actions.className = 'tile-tab-actions';
    // The DefaultTab idiom: preventDefault on pointerdown keeps an action
    // press from starting a drag/focus dance; the click still fires.
    actions.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    if (!isTerminal) {
      const popout = actionButton('tile-tab-popout', '↗', 'Open standalone page');
      popout.addEventListener('click', (e) => {
        if (e.defaultPrevented) return;
        e.preventDefault();
        const serviceId = this._panelId.startsWith(PLACEHOLDER_PREFIX)
          ? this._panelId.slice(PLACEHOLDER_PREFIX.length)
          : this._panelId;
        const url = standaloneUrlResolver?.(serviceId);
        if (url) window.open(url, '_blank', 'noopener');
      });
      actions.appendChild(popout);
    }
    const close = actionButton('tile-tab-close', '×', 'Close tile');
    close.addEventListener('click', (e) => {
      if (e.defaultPrevented) return;
      e.preventDefault();
      this._api?.close?.();
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
    if (this._title) {
      this._title.textContent = params.title ?? '';
      const sub = params.api?.onDidTitleChange?.((/** @type {any} */ e) => {
        if (this._title) this._title.textContent = e?.title ?? '';
      });
      if (sub?.dispose) this._disposables.push(sub);
    }
  }

  dispose() {
    for (const d of this._disposables.splice(0)) d.dispose();
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
