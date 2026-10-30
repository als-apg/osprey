// @ts-check
/* OSPREY Web Terminal — Tile Header-Bar Contributions (hub side)
 *
 * Receives embedded panels' `osprey-header-contribution` messages (see
 * design_system/static/js/header-contrib.js for the panel side and the item
 * vocabulary) and renders the items into the tile bar's contribution region.
 * The cross-iframe counterpart of the terminal tile's header adoption: the
 * bar stays the one header a tile has, and the panel's controls live IN it.
 *
 * Contributions are cached per panel id, not per TileTab: dock-tab.js
 * rebuilds a TileTab on every close/reopen/mode-switch, while a panel's
 * iframe (and therefore its last contribution) lives for the page. TileTabs
 * register their contribution host element here on construction; whichever
 * of (message arrived, host registered) happens second triggers the render.
 *
 * The hub can also fill a tile's header itself: contributeLocal takes the
 * same items (through the same sanitizer) plus a local dispatch that every
 * action of that panel's bar goes to instead of the iframe. From then on the
 * hub owns that header, so the panel's own contribution messages are
 * ignored — the Artifacts tile's iframe is a viewer of one artifact, and its
 * breadcrumb, pager and actions belong to the hub's sidebar.
 *
 * SECURITY: a message's panel identity is resolved by matching event.source
 * against the live panel iframes' contentWindow — never from a panel id
 * claimed in the payload, so one panel cannot inject controls into another
 * tile's bar. All rendering uses textContent (no innerHTML).
 *
 * Narrow tiles run an overflow LADDER (re-measured on every render and host
 * resize) in which nothing silently vanishes: text ellipsizes via CSS, then
 * a search field collapses to its magnifier (click/Enter re-expands it while
 * focused), then remaining controls fold lowest-priority-first into a
 * bar-owned ⋯ menu that keeps every folded control reachable. Text
 * (subtitle), the breadcrumb and separators never fold — text truncates —
 * and search never folds either: it collapses.
 *
 * Rendering is a keyed reconcile, not a rebuild. Items are matched on
 * `kind:id`; a survivor is patched in place and never detached, because a
 * contributed search box may be focused and mid-word when its panel re-sends.
 * The per-kind DOM lives in tile-header-items.js.
 */

import { sendHeaderActionToIframe } from './panel-iframe-sync.js';
import { createItem, disposeItem, patchItem, toggleOverflowMenu } from './tile-header-items.js';
import { svgIcon } from './svg-icons.js';
import { ICON_NAMES } from '/design-system/js/icons.js';

/** Hard caps so a misbehaving panel cannot flood the bar. */
const MAX_ITEMS = 12;
const MAX_TEXT = 120;

/**
 * @typedef {import('./tile-header-items.js').HeaderItem} HeaderItem
 * @typedef {import('./tile-header-items.js').Dispatch} Dispatch
 */

/** @type {Map<string, HeaderItem[]>} last valid contribution per panel id */
const contributions = new Map();
/** @type {Map<string, Dispatch>} action sink per panel whose header the hub contributes */
const localDispatch = new Map();
/** @type {Map<string, HTMLElement>} live contribution region per panel id */
const hosts = new Map();
/** @type {Map<string, ResizeObserver>} per-host overflow re-measurers */
const observers = new Map();

let wired = false;

/** Wire the window-level receive side (idempotent; called from app init). */
export function initHeaderContrib() {
  if (wired) return;
  wired = true;
  window.addEventListener('message', (e) => {
    if (e.origin !== window.location.origin) return;
    const data = e.data;
    if (!data || data.type !== 'osprey-header-contribution') return;
    const panelId = panelIdForSource(e.source);
    if (!panelId || localDispatch.has(panelId)) return;
    contributions.set(panelId, sanitizeItems(data.items));
    const host = hosts.get(panelId);
    if (host) render(panelId, host);
  });
}

/**
 * Resolve which panel iframe a message came from. Queries the live DOM each
 * time rather than keeping a registry: panel iframes are few, contributions
 * are rare, and the DOM is the ground truth (a detached iframe simply no
 * longer matches).
 * @param {MessageEventSource | null} source
 * @returns {string | null}
 */
function panelIdForSource(source) {
  if (!source) return null;
  for (const el of document.querySelectorAll('iframe.panel-iframe')) {
    const iframe = /** @type {HTMLIFrameElement} */ (el);
    if (iframe.contentWindow === source && iframe.dataset.panelId) {
      return iframe.dataset.panelId;
    }
  }
  return null;
}

/** @param {string} panelId @returns {HTMLIFrameElement | null} */
function iframeForPanel(panelId) {
  for (const el of document.querySelectorAll('iframe.panel-iframe')) {
    const iframe = /** @type {HTMLIFrameElement} */ (el);
    if (iframe.dataset.panelId === panelId) return iframe;
  }
  return null;
}

/**
 * Validate/clamp an incoming contribution to the closed vocabulary. Unknown
 * kinds and malformed entries are dropped, strings are stringified and
 * length-capped; the result is safe to render verbatim via textContent.
 * @param {unknown} raw
 * @returns {HeaderItem[]}
 */
function sanitizeItems(raw) {
  if (!Array.isArray(raw)) return [];
  /** @type {HeaderItem[]} */
  const out = [];
  /** Reconcile keys the DOM by `kind:id`; a repeat would alias one element. */
  const seen = new Set();
  for (const item of raw.slice(0, MAX_ITEMS)) {
    if (!item || typeof item !== 'object' || typeof (/** @type {any} */ (item).id) !== 'string') {
      continue;
    }
    const it = /** @type {any} */ (item);
    const key = `${it.kind}:${it.id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const base = { id: it.id, priority: typeof it.priority === 'number' ? it.priority : 0 };
    if (it.kind === 'text' && typeof it.text === 'string') {
      out.push({ ...base, kind: 'text', text: clip(it.text) });
    } else if (it.kind === 'search') {
      out.push({
        ...base,
        kind: 'search',
        placeholder: typeof it.placeholder === 'string' ? clip(it.placeholder) : '',
        value: typeof it.value === 'string' ? clip(it.value) : '',
      });
    } else if (it.kind === 'menu' && Array.isArray(it.items)) {
      const entries = it.items
        .slice(0, MAX_ITEMS)
        .filter(
          (/** @type {any} */ n) => n && typeof n.id === 'string' && typeof n.label === 'string'
        )
        .map((/** @type {any} */ n) => ({
          id: n.id,
          label: clip(n.label),
          // Absent stays absent: an entry without `checked` is a plain
          // action and must not render as an unticked checkbox.
          ...(typeof n.checked === 'boolean' ? { checked: n.checked } : {}),
          disabled: n.disabled === true,
        }));
      if (entries.length) {
        out.push({
          ...base,
          kind: 'menu',
          label: typeof it.label === 'string' ? clip(it.label) : undefined,
          items: entries,
        });
      }
    } else if (it.kind === 'button') {
      // An unknown icon name is dropped from the item: icon() throws on one.
      const icon =
        typeof it.icon === 'string' && ICON_NAMES.includes(it.icon) ? it.icon : undefined;
      const label = typeof it.label === 'string' ? clip(it.label) : undefined;
      const title = typeof it.title === 'string' ? clip(it.title) : undefined;
      // Every button needs an accessible name: its label, or for an icon
      // button without one, its title.
      if (icon ? Boolean(label || title) : label !== undefined) {
        out.push({
          ...base,
          kind: 'button',
          label,
          title,
          ...(icon ? { icon } : {}),
          tone: it.tone === 'accent' ? 'accent' : 'default',
          disabled: it.disabled === true,
        });
      }
    } else if (it.kind === 'crumb' && typeof it.root === 'string') {
      out.push({
        ...base,
        kind: 'crumb',
        root: clip(it.root),
        leaf: typeof it.leaf === 'string' ? clip(it.leaf) : '',
      });
    } else if (it.kind === 'sep') {
      out.push({ ...base, kind: 'sep' });
    } else if (it.kind === 'nav' && Array.isArray(it.items)) {
      const entries = it.items
        .slice(0, MAX_ITEMS)
        .filter(
          (/** @type {any} */ n) => n && typeof n.id === 'string' && typeof n.label === 'string'
        )
        .map((/** @type {any} */ n) => ({
          id: n.id,
          label: clip(n.label),
          active: n.active === true,
        }));
      if (entries.length) out.push({ ...base, kind: 'nav', items: entries });
    }
  }
  return out;
}

/** @param {string} s @returns {string} */
function clip(s) {
  return String(s).slice(0, MAX_TEXT);
}

/**
 * Fill a panel's header from the hub itself. The items pass the same
 * sanitizer as a panel's message and replace that panel's items; the bar
 * renders now if the tile is up, else when it registers. Every action of
 * that bar (buttons, nav, menu, search, the ⋯ fold rows) goes to `dispatch`
 * instead of the iframe, and the panel's own contribution messages are
 * ignored from now on, because the hub owns this header.
 * @param {string} panelId
 * @param {unknown} items
 * @param {Dispatch} dispatch
 */
export function contributeLocal(panelId, items, dispatch) {
  localDispatch.set(panelId, dispatch);
  contributions.set(panelId, sanitizeItems(items));
  const host = hosts.get(panelId);
  if (host) render(panelId, host);
}

/**
 * Register a TileTab's contribution region. Renders immediately if this
 * panel's contribution already arrived; starts overflow re-measuring.
 * @param {string} panelId
 * @param {HTMLElement} host
 */
export function registerContribHost(panelId, host) {
  hosts.set(panelId, host);
  // A collapsed search expands (and re-collapses on blur) via delegation on
  // the host: the wrap elements come and go with every reconcile, while the
  // host lives as long as its TileTab — one listener each, never stacked.
  host.addEventListener('click', (e) => {
    const wrap = e.target instanceof Element ? e.target.closest('.contrib-search-collapsed') : null;
    if (wrap instanceof HTMLElement) expandSearch(wrap);
  });
  host.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const wrap = e.target instanceof Element ? e.target.closest('.contrib-search-collapsed') : null;
    if (wrap instanceof HTMLElement) {
      e.preventDefault();
      expandSearch(wrap);
    }
  });
  host.addEventListener('focusout', (e) => {
    const wrap = e.target instanceof Element ? e.target.closest('.contrib-search') : null;
    if (wrap instanceof HTMLElement && wrap.dataset.searchPinned) {
      delete wrap.dataset.searchPinned;
      applyOverflowLadder(panelId, host);
    }
  });
  render(panelId, host);
  const ro = new ResizeObserver(() => applyOverflowLadder(panelId, host));
  ro.observe(host);
  observers.set(panelId, ro);
}

/**
 * Re-open a ladder-collapsed search for typing. The pin keeps the ladder from
 * re-collapsing it underneath the caret (the expansion itself re-triggers
 * measurement); focusout unpins and re-runs the ladder.
 * @param {HTMLElement} wrap
 */
function expandSearch(wrap) {
  wrap.dataset.searchPinned = '1';
  wrap.classList.remove('contrib-search-collapsed');
  wrap.removeAttribute('role');
  wrap.removeAttribute('tabindex');
  wrap.removeAttribute('aria-label');
  wrap.querySelector('input')?.focus();
}

/**
 * Drop a TileTab's region on dispose (only if it is still the registered
 * one — a rebuild registers the replacement before the old tab disposes).
 * @param {string} panelId
 * @param {HTMLElement} host
 */
export function unregisterContribHost(panelId, host) {
  if (hosts.get(panelId) !== host) return;
  hosts.delete(panelId);
  observers.get(panelId)?.disconnect();
  observers.delete(panelId);
  // A menu popover lives on document.body, so it would outlive its button.
  for (const node of host.children) disposeItem(/** @type {HTMLElement} */ (node));
}

/**
 * (Re)render a panel's contribution into its region.
 * @param {string} panelId
 * @param {HTMLElement} host
 */
function render(panelId, host) {
  const items = contributions.get(panelId) ?? [];
  host.classList.toggle('has-items', items.length > 0);
  // A breadcrumb names the tile, so CSS hides the tile's own title beside it.
  host.classList.toggle(
    'has-crumb',
    items.some((item) => item.kind === 'crumb')
  );

  /** Live elements from the previous contribution, keyed as they were built. */
  /** @type {Map<string, HTMLElement>} */
  const live = new Map();
  for (const node of [...host.children]) {
    const el = /** @type {HTMLElement} */ (node);
    if (el.dataset.contribKey) live.set(el.dataset.contribKey, el);
  }

  /** @type {Dispatch} */
  const dispatch = (id, value) => dispatchAction(panelId, id, value);

  /** @type {HTMLElement[]} */
  const ordered = [];
  for (const item of items) {
    const key = `${item.kind}:${item.id}`;
    const existing = live.get(key);
    let el;
    if (existing) {
      live.delete(key);
      el = patchItem(item, existing);
    } else {
      el = createItem(item, dispatch);
      el.dataset.contribKey = key;
    }
    el.dataset.priority = String(item.priority ?? 0);
    ordered.push(el);
  }

  for (const stale of live.values()) {
    disposeItem(stale);
    stale.remove();
  }

  // Place in contribution order WITHOUT touching elements already in the
  // right slot: re-inserting a node detaches it first, which would blur a
  // focused search box. Only a genuine reorder moves anything.
  let ref = host.firstChild;
  for (const el of ordered) {
    if (ref === el) ref = el.nextSibling;
    else host.insertBefore(el, ref);
  }

  applyOverflowLadder(panelId, host);
}

/**
 * Route one of a bar's actions: to the hub's own handler when the hub
 * contributes this header, else back to the panel's iframe. Looked up per
 * action, so elements built before a contributeLocal still route there.
 * @param {string} panelId
 * @param {string} id
 * @param {string} [value]
 */
function dispatchAction(panelId, id, value) {
  const local = localDispatch.get(panelId);
  if (!local) sendHeaderActionToIframe(iframeForPanel(panelId), id, value);
  else if (value === undefined) local(id);
  else local(id, value);
}

/**
 * The overflow ladder: reset to the full arrangement (so a widened tile gets
 * everything back), then give way in identity-preserving order —
 *
 *   1. the search field collapses to its magnifier (still clickable; the
 *      pin set by expandSearch exempts one being typed in), then
 *   2. remaining CONTROLS fold lowest-priority-first into a bar-owned
 *      ⋯ menu, so nothing becomes unreachable on a narrow tile.
 *
 * Text (subtitle), the breadcrumb and separators never fold — text
 * truncates via CSS — and search never folds past its magnifier. The ⋯
 * button is appended before the fold loop measures, so the space it
 * consumes is part of what the loop solves for.
 * @param {string} panelId
 * @param {HTMLElement} host
 */
function applyOverflowLadder(panelId, host) {
  const prev = host.querySelector('.contrib-overflow-btn');
  if (prev instanceof HTMLElement) {
    disposeItem(prev); // closes the popover if it is the open one
    prev.remove();
  }
  const children = /** @type {HTMLElement[]} */ ([...host.children]);
  for (const c of children) {
    c.classList.remove('contrib-hidden');
    if (c.classList.contains('contrib-search') && !c.dataset.searchPinned) {
      c.classList.remove('contrib-search-collapsed');
      c.removeAttribute('role');
      c.removeAttribute('tabindex');
      c.removeAttribute('aria-label');
    }
  }

  const crowded = () => host.scrollWidth > host.clientWidth + 1;
  if (!crowded()) return;

  const search = children.find(
    (c) => c.classList.contains('contrib-search') && !c.dataset.searchPinned
  );
  if (search) {
    search.classList.add('contrib-search-collapsed');
    // Collapsed, the wrap is itself the "open search" control.
    search.setAttribute('role', 'button');
    search.setAttribute('tabindex', '0');
    search.setAttribute('aria-label', search.querySelector('input')?.placeholder || 'Search');
    if (!crowded()) return;
  }

  const foldable = children.filter(
    (c) =>
      c.dataset.contribKey &&
      !c.classList.contains('contrib-search') &&
      !c.classList.contains('contrib-text') &&
      !c.classList.contains('osprey-crumb') &&
      !c.classList.contains('osprey-head-sep')
  );
  if (!foldable.length) return;

  const btn = buildOverflowButton(panelId);
  host.appendChild(btn);
  const byPriority = [...foldable].sort(
    (a, b) => Number(a.dataset.priority) - Number(b.dataset.priority)
  );
  /** @type {string[]} */
  const foldedKeys = [];
  for (const victim of byPriority) {
    if (!crowded()) break;
    victim.classList.add('contrib-hidden');
    if (victim.dataset.contribKey) foldedKeys.push(victim.dataset.contribKey);
  }
  if (!foldedKeys.length) {
    btn.remove();
    return;
  }
  btn.dataset.foldedKeys = foldedKeys.join('\n');
}

/**
 * The bar-owned ⋯ button holding whatever the fold loop took. Rows are
 * derived at CLICK time from the panel's current contribution, so a
 * re-contribution between fold and click can never show stale entries.
 * @param {string} panelId
 * @returns {HTMLButtonElement}
 */
function buildOverflowButton(panelId) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'contrib-menu-btn contrib-overflow-btn';
  btn.appendChild(svgIcon('ellipsis'));
  btn.title = 'More controls';
  btn.setAttribute('aria-label', 'More controls');
  btn.setAttribute('aria-haspopup', 'menu');
  btn.setAttribute('aria-expanded', 'false');
  btn.addEventListener('click', () => {
    const keys = btn.dataset.foldedKeys ? btn.dataset.foldedKeys.split('\n') : [];
    toggleOverflowMenu(btn, overflowRows(panelId, keys));
  });
  return btn;
}

/**
 * Flatten the folded controls into menu rows. A folded button is one row; a
 * folded nav contributes a row per view (active = checked); a folded menu's
 * entries are inlined. Each row dispatches exactly what the unfolded control
 * would have.
 * @param {string} panelId
 * @param {string[]} keys  contribKeys (`kind:id`) in folded order
 * @returns {import('./tile-header-items.js').OverflowRow[]}
 */
function overflowRows(panelId, keys) {
  const items = contributions.get(panelId) ?? [];
  /** @type {import('./tile-header-items.js').OverflowRow[]} */
  const rows = [];
  for (const key of keys) {
    const item = /** @type {any} */ (items.find((it) => `${it.kind}:${it.id}` === key));
    if (!item) continue;
    if (item.kind === 'button') {
      rows.push({
        // An icon-only button has no label; its title is its name.
        label: item.label || item.title || item.id,
        disabled: item.disabled === true,
        pick: () => dispatchAction(panelId, item.id),
      });
    } else if (item.kind === 'nav' || item.kind === 'menu') {
      for (const entry of item.items ?? []) {
        rows.push({
          label: entry.label,
          ...(item.kind === 'nav'
            ? { checked: entry.active === true }
            : typeof entry.checked === 'boolean'
              ? { checked: entry.checked }
              : {}),
          disabled: entry.disabled === true,
          pick: () => dispatchAction(panelId, item.id, entry.id),
        });
      }
    }
  }
  return rows;
}
