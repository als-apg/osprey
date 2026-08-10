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
 * SECURITY: a message's panel identity is resolved by matching event.source
 * against the live panel iframes' contentWindow — never from a panel id
 * claimed in the payload, so one panel cannot inject controls into another
 * tile's bar. All rendering uses textContent (no innerHTML).
 *
 * Narrow tiles: text items ellipsize via CSS first; when the region still
 * overflows, whole items are hidden lowest-priority-first (re-measured on
 * every render and host resize). No overflow menu — curated bars are sparse.
 */

import { sendHeaderActionToIframe } from './panel-iframe-sync.js';

/** Hard caps so a misbehaving panel cannot flood the bar. */
const MAX_ITEMS = 12;
const MAX_TEXT = 120;

/**
 * @typedef {import('/design-system/js/header-contrib.js').HeaderItem} HeaderItem
 */

/** @type {Map<string, HeaderItem[]>} last valid contribution per panel id */
const contributions = new Map();
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
    if (!panelId) return;
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
  for (const item of raw.slice(0, MAX_ITEMS)) {
    if (!item || typeof item !== 'object' || typeof (/** @type {any} */ (item).id) !== 'string') {
      continue;
    }
    const it = /** @type {any} */ (item);
    const base = { id: it.id, priority: typeof it.priority === 'number' ? it.priority : 0 };
    if (it.kind === 'text' && typeof it.text === 'string') {
      out.push({ ...base, kind: 'text', text: clip(it.text) });
    } else if (it.kind === 'button' && typeof it.label === 'string') {
      out.push({
        ...base,
        kind: 'button',
        label: clip(it.label),
        title: typeof it.title === 'string' ? clip(it.title) : undefined,
        tone: it.tone === 'accent' ? 'accent' : 'default',
        disabled: it.disabled === true,
      });
    } else if (it.kind === 'nav' && Array.isArray(it.items)) {
      const entries = it.items
        .slice(0, MAX_ITEMS)
        .filter(
          (/** @type {any} */ n) =>
            n && typeof n.id === 'string' && typeof n.label === 'string'
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
 * Register a TileTab's contribution region. Renders immediately if this
 * panel's contribution already arrived; starts overflow re-measuring.
 * @param {string} panelId
 * @param {HTMLElement} host
 */
export function registerContribHost(panelId, host) {
  hosts.set(panelId, host);
  render(panelId, host);
  const ro = new ResizeObserver(() => applyPriorityHide(host));
  ro.observe(host);
  observers.set(panelId, ro);
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
}

/**
 * (Re)render a panel's contribution into its region.
 * @param {string} panelId
 * @param {HTMLElement} host
 */
function render(panelId, host) {
  const items = contributions.get(panelId) ?? [];
  host.replaceChildren();
  host.classList.toggle('has-items', items.length > 0);
  for (const item of items) {
    const el = renderItem(panelId, item);
    el.dataset.priority = String(item.priority ?? 0);
    host.appendChild(el);
  }
  applyPriorityHide(host);
}

/**
 * @param {string} panelId
 * @param {HeaderItem} item
 * @returns {HTMLElement}
 */
function renderItem(panelId, item) {
  if (item.kind === 'text') {
    const span = document.createElement('span');
    span.className = 'contrib-text';
    span.textContent = item.text ?? '';
    return span;
  }
  if (item.kind === 'button') {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'contrib-btn';
    if (item.tone === 'accent') btn.classList.add('contrib-btn-accent');
    btn.textContent = item.label ?? '';
    if (item.title) btn.title = item.title;
    btn.disabled = item.disabled === true;
    btn.addEventListener('click', () => dispatchAction(panelId, item.id));
    return btn;
  }
  // nav
  const nav = document.createElement('span');
  nav.className = 'contrib-nav';
  for (const entry of item.items ?? []) {
    const link = document.createElement('button');
    link.type = 'button';
    link.className = 'contrib-nav-link';
    link.classList.toggle('active', entry.active === true);
    link.textContent = entry.label;
    link.addEventListener('click', () => dispatchAction(panelId, item.id, entry.id));
    nav.appendChild(link);
  }
  return nav;
}

/**
 * @param {string} panelId
 * @param {string} id
 * @param {string} [value]
 */
function dispatchAction(panelId, id, value) {
  sendHeaderActionToIframe(iframeForPanel(panelId), id, value);
}

/**
 * Hide lowest-priority items until the region no longer overflows. Reveals
 * everything first so a widened tile gets its items back.
 * @param {HTMLElement} host
 */
function applyPriorityHide(host) {
  const children = /** @type {HTMLElement[]} */ ([...host.children]);
  for (const c of children) c.classList.remove('contrib-hidden');
  const byPriority = [...children].sort(
    (a, b) => Number(a.dataset.priority) - Number(b.dataset.priority)
  );
  for (const victim of byPriority) {
    if (host.scrollWidth <= host.clientWidth) break;
    victim.classList.add('contrib-hidden');
  }
}
