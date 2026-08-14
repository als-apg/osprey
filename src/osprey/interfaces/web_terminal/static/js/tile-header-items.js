// @ts-check
/* OSPREY Web Terminal — Tile Header-Bar Contribution Items (hub side)
 *
 * Creates and patches the DOM for one contributed item. Split out of
 * tile-header-contrib.js, which keeps the message plumbing, the per-panel
 * store and the reconcile pass; this module knows only how a single item of
 * each kind looks and behaves, plus the shared body-level popover both menu
 * kinds ride (a contributed ⋯ menu and the overflow ladder's fold target —
 * see toggleOverflowMenu). The item vocabulary itself is documented once,
 * on the panel side: design_system/static/js/header-contrib.js.
 *
 * Create vs. patch matters more than it looks. A contribution is a whole
 * replace, and a panel re-sends on every state change — including while an
 * operator is typing in a contributed search box. Rebuilding that input would
 * drop focus and caret on the keystroke that triggered the re-send, so the
 * reconcile pass in tile-header-contrib.js patches live elements in place and
 * only ever creates the ones it has not seen. `patchItem` is therefore the
 * common path, not the exception, and it must never replace a node it can
 * update.
 *
 * All text reaches the DOM via textContent. Nothing here interpolates markup.
 */

import { svgIcon } from './svg-icons.js';

/** Debounce before a search item reports its text back to the panel. */
const SEARCH_DEBOUNCE_MS = 200;

/**
 * @typedef {import('/design-system/js/header-contrib.js').HeaderItem} HeaderItem
 * @typedef {(id: string, value?: string) => void} Dispatch
 */

/** The item a live element currently represents (menus read it on open). */
const itemState = new WeakMap();

/** The one open menu popover, if any. @type {null | {btn: HTMLElement, pop: HTMLElement, off: () => void}} */
let openMenu = null;

/**
 * Build the element for an item the bar has not rendered before.
 * @param {HeaderItem} item
 * @param {Dispatch} dispatch
 * @returns {HTMLElement}
 */
export function createItem(item, dispatch) {
  const el = build(item, dispatch);
  itemState.set(el, item);
  return el;
}

/**
 * Update a live element to a new contribution of the same kind and id.
 * @param {HeaderItem} item
 * @param {HTMLElement} el
 * @returns {HTMLElement}
 */
export function patchItem(item, el) {
  itemState.set(el, item);
  if (item.kind === 'text') {
    setText(el, item.text ?? '');
  } else if (item.kind === 'button') {
    patchButton(item, /** @type {HTMLButtonElement} */ (el));
  } else if (item.kind === 'nav') {
    patchNav(item, el);
  } else if (item.kind === 'search') {
    // Placeholder only. The value belongs to whoever is typing.
    const input = el.querySelector('input');
    if (input) input.placeholder = item.placeholder ?? '';
  } else if (item.kind === 'menu') {
    const label = item.label || 'More options';
    el.title = label;
    el.setAttribute('aria-label', label);
    // A menu open right now is showing entries this contribution may have
    // changed; rebuild its rows against the new item rather than let it lie.
    if (openMenu && openMenu.btn === el) renderMenuRows(el);
  }
  return el;
}

/**
 * Release anything an element owns outside its own subtree. Called before the
 * reconcile pass drops an element the latest contribution no longer names.
 * @param {HTMLElement} el
 */
export function disposeItem(el) {
  if (openMenu && openMenu.btn === el) closeMenu();
}

/** Close any open contribution menu (host teardown, panel switch). */
export function closeMenu() {
  if (!openMenu) return;
  openMenu.off();
  openMenu.pop.remove();
  openMenu.btn.setAttribute('aria-expanded', 'false');
  openMenu = null;
}

/* ---- construction ---- */

/**
 * @param {HeaderItem} item
 * @param {Dispatch} dispatch
 * @returns {HTMLElement}
 */
function build(item, dispatch) {
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
    btn.addEventListener('click', () => dispatch(item.id));
    patchButton(item, btn);
    return btn;
  }
  if (item.kind === 'search') return buildSearch(item, dispatch);
  if (item.kind === 'menu') return buildMenu(item, dispatch);
  return buildNav(item, dispatch);
}

/**
 * @param {HeaderItem} item
 * @param {Dispatch} dispatch
 * @returns {HTMLElement}
 */
function buildNav(item, dispatch) {
  const nav = document.createElement('span');
  nav.className = 'contrib-nav';
  nav.addEventListener('click', (e) => {
    const link = e.target instanceof Element ? e.target.closest('.contrib-nav-link') : null;
    if (link instanceof HTMLElement && link.dataset.entryId) {
      dispatch(item.id, link.dataset.entryId);
    }
  });
  patchNav(item, nav);
  return nav;
}

/**
 * The magnifier pill. Deliberately shaped like `.command-palette-trigger` in
 * palette.css — an operator should read a panel's filter and the hub's own
 * search as the same kind of thing.
 * @param {HeaderItem} item
 * @param {Dispatch} dispatch
 * @returns {HTMLElement}
 */
function buildSearch(item, dispatch) {
  const wrap = document.createElement('span');
  wrap.className = 'contrib-search';

  const icon = document.createElement('span');
  icon.className = 'contrib-search-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.appendChild(svgIcon('search'));

  const input = document.createElement('input');
  input.type = 'search';
  input.className = 'contrib-search-input';
  input.autocomplete = 'off';
  input.placeholder = item.placeholder ?? '';
  input.value = item.value ?? '';
  input.setAttribute('aria-label', item.placeholder || 'Filter');

  /** @type {ReturnType<typeof setTimeout> | null} */
  let timer = null;
  const report = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    dispatch(item.id, input.value);
  };
  input.addEventListener('input', () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(report, SEARCH_DEBOUNCE_MS);
  });
  input.addEventListener('keydown', (e) => {
    // The bar is a dockview tab: keep typing away from shell shortcuts.
    e.stopPropagation();
    if (e.key === 'Escape' && input.value !== '') {
      input.value = '';
      report();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      report();
    }
  });

  wrap.append(icon, input);
  return wrap;
}

/**
 * @param {HeaderItem} item
 * @param {Dispatch} dispatch
 * @returns {HTMLElement}
 */
function buildMenu(item, dispatch) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'contrib-menu-btn';
  btn.appendChild(svgIcon('ellipsis'));
  btn.setAttribute('aria-haspopup', 'menu');
  btn.setAttribute('aria-expanded', 'false');
  const label = item.label || 'More options';
  btn.title = label;
  btn.setAttribute('aria-label', label);
  btn.addEventListener('click', () => {
    if (openMenu && openMenu.btn === btn) closeMenu();
    else openMenuFor(btn, dispatch);
  });
  return btn;
}

/* ---- patching ---- */

/** @param {HTMLElement} el @param {string} text */
function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

/**
 * @param {HeaderItem} item
 * @param {HTMLButtonElement} btn
 */
function patchButton(item, btn) {
  setText(btn, item.label ?? '');
  btn.classList.toggle('contrib-btn-accent', item.tone === 'accent');
  btn.disabled = item.disabled === true;
  if (item.title) btn.title = item.title;
  else btn.removeAttribute('title');
}

/**
 * Patch a nav's entries in place where the shape allows, so switching views
 * does not rebuild the strip under the pointer.
 * @param {HeaderItem} item
 * @param {HTMLElement} nav
 */
function patchNav(item, nav) {
  const entries = /** @type {{id: string, label: string, active?: boolean}[]} */ (
    item.items ?? []
  );
  const links = /** @type {HTMLElement[]} */ ([...nav.children]);
  const sameShape =
    links.length === entries.length &&
    entries.every((entry, i) => links[i].dataset.entryId === entry.id);

  if (!sameShape) {
    nav.replaceChildren(
      ...entries.map((entry) => {
        const link = document.createElement('button');
        link.type = 'button';
        link.className = 'contrib-nav-link';
        link.dataset.entryId = entry.id;
        return link;
      })
    );
  }
  const live = /** @type {HTMLElement[]} */ ([...nav.children]);
  entries.forEach((entry, i) => {
    setText(live[i], entry.label);
    live[i].classList.toggle('active', entry.active === true);
    live[i].setAttribute('aria-current', entry.active === true ? 'true' : 'false');
  });
}

/* ---- menu popover ----
 * Appended to document.body rather than inside the bar: the contribution
 * region is overflow:hidden and only 36px tall, so an in-place popover would
 * be clipped to nothing. Open/close grammar matches panel-add-menu.js and
 * display-menu.js — capture-phase outside pointerdown, Escape, and a
 * scroll/resize close rather than a reposition loop. */

/**
 * Open an empty popover anchored to btn and register it as THE open menu
 * (there is at most one, shared by contributed menus and the bar's own
 * overflow menu). The caller fills in rows and click handling.
 * @param {HTMLElement} btn
 * @returns {HTMLElement} the popover element
 */
function openPopover(btn) {
  closeMenu();
  const pop = document.createElement('div');
  pop.className = 'contrib-menu-popover';
  pop.setAttribute('role', 'menu');
  document.body.appendChild(pop);

  const onDocDown = (/** @type {Event} */ e) => {
    const t = e.target;
    if (t instanceof Node && (pop.contains(t) || btn.contains(t))) return;
    closeMenu();
  };
  const onKey = (/** @type {KeyboardEvent} */ e) => {
    if (e.key === 'Escape') closeMenu();
  };
  const onReflow = () => closeMenu();
  document.addEventListener('pointerdown', onDocDown, true);
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('resize', onReflow);
  window.addEventListener('scroll', onReflow, true);

  openMenu = {
    btn,
    pop,
    off: () => {
      document.removeEventListener('pointerdown', onDocDown, true);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('resize', onReflow);
      window.removeEventListener('scroll', onReflow, true);
    },
  };
  btn.setAttribute('aria-expanded', 'true');
  return pop;
}

/**
 * @param {HTMLElement} btn
 * @param {Dispatch} dispatch
 */
function openMenuFor(btn, dispatch) {
  const pop = openPopover(btn);
  pop.addEventListener('click', (e) => {
    const row = e.target instanceof Element ? e.target.closest('.contrib-menu-item') : null;
    if (!(row instanceof HTMLElement) || !row.dataset.entryId) return;
    const id = /** @type {HeaderItem} */ (itemState.get(btn))?.id;
    closeMenu();
    if (id) dispatch(id, row.dataset.entryId);
  });
  renderMenuRows(btn);
  place(pop, btn);
}

/**
 * One row of the bar's overflow menu — a control the overflow ladder folded
 * out of a narrow bar, carrying its own pick action.
 * @typedef {object} OverflowRow
 * @property {string} label
 * @property {boolean} [checked]
 * @property {boolean} [disabled]
 * @property {() => void} pick
 */

/**
 * Toggle the bar-owned overflow menu (tile-header-contrib.js's fold target).
 * One-shot rows: a re-render of the bar rebuilds the fold set, so the open
 * popover is simply closed by disposeItem/closeMenu rather than re-rendered
 * the way a contributed menu is.
 * @param {HTMLElement} btn
 * @param {OverflowRow[]} rows
 */
export function toggleOverflowMenu(btn, rows) {
  if (openMenu && openMenu.btn === btn) {
    closeMenu();
    return;
  }
  const pop = openPopover(btn);
  pop.replaceChildren(...rows.map((row) => {
    const el = rowElement(row.label, row.checked, row.disabled === true);
    el.addEventListener('click', () => {
      closeMenu();
      row.pick();
    });
    return el;
  }));
  place(pop, btn);
}

/**
 * (Re)build the open popover's rows from the button's current item.
 * @param {HTMLElement} btn
 */
function renderMenuRows(btn) {
  if (!openMenu || openMenu.btn !== btn) return;
  const item = /** @type {HeaderItem} */ (itemState.get(btn));
  const entries = /** @type {{id: string, label: string, checked?: boolean, disabled?: boolean}[]} */ (
    item?.items ?? []
  );
  openMenu.pop.replaceChildren(
    ...entries.map((entry) => {
      const row = rowElement(entry.label, entry.checked, entry.disabled === true);
      row.dataset.entryId = entry.id;
      return row;
    })
  );
}

/**
 * Build one popover row: tick gutter + label, with the checkbox role only
 * when the entry is actually checkable — a plain action must not claim one.
 * @param {string} label
 * @param {boolean | undefined} checked
 * @param {boolean} disabled
 * @returns {HTMLButtonElement}
 */
function rowElement(label, checked, disabled) {
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'contrib-menu-item';
  row.disabled = disabled;
  if (checked === undefined) {
    row.setAttribute('role', 'menuitem');
  } else {
    row.setAttribute('role', 'menuitemcheckbox');
    row.setAttribute('aria-checked', checked ? 'true' : 'false');
    row.classList.toggle('checked', checked);
  }
  const tick = document.createElement('span');
  tick.className = 'contrib-menu-tick';
  tick.setAttribute('aria-hidden', 'true');
  tick.textContent = checked ? '✓' : '';
  const text = document.createElement('span');
  text.textContent = label;
  row.append(tick, text);
  return row;
}

/**
 * Pin the popover under the button, right edges aligned, clamped to the
 * viewport so a tile near the right or bottom edge still shows a full menu.
 * @param {HTMLElement} pop
 * @param {HTMLElement} btn
 */
function place(pop, btn) {
  const r = btn.getBoundingClientRect();
  const w = pop.offsetWidth;
  const h = pop.offsetHeight;
  const margin = 4;
  let left = r.right - w;
  let top = r.bottom + margin;
  left = Math.max(margin, Math.min(left, window.innerWidth - w - margin));
  if (top + h > window.innerHeight - margin) top = Math.max(margin, r.top - h - margin);
  pop.style.left = `${Math.round(left)}px`;
  pop.style.top = `${Math.round(top)}px`;
}
