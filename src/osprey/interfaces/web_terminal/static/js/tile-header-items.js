// @ts-check
/* OSPREY Web Terminal — Tile Header-Bar Contribution Items (hub side)
 *
 * Creates and patches the DOM for one contributed item. Split out of
 * tile-header-contrib.js, which keeps the message plumbing, the per-panel
 * store, the reconcile pass and the fold ladder; this module knows only how a
 * single item of each kind looks and behaves. The panel vocabulary itself is
 * documented once, on the panel side: design_system/static/js/header-contrib.js.
 *
 * Protocol v3 kinds:
 *
 *   - `icon`: one icon-only button in the panel-icon slot — an
 *     `.osprey-icon-btn.contrib-icon` named by its `title` (tooltip and
 *     accessible name), `aria-pressed` when the item says `pressed`.
 *   - `menu`: renders NO control. The entries live in the tile's one ⋯ menu
 *     (panel-menu-policy.js reads them from the contribution store), so the
 *     bar holds only an empty, hidden holder for the item. The holder keeps
 *     the item's `data-contrib-key` like every other item, which is what the
 *     keyed reconcile and the per-item accounting (one element per item)
 *     rely on.
 *
 * Legacy kinds, rendered exactly as before for CANOPY screens and panels not
 * yet migrated: `button` (text, icon-only or icon + label), `nav` (segmented
 * switcher), `search` (magnifier pill), plus the hub's own `crumb`
 * (`root › leaf` breadcrumb standing in for the tile's name) and `sep`
 * (hairline between control groups).
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

import { icon } from '/design-system/js/icons.js';
import { writeTitle } from '/design-system/js/tooltip.js';
import { svgIcon } from './svg-icons.js';

/** Debounce before a search item reports its text back to the panel. */
const SEARCH_DEBOUNCE_MS = 200;

/**
 * @typedef {import('/design-system/js/header-contrib.js').HeaderItem} PanelHeaderItem
 * @typedef {Omit<PanelHeaderItem, 'kind'> & {
 *   kind: PanelHeaderItem['kind'] | 'crumb' | 'sep',
 *   root?: string,
 *   leaf?: string,
 * }} HeaderItem  the panel vocabulary plus the hub-side kinds and fields
 * @typedef {(id: string, value?: string) => void} Dispatch
 */

/**
 * The child layout a live button was last built for: `''` for a plain
 * `.contrib-btn`, else the icon name and whether a label sits beside it. An
 * icon item's entry is its icon name alone.
 * @type {WeakMap<HTMLButtonElement, string>}
 */
const buttonShape = new WeakMap();

/**
 * Build the element for an item the bar has not rendered before.
 * @param {HeaderItem} item
 * @param {Dispatch} dispatch
 * @returns {HTMLElement}
 */
export function createItem(item, dispatch) {
  if (item.kind === 'icon') {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'osprey-icon-btn contrib-icon';
    btn.addEventListener('click', () => dispatch(item.id));
    patchIcon(item, btn);
    return btn;
  }
  if (item.kind === 'menu') {
    // The entries render in the tile ⋯, never here (module doc).
    const holder = document.createElement('span');
    holder.hidden = true;
    return holder;
  }
  if (item.kind === 'button') {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.addEventListener('click', () => dispatch(item.id));
    patchButton(item, btn);
    return btn;
  }
  if (item.kind === 'crumb') return buildCrumb(item);
  if (item.kind === 'sep') {
    const sep = document.createElement('span');
    sep.className = 'osprey-head-sep';
    sep.setAttribute('aria-hidden', 'true');
    return sep;
  }
  if (item.kind === 'search') return buildSearch(item, dispatch);
  return buildNav(item, dispatch);
}

/**
 * Update a live element to a new contribution of the same kind and id.
 * @param {HeaderItem} item
 * @param {HTMLElement} el
 * @returns {HTMLElement}
 */
export function patchItem(item, el) {
  if (item.kind === 'icon') {
    patchIcon(item, /** @type {HTMLButtonElement} */ (el));
  } else if (item.kind === 'button') {
    patchButton(item, /** @type {HTMLButtonElement} */ (el));
  } else if (item.kind === 'crumb') {
    patchCrumb(item, el);
  } else if (item.kind === 'nav') {
    patchNav(item, el);
  } else if (item.kind === 'search') {
    // Placeholder only. The value belongs to whoever is typing.
    const input = el.querySelector('input');
    if (input) input.placeholder = item.placeholder ?? '';
  }
  // A menu holder carries nothing to patch: the store has the new entries.
  return el;
}

/* ---- construction ---- */

/**
 * The `root › leaf` breadcrumb. All three parts always exist, so a leaf
 * that comes and goes is a visibility patch, not a rebuild.
 * @param {HeaderItem} item
 * @returns {HTMLElement}
 */
function buildCrumb(item) {
  const crumb = document.createElement('span');
  crumb.className = 'osprey-crumb';
  const root = document.createElement('span');
  root.className = 'osprey-crumb-root';
  const sep = document.createElement('span');
  sep.className = 'osprey-crumb-sep';
  sep.setAttribute('aria-hidden', 'true');
  sep.appendChild(icon('chevron-right'));
  const leaf = document.createElement('span');
  leaf.className = 'osprey-crumb-leaf';
  crumb.append(root, sep, leaf);
  patchCrumb(item, crumb);
  return crumb;
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
    clearTimeout(timer ?? undefined);
    timer = null;
    dispatch(item.id, input.value);
  };
  input.addEventListener('input', () => {
    clearTimeout(timer ?? undefined);
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

/* ---- patching ---- */

/** @param {HTMLElement} el @param {string} text */
function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

/**
 * Patch a v3 icon button in place: the pictogram is swapped only when the
 * icon name changes; title, disabled and pressed states are property patches,
 * so the button under the pointer is never replaced.
 * @param {HeaderItem} item
 * @param {HTMLButtonElement} btn
 */
function patchIcon(item, btn) {
  const name = item.icon ?? '';
  if (buttonShape.get(btn) !== name) {
    buttonShape.set(btn, name);
    btn.replaceChildren(icon(name));
  }
  const title = item.title ?? '';
  writeTitle(btn, title);
  btn.setAttribute('aria-label', title);
  btn.disabled = item.disabled === true;
  if (typeof item.pressed === 'boolean') {
    btn.setAttribute('aria-pressed', item.pressed ? 'true' : 'false');
  } else {
    btn.removeAttribute('aria-pressed');
  }
}

/**
 * Patch a legacy button in place. Its children are rebuilt only when its
 * layout changes (another icon, or a label that appears or goes); a new label
 * or disabled state on the same layout is a text or property patch, so the
 * button under the pointer is never replaced and its icon never wiped.
 * @param {HeaderItem} item
 * @param {HTMLButtonElement} btn
 */
function patchButton(item, btn) {
  const name = item.icon ?? '';
  const label = item.label ?? '';
  const iconOnly = name !== '' && label === '';
  const iconLabel = name !== '' && label !== '';
  const accent = item.tone === 'accent';
  const shape = name ? `${name}:${iconLabel ? 'label' : 'bare'}` : '';
  if (buttonShape.get(btn) !== shape) {
    buttonShape.set(btn, shape);
    btn.replaceChildren();
    if (name) btn.appendChild(icon(name));
    if (iconLabel) {
      const span = document.createElement('span');
      span.className = 'osprey-btn-label';
      btn.appendChild(span);
    }
  }
  btn.classList.toggle('contrib-btn', !name);
  btn.classList.toggle('contrib-btn-accent', !name && accent);
  btn.classList.toggle('osprey-icon-btn', iconOnly);
  btn.classList.toggle('osprey-btn', iconLabel);
  btn.classList.toggle('primary', iconLabel && accent);
  btn.disabled = item.disabled === true;

  if (!name) {
    setText(btn, label);
  } else if (iconLabel) {
    setText(/** @type {HTMLElement} */ (btn.querySelector('.osprey-btn-label')), label);
  }

  // An icon-only button has no text, so its title is its accessible name;
  // a labelled one is named by its label and only borrows it as a tooltip.
  const title = item.title || (iconLabel ? label : '');
  if (title) btn.title = title;
  else btn.removeAttribute('title');
  if (iconOnly) btn.setAttribute('aria-label', item.title ?? '');
  else btn.removeAttribute('aria-label');
}

/**
 * @param {HeaderItem} item
 * @param {HTMLElement} crumb
 */
function patchCrumb(item, crumb) {
  const [root, sep, leaf] = /** @type {HTMLElement[]} */ ([...crumb.children]);
  setText(root, item.root ?? '');
  const leafText = item.leaf ?? '';
  setText(leaf, leafText);
  if (leafText) leaf.title = leafText;
  else leaf.removeAttribute('title');
  sep.hidden = !leafText;
  leaf.hidden = !leafText;
}

/**
 * Patch a nav's entries in place where the shape allows, so switching views
 * does not rebuild the strip under the pointer.
 * @param {HeaderItem} item
 * @param {HTMLElement} nav
 */
function patchNav(item, nav) {
  const entries = /** @type {{id: string, label: string, active?: boolean}[]} */ (item.items ?? []);
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
