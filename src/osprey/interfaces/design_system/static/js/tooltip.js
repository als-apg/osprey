// @ts-check
/* OSPREY Design System — shared tooltip layer.
 *
 * One `div.osprey-tooltip[role=tooltip]`, appended to <body> on first use and
 * shared by every caller, so a scrolled container never cuts a tooltip and two
 * tooltips never show at once. It replaces native `title` tooltips where the
 * label must appear at once and in the theme's look: the collapsed sidebar,
 * whose rows are icons only.
 *
 * `initTooltips(root, {when})` shows an element's `data-tip` text when the
 * pointer enters it or it takes focus, for every `[data-tip]` element inside
 * `root`. The tooltip sits beside the element (10 px right of its right edge,
 * centred vertically). It hides on pointer leave, focus out and Escape.
 *
 * `when()` decides whether tooltips show at all: the shell passes "the sidebar
 * is collapsed", because the expanded sidebar already shows its labels. An
 * element whose `aria-controls` names an open `.osprey-popover` gets no
 * tooltip: the popover is already in that place.
 *
 * The tooltip only repeats a visible label; it is not the accessible name. The
 * element must carry its own `aria-label` (or text), so the layer sets no
 * `aria-describedby`.
 *
 * @module tooltip
 */

/** @type {HTMLDivElement | null} */
let layer = null;
let escapeBound = false;

/** @returns {HTMLDivElement} the shared layer, created on first use */
function tooltipLayer() {
  if (layer && layer.isConnected) return layer;
  layer = document.createElement('div');
  layer.className = 'osprey-tooltip';
  layer.setAttribute('role', 'tooltip');
  layer.hidden = true;
  document.body.append(layer);
  return layer;
}

/** Hide the shared tooltip. Safe to call when nothing shows. */
export function hideTooltip() {
  if (layer) layer.hidden = true;
}

/**
 * Whether `el` controls a popover that is open now.
 * @param {Element} el
 * @returns {boolean}
 */
function controlsOpenPopover(el) {
  const id = el.getAttribute('aria-controls');
  if (!id) return false;
  const controlled = document.getElementById(id);
  return !!controlled && controlled.classList.contains('osprey-popover') && !controlled.hidden;
}

/**
 * Show `el`'s `data-tip` beside it.
 * @param {HTMLElement} el
 */
function showTooltip(el) {
  const text = el.dataset.tip;
  if (!text) {
    hideTooltip();
    return;
  }
  const tip = tooltipLayer();
  tip.textContent = text;
  tip.hidden = false;
  const rect = el.getBoundingClientRect();
  tip.style.left = `${rect.right + 10}px`;
  tip.style.top = `${rect.top + rect.height / 2 - tip.offsetHeight / 2}px`;
}

/**
 * Show the tooltip layer for every `[data-tip]` element inside `root`.
 *
 * @param {HTMLElement} root the container to watch (the sidebar)
 * @param {{ when?: () => boolean }} [options] `when` returns false to keep
 *   every tooltip hidden; the default always shows
 * @returns {void}
 */
export function initTooltips(root, { when = () => true } = {}) {
  /** @param {Event} event @returns {HTMLElement | null} */
  const tipTarget = (event) => {
    const target = /** @type {Element | null} */ (event.target);
    const el = target && target.closest ? target.closest('[data-tip]') : null;
    return el instanceof HTMLElement && root.contains(el) ? el : null;
  };
  /** @param {Event} event */
  const maybeShow = (event) => {
    const el = tipTarget(event);
    if (!el || !when() || controlsOpenPopover(el)) {
      hideTooltip();
      return;
    }
    showTooltip(el);
  };

  root.addEventListener('pointerover', maybeShow);
  root.addEventListener('focusin', maybeShow);
  root.addEventListener('pointerleave', hideTooltip);
  root.addEventListener('focusout', hideTooltip);

  if (!escapeBound) {
    escapeBound = true;
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') hideTooltip();
    });
  }
}
