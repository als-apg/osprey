// @ts-check
/* OSPREY Design System — shared tooltip layer.
 *
 * One `div.osprey-tooltip[role=tooltip]`, appended to <body> on first use and
 * shared by every caller, so a scrolled container never cuts a tooltip and two
 * tooltips never show at once. It replaces native `title` tooltips where the
 * text must appear soon and in the theme's look.
 *
 * `initTooltips(root, {mode})` watches `root` (the shell passes <body>) and
 * shows a tip for the nearest element that carries one of:
 *   - `data-tip`: a short label, the name of the control;
 *   - `title`: a native hint, read as the label when there is no `data-tip`,
 *     so a control that only has a `title` still gets the styled tip;
 *   - `data-tip-detail`: one sentence that explains what the control does.
 * The tip is a bold label line over a muted detail line. `mode(el)` picks what
 * to show for an element: `'label+detail'` (the default) or `'detail'`, which
 * leaves the label out because the control already shows it (an expanded
 * sidebar row). A mode that yields no text shows no tip, and the native
 * `title` stays in place.
 *
 * Timing: the tip appears 450 ms after the pointer enters an element or the
 * element takes keyboard focus, and goes at once. A tip that shows within
 * 300 ms of the last one hiding skips the wait, so sweeping down a list reads
 * as one continuous tooltip.
 *
 * Placement: `data-tip-place` is `right` (default), `below` or `above`. The
 * tip sits 10 px from the element, flips to the opposite side when it would
 * leave the window, and is clamped inside an 8 px window margin.
 *
 * The tip hides on pointer leave, focus out, any pointer press, Escape, and
 * when it is the anchor of an `aria-controls` popover that opens. An element
 * whose `aria-controls` names an open `.osprey-popover` gets no tip: the
 * popover is already in that place. After a press or Escape the same element
 * stays quiet until the pointer or focus has left it.
 *
 * While the tip shows, the element's `aria-describedby` names the layer, so a
 * screen reader hears the detail with the control's own name. The tip is not
 * the accessible name: the element still needs its own `aria-label` or text.
 *
 * A native `title` would draw a second tooltip on top of the tip. While the
 * layer shows an element's tip, the `title` is parked in `data-tip-title` and
 * cleared; hiding puts it back. Code that retitles an element must go through
 * {@link readTitle} / {@link writeTitle}, so a title written while the element
 * is hovered lands in the parked copy instead of being lost or shown twice.
 *
 * @module tooltip
 */

/** @typedef {'label+detail' | 'detail'} TooltipMode */

/** Wait before a tip shows. */
const SHOW_DELAY_MS = 450;
/** A tip that shows this soon after one hid needs no wait. */
const WARM_PERIOD_MS = 300;
/** Space between the element and the tip. */
const GAP = 10;
/** The least space between the tip and a window edge. */
const MARGIN = 8;

const LAYER_ID = 'osprey-tooltip';

/** Where an element's native `title` waits while the layer shows its tip. */
const PARKED_TITLE_ATTR = 'data-tip-title';

/** Elements that can carry a tip. The parked title keeps a shown element matching. */
const TIP_SELECTOR = `[data-tip], [title], [${PARKED_TITLE_ATTR}], [data-tip-detail]`;

/** @type {HTMLDivElement | null} */
let layer = null;
let documentBound = false;

/** @type {HTMLElement | null} the element whose tip the layer shows now */
let current = null;

/** @type {HTMLElement | null} the element whose tip waits for the delay */
let pendingEl = null;

/** @type {ReturnType<typeof setTimeout> | undefined} */
let pendingTimer;

/** @type {HTMLElement | null} an element a press or Escape silenced until it is left */
let dismissed = null;

/** When the layer last went from shown to hidden. */
let lastHideAt = -Infinity;

/**
 * An element's own `title`, also while the layer has parked it.
 * @param {HTMLElement} el
 * @returns {string}
 */
export function readTitle(el) {
  return el.getAttribute(PARKED_TITLE_ATTR) ?? el.title;
}

/**
 * Set an element's own `title`; while the layer has parked it, update the
 * parked copy so the native tooltip stays off until the tip hides.
 * @param {HTMLElement} el
 * @param {string} text
 */
export function writeTitle(el, text) {
  if (el.hasAttribute(PARKED_TITLE_ATTR)) el.setAttribute(PARKED_TITLE_ATTR, text);
  else el.title = text;
}

/**
 * Take the layer's id out of an element's `aria-describedby`.
 * @param {HTMLElement} el
 */
function undescribe(el) {
  const rest = (el.getAttribute('aria-describedby') ?? '')
    .split(/\s+/)
    .filter((id) => id && id !== LAYER_ID);
  if (rest.length) el.setAttribute('aria-describedby', rest.join(' '));
  else el.removeAttribute('aria-describedby');
}

/**
 * Add the layer's id to an element's `aria-describedby`, keeping its own.
 * @param {HTMLElement} el
 */
function describe(el) {
  const ids = (el.getAttribute('aria-describedby') ?? '').split(/\s+/).filter(Boolean);
  if (!ids.includes(LAYER_ID)) ids.push(LAYER_ID);
  el.setAttribute('aria-describedby', ids.join(' '));
}

/** Put the shown element's parked `title` back and forget it. */
function releaseCurrent() {
  if (!current) return;
  undescribe(current);
  const parked = current.getAttribute(PARKED_TITLE_ATTR);
  if (parked !== null) {
    current.removeAttribute(PARKED_TITLE_ATTR);
    // A caller that set `title` directly while the tip showed wins.
    if (!current.title) current.title = parked;
  }
  current = null;
}

/** @returns {HTMLDivElement} the shared layer, created on first use */
function tooltipLayer() {
  if (layer && layer.isConnected) return layer;
  layer = document.createElement('div');
  layer.id = LAYER_ID;
  layer.className = 'osprey-tooltip';
  layer.setAttribute('role', 'tooltip');
  layer.hidden = true;
  document.body.append(layer);
  return layer;
}

/** Cancel a tip that is waiting for its delay. */
function cancelPending() {
  clearTimeout(pendingTimer);
  pendingTimer = undefined;
  pendingEl = null;
}

/** Hide the shared tooltip and cancel one that waits. Safe to call when nothing shows. */
export function hideTooltip() {
  cancelPending();
  if (layer && !layer.hidden) {
    layer.hidden = true;
    lastHideAt = Date.now();
  }
  releaseCurrent();
}

/** Hide the tip and keep its element quiet until it is left. */
function dismissTooltip() {
  dismissed = pendingEl ?? current ?? dismissed;
  hideTooltip();
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
 * What the tip for `el` says in `mode`.
 * @param {HTMLElement} el
 * @param {TooltipMode} mode
 * @returns {{ label: string, detail: string } | null} `null` when nothing is left to say
 */
function tipText(el, mode) {
  const detail = (el.dataset.tipDetail ?? '').trim();
  const label = mode === 'detail' ? '' : (el.dataset.tip || readTitle(el)).trim();
  return label || detail ? { label, detail } : null;
}

/**
 * Where a tip of `size` goes for an anchor at `rect`. It starts on the wanted
 * side, flips to the opposite one when it would overflow the window there, and
 * is then clamped inside the window margin.
 * @param {DOMRect} rect
 * @param {{ width: number, height: number }} size
 * @param {string | undefined} place `right`, `below` or `above`
 * @returns {{ left: number, top: number, side: 'right' | 'left' | 'below' | 'above' }}
 */
function placeTip(rect, size, place) {
  const maxLeft = window.innerWidth - MARGIN - size.width;
  const maxTop = window.innerHeight - MARGIN - size.height;
  /** @param {number} value @param {number} max */
  const clamp = (value, max) => Math.max(MARGIN, Math.min(value, max));
  const centreX = rect.left + rect.width / 2 - size.width / 2;
  const centreY = rect.top + rect.height / 2 - size.height / 2;

  if (place === 'below' || place === 'above') {
    const below = rect.bottom + GAP;
    const above = rect.top - GAP - size.height;
    const wantBelow = place === 'below';
    const fits = wantBelow ? below <= maxTop : above >= MARGIN;
    const useBelow = fits ? wantBelow : !wantBelow;
    return {
      left: clamp(centreX, maxLeft),
      top: clamp(useBelow ? below : above, maxTop),
      side: useBelow ? 'below' : 'above',
    };
  }
  const right = rect.right + GAP;
  const left = rect.left - GAP - size.width;
  const useRight = right <= maxLeft || left < MARGIN;
  return {
    left: clamp(useRight ? right : left, maxLeft),
    top: clamp(centreY, maxTop),
    side: useRight ? 'right' : 'left',
  };
}

/**
 * Show `el`'s tip now.
 * @param {HTMLElement} el
 * @param {{ label: string, detail: string }} text
 */
function presentTooltip(el, text) {
  const tip = tooltipLayer();
  if (current !== el) {
    releaseCurrent();
    current = el;
  }
  // Parked on every show, so a `title` set directly mid-hover replaces the
  // parked copy rather than showing beside the tip.
  if (el.hasAttribute('title')) {
    el.setAttribute(PARKED_TITLE_ATTR, el.title);
    el.removeAttribute('title');
  }
  describe(el);

  tip.replaceChildren();
  if (text.label) {
    const label = document.createElement('b');
    label.className = 'osprey-tooltip-label';
    label.textContent = text.label;
    tip.append(label);
  }
  if (text.detail) {
    const detail = document.createElement('span');
    detail.className = 'osprey-tooltip-detail';
    detail.textContent = text.detail;
    tip.append(detail);
  }
  // Measured at the window corner, so the size never depends on where the
  // previous tip stood.
  tip.style.left = '0px';
  tip.style.top = '0px';
  tip.hidden = false;
  const where = placeTip(
    el.getBoundingClientRect(),
    { width: tip.offsetWidth, height: tip.offsetHeight },
    el.dataset.tipPlace
  );
  tip.style.left = `${where.left}px`;
  tip.style.top = `${where.top}px`;
  tip.dataset.side = where.side;
}

/**
 * Whether a focused element shows its focus to the user (keyboard focus), as
 * opposed to the focus a mouse press gives it.
 * @param {HTMLElement} el
 * @returns {boolean}
 */
function showsFocus(el) {
  try {
    return el.matches(':focus-visible');
  } catch {
    return true;
  }
}

/**
 * Show a tip for every element inside `root` that carries `data-tip`,
 * `data-tip-detail` or a `title`.
 *
 * @param {HTMLElement} [root] the container to watch; the shell passes <body>
 * @param {{ mode?: (el: HTMLElement) => TooltipMode }} [options] `mode` picks
 *   what to show per element; the default shows label and detail everywhere
 * @returns {() => void} removes this call's listeners
 */
export function initTooltips(root = document.body, { mode = () => 'label+detail' } = {}) {
  /** @param {Event} event @returns {HTMLElement | null} */
  const tipTarget = (event) => {
    const target = /** @type {Element | null} */ (event.target);
    const el = target && target.closest ? target.closest(TIP_SELECTOR) : null;
    return el instanceof HTMLElement && root.contains(el) ? el : null;
  };

  /** @param {HTMLElement} el */
  const request = (el) => {
    if (el === current || el === pendingEl) return;
    hideTooltip();
    if (!tipText(el, mode(el)) || controlsOpenPopover(el)) return;
    /** Shows the tip if the element still has something to say. */
    const show = () => {
      cancelPending();
      const text = tipText(el, mode(el));
      if (text && !controlsOpenPopover(el)) presentTooltip(el, text);
    };
    if (Date.now() - lastHideAt < WARM_PERIOD_MS) {
      show();
      return;
    }
    pendingEl = el;
    pendingTimer = setTimeout(show, SHOW_DELAY_MS);
  };

  /** @param {Event} event */
  const onPointerOver = (event) => {
    const el = tipTarget(event);
    if (el !== dismissed) dismissed = null;
    if (!el) hideTooltip();
    else if (el !== dismissed) request(el);
  };

  /** @param {Event} event */
  const onFocusIn = (event) => {
    const el = tipTarget(event);
    if (!el || el === dismissed || !showsFocus(el)) return;
    request(el);
  };

  /** @param {Event} event */
  const onLeave = (event) => {
    const left = /** @type {HTMLElement} */ (event.target);
    if (left === dismissed) dismissed = null;
    if (left === root || left === (pendingEl ?? current)) hideTooltip();
  };

  /** @param {Event} event */
  const onFocusOut = (event) => {
    const el = tipTarget(event);
    if (el && el === dismissed) dismissed = null;
    if (el && el === (pendingEl ?? current)) hideTooltip();
  };

  /** @param {PointerEvent} event */
  const onPointerOut = (event) => {
    if (!event.relatedTarget) hideTooltip();
  };

  root.addEventListener('pointerover', onPointerOver);
  root.addEventListener('pointerout', onPointerOut);
  // `pointerleave` does not bubble, so the capture phase sees it for every
  // descendant; `onLeave` acts only when the element with the tip is left.
  root.addEventListener('pointerleave', onLeave, true);
  root.addEventListener('focusin', onFocusIn);
  root.addEventListener('focusout', onFocusOut);

  if (!documentBound) {
    documentBound = true;
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') dismissTooltip();
    });
    // Capture phase: a press hides the tip even when the control stops the event.
    document.addEventListener('pointerdown', dismissTooltip, true);
  }

  return () => {
    root.removeEventListener('pointerover', onPointerOver);
    root.removeEventListener('pointerout', onPointerOut);
    root.removeEventListener('pointerleave', onLeave, true);
    root.removeEventListener('focusin', onFocusIn);
    root.removeEventListener('focusout', onFocusOut);
  };
}
