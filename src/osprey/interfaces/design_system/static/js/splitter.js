// @ts-check
/*
 * OSPREY Design System — shared pane splitter.
 *
 * One draggable divider used by every two-pane browser in the product: the OKF
 * knowledge panel's sidebar/reader split, the artifact gallery's browse view,
 * and the PLAN panel's plan-list/detail split. The visual half lives in
 * css/base.css as `.osprey-splitter` (the grip-pill idiom borrowed from the
 * hub's dockview sashes); this module is the behaviour half.
 *
 * Why shared: these panels are served by three different processes — the hub
 * serves OKF and artifacts in-process, while the PLAN panel comes from the
 * bluesky-panels sidecar container. They cannot import each other's code, and
 * `/design-system/` is the one path all of them resolve (the web terminal's
 * panel proxy deliberately serves the hub's copy to embedded panels). Three
 * hand-copied splitters drifted silently because nothing failed when one of
 * them stopped matching; this is the single definition instead.
 *
 * Behaviour: pointer drag with pointer capture (so a fast drag that leaves the
 * 5px track keeps resizing), width clamped to a caller-supplied range, applied
 * as the pane's flex-basis, and persisted per origin so the chosen split
 * survives reloads. The handle is an ARIA separator, so Arrow keys nudge it —
 * a splitter that only responds to a pointer is unusable by keyboard.
 */

/**
 * Clamp a candidate pane width to the usable range.
 *
 * @param {number} width
 * @param {number} min
 * @param {number} max
 * @returns {number} the clamped width, rounded to whole pixels
 */
export function clampWidth(width, min, max) {
  return Math.min(max, Math.max(min, Math.round(width)));
}

/**
 * @typedef {object} SplitterApi
 * @property {(width: number) => number} applyWidth Size the pane; returns the applied (clamped) width.
 * @property {() => void} clearWidth Drop the inline sizes, handing the pane back to the stylesheet.
 * @property {() => number|null} restoreWidth Re-apply the persisted width; null when nothing usable is stored.
 */

/**
 * Wire a pane splitter. Degrades in two steps rather than all-or-nothing, so
 * callers need no null dance of their own:
 *
 * - no pane (a stripped-down embed that ships one pane): everything no-ops,
 *   the returned API is safe to call;
 * - pane but no handle: the sizing API still works — a caller that restores a
 *   persisted width on some other trigger keeps doing so — only the drag and
 *   keyboard wiring are skipped, because there is nothing to wire them to.
 *
 * @param {object} opts
 * @param {HTMLElement|null} opts.handle The separator element.
 * @param {HTMLElement|null} opts.pane The pane being resized (the one before the handle).
 * @param {string} opts.storageKey localStorage key holding the chosen width.
 * @param {number} opts.min Smallest useful pane width, px.
 * @param {number} opts.max Largest useful pane width, px.
 * @param {number} [opts.step] Arrow-key increment, px. Defaults to 16.
 * @param {() => boolean} [opts.isEnabled] Gate for both drag and keyboard, re-read
 *   per interaction. Lets a caller park the splitter while its layout is on a
 *   different axis (the gallery's stacked mode) without tearing down listeners.
 * @returns {SplitterApi}
 */
export function initSplitter({
  handle,
  pane,
  storageKey,
  min,
  max,
  step = 16,
  isEnabled = () => true,
}) {
  if (!pane) {
    return {
      applyWidth: (width) => clampWidth(width, min, max),
      clearWidth: () => {},
      restoreWidth: () => null,
    };
  }

  /** @type {HTMLElement} */
  const el = pane;

  /** @param {number} width @returns {number} the applied (clamped) width */
  const applyWidth = (width) => {
    const px = clampWidth(width, min, max);
    el.style.flexBasis = `${px}px`;
    el.style.maxWidth = `${px}px`;
    return px;
  };

  const clearWidth = () => {
    el.style.flexBasis = '';
    el.style.maxWidth = '';
  };

  /** @param {number} width */
  const persist = (width) => {
    try {
      localStorage.setItem(storageKey, String(width));
    } catch {
      /* storage blocked — the split still holds for this page lifetime */
    }
  };

  const restoreWidth = () => {
    let stored = NaN;
    try {
      stored = parseInt(localStorage.getItem(storageKey) ?? '', 10);
    } catch {
      /* storage blocked — keep the stylesheet default */
    }
    return Number.isFinite(stored) ? applyWidth(stored) : null;
  };

  if (handle) {
    const grip = handle;

    grip.addEventListener('pointerdown', (e) => {
      if (!isEnabled()) return;
      e.preventDefault();
      try {
        grip.setPointerCapture(e.pointerId);
      } catch {
        /* capture unavailable — the drag still tracks while the pointer is over
           the handle, which is the degraded case, not a broken one */
      }
      grip.classList.add('dragging');
      const startX = e.clientX;
      const startWidth = el.getBoundingClientRect().width;
      let width = startWidth;

      const move = (/** @type {PointerEvent} */ ev) => {
        width = applyWidth(startWidth + (ev.clientX - startX));
      };
      const end = () => {
        grip.classList.remove('dragging');
        grip.removeEventListener('pointermove', move);
        grip.removeEventListener('pointerup', end);
        grip.removeEventListener('pointercancel', end);
        persist(width);
      };
      grip.addEventListener('pointermove', move);
      grip.addEventListener('pointerup', end);
      grip.addEventListener('pointercancel', end);
    });

    grip.addEventListener('keydown', (e) => {
      if (!isEnabled()) return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const current = el.getBoundingClientRect().width;
      persist(applyWidth(current + (e.key === 'ArrowRight' ? step : -step)));
    });
  }

  return { applyWidth, clearWidth, restoreWidth };
}
