// @ts-check
/*
 * OSPREY Design System — shared pane splitter.
 *
 * One draggable divider behind every resizable pane in the product: the OKF
 * knowledge panel's sidebar/reader split, the artifact gallery's browse view,
 * the PLAN panel's plan-list/detail split, the LATTICE control sidebar, the
 * settings drawer, and the event dashboard's three layout zones. The visual
 * half lives in css/base.css as `.osprey-splitter`; this module is the
 * behaviour half.
 *
 * Why shared: these surfaces are served by three different processes — the hub
 * serves OKF and artifacts in-process, the PLAN panel comes from the
 * bluesky-web sidecar container, and the event dashboard is its own FastMCP
 * server. They cannot import each other's code, and `/design-system/` is the
 * one path all of them resolve. Hand-copied splitters drifted silently because
 * nothing failed when one stopped matching; this is the single definition.
 *
 * ---- The three dimensions ----
 *
 * A divider varies along three independent axes, and collapsing any two of
 * them is what forced the previous hand-rolled copies to exist:
 *
 *   axis    'x' | 'y'          which pointer coordinate drives the drag
 *   anchor  'start' | 'end'    whether the pane precedes or follows the handle
 *   sizing  'flex' | 'box'     which CSS property carries the size
 *
 * `sizing` is the one that is easy to miss. Writing `flex-basis` is right for a
 * flex child, but the settings drawer is `position: fixed` — a flex-basis on it
 * is inert. And on a column flex item an inline `flex-basis` outranks even
 * `height: …px !important`, so a flex-sized event zone would silently defeat
 * its own `.collapsed` rule. `sizing: 'box'` writes width/height directly for
 * those hosts.
 *
 * ---- Who owns the persisted state ----
 *
 * Most hosts let this module persist `{size, collapsed}` under a storage key of
 * their choosing. A host that already owns a layout record passes
 * `storageKey: null` plus `initialSize`/`initialCollapsed` and an `onCommit`
 * callback: the splitter then owns geometry only and never touches
 * localStorage. The event dashboard needs this — it keeps six layout fields and
 * a preset system in one record, and a second source of truth would let a stale
 * splitter value win over a freshly applied preset.
 *
 * ---- Why a drag never animates ----
 *
 * For the duration of a pointer drag the pane carries `data-splitter-dragging`,
 * which base.css uses to suppress transitions. A transition on the property a
 * drag is driving makes the pane trail the cursor by the transition duration —
 * the browser eases toward a target that has already moved. Discrete collapses
 * (double-click, a host's chevron) are *not* drags and keep their animation.
 */

/**
 * Clamp a candidate pane size to the usable range.
 *
 * @param {number} size
 * @param {number} min
 * @param {number} max
 * @returns {number} the clamped size, rounded to whole pixels
 */
export function clampSize(size, min, max) {
  return Math.min(max, Math.max(min, Math.round(size)));
}

/**
 * Resolve a bound that may be a live function. The drawer's ceiling is 90vw,
 * which changes when the window resizes; a number captured at construction
 * would go stale and let the drawer be dragged past its limit.
 *
 * @param {number | (() => number)} bound
 * @returns {number}
 */
function resolveBound(bound) {
  return typeof bound === 'function' ? bound() : bound;
}

/**
 * Parse whatever is in storage into the current record shape.
 *
 * Three legacy shapes predate `{size, collapsed}` and all three are still in
 * real browsers, so all three are read rather than discarded:
 *
 * - a bare integer — every splitter host before collapse existed;
 * - the string "true"/"false" — LATTICE's sidebar-collapsed flag;
 * - absent — which for LATTICE means *collapsed*, not "no preference", because
 *   its old read was `getItem(key) !== 'false'`. Hosts declare that with
 *   `defaultCollapsed`.
 *
 * @param {string|null} raw
 * @param {boolean} defaultCollapsed
 * @returns {{size: number|null, collapsed: boolean}}
 */
export function parseStoredState(raw, defaultCollapsed) {
  if (raw === null || raw === '') return { size: null, collapsed: defaultCollapsed };
  if (raw === 'true') return { size: null, collapsed: true };
  if (raw === 'false') return { size: null, collapsed: false };

  const asInt = parseInt(raw, 10);
  if (String(asInt) === raw.trim()) return { size: asInt, collapsed: false };

  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') {
      const size = Number.isFinite(parsed.size) ? parsed.size : null;
      return { size, collapsed: Boolean(parsed.collapsed) };
    }
  } catch {
    /* unparseable — fall through to the default */
  }
  return { size: null, collapsed: defaultCollapsed };
}

/**
 * @typedef {object} SplitterApi
 * @property {(size: number) => number} applySize Size the pane; returns the applied (clamped) size.
 * @property {() => void} clearSize Drop the inline sizes, handing the pane back to the stylesheet.
 * @property {() => number|null} restoreSize Re-apply the persisted state; null when nothing usable is stored.
 * @property {() => void} collapse Shrink to `collapsedSize`, remembering the current size.
 * @property {() => void} expand Restore the size held before the last collapse.
 * @property {() => void} toggleCollapse Collapse if expanded, expand if collapsed.
 * @property {() => boolean} isCollapsed Current collapsed state.
 * @property {() => void} destroy Detach everything without persisting.
 */

/**
 * Wire a pane splitter. Degrades in two steps rather than all-or-nothing, so
 * callers need no null dance of their own:
 *
 * - no pane (a stripped-down embed that ships one pane): everything no-ops,
 *   the returned API is safe to call;
 * - pane but no handle: the sizing API still works — a caller that restores a
 *   persisted size on some other trigger keeps doing so — only the drag and
 *   keyboard wiring are skipped, because there is nothing to wire them to.
 *
 * @param {object} opts
 * @param {HTMLElement|null} opts.handle The separator element.
 * @param {HTMLElement|null} opts.pane The pane being resized.
 * @param {string|null} [opts.storageKey] localStorage key, or null when the host persists.
 * @param {number | (() => number)} opts.min Smallest useful pane size, px.
 * @param {number | (() => number)} opts.max Largest useful pane size, px.
 * @param {'x'|'y'} [opts.axis] Which coordinate drives the drag. Defaults to 'x'.
 * @param {'start'|'end'} [opts.anchor] Whether the pane precedes ('start') or follows ('end') the handle.
 * @param {'flex'|'box'} [opts.sizing] Which property carries the size. Defaults to 'flex'.
 * @param {number} [opts.collapsedSize] Size the pane occupies when collapsed. Defaults to 0.
 * @param {boolean} [opts.defaultCollapsed] Collapsed state when nothing is stored.
 * @param {number|null} [opts.initialSize] Seed size for a host-persisted splitter.
 * @param {boolean} [opts.initialCollapsed] Seed collapsed state for a host-persisted splitter.
 * @param {((size: number, collapsed: boolean) => void)|null} [opts.onCommit] Called after every
 *   settled change (drag end, keyboard nudge, collapse toggle). The host's hook for persisting
 *   its own record and for side effects a resize implies, such as a plot relayout.
 * @param {number} [opts.step] Arrow-key increment, px. Defaults to 16.
 * @param {() => boolean} [opts.isEnabled] Gate for drag and keyboard, re-read per interaction.
 *   Lets a caller park a splitter while its layout is on a different axis.
 * @returns {SplitterApi}
 */
export function initSplitter({
  handle,
  pane,
  storageKey = null,
  min,
  max,
  axis = 'x',
  anchor = 'start',
  sizing = 'flex',
  collapsedSize = 0,
  defaultCollapsed = false,
  initialSize = null,
  initialCollapsed = false,
  onCommit = null,
  step = 16,
  isEnabled = () => true,
}) {
  if (!pane) {
    return {
      applySize: (size) => clampSize(size, resolveBound(min), resolveBound(max)),
      clearSize: () => {},
      restoreSize: () => null,
      collapse: () => {},
      expand: () => {},
      toggleCollapse: () => {},
      isCollapsed: () => false,
      destroy: () => {},
    };
  }

  /** @type {HTMLElement} */
  const el = pane;
  const horizontal = axis === 'x';
  /** The CSS box property `sizing: 'box'` writes. */
  const boxProp = horizontal ? 'width' : 'height';
  /** The cap property, which both sizing modes must keep out of the way. */
  const maxProp = horizontal ? 'maxWidth' : 'maxHeight';

  /**
   * Last size this module applied. The single source of truth for the drag
   * origin, the keyboard origin, and what `expand()` restores to.
   *
   * Deliberately not `getBoundingClientRect()`: under happy-dom (the unit
   * suite's environment) a rect reads 0, which would make every collapse
   * round-trip land on `min`. The rect is consulted exactly once, to seed a
   * first drag on a pane this module has not sized yet.
   */
  let appliedSize = /** @type {number|null} */ (initialSize);
  /** Size to return to on expand — never `collapsedSize`. */
  let expandedSize = /** @type {number|null} */ (initialSize);
  let collapsed = Boolean(initialCollapsed);

  const controller = new AbortController();
  const { signal } = controller;
  let frame = 0;
  let pendingSize = 0;

  /** Write geometry. `raw` bypasses the min clamp, which a collapse needs. */
  const write = (/** @type {number} */ px) => {
    if (sizing === 'box') {
      el.style[boxProp] = `${px}px`;
      el.style[maxProp] = 'none';
    } else {
      el.style.flexBasis = `${px}px`;
      el.style[maxProp] = `${px}px`;
    }
    appliedSize = px;
    if (handle) {
      handle.setAttribute('aria-valuenow', String(px));
      handle.setAttribute('aria-valuemin', String(resolveBound(min)));
      handle.setAttribute('aria-valuemax', String(resolveBound(max)));
    }
  };

  /** @param {number} size @returns {number} the applied (clamped) size */
  const applySize = (size) => {
    const px = clampSize(size, resolveBound(min), resolveBound(max));
    write(px);
    expandedSize = px;
    collapsed = false;
    return px;
  };

  const clearSize = () => {
    el.style.flexBasis = '';
    el.style[boxProp] = '';
    el.style[maxProp] = '';
    appliedSize = null;
  };

  /** The size a drag or nudge starts from, measuring only as a last resort. */
  const currentSize = () => {
    if (appliedSize !== null) return appliedSize;
    const rect = el.getBoundingClientRect();
    return horizontal ? rect.width : rect.height;
  };

  const commit = () => {
    const size = expandedSize ?? currentSize();
    if (storageKey !== null) {
      try {
        localStorage.setItem(storageKey, JSON.stringify({ size, collapsed }));
      } catch {
        /* storage blocked — the split still holds for this page lifetime */
      }
    }
    if (onCommit) onCommit(size, collapsed);
  };

  const collapse = () => {
    if (!collapsed) expandedSize = currentSize();
    collapsed = true;
    write(collapsedSize);
    commit();
  };

  const expand = () => {
    collapsed = false;
    // Clamp into `expandedSize`, not just into the write: a remembered size
    // from outside the current bounds (a narrower viewport, a host that
    // tightened its range) would otherwise be persisted as one value while a
    // different one is on screen.
    expandedSize = clampSize(expandedSize ?? currentSize(), resolveBound(min), resolveBound(max));
    write(expandedSize);
    commit();
  };

  const toggleCollapse = () => (collapsed ? expand() : collapse());

  const restoreSize = () => {
    let stored = /** @type {{size: number|null, collapsed: boolean}} */ (
      { size: initialSize, collapsed: Boolean(initialCollapsed) }
    );
    if (storageKey !== null) {
      let raw = null;
      try {
        raw = localStorage.getItem(storageKey);
      } catch {
        /* storage blocked — keep the stylesheet default */
      }
      stored = parseStoredState(raw, defaultCollapsed);
    }
    if (stored.size !== null) expandedSize = stored.size;
    if (stored.collapsed) {
      collapsed = true;
      write(collapsedSize);
      return collapsedSize;
    }
    return stored.size === null ? null : applySize(stored.size);
  };

  if (handle) {
    const grip = handle;
    grip.setAttribute('role', 'separator');
    grip.setAttribute('aria-orientation', horizontal ? 'vertical' : 'horizontal');

    /** A drag toward the pane grows it when the pane trails the handle. */
    const sign = anchor === 'end' ? -1 : 1;

    const cancelFrame = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    };

    grip.addEventListener(
      'pointerdown',
      (e) => {
        if (!isEnabled()) return;
        e.preventDefault();
        try {
          grip.setPointerCapture(e.pointerId);
        } catch {
          /* capture unavailable — the drag still tracks while the pointer is
             over the handle, which is the degraded case, not a broken one */
        }
        grip.classList.add('dragging');
        el.setAttribute('data-splitter-dragging', '');

        const start = horizontal ? e.clientX : e.clientY;
        const startSize = currentSize();
        pendingSize = startSize;
        // A press that never moves is a click, not a drag. Without this a
        // double-click — which is two full pointerdown/up pairs *before* the
        // dblclick event — would commit two zero-distance drags first, and
        // `applySize` would clear the collapsed flag and overwrite the
        // remembered expanded size. The collapse gesture would then never
        // restore, because there would be nothing left to restore to.
        let moved = false;

        const move = (/** @type {PointerEvent} */ ev) => {
          const now = horizontal ? ev.clientX : ev.clientY;
          if (now !== start) moved = true;
          pendingSize = startSize + sign * (now - start);
          // Coalesce to one geometry write per displayed frame. Pointer events
          // outrun the refresh rate, and every uncoalesced write forces another
          // synchronous layout of a subtree only the last one will show.
          if (!frame) {
            frame = requestAnimationFrame(() => {
              frame = 0;
              applySize(pendingSize);
            });
          }
        };

        const end = () => {
          grip.removeEventListener('pointermove', move);
          grip.removeEventListener('pointerup', end);
          grip.removeEventListener('pointercancel', end);
          // Order matters. Land the final size while transitions are still
          // suppressed, or the pane animates its last few pixels — the exact
          // trailing-behind-the-cursor artefact this module exists to avoid.
          cancelFrame();
          if (moved) applySize(pendingSize);
          el.removeAttribute('data-splitter-dragging');
          grip.classList.remove('dragging');
          if (moved) commit();
        };

        grip.addEventListener('pointermove', move, { signal });
        grip.addEventListener('pointerup', end, { signal });
        grip.addEventListener('pointercancel', end, { signal });
      },
      { signal }
    );

    grip.addEventListener(
      'dblclick',
      () => {
        if (!isEnabled()) return;
        toggleCollapse();
      },
      { signal }
    );

    grip.addEventListener(
      'keydown',
      (e) => {
        if (!isEnabled()) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleCollapse();
          return;
        }
        const grow = horizontal ? 'ArrowRight' : 'ArrowDown';
        const shrink = horizontal ? 'ArrowLeft' : 'ArrowUp';
        if (e.key !== grow && e.key !== shrink) return;
        e.preventDefault();
        applySize(currentSize() + sign * (e.key === grow ? step : -step));
        commit();
      },
      { signal }
    );
  }

  const destroy = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    el.removeAttribute('data-splitter-dragging');
    if (handle) handle.classList.remove('dragging');
    // Deliberately no commit(): an interrupted drag was never a deliberate
    // choice of size, and persisting it would corrupt the next restore.
    controller.abort();
  };

  return {
    applySize,
    clearSize,
    restoreSize,
    collapse,
    expand,
    toggleCollapse,
    isCollapsed: () => collapsed,
    destroy,
  };
}
