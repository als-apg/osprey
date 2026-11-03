// @ts-check
/* OSPREY Design System — movable, resizable table columns.
 *
 * Two halves, like the splitter:
 *
 *   - a pure LAYOUT MODEL — which columns a table shows, in what order, at what
 *     width — with the operations a header menu offers (move, resize, hide,
 *     reset) and a storage round-trip that survives a table growing or losing
 *     a column between releases;
 *   - a DOM CONTROLLER that turns pointer gestures on a header row into those
 *     operations: drag a header to move it, drag its right edge to resize it,
 *     double-click the edge to fit it to its content.
 *
 * The controller never renders. It reads the table the caller drew, reports a
 * gesture through a callback, and the caller updates its layout and redraws.
 * That keeps one renderer per table and makes the controller safe to attach to
 * a region whose table is replaced on every search.
 *
 * Header contract (what the controller reads):
 *
 *   <table><colgroup><col data-col="device" style="width:150px">…</colgroup>
 *     <thead><tr>
 *       <th data-col="device">Device<span class="col-resize" data-resize="device"></span></th>
 *       <th data-col="chk" data-pinned>…</th>          ← pinned: neither moved nor a drop target
 *
 * Pinned columns (a selection checkbox, a trailing actions column) keep their
 * edge: `pin: 'start'` columns always lead, `pin: 'end'` columns always trail,
 * and the order the model stores is the order of the movable columns between.
 */

/**
 * One column as a table declares it.
 * @typedef {object} ColumnSpec
 * @property {string} key - stable identity; also the `data-col` value
 * @property {string} label - the header text, and the name the column menu shows
 * @property {number} width - default width, px
 * @property {number} [minWidth] - narrowest a resize may go, px (default 40)
 * @property {boolean} [hidden] - hidden by default
 * @property {'start'|'end'} [pin] - pinned to an edge; never moved, hidden or resized
 */

/**
 * The operator's arrangement of one table.
 * @typedef {object} ColumnLayout
 * @property {string[]} order - keys of the movable columns, in display order
 * @property {Record<string, number>} widths - key → width, px
 * @property {Set<string>} hidden - keys of hidden columns
 */

/**
 * What a layout stores as. Plain JSON, keyed by column key, so an added or
 * dropped column costs nothing on restore.
 * @typedef {{order: string[], widths: Record<string, number>, hidden: string[]}} StoredLayout
 */

export const DEFAULT_MIN_WIDTH = 40;
export const MAX_WIDTH = 800;

/** @param {ColumnSpec} spec */
const isPinned = (spec) => spec.pin === 'start' || spec.pin === 'end';

/**
 * The layout a table opens with when the operator has arranged nothing.
 * @param {ColumnSpec[]} specs
 * @returns {ColumnLayout}
 */
export function defaultLayout(specs) {
  return {
    order: specs.filter((s) => !isPinned(s)).map((s) => s.key),
    widths: Object.fromEntries(specs.map((s) => [s.key, s.width])),
    hidden: new Set(specs.filter((s) => s.hidden && !isPinned(s)).map((s) => s.key)),
  };
}

/**
 * Every column in display order: leading pins, the movable order, trailing pins.
 * @param {ColumnSpec[]} specs
 * @param {ColumnLayout} layout
 * @returns {ColumnSpec[]}
 */
export function orderedColumns(specs, layout) {
  const byKey = new Map(specs.map((s) => [s.key, s]));
  const middle = layout.order.map((k) => byKey.get(k)).filter((s) => s !== undefined);
  return [
    ...specs.filter((s) => s.pin === 'start'),
    .../** @type {ColumnSpec[]} */ (middle),
    ...specs.filter((s) => s.pin === 'end'),
  ];
}

/**
 * The columns a table draws, in order, each with its current width.
 * @param {ColumnSpec[]} specs
 * @param {ColumnLayout} layout
 * @returns {Array<ColumnSpec & {width: number}>}
 */
export function visibleColumns(specs, layout) {
  return orderedColumns(specs, layout)
    .filter((s) => !layout.hidden.has(s.key))
    .map((s) => ({ ...s, width: layout.widths[s.key] ?? s.width }));
}

/**
 * Move one column before or after another. Pinned columns are neither moved
 * nor used as targets; a move onto itself is a no-op.
 * @param {ColumnLayout} layout
 * @param {string} key - the column being moved
 * @param {string} targetKey - the column it is dropped against
 * @param {boolean} before - true to land before the target, false after
 * @returns {boolean} whether the order changed
 */
export function moveColumn(layout, key, targetKey, before) {
  const from = layout.order.indexOf(key);
  if (from === -1 || key === targetKey || !layout.order.includes(targetKey)) return false;
  const next = layout.order.filter((k) => k !== key);
  const at = next.indexOf(targetKey) + (before ? 0 : 1);
  next.splice(at, 0, key);
  if (next.every((k, i) => k === layout.order[i])) return false;
  layout.order = next;
  return true;
}

/**
 * Step one column one place left or right among the VISIBLE movable columns,
 * which is what a keyboard or menu "Move left" means to the person looking.
 * @param {ColumnLayout} layout
 * @param {string} key
 * @param {-1|1} delta
 * @returns {boolean} whether the order changed
 */
export function nudgeColumn(layout, key, delta) {
  const shown = layout.order.filter((k) => !layout.hidden.has(k));
  const i = shown.indexOf(key);
  const j = i + delta;
  if (i === -1 || j < 0 || j >= shown.length) return false;
  return moveColumn(layout, key, shown[j], delta < 0);
}

/**
 * Set one column's width, clamped to its minimum and the shared maximum.
 * @param {ColumnSpec[]} specs
 * @param {ColumnLayout} layout
 * @param {string} key
 * @param {number} width
 * @returns {number} the width applied
 */
export function resizeColumn(specs, layout, key, width) {
  const spec = specs.find((s) => s.key === key);
  const min = spec?.minWidth ?? DEFAULT_MIN_WIDTH;
  const px = Math.round(Math.min(MAX_WIDTH, Math.max(min, Number(width) || min)));
  layout.widths[key] = px;
  return px;
}

/**
 * Show or hide one column. The last visible movable column cannot be hidden —
 * a table of nothing but a checkbox is not a state anyone means to reach.
 * @param {ColumnSpec[]} specs
 * @param {ColumnLayout} layout
 * @param {string} key
 * @param {boolean} hidden
 * @returns {boolean} whether anything changed
 */
export function setColumnHidden(specs, layout, key, hidden) {
  const spec = specs.find((s) => s.key === key);
  if (!spec || isPinned(spec) || layout.hidden.has(key) === hidden) return false;
  if (hidden && layout.order.filter((k) => !layout.hidden.has(k)).length <= 1) return false;
  if (hidden) layout.hidden.add(key);
  else layout.hidden.delete(key);
  return true;
}

/**
 * Whether a layout still matches the table's defaults — what decides if
 * "Reset columns" has anything to do.
 * @param {ColumnSpec[]} specs
 * @param {ColumnLayout} layout
 * @returns {boolean}
 */
export function isDefaultLayout(specs, layout) {
  return (
    JSON.stringify(serializeLayout(layout)) ===
    JSON.stringify(serializeLayout(defaultLayout(specs)))
  );
}

/**
 * @param {ColumnLayout} layout
 * @returns {StoredLayout}
 */
export function serializeLayout(layout) {
  return {
    order: [...layout.order],
    widths: { ...layout.widths },
    hidden: [...layout.hidden].sort(),
  };
}

/**
 * Rebuild a layout from what storage held, against the columns the table has
 * today. Unknown keys are dropped, a column the stored order never named is
 * inserted where its default position puts it, and a width that is not a
 * finite number falls back to the default. Anything unreadable yields the
 * default layout rather than an error.
 * @param {ColumnSpec[]} specs
 * @param {unknown} stored - a StoredLayout, a JSON string of one, or anything else
 * @returns {ColumnLayout}
 */
export function restoreLayout(specs, stored) {
  const base = defaultLayout(specs);
  /** @type {any} */
  let raw = stored;
  if (typeof raw === 'string') {
    try {
      raw = JSON.parse(raw);
    } catch {
      return base;
    }
  }
  if (!raw || typeof raw !== 'object') return base;

  const movable = new Set(base.order);
  const order = Array.isArray(raw.order)
    ? raw.order.filter((/** @type {unknown} */ k) => typeof k === 'string' && movable.has(k))
    : [];
  const unique = [...new Set(/** @type {string[]} */ (order))];
  // A column added since the layout was stored goes in after its default
  // predecessor, so a new release's column appears where it was designed to.
  for (const [i, key] of base.order.entries()) {
    if (unique.includes(key)) continue;
    const prev = base.order
      .slice(0, i)
      .reverse()
      .find((k) => unique.includes(k));
    unique.splice(prev === undefined ? 0 : unique.indexOf(prev) + 1, 0, key);
  }

  const widths = { ...base.widths };
  if (raw.widths && typeof raw.widths === 'object') {
    for (const spec of specs) {
      const w = Number(raw.widths[spec.key]);
      if (!isPinned(spec) && Number.isFinite(w)) {
        widths[spec.key] = Math.min(
          MAX_WIDTH,
          Math.max(spec.minWidth ?? DEFAULT_MIN_WIDTH, Math.round(w))
        );
      }
    }
  }

  const hidden = Array.isArray(raw.hidden)
    ? new Set(
        raw.hidden.filter((/** @type {unknown} */ k) => typeof k === 'string' && movable.has(k))
      )
    : base.hidden;
  const layout = { order: unique, widths, hidden: /** @type {Set<string>} */ (hidden) };
  // Never restore into a table with every movable column hidden.
  if (layout.order.every((k) => layout.hidden.has(k))) layout.hidden = new Set();
  return layout;
}

/**
 * The width that fits a column's widest cell, header included. Reads
 * `scrollWidth`, which reports the full content width of a cell even while
 * `overflow: hidden` and `text-overflow: ellipsis` are clipping it.
 * @param {HTMLTableElement} table
 * @param {string} key
 * @returns {number|null} the fitted width, or null when the column is not drawn
 */
export function measureColumn(table, key) {
  const head = table.querySelector(`thead th[data-col="${CSS.escape(key)}"]`);
  if (!(head instanceof HTMLTableCellElement)) return null;
  // Position among the header's siblings rather than `cellIndex`: the header
  // row carries no colspans, and DOM shims disagree about `cellIndex`.
  const index = [...(head.parentElement?.children ?? [])].indexOf(head);
  let widest = head.scrollWidth;
  for (const row of table.querySelectorAll(':scope > tbody > tr')) {
    const cell = row.children[index];
    // A full-width message row (colspan) belongs to no column.
    if (cell instanceof HTMLElement && Number(cell.getAttribute('colspan') || 1) === 1) {
      widest = Math.max(widest, cell.scrollWidth);
    }
  }
  return widest + 2;
}

/**
 * @typedef {object} ColumnGestures
 * @property {(key: string, width: number) => void} onResizing - live, every frame of an edge drag
 * @property {(key: string, width: number) => void} onResize - once, when an edge drag ends
 * @property {(key: string, targetKey: string, before: boolean) => void} onMove - a header drop
 * @property {(key: string) => void} onAutosize - a double-click on an edge
 */

/** How far a press must travel before it is a drag rather than a click, px. */
const DRAG_THRESHOLD = 4;

/**
 * Wire header gestures on `region`, the stable element the table is drawn
 * into. Listeners are delegated, so redrawing the table inside the region
 * needs no re-wiring. A click that ended a drag is swallowed, so a caller's
 * click-to-sort on the header does not also fire.
 *
 * @param {HTMLElement} region
 * @param {ColumnGestures} gestures
 * @returns {() => void} detach
 */
export function attachColumnGestures(region, gestures) {
  const controller = new AbortController();
  const { signal } = controller;
  const doc = region.ownerDocument;

  /** @type {null | {kind: 'resize', key: string, x0: number, w0: number}
   *   | {kind: 'press' | 'move', key: string, x0: number, th: HTMLElement,
   *      target: string | null, before: boolean, ghost: HTMLElement | null, line: HTMLElement | null}} */
  let drag = null;
  let swallowClick = false;

  const movableHeads = () =>
    /** @type {HTMLElement[]} */ ([
      ...region.querySelectorAll('thead th[data-col]:not([data-pinned])'),
    ]);

  const endDrag = () => {
    if (drag && drag.kind !== 'resize') {
      drag.th.classList.remove('col-drag-source');
      drag.ghost?.remove();
      drag.line?.remove();
      doc.body.classList.remove('col-dragging');
    }
    doc.body.classList.remove('col-resizing');
    drag = null;
  };

  region.addEventListener(
    'pointerdown',
    (e) => {
      if (e.button !== 0 || !(e.target instanceof Element)) return;
      const handle = e.target.closest('.col-resize[data-resize]');
      if (handle instanceof HTMLElement) {
        const key = handle.dataset.resize ?? '';
        const th = handle.closest('th');
        drag = {
          kind: 'resize',
          key,
          x0: e.clientX,
          w0: th ? th.getBoundingClientRect().width : 0,
        };
        handle.setPointerCapture?.(e.pointerId);
        doc.body.classList.add('col-resizing');
        e.preventDefault();
        return;
      }
      const th = e.target.closest('thead th[data-col]:not([data-pinned])');
      if (th instanceof HTMLElement && !e.target.closest('input, button, a')) {
        // No text selection from a header press: the press may become a drag,
        // and the click that sorts still fires without the default.
        e.preventDefault();
        drag = {
          kind: 'press',
          key: th.dataset.col ?? '',
          x0: e.clientX,
          th,
          target: null,
          before: true,
          ghost: null,
          line: null,
        };
      }
    },
    { signal }
  );

  region.addEventListener(
    'pointermove',
    (e) => {
      if (!drag) return;
      if (drag.kind === 'resize') {
        gestures.onResizing(drag.key, drag.w0 + (e.clientX - drag.x0));
        return;
      }
      if (drag.kind === 'press') {
        if (Math.abs(e.clientX - drag.x0) < DRAG_THRESHOLD) return;
        drag.kind = 'move';
        // A press on header text may already have begun a text selection that
        // would otherwise sweep across the rows for the rest of the drag.
        doc.getSelection?.()?.removeAllRanges();
        drag.th.setPointerCapture?.(e.pointerId);
        drag.th.classList.add('col-drag-source');
        doc.body.classList.add('col-dragging');
        drag.ghost = doc.createElement('div');
        drag.ghost.className = 'col-drag-ghost';
        drag.ghost.textContent = drag.th.textContent?.trim() ?? '';
        doc.body.appendChild(drag.ghost);
        drag.line = doc.createElement('div');
        drag.line.className = 'col-drop-line';
        doc.body.appendChild(drag.line);
      }
      if (drag.ghost) {
        drag.ghost.style.left = `${e.clientX + 12}px`;
        drag.ghost.style.top = `${e.clientY + 12}px`;
      }
      // The drop slot is the gap nearest the cursor: before the first header
      // whose midpoint lies right of it, else after the last.
      const heads = movableHeads();
      let target = null;
      let before = true;
      for (const head of heads) {
        const r = head.getBoundingClientRect();
        target = head;
        if (e.clientX < r.left + r.width / 2) {
          before = true;
          break;
        }
        before = false;
      }
      drag.target = target?.dataset.col ?? null;
      drag.before = before;
      if (target && drag.line) {
        const r = target.getBoundingClientRect();
        const wrap = target.closest('table')?.parentElement?.getBoundingClientRect() ?? r;
        drag.line.style.left = `${(before ? r.left : r.right) - 1}px`;
        drag.line.style.top = `${r.top}px`;
        drag.line.style.height = `${Math.max(r.height, wrap.bottom - r.top)}px`;
      }
    },
    { signal }
  );

  const finish = (/** @type {PointerEvent} */ e) => {
    if (!drag) return;
    if (drag.kind === 'resize') {
      gestures.onResize(drag.key, drag.w0 + (e.clientX - drag.x0));
      swallowClick = true;
    } else if (drag.kind === 'move') {
      swallowClick = true;
      if (drag.target && drag.target !== drag.key)
        gestures.onMove(drag.key, drag.target, drag.before);
    }
    endDrag();
  };
  region.addEventListener('pointerup', finish, { signal });
  region.addEventListener('pointercancel', () => endDrag(), { signal });

  // Capture phase: runs before the caller's own click handler on the header.
  region.addEventListener(
    'click',
    (e) => {
      if (!swallowClick) return;
      swallowClick = false;
      e.stopPropagation();
      e.preventDefault();
    },
    { signal, capture: true }
  );
  // A gesture that ended without a click (released outside) must not leave
  // the flag armed for the next, unrelated click.
  region.addEventListener(
    'pointerdown',
    () => {
      swallowClick = false;
    },
    { signal, capture: true }
  );

  region.addEventListener(
    'dblclick',
    (e) => {
      if (!(e.target instanceof Element)) return;
      const handle = e.target.closest('.col-resize[data-resize]');
      if (handle instanceof HTMLElement) {
        e.preventDefault();
        gestures.onAutosize(handle.dataset.resize ?? '');
      }
    },
    { signal }
  );

  return () => {
    endDrag();
    controller.abort();
  };
}
