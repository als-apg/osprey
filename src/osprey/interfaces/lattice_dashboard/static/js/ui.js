// @ts-check
/* OSPREY Lattice Dashboard — UI Chrome Layer
 *
 * The control sidebar (the shared panel sidebar), the stacked/grid layout,
 * the sidebar's Magnets | Settings switch, and figure-panel drag-and-drop
 * rearrangement — the dashboard's persistent UI chrome. Each preference is
 * held in one of the four `lattice-*` localStorage keys below, resolved per
 * person through scopedStorageKey().
 *
 * The full drag-and-drop feature (`setupDragAndDrop`, `savePanelOrder`,
 * `restorePanelOrder`) lives together in one Drag-and-Drop section below.
 *
 * No dependency on net.js/render.js — Plotly reflows after a
 * layout/sidebar change are the only side effect threaded in, via the
 * figure catalog passed to createUI().
 */

import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

const SIDEBAR_TAB_KEY = 'lattice-sidebar-tab';
const PANEL_ORDER_KEY = 'lattice-panel-order';
const LAYOUT_KEY = 'lattice-layout-mode';
// The key's name predates the panel sidebar; it keeps it, so an operator's
// stored width and open state carry over.
const SIDEBAR_KEY = 'lattice-sidebar-collapsed';

/** How long the figure pane must hold its width before the figures reflow. */
const REFLOW_DELAY_MS = 150;

/**
 * Create the dashboard's sidebar/layout/tabs/drag-and-drop controller,
 * bound to the figure catalog it needs for post-toggle Plotly reflows.
 * @param {string[]} figureNames - the full figure catalog (fast + verification)
 */
export function createUI(figureNames) {
  /** Plotly figures need to reflow after a layout/sidebar-width change. */
  function _reflowFigures() {
    figureNames.forEach((name) => {
      const plotEl = /** @type {any} */ (document.getElementById(`plot-${name}`));
      if (plotEl && plotEl.data) {
        Plotly.relayout(plotEl, { autosize: true });
      }
    });
  }

  // ── Sidebar: the shared panel sidebar ───────────────────

  /** @type {import('/design-system/js/panel-sidebar.js').PanelSidebarApi | null} */
  let sidebarApi = null;

  /**
   * Wire the control sidebar: open on wide tiles (the sliders are the main
   * control), an overlay on narrow ones, toggled from the toolbar. Plotly
   * does not re-measure when its pane changes width without a window
   * resize, so the figures reflow once the pane settles.
   * @returns {import('/design-system/js/panel-sidebar.js').PanelSidebarApi | null}
   */
  function initSidebar() {
    if (sidebarApi) return sidebarApi;
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return null;
    sidebarApi = initPanelSidebar({
      sidebar,
      handle: document.getElementById('sidebar-resizer'),
      toggle: document.getElementById('btn-sidebar'),
      storageKey: SIDEBAR_KEY,
      defaultOpen: true,
    });
    const pane = document.getElementById('figure-pane');
    if (pane && typeof ResizeObserver === 'function') {
      /** @type {ReturnType<typeof setTimeout> | undefined} */
      let timer;
      new ResizeObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(_reflowFigures, REFLOW_DELAY_MS);
      }).observe(pane);
    }
    return sidebarApi;
  }

  // ── Layout Mode ─────────────────────────────────────────

  /** The layout the figures show: `stacked` or `grid`. */
  function currentLayout() {
    return document.getElementById('figure-area')?.classList.contains('layout-stacked')
      ? 'stacked'
      : 'grid';
  }

  /** @returns {string} the stored layout, now applied */
  function initLayout() {
    const mode = localStorage.getItem(scopedStorageKey(LAYOUT_KEY)) || 'stacked';
    applyLayout(mode);
    return currentLayout();
  }

  /** @param {string} mode */
  function applyLayout(mode) {
    const figArea = document.getElementById('figure-area');
    if (!figArea) return;
    figArea.classList.toggle('layout-stacked', mode === 'stacked');
    localStorage.setItem(scopedStorageKey(LAYOUT_KEY), currentLayout());
    _reflowFigures();
  }

  // ── Sidebar Tabs ────────────────────────────────────────

  /** Bind the Magnets | Settings switch and restore the last one shown. */
  function initSidebarTabs() {
    document.querySelectorAll('.sidebar-tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        switchTab(/** @type {string} */ (/** @type {HTMLElement} */ (tab).dataset.tab));
      });
    });

    const savedTab = localStorage.getItem(scopedStorageKey(SIDEBAR_TAB_KEY));
    if (savedTab) switchTab(savedTab);
  }

  /** @param {string} tabName */
  function switchTab(tabName) {
    document.querySelectorAll('.sidebar-tab').forEach((t) => {
      const shown = /** @type {HTMLElement} */ (t).dataset.tab === tabName;
      t.setAttribute('aria-pressed', String(shown));
    });
    document.querySelectorAll('.sidebar-tab-content').forEach((panel) => {
      panel.classList.toggle('sidebar-tab-content--active', panel.id === `tab-${tabName}`);
    });
    localStorage.setItem(scopedStorageKey(SIDEBAR_TAB_KEY), tabName);
  }

  // ── Drag-and-Drop Panel Rearrangement (unified) ────────

  /**
   * @param {Element} cellA
   * @param {Element} cellB
   */
  function swapCells(cellA, cellB) {
    const parentA = /** @type {Node} */ (cellA.parentNode);
    const parentB = /** @type {Node} */ (cellB.parentNode);
    const nextA = cellA.nextSibling;
    // Insert A before B, then B into A's old position
    parentB.insertBefore(cellA, cellB);
    if (nextA) {
      parentA.insertBefore(cellB, nextA);
    } else {
      parentA.appendChild(cellB);
    }
  }

  function savePanelOrder() {
    const cells = document.querySelectorAll('.figure-cell');
    const order = Array.from(cells).map((c) => /** @type {HTMLElement} */ (c).dataset.figure);
    localStorage.setItem(scopedStorageKey(PANEL_ORDER_KEY), JSON.stringify(order));
  }

  function setupDragAndDrop() {
    const cells = document.querySelectorAll('.figure-cell');
    /** @type {Element|null} */
    let draggedCell = null;

    cells.forEach((cell) => {
      const header = cell.querySelector('.figure-header');
      if (!header) return;

      // Only the header bar is the drag handle
      header.setAttribute('draggable', 'true');

      header.addEventListener('dragstart', (e) => {
        draggedCell = cell;
        cell.classList.add('dragging');
        document.body.classList.add('drag-active');
        const dragEvent = /** @type {DragEvent} */ (e);
        /** @type {DataTransfer} */ (dragEvent.dataTransfer).effectAllowed = 'move';
        /** @type {DataTransfer} */ (dragEvent.dataTransfer).setData('text/plain', cell.id);
      });

      // Drop target is the whole cell
      cell.addEventListener('dragover', (e) => {
        e.preventDefault();
        /** @type {DataTransfer} */ (/** @type {DragEvent} */ (e).dataTransfer).dropEffect = 'move';
        if (cell !== draggedCell) {
          cell.classList.add('drag-over');
        }
      });

      cell.addEventListener('dragleave', (e) => {
        // Only remove highlight when actually leaving the cell,
        // not when entering a child element
        const relatedTarget = /** @type {Node|null} */ (/** @type {DragEvent} */ (e).relatedTarget);
        if (!cell.contains(relatedTarget)) {
          cell.classList.remove('drag-over');
        }
      });

      cell.addEventListener('drop', (e) => {
        e.preventDefault();
        cell.classList.remove('drag-over');
        if (draggedCell && draggedCell !== cell) {
          swapCells(draggedCell, cell);
          savePanelOrder();

          // Plotly needs resize after DOM reparenting
          const plotA = /** @type {any} */ (draggedCell.querySelector('.figure-plot'));
          const plotB = /** @type {any} */ (cell.querySelector('.figure-plot'));
          if (plotA) Plotly.relayout(plotA, { autosize: true });
          if (plotB) Plotly.relayout(plotB, { autosize: true });
        }
      });

      header.addEventListener('dragend', () => {
        document.body.classList.remove('drag-active');
        if (draggedCell) {
          draggedCell.classList.remove('dragging');
          draggedCell = null;
        }
        // Clean up any stale drag-over highlights
        cells.forEach((c) => c.classList.remove('drag-over'));
      });
    });
  }

  function restorePanelOrder() {
    const saved = localStorage.getItem(scopedStorageKey(PANEL_ORDER_KEY));
    if (!saved) return;

    try {
      /** @type {string[]} */
      const order = JSON.parse(saved);

      // Collect all containers that hold figure cells (grid + verification row)
      const containers = document.querySelectorAll('.figure-grid, .verification-row');
      /** @type {{container: Element, placeholder: Element}[]} */
      const allSlots = [];
      containers.forEach((container) => {
        Array.from(container.children).forEach((child) => {
          if (child.classList.contains('figure-cell')) {
            allSlots.push({ container, placeholder: child });
          }
        });
      });

      // Build lookup of cells by figure name
      /** @type {Record<string, Element>} */
      const cellMap = {};
      allSlots.forEach((slot) => {
        const figureName = /** @type {HTMLElement} */ (slot.placeholder).dataset.figure;
        if (figureName) cellMap[figureName] = slot.placeholder;
      });

      // Validate saved order matches current DOM cells — if panel names
      // have changed (e.g. fma → lma), discard stale order
      const currentNames = new Set(Object.keys(cellMap));
      const savedNames = new Set(order);
      if (order.length !== allSlots.length || ![...currentNames].every((n) => savedNames.has(n))) {
        localStorage.removeItem(scopedStorageKey(PANEL_ORDER_KEY));
        return;
      }

      // Reorder: place cells into slots according to saved order
      const slotPositions = allSlots.map((s) => ({
        container: s.container,
        nextSibling: s.placeholder.nextSibling,
      }));

      // Detach all cells
      allSlots.forEach((s) => s.placeholder.remove());

      // Re-insert in saved order
      order.forEach((figureName, i) => {
        const cell = cellMap[figureName];
        if (!cell || i >= slotPositions.length) return;
        const pos = slotPositions[i];
        if (pos.nextSibling) {
          pos.container.insertBefore(cell, pos.nextSibling);
        } else {
          pos.container.appendChild(cell);
        }
      });
    } catch (e) {
      console.warn('Failed to restore panel order:', e);
      localStorage.removeItem(scopedStorageKey(PANEL_ORDER_KEY));
    }
  }

  return {
    initSidebar,
    initLayout,
    applyLayout,
    currentLayout,
    initSidebarTabs,
    setupDragAndDrop,
    restorePanelOrder,
  };
}
