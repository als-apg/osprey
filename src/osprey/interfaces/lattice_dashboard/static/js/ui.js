// @ts-check
/* OSPREY Lattice Dashboard — UI Chrome Layer
 *
 * Sidebar collapse, stacked/grid layout toggle, sidebar tab switching, and
 * figure-panel drag-and-drop rearrangement — the dashboard's persistent UI
 * chrome. Each preference is held in one of the four `lattice-*`
 * localStorage keys below.
 *
 * The full drag-and-drop feature (`setupDragAndDrop`, `savePanelOrder`,
 * `restorePanelOrder`) lives together in one Drag-and-Drop section below.
 *
 * No dependency on net.js/render.js — Plotly reflows after a
 * layout/sidebar change are the only side effect threaded in, via the
 * figure catalog passed to createUI().
 */

import { initSplitter } from '/design-system/js/splitter.js';

const SIDEBAR_TAB_KEY = 'lattice-sidebar-tab';
const PANEL_ORDER_KEY = 'lattice-panel-order';
const LAYOUT_KEY = 'lattice-layout-mode';
const SIDEBAR_KEY = 'lattice-sidebar-collapsed';

/**
 * Look up a required descendant and assert it as HTMLElement (for `.style`/
 * `.textContent` access). Purely a type-checking aid — if `selector` isn't
 * found this throws at the access site exactly as the un-cast lookup would.
 * @param {Element} scope
 * @param {string} selector
 * @returns {HTMLElement}
 */
function _requireEl(scope, selector) {
  return /** @type {HTMLElement} */ (scope.querySelector(selector));
}

/**
 * Create the dashboard's sidebar/layout/tabs/drag-and-drop controller,
 * bound to the figure catalog it needs for post-toggle Plotly reflows.
 * @param {string[]} figureNames - the full figure catalog (fast + verification)
 */
export function createUI(figureNames) {
  /** Plotly figures need to reflow after a layout/sidebar-width change. */
  function _reflowFigures() {
    figureNames.forEach(name => {
      const plotEl = /** @type {any} */ (document.getElementById(`plot-${name}`));
      if (plotEl && plotEl.data) {
        Plotly.relayout(plotEl, { autosize: true });
      }
    });
  }

  // ── Sidebar: collapse + resize ──────────────────────────

  /**
   * The shared splitter, which owns both the sidebar's width and its collapsed
   * state. Three call sites used to write `.sidebar-collapsed` and the storage
   * key independently — the chevron, the boot path, and the tab strip's
   * expand-on-click — and nothing kept them agreeing. They all route here now.
   */
  /** @type {import('/design-system/js/splitter.js').SplitterApi|null} */
  let sidebarSplitter = null;

  /**
   * Idempotent, and safe to call from any entry point. `createUI()` hands back
   * independent functions that a caller may invoke in any order, so the
   * chevron and the tab strip both ensure the splitter exists rather than
   * assuming boot already made it — otherwise whichever runs first silently
   * no-ops.
   */
  function initSidebar() {
    if (sidebarSplitter) return;
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    sidebarSplitter = initSplitter({
      handle: document.getElementById('sidebar-resizer'),
      pane: sidebar,
      storageKey: SIDEBAR_KEY,
      axis: 'x',
      anchor: 'start',
      // #sidebar is a plain flex child sized by `width`, and its
      // `.sidebar-collapsed` rule is a non-important `width: 36px` — an inline
      // flex-basis would outrank it and the collapse would stop working.
      sizing: 'box',
      min: 160,
      max: 480,
      collapsedSize: 36,
      // The pre-splitter read was `getItem(key) !== 'false'`, so an absent key
      // means COLLAPSED here — the inverse of every other host. Without this,
      // every operator who never touched the chevron silently gets an expanded
      // sidebar on first load after the upgrade.
      defaultCollapsed: true,
      // Plotly does not re-measure on its own; a width change that skips this
      // leaves every figure rendered at the old width.
      onCommit: () => {
        sidebar.classList.toggle('sidebar-collapsed', !!sidebarSplitter?.isCollapsed());
        setTimeout(_reflowFigures, 250);
      },
    });
    sidebarSplitter.restoreSize();
    sidebar.classList.toggle('sidebar-collapsed', sidebarSplitter.isCollapsed());
  }

  function toggleSidebar() {
    initSidebar();
    sidebarSplitter?.toggleCollapse();
  }

  /** Expand the sidebar if it is collapsed; a no-op otherwise. */
  function expandSidebar() {
    initSidebar();
    if (sidebarSplitter?.isCollapsed()) sidebarSplitter.expand();
  }

  // ── Layout Mode ─────────────────────────────────────────

  function initLayout() {
    const mode = localStorage.getItem(LAYOUT_KEY) || 'stacked';
    applyLayout(mode);
  }

  function toggleLayout() {
    const figArea = /** @type {HTMLElement} */ (document.getElementById('figure-area'));
    const isStacked = figArea.classList.contains('layout-stacked');
    applyLayout(isStacked ? 'grid' : 'stacked');
  }

  /** @param {string} mode */
  function applyLayout(mode) {
    const figArea = document.getElementById('figure-area');
    if (!figArea) return;
    const btn = document.getElementById('btn-layout');

    if (mode === 'stacked') {
      figArea.classList.add('layout-stacked');
      if (btn) {
        _requireEl(btn, '.layout-label').textContent = 'Grid';
        _requireEl(btn, '.icon-grid').style.display = '';
        _requireEl(btn, '.icon-stack').style.display = 'none';
      }
    } else {
      figArea.classList.remove('layout-stacked');
      if (btn) {
        _requireEl(btn, '.layout-label').textContent = 'Stack';
        _requireEl(btn, '.icon-grid').style.display = 'none';
        _requireEl(btn, '.icon-stack').style.display = '';
      }
    }

    localStorage.setItem(LAYOUT_KEY, mode);
    _reflowFigures();
  }

  // ── Sidebar Tabs ────────────────────────────────────────

  function initSidebarTabs() {
    const tabs = document.querySelectorAll('.sidebar-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = /** @type {HTMLElement} */ (tab).dataset.tab;
        // Picking a tab implies wanting to see it. Routed through the splitter
        // so the class, the persisted state and the Plotly reflow stay in step.
        expandSidebar();
        switchTab(/** @type {string} */ (tabName));
      });
    });

    // Restore last active tab
    const savedTab = localStorage.getItem(SIDEBAR_TAB_KEY);
    if (savedTab) switchTab(savedTab);
  }

  /** @param {string} tabName */
  function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.sidebar-tab').forEach(t => {
      t.classList.toggle('sidebar-tab--active', /** @type {HTMLElement} */ (t).dataset.tab === tabName);
    });
    // Update tab content
    document.querySelectorAll('.sidebar-tab-content').forEach(panel => {
      panel.classList.toggle('sidebar-tab-content--active', panel.id === `tab-${tabName}`);
    });
    localStorage.setItem(SIDEBAR_TAB_KEY, tabName);
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
    const order = Array.from(cells).map(c => /** @type {HTMLElement} */ (c).dataset.figure);
    localStorage.setItem(PANEL_ORDER_KEY, JSON.stringify(order));
  }

  function setupDragAndDrop() {
    const cells = document.querySelectorAll('.figure-cell');
    /** @type {Element|null} */
    let draggedCell = null;

    cells.forEach(cell => {
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
        cells.forEach(c => c.classList.remove('drag-over'));
      });
    });
  }

  function restorePanelOrder() {
    const saved = localStorage.getItem(PANEL_ORDER_KEY);
    if (!saved) return;

    try {
      /** @type {string[]} */
      const order = JSON.parse(saved);

      // Collect all containers that hold figure cells (grid + verification row)
      const containers = document.querySelectorAll('.figure-grid, .verification-row');
      /** @type {{container: Element, placeholder: Element}[]} */
      const allSlots = [];
      containers.forEach(container => {
        Array.from(container.children).forEach(child => {
          if (child.classList.contains('figure-cell')) {
            allSlots.push({ container, placeholder: child });
          }
        });
      });

      // Build lookup of cells by figure name
      /** @type {Record<string, Element>} */
      const cellMap = {};
      allSlots.forEach(slot => {
        const figureName = /** @type {HTMLElement} */ (slot.placeholder).dataset.figure;
        if (figureName) cellMap[figureName] = slot.placeholder;
      });

      // Validate saved order matches current DOM cells — if panel names
      // have changed (e.g. fma → lma), discard stale order
      const currentNames = new Set(Object.keys(cellMap));
      const savedNames = new Set(order);
      if (order.length !== allSlots.length || ![...currentNames].every(n => savedNames.has(n))) {
        localStorage.removeItem(PANEL_ORDER_KEY);
        return;
      }

      // Reorder: place cells into slots according to saved order
      const slotPositions = allSlots.map(s => ({
        container: s.container,
        nextSibling: s.placeholder.nextSibling,
      }));

      // Detach all cells
      allSlots.forEach(s => s.placeholder.remove());

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
      localStorage.removeItem(PANEL_ORDER_KEY);
    }
  }

  return {
    initSidebar,
    toggleSidebar,
    initLayout,
    toggleLayout,
    initSidebarTabs,
    setupDragAndDrop,
    restorePanelOrder,
    reflowFigures: _reflowFigures,
  };
}
