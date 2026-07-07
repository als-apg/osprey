/**
 * Unit tests for the Artifact Gallery sidebar rendering layer (render.js:
 * filter bar, gallery-card template, sidebar dispatcher + tree/activity
 * renderers + attachSidebarHandlers, and the split-pane resize handle).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/render.test.mjs
 *
 * render.js reads/writes state.js's module-singleton artifact list/filter
 * (mirrors state.test.mjs's convention: no vi.resetModules, just call the
 * setters fresh per test), and formats via types.js (real implementations
 * — no network calls are exercised by anything covered here).
 *
 * Covers exactly what the task calls for: tree-mode grouping by type from
 * fixture artifacts, activity-mode chronological ordering, and filter-chip
 * active-state logic. Also covers the split-pane resize handle (pure
 * export) and the injected-callback contract (onSelect/onPreviewNeeded/
 * onEnterFullscreen) since those are the two pieces genuinely new to this
 * extraction (createSidebarRenderer's factory shape, and the still-in-
 * gallery.js effects it defers to).
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import {
  setArtifacts,
  setSelectedArtifact,
  setActiveFilter,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';
import {
  initSplitPaneResize,
  createSidebarRenderer,
} from '../../../src/osprey/interfaces/artifacts/static/js/render.js';

/** Minimal DOM fixture matching artifacts/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `
    <nav class="filter-bar" id="filter-bar">
      <div class="filter-bar-inner">
        <button class="filter-chip active" data-filter="all">ALL</button>
        <button class="filter-chip" data-filter="pinned" title="Pinned items" hidden>
          Pinned <span class="chip-count"></span>
        </button>
        <span class="filter-chip-separator" id="filter-type-chips"></span>
      </div>
    </nav>
    <input id="search" />
    <aside class="browse-sidebar" id="browse-sidebar">
      <div class="sidebar-body" id="sidebar-body"></div>
    </aside>
    <div class="resize-handle" id="resize-handle"><div class="resize-handle-grip"></div></div>
  `;
}

function makeFixtureArtifacts() {
  return [
    { id: '1', title: 'Beam Profile', filename: 'beam_profile.png', artifact_type: 'plot_png', category: 'visualization', pinned: false, timestamp: '2026-07-01T10:00:00Z', size_bytes: 2048 },
    { id: '2', title: 'Channel Values', filename: 'channels.json', artifact_type: 'json', category: 'channel_values', pinned: true, timestamp: '2026-07-03T10:00:00Z', size_bytes: 512 },
    { id: '3', title: 'Lattice Table', filename: 'lattice.html', artifact_type: 'table_html', category: 'visualization', pinned: false, timestamp: '2026-07-02T10:00:00Z', size_bytes: 4096 },
    { id: '4', title: 'Old Report', filename: 'report.md', artifact_type: 'markdown', category: 'document', pinned: true, timestamp: '2026-06-30T10:00:00Z', size_bytes: 1024 },
  ];
}

function makeCallbacks() {
  return {
    onSelect: vi.fn(),
    onPreviewNeeded: vi.fn(),
    onEnterFullscreen: vi.fn(),
  };
}

beforeEach(() => {
  mountFixture();
  setActiveFilter('all');
  setSelectedArtifact(null);
});

describe('tree-mode grouping by type', () => {
  test('groups fixture artifacts by category/artifact_type, most-populous group first', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());

    sidebarRenderer.renderSidebar(); // defaults to tree mode

    const sections = document.querySelectorAll('#sidebar-body .tree-section');
    // 'visualization' (2 artifacts: Beam Profile, Lattice Table) sorts before
    // the two 1-artifact groups ('channel_values', 'document'), which then
    // sort alphabetically.
    expect(Array.from(sections).map((s) => s.dataset.type)).toEqual([
      'visualization', 'channel_values', 'document',
    ]);

    const visSection = document.querySelector('.tree-section[data-type="visualization"]');
    expect(visSection.querySelectorAll('.tree-item').length).toBe(2);
    expect(visSection.querySelector('.tree-section-count').textContent).toBe('2');
  });

  test('each tree item renders its title, pin indicator, and size', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const pinnedItem = document.querySelector('.tree-item[data-id="2"]');
    expect(pinnedItem).not.toBeNull();
    expect(pinnedItem.classList.contains('pinned')).toBe(true);
    expect(pinnedItem.querySelector('.tree-item-name').textContent).toBe('Channel Values');
    expect(pinnedItem.querySelector('.pin-indicator')).not.toBeNull();
  });

  test('an empty (fully filtered-out) artifact list shows the empty state instead of tree sections', () => {
    setArtifacts([]);
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    expect(document.querySelector('.sidebar-empty')).not.toBeNull();
    expect(document.querySelectorAll('.tree-section').length).toBe(0);
  });
});

describe('activity-mode chronological ordering', () => {
  test('groups by date label and orders newest-first within "today"-equivalent single-day fixtures', () => {
    // All four fixtures land on distinct days, so each date group holds one
    // item; assert the date GROUPS themselves preserve encounter order
    // (getFilteredArtifacts already sorted pinned-first/newest-first, and
    // renderActivityMode groups in that same iteration order without
    // re-sorting) — i.e. chronological-by-recency across groups.
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.setBrowseMode('activity');

    sidebarRenderer.renderSidebar();

    const items = document.querySelectorAll('#sidebar-body .timeline-item');
    // getFilteredArtifacts sorts pinned-first then newest-first: '2' (pinned,
    // Jul 3) and '4' (pinned, Jun 30) before '3' (Jul 2) and '1' (Jul 1) —
    // pinned-first takes priority over date recency, matching state.js's
    // contract (already pinned to the identical order in state.test.mjs).
    expect(Array.from(items).map((el) => el.dataset.id)).toEqual(['2', '4', '3', '1']);
  });

  test('within a shared date group, items keep the incoming (pinned-first/newest-first) order', () => {
    const sameDay = [
      { id: 'a', title: 'Morning', filename: 'a.png', artifact_type: 'image', category: 'screenshot', pinned: false, timestamp: '2026-07-01T08:00:00Z', size_bytes: 100 },
      { id: 'b', title: 'Evening', filename: 'b.png', artifact_type: 'image', category: 'screenshot', pinned: false, timestamp: '2026-07-01T20:00:00Z', size_bytes: 100 },
    ];
    setArtifacts(sameDay);
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.setBrowseMode('activity');
    sidebarRenderer.renderSidebar();

    const groups = document.querySelectorAll('.timeline-group');
    expect(groups.length).toBe(1); // same calendar day -> one date group
    const items = groups[0].querySelectorAll('.timeline-item');
    expect(Array.from(items).map((el) => el.dataset.id)).toEqual(['b', 'a']); // newest-first
  });
});

describe('filter-chip active-state logic', () => {
  test('initFilterBar wires clicks so the clicked chip becomes active and others deactivate', () => {
    setArtifacts(makeFixtureArtifacts()); // gives the "pinned" chip a count > 0, unhiding it
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.initFilterBar();

    const allChip = document.querySelector('.filter-chip[data-filter="all"]');
    const pinnedChip = document.querySelector('.filter-chip[data-filter="pinned"]');
    expect(allChip.classList.contains('active')).toBe(true);
    expect(pinnedChip.hidden).toBe(false);
    expect(pinnedChip.querySelector('.chip-count').textContent).toBe('2');

    pinnedChip.click();

    expect(pinnedChip.classList.contains('active')).toBe(true);
    expect(allChip.classList.contains('active')).toBe(false);
  });

  test('rebuildTypeChips hides the pinned chip and resets the filter when no artifacts are pinned', () => {
    setActiveFilter('pinned');
    setArtifacts(makeFixtureArtifacts().map((a) => ({ ...a, pinned: false })));
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());

    sidebarRenderer.rebuildTypeChips();

    const pinnedChip = document.querySelector('.filter-chip[data-filter="pinned"]');
    expect(pinnedChip.hidden).toBe(true);
    // Active filter is reset since "pinned" no longer has any matches.
    expect(document.querySelector('.filter-chip[data-filter="all"]').classList.contains('active')).toBe(true);
  });

  test('clicking a chip re-renders the sidebar filtered to that selection', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.initFilterBar();
    sidebarRenderer.renderSidebar();

    document.querySelector('.filter-chip[data-filter="pinned"]').click();

    // Only pinned artifacts (ids 2, 4) should now be rendered.
    const ids = Array.from(document.querySelectorAll('#sidebar-body [data-id]')).map((el) => el.dataset.id).sort();
    expect(ids).toEqual(['2', '4']);
  });
});

describe('shared item handlers (click/dblclick/drag-to-terminal)', () => {
  test('clicking an item selects it, invokes onSelect, and invokes onPreviewNeeded only on a real selection change', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    const sidebarRenderer = createSidebarRenderer(callbacks);
    sidebarRenderer.renderSidebar();

    const item = document.querySelector('.tree-item[data-id="1"]');
    item.click();

    expect(callbacks.onSelect).toHaveBeenCalledTimes(1);
    expect(callbacks.onSelect.mock.calls[0][0].id).toBe('1');
    expect(callbacks.onPreviewNeeded).toHaveBeenCalledTimes(1);
    expect(item.classList.contains('selected')).toBe(true);

    // Re-clicking the already-selected item still fires onSelect (matches
    // setAsFocus's original behavior) but must NOT re-trigger the preview.
    item.click();
    expect(callbacks.onSelect).toHaveBeenCalledTimes(2);
    expect(callbacks.onPreviewNeeded).toHaveBeenCalledTimes(1);
  });

  test('double-clicking an item invokes onEnterFullscreen with that artifact', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    const sidebarRenderer = createSidebarRenderer(callbacks);
    sidebarRenderer.renderSidebar();

    document.querySelector('.tree-item[data-id="3"]').dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));

    expect(callbacks.onEnterFullscreen).toHaveBeenCalledTimes(1);
    expect(callbacks.onEnterFullscreen.mock.calls[0][0].id).toBe('3');
  });

  test('dragging an item sets the terminal-paste reference text on the DataTransfer', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const item = document.querySelector('.tree-item[data-id="1"]');
    expect(item.draggable).toBe(true);

    const dataTransfer = { setData: vi.fn(), effectAllowed: '' };
    const dragEvent = new Event('dragstart', { bubbles: true });
    dragEvent.dataTransfer = dataTransfer;
    item.dispatchEvent(dragEvent);

    expect(dataTransfer.setData).toHaveBeenCalledWith(
      'text/plain',
      'Please have a look at _agent_data/artifacts/beam_profile.png'
    );
    expect(dataTransfer.effectAllowed).toBe('copy');
  });

  test('clicking a tree-section header collapses/expands its section without triggering item selection', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    const sidebarRenderer = createSidebarRenderer(callbacks);
    sidebarRenderer.renderSidebar();

    const section = document.querySelector('.tree-section[data-type="visualization"]');
    const header = section.querySelector('.tree-section-header');
    header.click();

    expect(section.classList.contains('collapsed')).toBe(true);
    expect(callbacks.onSelect).not.toHaveBeenCalled();
  });
});

describe('split-pane resize (initSplitPaneResize)', () => {
  test('is a no-op when either element is missing', () => {
    expect(() => initSplitPaneResize(null, document.getElementById('browse-sidebar'))).not.toThrow();
    expect(() => initSplitPaneResize(document.getElementById('resize-handle'), null)).not.toThrow();
  });

  test('dragging the handle resizes the sidebar within the [180, 60vw] clamp', () => {
    const handle = document.getElementById('resize-handle');
    const sidebarEl = document.getElementById('browse-sidebar');
    Object.defineProperty(sidebarEl, 'offsetWidth', { value: 240, configurable: true });
    Object.defineProperty(window, 'innerWidth', { value: 1000, configurable: true });

    initSplitPaneResize(handle, sidebarEl);

    handle.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: 100 }));
    document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 150 })); // +50
    expect(sidebarEl.style.width).toBe('290px');

    // Clamp at the low end: a huge negative delta still leaves >= 180px.
    document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: -1000 }));
    expect(sidebarEl.style.width).toBe('180px');

    // Clamp at the high end: 60% of 1000px innerWidth = 600px.
    document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 5000 }));
    expect(sidebarEl.style.width).toBe('600px');
  });
});
