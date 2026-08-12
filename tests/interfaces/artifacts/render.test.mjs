/**
 * Unit tests for the Artifact Gallery sidebar rendering layer (render.js:
 * gallery-card template, sidebar dispatcher + tree/activity renderers +
 * attachSidebarHandlers, pinned promotion). The browse split's orientation
 * and divider are covered by browse-layout.test.mjs; the ⋯ overflow menu
 * by sidebar-menu.test.mjs.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/render.test.mjs
 *
 * render.js reads/writes state.js's module-singleton artifact list
 * (mirrors state.test.mjs's convention: no vi.resetModules, just call the
 * setters fresh per test), and formats via types.js (real implementations
 * — no network calls are exercised by anything covered here).
 *
 * Covers tree-mode grouping by type from fixture artifacts (with pinned
 * artifacts promoted into a top "Pinned" section) and activity-mode
 * chronological ordering. Also covers the injected-callback contract
 * (onSelect/onPreviewNeeded/onEnterFullscreen) since that is the piece
 * genuinely new to this extraction (createSidebarRenderer's factory shape,
 * and the still-in-gallery.js effects it defers to).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  setArtifacts,
  setSelectedArtifact,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';
import {
  createSidebarRenderer,
} from '../../../src/osprey/interfaces/artifacts/static/js/render.js';
import { initTypeRegistry } from '../../../src/osprey/interfaces/artifacts/static/js/types.js';
import { qs, byId } from '../_support/dom.mjs';

/** Minimal DOM fixture matching artifacts/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `
    <input id="search" />
    <aside class="browse-sidebar" id="browse-sidebar">
      <div class="sidebar-body" id="sidebar-body"></div>
    </aside>
    <div class="resize-handle osprey-splitter" id="resize-handle"></div>
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
  setSelectedArtifact(null);
});

describe('tree-mode grouping by type (pinned promoted)', () => {
  test('promotes pinned artifacts into a top "Pinned" section, then groups the rest by category, most-populous first', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());

    sidebarRenderer.renderSidebar(); // defaults to tree mode

    const sections = document.querySelectorAll('#sidebar-body .tree-section');
    // Pinned ids 2 + 4 are PROMOTED out of their type groups (leaving
    // 'channel_values' and 'document' empty — those groups vanish), so
    // 'visualization' (Beam Profile, Lattice Table) is the only type group.
    expect(Array.from(sections).map((s) => /** @type {HTMLElement} */ (s).dataset.type)).toEqual([
      'pinned', 'visualization',
    ]);

    const pinnedSection = qs(document, '.tree-section[data-type="pinned"]');
    // getFilteredArtifacts orders pinned-first/newest-first: 2 (Jul 3), 4 (Jun 30).
    expect(
      Array.from(pinnedSection.querySelectorAll('.tree-item')).map((el) => /** @type {HTMLElement} */ (el).dataset.id)
    ).toEqual(['2', '4']);
    expect(qs(pinnedSection, '.tree-section-count').textContent).toBe('2');

    const visSection = qs(document, '.tree-section[data-type="visualization"]');
    expect(visSection.querySelectorAll('.tree-item').length).toBe(2);
    expect(qs(visSection, '.tree-section-count').textContent).toBe('2');
  });

  test('no "Pinned" section renders when nothing is pinned', () => {
    setArtifacts(makeFixtureArtifacts().map((a) => ({ ...a, pinned: false })));
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    expect(document.querySelector('.tree-section[data-type="pinned"]')).toBeNull();
    const sections = document.querySelectorAll('#sidebar-body .tree-section');
    expect(Array.from(sections).map((s) => /** @type {HTMLElement} */ (s).dataset.type)).toEqual([
      'visualization', 'channel_values', 'document',
    ]);
  });

  test('each tree item renders its title, pin indicator, and size', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const pinnedItem = document.querySelector('.tree-item[data-id="2"]');
    expect(pinnedItem).not.toBeNull();
    if (pinnedItem === null) throw new Error('unreachable: asserted non-null above');
    expect(pinnedItem.classList.contains('pinned')).toBe(true);
    expect(qs(pinnedItem, '.tree-item-name').textContent).toBe('Channel Values');
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
    expect(Array.from(items).map((el) => /** @type {HTMLElement} */ (el).dataset.id)).toEqual(['2', '4', '3', '1']);
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
    expect(Array.from(items).map((el) => /** @type {HTMLElement} */ (el).dataset.id)).toEqual(['b', 'a']); // newest-first
  });
});

describe('shared item handlers (click/dblclick/drag-to-terminal)', () => {
  test('clicking an item selects it, invokes onSelect, and invokes onPreviewNeeded only on a real selection change', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    const sidebarRenderer = createSidebarRenderer(callbacks);
    sidebarRenderer.renderSidebar();

    const item = qs(document, '.tree-item[data-id="1"]');
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

    qs(document, '.tree-item[data-id="3"]').dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));

    expect(callbacks.onEnterFullscreen).toHaveBeenCalledTimes(1);
    expect(callbacks.onEnterFullscreen.mock.calls[0][0].id).toBe('3');
  });

  test('dragging an item sets the terminal-paste reference text on the DataTransfer', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const item = qs(document, '.tree-item[data-id="1"]');
    expect(item.draggable).toBe(true);

    const dataTransfer = { setData: vi.fn(), effectAllowed: '' };
    const dragEvent = /** @type {Event & { dataTransfer: typeof dataTransfer }} */ (new Event('dragstart', { bubbles: true }));
    dragEvent.dataTransfer = dataTransfer;
    item.dispatchEvent(dragEvent);

    expect(dataTransfer.setData).toHaveBeenCalledWith(
      'text/plain',
      'Please have a look at var/agent_data/artifacts/beam_profile.png'
    );
    expect(dataTransfer.effectAllowed).toBe('copy');
  });

  test('clicking a tree-section header collapses/expands its section without triggering item selection', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    const sidebarRenderer = createSidebarRenderer(callbacks);
    sidebarRenderer.renderSidebar();

    const section = qs(document, '.tree-section[data-type="visualization"]');
    const header = qs(section, '.tree-section-header');
    header.click();

    expect(section.classList.contains('collapsed')).toBe(true);
    expect(callbacks.onSelect).not.toHaveBeenCalled();
  });
});

describe('XSS hardening (Task 1.3 — escape-metadata-sinks)', () => {
  const HOSTILE = '"><img src=x onerror=alert(1)>';

  /**
   * Seed state with one artifact whose category/artifact_type carry the
   * hostile payload (title/filename are inert and unasserted).
   * @param {string} id
   */
  function setHostileArtifact(id) {
    setArtifacts([
      { id, title: `Hostile ${id}`, filename: `${id}.png`, artifact_type: HOSTILE, category: HOSTILE, pinned: false, timestamp: '2026-07-04T10:00:00Z', size_bytes: 10 },
    ]);
  }

  afterEach(async () => {
    // Reset the module-level type registry so a test that re-inits it with a
    // hostile label can't leak into later tests — even when its assertions fail.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: () => Promise.resolve({}) }));
    await initTypeRegistry();
    vi.unstubAllGlobals();
  });

  test('a hostile category/artifact_type is escaped in tree-mode data-type attributes — no live <img>, no unescaped attribute breakout', () => {
    setHostileArtifact('x1');
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar(); // default: tree mode, list layout

    // No live <img> element was injected anywhere in the sidebar.
    expect(document.querySelector('#sidebar-body img')).toBeNull();
    // No unescaped '"><' breakout sequence made it into the serialized markup.
    expect(byId('sidebar-body').innerHTML).not.toMatch(/"><img/);

    const section = document.querySelector('.tree-section');
    expect(section).not.toBeNull();
    if (!(section instanceof HTMLElement)) throw new Error('unreachable: asserted non-null above');
    // dataset/getAttribute auto-decode entities, so reading the attribute back
    // round-trips to the raw hostile string (entity-decoded) — that's expected
    // and safe; what matters is the SERIALIZED markup never contained a live
    // '"><' breakout, asserted above.
    expect(section.dataset.type).toBe(HOSTILE);
    expect(qs(section, '.tree-section-header').dataset.type).toBe(HOSTILE);
  });

  test('a hostile category is escaped in the activity-mode timeline-item data-type attribute', () => {
    setHostileArtifact('x2');
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.setBrowseMode('activity');
    sidebarRenderer.renderSidebar();

    expect(document.querySelector('#sidebar-body img')).toBeNull();
    expect(byId('sidebar-body').innerHTML).not.toMatch(/"><img/);

    const item = document.querySelector('.timeline-item');
    expect(item).not.toBeNull();
    if (!(item instanceof HTMLElement)) throw new Error('unreachable: asserted non-null above');
    expect(item.dataset.type).toBe(HOSTILE);
  });

  test('a hostile gallery-layout category is escaped in the gallery-card data-type attribute', () => {
    setHostileArtifact('x3');
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.setSidebarLayout('gallery');
    sidebarRenderer.renderSidebar();

    expect(document.querySelector('#sidebar-body img')).toBeNull();
    const card = document.querySelector('.gallery-card');
    expect(card).not.toBeNull();
    if (!(card instanceof HTMLElement)) throw new Error('unreachable: asserted non-null above');
    expect(card.dataset.type).toBe(HOSTILE);
  });

  test('a hostile registry label reaches the tree-section header only as escaped text', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ categories: { evilcat: { label: '<img src=x onerror=alert(1)>' } } }),
    }));
    await initTypeRegistry();

    setArtifacts([
      { id: 'c1', title: 'Section Test', filename: 'c.png', artifact_type: 'evilcat', category: 'evilcat', pinned: false, timestamp: '2026-07-04T10:00:00Z', size_bytes: 10 },
    ]);
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const header = document.querySelector('.tree-section-header');
    expect(header).not.toBeNull();
    if (header === null) throw new Error('unreachable: asserted non-null above');
    expect(header.querySelector('img')).toBeNull();
    // typeIcon's own SVG markup (audited: `type` never reaches its output) is
    // untouched by this escaping and must still render.
    expect(header.querySelector('.tree-section-icon svg')).not.toBeNull();
    expect(header.innerHTML).toContain('&lt;img');
    expect(header.innerHTML).not.toContain('<img');
  });

  test('benign categories render byte-identical output (regression guard)', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const visSection = document.querySelector('.tree-section[data-type="visualization"]');
    expect(visSection).not.toBeNull();
    if (!(visSection instanceof HTMLElement)) throw new Error('unreachable: asserted non-null above');
    expect(visSection.dataset.type).toBe('visualization');
    expect(qs(visSection, '.tree-section-header').dataset.type).toBe('visualization');
    // The serialized markup is also byte-identical for benign values —
    // escaping must introduce no stray entities.
    expect(visSection.outerHTML).toContain('data-type="visualization"');
  });
});
