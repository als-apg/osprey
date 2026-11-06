/**
 * Unit tests for the Artifact Gallery sidebar rendering layer (render.js:
 * gallery-card template, sidebar dispatcher + tree/activity renderers +
 * attachSidebarHandlers). The browse split's orientation
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
 * Covers tree-mode grouping by type from fixture artifacts and activity-mode
 * chronological ordering. Also covers the injected-callback contract
 * (onSelect/onPreviewNeeded/onEnterFullscreen) since that is the piece
 * genuinely new to this extraction (createSidebarRenderer's factory shape,
 * and the still-in-gallery.js effects it defers to).
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  setArtifacts,
  setSelectedArtifact,
  getPickedIds,
  setPicked,
  clearPicked,
  getPickAnchor,
  setPickAnchor,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';
import {
  createSidebarRenderer,
  isNearListEnd,
  pickRange,
  revealDelayMs,
} from '../../../src/osprey/interfaces/artifacts/static/js/render.js';
import { initTypeRegistry } from '../../../src/osprey/interfaces/artifacts/static/js/types.js';
import { qs, byId } from '../_support/dom.mjs';
import { stampFacilityZone } from '../_support/facility-zone.mjs';

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
    {
      id: '1',
      title: 'Beam Profile',
      filename: 'beam_profile.png',
      artifact_type: 'plot_png',
      form: 'plots',
      form_label: 'Plots',
      timestamp: '2026-07-01T10:00:00Z',
      size_bytes: 2048,
    },
    {
      id: '2',
      title: 'Channel Values',
      filename: 'channels.json',
      artifact_type: 'json',
      form: 'data',
      form_label: 'Data',
      timestamp: '2026-07-03T10:00:00Z',
      size_bytes: 512,
    },
    {
      id: '3',
      title: 'Lattice Table',
      filename: 'lattice.html',
      artifact_type: 'plot_html',
      form: 'plots',
      form_label: 'Plots',
      timestamp: '2026-07-02T10:00:00Z',
      size_bytes: 4096,
    },
    {
      id: '4',
      title: 'Old Report',
      filename: 'report.md',
      artifact_type: 'markdown',
      form: 'documents',
      form_label: 'Documents',
      timestamp: '2026-06-30T10:00:00Z',
      size_bytes: 1024,
    },
  ];
}

/** The shared /api/type-registry fixture the hub sidebar suite reads too. */
const REGISTRY = JSON.parse(
  readFileSync(
    resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/type-registry.json'),
    'utf8'
  )
);

/** @param {any} registry */
async function loadRegistry(registry) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: () => Promise.resolve(registry) }));
  await initTypeRegistry();
  vi.unstubAllGlobals();
}

function makeCallbacks() {
  return {
    onSelect: vi.fn(),
    onPreviewNeeded: vi.fn(),
    onEnterFullscreen: vi.fn(),
  };
}

beforeEach(async () => {
  await loadRegistry(REGISTRY);
  mountFixture();
  setSelectedArtifact(null);
  clearPicked();
  setPickAnchor(null);
});

describe('tree-mode grouping by type', () => {
  test('groups by form, with no "Pinned" section even for a legacy pinned flag', () => {
    setArtifacts(makeFixtureArtifacts().map((a) => ({ ...a, pinned: true })));
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    expect(document.querySelector('.tree-section[data-type="pinned"]')).toBeNull();
    const sections = document.querySelectorAll('#sidebar-body .tree-section');
    expect(Array.from(sections).map((s) => /** @type {HTMLElement} */ (s).dataset.type)).toEqual([
      'plots',
      'documents',
      'data',
    ]);
  });

  test('each tree item renders its title and no pin indicator', () => {
    setArtifacts(makeFixtureArtifacts().map((a) => ({ ...a, pinned: true })));
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const item = document.querySelector('.tree-item[data-id="2"]');
    expect(item).not.toBeNull();
    if (item === null) throw new Error('unreachable: asserted non-null above');
    expect(item.classList.contains('pinned')).toBe(false);
    expect(qs(item, '.tree-item-name').textContent).toBe('Channel Values');
    expect(item.querySelector('.pin-indicator')).toBeNull();
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
  // Date groups are the facility calendar's days, and the fixtures are UTC
  // instants, so the facility zone is UTC here in any runner zone.
  beforeEach(() => stampFacilityZone('UTC'));
  afterEach(() => stampFacilityZone(null));

  test('groups by date label and orders newest-first within "today"-equivalent single-day fixtures', () => {
    // All four fixtures land on distinct days, so each date group holds one
    // item; assert the date GROUPS themselves preserve encounter order
    // (getFilteredArtifacts already sorted newest-first, and
    // renderActivityMode groups in that same iteration order without
    // re-sorting) — i.e. chronological-by-recency across groups.
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.setBrowseMode('activity');

    sidebarRenderer.renderSidebar();

    const items = document.querySelectorAll('#sidebar-body .timeline-item');
    // getFilteredArtifacts sorts newest-first: '2' (Jul 3), '3' (Jul 2),
    // '1' (Jul 1), '4' (Jun 30) — state.js's contract, held to the identical
    // order in state.test.mjs.
    expect(Array.from(items).map((el) => /** @type {HTMLElement} */ (el).dataset.id)).toEqual([
      '2',
      '3',
      '1',
      '4',
    ]);
  });

  test('within a shared date group, items keep the incoming (newest-first) order', () => {
    const sameDay = [
      {
        id: 'a',
        title: 'Morning',
        filename: 'a.png',
        artifact_type: 'image',
        form: 'images',
        form_label: 'Images',
        timestamp: '2026-07-01T08:00:00Z',
        size_bytes: 100,
      },
      {
        id: 'b',
        title: 'Evening',
        filename: 'b.png',
        artifact_type: 'image',
        form: 'images',
        form_label: 'Images',
        timestamp: '2026-07-01T20:00:00Z',
        size_bytes: 100,
      },
    ];
    setArtifacts(sameDay);
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.setBrowseMode('activity');
    sidebarRenderer.renderSidebar();

    const groups = document.querySelectorAll('.timeline-group');
    expect(groups.length).toBe(1); // same calendar day -> one date group
    const items = groups[0].querySelectorAll('.timeline-item');
    expect(Array.from(items).map((el) => /** @type {HTMLElement} */ (el).dataset.id)).toEqual([
      'b',
      'a',
    ]); // newest-first
  });
});

describe('shipped example (origin "demo")', () => {
  const example = {
    id: 'ex',
    title: 'Example: an interactive plot',
    filename: 'example.html',
    artifact_type: 'plot_html',
    form: 'plots',
    form_label: 'Plots',
    origin: 'demo',
    // Newer than everything else: seeded at gallery start. Must still sort last.
    timestamp: '2026-07-09T10:00:00Z',
    size_bytes: 50000,
  };

  test('tree mode lists it in a trailing "Examples" section, outside the type groups, with an "example" badge and no "new" badge', () => {
    setArtifacts([...makeFixtureArtifacts(), example]);
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const sections = document.querySelectorAll('#sidebar-body .tree-section');
    expect(Array.from(sections).map((s) => /** @type {HTMLElement} */ (s).dataset.type)).toEqual([
      'plots',
      'documents',
      'data',
      'examples',
    ]);

    const examples = qs(document, '.tree-section[data-type="examples"]');
    expect(examples.textContent).toContain('Examples');
    const item = qs(examples, '.tree-item[data-id="ex"]');
    expect(qs(item, '.tree-item-badge.demo').textContent).toBe('example');
    expect(item.querySelector('.tree-item-badge.new')).toBeNull();
  });

  test('no "Examples" section renders when the store holds none', () => {
    setArtifacts(makeFixtureArtifacts());
    createSidebarRenderer(makeCallbacks()).renderSidebar();
    expect(document.querySelector('.tree-section[data-type="examples"]')).toBeNull();
  });

  test('an example alone still renders the tree, not the empty state', () => {
    setArtifacts([example]);
    createSidebarRenderer(makeCallbacks()).renderSidebar();
    expect(document.querySelector('.sidebar-empty')).toBeNull();
    expect(document.querySelectorAll('.tree-section[data-type="examples"] .tree-item').length).toBe(
      1
    );
  });

  test('activity mode puts it in a trailing "Examples" group instead of the day it was seeded', () => {
    setArtifacts([...makeFixtureArtifacts(), example]);
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.setBrowseMode('activity');
    sidebarRenderer.renderSidebar();

    const labels = Array.from(document.querySelectorAll('.timeline-group-label')).map(
      (el) => el.textContent
    );
    expect(labels.at(-1)).toBe('Examples');
    expect(labels.filter((l) => l === 'Examples').length).toBe(1);
    const last = /** @type {HTMLElement} */ (
      document.querySelectorAll('.timeline-group')[labels.length - 1]
    );
    expect(
      Array.from(last.querySelectorAll('.timeline-item')).map(
        (el) => /** @type {HTMLElement} */ (el).dataset.id
      )
    ).toEqual(['ex']);
    expect(qs(last, '.tree-item-badge.demo').textContent).toBe('example');
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

    qs(document, '.tree-item[data-id="3"]').dispatchEvent(
      new MouseEvent('dblclick', { bubbles: true })
    );

    expect(callbacks.onEnterFullscreen).toHaveBeenCalledTimes(1);
    expect(callbacks.onEnterFullscreen.mock.calls[0][0].id).toBe('3');
  });

  test('dragging an item sets the terminal-paste reference text on the DataTransfer', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const item = qs(document, '.tree-item[data-id="1"]');
    // Carried in the markup, so a rebuilt row is draggable without a per-row property set.
    expect(item.getAttribute('draggable')).toBe('true');

    const meta = document.createElement('meta');
    meta.setAttribute('name', 'osprey-artifact-dir');
    meta.setAttribute('content', 'state/agent/artifacts');
    document.head.appendChild(meta);
    try {
      const dataTransfer = { setData: vi.fn(), effectAllowed: '' };
      const dragEvent = /** @type {Event & { dataTransfer: typeof dataTransfer }} */ (
        new Event('dragstart', { bubbles: true })
      );
      dragEvent.dataTransfer = dataTransfer;
      item.dispatchEvent(dragEvent);

      expect(dataTransfer.setData).toHaveBeenCalledWith(
        'text/plain',
        'Please have a look at state/agent/artifacts/beam_profile.png'
      );
      expect(dataTransfer.effectAllowed).toBe('copy');
    } finally {
      meta.remove(); // this file's document is shared across tests
    }
  });

  test('clicking a tree-section header collapses/expands its section without triggering item selection', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    const sidebarRenderer = createSidebarRenderer(callbacks);
    sidebarRenderer.renderSidebar();

    const section = qs(document, '.tree-section[data-type="plots"]');
    const header = qs(section, '.tree-section-header');
    header.click();

    expect(section.classList.contains('collapsed')).toBe(true);
    expect(callbacks.onSelect).not.toHaveBeenCalled();
  });

  // Every new artifact re-renders the sidebar; a section the user collapsed
  // re-expanded each time the agent wrote another file into it.
  test('a collapsed section stays collapsed when a new artifact re-renders the sidebar', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();
    qs(document, '.tree-section[data-type="plots"] .tree-section-header').click();

    setArtifacts([
      ...makeFixtureArtifacts(),
      {
        id: '5',
        title: 'New Plot',
        filename: 'new.png',
        artifact_type: 'plot_png',
        form: 'plots',
        form_label: 'Plots',
        timestamp: '2026-07-04T10:00:00Z',
        size_bytes: 100,
      },
    ]);
    sidebarRenderer.renderSidebar();

    expect(qs(document, '.tree-section[data-type="plots"]').classList.contains('collapsed')).toBe(
      true
    );
    expect(qs(document, '.tree-section[data-type="data"]').classList.contains('collapsed')).toBe(
      false
    );
  });
});

describe('XSS hardening (Task 1.3 — escape-metadata-sinks)', () => {
  const HOSTILE = '"><img src=x onerror=alert(1)>';

  /**
   * Seed state with one artifact whose form/form_label/artifact_type carry the
   * hostile payload (title/filename are inert and unasserted).
   * @param {string} id
   */
  function setHostileArtifact(id) {
    setArtifacts([
      {
        id,
        title: `Hostile ${id}`,
        filename: `${id}.png`,
        artifact_type: HOSTILE,
        category: HOSTILE,
        form: HOSTILE,
        form_label: HOSTILE,
        timestamp: '2026-07-04T10:00:00Z',
        size_bytes: 10,
      },
    ]);
  }

  afterEach(async () => {
    // Reset the module-level type registry so a test that re-inits it with a
    // hostile label can't leak into later tests — even when its assertions fail.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: () => Promise.resolve({}) }));
    await initTypeRegistry();
    vi.unstubAllGlobals();
  });

  test('a hostile form/form_label is escaped in tree-mode data-type attributes — no live <img>, no unescaped attribute breakout', () => {
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

  test('a hostile form is escaped in the activity-mode timeline-item data-type attribute', () => {
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

  test('a hostile gallery-layout form is escaped in the gallery-card data-type attribute', () => {
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
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        json: () =>
          Promise.resolve({
            base_forms: [{ key: 'evilform', label: '<img src=x onerror=alert(1)>' }],
          }),
      })
    );
    await initTypeRegistry();

    setArtifacts([
      {
        id: 'c1',
        title: 'Section Test',
        filename: 'c.png',
        artifact_type: 'json',
        form: 'evilform',
        form_label: 'Evil',
        timestamp: '2026-07-04T10:00:00Z',
        size_bytes: 10,
      },
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

  test('benign forms render byte-identical output (regression guard)', () => {
    setArtifacts(makeFixtureArtifacts());
    const sidebarRenderer = createSidebarRenderer(makeCallbacks());
    sidebarRenderer.renderSidebar();

    const visSection = document.querySelector('.tree-section[data-type="plots"]');
    expect(visSection).not.toBeNull();
    if (!(visSection instanceof HTMLElement))
      throw new Error('unreachable: asserted non-null above');
    expect(visSection.dataset.type).toBe('plots');
    expect(qs(visSection, '.tree-section-header').dataset.type).toBe('plots');
    // The serialized markup is also byte-identical for benign values —
    // escaping must introduce no stray entities.
    expect(visSection.outerHTML).toContain('data-type="plots"');
  });
});

describe('isNearListEnd', () => {
  test('is true at the exact bottom', () => {
    expect(isNearListEnd({ scrollHeight: 1000, scrollTop: 600, clientHeight: 400 })).toBe(true);
  });

  test('is true within the threshold', () => {
    expect(isNearListEnd({ scrollHeight: 1000, scrollTop: 450, clientHeight: 400 })).toBe(true);
  });

  test('is false above the threshold', () => {
    expect(isNearListEnd({ scrollHeight: 1000, scrollTop: 399, clientHeight: 400 })).toBe(false);
  });

  test('honours a caller-supplied threshold', () => {
    const el = { scrollHeight: 1000, scrollTop: 450, clientHeight: 400 };
    expect(isNearListEnd(el, 100)).toBe(false);
    expect(isNearListEnd(el, 150)).toBe(true);
  });
});

describe('picking several rows', () => {
  /**
   * @param {string} id
   * @param {MouseEventInit} [init]
   */
  function clickRow(id, init = {}) {
    qs(document, `[data-id="${id}"]`).dispatchEvent(
      new MouseEvent('click', { bubbles: true, ...init })
    );
  }

  /** @returns {string[]} ids of the rows carrying the `picked` class, in DOM order */
  function pickedRows() {
    return Array.from(document.querySelectorAll('#sidebar-body .picked')).map(
      (el) => /** @type {HTMLElement} */ (el).dataset.id || ''
    );
  }

  /** @param {any} callbacks */
  function renderer(callbacks) {
    const r = createSidebarRenderer(callbacks);
    r.renderSidebar();
    return r;
  }

  describe('pickRange', () => {
    const order = ['a', 'b', 'c', 'd', 'e'];

    test('returns the inclusive run from the anchor down to the target', () => {
      expect(pickRange(order, 'b', 'd')).toEqual(['b', 'c', 'd']);
    });

    test('returns the inclusive run from the anchor up to the target, in list order', () => {
      expect(pickRange(order, 'd', 'b')).toEqual(['b', 'c', 'd']);
    });

    test('returns just the target when it is the anchor', () => {
      expect(pickRange(order, 'c', 'c')).toEqual(['c']);
    });

    test('returns just the target when the anchor is not in the list', () => {
      expect(pickRange(order, 'zz', 'c')).toEqual(['c']);
    });
  });

  test('in tree mode a Shift-click picks the run in DOM order across section boundaries', () => {
    // Tree order: plots [3, 1], documents [4], data [2].
    setArtifacts(makeFixtureArtifacts());
    renderer(makeCallbacks());

    clickRow('1');
    clickRow('2', { shiftKey: true });

    expect([...getPickedIds()]).toEqual(['1', '4', '2']);
    expect(pickedRows()).toEqual(['1', '4', '2']);
  });

  test('a Shift-click skips rows inside a collapsed section', () => {
    // Plots [3, 1], documents [4], data [2].
    setArtifacts(makeFixtureArtifacts());
    renderer(makeCallbacks());

    qs(document, '.tree-section[data-type="data"] .tree-section-header').click();
    clickRow('3');
    clickRow('4', { shiftKey: true });

    expect([...getPickedIds()]).toEqual(['3', '1', '4']);
  });

  test('a Shift-click with no anchor starts the run at the selected row', () => {
    setArtifacts(makeFixtureArtifacts());
    setSelectedArtifact(makeFixtureArtifacts()[1]); // id 2
    renderer(makeCallbacks());

    clickRow('4', { shiftKey: true });
    expect([...getPickedIds()]).toEqual(['4', '2']);
  });

  test('a Shift-click keeps the anchor, so a second one re-picks the run from the same row', () => {
    setArtifacts(makeFixtureArtifacts());
    renderer(makeCallbacks());

    clickRow('4');
    clickRow('3', { shiftKey: true });
    expect([...getPickedIds()]).toEqual(['3', '1', '4']);
    clickRow('2', { shiftKey: true });
    expect([...getPickedIds()]).toEqual(['4', '2']);
    expect(getPickAnchor()).toBe('4');
  });

  test.each([['metaKey'], ['ctrlKey']])(
    'a %s-click toggles one row and moves the anchor',
    (key) => {
      setArtifacts(makeFixtureArtifacts());
      renderer(makeCallbacks());

      clickRow('2');
      clickRow('3', { [key]: true });
      expect([...getPickedIds()].sort()).toEqual(['2', '3']);
      expect(getPickAnchor()).toBe('3');

      clickRow('1', { [key]: true });
      expect([...getPickedIds()].sort()).toEqual(['1', '2', '3']);

      clickRow('3', { [key]: true });
      expect([...getPickedIds()].sort()).toEqual(['1', '2']);
      expect(getPickAnchor()).toBe('3');
      expect(pickedRows().sort()).toEqual(['1', '2']);
    }
  );

  test('a plain click clears the picks and sets the anchor', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    renderer(callbacks);

    clickRow('1');
    clickRow('2', { shiftKey: true });
    expect(getPickedIds().size).toBe(3);

    clickRow('3');
    expect(getPickedIds().size).toBe(0);
    expect(getPickAnchor()).toBe('3');
    expect(pickedRows()).toEqual([]);
    expect(qs(document, '[data-id="3"]').classList.contains('selected')).toBe(true);
  });

  test('picked rows carry the class `picked` and aria-selected="true"; others carry neither', () => {
    setArtifacts(makeFixtureArtifacts());
    renderer(makeCallbacks());

    clickRow('2');
    clickRow('4', { shiftKey: true });

    const picked = qs(document, '[data-id="4"]');
    expect(picked.classList.contains('picked')).toBe(true);
    expect(picked.getAttribute('aria-selected')).toBe('true');
    const other = qs(document, '[data-id="1"]');
    expect(other.classList.contains('picked')).toBe(false);
    expect(other.getAttribute('aria-selected')).toBeNull();
  });

  test('a Shift- or Cmd/Ctrl-click calls neither onSelect nor onPreviewNeeded', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = makeCallbacks();
    renderer(callbacks);

    clickRow('2');
    expect(callbacks.onSelect).toHaveBeenCalledTimes(1);
    expect(callbacks.onPreviewNeeded).toHaveBeenCalledTimes(1);

    clickRow('3', { shiftKey: true });
    clickRow('1', { metaKey: true });
    clickRow('4', { ctrlKey: true });

    expect(callbacks.onSelect).toHaveBeenCalledTimes(1);
    expect(callbacks.onPreviewNeeded).toHaveBeenCalledTimes(1);
    expect(qs(document, '[data-id="2"]').classList.contains('selected')).toBe(true);
  });

  test('every pick change and every render calls onPicksChanged', () => {
    setArtifacts(makeFixtureArtifacts());
    const callbacks = { ...makeCallbacks(), onPicksChanged: vi.fn() };
    renderer(callbacks);
    expect(callbacks.onPicksChanged).toHaveBeenCalledTimes(1);

    clickRow('2');
    clickRow('3', { shiftKey: true });
    clickRow('1', { metaKey: true });
    expect(callbacks.onPicksChanged).toHaveBeenCalledTimes(4);
  });

  test('the picks survive a renderSidebar() for rows still shown, and drop rows no longer shown', () => {
    setArtifacts(makeFixtureArtifacts());
    const r = renderer(makeCallbacks());

    clickRow('1');
    clickRow('2', { shiftKey: true });
    r.renderSidebar();
    expect(pickedRows()).toEqual(['1', '4', '2']);

    setArtifacts(makeFixtureArtifacts().filter((a) => a.id !== '4'));
    r.renderSidebar();
    expect([...getPickedIds()]).toEqual(['1', '2']);
    expect(pickedRows()).toEqual(['1', '2']);
  });

  test('the same run works in activity mode', () => {
    stampFacilityZone('UTC');
    try {
      setArtifacts(makeFixtureArtifacts());
      const r = createSidebarRenderer(makeCallbacks());
      r.setBrowseMode('activity');
      r.renderSidebar();

      const order = Array.from(document.querySelectorAll('.timeline-item')).map(
        (el) => /** @type {HTMLElement} */ (el).dataset.id
      );
      clickRow(/** @type {string} */ (order[0]));
      clickRow(/** @type {string} */ (order[2]), { shiftKey: true });

      expect([...getPickedIds()]).toEqual(order.slice(0, 3));
      expect(pickedRows()).toEqual(order.slice(0, 3));
      expect(document.querySelectorAll('.timeline-item.picked').length).toBe(3);
    } finally {
      stampFacilityZone(null);
    }
  });

  test('the same run works in the gallery card layout', () => {
    setArtifacts(makeFixtureArtifacts());
    const r = createSidebarRenderer(makeCallbacks());
    r.setSidebarLayout('gallery');
    r.renderSidebar();

    clickRow('1');
    clickRow('2', { shiftKey: true });

    expect([...getPickedIds()]).toEqual(['1', '4', '2']);
    expect(document.querySelectorAll('.gallery-card.picked').length).toBe(3);
  });

  test('picked rows render with the class after a fresh render', () => {
    setArtifacts(makeFixtureArtifacts());
    setPicked(['1', '3']);
    renderer(makeCallbacks());
    expect(pickedRows()).toEqual(['3', '1']);
  });
});

describe('reveal stagger with many rows (#924)', () => {
  /** @param {number} n */
  function manyArtifacts(n) {
    return Array.from({ length: n }, (_, i) => ({
      id: String(i + 1),
      title: `Paged Artifact ${String(i).padStart(4, '0')}`,
      filename: `paged-${i}.md`,
      artifact_type: 'markdown',
      form: i % 3 ? 'documents' : 'plots',
      timestamp: `2026-07-01T10:${String(i % 60).padStart(2, '0')}:00Z`,
      size_bytes: 10,
    }));
  }

  test('the per-row delay is staggered over the first rows only, then flat', () => {
    expect(revealDelayMs(0)).toBe(0);
    expect(revealDelayMs(1)).toBe(30);
    expect(revealDelayMs(12)).toBe(360);
    expect(revealDelayMs(13)).toBe(360);
    expect(revealDelayMs(4000)).toBe(360);
    expect(revealDelayMs(4000, 25)).toBe(300);
  });

  test('the deepest row of a large render is not held invisible for the length of the list', () => {
    mountFixture();
    setArtifacts(manyArtifacts(500));
    const renderer = createSidebarRenderer(makeCallbacks());
    renderer.renderSidebar();
    const rows = document.querySelectorAll('.tree-item');
    expect(rows.length).toBe(500);
    const delays = [...rows].map((row) =>
      parseInt((row.getAttribute('style') ?? '').match(/animation-delay: (\d+)ms/)?.[1] ?? '-1', 10)
    );
    expect(delays[0]).toBe(0);
    expect(Math.max(...delays)).toBe(revealDelayMs(499));
    expect(Math.max(...delays)).toBeLessThanOrEqual(360);
  });

  test('row handlers are installed once on the container, so a rebuilt row selects once', () => {
    mountFixture();
    setArtifacts(manyArtifacts(30));
    const onSelect = vi.fn();
    const renderer = createSidebarRenderer({ ...makeCallbacks(), onSelect });
    renderer.renderSidebar();
    renderer.renderSidebar();
    renderer.renderSidebar();
    qs(document, '.tree-item[data-id="7"]').click();
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0].id).toBe('7');
  });
});

describe('tree mode groups by form, runs expand to their parts (shared registry fixture)', () => {
  /** Every registry label that names a file format or a legacy category. */
  const FORMAT_LABELS = [
    ...Object.values(REGISTRY.artifact_types),
    ...Object.values(REGISTRY.categories),
  ].map((/** @type {any} */ t) => t.label);

  /**
   * @param {string} id
   * @param {object} extra
   * @returns {any}
   */
  function entry(id, extra) {
    return {
      id,
      title: `T ${id}`,
      filename: `${id}.bin`,
      artifact_type: 'json',
      timestamp: `2026-07-0${(Number(id.replace(/\D/g, '')) % 9) + 1}T10:00:00Z`,
      size_bytes: 10,
      ...extra,
    };
  }

  /** @returns {string[]} the section headers' label texts, in order */
  function sectionLabels() {
    return Array.from(document.querySelectorAll('.tree-section-header > span:not([class])')).map(
      (el) => (el.textContent || '').trim()
    );
  }

  /** @param {string} type @returns {string[]} */
  function sectionIds(type) {
    return Array.from(
      document.querySelectorAll(`.tree-section[data-type="${type}"] .tree-item`)
    ).map((el) => /** @type {HTMLElement} */ (el).dataset.id || '');
  }

  const RUN = {
    head: entry('h1', {
      title: 'Orbit correction',
      form: 'runs',
      form_label: 'Runs',
      exec_id: 'e1',
      run_part: 'result',
      run_status: 'succeeded',
      run_parts: [
        { id: 'p1', role: 'preview', title: 'Review: orbit', form: 'documents' },
        { id: 'n1', role: 'code', title: 'Notebook: orbit', form: 'documents' },
        { id: 'o1', role: 'output', title: 'Figure: orbit', form: 'plots' },
      ],
    }),
    preview: entry('p1', {
      artifact_type: 'markdown',
      form: null,
      form_label: null,
      exec_id: 'e1',
      run_part: 'preview',
    }),
    code: entry('n1', {
      artifact_type: 'notebook',
      form: null,
      form_label: null,
      exec_id: 'e1',
      run_part: 'code',
    }),
    output: entry('o1', {
      artifact_type: 'plot_html',
      form: 'plots',
      form_label: 'Plots',
      exec_id: 'e1',
      run_part: 'output',
    }),
  };
  const runEntries = () => [RUN.head, RUN.preview, RUN.code, RUN.output];

  test('groups follow the base forms in order, then extension types, labelled by form, never by format', () => {
    setArtifacts([
      entry('d1', { form: 'data', form_label: 'Data' }),
      entry('b1', { artifact_type: 'table_html', form: 'beam_log', form_label: 'Beam Log' }),
      entry('b2', { artifact_type: 'table_html', form: 'beam_log', form_label: 'Beam Log' }),
      entry('b3', { artifact_type: 'table_html', form: 'beam_log', form_label: 'Beam Log' }),
      entry('i1', { artifact_type: 'image', form: 'images', form_label: 'Images' }),
      entry('s1', {
        artifact_type: 'dashboard_html',
        form: 'dashboards',
        form_label: 'Dashboards',
      }),
      entry('m1', { artifact_type: 'markdown', form: 'documents', form_label: 'Documents' }),
      entry('t1', { artifact_type: 'table_html', form: 'tables', form_label: 'Tables' }),
      entry('a1', { form: 'trends', form_label: 'Trends', category: 'archiver_data' }),
      entry('f1', { artifact_type: 'plot_html', form: 'orbit_fit', form_label: 'Orbit Fit' }),
      entry('p2', { artifact_type: 'plot_png', form: 'plots', form_label: 'Plots' }),
      ...runEntries(),
    ]);
    createSidebarRenderer(makeCallbacks()).renderSidebar();

    const order = Array.from(document.querySelectorAll('#sidebar-body .tree-section')).map(
      (el) => /** @type {HTMLElement} */ (el).dataset.type
    );
    // Base forms in BASE_FORMS order, then the registry's extension types in
    // its order -- the most populous group (beam_log, 3) does not jump ahead.
    expect(order).toEqual([
      'runs',
      'plots',
      'trends',
      'tables',
      'documents',
      'dashboards',
      'images',
      'data',
      'beam_log',
      'orbit_fit',
    ]);
    const labels = sectionLabels();
    expect(labels).toEqual([
      'Runs',
      'Plots',
      'Trends',
      'Tables',
      'Documents',
      'Dashboards',
      'Images',
      'Data',
      'Beam Log',
      'Orbit Fit',
    ]);
    for (const label of labels) expect(FORMAT_LABELS).not.toContain(label);
  });

  test('a run lists once, as its head under Runs; its preview and code list only under it, its output also under its form', () => {
    setArtifacts(runEntries());
    createSidebarRenderer(makeCallbacks()).renderSidebar();

    expect(sectionIds('runs')).toEqual(['h1']);
    expect(qs(document, '.tree-section[data-type="runs"] .tree-section-count').textContent).toBe(
      '1'
    );
    expect(sectionIds('plots')).toEqual(['o1']);
    expect(document.querySelector('.tree-item[data-id="p1"]')).toBeNull();
    expect(document.querySelector('.tree-item[data-id="n1"]')).toBeNull();
    expect(document.querySelector('.tree-section[data-type="run_part"]')).toBeNull();
    const head = qs(document, '.tree-item[data-id="h1"]');
    expect(head.getAttribute('aria-expanded')).toBe('false');
  });

  test('the head expands to its parts with the sidebar labels, a part selects, and the head folds again', () => {
    setArtifacts(runEntries());
    const callbacks = makeCallbacks();
    const sidebar = createSidebarRenderer(callbacks);
    sidebar.renderSidebar();

    qs(document, '.tree-item[data-id="h1"] .tree-run-toggle').click();
    expect(callbacks.onSelect).not.toHaveBeenCalled();
    expect(qs(document, '.tree-item[data-id="h1"]').getAttribute('aria-expanded')).toBe('true');

    const parts = Array.from(document.querySelectorAll('.tree-run-part'));
    expect(parts.map((el) => /** @type {HTMLElement} */ (el).dataset.id)).toEqual([
      'o1',
      'p1',
      'n1',
    ]);
    expect(parts.map((el) => qs(el, '.tree-item-name').textContent)).toEqual([
      'Figure: orbit',
      'Approval preview',
      'Code (notebook)',
    ]);
    for (const el of parts) expect(/** @type {HTMLElement} */ (el).dataset.runHead).toBe('h1');

    // Expansion survives a rebuild (a new artifact arriving).
    sidebar.renderSidebar();
    expect(document.querySelectorAll('.tree-run-part').length).toBe(3);

    /** @type {HTMLElement} */ (qs(document, '.tree-run-part[data-id="p1"]')).click();
    expect(callbacks.onSelect).toHaveBeenCalledTimes(1);
    expect(callbacks.onSelect.mock.calls[0][0].id).toBe('p1');

    qs(document, '.tree-item[data-id="h1"] .tree-run-toggle').click();
    expect(document.querySelectorAll('.tree-run-part').length).toBe(0);
  });

  test.each([
    ['succeeded', 'succeeded', null],
    ['failed', 'failed', 'failed'],
    ['not_run', 'not_run', 'not run'],
    [null, 'running', 'running'],
  ])('a head with run_status %s is marked %s', (runStatus, key, text) => {
    setArtifacts([{ ...RUN.head, run_status: runStatus, run_parts: [] }]);
    createSidebarRenderer(makeCallbacks()).renderSidebar();

    const head = qs(document, '.tree-item[data-id="h1"]');
    expect(head.dataset.runStatus).toBe(key);
    expect(head.classList.contains(`run-status-${key}`)).toBe(true);
    const badge = head.querySelector('.tree-item-badge.run-status');
    expect(badge ? badge.textContent : null).toBe(text);
    // No parts: nothing to expand.
    expect(head.querySelector('.tree-run-toggle')).toBeNull();
  });

  test('an in-flight run headed by its preview (no result yet) lists under Runs and expands to its code', () => {
    setArtifacts([
      entry('p9', {
        artifact_type: 'markdown',
        title: 'Review: pending',
        form: 'runs',
        form_label: 'Runs',
        exec_id: 'e9',
        run_part: 'preview',
        run_status: null,
        run_parts: [{ id: 'n9', role: 'code', title: 'Notebook: pending', form: 'documents' }],
      }),
      entry('n9', { artifact_type: 'notebook', form: null, exec_id: 'e9', run_part: 'code' }),
    ]);
    createSidebarRenderer(makeCallbacks()).renderSidebar();

    expect(sectionIds('runs')).toEqual(['p9']);
    expect(qs(document, '.tree-item[data-id="p9"]').dataset.runStatus).toBe('running');
    qs(document, '.tree-item[data-id="p9"] .tree-run-toggle').click();
    expect(
      Array.from(document.querySelectorAll('.tree-run-part')).map(
        (el) => /** @type {HTMLElement} */ (el).dataset.id
      )
    ).toEqual(['n9']);
  });

  test('a part whose head is not shown lists under Runs rather than vanishing', () => {
    setArtifacts([RUN.preview]);
    createSidebarRenderer(makeCallbacks()).renderSidebar();

    expect(sectionIds('runs')).toEqual(['p1']);
  });

  test('card and timeline badges read form_label (a run part reads its role), never a format', () => {
    setArtifacts([
      entry('x1', { artifact_type: 'plot_html', form: 'plots', form_label: 'Plots' }),
      entry('x2', { artifact_type: 'table_html', form: 'beam_log', form_label: 'Beam Log' }),
      RUN.preview,
    ]);
    const sidebar = createSidebarRenderer(makeCallbacks());
    sidebar.setSidebarLayout('gallery');
    sidebar.setBrowseMode('activity');
    sidebar.renderSidebar();

    /** @param {string} id */
    const cardBadge = (id) =>
      qs(document, `.gallery-card[data-id="${id}"] .gallery-card-type`).textContent;
    expect(cardBadge('x1')).toBe('Plots');
    expect(cardBadge('x2')).toBe('Beam Log');
    expect(cardBadge('p1')).toBe('Approval preview');
    expect(qs(document, '.gallery-card[data-id="x1"]').dataset.type).toBe('plots');

    sidebar.setSidebarLayout('list');
    sidebar.renderSidebar();
    expect(qs(document, '.timeline-item[data-id="x1"] .timeline-item-type').textContent).toBe(
      'Plots'
    );
  });

  test('a run head card in the gallery layout expands to its parts too', () => {
    setArtifacts(runEntries());
    const sidebar = createSidebarRenderer(makeCallbacks());
    sidebar.setSidebarLayout('gallery');
    sidebar.renderSidebar();

    const toggle = qs(document, '.gallery-card[data-id="h1"] .tree-run-toggle');
    expect(toggle.textContent).toBe('3');
    toggle.click();
    expect(document.querySelectorAll('.tree-run-part').length).toBe(3);
  });
});
