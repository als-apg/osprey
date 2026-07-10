// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Consolidated hostile-metadata security regression suite (Phase 4, Task
 * 1.5) — the machine-checkable gate proving Phase 1's security fixes
 * (1.1 quote-safe-canonical-escaper, 1.2 consolidate-artifacts-escapers,
 * 1.3 escape-metadata-sinks, 1.4 percent-encode-url-id-sinks) hold across
 * every render path that touches agent-supplied artifact metadata, and
 * cannot silently regress.
 *
 * Unlike the per-module suites (render.test.mjs, preview.test.mjs,
 * types.test.mjs, timeseries.test.mjs), each of which owns one module's
 * "XSS hardening" describe for its own sinks, this file drives the RENDER,
 * PREVIEW, TYPES, TIMESERIES, and LOGBOOK-PICKER paths together with one
 * broader hostile-fixture set, and pins the already-escaped fields
 * (description/tool_source/filename) against a future unescaping
 * regression. A few core assertions unavoidably overlap the per-module
 * suites; that overlap is intentional (this is the cross-path contract).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/security_render.test.mjs
 *
 * Conventions mirror the sibling suites (no vi.resetModules; state.js
 * setters reseed the module-singleton artifact list per test). This file
 * deliberately does NOT vi.mock logbook.js/print.js (unlike
 * preview.test.mjs) — the real, unmocked implementations are safe no-op/
 * button-appenders against these fixtures, and Task 1.5 item 5 requires
 * driving the real artifact-picker id sink (logbook.js's `.value` DOM
 * property assignment, the Task 1.4 attribute-sink fix) through the actual
 * render pipeline in this same file, which would be impossible if the
 * module were mocked here.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  setArtifacts,
  setSelectedArtifact,
  setActiveFilter,
  setFocusedArtifact,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';
import {
  createSidebarRenderer,
} from '../../../src/osprey/interfaces/artifacts/static/js/render.js';
import {
  initTypeRegistry,
  typeBadge,
  thumbnailHtml,
  openUrl,
} from '../../../src/osprey/interfaces/artifacts/static/js/types.js';
import { createPreviewRenderer } from '../../../src/osprey/interfaces/artifacts/static/js/preview.js';
import {
  renderTimeseriesTable,
  renderTimeseriesView,
} from '../../../src/osprey/interfaces/artifacts/static/js/timeseries.js';
import '../../../src/osprey/interfaces/artifacts/static/js/logbook.js';

// ---- Shared hostile payload set ---- //

/**
 * Double/single-quote attribute-breakout + live-element-injection payloads.
 * Every one of these MUST be broken (never appear verbatim) in any
 * serialized HTML this suite inspects, because escaping at least one of
 * their `"`/`'`/`<`/`>` characters is required to render them inert.
 */
const HOSTILE = {
  DQ_IMG: '"><img src=x onerror=alert(1)>',
  DQ_ATTR: '" onmouseover=alert(1) x="',
  SQ_IMG: "'><img src=x onerror=alert(1)>",
};

/**
 * javascript: URL-scheme payload. Contains no `"`/`<`/`>`/`'`/`&`, so
 * escapeHtml leaves it byte-identical — the invariant it must satisfy is
 * different: it must only ever land in a text/attribute-value sink, never
 * become a live `href`/`src`.
 */
const JS_SCHEME = 'javascript:alert(1)';

/** Hostile artifact id for URL path-segment sinks (percent-encoding contract, Task 1.4). */
const HOSTILE_ID = 'a/../b?x="y"';

/** Hostile artifact id for the logbook picker's attribute-context id sink (Task 1.4). */
const HOSTILE_PICKER_ID = '"><input x="';

/** No element anywhere under `root` has live injected markup or a bare event-handler attribute. */
function expectNoLiveInjection(root) {
  expect(root.querySelector('img[onerror]')).toBeNull();
  expect(root.querySelector('[onmouseover]')).toBeNull();
}

// ---- Shared DOM fixtures / lifecycle helpers ---- //

/** Preview-pane DOM skeleton used by both the PREVIEW and LOGBOOK-picker paths. */
function mountPreviewFixture() {
  document.body.className = '';
  document.body.innerHTML = `
    <aside class="browse-sidebar" id="browse-sidebar">
      <div class="sidebar-body" id="sidebar-body"></div>
    </aside>
    <div class="browse-preview-pane" id="browse-preview-pane">
      <div class="preview-empty" id="preview-empty"></div>
      <div class="preview-content hidden" id="preview-content"></div>
    </div>
  `;
}

function makePreviewCallbacks() {
  return {
    onArtifactDeleted: vi.fn(),
    onPinToggled: vi.fn(),
    onFullscreenExit: vi.fn(),
    onTimeseriesNeeded: vi.fn(),
  };
}

/**
 * Reset the module-level type registry so a test that re-inits it with a
 * hostile label can't leak into a later test, even on assertion failure.
 * For use in afterEach of any describe that touches initTypeRegistry.
 */
async function resetTypeRegistry() {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: () => Promise.resolve({}) }));
  await initTypeRegistry();
  vi.unstubAllGlobals();
}

// =========================================================================
// RENDER path — render.js's sidebar (tree / activity / gallery layouts) +
// filter-bar chip labels.
// =========================================================================

describe('RENDER path (render.js) — hostile metadata in sidebar renders', () => {
  function mountSidebarFixture() {
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
    `;
  }

  function makeSidebarCallbacks() {
    return { onSelect: vi.fn(), onPreviewNeeded: vi.fn(), onEnterFullscreen: vi.fn() };
  }

  /** @param {string} payload */
  function makeHostileArtifact(payload) {
    return [{
      id: 'r1',
      title: payload,
      filename: 'x.png',
      artifact_type: payload,
      category: payload,
      pinned: false,
      timestamp: '2026-07-04T10:00:00Z',
      size_bytes: 10,
    }];
  }

  beforeEach(() => {
    mountSidebarFixture();
    setActiveFilter('all');
    setSelectedArtifact(null);
  });

  afterEach(resetTypeRegistry);

  test.each(Object.entries(HOSTILE))(
    'tree mode: a hostile %s payload in title/category/artifact_type renders inert',
    (_name, payload) => {
      setArtifacts(makeHostileArtifact(payload));
      const renderer = createSidebarRenderer(makeSidebarCallbacks());
      renderer.renderSidebar(); // default: tree mode

      const sidebarBody = document.getElementById('sidebar-body');
      // The real security property: no live element/attribute got injected.
      // (A raw-substring check on the serialized HTML is NOT a reliable
      // proxy for this — HTML text-node serialization is not required to
      // re-escape `"`/`'`, only `&`/`<`/`>`, since quotes carry no special
      // meaning outside an attribute-value context; a quote-only payload
      // can legitimately reappear verbatim in TEXT content while still
      // being fully inert.)
      expectNoLiveInjection(sidebarBody);
      // Exactly one section/item exists — an attribute breakout that split
      // the hostile string into new attributes/elements would change this.
      expect(sidebarBody.querySelectorAll('.tree-section').length).toBe(1);
      expect(sidebarBody.querySelectorAll('.tree-item').length).toBe(1);

      const section = sidebarBody.querySelector('.tree-section');
      // Round-trip proof that the WHOLE payload landed intact as ONE
      // attribute value (an early-closed `"` would truncate this).
      expect(section.dataset.type).toBe(payload);
      expect(sidebarBody.querySelector('.tree-item-name').textContent).toBe(payload);
    }
  );

  test.each(Object.entries(HOSTILE))(
    'activity mode: a hostile %s payload in category/artifact_type renders inert in the timeline-item',
    (_name, payload) => {
      setArtifacts(makeHostileArtifact(payload));
      const renderer = createSidebarRenderer(makeSidebarCallbacks());
      renderer.setBrowseMode('activity');
      renderer.renderSidebar();

      const sidebarBody = document.getElementById('sidebar-body');
      expectNoLiveInjection(sidebarBody);
      expect(sidebarBody.querySelectorAll('.timeline-item').length).toBe(1);

      const item = sidebarBody.querySelector('.timeline-item');
      expect(item.dataset.type).toBe(payload);
      // The title markup interleaves a conditional pin-indicator span
      // around the escaped title, so the node carries surrounding
      // whitespace — trim before the round-trip comparison.
      expect(item.querySelector('.timeline-item-title').textContent.trim()).toBe(payload);
    }
  );

  test.each(Object.entries(HOSTILE))(
    'gallery layout: a hostile %s payload in category/artifact_type renders inert in the gallery-card',
    (_name, payload) => {
      setArtifacts(makeHostileArtifact(payload));
      const renderer = createSidebarRenderer(makeSidebarCallbacks());
      renderer.setSidebarLayout('gallery');
      renderer.renderSidebar();

      const sidebarBody = document.getElementById('sidebar-body');
      expectNoLiveInjection(sidebarBody);
      expect(sidebarBody.querySelectorAll('.gallery-card').length).toBe(1);

      const card = sidebarBody.querySelector('.gallery-card');
      expect(card.dataset.type).toBe(payload);
      // Same interleaved pin-indicator layout as the timeline-item title.
      expect(card.querySelector('.gallery-card-title').textContent.trim()).toBe(payload);
    }
  );

  test('initFilterBar: a hostile registry label is escaped as text in the chip innerHTML sink, typeIcon SVG survives', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ categories: { evilchip: { label: HOSTILE.DQ_ATTR } } }),
    }));
    await initTypeRegistry();

    setArtifacts([
      { id: 'c1', title: 'Chip Test', filename: 'c.png', artifact_type: 'evilchip', category: 'evilchip', pinned: false, timestamp: '2026-07-04T10:00:00Z', size_bytes: 10 },
    ]);
    const renderer = createSidebarRenderer(makeSidebarCallbacks());
    renderer.initFilterBar();

    const chip = document.querySelector('.filter-chip[data-filter="evilchip"]');
    expect(chip).not.toBeNull();
    expectNoLiveInjection(document.body);
    expect(chip.querySelector('.chip-icon svg')).not.toBeNull();
  });
});

// =========================================================================
// PREVIEW path — preview.js's renderPreview: badge class, title,
// description, tool_source, filename, and id URL sinks.
// =========================================================================

describe('PREVIEW path (preview.js) — hostile metadata in the preview pane', () => {
  /** @returns {any} */
  function makeArtifact(overrides = {}) {
    return {
      id: 'a1',
      title: 'Beam Profile',
      filename: 'beam_profile.png',
      artifact_type: 'plot_png',
      category: 'visualization',
      pinned: false,
      timestamp: '2026-07-01T10:00:00Z',
      size_bytes: 2048,
      ...overrides,
    };
  }

  beforeEach(() => {
    mountPreviewFixture();
    setArtifacts([]);
    setSelectedArtifact(null);
    setFocusedArtifact(null);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  test.each(Object.entries(HOSTILE))(
    'a hostile %s payload in title/category/artifact_type is inert in the header badge + title',
    (_name, payload) => {
      setSelectedArtifact(makeArtifact({ title: payload, category: payload, artifact_type: payload }));
      createPreviewRenderer(makePreviewCallbacks()).renderPreview();

      const previewContent = document.getElementById('preview-content');
      expectNoLiveInjection(previewContent);
      expect(previewContent.querySelectorAll('.badge').length).toBe(1);

      const badge = previewContent.querySelector('.badge');
      // Round-trip proof the whole payload landed intact as ONE class
      // attribute value (an early-closed `"` would truncate/split this).
      expect(badge.className).toBe(`badge badge-${payload}`);
      expect(previewContent.querySelector('.preview-header-title').textContent).toBe(payload);
    }
  );

  test('description (currently-escaped field) renders as inert text — regression pin against future unescaping', () => {
    // artifact_type falls through to the plain download-link branch (no
    // mime_type, no recognized type) so renderPreview never fires a real
    // network fetch (unlike "json"/"markdown", which load their own content).
    setSelectedArtifact(makeArtifact({ artifact_type: 'totally_unrecognized_type', category: 'json', description: HOSTILE.DQ_IMG }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = document.getElementById('preview-content');
    expectNoLiveInjection(previewContent);
    expect(previewContent.innerHTML).not.toContain(HOSTILE.DQ_IMG);

    const desc = previewContent.querySelector('.preview-desc');
    expect(desc).not.toBeNull();
    expect(desc.textContent).toBe(HOSTILE.DQ_IMG);
  });

  test('tool_source (currently-escaped field) renders as inert text — regression pin against future unescaping', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'totally_unrecognized_type', category: 'json', tool_source: HOSTILE.DQ_IMG }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = document.getElementById('preview-content');
    expectNoLiveInjection(previewContent);
    expect(previewContent.innerHTML).not.toContain(HOSTILE.DQ_IMG);

    const metaValues = Array.from(previewContent.querySelectorAll('.preview-meta-value')).map((el) => el.textContent);
    expect(metaValues).toContain(HOSTILE.DQ_IMG);
  });

  test('filename (currently-escaped field) renders as inert text at both the download link and copy-path sinks — regression pin', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'totally_unrecognized_type', category: 'json', filename: HOSTILE.DQ_IMG }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = document.getElementById('preview-content');
    expectNoLiveInjection(previewContent);
    expect(previewContent.innerHTML).not.toContain(HOSTILE.DQ_IMG);

    const downloadLink = previewContent.querySelector('.preview-download a');
    expect(downloadLink).not.toBeNull();
    expect(downloadLink.textContent).toContain(HOSTILE.DQ_IMG);

    const pathText = previewContent.querySelector('.preview-path-text');
    expect(pathText.textContent).toBe(`_agent_data/artifacts/${HOSTILE.DQ_IMG}`);
  });

  test('a hostile id is percent-encoded at both the "open in new tab" href and the download-link href', () => {
    setSelectedArtifact(makeArtifact({ id: HOSTILE_ID, artifact_type: 'totally_unrecognized_type', category: 'json', filename: 'x.bin' }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = document.getElementById('preview-content');
    const expectedUrl = `/files/${encodeURIComponent(HOSTILE_ID)}/${encodeURIComponent('x.bin')}`;

    const openInNewTab = previewContent.querySelector('.preview-header-actions a[target="_blank"]');
    expect(openInNewTab.getAttribute('href')).toBe(expectedUrl);
    expect(openInNewTab.getAttribute('href')).not.toMatch(/["?]/);

    const downloadLink = previewContent.querySelector('.preview-download a');
    expect(downloadLink.getAttribute('href')).toBe(expectedUrl);
  });

  test('a javascript: scheme payload in the title renders as inert text, never as a live href', () => {
    setSelectedArtifact(makeArtifact({ title: JS_SCHEME }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = document.getElementById('preview-content');
    expect(previewContent.querySelector('.preview-header-title').textContent).toBe(JS_SCHEME);
    expect(previewContent.querySelector('a[href^="javascript:"]')).toBeNull();
  });
});

// =========================================================================
// TYPES path — types.js's typeBadge / thumbnailHtml / openUrl-fileUrl id
// encoding (spot checks; detailed id-encoding assertions live in
// types.test.mjs/state.test.mjs).
// =========================================================================

describe('TYPES path (types.js) — typeBadge / thumbnailHtml / id-encoding', () => {
  afterEach(resetTypeRegistry);

  test('typeBadge escapes a hostile registry label to entity-safe text', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ categories: { evil: { label: HOSTILE.DQ_IMG } } }),
    }));
    await initTypeRegistry();

    const result = typeBadge('evil');
    expect(result).not.toContain(HOSTILE.DQ_IMG);
    expect(result).toContain('&lt;img');
    expect(result).not.toContain('<img');
  });

  test('typeBadge escapes a hostile type string when it falls back to the raw type (registry miss)', () => {
    const result = typeBadge(HOSTILE.SQ_IMG);
    expect(result).not.toContain(HOSTILE.SQ_IMG);
    expect(result).not.toContain('<img');
  });

  test('thumbnailHtml escapes a hostile summary value at the summary-dump sink, no live element', () => {
    const html = thumbnailHtml({
      id: 'th1',
      artifact_type: 'unknown_xyz',
      filename: 'f.bin',
      summary: { note: HOSTILE.DQ_IMG },
    });

    expect(html).not.toContain(HOSTILE.DQ_IMG);

    const container = document.createElement('div');
    container.innerHTML = html;
    expectNoLiveInjection(container);
    expect(container.querySelector('.thumb-summary').textContent).toContain(HOSTILE.DQ_IMG);
  });

  test('openUrl percent-encodes a hostile id at the markdown/notebook rendered-endpoint sinks', () => {
    const mdUrl = openUrl({ id: HOSTILE_ID, artifact_type: 'markdown', filename: 'x.md' });
    expect(mdUrl).toBe(`/api/markdown/${encodeURIComponent(HOSTILE_ID)}/rendered`);
    expect(mdUrl).not.toMatch(/["?]/);

    const nbUrl = openUrl({ id: HOSTILE_ID, artifact_type: 'notebook', filename: 'x.ipynb' });
    expect(nbUrl).toBe(`/api/notebooks/${encodeURIComponent(HOSTILE_ID)}/rendered`);
    expect(nbUrl).not.toMatch(/["?]/);
  });

  test('openUrl percent-encodes a hostile id at the default (fileUrl) sink', () => {
    const url = openUrl({ id: HOSTILE_ID, artifact_type: 'mystery', filename: 'x.bin' });
    expect(url).toBe(`/files/${encodeURIComponent(HOSTILE_ID)}/${encodeURIComponent('x.bin')}`);
    expect(url).not.toMatch(/["?]/);
  });
});

// =========================================================================
// TIMESERIES paths — timeseries.js's renderTimeseriesTable (column-header
// text sink) AND renderTimeseriesView (data-ch-name=/title= attribute
// sinks at the channel-toggle buttons).
// =========================================================================

describe('TIMESERIES paths (timeseries.js) — hostile column name', () => {
  /** Fixture chart-format response (`/api/artifacts/{id}/data?format=chart`). */
  function makeChartData(overrides = {}) {
    return {
      columns: [HOSTILE.DQ_IMG],
      index: ['2026-07-01T00:00:00Z'],
      data: [[1.0]],
      total_rows: 1,
      downsampled: false,
      returned_points: 1,
      ...overrides,
    };
  }

  /** Fixture table-format response (`/api/artifacts/{id}/data?format=table`). */
  function makeTableData(overrides = {}) {
    return {
      columns: [HOSTILE.DQ_IMG],
      index: ['2026-07-01T00:00:00Z'],
      data: [[1.0]],
      total_rows: 1,
      offset: 0,
      limit: 50,
      returned_rows: 1,
      ...overrides,
    };
  }

  /** Mirrors timeseries.test.mjs's stubScriptLoad(): fake-loads an injected `<script>`. */
  function stubScriptLoad() {
    const originalAppendChild = document.head.appendChild.bind(document.head);
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      if (node && node.tagName === 'SCRIPT') {
        queueMicrotask(() => node.onload && node.onload());
        return node;
      }
      return originalAppendChild(node);
    });
  }

  beforeEach(() => {
    vi.stubGlobal('Plotly', {
      newPlot: vi.fn((el) => { el.data = []; }),
      restyle: vi.fn(),
      relayout: vi.fn(),
    });
    stubScriptLoad();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe('renderTimeseriesTable', () => {
    test('a hostile column name renders as inert text in the <th> header cell', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(makeTableData()) }));
      const el = document.createElement('div');

      await renderTimeseriesTable(el, 'ts1', [HOSTILE.DQ_IMG], 0);

      expectNoLiveInjection(el);
      expect(el.innerHTML).not.toContain(HOSTILE.DQ_IMG);
      const th = el.querySelector('thead th:not(:first-child)');
      expect(th.textContent).toBe(HOSTILE.DQ_IMG);
    });
  });

  describe('renderTimeseriesView', () => {
    test('a hostile column name round-trips through the data-ch-name/title attribute sinks with no breakout', async () => {
      vi.stubGlobal('fetch', vi.fn((url) => {
        if (url.includes('format=chart')) return Promise.resolve({ ok: true, json: () => Promise.resolve(makeChartData()) });
        return Promise.resolve({ ok: true, json: () => Promise.resolve(makeTableData()) });
      }));
      const container = document.createElement('div');
      document.body.appendChild(container);

      await renderTimeseriesView(container, { id: 'ts1' });

      expectNoLiveInjection(container);
      expect(container.innerHTML).not.toContain(HOSTILE.DQ_IMG);

      const toggle = container.querySelector('.ts-ch-toggle');
      expect(toggle).not.toBeNull();
      // dataset/getAttribute auto-decode entities: reading back round-trips
      // to the raw hostile column name — what matters is the SERIALIZED
      // markup above never contained the raw breakout.
      expect(toggle.dataset.chName).toBe(HOSTILE.DQ_IMG);
      expect(toggle.getAttribute('title')).toBe(HOSTILE.DQ_IMG);
    });

    test('the info-bar channel badge escapes a hostile column name, no live element', async () => {
      vi.stubGlobal('fetch', vi.fn((url) => {
        if (url.includes('format=chart')) return Promise.resolve({ ok: true, json: () => Promise.resolve(makeChartData({ columns: [HOSTILE.SQ_IMG] })) });
        return Promise.resolve({ ok: true, json: () => Promise.resolve(makeTableData({ columns: [HOSTILE.SQ_IMG] })) });
      }));
      const container = document.createElement('div');
      document.body.appendChild(container);

      await renderTimeseriesView(container, { id: 'ts1' });

      // SQ_IMG contains `<`/`>`/`'` but no `"`; the toggle button's
      // data-ch-name/title attributes are double-quoted, so `'`/`<`/`>`
      // need no re-escaping there to stay inert (matches the dedicated
      // attribute-sink test above) — the property this test locks down is
      // the info-bar's TEXT-content sink, which must still escape `<`/`>`.
      expectNoLiveInjection(container);
      const badge = container.querySelector('.ts-badge-channel');
      expect(badge).not.toBeNull();
      expect(badge.innerHTML).not.toMatch(/<img/);
    });
  });
});

// =========================================================================
// LOGBOOK picker path — the deferred Task 1.4 regression: the artifact-
// picker checkbox id is assigned via the `.value` DOM property, not
// interpolated into innerHTML. Driven through the real (unmocked)
// injectLogbookButtons() export via preview.js's actual render pipeline
// (preview.js calls injectLogbookButtons() directly at the end of
// renderPreview — see preview.js's module doc-comment), then a real
// button click + radio change, mirroring the true user flow.
// =========================================================================

describe('LOGBOOK picker path (logbook.js) — hostile artifact id at the checkbox value sink', () => {
  beforeEach(() => {
    mountPreviewFixture();
    setArtifacts([]);
    setSelectedArtifact(null);
    setFocusedArtifact(null);
  });

  afterEach(() => {
    // Close the modal (if the test opened one) so logbook.js's module-level
    // `modal`/`allArtifacts` singleton state doesn't leak into a later test.
    const closeBtn = document.querySelector('.logbook-modal-close');
    if (closeBtn) closeBtn.click();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  test('a hostile artifact id round-trips via the checkbox .value property, no injected <input>, no unescaped breakout', async () => {
    setSelectedArtifact({
      id: 'current-artifact',
      title: 'Current',
      filename: 'current.png',
      artifact_type: 'plot_png',
      category: 'visualization',
      pinned: false,
      timestamp: '2026-07-01T10:00:00Z',
      size_bytes: 10,
    });
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    // renderPreview() calls the real injectLogbookButtons() internally,
    // which appends the compose-modal trigger to the preview header.
    const logbookBtn = document.querySelector('.logbook-action-btn');
    expect(logbookBtn).not.toBeNull();
    logbookBtn.click(); // opens the compose modal, phase = steering

    const chooseRadio = document.querySelector('input[name="logbook-artifact-scope"][value="choose"]');
    expect(chooseRadio).not.toBeNull();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({
        artifacts: [{ id: HOSTILE_PICKER_ID, title: 'Hostile Artifact', artifact_type: 'json' }],
      }),
    }));

    chooseRadio.checked = true;
    chooseRadio.dispatchEvent(new Event('change', { bubbles: true }));

    // loadArtifactPicker() -> fetch().then(json).then(renderArtifactPicker):
    // two microtask turns to flush the promise chain.
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    const list = document.getElementById('logbook-artifact-picker-list');
    expect(list).not.toBeNull();

    const checkboxes = list.querySelectorAll('input[type=checkbox]');
    expect(checkboxes.length).toBe(1); // no extra <input> injected by the hostile id
    expect(checkboxes[0].value).toBe(HOSTILE_PICKER_ID); // raw round-trip via the .value property

    // The id was assigned as a DOM property, never interpolated into the
    // innerHTML string, so it can never appear as a serialized attribute
    // breakout in the picker's markup.
    expect(list.innerHTML).not.toContain(HOSTILE_PICKER_ID);
  });
});
