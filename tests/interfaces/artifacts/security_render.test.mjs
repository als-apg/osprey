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
 *
 * The MARKDOWN path (Task 6.3) drives the gallery's markdown preview
 * (md-render.js via preview-content.js) with the vendored marked and
 * DOMPurify builds — byte-checked against vendor_manifest.json — so
 * script-bearing agent markdown is proven inert with the real sanitizer.
 * DOMPurify does not sanitise under happy-dom, so it is instantiated on a
 * jsdom window; its output string then lands in this file's happy-dom DOM
 * exactly as the page's innerHTML sink receives it.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { test, expect, describe, beforeAll, beforeEach, afterEach, vi } from 'vitest';

import { qs, byId } from '../_support/dom.mjs';

import {
  setArtifacts,
  setSelectedArtifact,
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
import {
  renderMathInMarkdown,
  renderMarkdownView,
} from '../../../src/osprey/interfaces/artifacts/static/js/preview-content.js';

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

/**
 * No element anywhere under `root` has live injected markup or a bare event-handler attribute.
 * @param {ParentNode} root
 */
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
// tree-section header labels.
// =========================================================================

describe('RENDER path (render.js) — hostile metadata in sidebar renders', () => {
  function mountSidebarFixture() {
    document.body.innerHTML = `
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
    setSelectedArtifact(null);
  });

  afterEach(resetTypeRegistry);

  test.each(Object.entries(HOSTILE))(
    'tree mode: a hostile %s payload in title/category/artifact_type renders inert',
    (_name, payload) => {
      setArtifacts(makeHostileArtifact(payload));
      const renderer = createSidebarRenderer(makeSidebarCallbacks());
      renderer.renderSidebar(); // default: tree mode

      const sidebarBody = byId('sidebar-body');
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

      const section = qs(sidebarBody, '.tree-section');
      // Round-trip proof that the WHOLE payload landed intact as ONE
      // attribute value (an early-closed `"` would truncate this).
      expect(section.dataset.type).toBe(payload);
      expect(qs(sidebarBody, '.tree-item-name').textContent).toBe(payload);
    }
  );

  test.each(Object.entries(HOSTILE))(
    'activity mode: a hostile %s payload in category/artifact_type renders inert in the timeline-item',
    (_name, payload) => {
      setArtifacts(makeHostileArtifact(payload));
      const renderer = createSidebarRenderer(makeSidebarCallbacks());
      renderer.setBrowseMode('activity');
      renderer.renderSidebar();

      const sidebarBody = byId('sidebar-body');
      expectNoLiveInjection(sidebarBody);
      expect(sidebarBody.querySelectorAll('.timeline-item').length).toBe(1);

      const item = qs(sidebarBody, '.timeline-item');
      expect(item.dataset.type).toBe(payload);
      // The title markup interleaves a conditional pin-indicator span
      // around the escaped title, so the node carries surrounding
      // whitespace — trim before the round-trip comparison.
      expect(qs(item, '.timeline-item-title').textContent?.trim()).toBe(payload);
    }
  );

  test.each(Object.entries(HOSTILE))(
    'gallery layout: a hostile %s payload in category/artifact_type renders inert in the gallery-card',
    (_name, payload) => {
      setArtifacts(makeHostileArtifact(payload));
      const renderer = createSidebarRenderer(makeSidebarCallbacks());
      renderer.setSidebarLayout('gallery');
      renderer.renderSidebar();

      const sidebarBody = byId('sidebar-body');
      expectNoLiveInjection(sidebarBody);
      expect(sidebarBody.querySelectorAll('.gallery-card').length).toBe(1);

      const card = qs(sidebarBody, '.gallery-card');
      expect(card.dataset.type).toBe(payload);
      // Same interleaved pin-indicator layout as the timeline-item title.
      expect(qs(card, '.gallery-card-title').textContent?.trim()).toBe(payload);
    }
  );

  test('a hostile registry label is escaped as text in the tree-section header innerHTML sink, typeIcon SVG survives', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ categories: { evilcat: { label: HOSTILE.DQ_ATTR } } }),
    }));
    await initTypeRegistry();

    setArtifacts([
      { id: 'c1', title: 'Section Test', filename: 'c.png', artifact_type: 'evilcat', category: 'evilcat', pinned: false, timestamp: '2026-07-04T10:00:00Z', size_bytes: 10 },
    ]);
    const renderer = createSidebarRenderer(makeSidebarCallbacks());
    renderer.renderSidebar();

    const header = document.querySelector('.tree-section-header[data-type="evilcat"]');
    expect(header).not.toBeNull();
    if (header === null) throw new Error('unreachable: header asserted non-null above');
    expectNoLiveInjection(document.body);
    expect(header.querySelector('.tree-section-icon svg')).not.toBeNull();
  });
});

// =========================================================================
// PREVIEW path — preview.js's renderPreview: badge class, title,
// description, tool_source, filename, and id URL sinks.
// =========================================================================

describe('PREVIEW path (preview.js) — hostile metadata in the preview pane', () => {
  /** @param {Record<string, unknown>} [overrides] */
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

      const previewContent = byId('preview-content');
      expectNoLiveInjection(previewContent);
      expect(previewContent.querySelectorAll('.badge').length).toBe(1);

      const badge = qs(previewContent, '.badge');
      // Round-trip proof the whole payload landed intact as ONE class
      // attribute value (an early-closed `"` would truncate/split this).
      expect(badge.className).toBe(`badge badge-${payload}`);
      expect(qs(previewContent, '.preview-header-title').textContent).toBe(payload);
    }
  );

  test('description (currently-escaped field) renders as inert text — regression pin against future unescaping', () => {
    // artifact_type falls through to the plain download-link branch (no
    // mime_type, no recognized type) so renderPreview never fires a real
    // network fetch (unlike "json"/"markdown", which load their own content).
    setSelectedArtifact(makeArtifact({ artifact_type: 'totally_unrecognized_type', category: 'json', description: HOSTILE.DQ_IMG }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = byId('preview-content');
    expectNoLiveInjection(previewContent);
    expect(previewContent.innerHTML).not.toContain(HOSTILE.DQ_IMG);

    const desc = previewContent.querySelector('.preview-desc');
    expect(desc).not.toBeNull();
    if (desc === null) throw new Error('unreachable: desc asserted non-null above');
    expect(desc.textContent).toBe(HOSTILE.DQ_IMG);
  });

  test('tool_source (currently-escaped field) renders as inert text — regression pin against future unescaping', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'totally_unrecognized_type', category: 'json', tool_source: HOSTILE.DQ_IMG }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = byId('preview-content');
    expectNoLiveInjection(previewContent);
    expect(previewContent.innerHTML).not.toContain(HOSTILE.DQ_IMG);

    const metaValues = Array.from(previewContent.querySelectorAll('.preview-meta-value')).map((el) => el.textContent);
    expect(metaValues).toContain(HOSTILE.DQ_IMG);
  });

  test('filename (currently-escaped field) renders as inert text at both the download link and copy-path sinks — regression pin', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'totally_unrecognized_type', category: 'json', filename: HOSTILE.DQ_IMG }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = byId('preview-content');
    expectNoLiveInjection(previewContent);
    expect(previewContent.innerHTML).not.toContain(HOSTILE.DQ_IMG);

    const downloadLink = previewContent.querySelector('.preview-download a');
    expect(downloadLink).not.toBeNull();
    if (downloadLink === null) throw new Error('unreachable: downloadLink asserted non-null above');
    expect(downloadLink.textContent).toContain(HOSTILE.DQ_IMG);

    const pathText = qs(previewContent, '.preview-path-text');
    expect(pathText.textContent).toBe(`var/agent_data/artifacts/${HOSTILE.DQ_IMG}`);
  });

  test('a hostile id is percent-encoded at both the "open in new tab" href and the download-link href', () => {
    setSelectedArtifact(makeArtifact({ id: HOSTILE_ID, artifact_type: 'totally_unrecognized_type', category: 'json', filename: 'x.bin' }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = byId('preview-content');
    const expectedUrl = `/files/${encodeURIComponent(HOSTILE_ID)}/${encodeURIComponent('x.bin')}`;

    const openInNewTab = qs(previewContent, '.preview-header-actions a[target="_blank"]');
    expect(openInNewTab.getAttribute('href')).toBe(expectedUrl);
    expect(openInNewTab.getAttribute('href')).not.toMatch(/["?]/);

    const downloadLink = qs(previewContent, '.preview-download a');
    expect(downloadLink.getAttribute('href')).toBe(expectedUrl);
  });

  test('a javascript: scheme payload in the title renders as inert text, never as a live href', () => {
    setSelectedArtifact(makeArtifact({ title: JS_SCHEME }));
    createPreviewRenderer(makePreviewCallbacks()).renderPreview();

    const previewContent = byId('preview-content');
    expect(qs(previewContent, '.preview-header-title').textContent).toBe(JS_SCHEME);
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
    expect(qs(container, '.thumb-summary').textContent).toContain(HOSTILE.DQ_IMG);
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
  /**
   * Fixture chart-format response (`/api/artifacts/{id}/data?format=chart`) --
   * one entry per channel. `overrides.columns` replaces the channel's name.
   */
  function makeChartData({ columns = [HOSTILE.DQ_IMG], ...overrides } = {}) {
    return {
      channels: [
        {
          channel: columns[0],
          timestamps: ['2026-07-01T00:00:00Z'],
          values: [1.0],
          total_points: 1,
          returned_points: 1,
          numeric: true,
        },
      ],
      metadata: {},
      summary: { total_points: 1, returned_points: 1, downsampled: false, row_count: 1 },
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
      const script = /** @type {{ tagName?: string; onload?: (() => void) | null }} */ (node);
      if (script && script.tagName === 'SCRIPT') {
        queueMicrotask(() => script.onload && script.onload());
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
      update: vi.fn(),
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

      await renderTimeseriesTable(el, 'ts1', 0);

      expectNoLiveInjection(el);
      expect(el.innerHTML).not.toContain(HOSTILE.DQ_IMG);
      const th = qs(el, 'thead th:not(:first-child)');
      expect(th.textContent).toBe(HOSTILE.DQ_IMG);
    });
  });

  describe('renderTimeseriesView', () => {
    test('a hostile column name round-trips through the data-ch-name/title/aria-label attribute sinks with no breakout', async () => {
      vi.stubGlobal('fetch', vi.fn((url) => {
        if (url.includes('format=chart')) return Promise.resolve({ ok: true, json: () => Promise.resolve(makeChartData()) });
        return Promise.resolve({ ok: true, json: () => Promise.resolve(makeTableData()) });
      }));
      const container = document.createElement('div');
      document.body.appendChild(container);

      await renderTimeseriesView(container, { id: 'ts1' });

      expectNoLiveInjection(container);
      expect(container.innerHTML).not.toContain(HOSTILE.DQ_IMG);

      const toggle = /** @type {HTMLElement | null} */ (container.querySelector('.ts-ch-toggle'));
      expect(toggle).not.toBeNull();
      if (toggle === null) throw new Error('unreachable: toggle asserted non-null above');
      // dataset/getAttribute auto-decode entities: reading back round-trips
      // to the raw hostile column name — what matters is the SERIALIZED
      // markup above never contained the raw breakout.
      expect(toggle.dataset.chName).toBe(HOSTILE.DQ_IMG);
      expect(toggle.getAttribute('title')).toBe(HOSTILE.DQ_IMG);
      // `aria-label` is a third double-quoted sink for the same agent-supplied
      // channel name -- the double-quote in DQ_IMG is the breakout character.
      expect(toggle.getAttribute('aria-label')).toBe(HOSTILE.DQ_IMG);
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
      if (badge === null) throw new Error('unreachable: badge asserted non-null above');
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
    // `modal` singleton state doesn't leak into a later test.
    const closeBtn = /** @type {HTMLElement | null} */ (document.querySelector('.logbook-modal-close'));
    if (closeBtn) closeBtn.click();
    setArtifacts([]);
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
    const logbookBtn = /** @type {HTMLElement | null} */ (document.querySelector('.logbook-action-btn'));
    expect(logbookBtn).not.toBeNull();
    if (logbookBtn === null) throw new Error('unreachable: logbookBtn asserted non-null above');
    // Opening the modal fills its model selector from /api/logbook/models.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ provider: 'anthropic', default: '', models: [] }),
    }));
    logbookBtn.click(); // opens the compose modal, phase = steering

    const chooseRadio = /** @type {HTMLInputElement | null} */ (document.querySelector('input[name="logbook-artifact-scope"][value="choose"]'));
    expect(chooseRadio).not.toBeNull();
    if (chooseRadio === null) throw new Error('unreachable: chooseRadio asserted non-null above');

    setArtifacts([{ id: HOSTILE_PICKER_ID, title: 'Hostile Artifact', artifact_type: 'json' }]);

    chooseRadio.checked = true;
    chooseRadio.dispatchEvent(new Event('change', { bubbles: true }));

    const list = document.getElementById('logbook-artifact-picker-list');
    expect(list).not.toBeNull();
    if (list === null) throw new Error('unreachable: list asserted non-null above');

    const checkboxes = /** @type {NodeListOf<HTMLInputElement>} */ (list.querySelectorAll('input[type=checkbox]'));
    expect(checkboxes.length).toBe(1); // no extra <input> injected by the hostile id
    expect(checkboxes[0].value).toBe(HOSTILE_PICKER_ID); // raw round-trip via the .value property

    // The id was assigned as a DOM property, never interpolated into the
    // innerHTML string, so it can never appear as a serialized attribute
    // breakout in the picker's markup.
    expect(list.innerHTML).not.toContain(HOSTILE_PICKER_ID);
  });
});

// =========================================================================
// MARKDOWN path (md-render.js via preview-content.js) — agent-written
// markdown carrying script, rendered with the vendored marked + DOMPurify.
// =========================================================================

describe('MARKDOWN path (md-render.js / preview-content.js) — script-bearing markdown', () => {
  const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
  const MANIFEST = path.join(REPO_ROOT, 'src/osprey/interfaces/vendor_manifest.json');
  /** @type {Readonly<Record<string, string>>} */
  const BUILDS = Object.freeze({
    marked: 'node_modules/marked/marked.min.js',
    DOMPurify: 'node_modules/dompurify/dist/purify.min.js',
  });

  /** @param {string} name */
  function pinned(name) {
    const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
    const asset = manifest.assets.find((/** @type {{ name: string }} */ a) => a.name === name);
    if (!asset) throw new Error(`vendor_manifest.json names no asset ${name}`);
    return asset;
  }

  /** @param {string} name */
  function buildBytes(name) {
    return readFileSync(path.join(REPO_ROOT, BUILDS[name]));
  }

  /**
   * Evaluate a UMD build as a classic <script> would, then take its global back.
   * @param {string} name
   * @returns {any}
   */
  function loadClassicScript(name) {
    new Function('exports', 'module', 'define', 'require', buildBytes(name).toString('utf8'))();
    const g = /** @type {any} */ (globalThis);
    const lib = g[name];
    delete g[name];
    return lib;
  }

  /**
   * A KaTeX stand-in shaped like KaTeX's real output (quoted class/style
   * attributes, escaped source), so the test sees the markup the swap splices in.
   */
  const fakeKatex = {
    renderToString: (/** @type {string} */ expr, /** @type {{displayMode: boolean}} */ opts) =>
      `<span class="katex${opts.displayMode ? ' katex-display' : ''}">` +
      `<span class="katex-html" aria-hidden="true" style="height:0.8em">` +
      `<span class="mord mathnormal">${escapeForFake(expr)}</span></span></span>`,
  };

  /** @param {string} s */
  function escapeForFake(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /** Hostile markdown bodies an agent could write into a .md artifact. */
  const HOSTILE_MD = {
    SCRIPT: '# Report\n\n<script>window.__pwned = 1</script>\n\nbody',
    IMG_ONERROR: 'see <img src=x onerror="window.__pwned=1"> here',
    JS_LINK: '[click me](javascript:window.__pwned=1)',
    RAW_JS_ANCHOR: '<a href="javascript:window.__pwned=1">x</a>',
    IFRAME: '<iframe srcdoc="<script>parent.__pwned=1</script>"></iframe>',
    SVG_ONLOAD: '<svg onload="window.__pwned=1"><circle r="4"/></svg>',
    // A math placeholder inside a link target: splicing KaTeX's quoted markup
    // into the href after sanitizing would break out of the attribute.
    MATH_IN_HREF: '[x](http://example.com/$a" onmouseover="window.__pwned=1$)',
  };

  /**
   * Nothing executable anywhere under `root`.
   * @param {ParentNode} root
   */
  function expectInert(root) {
    expect(root.querySelector('script')).toBeNull();
    expect(root.querySelector('iframe')).toBeNull();
    for (const el of root.querySelectorAll('*')) {
      for (const attr of Array.from(el.attributes)) {
        expect(attr.name.startsWith('on'), `${el.tagName} carries ${attr.name}`).toBe(false);
        if (attr.name === 'href' || attr.name === 'src' || attr.name === 'xlink:href') {
          expect(attr.value.trim().toLowerCase().startsWith('javascript:')).toBe(false);
        }
      }
    }
  }

  /** @type {{ marked: any, DOMPurify: any }} */
  let real;

  beforeAll(() => {
    // jsdom ships no type declarations; it is loaded untyped on purpose.
    const { JSDOM } = createRequire(import.meta.url)('jsdom');
    const window = new JSDOM('').window;
    real = {
      marked: loadClassicScript('marked'),
      DOMPurify: loadClassicScript('DOMPurify')(window),
    };
  });

  beforeEach(() => {
    vi.stubGlobal('marked', real.marked);
    vi.stubGlobal('DOMPurify', real.DOMPurify);
    vi.stubGlobal('katex', fakeKatex);
    delete (/** @type {any} */ (globalThis)).__pwned;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
  });

  for (const name of Object.keys(BUILDS)) {
    test(`${name} in node_modules is the build the vendor manifest pins`, () => {
      const digest = createHash('sha256').update(buildBytes(name)).digest('hex');
      expect(digest).toBe(pinned(name).sha256);
    });
  }

  test('the vendor manifest delivers DOMPurify to the artifacts app', () => {
    expect(pinned('DOMPurify').targets).toContain('artifacts/static/js/vendor/');
  });

  test('the gallery page loads DOMPurify before its modules', () => {
    const page = readFileSync(
      path.join(REPO_ROOT, 'src/osprey/interfaces/artifacts/static/index.html'), 'utf8');
    const tag = page.indexOf("vendor_url('DOMPurify', '/static/js/vendor/purify.min.js')");
    expect(tag).toBeGreaterThan(-1);
    expect(tag).toBeLessThan(page.indexOf('type="module"'));
  });

  for (const [label, md] of Object.entries(HOSTILE_MD)) {
    test(`renderMathInMarkdown leaves ${label} inert`, () => {
      const root = document.createElement('div');
      root.innerHTML = renderMathInMarkdown(md);
      expectInert(root);
    });

    test(`the preview pane renders ${label} with nothing executable`, async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, text: () => Promise.resolve(md) }));
      const container = document.createElement('div');
      document.body.appendChild(container);

      await renderMarkdownView(container, { id: 'm1', filename: 'r.md', artifact_type: 'markdown' });

      const rendered = container.querySelector('.osprey-md-rendered');
      expect(rendered).not.toBeNull();
      expectInert(container);
      expect(/** @type {any} */ (globalThis).__pwned).toBeUndefined();
    });
  }

  test('ordinary markdown still renders as markup', () => {
    const root = document.createElement('div');
    root.innerHTML = renderMathInMarkdown('# Title\n\n**bold** and [link](https://example.com)\n\n```py\nx = 1\n```');
    expect(root.querySelector('h1')?.textContent).toBe('Title');
    expect(root.querySelector('strong')?.textContent).toBe('bold');
    expect(root.querySelector('a')?.getAttribute('href')).toBe('https://example.com');
    expect(root.querySelector('pre code')).not.toBeNull();
  });

  test('math still renders through the sanitizer', () => {
    const root = document.createElement('div');
    root.innerHTML = renderMathInMarkdown('Inline $x^2$ and display $$E = mc^2$$ done.');
    const math = root.querySelectorAll('.katex');
    expect(math.length).toBe(2);
    expect(root.querySelector('.katex-display .katex-html')?.getAttribute('style')).toBe('height:0.8em');
    expect(root.textContent).toContain('x^2');
    expect(root.textContent).toContain('E = mc^2');
    expect(root.innerHTML).not.toContain('\x00MATH');
  });

  test('without DOMPurify the preview shows the markdown as plain text', async () => {
    vi.stubGlobal('DOMPurify', undefined);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true, text: () => Promise.resolve(HOSTILE_MD.IMG_ONERROR),
    }));
    const container = document.createElement('div');

    await renderMarkdownView(container, { id: 'm1', filename: 'r.md', artifact_type: 'markdown' });

    expect(container.querySelector('img')).toBeNull();
    expect(container.textContent).toContain('onerror');
  });
});
