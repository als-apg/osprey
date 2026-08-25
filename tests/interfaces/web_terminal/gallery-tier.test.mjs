// @ts-check
/**
 * Unit tests for the client half of `web.scaffold_gallery.write_enabled`:
 *
 *   npx vitest run tests/interfaces/web_terminal/gallery-tier.test.mjs
 *
 * The server is the enforcement — every write/delete verb under
 * `/api/scaffold` answers 403 with the key off (routes/scaffold.py), and
 * `GET /api/panels` publishes the posture as `scaffold_write_enabled`. These
 * tests pin what the BROWSER does with that fact: a deployment whose write
 * surface refuses must not paint controls that reach for it.
 *
 * What is pinned, in the order a reviewer would ask about it:
 *
 * - `false` withdraws every write control the gallery renders — the untracked
 *   banner's Register and Delete, the per-category "+" create button, the
 *   detail header's ownership button (Take Ownership / Release to Framework),
 *   and the Save button — while every read affordance (cards, Preview, Diff,
 *   the banner itself) survives. Looking is not authoring;
 * - `true`, a payload with no such key, and a null payload (a failed or hung
 *   `/api/panels`) all LEAVE THE CONTROLS. Absent means enabled, mirroring the
 *   server's own `getattr(..., True)` default; a read that never landed is not
 *   a statement about the deployment's posture;
 * - the withdrawal is by ABSENCE, not by a disabled attribute, for everything
 *   but the Edit mode tab — which stays rendered-but-disabled with a reason,
 *   exactly as the `read_only` badge discipline already does for a reserved
 *   artifact, because a Preview/Diff/Edit tab strip missing its third entry
 *   reads as a broken panel rather than a gated one;
 * - the settings.json structured editor falls back to its read-only view, the
 *   same branch a reserved artifact takes: a field an operator can type into,
 *   backed by a save the server refuses, is worse than one that plainly cannot
 *   be typed into.
 *
 * Module isolation: write-gate.js keeps module-private state (the resolved
 * posture), so every test re-imports it and its consumers through
 * `vi.resetModules()` — the pattern the palette suites use for the same
 * reason. That also pins the shipped default: a freshly imported gate is
 * enabled until something says otherwise.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

/**
 * Fresh copies of the gate and the two renderers that consult it.
 * @returns {Promise<{
 *   gate: typeof import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/write-gate.js'),
 *   view: typeof import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/view.js'),
 *   detail: typeof import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/detail.js'),
 *   content: typeof import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/detail-content.js'),
 * }>}
 */
async function loadModules() {
  vi.resetModules();
  const gate = await import(`${JS}/scaffold/write-gate.js`);
  const view = await import(`${JS}/scaffold/view.js`);
  const detail = await import(`${JS}/scaffold/detail.js`);
  const content = await import(`${JS}/scaffold/detail-content.js`);
  return { gate, view, detail, content };
}

/** A gallery host with real elements for every DOM ref the renderers touch. */
function makeGallery(overrides = {}) {
  return /** @type {any} */ ({
    artifacts: [],
    untrackedFiles: [],
    selectedArtifact: null,
    currentView: 'gallery',
    detailMode: 'preview',
    editDirty: false,
    searchQuery: '',
    filterCategory: null,
    filterProjectOwned: false,
    filterOpen: false,
    collapsedCategories: new Set(),
    seenCategories: new Set(),
    pinnedCategories: [],
    categoryOverrides: {},
    categoryRemaps: {},
    showSearch: true,
    showSummary: true,
    showFilterChips: true,
    summary: { total: 0, framework: 0, userOwned: 0 },
    galleryView: document.createElement('div'),
    detailView: document.createElement('div'),
    untrackedBannerEl: document.createElement('div'),
    filterChipsEl: document.createElement('div'),
    filterPanelEl: document.createElement('div'),
    filterToggleEl: document.createElement('button'),
    clearFilterEl: document.createElement('button'),
    summaryEl: document.createElement('div'),
    searchInput: document.createElement('input'),
    categoriesEl: document.createElement('div'),
    detailHeaderEl: document.createElement('div'),
    detailModesEl: document.createElement('div'),
    detailContentEl: document.createElement('div'),
    onDetailOpen: null,
    onDetailClose: null,
    load: () => Promise.resolve(),
    reloadFull: () => Promise.resolve(),
    renderGallery: () => {},
    renderDetailModes: () => {},
    renderDetailContent: () => {},
    renderEdit: vi.fn(() => Promise.resolve()),
    registerUntracked: vi.fn(() => Promise.resolve()),
    deleteUntracked: vi.fn(() => Promise.resolve()),
    openDetail: vi.fn(),
    showCreateDialog: vi.fn(),
    closeDetail: vi.fn(),
    takeOwnership: vi.fn(),
    releaseToFramework: vi.fn(),
    handleEditFramework: vi.fn(),
    discardEdits: vi.fn(),
    saveOverride: vi.fn(),
    ...overrides,
  });
}

/** One untracked file, so the banner has a row with actions to render. */
const UNTRACKED = [{ canonical_name: 'rules/stray', output_path: '.claude/rules/stray.md' }];

/** One creatable-category artifact, so a category header gets its "+" button. */
const ARTIFACTS = [
  {
    name: 'my-rule',
    category: 'rules',
    displayCategory: 'rules',
    status: 'user-owned',
    output_path: '.claude/rules/my-rule.md',
  },
];

beforeEach(() => {
  vi.stubGlobal('confirm', vi.fn(() => true));
  vi.stubGlobal('alert', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/**
 * Render every control-bearing surface once and report which write controls
 * survived. One helper, because the posture is ONE privilege: a control that
 * quietly stayed painted is the whole point of the suite.
 * @param {any} modules
 * @returns {Promise<Record<string, boolean>>}
 */
function renderControls(modules) {
  const gallery = makeGallery({ untrackedFiles: UNTRACKED, artifacts: ARTIFACTS });
  const view = modules.view.createScaffoldGalleryView(gallery);
  view.renderGallery();

  const detailGallery = makeGallery({
    selectedArtifact: { ...ARTIFACTS[0], custom: false },
    detailMode: 'edit',
    editDirty: true,
  });
  const detail = modules.detail.createScaffoldGalleryDetail(detailGallery);
  detail.renderDetailHeader();
  detail.renderDetailModes();

  const editBtn = [...detailGallery.detailModesEl.querySelectorAll('.prompts-mode-btn')].find(
    (b) => b.textContent === 'Edit'
  );

  return Promise.resolve({
    register: !!gallery.untrackedBannerEl.querySelector('.prompts-untracked-register'),
    deleteUntracked: !!gallery.untrackedBannerEl.querySelector('.prompts-untracked-delete'),
    create: !!gallery.categoriesEl.querySelector('.prompts-category-add'),
    ownership: !!detailGallery.detailHeaderEl.querySelector('.prompts-ownership-btn'),
    save: !!detailGallery.detailModesEl.querySelector('.prompts-save-btn'),
    editTabEnabled: !!editBtn && !(/** @type {HTMLButtonElement} */ (editBtn).disabled),
    // Read affordances, asserted alongside so "hid everything" cannot pass.
    banner: gallery.untrackedBannerEl.style.display !== 'none',
    // Discarding local edits touches nothing on the server, so it must survive
    // a withdrawn write surface — an operator left inside a dirty editor with
    // no way out is a worse answer than a gated one.
    discard: !!detailGallery.detailModesEl.querySelector('.prompts-discard-btn'),
    cards: !!gallery.categoriesEl.querySelector('.prompts-category-section'),
    previewTab: !!detailGallery.detailModesEl.querySelector('.prompts-mode-btn'),
  });
}

const ALL_WRITE_CONTROLS = ['register', 'deleteUntracked', 'create', 'ownership', 'save'];

describe('writes disabled', () => {
  test('every write control is withdrawn, and every read affordance stays', async () => {
    const modules = await loadModules();
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    const controls = await renderControls(modules);
    for (const name of ALL_WRITE_CONTROLS) {
      expect(controls[name], `${name} control should be withdrawn`).toBe(false);
    }
    expect(controls.banner).toBe(true);
    expect(controls.cards).toBe(true);
    expect(controls.previewTab).toBe(true);
    expect(controls.discard).toBe(true);
  });

  test('the Edit tab stays rendered but disabled, with a reason', async () => {
    const modules = await loadModules();
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    const gallery = makeGallery({
      selectedArtifact: { ...ARTIFACTS[0], custom: false },
      detailMode: 'preview',
    });
    const detail = modules.detail.createScaffoldGalleryDetail(gallery);
    detail.renderDetailModes();
    const buttons = [...gallery.detailModesEl.querySelectorAll('.prompts-mode-btn')];
    expect(buttons.map((b) => b.textContent)).toEqual(['Preview', 'Diff', 'Edit']);
    const editBtn = /** @type {HTMLButtonElement} */ (buttons[2]);
    expect(editBtn.disabled).toBe(true);
    expect(editBtn.title).toBeTruthy();
  });

  test('the banner still names the untracked files it can no longer act on', async () => {
    const modules = await loadModules();
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    const gallery = makeGallery({ untrackedFiles: UNTRACKED });
    modules.view.createScaffoldGalleryView(gallery).renderGallery();
    expect(gallery.untrackedBannerEl.textContent).toContain('rules/stray');
  });

  test('the settings.json preview falls back to the read-only structured view', async () => {
    const modules = await loadModules();
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ content: '{"permissions": {}}', language: 'json' }),
        })
      )
    );
    const gallery = makeGallery({
      selectedArtifact: { name: 'settings-json', language: 'json', status: 'user-owned' },
      detailMode: 'preview',
    });
    const content = modules.content.createScaffoldGalleryDetailContent(gallery);
    await content.renderPreview();
    // The structured view renders, and offers no control an operator could
    // type into -- so a withdrawn-writes deployment never shows a field
    // backed by a save that answers 403.
    expect(gallery.detailContentEl.querySelector('.config-structured-view')).toBeTruthy();
    expect(gallery.detailContentEl.querySelectorAll('input, select, textarea')).toHaveLength(0);
  });
});

describe('writes enabled — the shipped posture', () => {
  /** @type {Array<[string, any]>} */
  const postures = [
    ['never applied (fresh boot)', undefined],
    ['explicit true', { scaffold_write_enabled: true }],
    ['payload without the key', { ui_mode: 'expert' }],
    ['null payload (failed /api/panels)', null],
  ];

  for (const [label, payload] of postures) {
    test(`${label} leaves every write control painted`, async () => {
      const modules = await loadModules();
      if (payload !== undefined) modules.gate.applyScaffoldWriteGate(payload);
      const controls = await renderControls(modules);
      for (const name of ALL_WRITE_CONTROLS) {
        expect(controls[name], `${name} control should be painted`).toBe(true);
      }
      expect(controls.editTabEnabled).toBe(true);
    });
  }

  test('a false payload followed by a true one re-opens the controls', async () => {
    const modules = await loadModules();
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    expect(modules.gate.scaffoldWritesEnabled()).toBe(false);
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: true });
    expect(modules.gate.scaffoldWritesEnabled()).toBe(true);
    const controls = await renderControls(modules);
    for (const name of ALL_WRITE_CONTROLS) {
      expect(controls[name]).toBe(true);
    }
  });

  test('a null payload never overwrites a posture already resolved', async () => {
    const modules = await loadModules();
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    modules.gate.applyScaffoldWriteGate(null);
    expect(modules.gate.scaffoldWritesEnabled()).toBe(false);
  });
});

describe('element-absence safety', () => {
  test('rendering with no untracked files and no DOM refs does not throw', async () => {
    const modules = await loadModules();
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    const gallery = makeGallery({
      untrackedBannerEl: null,
      categoriesEl: null,
      summaryEl: null,
      filterChipsEl: null,
      searchInput: null,
      detailHeaderEl: null,
      detailModesEl: null,
    });
    expect(() => modules.view.createScaffoldGalleryView(gallery).renderGallery()).not.toThrow();
    const detail = modules.detail.createScaffoldGalleryDetail(gallery);
    expect(() => detail.renderDetailHeader()).not.toThrow();
    expect(() => detail.renderDetailModes()).not.toThrow();
  });

  test('scaffoldWritesEnabled reports the resolved posture', async () => {
    const modules = await loadModules();
    expect(modules.gate.scaffoldWritesEnabled()).toBe(true);
    modules.gate.applyScaffoldWriteGate({ scaffold_write_enabled: false });
    expect(modules.gate.scaffoldWritesEnabled()).toBe(false);
  });
});
