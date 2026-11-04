// @ts-check
/**
 * The ARIEL reader's "Enhancements" tab (inspector.js):
 *   npx vitest run tests/interfaces/ariel/inspector.test.mjs
 *
 * A reader-seam tab with key `e`. It fetches `/entries/{id}/enhancements`
 * only when the tab opens and shows one card per module in execution order:
 * label, feeds, state chip, times, marker, error, attempts, then a detail
 * block for a module the inspector knows and nothing more for any other. A
 * card with a `command` shows it with a copy button that writes the clipboard
 * and raises a toast; a stale `text_embedding` card says `reembed` rebuilds
 * the whole table. There is no re-run control, and no vector is rendered.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    entriesApi: {
      ...actual.entriesApi,
      get: vi.fn(),
      thread: vi.fn(),
      enhancements: vi.fn(),
    },
  };
});

vi.mock('/design-system/js/toast.js', () => ({ showToast: vi.fn() }));

import { entriesApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import { showToast } from '/design-system/js/toast.js';
import { seam } from '../../../src/osprey/interfaces/ariel/static/js/reader-seam.js';
import {
  showInReader,
  runReaderKey,
} from '../../../src/osprey/interfaces/ariel/static/js/reader.js';
import {
  inspectorTab,
  registerInspector,
  renderInspector,
  renderModuleCard,
} from '../../../src/osprey/interfaces/ariel/static/js/inspector.js';

/**
 * @param {string} id
 * @returns {any}
 */
function entry(id) {
  return {
    entry_id: id,
    timestamp: '2026-01-01T00:00:00Z',
    author: 'author',
    source_system: 'sys',
    raw_text: `subject of ${id}\nbody of ${id}`,
    keywords: [],
    attachments: [],
  };
}

/**
 * @param {Partial<import('../../../src/osprey/interfaces/ariel/static/js/api.js').ModuleCard>} over
 * @returns {import('../../../src/osprey/interfaces/ariel/static/js/api.js').ModuleCard}
 */
function card(over) {
  return {
    module: 'semantic_processor',
    label: 'Summary & keywords',
    order: 0,
    feeds: ['keyword search'],
    state: 'complete',
    completed_at: '2026-01-02T03:04:05Z',
    failed_at: null,
    marker: null,
    error: null,
    attempts: 1,
    command: null,
    detail: null,
    ...over,
  };
}

/** @type {HTMLElement} */
let container;

const baseTabs = [...seam.tabs];

/** @returns {HTMLElement} */
const panel = () => /** @type {HTMLElement} */ (container.querySelector('[data-reader-tab-panel]'));

/** Let the tab's async render settle. */
const settle = () => new Promise((r) => setTimeout(r, 0));

/** @param {HTMLElement} root */
const cardModules = (root) =>
  [...root.querySelectorAll('[data-inspector-card]')].map(
    (c) => /** @type {HTMLElement} */ (c).dataset.module
  );

beforeEach(() => {
  document.body.innerHTML = '<div id="ariel-reader"><div id="reader-body"></div></div>';
  container = /** @type {HTMLElement} */ (document.getElementById('reader-body'));
  vi.clearAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  seam.tabs.splice(0, seam.tabs.length, ...baseTabs);
  registerInspector();
  vi.mocked(entriesApi.get).mockResolvedValue(entry('e-1'));
  vi.mocked(entriesApi.thread).mockResolvedValue({ parent_id: null, parent: null, children: [] });
});

afterEach(() => {
  vi.restoreAllMocks();
  seam.tabs.splice(0, seam.tabs.length, ...baseTabs);
});

describe('registration', () => {
  test('registers one "Enhancements" tab with key e', () => {
    registerInspector();
    const tabs = seam.tabs.filter((t) => t.id === 'enhancements');
    expect(tabs).toHaveLength(1);
    expect(tabs[0]).toBe(inspectorTab);
    expect(inspectorTab.label).toBe('Enhancements');
    expect(inspectorTab.key).toBe('e');
  });
});

describe('the tab in the reader', () => {
  test('loads /enhancements only when the tab opens', async () => {
    vi.mocked(entriesApi.enhancements).mockResolvedValue({ entry_id: 'e-1', modules: [card({})] });
    await showInReader(container, 'e-1');
    expect(entriesApi.enhancements).not.toHaveBeenCalled();
    const tab = /** @type {HTMLElement} */ (
      container.querySelector('[data-reader-tab="enhancements"]')
    );
    expect(tab.textContent).toBe('Enhancements');
    tab.click();
    await settle();
    expect(entriesApi.enhancements).toHaveBeenCalledWith('e-1');
    expect(cardModules(panel())).toEqual(['semantic_processor']);
  });

  test('the e key opens the tab', async () => {
    vi.mocked(entriesApi.enhancements).mockResolvedValue({ entry_id: 'e-1', modules: [card({})] });
    await showInReader(container, 'e-1');
    expect(runReaderKey(container, 'e')).toBe(true);
    await settle();
    expect(entriesApi.enhancements).toHaveBeenCalledTimes(1);
    expect(panel().hidden).toBe(false);
    expect(cardModules(panel())).toEqual(['semantic_processor']);
  });

  test('a failed fetch shows an error line, never cards', async () => {
    vi.mocked(entriesApi.enhancements).mockRejectedValue(new Error('boom'));
    await showInReader(container, 'e-1');
    runReaderKey(container, 'e');
    await settle();
    expect(cardModules(panel())).toEqual([]);
    expect(panel().textContent).toContain('could not be loaded');
  });
});

describe('cards', () => {
  test('are listed in execution order', () => {
    const el = renderInspector({
      entry_id: 'e-1',
      modules: [
        card({ module: 'qmd_export', label: 'Mirror', order: 3 }),
        card({ module: 'text_embedding', label: 'Text embedding', order: 1 }),
        card({ module: 'semantic_processor', order: 0 }),
      ],
    });
    expect(cardModules(el)).toEqual(['semantic_processor', 'text_embedding', 'qmd_export']);
  });

  test('no modules: an empty-state line', () => {
    const el = renderInspector({ entry_id: 'e-1', modules: [] });
    expect(cardModules(el)).toEqual([]);
    expect(el.textContent).toContain('No enhancement modules');
  });

  test('show label, feeds, state chip, times, marker, error and attempts', () => {
    const html = renderModuleCard(
      card({
        module: 'image_caption',
        label: 'Captions',
        feeds: ['keyword search', 'semantic search'],
        state: 'failed',
        completed_at: null,
        failed_at: '2026-02-03T04:05:06Z',
        marker: 'model-x',
        error: 'provider timed out',
        attempts: 3,
      })
    );
    const el = document.createElement('div');
    el.innerHTML = html;
    const c = /** @type {HTMLElement} */ (el.querySelector('[data-inspector-card]'));
    expect(c.dataset.state).toBe('failed');
    expect(c.querySelector('.inspector-card-title')?.textContent).toBe('Captions');
    const chip = /** @type {HTMLElement} */ (c.querySelector('[data-inspector-chip]'));
    expect(chip.textContent).toBe('failed');
    expect(chip.className).toContain('ariel-strip-cell-failed');
    const text = c.textContent ?? '';
    expect(text).toContain('keyword search, semantic search');
    expect(text).toContain('model-x');
    expect(text).toContain('provider timed out');
    expect(text).toContain('3');
    expect(text).toContain('Failed');
    // A missing value reads as an em dash.
    const completed = [...c.querySelectorAll('dt')].find((d) => d.textContent === 'Completed');
    expect(completed?.nextElementSibling?.textContent).toBe('—');
  });

  test('escape server text', () => {
    const html = renderModuleCard(
      card({ label: '<img src=x onerror=alert(1)>', error: '<script>x</script>', state: 'failed' })
    );
    const el = document.createElement('div');
    el.innerHTML = html;
    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('script')).toBeNull();
    expect(el.textContent).toContain('<script>x</script>');
  });

  test('carry no re-run control', () => {
    const el = renderInspector({
      entry_id: 'e-1',
      modules: [card({ state: 'failed', command: 'osprey ariel enhance --module x --entry e-1' })],
    });
    const buttons = [...el.querySelectorAll('button')];
    expect(buttons.map((b) => b.textContent?.trim())).toEqual(['Copy']);
    expect(el.textContent).not.toMatch(/re-?run now|retry/i);
  });
});

describe('the command', () => {
  test('is absent without a command', () => {
    const el = renderInspector({ entry_id: 'e-1', modules: [card({})] });
    expect(el.querySelector('[data-inspector-command]')).toBeNull();
    expect(el.querySelector('[data-inspector-copy]')).toBeNull();
  });

  test('is shown and copied to the clipboard with a toast', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const command = "osprey ariel enhance --module semantic_processor --entry 'e-1'";
    const el = renderInspector({
      entry_id: 'e-1',
      modules: [card({ state: 'failed', command })],
    });
    expect(el.querySelector('[data-inspector-command] code')?.textContent).toBe(command);
    /** @type {HTMLElement} */ (el.querySelector('[data-inspector-copy]')).click();
    await settle();
    expect(writeText).toHaveBeenCalledWith(command);
    expect(showToast).toHaveBeenCalledTimes(1);
    expect(String(vi.mocked(showToast).mock.calls[0][0])).toMatch(/copied/i);
  });

  test('a refused clipboard write raises a warning toast', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error('denied'));
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const el = renderInspector({
      entry_id: 'e-1',
      modules: [card({ state: 'failed', command: 'cmd' })],
    });
    /** @type {HTMLElement} */ (el.querySelector('[data-inspector-copy]')).click();
    await settle();
    expect(showToast).toHaveBeenCalledTimes(1);
    expect(String(vi.mocked(showToast).mock.calls[0][0])).toMatch(/could not copy/i);
  });

  test('a stale text_embedding card says reembed rebuilds the whole table', () => {
    const stale = renderModuleCard(
      card({ module: 'text_embedding', state: 'stale', command: 'osprey ariel reembed --model m' })
    );
    expect(stale).toContain('rebuilds the whole table');
    const failed = renderModuleCard(
      card({ module: 'text_embedding', state: 'failed', command: 'osprey ariel enhance ...' })
    );
    expect(failed).not.toContain('rebuilds the whole table');
    const otherStale = renderModuleCard(
      card({ module: 'image_embedding', state: 'stale', command: 'x' })
    );
    expect(otherStale).not.toContain('rebuilds the whole table');
  });
});

describe('detail blocks', () => {
  /** @param {import('../../../src/osprey/interfaces/ariel/static/js/api.js').ModuleCard} c */
  const render = (c) => {
    const el = document.createElement('div');
    el.innerHTML = renderModuleCard(c);
    return el;
  };

  test('semantic_processor: summary and keywords', () => {
    const el = render(
      card({ detail: { summary: 'Beam lost at 3am', keywords: ['beam', 'loss'] } })
    );
    const detail = el.querySelector('[data-inspector-detail]');
    expect(detail?.textContent).toContain('Beam lost at 3am');
    expect(detail?.textContent).toContain('beam');
    expect(detail?.textContent).toContain('loss');
  });

  test('text_embedding: table, model, dimensions, input key and would-embed text; never a vector', () => {
    const el = render(
      card({
        module: 'text_embedding',
        detail: {
          tables: [
            {
              table: 'text_embeddings_m',
              model: 'm',
              dimensions: 768,
              created_at: '2026-01-01T00:00:00Z',
              input_key: 'k1',
              would_embed_now: { input_key: 'k2', text: 'the text to embed' },
              embedding: [0.123456, 0.654321],
            },
          ],
          vector: [0.999111],
        },
      })
    );
    const text = el.querySelector('[data-inspector-detail]')?.textContent ?? '';
    expect(text).toContain('text_embeddings_m');
    expect(text).toContain('768');
    expect(text).toContain('k1');
    expect(text).toContain('k2');
    expect(text).toContain('the text to embed');
    expect(el.textContent).not.toContain('0.123456');
    expect(el.textContent).not.toContain('0.999111');
  });

  test('image_embedding: table and one row per picture', () => {
    const el = render(
      card({
        module: 'image_embedding',
        detail: {
          table: 'image_embeddings',
          model: 'clip',
          dimensions: 512,
          table_exists: true,
          rows: [
            {
              attachment_id: 'a1',
              stored: true,
              dimensions: 512,
              skip_reason: null,
              model_ref: 'clip@1',
              created_at: '2026-01-01T00:00:00Z',
            },
            { attachment_id: 'a2', stored: false },
          ],
        },
      })
    );
    const detail = el.querySelector('[data-inspector-detail]');
    expect(detail?.textContent).toContain('image_embeddings');
    expect(detail?.querySelectorAll('[data-inspector-row]')).toHaveLength(2);
    expect(detail?.textContent).toContain('not stored');
  });

  test('image_caption: captions per picture and the attachment text', () => {
    const el = render(
      card({
        module: 'image_caption',
        detail: {
          model: 'cap-model',
          pictures: [{ attachment_id: 'a1', captions: { 'cap-model': 'a plot of beam current' } }],
          attachment_text: 'composed text',
        },
      })
    );
    const text = el.querySelector('[data-inspector-detail]')?.textContent ?? '';
    expect(text).toContain('a plot of beam current');
    expect(text).toContain('composed text');
  });

  test('qmd_export: mirror path and whether it exists', () => {
    const el = render(
      card({ module: 'qmd_export', detail: { mirror_path: '/mirror/e-1.qmd', exists: false } })
    );
    const text = el.querySelector('[data-inspector-detail]')?.textContent ?? '';
    expect(text).toContain('/mirror/e-1.qmd');
    expect(text).toContain('no');
  });

  test('an unknown module gets a status-only card, whatever its detail', () => {
    const el = render(
      card({ module: 'custom_thing', label: 'Custom', detail: { secret: 'do-not-show' } })
    );
    expect(el.querySelector('[data-inspector-detail]')).toBeNull();
    expect(el.textContent).not.toContain('do-not-show');
    expect(el.textContent).toContain('Custom');
  });

  test('a known module with null detail is status-only', () => {
    const el = render(card({ detail: null }));
    expect(el.querySelector('[data-inspector-detail]')).toBeNull();
  });
});
