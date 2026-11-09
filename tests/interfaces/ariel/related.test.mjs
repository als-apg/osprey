// @ts-check
/**
 * The ARIEL reader's "Related" section:
 *   npx vitest run tests/interfaces/ariel/related.test.mjs
 *
 * An after-body reader-seam section, below the thread. It fetches
 * `/entries/{id}/related` (limit 5) and lists up to five rows as the thread's
 * `data-thread-entry` buttons, which the reader routes to `onSelectEntry`.
 * `disabled` hides it, `no_embedding` shows a one-line note, and a failed
 * fetch leaves it out, logs the error and raises no toast.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    entriesApi: { ...actual.entriesApi, get: vi.fn(), thread: vi.fn(), related: vi.fn() },
  };
});

vi.mock('/design-system/js/toast.js', () => ({ showToast: vi.fn() }));

import { entriesApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import { showToast } from '/design-system/js/toast.js';
import { seam } from '../../../src/osprey/interfaces/ariel/static/js/reader-seam.js';
import { showInReader } from '../../../src/osprey/interfaces/ariel/static/js/reader.js';
import {
  relatedSection,
  registerRelated,
  renderRelatedSection,
  RELATED_LIMIT,
} from '../../../src/osprey/interfaces/ariel/static/js/related.js';

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
 * @param {string} id
 * @returns {any}
 */
function row(id) {
  return {
    entry_id: id,
    timestamp: '2026-01-01T00:00:00Z',
    author: 'a',
    subject: `subject of ${id}`,
  };
}

/** @type {HTMLElement} */
let container;

/** @returns {Element|null} */
const relatedSlot = () => container.querySelector('[data-reader-slot="related"]');

/** @returns {Element|null} */
const related = () => container.querySelector('[data-entry-related]');

const baseSections = [...seam.sections];

beforeEach(() => {
  document.body.innerHTML = '<div id="ariel-reader"><div id="reader-body"></div></div>';
  container = /** @type {HTMLElement} */ (document.getElementById('reader-body'));
  vi.clearAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  seam.sections.splice(0, seam.sections.length, ...baseSections);
  registerRelated();
  vi.mocked(entriesApi.get).mockResolvedValue(entry('e-1'));
  vi.mocked(entriesApi.thread).mockResolvedValue({ parent_id: null, parent: null, children: [] });
});

afterEach(() => {
  vi.restoreAllMocks();
  seam.sections.splice(0, seam.sections.length, ...baseSections);
});

describe('registration', () => {
  test('registers one section, after the thread', () => {
    const ids = seam.sections.map((s) => s.id);
    expect(ids.filter((id) => id === 'related')).toHaveLength(1);
    expect(ids.indexOf('related')).toBeGreaterThan(ids.indexOf('thread'));
    registerRelated();
    expect(seam.sections.filter((s) => s.id === 'related')).toHaveLength(1);
    expect(seam.sections.find((s) => s.id === 'related')).toBe(relatedSection);
  });
});

describe('the related section in the reader', () => {
  test('fetches five related entries for the shown entry after the body', async () => {
    vi.mocked(entriesApi.related).mockResolvedValueOnce({
      state: 'ok',
      entries: [row('r-1'), row('r-2')],
    });
    await showInReader(container, 'e-1');
    expect(entriesApi.related).toHaveBeenCalledWith('e-1', 5);
    expect(RELATED_LIMIT).toBe(5);
    const section = related();
    expect(section?.textContent).toContain('Related');
    const links = [...(section?.querySelectorAll('button[data-thread-entry]') ?? [])].map(
      (b) => /** @type {HTMLElement} */ (b).dataset.threadEntry
    );
    expect(links).toEqual(['r-1', 'r-2']);
    expect(section?.textContent).toContain('subject of r-1');
    expect(container.querySelector('.entry-detail-title')?.textContent).toBe('subject of e-1');
  });

  test('shows at most five rows', async () => {
    vi.mocked(entriesApi.related).mockResolvedValueOnce({
      state: 'ok',
      entries: ['1', '2', '3', '4', '5', '6', '7'].map((n) => row(`r-${n}`)),
    });
    await showInReader(container, 'e-1');
    expect(related()?.querySelectorAll('[data-thread-entry]')).toHaveLength(5);
  });

  test('a related row selects its entry through onSelectEntry', async () => {
    vi.mocked(entriesApi.related).mockResolvedValueOnce({ state: 'ok', entries: [row('r-9')] });
    const onSelectEntry = vi.fn();
    await showInReader(container, 'e-1', { onSelectEntry });
    const button = /** @type {HTMLElement} */ (related()?.querySelector('[data-thread-entry]'));
    button.click();
    expect(onSelectEntry).toHaveBeenCalledWith('r-9');
  });

  test('disabled hides the section', async () => {
    vi.mocked(entriesApi.related).mockResolvedValueOnce({ state: 'disabled', entries: [] });
    await showInReader(container, 'e-1');
    expect(related()).toBeNull();
    expect(relatedSlot()?.innerHTML.trim()).toBe('');
  });

  test('no_embedding shows a one-line note and no rows', async () => {
    vi.mocked(entriesApi.related).mockResolvedValueOnce({ state: 'no_embedding', entries: [] });
    await showInReader(container, 'e-1');
    const section = related();
    expect(section).not.toBeNull();
    expect(section?.querySelector('[data-thread-entry]')).toBeNull();
    const note = section?.querySelector('.entry-related-note');
    expect(note?.textContent?.trim()).toMatch(/embedded/);
    expect(note?.textContent?.trim().split('\n')).toHaveLength(1);
  });

  test('ok with no rows shows nothing', async () => {
    vi.mocked(entriesApi.related).mockResolvedValueOnce({ state: 'ok', entries: [] });
    await showInReader(container, 'e-1');
    expect(related()).toBeNull();
  });

  test('a failed fetch leaves the section out, logs the error, raises no toast', async () => {
    const failure = new Error('500');
    vi.mocked(entriesApi.related).mockRejectedValueOnce(failure);
    const shown = await showInReader(container, 'e-1');
    expect(shown.entry_id).toBe('e-1');
    expect(related()).toBeNull();
    expect(relatedSlot()?.innerHTML.trim()).toBe('');
    expect(console.error).toHaveBeenCalledWith(expect.any(String), failure);
    expect(showToast).not.toHaveBeenCalled();
    expect(container.querySelector('.entry-detail-title')?.textContent).toBe('subject of e-1');
  });

  test('a late response for a superseded entry renders nothing', async () => {
    /** @type {(v: any) => void} */
    let resolveOld = () => {};
    vi.mocked(entriesApi.get).mockImplementation(async (id) => entry(String(id)));
    vi.mocked(entriesApi.related)
      .mockReturnValueOnce(new Promise((res) => (resolveOld = res)))
      .mockResolvedValueOnce({ state: 'ok', entries: [row('new-r')] });
    const p1 = showInReader(container, 'old');
    await new Promise((r) => setTimeout(r, 0));
    await showInReader(container, 'new');
    resolveOld({ state: 'ok', entries: [row('old-r')] });
    await p1;
    const ids = [...container.querySelectorAll('[data-entry-related] [data-thread-entry]')].map(
      (b) => /** @type {HTMLElement} */ (b).dataset.threadEntry
    );
    expect(ids).toEqual(['new-r']);
  });
});

describe('renderRelatedSection', () => {
  test('escapes server text', () => {
    const html = renderRelatedSection({
      state: 'ok',
      entries: [{ ...row('x'), subject: '<img src=x onerror=alert(1)>' }],
    });
    expect(html).not.toContain('<img');
    expect(html).toContain('&lt;img');
  });

  test('an unknown state renders nothing', () => {
    expect(renderRelatedSection(/** @type {any} */ ({ state: 'weird', entries: [row('x')] }))).toBe(
      ''
    );
  });
});
