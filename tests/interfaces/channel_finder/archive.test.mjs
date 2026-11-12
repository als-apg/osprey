// @ts-check
/**
 * The Archive view (archive.js): the read on mount, its states, Refresh and
 * Copy. The answers are the shared route fixture's.
 *   npx vitest run tests/interfaces/channel_finder/archive.test.mjs
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test, expect, vi, afterEach, describe } from 'vitest';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

/** @type {Record<string, any>} */
const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);

/** @type {(() => void)|null} */
let unmount = null;

afterEach(() => {
  unmount?.();
  unmount = null;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.replaceChildren();
});

/**
 * Mount the view against an archive route answering `answers` in turn; a
 * function answer is called, an Error rejects.
 * @param {any[]} answers
 */
async function mount(answers) {
  /** @type {string[]} */
  const reads = [];
  /** @type {{ url: string, method?: string }[]} */
  const calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ opts = {}) => {
      calls.push({ url: String(url), method: opts.method });
      reads.push(String(url));
      const answer = answers[Math.min(reads.length - 1, answers.length - 1)];
      if (answer instanceof Error) throw answer;
      return { ok: true, status: 200, json: async () => answer };
    })
  );
  vi.resetModules();
  const view = await import(`${JS}/archive.js`);
  const container = document.createElement('div');
  document.body.append(container);
  view.mountArchive(container);
  unmount = view.unmountArchive;
  return { container, reads, calls };
}

/** @param {HTMLElement} container */
const message = (container) => container.querySelector('.osprey-state-message')?.textContent;

describe('reading', () => {
  test('mount reads the archive route once, with GET, and draws the page', async () => {
    const { container, calls } = await mount([ROUTES.archive]);
    await vi.waitFor(() => expect(container.querySelector('.archive-tiles')).not.toBeNull());
    expect(calls).toEqual([{ url: '/api/directory/archive', method: undefined }]);
    expect(container.querySelector('.archive-title')?.textContent).toBe('Archive');
    expect(container.querySelector('.osprey-state')).toBeNull();
  });

  test('Refresh reads again', async () => {
    const { container, reads } = await mount([ROUTES.archive]);
    await vi.waitFor(() => expect(container.querySelector('.archive-tiles')).not.toBeNull());
    /** @type {HTMLElement} */ (container.querySelector('[data-action="refresh"]')).click();
    await vi.waitFor(() => expect(reads).toHaveLength(2));
  });

  test('a late answer for an unmounted view draws nothing', async () => {
    /** @type {(value: any) => void} */
    let release = () => {};
    const held = new Promise((resolve) => {
      release = resolve;
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, status: 200, json: async () => held }))
    );
    vi.resetModules();
    const view = await import(`${JS}/archive.js`);
    const container = document.createElement('div');
    view.mountArchive(container);
    view.unmountArchive();
    release(ROUTES.archive);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(container.querySelector('.archive-tiles')).toBeNull();
  });
});

describe('states', () => {
  test('while reading, the view says so', async () => {
    const { container } = await mount([new Promise(() => {})]);
    expect(message(container)).toBe('Reading the archive');
  });

  test('another target than the archive serves reads not available, with Refresh', async () => {
    const answer = {
      outcome: 'unavailable',
      source: { target: 'live', service: 'none', label: 'All PVs', rehearsal: false },
      reason: 'not available for this target',
      configuration: { archiver: { type: 'mongodb_archiver' } },
      targets: [{ target: 'live', archive: false }],
    };
    const { container } = await mount([answer]);
    await vi.waitFor(() => expect(message(container)).toBe('Not available for this target'));
    expect(container.querySelector('.osprey-state button')?.textContent).toBe('Refresh');
  });

  test('an archive that cannot be reached gives its reason', async () => {
    const answer = {
      outcome: 'unavailable',
      source: { target: 'va', service: 'none', label: 'All PVs', rehearsal: false },
      reason: 'cannot reach the archive: connection refused',
      configuration: {},
      targets: [],
    };
    const { container } = await mount([answer]);
    await vi.waitFor(() => expect(message(container)).toBe('Archive not available'));
    expect(container.querySelector('.osprey-state-detail')?.textContent).toBe(
      'cannot reach the archive: connection refused'
    );
  });

  test('a failed read offers Retry, which reads again', async () => {
    const { container, reads } = await mount([new Error('network'), ROUTES.archive]);
    await vi.waitFor(() => expect(message(container)).toBe('Archive summary not available'));
    /** @type {HTMLElement} */ (container.querySelector('.osprey-state button')).click();
    await vi.waitFor(() => expect(container.querySelector('.archive-tiles')).not.toBeNull());
    expect(reads).toHaveLength(2);
  });
});

describe('copy and source', () => {
  test('Copy hands the configuration text to the clipboard', async () => {
    const writeText = vi.fn(async () => {});
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    const { container } = await mount([ROUTES.archive]);
    await vi.waitFor(() => expect(container.querySelector('[data-action="copy"]')).not.toBeNull());
    /** @type {HTMLElement} */ (container.querySelector('[data-action="copy"]')).click();
    await vi.waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(String(writeText.mock.calls[0])).toContain('type: mongodb_archiver');
  });

  test('the source popover names the archiver and the target', async () => {
    const { container } = await mount([ROUTES.archive]);
    await vi.waitFor(() =>
      expect(container.querySelector('[data-action="source"]')).not.toBeNull()
    );
    /** @type {HTMLElement} */ (container.querySelector('[data-action="source"]')).click();
    const popover = document.querySelector('.osprey-info-popover');
    expect(popover?.textContent).toContain('mongodb_archiver');
    expect(popover?.textContent).toContain('va');
  });
});
