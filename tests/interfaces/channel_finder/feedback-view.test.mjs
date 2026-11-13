// @ts-check
/**
 * The Feedback view on the panel standard (feedback.js, DESIGN.md §12):
 * loading, failed and switched off are centred states; the entry counts are the
 * toolbar's status text; the entries section header carries New entry, Export
 * and Clear all as icon buttons, Clear all disabled with a reason when there is
 * nothing to clear; the detail view's breadcrumb is the crumb primitive.
 *   npx vitest run tests/interfaces/channel_finder/feedback-view.test.mjs
 */

import { test, expect, vi, afterEach, describe } from 'vitest';

import {
  mountFeedback,
  unmountFeedback,
} from '../../../src/osprey/interfaces/channel_finder/static/js/feedback.js';

afterEach(() => {
  unmountFeedback();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

const ENTRY = {
  key: 'k1',
  query: 'bpms',
  facility: 'ERF',
  success_count: 3,
  failure_count: 1,
  last_activity: '2026-08-27T10:00:00Z',
};

/** @param {any[]} entries */
const storeRoutes = (entries) => ({
  '/api/feedback/status': { available: true, paradigm: 'hierarchical' },
  '/api/feedback': { entries },
  '/api/pending-reviews/status': { available: false },
});

/**
 * Answer each GET with the payload registered for its path: a function is
 * called, an Error rejects, an unregistered path rejects.
 * @param {Record<string, any>} routes
 */
function stubApi(routes) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ path) => {
      if (!(path in routes)) throw new Error(`unstubbed fetch: ${path}`);
      const answer = typeof routes[path] === 'function' ? await routes[path]() : routes[path];
      if (answer instanceof Error) throw answer;
      return { ok: true, status: 200, json: async () => answer };
    })
  );
}

/** The panel toolbar's status slot (index.html) and a mount point. */
function freshPanel() {
  document.body.innerHTML = '<span id="cf-result-count" hidden></span><div id="view"></div>';
  return /** @type {HTMLElement} */ (document.getElementById('view'));
}

const count = () => /** @type {HTMLElement} */ (document.getElementById('cf-result-count'));
/** @param {HTMLElement} container */
const message = (container) => container.querySelector('.osprey-state-message')?.textContent;
/** @param {HTMLElement} container @param {string} action */
const button = (container, action) =>
  /** @type {HTMLButtonElement|null} */ (container.querySelector(`[data-action="${action}"]`));

describe('states', () => {
  test('loading is a centred state, not a spinner', async () => {
    const container = freshPanel();
    stubApi({ '/api/feedback/status': () => new Promise(() => {}) });
    mountFeedback(container);
    expect(container.querySelector('.osprey-state')).not.toBeNull();
    expect(message(container)).toBe('Loading feedback');
    expect(container.querySelector('.loading-spinner')).toBeNull();
  });

  test('a failed read is a state with Retry, and Retry reads again', async () => {
    const container = freshPanel();
    let calls = 0;
    stubApi({
      ...storeRoutes([ENTRY]),
      '/api/feedback/status': () =>
        ++calls === 1 ? new Error('connection refused') : { available: true },
    });
    mountFeedback(container);
    await vi.waitFor(() => expect(message(container)).toBe('Feedback could not be read'));
    const retry = /** @type {HTMLButtonElement} */ (
      container.querySelector('.osprey-state button')
    );
    expect(retry.textContent).toBe('Retry');
    expect(container.querySelector('.osprey-state')?.textContent).not.toMatch(
      /[\u{1F300}-\u{1FAFF}]/u
    );
    retry.click();
    await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());
  });

  test('switched off is a state with no action and no emoji', async () => {
    const container = freshPanel();
    stubApi({ '/api/feedback/status': { available: false, paradigm: 'hierarchical' } });
    mountFeedback(container);
    await vi.waitFor(() =>
      expect(message(container)).toBe('Feedback is switched off for this deployment')
    );
    expect(container.querySelector('.osprey-state button')).toBeNull();
    expect(container.innerHTML).not.toMatch(/&#128274;|\u{1F512}/u);
  });

  test('no entries is a state below the section header', async () => {
    const container = freshPanel();
    stubApi(storeRoutes([]));
    mountFeedback(container);
    await vi.waitFor(() => expect(message(container)).toBe('No feedback entries yet'));
    expect(container.querySelector('.fb-section-head')).not.toBeNull();
  });
});

describe('the toolbar status text', () => {
  test('carries the entry counts, and is cleared on unmount', async () => {
    const container = freshPanel();
    stubApi(storeRoutes([ENTRY, { ...ENTRY, key: 'k2', success_count: 27, failure_count: 3 }]));
    mountFeedback(container);
    await vi.waitFor(() => expect(count().hidden).toBe(false));
    expect(count().textContent).toBe('2 entries · 30 successes · 4 failures');
    expect(container.querySelector('.fb-stat')).toBeNull();
    unmountFeedback();
    expect(count().hidden).toBe(true);
    expect(count().textContent).toBe('');
  });

  test('speaks of one entry in the singular', async () => {
    const container = freshPanel();
    stubApi(storeRoutes([{ ...ENTRY, success_count: 1, failure_count: 0 }]));
    mountFeedback(container);
    await vi.waitFor(() => expect(count().hidden).toBe(false));
    expect(count().textContent).toBe('1 entry · 1 success · 0 failures');
  });

  test('a read that lands after unmount writes no count', async () => {
    const container = freshPanel();
    /** @type {(value: any) => void} */
    let release = () => {};
    let asked = false;
    stubApi({
      ...storeRoutes([ENTRY]),
      '/api/feedback': () =>
        new Promise((resolve) => {
          asked = true;
          release = resolve;
        }),
    });
    mountFeedback(container);
    await vi.waitFor(() => expect(asked).toBe(true));
    unmountFeedback();
    release({ entries: [ENTRY] });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(count().hidden).toBe(true);
  });
});

describe('the entries section header', () => {
  test('holds New entry, Export and Clear all as icon buttons', async () => {
    const container = freshPanel();
    stubApi(storeRoutes([ENTRY]));
    mountFeedback(container);
    await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());
    const head = /** @type {HTMLElement} */ (container.querySelector('.fb-section-head'));
    const actions = [...head.querySelectorAll('button')].map((b) => [
      b.dataset.action,
      b.getAttribute('aria-label'),
      b.className,
      !!b.querySelector('svg'),
    ]);
    expect(actions).toEqual([
      ['new', 'New entry', 'osprey-icon-btn', true],
      ['export', 'Export', 'osprey-icon-btn', true],
      ['clear', 'Clear all', 'osprey-icon-btn', true],
    ]);
    expect(button(container, 'clear')?.disabled).toBe(false);
  });

  test('Clear all is disabled with its reason when there are no entries, never hidden', async () => {
    const container = freshPanel();
    stubApi(storeRoutes([]));
    mountFeedback(container);
    await vi.waitFor(() => expect(button(container, 'clear')).not.toBeNull());
    const clear = /** @type {HTMLButtonElement} */ (button(container, 'clear'));
    expect(clear.disabled).toBe(true);
    expect(clear.hidden).toBe(false);
    expect(clear.title).toBe('No entries to clear');
  });

  test("a row's delete is a mini button with an icon", async () => {
    const container = freshPanel();
    stubApi(storeRoutes([ENTRY]));
    mountFeedback(container);
    await vi.waitFor(() => expect(container.querySelector('.fb-row-delete')).not.toBeNull());
    const remove = /** @type {HTMLElement} */ (container.querySelector('.fb-row-delete'));
    expect(remove.classList.contains('osprey-mini')).toBe(true);
    expect(remove.querySelector('svg')).not.toBeNull();
    expect(remove.getAttribute('aria-label')).toBe('Delete entry');
  });

  test('no inline style and no legacy button in the list', async () => {
    const container = freshPanel();
    stubApi(storeRoutes([ENTRY]));
    mountFeedback(container);
    await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());
    expect(container.querySelector('[style]')).toBeNull();
    expect(container.querySelector('.btn, .item-action-btn')).toBeNull();
  });
});

describe('the detail view', () => {
  const DETAIL = {
    _meta: { query: 'bpms', facility: 'ERF' },
    successes: [{ selections: { system: 'BPM' }, channel_count: 4, timestamp: 't1' }],
    failures: [],
  };

  /** Mount, open the entry, and wait for its detail. */
  async function openDetail() {
    const container = freshPanel();
    stubApi({ ...storeRoutes([ENTRY]), '/api/feedback/k1': DETAIL });
    mountFeedback(container);
    await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());
    /** @type {HTMLElement} */ (container.querySelector('.fb-table-row td')).click();
    await vi.waitFor(() => expect(container.querySelector('.osprey-crumb')).not.toBeNull());
    return container;
  }

  test('the breadcrumb is the crumb primitive and leads back to the list', async () => {
    const container = await openDetail();
    const crumb = /** @type {HTMLElement} */ (container.querySelector('.osprey-crumb'));
    expect(crumb.querySelector('.osprey-crumb-root')?.textContent).toBe('Feedback');
    expect(crumb.querySelector('.osprey-crumb-sep svg')).not.toBeNull();
    expect(crumb.querySelector('.osprey-crumb-leaf')?.textContent).toBe('"bpms" (ERF)');
    expect(count().hidden).toBe(true);
    /** @type {HTMLElement} */ (crumb.querySelector('.osprey-crumb-root')).click();
    await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());
  });

  test('its buttons are design-system buttons, with no inline style', async () => {
    const container = await openDetail();
    const add = [...container.querySelectorAll('[data-action^="add-"]')];
    expect(add.map((b) => [b.className, b.getAttribute('aria-label')])).toEqual([
      ['osprey-icon-btn', 'Add success'],
      ['osprey-icon-btn', 'Add failure'],
    ]);
    for (const b of container.querySelectorAll('.fb-record-actions button')) {
      expect(b.classList.contains('osprey-mini')).toBe(true);
      expect(b.querySelector('svg')).not.toBeNull();
    }
    expect(container.querySelector('[style]')).toBeNull();
    expect(container.querySelector('.btn, .item-action-btn')).toBeNull();
    expect(container.innerHTML).not.toContain('&#9998;');
    expect(container.textContent).not.toContain('✎');
  });
});
