// @ts-check
/**
 * The Alarms tab's pill (app.js): only the newest counts read writes it.
 *   npx vitest run tests/interfaces/channel_finder/app.test.mjs
 *
 * The panel reads the counts on boot and again when it is shown. The reads
 * await their answers, so the answers may arrive in either order; the read
 * started last is the one the pill shows, and starting it aborts the one
 * before.
 *
 * Runs under happy-dom (vitest.config.js). app.js boots on import, so the
 * toolbar's view switch and the fetch stub are in place before it is imported.
 */

import { afterEach, expect, test, vi } from 'vitest';

const APP_MODULE = '../../../src/osprey/interfaces/channel_finder/static/js/app.js';

const INFO = {
  pipeline_type: 'hierarchical',
  metadata: {},
  available_pipelines: ['hierarchical'],
  db_path: null,
  directory: { label: 'All PVs', configured_services: ['alarm_state'] },
};

/**
 * @param {unknown} body
 */
function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

/**
 * The counts answer for `counts`, by severity.
 * @param {Record<string, number>} counts
 */
function countsAnswer(counts) {
  return { outcome: 'found', counts, rows: [] };
}

/** The counts reads, in the order they were sent. */
/** @type {{ signal: AbortSignal|undefined, release: (body: unknown) => void }[]} */
let reads = [];

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  reads = [];
});

function pill() {
  return document.querySelector('#cf-views button[data-view="alarms"] .osprey-count');
}

test('of two counts reads, the pill shows the newer answer when the older answers last', async () => {
  document.body.innerHTML = '<div id="cf-views"></div>';
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init = {}) => {
      if (url === '/api/info') return jsonResponse(INFO);
      if (url.startsWith('/api/directory/alarms')) {
        return new Promise((resolve) => {
          reads.push({
            signal: init.signal ?? undefined,
            release: (b) => resolve(jsonResponse(b)),
          });
        });
      }
      return jsonResponse({});
    })
  );
  vi.resetModules();
  await import(APP_MODULE);
  await vi.waitFor(() => expect(reads).toHaveLength(1));

  document.dispatchEvent(new Event('visibilitychange'));
  await vi.waitFor(() => expect(reads).toHaveLength(2));
  expect(reads[0].signal?.aborted).toBe(true);

  reads[1].release(countsAnswer({ MAJOR: 2 }));
  await vi.waitFor(() => expect(pill()?.textContent).toBe('2'));

  reads[0].release(countsAnswer({ MINOR: 5 }));
  // The released answer's path through the panel is promise jobs only, so one
  // task turn drains it.
  await new Promise((r) => setTimeout(r, 0));

  expect(pill()?.textContent).toBe('2');
  expect(/** @type {HTMLElement} */ (pill()).dataset.severity).toBe('major');
});
