/**
 * Unit tests for the BLUESKY panel's Plans view (plans-view.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/bluesky_web/plans-view.test.mjs
 *
 * The view is booted whole, as panel.js boots it: over the body of the
 * shipped index.html, with a stubbed `fetch` that answers each sidecar route
 * (some of them only when a test says so) and a fake `EventSource` for the
 * draft stream. What the tests assert is what the view sends: the `PATCH
 * /draft` and `POST /queue/items` bodies, and the banner it shows instead.
 */

import { readFileSync } from 'node:fs';
import { cwd } from 'node:process';

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { stubEventSource } from '../_support/event-source.mjs';
import { createPlansView } from '../../../src/osprey/interfaces/bluesky_web/panels/bluesky/plans-view.js';

const BUNDLE = `${cwd()}/src/osprey/interfaces/bluesky_web/panels/bluesky/`;

/**
 * Two plans that share a parameter name with different defaults, so a form
 * built from the wrong plan sends a value the selected plan never showed.
 */
const PLAN_A = {
  name: 'plan_a',
  provenance: 'shipped',
  schema: { type: 'object', properties: { num: { type: 'integer', default: 99, title: 'num' } } },
};
const PLAN_B = {
  name: 'plan_b',
  provenance: 'shipped',
  schema: {
    type: 'object',
    properties: { num: { type: 'integer', default: 5, maximum: 10, title: 'num' } },
  },
};

/** @param {{name: string}} plan */
function sourceOf(plan) {
  return {
    name: plan.name,
    provenance: 'shipped',
    validated: true,
    truncated: false,
    source: `def ${plan.name}(): ...`,
  };
}

/**
 * @param {number} status
 * @param {unknown} body
 */
function jsonResponse(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

/** A promise and the function that settles it. */
function deferred() {
  /** @type {(value: any) => void} */
  let resolve = () => {};
  const promise = new Promise((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

/** Put the body of the shipped markup (scripts left out) into the document. */
function mountMarkup() {
  const html = readFileSync(`${BUNDLE}index.html`, 'utf-8');
  const body = html
    .slice(html.indexOf('<body>') + '<body>'.length, html.indexOf('</body>'))
    .replace(/<script[\s\S]*?<\/script>/g, '');
  const parsed = new DOMParser().parseFromString(`<body>${body}</body>`, 'text/html');
  document.body.replaceChildren(...parsed.body.childNodes);
}

/**
 * Boot the view over the shipped markup.
 *
 * `sources` maps a plan name to the source response, or to a `deferred()`
 * the test settles itself. `patch` answers `PATCH /draft`.
 *
 * @param {{
 *   plans: any[],
 *   sources: Record<string, any>,
 *   draft?: any,
 *   patch?: (body: any) => {status: number, body: any},
 * }} options
 */
function boot({ plans, sources, draft = { draft: null, revision: 0 }, patch }) {
  mountMarkup();
  const stream = stubEventSource();
  /** @type {Array<{method: string, path: string, body: any}>} */
  const sent = [];
  const answerPatch = patch || (() => ({ status: 200, body: { revision: 3, changed: [] } }));
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init = {}) => {
      const method = init.method || 'GET';
      const path = url.split('?')[0];
      const body = typeof init.body === 'string' ? JSON.parse(init.body) : null;
      sent.push({ method, path, body });
      if (method === 'GET' && path === '/plans') return jsonResponse(200, plans);
      if (method === 'GET' && path === '/bridge/health') {
        return jsonResponse(200, {
          capability: { can_execute: true, reason: 'ok', detail: '', retryable: false },
        });
      }
      const sourceMatch = path.match(/^\/plans\/([^/]+)\/source$/);
      if (method === 'GET' && sourceMatch) {
        const answer = sources[decodeURIComponent(sourceMatch[1])];
        const resolved = answer && answer.promise ? await answer.promise : answer;
        return jsonResponse(200, resolved);
      }
      if (method === 'GET' && path === '/draft') return jsonResponse(200, draft);
      if (method === 'PATCH' && path === '/draft') {
        const answer = answerPatch(body);
        return jsonResponse(answer.status, answer.body);
      }
      if (method === 'POST' && path === '/queue/items') {
        return jsonResponse(200, { run_id: 'run-1', revision: 3 });
      }
      return jsonResponse(404, { detail: 'not stubbed' });
    })
  );
  createPlansView({ root: document.body, api: (path) => path, onOpenRun: () => {} });
  const el = (/** @type {string} */ id) => /** @type {HTMLElement} */ (document.getElementById(id));
  return {
    sent,
    stream,
    title: () => el('detail-title').textContent,
    queueButton: () => /** @type {HTMLButtonElement} */ (el('queue-add-btn')),
    banner: () => el('launch-outcome-banner'),
    numInput: () =>
      /** @type {HTMLInputElement} */ (el('param-form').querySelector('input[type="number"]')),
    /** @param {string} method @param {string} path */
    calls: (method, path) => sent.filter((call) => call.method === method && call.path === path),
  };
}

/** Let every queued promise chain run (a macrotask hop). */
async function settle() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

beforeEach(() => {
  // The capability refresh interval would outlive the test.
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.replaceChildren();
});

describe('plans view — the form and the enqueue follow the newest selection', () => {
  test('a source answer for an earlier selection does not repaint the form', async () => {
    const sourceA = deferred();
    const sourceB = deferred();
    const view = boot({
      plans: [PLAN_A, PLAN_B],
      sources: { plan_a: sourceA, plan_b: sourceB },
    });
    // The first plan is selected on load; its source is still on the way.
    await vi.waitFor(() => expect(view.calls('GET', '/plans/plan_a/source')).toHaveLength(1));

    const rowB = /** @type {HTMLElement} */ (
      document.querySelector('#plan-tree button[data-plan-name="plan_b"]')
    );
    rowB.click();
    await vi.waitFor(() => expect(view.calls('GET', '/plans/plan_b/source')).toHaveLength(1));

    // B answers first, then A.
    sourceB.resolve(sourceOf(PLAN_B));
    await settle();
    sourceA.resolve(sourceOf(PLAN_A));
    await settle();

    expect(view.title()).toBe('plan_b');
    expect(view.numInput().value).toBe('5');

    await vi.waitFor(() => expect(view.queueButton().disabled).toBe(false));
    view.queueButton().click();
    await vi.waitFor(() => expect(view.calls('POST', '/queue/items')).toHaveLength(1));

    const patches = view.calls('PATCH', '/draft');
    expect(patches).toHaveLength(1);
    expect(patches[0].body.plan_name).toBe('plan_b');
    expect(patches[0].body.plan_args_patch).toEqual({ num: 5 });
  });
});

describe('plans view — the enqueue stops when the screen and the draft differ', () => {
  test('a value the form cannot read is reported on its field, and the add sends nothing', async () => {
    const view = boot({ plans: [PLAN_B], sources: { plan_b: sourceOf(PLAN_B) } });
    await vi.waitFor(() => expect(view.queueButton().disabled).toBe(false));

    const input = view.numInput();
    input.value = '3.7';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(input.getAttribute('aria-invalid')).toBe('true');
    // The browser shows a field's own message when its validity is reported.
    const reported = vi.fn();
    input.addEventListener('invalid', reported);

    view.queueButton().click();
    await settle();

    expect(reported).toHaveBeenCalledTimes(1);
    expect(input.validationMessage).toBe('Enter a whole number.');
    expect(view.calls('PATCH', '/draft')).toHaveLength(0);
    expect(view.calls('POST', '/queue/items')).toHaveLength(0);
    expect(view.banner().hidden).toBe(false);
    expect(view.banner().className).toContain('banner-err');
    expect(view.banner().textContent).toBe('num: Enter a whole number.');
  });

  test('bound to the draft, an edit the bridge refused stops the add with its sentence', async () => {
    const draft = { draft: { plan_name: 'plan_b', plan_args: { num: 5 } }, revision: 1 };
    const view = boot({
      plans: [PLAN_B],
      sources: { plan_b: sourceOf(PLAN_B) },
      draft,
      patch: () => ({
        status: 422,
        body: { detail: { field: 'num', error: 'Input should be less than or equal to 10' } },
      }),
    });
    view.stream.emit({ type: 'hello', ...draft });
    await vi.waitFor(() =>
      expect(/** @type {HTMLElement} */ (document.getElementById('draft-indicator')).hidden).toBe(
        false
      )
    );
    await vi.waitFor(() => expect(view.queueButton().disabled).toBe(false));

    const input = view.numInput();
    input.value = '999';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    // The debounced flush sends the edit, and the bridge refuses it.
    await vi.waitFor(() => expect(view.calls('PATCH', '/draft')).toHaveLength(1));
    await settle();

    // Nothing is pending any more: the click flushes nothing new.
    view.queueButton().click();
    await vi.waitFor(() => expect(view.banner().hidden).toBe(false));

    expect(view.calls('PATCH', '/draft')).toHaveLength(1);
    expect(view.calls('POST', '/queue/items')).toHaveLength(0);
    expect(view.banner().className).toContain('banner-err');
    expect(view.banner().textContent).toBe(
      'could not stage the draft: num: Input should be less than or equal to 10'
    );
    // The typed value stays on screen.
    expect(input.value).toBe('999');
  });
});
