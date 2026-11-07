// @ts-check
/**
 * The gallery page entry (gallery.js): how the viewer applies the hub's change
 * messages, and which listing it holds.
 *   npx vitest run tests/interfaces/artifacts/gallery.test.mjs
 *
 * The page opens no event stream: each one holds one of the browser's six
 * connections to the hub's origin, which every panel iframe shares, so the hub
 * forwards the shown artifact's `osprey-artifact-changed` instead.
 *
 * gallery.js wires itself on import, so each test sets the URL, imports a
 * fresh copy (vi.resetModules) and reads the state through the same copy of
 * state.js. Seams mocked: the preview renderer, print, the logbook composer, the
 * timeseries view, the type registry and the theme runtime. The real
 * viewer.js and state.js run, over a stubbed fetch and EventSource. The
 * listeners each import adds to window and document are removed after the
 * test, so an earlier copy never answers a later test's message.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { stubEventSource } from '../_support/event-source.mjs';

const JS = '../../../src/osprey/interfaces/artifacts/static/js';

const mocks = vi.hoisted(() => ({
  /** The preview renderer the latest import made. */
  preview: /** @type {any} */ (null),
  openComposeModal: vi.fn(),
}));

vi.mock('/design-system/js/theme-manager.js', () => ({
  getTheme: () => 'dark',
  initTheme: () => {},
  subscribe: () => {},
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/preview.js', () => ({
  createPreviewRenderer: () => {
    mocks.preview = {
      renderPreview: vi.fn(),
      deleteArtifact: vi.fn(),
    };
    return mocks.preview;
  },
}));

vi.mock('../../../src/osprey/interfaces/artifacts/static/js/print.js', () => ({
  printArtifact: () => {},
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/logbook.js', () => ({
  openComposeModal: mocks.openComposeModal,
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/timeseries.js', () => ({
  renderTimeseriesView: () => {},
  restyleMountedCharts: () => {},
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/types.js', () => ({
  initTypeRegistry: () => Promise.resolve(),
}));

/**
 * @param {string} id
 * @param {object} [extra]
 */
const art = (id, extra = {}) => ({
  id,
  artifact_type: 'markdown',
  title: `Artifact ${id}`,
  filename: `${id}.md`,
  ...extra,
});

/** @type {ReturnType<typeof stubEventSource>} */
let events;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;
/** @type {typeof import('../../../src/osprey/interfaces/artifacts/static/js/state.js')} */
let state;
/** The listeners the import added, removed again after the test. */
/** @type {[EventTarget, string, any, any][]} */
let added;

/** @param {unknown} body @param {number} [status] */
const reply = (body, status = 200) => ({
  ok: status < 400,
  status,
  json: async () => body,
  text: async () => JSON.stringify(body),
});

/** The list the server holds, served by `GET /api/artifacts`. */
let listed = [art('a1'), art('a2')];

/** The session view's list and the transcript its key resolves to, served by `POST /api/artifacts/session-list`. */
let sessionListed = [art('s1', { owner: 'T1', operator_visible: true })];
/** @type {string|null} */
let transcript = 'T1';

/** @param {string} url @param {any} [init] */
function route(url, init) {
  const path = new URL(url, 'http://gallery').pathname;
  if (path === '/api/artifacts/session-list' && init?.method === 'POST') {
    return reply({ artifacts: sessionListed, next_cursor: null, transcript, ledger: 'ok' });
  }
  if (path === '/api/artifacts') return reply({ artifacts: listed, next_cursor: null });
  const one = path.match(/^\/api\/artifacts\/(.+)$/);
  const entry = one && listed.find((a) => a.id === decodeURIComponent(one[1]));
  return entry ? reply(entry) : reply({ detail: 'not found' }, 404);
}

/** Let every pending fetch and promise callback run. */
async function settle() {
  for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Load the page and let its first fetches land.
 */
async function load() {
  window.history.replaceState({}, '', '/?embedded=true');
  await import(`${JS}/gallery.js`);
  state = await import(`${JS}/state.js`);
  await settle();
}

/** @param {unknown} data */
function fromHub(data) {
  window.dispatchEvent(new MessageEvent('message', { data, origin: window.location.origin }));
}

/** Every request path the page made. */
function paths() {
  return fetchMock.mock.calls.map(([url]) => new URL(String(url), 'http://gallery').pathname);
}

beforeEach(() => {
  vi.resetModules();
  listed = [art('a1'), art('a2')];
  mocks.preview = null;
  mocks.openComposeModal.mockClear();
  added = [];
  for (const target of [window, document]) {
    const original = target.addEventListener.bind(target);
    vi.spyOn(target, 'addEventListener').mockImplementation((type, listener, options) => {
      added.push([target, type, listener, options]);
      original(type, listener, options);
    });
  }
  sessionListed = [art('s1', { owner: 'T1', operator_visible: true })];
  transcript = 'T1';
  fetchMock = vi.fn(async (/** @type {string} */ url, /** @type {any} */ init) => route(url, init));
  vi.stubGlobal('fetch', fetchMock);
  vi.stubGlobal('requestAnimationFrame', () => 0);
  events = stubEventSource();
  // The page's own slots (index.html).
  document.body.innerHTML = `
    <div id="preview-empty"></div>
    <div id="preview-content"></div>
  `;
});

afterEach(() => {
  vi.restoreAllMocks();
  for (const [target, type, listener, options] of added) {
    target.removeEventListener(type, listener, options);
  }
  vi.unstubAllGlobals();
  document.body.replaceChildren();
  document.body.className = '';
  window.history.replaceState({}, '', '/');
});

describe('the event stream', () => {
  test('the page opens no stream and leaves the focus to the hub', async () => {
    await load();

    expect(events.opened).toEqual([]);
    expect(paths()).toEqual(['/api/artifacts']);
  });
});

describe('the hub says when the shown artifact changed', () => {
  /** Load the embedded page and have the hub show a1. */
  async function showA1() {
    await load();
    fromHub({ type: 'osprey-artifact-select', artifact_id: 'a1' });
    expect(state.getSelectedArtifact()?.id).toBe('a1');
    mocks.preview.renderPreview.mockClear();
  }

  test('an update re-renders it with the copy the hub sent', async () => {
    await showA1();

    fromHub({
      type: 'osprey-artifact-changed',
      artifact_id: 'a1',
      change: 'updated',
      artifact: art('a1', { title: 'Renamed' }),
    });

    expect(state.getSelectedArtifact()).toEqual(art('a1', { title: 'Renamed' }));
    expect(state.getArtifacts().find((a) => a.id === 'a1')?.title).toBe('Renamed');
    expect(mocks.preview.renderPreview).toHaveBeenCalledTimes(1);
  });

  test('an update of an artifact not shown re-renders nothing', async () => {
    await showA1();

    fromHub({
      type: 'osprey-artifact-changed',
      artifact_id: 'a2',
      change: 'updated',
      artifact: art('a2', { title: 'Renamed' }),
    });

    expect(state.getSelectedArtifact()?.id).toBe('a1');
    expect(mocks.preview.renderPreview).not.toHaveBeenCalled();
  });

  test('a delete drops it and empties the viewer', async () => {
    await showA1();

    fromHub({ type: 'osprey-artifact-changed', artifact_id: 'a1', change: 'deleted' });

    expect(state.getSelectedArtifact()).toBeNull();
    expect(state.getArtifacts().map((a) => a.id)).toEqual(['a2']);
    expect(mocks.preview.renderPreview).toHaveBeenCalledTimes(1);
  });

  test('a change from another origin is ignored', async () => {
    await showA1();

    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'osprey-artifact-changed', artifact_id: 'a1', change: 'deleted' },
        origin: 'https://evil.example',
      })
    );

    expect(state.getSelectedArtifact()?.id).toBe('a1');
    expect(mocks.preview.renderPreview).not.toHaveBeenCalled();
  });

  test('the logbook action reads the list again before the composer opens', async () => {
    await showA1();
    listed = [art('a3'), art('a1'), art('a2')];
    fetchMock.mockClear();

    fromHub({ type: 'osprey-artifact-action', artifact_id: 'a1', action: 'logbook' });
    expect(mocks.openComposeModal).not.toHaveBeenCalled();
    await settle();

    expect(paths()).toEqual(['/api/artifacts']);
    expect(state.getArtifacts().map((a) => a.id)).toEqual(['a3', 'a1', 'a2']);
    expect(mocks.openComposeModal).toHaveBeenCalledWith({ artifact_id: 'a1' });
  });
});

describe('the session view is the sidebar row', () => {
  /** Load the page and have the host name session key k1. */
  async function enterSession() {
    await load();
    fromHub({ type: 'osprey-session-change', session_id: 'k1' });
    await settle();
  }

  /** The session-list requests the page made. */
  const sessionLists = () =>
    fetchMock.mock.calls.filter(([url]) => String(url) === '/api/artifacts/session-list');

  test('no session id shows all sessions through GET /api/artifacts', async () => {
    await load();

    expect(paths()).toContain('/api/artifacts');
    expect(sessionLists()).toEqual([]);
    expect(state.getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);
  });

  test('a session key posts the exact gallery body and holds the transcript', async () => {
    await enterSession();

    expect(sessionLists()).toHaveLength(1);
    const [, init] = sessionLists()[0];
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ key: 'k1', search: '', cursor: null, runs: 'grouped' });
    expect(state.getHeldTranscript()).toBe('T1');
    expect(state.getArtifacts().map((a) => a.id)).toEqual(['s1']);
  });

  test('a same-key change reads the list again', async () => {
    await enterSession();
    fetchMock.mockClear();

    fromHub({ type: 'osprey-session-change', session_id: 'k1' });
    await settle();

    expect(state.getCurrentSessionId()).toBe('k1');
    expect(sessionLists()).toHaveLength(1);
  });

  test('a new key lists that session', async () => {
    await enterSession();
    fetchMock.mockClear();

    fromHub({ type: 'osprey-session-change', session_id: 'k2' });
    await settle();

    expect(JSON.parse(sessionLists()[0][1].body).key).toBe('k2');
  });
});
