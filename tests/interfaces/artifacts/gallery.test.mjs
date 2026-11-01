// @ts-check
/**
 * The gallery page entry (gallery.js): which mode follows the panel's event
 * stream, and how the embedded viewer applies the hub's change messages.
 *   npx vitest run tests/interfaces/artifacts/gallery.test.mjs
 *
 * Standalone, the page opens `/api/events` itself. Embedded (`?embedded=true`)
 * it opens no stream: each one holds one of the browser's six connections to
 * the hub's origin, which every panel iframe shares, so the hub forwards the
 * shown artifact's `osprey-artifact-changed` instead.
 *
 * gallery.js wires itself on import, so each test sets the URL, imports a
 * fresh copy (vi.resetModules) and reads the state through the same copy of
 * state.js. Seams mocked: the preview and sidebar renderers, the browse
 * layout, the ⋯ menu, the pick bar, print, the logbook composer, the
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
vi.mock('/design-system/js/components/osprey-display-menu.js', () => ({}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/preview.js', () => ({
  createPreviewRenderer: () => {
    mocks.preview = {
      renderPreview: vi.fn(),
      isFullscreen: () => false,
      enterFullscreen: vi.fn(),
      exitFullscreen: vi.fn(),
      noteNewArtifactArrival: vi.fn(),
      updateNewArtifactBadge: vi.fn(),
      setPinned: vi.fn(),
      deleteArtifact: vi.fn(),
      setAsFocus: vi.fn(),
    };
    return mocks.preview;
  },
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/render.js', () => ({
  createSidebarRenderer: () => ({
    renderSidebar: () => {},
    getBrowseMode: () => 'tree',
    setBrowseMode: () => {},
    getSidebarLayout: () => 'list',
    setSidebarLayout: () => {},
  }),
  isNearListEnd: () => false,
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/browse-layout.js', () => ({
  initBrowseLayout: () => {},
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/sidebar-menu.js', () => ({
  initSidebarMenu: () => {},
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/pick-bar.js', () => ({
  mountPickBar: () => ({ render: () => {}, clear: () => false }),
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
  pinned: false,
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

/** @param {string} url */
function route(url) {
  const path = new URL(url, 'http://gallery').pathname;
  if (path === '/api/artifacts') return reply({ artifacts: listed, next_cursor: null });
  if (path === '/api/focus') return reply({ focused: false, artifact: null });
  const one = path.match(/^\/api\/artifacts\/(.+)$/);
  const entry = one && listed.find((a) => a.id === decodeURIComponent(one[1]));
  return entry ? reply(entry) : reply({ detail: 'not found' }, 404);
}

/** Let every pending fetch and promise callback run. */
async function settle() {
  for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Load the page at `search` and let its first fetches land.
 * @param {string} search
 */
async function load(search) {
  window.history.replaceState({}, '', `/${search}`);
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
  fetchMock = vi.fn(async (/** @type {string} */ url) => route(url));
  vi.stubGlobal('fetch', fetchMock);
  vi.stubGlobal('requestAnimationFrame', () => 0);
  events = stubEventSource();
  // The page's own slots gallery.js reads on boot (index.html); the embedded
  // page keeps them too, hidden.
  document.body.innerHTML = `
    <div id="browse-sidebar">
      <input id="search" type="search">
      <div id="sidebar-body"></div>
    </div>
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
  test('standalone, the page opens the panel stream and reads the focus', async () => {
    await load('');

    expect(events.opened.map((source) => source.url)).toEqual(['/api/events']);
    expect(paths()).toContain('/api/focus');
  });

  test('embedded, the page opens no stream and leaves the focus to the hub', async () => {
    await load('?embedded=true');

    expect(events.opened).toEqual([]);
    expect(paths()).toEqual(['/api/artifacts']);
  });

  test('standalone, an update frame re-renders the shown artifact', async () => {
    await load('');
    state.setSelectedArtifact(state.getArtifacts()[0]);
    mocks.preview.renderPreview.mockClear();

    events.emit({ type: 'artifact_updated', ...art('a1', { pinned: true }) });

    expect(state.getSelectedArtifact()).toEqual(art('a1', { pinned: true }));
    expect(mocks.preview.renderPreview).toHaveBeenCalledTimes(1);
  });
});

describe('embedded: the hub says when the shown artifact changed', () => {
  /** Load the embedded page and have the hub show a1. */
  async function showA1() {
    await load('?embedded=true');
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
      artifact: art('a1', { pinned: true }),
    });

    expect(state.getSelectedArtifact()).toEqual(art('a1', { pinned: true }));
    expect(state.getArtifacts().find((a) => a.id === 'a1')?.pinned).toBe(true);
    expect(mocks.preview.renderPreview).toHaveBeenCalledTimes(1);
  });

  test('an update of an artifact not shown re-renders nothing', async () => {
    await showA1();

    fromHub({
      type: 'osprey-artifact-changed',
      artifact_id: 'a2',
      change: 'updated',
      artifact: art('a2', { pinned: true }),
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
