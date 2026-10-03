/**
 * Unit tests for the Artifact Gallery's Simple layout (simple-view.js): the
 * latest-result card and the "Results from this session" list.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/simple-view.test.mjs
 *
 * simple-view.js reads state.js's module-singleton artifact/selection/focus
 * state (no vi.resetModules, just call the setters fresh per test) and formats
 * via types.js (real implementations). The list is filled the way production
 * fills it — one stubbed listing fetch through fetchArtifacts() — because
 * state.js has no setter for the server total the list count and "Show all"
 * read. onTimeseriesNeeded is the one spy: rendering a timeseries chart/table
 * is timeseries.js's job, so only the hand-off matters here.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  fetchArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  setFocusedArtifact,
  fileUrl,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';
import { formatFullTime, formatTime } from '../../../src/osprey/interfaces/artifacts/static/js/types.js';
import { createSimpleView } from '../../../src/osprey/interfaces/artifacts/static/js/simple-view.js';
import { qs, byId } from '../_support/dom.mjs';

/** The Simple section of artifacts/static/index.html, ids and classes as served. */
function mountFixture() {
  document.body.innerHTML = `
    <section id="view-artifacts-simple">
      <div class="simple-empty" id="simple-empty"></div>
      <div class="simple-result hidden" id="simple-result">
        <span class="simple-result-title" id="simple-result-title"></span>
        <span class="simple-badge-new" id="simple-result-badge" hidden>NEW</span>
        <a class="btn btn-primary" id="simple-open-full" target="_blank" rel="noopener">Open full size</a>
        <a class="btn btn-secondary" id="simple-save">Save</a>
        <div class="simple-result-preview preview-viewport" id="simple-result-preview"></div>
        <div class="simple-result-caption" id="simple-result-caption"></div>
      </div>
      <span class="simple-list-count" id="simple-list-count">0</span>
      <button class="simple-show-all" id="simple-show-all" hidden>Show all &rarr;</button>
      <div class="simple-list-body" id="simple-list-body"></div>
    </section>
  `;
}

/** A timestamp an hour after this page loaded: "new this session". */
const FRESH = new Date(Date.now() + 3_600_000).toISOString();

/** @returns {any} */
function makeArtifact(overrides = {}) {
  return {
    id: 'a1', title: 'Beam Profile', filename: 'beam_profile.png', artifact_type: 'plot_png',
    category: 'visualization', pinned: false, timestamp: '2026-07-01T10:00:00Z', size_bytes: 2048,
    ...overrides,
  };
}

/**
 * `n` plot artifacts, oldest first by timestamp, ids p1..pn.
 * @param {number} n
 * @returns {any[]}
 */
function makeMany(n) {
  return Array.from({ length: n }, (_, i) =>
    makeArtifact({ id: `p${i + 1}`, title: `Plot ${i + 1}`, timestamp: `2026-07-0${1 + Math.floor(i / 9)}T${String(10 + (i % 9)).padStart(2, '0')}:00:00Z` })
  );
}

/**
 * Fill state.js the way production does: one listing fetch holding `artifacts`
 * with a server total of `total` (defaults to the list's length).
 * @param {any[]} artifacts
 * @param {number} [total]
 */
async function seed(artifacts, total = artifacts.length) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({ artifacts, total, next_cursor: null }),
  }));
  await fetchArtifacts();
}

function rows() {
  return Array.from(document.querySelectorAll('#simple-list-body .simple-list-item'));
}

/** @type {import('vitest').Mock<(container: HTMLElement, artifact: any) => void>} */
let onTimeseriesNeeded;

beforeEach(async () => {
  mountFixture();
  document.documentElement.setAttribute('data-ui-mode', 'simple');
  setSelectedArtifact(null);
  setFocusedArtifact(null);
  onTimeseriesNeeded = vi.fn();
  await seed([]);
});

afterEach(() => {
  document.documentElement.removeAttribute('data-ui-mode');
  vi.unstubAllGlobals();
});

function view() {
  return createSimpleView({ onTimeseriesNeeded });
}

function hidden(/** @type {string} */ id) {
  return byId(id).classList.contains('hidden');
}

function title() {
  return byId('simple-result-title').textContent;
}

function selectedFlags() {
  return rows().map((r) => r.classList.contains('selected'));
}

describe('render', () => {
  test('writes nothing while the page is not in Simple mode', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    await seed([makeArtifact()]);
    view().render();

    expect(rows()).toEqual([]);
    expect(hidden('simple-empty')).toBe(false);
    expect(hidden('simple-result')).toBe(true);
  });

  test('with no artifacts shows the empty card, a count of 0 and no rows', () => {
    view().render();

    expect(hidden('simple-empty')).toBe(false);
    expect(hidden('simple-result')).toBe(true);
    expect(byId('simple-list-count').textContent).toBe('0');
    expect(byId('simple-show-all').hidden).toBe(true);
    expect(rows()).toEqual([]);
  });
});

describe('the result card', () => {
  test('shows the newest artifact when nothing is selected or focused, skipping the shipped example', async () => {
    await seed([
      makeArtifact({ id: 'old', title: 'Old', timestamp: '2026-07-01T10:00:00Z' }),
      makeArtifact({ id: 'demo', title: 'Demo', origin: 'demo', timestamp: '2026-07-03T10:00:00Z' }),
      makeArtifact({ id: 'new', title: 'New', timestamp: '2026-07-02T10:00:00Z' }),
    ]);
    view().render();

    expect(hidden('simple-empty')).toBe(true);
    expect(hidden('simple-result')).toBe(false);
    expect(title()).toBe('New');
  });

  test('shows the shipped example only when it is all there is', async () => {
    await seed([makeArtifact({ id: 'demo', title: 'Example', origin: 'demo' })]);
    view().render();

    expect(title()).toBe('Example');
  });

  test('prefers the selected artifact, then the focused one, when the list holds them', async () => {
    const a = makeArtifact({ id: 'a', title: 'A', timestamp: '2026-07-03T10:00:00Z' });
    const b = makeArtifact({ id: 'b', title: 'B', timestamp: '2026-07-02T10:00:00Z' });
    const c = makeArtifact({ id: 'c', title: 'C', timestamp: '2026-07-01T10:00:00Z' });
    await seed([a, b, c]);
    const simple = view();

    setFocusedArtifact(c);
    simple.render();
    expect(title()).toBe('C');

    setSelectedArtifact(b);
    simple.render();
    expect(title()).toBe('B');

    setSelectedArtifact(makeArtifact({ id: 'gone' }));
    setFocusedArtifact(makeArtifact({ id: 'gone-too' }));
    simple.render();
    expect(title()).toBe('A');
  });

  test('links Open full size to the artifact page (theme-stamped) and Save to the file download', async () => {
    const a = makeArtifact({ id: 'a 1', filename: 'beam profile.png' });
    await seed([a]);
    view().render();

    const open = byId('simple-open-full');
    expect(open.getAttribute('href')).toBe(fileUrl(a));
    expect(open.hasAttribute('data-theme-link')).toBe(true);
    const save = byId('simple-save');
    expect(save.getAttribute('href')).toBe('/files/a%201/beam%20profile.png');
    expect(save.getAttribute('download')).toBe('beam profile.png');
  });

  test('opens a markdown artifact through its rendered page', async () => {
    await seed([makeArtifact({ id: 'md', artifact_type: 'markdown', filename: 'notes.md' })]);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, text: () => Promise.resolve('# hi') }));
    view().render();

    expect(byId('simple-open-full').getAttribute('href')).toBe('/api/markdown/md/rendered');
  });

  test('captions with the description, else the title and full time', async () => {
    const simple = view();
    await seed([makeArtifact({ description: 'Orbit after correction' })]);
    simple.render();
    expect(byId('simple-result-caption').textContent).toBe('Orbit after correction');

    const bare = makeArtifact({ id: 'b', title: 'Bare', description: '' });
    await seed([bare]);
    simple.render();
    expect(byId('simple-result-caption').textContent).toBe(`Bare · ${formatFullTime(bare.timestamp)}`);
  });

  test('marks an artifact created since page load NEW, and nothing older or shipped', async () => {
    const simple = view();
    await seed([makeArtifact({ id: 'fresh', timestamp: FRESH })]);
    simple.render();
    expect(byId('simple-result-badge').hidden).toBe(false);

    await seed([makeArtifact({ id: 'old' })]);
    simple.render();
    expect(byId('simple-result-badge').hidden).toBe(true);

    await seed([makeArtifact({ id: 'demo', origin: 'demo', timestamp: FRESH })]);
    simple.render();
    expect(byId('simple-result-badge').hidden).toBe(true);
  });

  test('renders the result through the shared viewport dispatch', async () => {
    const a = makeArtifact();
    await seed([a]);
    view().render();

    const img = qs(byId('simple-result-preview'), 'img', HTMLImageElement);
    expect(img.getAttribute('src')).toBe(fileUrl(a));
  });

  test('hands a timeseries artifact to onTimeseriesNeeded with its mounted container', async () => {
    const ts = makeArtifact({
      id: 'ts', title: 'Orbit history', filename: 'orbit.json', artifact_type: 'json',
      category: 'archiver_data', metadata: { data_type: 'timeseries', data_file: 'orbit.jsonl' },
    });
    await seed([ts]);
    view().render();

    expect(onTimeseriesNeeded).toHaveBeenCalledTimes(1);
    const [container, artifact] = onTimeseriesNeeded.mock.calls[0];
    expect(container).toBe(qs(byId('simple-result-preview'), '.ts-viewport-container'));
    expect(artifact).toBe(ts);
  });
});

describe('the session list', () => {
  test('lists newest first, marks the shown artifact selected, and names each row by id, title and time', async () => {
    await seed([
      makeArtifact({ id: 'b', title: 'Second', timestamp: '2026-07-02T10:00:00Z' }),
      makeArtifact({ id: 'a', title: 'First', timestamp: '2026-07-01T10:00:00Z' }),
      makeArtifact({ id: 'c', title: 'Third', timestamp: '2026-07-03T10:00:00Z' }),
    ]);
    view().render();

    expect(rows().map((r) => r.getAttribute('data-id'))).toEqual(['c', 'b', 'a']);
    expect(selectedFlags()).toEqual([true, false, false]);
    expect(qs(rows()[1], '.simple-list-item-name').textContent).toBe('Second');
    expect(qs(rows()[1], '.simple-list-item-time').textContent).toBe(formatTime('2026-07-02T10:00:00Z'));
    expect(byId('simple-list-count').textContent).toBe('3');
    expect(byId('simple-show-all').hidden).toBe(true);
  });

  test('escapes titles and ids in the rows', async () => {
    await seed([makeArtifact({ id: 'x"y', title: '<b>bold</b>' })]);
    view().render();

    const row = rows()[0];
    expect(row.getAttribute('data-id')).toBe('x"y');
    expect(qs(row, '.simple-list-item-name').textContent).toBe('<b>bold</b>');
    expect(row.querySelector('b')).toBeNull();
  });

  test('badges a row created since page load NEW', async () => {
    await seed([makeArtifact({ id: 'fresh', timestamp: FRESH }), makeArtifact({ id: 'old' })]);
    view().render();

    expect(rows()[0].querySelector('.simple-badge-new')).not.toBeNull();
    expect(rows()[1].querySelector('.simple-badge-new')).toBeNull();
  });

  test('shows the six newest with "Show all" when the server total is larger', async () => {
    await seed(makeMany(8));
    view().render();

    expect(rows().map((r) => r.getAttribute('data-id'))).toEqual(['p8', 'p7', 'p6', 'p5', 'p4', 'p3']);
    expect(byId('simple-list-count').textContent).toBe('8');
    expect(byId('simple-show-all').hidden).toBe(false);
  });

  test('"Show all" reveals every held row and stays expanded across re-renders', async () => {
    await seed(makeMany(8));
    const simple = view();
    simple.render();
    byId('simple-show-all').click();
    expect(rows()).toHaveLength(8);

    simple.render();
    expect(rows()).toHaveLength(8);
  });

  test('offers "Show all" from the server total, not the rows held', async () => {
    await seed(makeMany(3), 12);
    view().render();

    expect(rows()).toHaveLength(3);
    expect(byId('simple-list-count').textContent).toBe('12');
    expect(byId('simple-show-all').hidden).toBe(false);
  });

  test('clicking a row selects that artifact and promotes it to the card', async () => {
    await seed(makeMany(3));
    view().render();
    qs(rows()[2], '.simple-list-item-name').click();

    expect(getSelectedArtifact()?.id).toBe('p1');
    expect(title()).toBe('Plot 1');
    expect(selectedFlags()).toEqual([false, false, true]);
  });

  test('a click outside any row changes nothing', async () => {
    await seed(makeMany(2));
    view().render();
    byId('simple-list-body').click();

    expect(getSelectedArtifact()).toBeNull();
    expect(title()).toBe('Plot 2');
  });
});
