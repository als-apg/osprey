/**
 * Unit tests for the Artifact Gallery timeseries preview (timeseries.js:
 * the lazy Plotly loader, `renderTimeseriesView`
 * (toolbar/channel-toggle/export), `_tsChartTheme`, `renderTimeseriesChart`,
 * and `renderTimeseriesTable`).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal — mirrors preview.test.mjs/render.test.mjs.
 *   npx vitest run tests/interfaces/artifacts/timeseries.test.mjs
 *
 * `Plotly` is a vendored classic-script global (see vendor-globals.d.ts),
 * stubbed via vi.stubGlobal like lattice_dashboard/render.test.mjs does for
 * its own Plotly usage. Unlike that module, this one lazily injects Plotly
 * via a `<script>` tag (`ensurePlotlyLoaded`) rather than assuming it's
 * already loaded — `stubScriptLoad()` below spies `document.head.appendChild`
 * so an injected `<script>` "loads" synchronously (next microtask) instead
 * of happy-dom attempting a real network fetch of the vendored path (which
 * would reject with a real ECONNREFUSED). `_plotlyLoaded` is a module
 * singleton (matches state.js/types.js's convention — no vi.resetModules),
 * so only the first test to actually load a chart exercises the script-load
 * path; later tests skip straight past it. The stub is installed in every
 * test's beforeEach regardless, since it's a no-op once already loaded.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { qs } from '../_support/dom.mjs';

import {
  renderTimeseriesView,
  renderTimeseriesChart,
  renderTimeseriesTable,
  _tsChartTheme,
} from '../../../src/osprey/interfaces/artifacts/static/js/timeseries.js';

/**
 * A minimal fake of the parts of `Response` these tests' fetch stubs expose.
 * @typedef {{ ok: boolean, status?: number, json?: () => Promise<unknown> }} FakeResponse
 */

/**
 * An injected `<script>` node as the loader mocks see it: the loader assigns
 * bare zero-arg `onload`/`onerror` callbacks, so narrow those handlers here.
 * @typedef {HTMLScriptElement & { onload: (() => void) | null, onerror: (() => void) | null }} InjectedScript
 */

/**
 * Fixture chart-format *channel* entry (one element of the `channels` array
 * `/api/artifacts/{id}/data?format=chart` returns). Each channel carries its
 * own `timestamps`/`values` -- no shared axis.
 */
function makeChartChannel(overrides = {}) {
  return {
    channel: 'SR:MAG:QF1:I',
    timestamps: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z'],
    values: [1.0, 1.5],
    total_points: 2,
    returned_points: 2,
    numeric: true,
    ...overrides,
  };
}

/**
 * The `summary` block app.py's chart branch derives from its channels.
 * `row_count` counts the *unioned* timestamp axis and is not derivable
 * client-side; it defaults to the fixture's shared two-timestamp axis.
 * @param {ReturnType<typeof makeChartChannel>[]} channels
 * @param {Record<string, any>} [overrides]
 */
function makeChartSummary(channels, overrides = {}) {
  return {
    total_points: channels.reduce((sum, ch) => sum + ch.total_points, 0),
    returned_points: channels.reduce((sum, ch) => sum + ch.returned_points, 0),
    downsampled: channels.some((ch) => ch.returned_points < ch.total_points),
    row_count: 2,
    ...overrides,
  };
}

/**
 * Fixture chart-format response (`/api/artifacts/{id}/data?format=chart`).
 * @param {Record<string, any>} [overrides]
 */
function makeChartData(overrides = {}) {
  const channels = overrides.channels ?? [
    makeChartChannel({ channel: 'SR:MAG:QF1:I', values: [1.0, 1.5] }),
    makeChartChannel({ channel: 'SR:MAG:QF2:I', values: [2.0, 2.5] }),
  ];
  return {
    channels,
    metadata: {},
    summary: makeChartSummary(channels),
    ...overrides,
  };
}

/** Fixture table-format response (`/api/artifacts/{id}/data?format=table`). */
function makeTableData(overrides = {}) {
  return {
    columns: ['SR:MAG:QF1:I', 'SR:MAG:QF2:I'],
    index: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z'],
    data: [[1.0, 2.0], [1.5, 2.5]],
    total_rows: 2,
    offset: 0,
    limit: 50,
    returned_rows: 2,
    ...overrides,
  };
}

/**
 * Route the shared fetch mock by URL: format=chart -> chartResp,
 * format=table -> tableResp. Both default to a resolved ok response over
 * the matching fixture.
 * @param {{ chartResp?: Promise<FakeResponse>, tableResp?: Promise<FakeResponse> }} [opts]
 */
function stubFetchRouting({ chartResp, tableResp } = {}) {
  vi.stubGlobal('fetch', vi.fn((url) => {
    if (url.includes('format=chart')) {
      return chartResp ?? Promise.resolve({ ok: true, json: () => Promise.resolve(makeChartData()) });
    }
    if (url.includes('format=table')) {
      return tableResp ?? Promise.resolve({ ok: true, json: () => Promise.resolve(makeTableData()) });
    }
    return Promise.reject(new Error('unexpected fetch URL: ' + url));
  }));
}

/**
 * Stub the global `fetch` to resolve ok with `makeTableData(overrides)` for
 * every call — the common `renderTimeseriesTable` setup. Returns the mock
 * for tests that assert on fetched URLs.
 */
function stubTableFetch(overrides = {}) {
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(makeTableData(overrides)) });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/**
 * Make an injected `<script>` "load" on the next microtask instead of
 * happy-dom actually processing it — happy-dom's default browser settings
 * disable script file loading outright (`disableJavaScriptFileLoading`),
 * which synchronously fires the script's `error` event the moment it's
 * inserted (there's no network round-trip to intercept, since happy-dom
 * never attempts one). So for `<script>` nodes this skips real insertion
 * entirely rather than delegating to the original `appendChild`.
 */
function stubScriptLoad() {
  const originalAppendChild = document.head.appendChild.bind(document.head);
  vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
    const script = /** @type {InjectedScript} */ (node);
    if (node && script.tagName === 'SCRIPT') {
      queueMicrotask(() => script.onload && script.onload());
      return node;
    }
    return originalAppendChild(node);
  });
}

/**
 * Sets the chart-theme CSS custom properties (+ the sentinel) inline.
 * @param {{ bgPrimary?: string, paperBg: string, plotBg: string, axisText: string, grid: string, border: string }} vars
 */
function setChartVars({ bgPrimary = '#000', paperBg, plotBg, axisText, grid, border }) {
  const root = document.documentElement.style;
  root.setProperty('--bg-primary', bgPrimary);
  root.setProperty('--chart-paper-bg', paperBg);
  root.setProperty('--chart-plot-bg', plotBg);
  root.setProperty('--chart-axis-text', axisText);
  root.setProperty('--chart-grid', grid);
  root.setProperty('--chart-axis-line', border);
}

beforeEach(() => {
  vi.stubGlobal('Plotly', {
    newPlot: vi.fn((el) => { el.data = []; }),
    restyle: vi.fn(),
    relayout: vi.fn(),
    update: vi.fn(),
  });
  stubScriptLoad();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  ['--bg-primary', '--chart-paper-bg', '--chart-plot-bg', '--chart-axis-text', '--chart-grid', '--chart-axis-line']
    .forEach((v) => document.documentElement.style.removeProperty(v));
});

describe('_tsChartTheme', () => {
  test('reflects a dark-theme set of chart CSS custom properties', () => {
    setChartVars({ paperBg: '#0b0f14', plotBg: '#11161d', axisText: '#e6edf3', grid: '#232a33', border: '#2d3542' });

    const t = _tsChartTheme();

    expect(t.paper_bgcolor).toBe('#0b0f14');
    expect(t.plot_bgcolor).toBe('#11161d');
    expect(t.font.color).toBe('#e6edf3');
    expect(t.xaxis.gridcolor).toBe('#232a33');
    expect(t.yaxis.gridcolor).toBe('#232a33');
    expect(t.line).toBe('#2d3542');
    expect(t.legendBg).toBe('#0b0f14');
    expect(t.legendBorder).toBe('#2d3542');
  });

  test('reflects a light-theme set of chart CSS custom properties', () => {
    setChartVars({ paperBg: '#ffffff', plotBg: '#f6f8fa', axisText: '#1f2328', grid: '#d0d7de', border: '#c9d1d9' });

    const t = _tsChartTheme();

    expect(t.paper_bgcolor).toBe('#ffffff');
    expect(t.plot_bgcolor).toBe('#f6f8fa');
    expect(t.font.color).toBe('#1f2328');
    expect(t.xaxis.gridcolor).toBe('#d0d7de');
    expect(t.line).toBe('#c9d1d9');
    expect(t.legendBg).toBe('#ffffff');
  });

  test('falls back to the grid color for the line when --border-default is unset', () => {
    setChartVars({ paperBg: '#0b0f14', plotBg: '#11161d', axisText: '#e6edf3', grid: '#232a33', border: '' });

    const t = _tsChartTheme();

    expect(t.line).toBe('#232a33');
    expect(t.legendBorder).toBe('');
  });
});

describe('renderTimeseriesView', () => {
  /** @type {HTMLElement} */
  let container;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
  });

  test('info-bar totals come from the server summary, not a client-side re-derivation', async () => {
    // The per-channel numbers are deliberately inconsistent with `summary` so
    // the assertion can only pass if the server's figures are the ones shown.
    stubFetchRouting({
      chartResp: Promise.resolve({
        ok: true,
        json: () => Promise.resolve(makeChartData({
          summary: { total_points: 999, returned_points: 40, downsampled: true, row_count: 7 },
        })),
      }),
    });

    await renderTimeseriesView(container, { id: 'ts1' });

    expect(qs(container, '.ts-badge-points').textContent).toContain('999');
    expect(qs(container, '.ts-badge-downsampled').textContent).toContain('40');
    expect(qs(container, '.ts-badge-rows').textContent).toContain('7');
  });

  test('shows a loading placeholder, then the info bar, toolbar, chart, and table containers on success', async () => {
    stubFetchRouting();

    const pending = renderTimeseriesView(container, { id: 'ts1' });
    expect(container.querySelector('.ts-loading')).not.toBeNull();

    await pending;

    expect(container.querySelector('.ts-info-bar')).not.toBeNull();
    expect(container.querySelectorAll('.ts-badge-channel').length).toBe(2);
    // summary.total_points: two channels, 2 points each = 4.
    expect(qs(container, '.ts-badge-points').textContent).toContain('4');
    expect(container.querySelector('[data-ts-chart]')).not.toBeNull();
    expect(container.querySelector('[data-ts-table]')).not.toBeNull();
    expect(container.querySelectorAll('.ts-ch-toggle').length).toBe(2);
    // The `[data-ts-table]` div is emitted whether or not the table renders,
    // so assert real populated rows.
    expect(container.querySelectorAll('.ts-data-table tbody tr').length).toBe(2);
  });

  test('shows a downsampled badge only when at least one channel reports downsampling, summed across channels', async () => {
    stubFetchRouting({
      chartResp: Promise.resolve({
        ok: true,
        json: () => Promise.resolve(makeChartData({
          channels: [
            makeChartChannel({ channel: 'A', returned_points: 500, total_points: 5000 }),
            makeChartChannel({ channel: 'B', returned_points: 2, total_points: 2 }),
          ],
        })),
      }),
    });

    await renderTimeseriesView(container, { id: 'ts1' });

    // 500 (downsampled channel) + 2 (untouched channel) = 502.
    expect(qs(container, '.ts-badge-downsampled').textContent).toContain('502');
  });

  test('does not show a downsampled badge when no channel was downsampled', async () => {
    stubFetchRouting();

    await renderTimeseriesView(container, { id: 'ts1' });

    // Positive anchor: confirming the points badge rendered proves this is
    // "no downsampling happened", not "the view died".
    expect(qs(container, '.ts-badge-points')).not.toBeNull();
    expect(container.querySelector('.ts-badge-downsampled')).toBeNull();
  });

  test('on a chart-fetch failure: shows the failure fallback', async () => {
    stubFetchRouting({ chartResp: Promise.resolve({ ok: false, status: 500 }) });

    await renderTimeseriesView(container, { id: 'ts1' });

    expect(container.textContent).toContain('Failed to load timeseries data');
  });

  test('the toolbar marks a channel non-numeric from the `numeric` flag alone, even when its values look like numbers', async () => {
    // A dtype check that re-derives from the values instead of trusting
    // `numeric: false` would treat 0/1 as numeric and never mark the toggle.
    stubFetchRouting({
      chartResp: Promise.resolve({
        ok: true,
        json: () => Promise.resolve(makeChartData({
          channels: [
            makeChartChannel({
              channel: 'SR:RF:INTERLOCK_STATE',
              values: [0, 1],
              numeric: false,
            }),
          ],
        })),
      }),
    });

    await renderTimeseriesView(container, { id: 'ts1' });

    const toggle = qs(container, '.ts-ch-toggle');
    expect(toggle.querySelector('.ts-ch-axis-tag')).not.toBeNull();
    expect(toggle.getAttribute('title')).toContain('status/enum');
  });

  test('an absent `numeric` flag means numeric at every site -- only `numeric === false` is an enum channel', async () => {
    // An absent JSON key reads as undefined and means numeric; every site must
    // test `numeric === false`, never `!ch.numeric`.
    stubFetchRouting({
      chartResp: Promise.resolve({
        ok: true,
        json: () => Promise.resolve(makeChartData({
          channels: [
            makeChartChannel({ channel: 'SR:MAG:QF1:I', numeric: undefined }),
            makeChartChannel({ channel: 'SR:MAG:QF2:I', values: [2.0, 2.5], numeric: undefined }),
          ],
        })),
      }),
    });

    await renderTimeseriesView(container, { id: 'ts1' });

    const toggles = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.ts-ch-toggle'));
    // Site: the toolbar marker/title.
    expect(container.querySelector('.ts-ch-axis-tag')).toBeNull();
    expect(toggles[0].getAttribute('aria-label')).toBe('SR:MAG:QF1:I');
    // Site: the trace's axis routing.
    const [, traces, layout] = Plotly.newPlot.mock.calls[0];
    expect(traces[0].yaxis).toBeUndefined();
    expect(traces[0].y).toEqual([1.0, 1.5]);
    expect(traces[0].type).toBe('scattergl');
    // Site: whether the layout gets a secondary axis at all.
    expect(layout.yaxis2).toBeUndefined();
    // Site: the toggle handler's show/hide of that axis.
    toggles[0].click();
    expect(Plotly.relayout).not.toHaveBeenCalled();
    // ...and the same answer drives Reset Zoom's per-axis autorange keys.
    qs(container, '[data-action="zoom-reset"]').click();
    expect(Plotly.relayout).toHaveBeenLastCalledWith(
      container.querySelector('[data-ts-chart]'),
      { 'xaxis.autorange': true, 'yaxis.autorange': true }
    );
  });

  test('the enum axis marker is kept out of the accessible name, which comes from aria-label', async () => {
    // The "R" glyph sits inside the button content and would land in the
    // accessible name; an explicit aria-label wins over content, and
    // aria-hidden keeps the glyph visual-only.
    stubFetchRouting({
      chartResp: Promise.resolve({
        ok: true,
        json: () => Promise.resolve(makeChartData({
          channels: [
            makeChartChannel({ channel: 'SR:MAG:QF1:I', numeric: true }),
            makeChartChannel({ channel: 'SR:RF:INTERLOCK_STATE', values: [0, 1], numeric: false }),
          ],
        })),
      }),
    });

    await renderTimeseriesView(container, { id: 'ts1' });

    const toggles = Array.from(container.querySelectorAll('.ts-ch-toggle'));
    const enumToggle = /** @type {HTMLElement} */ (
      toggles.find((b) => /** @type {HTMLElement} */ (b).dataset.chName === 'SR:RF:INTERLOCK_STATE')
    );
    const tag = qs(enumToggle, '.ts-ch-axis-tag');
    expect(tag.getAttribute('aria-hidden')).toBe('true');
    // Full PV plus the explanation -- not the display text with a stray "R".
    expect(enumToggle.getAttribute('aria-label')).toBe('SR:RF:INTERLOCK_STATE (status/enum -- plotted on right axis)');
    // A numeric channel has no marker, so its name is just the full PV.
    expect(toggles[0].getAttribute('aria-label')).toBe('SR:MAG:QF1:I');
    // type="button" so these never act as submit buttons inside a form.
    expect(toggles.every((b) => b.getAttribute('type') === 'button')).toBe(true);
  });

  describe('channel toggling', () => {
    test('a toggle reports its shown/hidden state via aria-pressed, not by CSS class alone', async () => {
      // The `ts-ch-off` class alone is invisible to assistive tech.
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.ts-ch-toggle'));
      expect(Array.from(toggles).map((b) => b.getAttribute('aria-pressed'))).toEqual(['true', 'true']);

      toggles[0].click(); // hide

      expect(toggles[0].getAttribute('aria-pressed')).toBe('false');
      expect(toggles[1].getAttribute('aria-pressed')).toBe('true');

      toggles[0].click(); // show again

      expect(toggles[0].getAttribute('aria-pressed')).toBe('true');
    });

    test('a refused hide (last visible channel) leaves aria-pressed alone', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.ts-ch-toggle'));
      toggles[0].click(); // hide channel 1, leaving channel 2 the only visible one

      toggles[1].click(); // refused

      // The state attribute must not drift from the state the class shows.
      expect(toggles[1].classList.contains('ts-ch-off')).toBe(false);
      expect(toggles[1].getAttribute('aria-pressed')).toBe('true');
    });

    test('clicking a channel button hides it and calls Plotly.update with the updated visibility mask', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.ts-ch-toggle'));
      const chartEl = container.querySelector('[data-ts-chart]');

      toggles[0].click();

      expect(toggles[0].classList.contains('ts-ch-off')).toBe(true);
      expect(Plotly.update).toHaveBeenCalledWith(chartEl, { visible: [false, true] }, {});
    });

    test('refuses to hide the last visible channel', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.ts-ch-toggle'));
      toggles[0].click(); // hide channel 1, leaving channel 2 visible
      Plotly.update.mockClear();

      toggles[1].click(); // attempt to hide the only remaining visible channel

      expect(toggles[1].classList.contains('ts-ch-off')).toBe(false);
      expect(Plotly.update).not.toHaveBeenCalled();
    });

    test('re-clicking a hidden channel shows it again', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.ts-ch-toggle'));
      toggles[0].click(); // hide
      toggles[0].click(); // show again

      expect(toggles[0].classList.contains('ts-ch-off')).toBe(false);
      expect(Plotly.update).toHaveBeenLastCalledWith(expect.anything(), { visible: [true, true] }, {});
    });

    test('toggles are keyed by channel name (data-ch-name), not by a column index', async () => {
      stubFetchRouting({
        chartResp: Promise.resolve({
          ok: true,
          json: () => Promise.resolve(makeChartData({
            channels: [
              makeChartChannel({ channel: 'SR:MAG:QF1:I' }),
              makeChartChannel({ channel: 'SR:MAG:QF2:I' }),
              makeChartChannel({ channel: 'SR:BPM:X:1' }),
            ],
          })),
        }),
      });
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = Array.from(container.querySelectorAll('.ts-ch-toggle'));
      expect(toggles.map((b) => /** @type {HTMLElement} */ (b).dataset.chName)).toEqual([
        'SR:MAG:QF1:I', 'SR:MAG:QF2:I', 'SR:BPM:X:1',
      ]);
      // The old `data-ch-index` key must not be relied on any more.
      expect(/** @type {HTMLElement} */ (toggles[0]).dataset.chIndex).toBeUndefined();

      const chartEl = container.querySelector('[data-ts-chart]');
      /** @type {HTMLElement} */ (toggles[1]).click(); // hide the middle channel, by name

      expect(Plotly.update).toHaveBeenCalledWith(chartEl, { visible: [true, false, true] }, {});
    });

    test('hiding the only visible non-numeric channel hides its secondary axis; showing it again restores it', async () => {
      stubFetchRouting({
        chartResp: Promise.resolve({
          ok: true,
          json: () => Promise.resolve(makeChartData({
            channels: [
              makeChartChannel({ channel: 'SR:MAG:QF1:I', numeric: true }),
              makeChartChannel({ channel: 'SR:RF:STATE', values: ['STANDBY', 'CW'], numeric: false }),
            ],
          })),
        }),
      });
      await renderTimeseriesView(container, { id: 'ts1' });

      const chartEl = container.querySelector('[data-ts-chart]');
      const toggles = Array.from(container.querySelectorAll('.ts-ch-toggle'));
      const enumToggle = /** @type {HTMLElement} */ (
        toggles.find((b) => /** @type {HTMLElement} */ (b).dataset.chName === 'SR:RF:STATE')
      );

      enumToggle.click(); // hide the only non-numeric channel

      // Otherwise an empty category yaxis2 is left sitting on the right with
      // nothing plotted against it. One call: trace and layout deltas are
      // applied together so a toggle costs a single redraw.
      expect(Plotly.update).toHaveBeenLastCalledWith(
        chartEl, { visible: [true, false] }, { 'yaxis2.visible': false }
      );

      enumToggle.click(); // show it again

      expect(Plotly.update).toHaveBeenLastCalledWith(
        chartEl, { visible: [true, true] }, { 'yaxis2.visible': true }
      );
    });

    test('does not touch yaxis2 visibility when no channel is non-numeric', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.ts-ch-toggle'));
      toggles[0].click();

      expect(Plotly.relayout).not.toHaveBeenCalled();
      // The layout half of the combined update stays empty.
      expect(Plotly.update).toHaveBeenLastCalledWith(expect.anything(), expect.anything(), {});
    });
  });

  describe('toolbar actions', () => {
    /**
     * Capture the `download` filename each exporter puts on its anchor.
     * @returns {string[]}
     */
    function spyOnDownloadNames() {
      /** @type {string[]} */
      const names = [];
      vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(
        /** @this {HTMLAnchorElement} */
        function () { names.push(this.download); }
      );
      return names;
    }

    test('zoom-reset resets both axes to autorange', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      qs(container, '[data-action="zoom-reset"]').click();

      expect(Plotly.relayout).toHaveBeenCalledWith(
        container.querySelector('[data-ts-chart]'),
        { 'xaxis.autorange': true, 'yaxis.autorange': true }
      );
    });

    test('zoom-reset also autoranges yaxis2 when a non-numeric channel is present', async () => {
      stubFetchRouting({
        chartResp: Promise.resolve({
          ok: true,
          json: () => Promise.resolve(makeChartData({
            channels: [
              makeChartChannel({ channel: 'SR:MAG:QF1:I', numeric: true }),
              makeChartChannel({ channel: 'SR:RF:STATE', values: ['STANDBY', 'CW'], numeric: false }),
            ],
          })),
        }),
      });
      await renderTimeseriesView(container, { id: 'ts1' });

      qs(container, '[data-action="zoom-reset"]').click();

      // Otherwise Reset Zoom leaves the enum channel's secondary axis clipped,
      // reading as "that channel has no data".
      expect(Plotly.relayout).toHaveBeenCalledWith(
        container.querySelector('[data-ts-chart]'),
        { 'xaxis.autorange': true, 'yaxis.autorange': true, 'yaxis2.autorange': true }
      );
    });

    test('export-csv builds a long-format CSV blob (channel,timestamp,value per sample) and triggers a download', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-csv');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      vi.spyOn(window, 'open').mockImplementation(() => null);

      qs(container, '[data-action="export-csv"]').click();

      expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
      const blob = /** @type {Blob} */ (createObjectURL.mock.calls[0][0]);
      expect(blob.type).toBe('text/csv');
      const lines = (await blob.text()).split('\n');
      // Header + 2 channels x 2 samples each = 5 lines. Long format: channels
      // have independent timestamps, so there is no shared axis to pivot wide.
      expect(lines[0]).toBe('channel,timestamp,value');
      expect(lines).toHaveLength(5);
      expect(lines[1]).toBe('SR:MAG:QF1:I,2026-07-01T00:00:00Z,1');
      expect(lines[2]).toBe('SR:MAG:QF1:I,2026-07-01T00:01:00Z,1.5');
      expect(lines[3]).toBe('SR:MAG:QF2:I,2026-07-01T00:00:00Z,2');
      expect(lines[4]).toBe('SR:MAG:QF2:I,2026-07-01T00:01:00Z,2.5');
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-csv');
    });

    test('export-csv quotes/escapes a value containing a comma or a double quote', async () => {
      // Channel names and enum values are CSV fields -- an unquoted comma or
      // embedded quote would silently corrupt the row.
      stubFetchRouting({
        chartResp: Promise.resolve({
          ok: true,
          json: () => Promise.resolve(makeChartData({
            channels: [
              makeChartChannel({
                channel: 'SR:RF:STATE',
                timestamps: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z'],
                values: ['OFF, LOCAL', 'STANDBY "armed"'],
                numeric: false,
              }),
            ],
          })),
        }),
      });
      await renderTimeseriesView(container, { id: 'ts1' });

      const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-csv');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

      qs(container, '[data-action="export-csv"]').click();

      const blob = /** @type {Blob} */ (createObjectURL.mock.calls[0][0]);
      const lines = (await blob.text()).split('\n');
      expect(lines[1]).toBe('SR:RF:STATE,2026-07-01T00:00:00Z,"OFF, LOCAL"');
      expect(lines[2]).toBe('SR:RF:STATE,2026-07-01T00:01:00Z,"STANDBY ""armed"""');
    });

    test('a null value exports as an empty CSV field, not the string "null"', async () => {
      // Gaps arrive as real nulls; a bare `String(v)` would write the literal
      // text "null", indistinguishable from a genuine status string.
      stubFetchRouting({
        chartResp: Promise.resolve({
          ok: true,
          json: () => Promise.resolve(makeChartData({
            channels: [
              makeChartChannel({
                channel: 'SR:RF:STATE',
                timestamps: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z'],
                values: ['ON', null],
                numeric: false,
              }),
            ],
          })),
        }),
      });
      await renderTimeseriesView(container, { id: 'ts1' });

      const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-csv');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

      qs(container, '[data-action="export-csv"]').click();

      const blob = /** @type {Blob} */ (createObjectURL.mock.calls[0][0]);
      const lines = (await blob.text()).split('\n');
      expect(lines[1]).toBe('SR:RF:STATE,2026-07-01T00:00:00Z,ON');
      expect(lines[2]).toBe('SR:RF:STATE,2026-07-01T00:01:00Z,');
    });

    test('a channel with no samples contributes no CSV rows, and the header is emitted exactly once', async () => {
      // An empty channel must contribute nothing, and the header must not be
      // re-seeded per channel.
      stubFetchRouting({
        chartResp: Promise.resolve({
          ok: true,
          json: () => Promise.resolve(makeChartData({
            channels: [
              makeChartChannel({ channel: 'EMPTY', timestamps: [], values: [], total_points: 0, returned_points: 0 }),
              makeChartChannel({ channel: 'SR:MAG:QF1:I', values: [1.0, 1.5] }),
            ],
          })),
        }),
      });
      await renderTimeseriesView(container, { id: 'ts1' });

      const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-csv');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

      qs(container, '[data-action="export-csv"]').click();

      const blob = /** @type {Blob} */ (createObjectURL.mock.calls[0][0]);
      const lines = (await blob.text()).split('\n');
      expect(lines).toEqual([
        'channel,timestamp,value',
        'SR:MAG:QF1:I,2026-07-01T00:00:00Z,1',
        'SR:MAG:QF1:I,2026-07-01T00:01:00Z,1.5',
      ]);
    });

    test('each exporter names its download with its own extension', async () => {
      // The filename extension is the one observable that distinguishes the
      // two exporters.
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      const names = spyOnDownloadNames();

      qs(container, '[data-action="export-csv"]').click();
      qs(container, '[data-action="export-json"]').click();

      expect(names).toHaveLength(2);
      expect(names[0]).toMatch(/^timeseries_\d+\.csv$/);
      expect(names[1]).toMatch(/^timeseries_\d+\.json$/);
    });

    test('export-json builds a JSON blob of the full chart payload', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-json');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      vi.spyOn(window, 'open').mockImplementation(() => null);

      qs(container, '[data-action="export-json"]').click();

      const blob = /** @type {Blob} */ (createObjectURL.mock.calls[0][0]);
      expect(blob.type).toBe('application/json');
      const parsed = JSON.parse(await blob.text());
      expect(parsed.channels.map((/** @type {any} */ ch) => ch.channel)).toEqual(['SR:MAG:QF1:I', 'SR:MAG:QF2:I']);
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-json');
    });
  });
});

describe('renderTimeseriesChart', () => {
  test('hands Plotly.newPlot one trace per channel, each with its own x, and a themed layout', async () => {
    const el = document.createElement('div');
    const chartData = makeChartData();

    await renderTimeseriesChart(el, chartData);

    expect(Plotly.newPlot).toHaveBeenCalledTimes(1);
    const [plotEl, traces, layout, config] = Plotly.newPlot.mock.calls[0];
    expect(plotEl).toBe(el);
    expect(traces).toEqual([
      { x: chartData.channels[0].timestamps, y: [1.0, 1.5], name: 'SR:MAG:QF1:I', type: 'scattergl', mode: 'lines', hovertemplate: '%{y:.4g}<extra>%{fullData.name}</extra>' },
      { x: chartData.channels[1].timestamps, y: [2.0, 2.5], name: 'SR:MAG:QF2:I', type: 'scattergl', mode: 'lines', hovertemplate: '%{y:.4g}<extra>%{fullData.name}</extra>' },
    ]);
    expect(layout.hovermode).toBe('x unified');
    expect(layout.margin).toEqual({ t: 30, r: 20, b: 50, l: 60 });
    expect(config).toEqual({
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });
  });

  test('two channels of different cadence each keep their own independent x -- not a shared axis', async () => {
    const el = document.createElement('div');
    const chartData = makeChartData({
      channels: [
        // Slow-cadence channel: 3 samples.
        makeChartChannel({
          channel: 'SR:MAG:QF1:I',
          timestamps: ['2026-07-01T00:00:00Z', '2026-07-01T00:10:00Z', '2026-07-01T00:20:00Z'],
          values: [1.0, 1.1, 1.2],
          total_points: 3,
          returned_points: 3,
        }),
        // Fast-cadence channel: 5 samples on a different timestamp grid.
        makeChartChannel({
          channel: 'SR:BPM:X:1',
          timestamps: [
            '2026-07-01T00:00:00Z', '2026-07-01T00:02:00Z', '2026-07-01T00:04:00Z',
            '2026-07-01T00:06:00Z', '2026-07-01T00:08:00Z',
          ],
          values: [0.1, 0.2, 0.3, 0.4, 0.5],
          total_points: 5,
          returned_points: 5,
        }),
      ],
    });

    await renderTimeseriesChart(el, chartData);

    const [, traces] = Plotly.newPlot.mock.calls[0];
    expect(traces[0].x).toEqual(chartData.channels[0].timestamps);
    expect(traces[0].x).toHaveLength(3);
    expect(traces[1].x).toEqual(chartData.channels[1].timestamps);
    expect(traces[1].x).toHaveLength(5);
    // Genuinely independent arrays, not the same shared axis sliced twice.
    expect(traces[0].x).not.toEqual(traces[1].x);
  });

  test('a non-numeric (enum/status) channel is not dropped -- it gets its own categorical trace on a secondary y-axis', async () => {
    const el = document.createElement('div');
    const chartData = makeChartData({
      channels: [
        makeChartChannel({ channel: 'SR:MAG:QF1:I', numeric: true }),
        makeChartChannel({
          channel: 'SR:RF:STATE',
          timestamps: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z'],
          values: ['STANDBY', 'CW'],
          numeric: false,
        }),
      ],
    });

    await renderTimeseriesChart(el, chartData);

    const [, traces, layout] = Plotly.newPlot.mock.calls[0];
    expect(traces).toHaveLength(2);

    const numericTrace = traces.find((/** @type {any} */ t) => t.name === 'SR:MAG:QF1:I');
    expect(numericTrace.yaxis).toBeUndefined(); // default primary axis
    expect(numericTrace.y).toEqual([1.0, 1.5]);

    const statusTrace = traces.find((/** @type {any} */ t) => t.name === 'SR:RF:STATE');
    // The category is namespaced by channel so two enum channels can't
    // interleave on the shared axis; real values ride along in customdata.
    expect(statusTrace.y).toEqual(['SR:RF:STATE: STANDBY', 'SR:RF:STATE: CW']);
    expect(statusTrace.customdata).toEqual(['STANDBY', 'CW']);
    expect(statusTrace.yaxis).toBe('y2');
    expect(statusTrace.line).toEqual({ shape: 'hv' }); // step trace: holds until the next transition
    // Hover shows the real state, not the namespaced label, and no :.4g on a string.
    expect(statusTrace.hovertemplate).toBe('%{customdata}<extra>%{fullData.name}</extra>');

    expect(layout.yaxis2).toMatchObject({ overlaying: 'y', side: 'right', type: 'category' });
  });

  test('two enum channels get disjoint rungs on the shared category axis', async () => {
    // Plotly unions a category axis's rungs across traces; bare values would
    // interleave both vocabularies on one ladder. Namespacing keeps each
    // channel's rungs contiguous.
    const el = document.createElement('div');
    const chartData = makeChartData({
      channels: [
        makeChartChannel({
          channel: 'SR:RF:STATE',
          timestamps: ['2026-07-01T00:00:00Z'],
          values: ['CW'],
          numeric: false,
        }),
        makeChartChannel({
          channel: 'SR:MODE',
          timestamps: ['2026-07-01T00:00:00Z'],
          values: ['CW'],
          numeric: false,
        }),
      ],
    });

    await renderTimeseriesChart(el, chartData);

    const [, traces] = Plotly.newPlot.mock.calls[0];
    const [rf, mode] = traces;

    // Same underlying value 'CW' on both channels, but distinct rungs.
    expect(rf.customdata).toEqual(['CW']);
    expect(mode.customdata).toEqual(['CW']);
    expect(rf.y).not.toEqual(mode.y);
    expect(new Set([...rf.y, ...mode.y]).size).toBe(2);
  });

  test('a null gap inside an enum channel stays a gap -- it never becomes a fabricated "CH: null" rung', async () => {
    // A status channel can carry `null` in `values`. Namespacing it as
    // 'SR:RF:STATE: null' would register a real rung -- a state the channel
    // was never in; a real null instead hits Plotly's `!= null` category
    // guard and breaks the line.
    const el = document.createElement('div');
    const chartData = makeChartData({
      channels: [
        makeChartChannel({
          channel: 'SR:RF:STATE',
          timestamps: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z', '2026-07-01T00:02:00Z'],
          values: ['ON', null, 'OFF'],
          numeric: false,
        }),
      ],
    });

    await renderTimeseriesChart(el, chartData);

    const [, traces] = Plotly.newPlot.mock.calls[0];
    expect(traces[0].y).toEqual(['SR:RF:STATE: ON', null, 'SR:RF:STATE: OFF']);
    // The namespacing still applies to every real value -- only the gap is exempt.
    expect(traces[0].customdata).toEqual(['ON', null, 'OFF']);
  });

  test('non-numeric routing is driven by the `numeric` flag, not by sniffing whether the values look like numbers', async () => {
    // Numeric-looking enum codes with `numeric: false`: sniffing the values
    // would route them onto the linear axis, defeating the flag.
    const el = document.createElement('div');
    const chartData = makeChartData({
      channels: [
        makeChartChannel({
          channel: 'SR:RF:INTERLOCK_STATE',
          timestamps: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z', '2026-07-01T00:02:00Z'],
          values: [0, 1, 0],
          numeric: false,
        }),
      ],
    });

    await renderTimeseriesChart(el, chartData);

    const [, traces, layout] = Plotly.newPlot.mock.calls[0];
    const trace = traces[0];
    // Codes stay numeric in customdata and become category labels on the axis.
    expect(trace.customdata).toEqual([0, 1, 0]);
    expect(trace.y).toEqual([
      'SR:RF:INTERLOCK_STATE: 0',
      'SR:RF:INTERLOCK_STATE: 1',
      'SR:RF:INTERLOCK_STATE: 0',
    ]);
    expect(trace.yaxis).toBe('y2');
    expect(trace.type).toBe('scatter'); // not scattergl -- the non-numeric branch
    expect(trace.line).toEqual({ shape: 'hv' });
    expect(trace.hovertemplate).toBe('%{customdata}<extra>%{fullData.name}</extra>'); // no :.4g
    expect(layout.yaxis2).toBeDefined();
  });

  test('the secondary category axis automargins, so its long namespaced rung labels are not clipped', async () => {
    // yaxis2's rung labels ("<full PV name>: <VALUE>") are the longest strings
    // in the figure and sit on the right against `margin.r: 20`; in the
    // vendored plotly-3.3.1 `automargin` resolves false for this axis, so
    // without it the labels are drawn past the paper edge and clipped.
    const el = document.createElement('div');

    await renderTimeseriesChart(el, makeChartData({
      channels: [makeChartChannel({ channel: 'SR:C03:BPM:STATUS', values: ['OPERATIONAL', 'FAULT'], numeric: false })],
    }));

    const [, , layout] = Plotly.newPlot.mock.calls[0];
    expect(layout.yaxis2.automargin).toBe(true);
    // The numeric axes keep the unchanged margins.
    expect(layout.margin).toEqual({ t: 30, r: 20, b: 50, l: 60 });
  });

  test('the secondary category axis draws no gridlines of its own', async () => {
    // In the vendored bundle an overlaying axis's `showgrid` defaults true, so
    // yaxis2 would draw one un-themed gridline per rung, ignoring the
    // --chart-* tokens in both themes.
    const el = document.createElement('div');

    await renderTimeseriesChart(el, makeChartData({
      channels: [makeChartChannel({ channel: 'SR:RF:STATE', values: ['STANDBY', 'CW'], numeric: false })],
    }));

    const [, , layout] = Plotly.newPlot.mock.calls[0];
    expect(layout.yaxis2.showgrid).toBe(false);
    // The themed numeric grid is untouched.
    expect(layout.yaxis.gridcolor).toBe(_tsChartTheme().yaxis.gridcolor);
  });

  test('does not add a secondary y-axis when every channel is numeric', async () => {
    const el = document.createElement('div');

    await renderTimeseriesChart(el, makeChartData());

    const [, , layout] = Plotly.newPlot.mock.calls[0];
    expect(layout.yaxis2).toBeUndefined();
  });

  test('is a no-op when the target element is falsy (but still awaits the Plotly load)', async () => {
    await expect(renderTimeseriesChart(null, makeChartData())).resolves.toBeUndefined();
    expect(Plotly.newPlot).not.toHaveBeenCalled();
  });
});

describe('renderTimeseriesTable', () => {
  /** @type {HTMLElement} */
  let el;

  beforeEach(() => {
    el = document.createElement('div');
  });

  test('header comes from the table response, not from the caller-supplied columns', async () => {
    // A separate `format=chart` request can disagree with this one; the table
    // response carries the very column list its rows were built from.
    stubTableFetch({ columns: ['FRESH:A', 'FRESH:B'] });

    await renderTimeseriesTable(el, 'ts1', 0);

    const headers = Array.from(el.querySelectorAll('thead th')).map((th) => th.textContent);
    expect(headers).toEqual(['Index', 'FRESH:A', 'FRESH:B']);
  });

  test('renders a header row and one data row per index entry, from fixture series', async () => {
    stubTableFetch();

    await renderTimeseriesTable(el, 'ts1', 0);

    const headers = Array.from(el.querySelectorAll('thead th')).map((th) => th.textContent);
    expect(headers).toEqual(['Index', 'SR:MAG:QF1:I', 'SR:MAG:QF2:I']);

    const rows = el.querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);
    // Index cell goes through _tsShortTime: exact rendering is locale-dependent,
    // so assert shape (no year, seconds retained) rather than an exact string.
    const indexCellText = qs(rows[0], '.ts-index-cell').textContent;
    expect(indexCellText).not.toMatch(/\b(19|20)\d{2}\b/);
    expect(indexCellText).toMatch(/\d{1,2}:\d{2}:\d{2}/);
    // Value cells go through _tsFormatValue: <=5 significant figures.
    const valueCells = Array.from(rows[0].querySelectorAll('td')).slice(1).map((td) => td.textContent);
    expect(valueCells).toEqual(['1.0000', '2.0000']);
  });

  test('renders null values as "--" rather than the string "null"', async () => {
    stubTableFetch({ data: [[null, 2.0]], index: ['2026-07-01T00:00:00Z'] });

    await renderTimeseriesTable(el, 'ts1', 0);

    const cells = Array.from(el.querySelectorAll('tbody tr td')).map((td) => td.textContent);
    expect(cells[1]).toBe('--');
    expect(cells[2]).toBe('2.0000');
  });

  describe('value-cell formatting (_tsFormatValue, exercised via rendered cells)', () => {
    test('null/undefined -> "--", 0 -> "0", ordinary magnitude -> <=5 significant figures', async () => {
      stubTableFetch({
        columns: ['a', 'b', 'c'],
        index: ['2026-07-01T00:00:00Z'],
        data: [[null, 0, 1.23456789]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      const cells = Array.from(el.querySelectorAll('tbody tr td')).map((td) => td.textContent).slice(1);
      expect(cells).toEqual(['--', '0', '1.2346']);
    });

    test('very large and very small nonzero magnitudes render in scientific notation (toExponential(3))', async () => {
      stubTableFetch({
        columns: ['big', 'tiny'],
        index: ['2026-07-01T00:00:00Z'],
        data: [[1e7, 0.000123]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      const cells = Array.from(el.querySelectorAll('tbody tr td')).map((td) => td.textContent).slice(1);
      expect(cells).toEqual([(1e7).toExponential(3), (0.000123).toExponential(3)]);
    });

    test('a non-number string cell and a NaN cell fall back to String(...)', async () => {
      stubTableFetch({
        columns: ['s', 'n'],
        index: ['2026-07-01T00:00:00Z'],
        data: [['not-a-number', NaN]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      const cells = Array.from(el.querySelectorAll('tbody tr td')).map((td) => td.textContent).slice(1);
      expect(cells).toEqual(['not-a-number', 'NaN']);
    });
  });

  describe('index-cell short-time formatting (_tsShortTime, exercised via rendered cells)', () => {
    test('a valid ISO timestamp renders without a 4-digit year and with seconds retained', async () => {
      stubTableFetch({
        columns: ['a'],
        index: ['2026-07-06T12:34:56Z'],
        data: [[1.0]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      const indexCellText = qs(el, '.ts-index-cell').textContent;
      // Locale-dependent exact rendering -- assert shape, not an exact string.
      // (No month-glyph assertion: month:"short" has no ASCII letters under
      // CJK/numeric-month locales, so that check would flake by CI locale.)
      expect(indexCellText).not.toMatch(/\b(19|20)\d{2}\b/); // no year
      expect(indexCellText).toMatch(/\d{1,2}:\d{2}:\d{2}/); // hour:minute:second
    });

    test('null and numeric index values render honestly instead of as fabricated dates', async () => {
      stubTableFetch({
        columns: ['a'],
        index: [null, 0, '0'],
        data: [[1.0], [2.0], [3.0]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      const cells = Array.from(el.querySelectorAll('.ts-index-cell')).map((c) => c.textContent);
      // new Date(null) is epoch 0 and new Date('0') parses as year 2000 --
      // neither may leak into the table as an invented timestamp.
      expect(cells).toEqual(['--', '0', '0']);
    });

    test('an unparseable index value falls back to String(iso)', async () => {
      stubTableFetch({
        columns: ['a'],
        index: ['not-a-date'],
        data: [[1.0]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      expect(qs(el, '.ts-index-cell').textContent).toBe('not-a-date');
    });
  });

  describe('hostile cell values (MI-1 regression: agent-supplied strings must render inert)', () => {
    const XSS_PAYLOAD = '"><img src=x onerror=alert(1)>';

    test('a hostile string value cell renders escaped, with no live <img>', async () => {
      stubTableFetch({
        columns: ['a'],
        index: ['2026-07-01T00:00:00Z'],
        data: [[XSS_PAYLOAD]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      expect(el.querySelectorAll('img').length).toBe(0);
      expect(el.innerHTML).toContain('&lt;img');
      expect(el.innerHTML).not.toContain('<img');
      const valueCell = el.querySelectorAll('tbody tr td')[1];
      expect(valueCell.textContent).toBe(XSS_PAYLOAD);
    });

    test('a hostile string index value renders escaped, with no live <img>', async () => {
      stubTableFetch({
        columns: ['a'],
        index: [XSS_PAYLOAD],
        data: [[1.0]],
      });

      await renderTimeseriesTable(el, 'ts1', 0);

      expect(el.querySelectorAll('img').length).toBe(0);
      expect(el.innerHTML).toContain('&lt;img');
      expect(el.innerHTML).not.toContain('<img');
      expect(qs(el, '.ts-index-cell').textContent).toBe(XSS_PAYLOAD);
    });
  });

  describe('pagination', () => {
    test('Prev is disabled at offset 0; Next is disabled once all rows are on the page', async () => {
      stubTableFetch({ total_rows: 2 });

      await renderTimeseriesTable(el, 'ts1', 0);

      expect(qs(el, '[data-ts-prev]', HTMLButtonElement).disabled).toBe(true);
      expect(qs(el, '[data-ts-next]', HTMLButtonElement).disabled).toBe(true); // total_rows (2) <= offset(0) + limit(50)
      expect(qs(el, '.ts-page-info').textContent).toBe('Page 1 of 1');
    });

    test('Next is enabled when more rows remain, and clicking it re-fetches at the next offset', async () => {
      const fetchMock = stubTableFetch({ total_rows: 120 });

      await renderTimeseriesTable(el, 'ts1', 0);

      const nextBtn = qs(el, '[data-ts-next]', HTMLButtonElement);
      expect(nextBtn.disabled).toBe(false);
      expect(qs(el, '[data-ts-prev]', HTMLButtonElement).disabled).toBe(true);

      nextBtn.click();
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('offset=50')));
    });

    test('Prev clamps to offset 0 rather than going negative', async () => {
      const fetchMock = stubTableFetch({ total_rows: 120, offset: 20 });

      await renderTimeseriesTable(el, 'ts1', 20);
      fetchMock.mockClear();

      qs(el, '[data-ts-prev]', HTMLButtonElement).click();
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('offset=0')));
    });
  });

  test('on a fetch failure: shows the failure fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }));

    await renderTimeseriesTable(el, 'ts1', 0);

    expect(el.textContent).toContain('Failed to load table data');
  });

  test('is a no-op when the target element is falsy', async () => {
    await expect(renderTimeseriesTable(null, 'ts1', 0)).resolves.toBeUndefined();
  });
});

describe('Plotly loader edge cases (isolated: fresh module instance per test)', () => {
  // `_plotlyLoaded`/`_plotlyLoading` are a module singleton (see the file
  // header) — by the time any test above this point has run, `_plotlyLoaded`
  // is permanently `true` for the rest of the file, so the loader's onerror
  // path and its concurrent-call coalescing are both untestable against the
  // shared `timeseries` import above. Each test here instead calls
  // vi.resetModules() and re-imports the module fresh via a dynamic import,
  // getting its own private `_plotlyLoaded`/`_plotlyLoading` closure state
  // that starts unloaded, independent of every other test in this file.

  beforeEach(() => {
    vi.resetModules();
  });

  test('a script load failure (onerror) rejects ensurePlotlyLoaded, surfacing as the rendered failure fallback', async () => {
    vi.stubGlobal('Plotly', { newPlot: vi.fn() });
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      const script = /** @type {InjectedScript} */ (node);
      if (node && script.tagName === 'SCRIPT') {
        queueMicrotask(() => script.onerror && script.onerror());
        return node;
      }
      throw new Error('unexpected non-script appendChild in this isolated test');
    });
    stubFetchRouting();

    const fresh = await import('../../../src/osprey/interfaces/artifacts/static/js/timeseries.js');
    const container = document.createElement('div');

    await fresh.renderTimeseriesView(container, { id: 'ts1' });

    expect(container.textContent).toContain('Failed to load timeseries data');
    expect(Plotly.newPlot).not.toHaveBeenCalled();
  });

  test('a script load failure does not permanently poison the loader: the next render re-injects a fresh <script> and can succeed', async () => {
    vi.stubGlobal('Plotly', { newPlot: vi.fn((el) => { el.data = []; }) });
    stubFetchRouting();

    /** @type {InjectedScript[]} */
    const scriptAppends = [];
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      const script = /** @type {InjectedScript} */ (node);
      if (node && script.tagName === 'SCRIPT') {
        scriptAppends.push(script);
        if (scriptAppends.length === 1) {
          // First injection: simulate a load failure, as the earlier test does.
          queueMicrotask(() => script.onerror && script.onerror());
        }
        // The second (retry) injection is left pending here -- the test
        // fires its onload manually below, once it's confirmed injected.
        return node;
      }
      throw new Error('unexpected non-script appendChild in this isolated test');
    });

    const fresh = await import('../../../src/osprey/interfaces/artifacts/static/js/timeseries.js');

    const container1 = document.createElement('div');
    await fresh.renderTimeseriesView(container1, { id: 'ts1' });

    expect(container1.textContent).toContain('Failed to load timeseries data');
    expect(scriptAppends.length).toBe(1);

    const container2 = document.createElement('div');
    const renderPromise = fresh.renderTimeseriesView(container2, { id: 'ts2' });

    // A second render must re-inject its own fresh <script> rather than
    // reusing the (permanently rejected) first loading promise.
    await vi.waitFor(() => expect(scriptAppends.length).toBe(2));
    expect(scriptAppends[1]).not.toBe(scriptAppends[0]);

    /** @type {() => void} */ (scriptAppends[1].onload)();
    await renderPromise;

    expect(container2.textContent).not.toContain('Failed to load timeseries data');
    expect(Plotly.newPlot).toHaveBeenCalledTimes(1);
  });

  test('concurrent renderTimeseriesChart calls coalesce onto one _plotlyLoading promise (exactly one <script> injected)', async () => {
    vi.stubGlobal('Plotly', { newPlot: vi.fn((el) => { el.data = []; }) });
    /** @type {InjectedScript[]} */
    const scriptAppends = [];
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      const script = /** @type {InjectedScript} */ (node);
      if (node && script.tagName === 'SCRIPT') {
        scriptAppends.push(script);
        return node; // deliberately NOT firing onload yet, to inspect the in-flight state below
      }
      throw new Error('unexpected non-script appendChild in this isolated test');
    });

    const fresh = await import('../../../src/osprey/interfaces/artifacts/static/js/timeseries.js');
    const elA = /** @type {HTMLDivElement & { data?: unknown[] }} */ (document.createElement('div'));
    const elB = /** @type {HTMLDivElement & { data?: unknown[] }} */ (document.createElement('div'));
    const chartData = makeChartData();

    // Two callers racing the same not-yet-loaded Plotly: both start before
    // either resolves.
    const pA = fresh.renderTimeseriesChart(elA, chartData);
    const pB = fresh.renderTimeseriesChart(elB, chartData);

    expect(scriptAppends.length).toBe(1); // only the first call injected a <script>; the second reused its pending promise

    /** @type {() => void} */ (scriptAppends[0].onload)();
    await Promise.all([pA, pB]);

    expect(Plotly.newPlot).toHaveBeenCalledTimes(2);
    expect(elA.data).toEqual([]);
    expect(elB.data).toEqual([]);
  });

  // The injected src is resolved server-side by the vendor_url() template
  // global (CDN by default, the vendored copy offline) and handed to this
  // classic-script loader via index.html's osprey-vendor-plotly meta tag —
  // static JS cannot call a Jinja global. Both halves of that contract:
  // the meta wins when present, the vendored spelling survives without it.

  test('the injected <script> src comes from the osprey-vendor-plotly meta when the page carries one', async () => {
    vi.stubGlobal('Plotly', { newPlot: vi.fn((el) => { el.data = []; }) });
    stubFetchRouting();

    const meta = document.createElement('meta');
    meta.setAttribute('name', 'osprey-vendor-plotly');
    meta.setAttribute('content', 'https://cdn.plot.ly/plotly-3.3.1.min.js');
    document.head.appendChild(meta); // before the spy below hijacks appendChild

    /** @type {InjectedScript[]} */
    const scriptAppends = [];
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      const script = /** @type {InjectedScript} */ (node);
      if (node && script.tagName === 'SCRIPT') {
        scriptAppends.push(script);
        queueMicrotask(() => script.onload && script.onload());
        return node;
      }
      throw new Error('unexpected non-script appendChild in this isolated test');
    });

    try {
      const fresh = await import('../../../src/osprey/interfaces/artifacts/static/js/timeseries.js');
      const container = document.createElement('div');
      await fresh.renderTimeseriesView(container, { id: 'ts1' });

      expect(scriptAppends.length).toBe(1);
      expect(scriptAppends[0].getAttribute('src')).toBe('https://cdn.plot.ly/plotly-3.3.1.min.js');
    } finally {
      meta.remove(); // this file's document is shared across tests
    }
  });

  test('without the meta the loader falls back to the vendored path', async () => {
    vi.stubGlobal('Plotly', { newPlot: vi.fn((el) => { el.data = []; }) });
    stubFetchRouting();

    /** @type {InjectedScript[]} */
    const scriptAppends = [];
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      const script = /** @type {InjectedScript} */ (node);
      if (node && script.tagName === 'SCRIPT') {
        scriptAppends.push(script);
        queueMicrotask(() => script.onload && script.onload());
        return node;
      }
      throw new Error('unexpected non-script appendChild in this isolated test');
    });

    const fresh = await import('../../../src/osprey/interfaces/artifacts/static/js/timeseries.js');
    const container = document.createElement('div');
    await fresh.renderTimeseriesView(container, { id: 'ts1' });

    expect(scriptAppends.length).toBe(1);
    expect(scriptAppends[0].getAttribute('src')).toBe('/static/js/vendor/plotly-3.3.1.min.js');
  });
});
