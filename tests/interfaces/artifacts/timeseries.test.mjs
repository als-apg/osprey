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

import {
  renderTimeseriesView,
  renderTimeseriesChart,
  renderTimeseriesTable,
  _tsChartTheme,
} from '../../../src/osprey/interfaces/artifacts/static/js/timeseries.js';

/** Fixture chart-format response (`/api/artifacts/{id}/data?format=chart`). */
function makeChartData(overrides = {}) {
  return {
    columns: ['SR:MAG:QF1:I', 'SR:MAG:QF2:I'],
    index: ['2026-07-01T00:00:00Z', '2026-07-01T00:01:00Z'],
    data: [[1.0, 2.0], [1.5, 2.5]],
    total_rows: 2,
    downsampled: false,
    returned_points: 2,
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
    if (node && node.tagName === 'SCRIPT') {
      queueMicrotask(() => node.onload && node.onload());
      return node;
    }
    return originalAppendChild(node);
  });
}

/** Sets the chart-theme CSS custom properties (+ the sentinel) inline. */
function setChartVars({ bgPrimary = '#000', paperBg, plotBg, axisText, grid, border }) {
  const root = document.documentElement.style;
  root.setProperty('--bg-primary', bgPrimary);
  root.setProperty('--chart-paper-bg', paperBg);
  root.setProperty('--chart-plot-bg', plotBg);
  root.setProperty('--chart-axis-text', axisText);
  root.setProperty('--chart-grid', grid);
  root.setProperty('--border-default', border);
}

beforeEach(() => {
  vi.stubGlobal('Plotly', {
    newPlot: vi.fn((el) => { el.data = []; }),
    restyle: vi.fn(),
    relayout: vi.fn(),
  });
  stubScriptLoad();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  ['--bg-primary', '--chart-paper-bg', '--chart-plot-bg', '--chart-axis-text', '--chart-grid', '--border-default']
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

  test('shows a loading placeholder, then the info bar, toolbar, chart, and table containers on success', async () => {
    stubFetchRouting();

    const pending = renderTimeseriesView(container, { id: 'ts1' });
    expect(container.querySelector('.ts-loading')).not.toBeNull();

    await pending;

    expect(container.querySelector('.ts-info-bar')).not.toBeNull();
    expect(container.querySelectorAll('.ts-badge-channel').length).toBe(2);
    expect(container.querySelector('.ts-badge-rows').textContent).toContain('2');
    expect(container.querySelector('[data-ts-chart]')).not.toBeNull();
    expect(container.querySelector('[data-ts-table]')).not.toBeNull();
    expect(container.querySelectorAll('.ts-ch-toggle').length).toBe(2);
  });

  test('shows a downsampled badge only when the chart response reports downsampling', async () => {
    stubFetchRouting({
      chartResp: Promise.resolve({
        ok: true,
        json: () => Promise.resolve(makeChartData({ downsampled: true, returned_points: 500, total_rows: 5000 })),
      }),
    });

    await renderTimeseriesView(container, { id: 'ts1' });

    expect(container.querySelector('.ts-badge-downsampled').textContent).toContain('500');
  });

  test('on a chart-fetch failure: shows the failure fallback', async () => {
    stubFetchRouting({ chartResp: Promise.resolve({ ok: false, status: 500 }) });

    await renderTimeseriesView(container, { id: 'ts1' });

    expect(container.textContent).toContain('Failed to load timeseries data');
  });

  describe('channel toggling', () => {
    test('clicking a channel button hides it and calls Plotly.restyle with the updated visibility mask', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = container.querySelectorAll('.ts-ch-toggle');
      const chartEl = container.querySelector('[data-ts-chart]');

      toggles[0].click();

      expect(toggles[0].classList.contains('ts-ch-off')).toBe(true);
      expect(Plotly.restyle).toHaveBeenCalledWith(chartEl, { visible: [false, true] });
    });

    test('refuses to hide the last visible channel', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = container.querySelectorAll('.ts-ch-toggle');
      toggles[0].click(); // hide channel 1, leaving channel 2 visible
      Plotly.restyle.mockClear();

      toggles[1].click(); // attempt to hide the only remaining visible channel

      expect(toggles[1].classList.contains('ts-ch-off')).toBe(false);
      expect(Plotly.restyle).not.toHaveBeenCalled();
    });

    test('re-clicking a hidden channel shows it again', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      const toggles = container.querySelectorAll('.ts-ch-toggle');
      toggles[0].click(); // hide
      toggles[0].click(); // show again

      expect(toggles[0].classList.contains('ts-ch-off')).toBe(false);
      expect(Plotly.restyle).toHaveBeenLastCalledWith(expect.anything(), { visible: [true, true] });
    });
  });

  describe('toolbar actions', () => {
    test('zoom-reset resets both axes to autorange', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      container.querySelector('[data-action="zoom-reset"]').click();

      expect(Plotly.relayout).toHaveBeenCalledWith(
        container.querySelector('[data-ts-chart]'),
        { 'xaxis.autorange': true, 'yaxis.autorange': true }
      );
    });

    test('export-csv builds a CSV blob (header + one row per index entry) and triggers a download', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-csv');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      vi.spyOn(window, 'open').mockImplementation(() => null);

      container.querySelector('[data-action="export-csv"]').click();

      expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
      const blob = URL.createObjectURL.mock.calls[0][0];
      expect(blob.type).toBe('text/csv');
      const text = await blob.text();
      expect(text.split('\n')[0]).toBe('timestamp,SR:MAG:QF1:I,SR:MAG:QF2:I');
      expect(text.split('\n')).toHaveLength(3); // header + 2 data rows
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-csv');
    });

    test('export-json builds a JSON blob of the full chart payload', async () => {
      stubFetchRouting();
      await renderTimeseriesView(container, { id: 'ts1' });

      vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-json');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      vi.spyOn(window, 'open').mockImplementation(() => null);

      container.querySelector('[data-action="export-json"]').click();

      const blob = URL.createObjectURL.mock.calls[0][0];
      expect(blob.type).toBe('application/json');
      const parsed = JSON.parse(await blob.text());
      expect(parsed.columns).toEqual(['SR:MAG:QF1:I', 'SR:MAG:QF2:I']);
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-json');
    });
  });
});

describe('renderTimeseriesChart', () => {
  test('hands Plotly.newPlot traces built from the chart columns/index/data and a themed layout', async () => {
    const el = document.createElement('div');
    const chartData = makeChartData();

    await renderTimeseriesChart(el, chartData);

    expect(Plotly.newPlot).toHaveBeenCalledTimes(1);
    const [plotEl, traces, layout, config] = Plotly.newPlot.mock.calls[0];
    expect(plotEl).toBe(el);
    expect(traces).toEqual([
      { x: chartData.index, y: [1.0, 1.5], name: 'SR:MAG:QF1:I', type: 'scattergl', mode: 'lines', hovertemplate: '%{y:.4g}<extra>%{fullData.name}</extra>' },
      { x: chartData.index, y: [2.0, 2.5], name: 'SR:MAG:QF2:I', type: 'scattergl', mode: 'lines', hovertemplate: '%{y:.4g}<extra>%{fullData.name}</extra>' },
    ]);
    expect(layout.hovermode).toBe('x unified');
    expect(layout.margin).toEqual({ t: 30, r: 20, b: 50, l: 60 });
    expect(config).toEqual({
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });
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

  test('renders a header row and one data row per index entry, from fixture series', async () => {
    stubTableFetch();

    await renderTimeseriesTable(el, 'ts1', ['SR:MAG:QF1:I', 'SR:MAG:QF2:I'], 0);

    const headers = Array.from(el.querySelectorAll('thead th')).map((th) => th.textContent);
    expect(headers).toEqual(['Index', 'SR:MAG:QF1:I', 'SR:MAG:QF2:I']);

    const rows = el.querySelectorAll('tbody tr');
    expect(rows.length).toBe(2);
    // Index cell goes through _tsShortTime: exact rendering is locale-dependent,
    // so assert shape (no year, seconds retained) rather than an exact string.
    const indexCellText = rows[0].querySelector('.ts-index-cell').textContent;
    expect(indexCellText).not.toMatch(/\b(19|20)\d{2}\b/);
    expect(indexCellText).toMatch(/\d{1,2}:\d{2}:\d{2}/);
    // Value cells go through _tsFormatValue: <=5 significant figures.
    const valueCells = Array.from(rows[0].querySelectorAll('td')).slice(1).map((td) => td.textContent);
    expect(valueCells).toEqual(['1.0000', '2.0000']);
  });

  test('renders null values as "--" rather than the string "null"', async () => {
    stubTableFetch({ data: [[null, 2.0]], index: ['2026-07-01T00:00:00Z'] });

    await renderTimeseriesTable(el, 'ts1', ['a', 'b'], 0);

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

      await renderTimeseriesTable(el, 'ts1', ['a', 'b', 'c'], 0);

      const cells = Array.from(el.querySelectorAll('tbody tr td')).map((td) => td.textContent).slice(1);
      expect(cells).toEqual(['--', '0', '1.2346']);
    });

    test('very large and very small nonzero magnitudes render in scientific notation (toExponential(3))', async () => {
      stubTableFetch({
        columns: ['big', 'tiny'],
        index: ['2026-07-01T00:00:00Z'],
        data: [[1e7, 0.000123]],
      });

      await renderTimeseriesTable(el, 'ts1', ['big', 'tiny'], 0);

      const cells = Array.from(el.querySelectorAll('tbody tr td')).map((td) => td.textContent).slice(1);
      expect(cells).toEqual([(1e7).toExponential(3), (0.000123).toExponential(3)]);
    });

    test('a non-number string cell and a NaN cell fall back to String(...)', async () => {
      stubTableFetch({
        columns: ['s', 'n'],
        index: ['2026-07-01T00:00:00Z'],
        data: [['not-a-number', NaN]],
      });

      await renderTimeseriesTable(el, 'ts1', ['s', 'n'], 0);

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

      await renderTimeseriesTable(el, 'ts1', ['a'], 0);

      const indexCellText = el.querySelector('.ts-index-cell').textContent;
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

      await renderTimeseriesTable(el, 'ts1', ['a'], 0);

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

      await renderTimeseriesTable(el, 'ts1', ['a'], 0);

      expect(el.querySelector('.ts-index-cell').textContent).toBe('not-a-date');
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

      await renderTimeseriesTable(el, 'ts1', ['a'], 0);

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

      await renderTimeseriesTable(el, 'ts1', ['a'], 0);

      expect(el.querySelectorAll('img').length).toBe(0);
      expect(el.innerHTML).toContain('&lt;img');
      expect(el.innerHTML).not.toContain('<img');
      expect(el.querySelector('.ts-index-cell').textContent).toBe(XSS_PAYLOAD);
    });
  });

  describe('pagination', () => {
    test('Prev is disabled at offset 0; Next is disabled once all rows are on the page', async () => {
      stubTableFetch({ total_rows: 2 });

      await renderTimeseriesTable(el, 'ts1', ['a', 'b'], 0);

      expect(el.querySelector('[data-ts-prev]').disabled).toBe(true);
      expect(el.querySelector('[data-ts-next]').disabled).toBe(true); // total_rows (2) <= offset(0) + limit(50)
      expect(el.querySelector('.ts-page-info').textContent).toBe('Page 1 of 1');
    });

    test('Next is enabled when more rows remain, and clicking it re-fetches at the next offset', async () => {
      const fetchMock = stubTableFetch({ total_rows: 120 });

      await renderTimeseriesTable(el, 'ts1', ['a', 'b'], 0);

      const nextBtn = el.querySelector('[data-ts-next]');
      expect(nextBtn.disabled).toBe(false);
      expect(el.querySelector('[data-ts-prev]').disabled).toBe(true);

      nextBtn.click();
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('offset=50')));
    });

    test('Prev clamps to offset 0 rather than going negative', async () => {
      const fetchMock = stubTableFetch({ total_rows: 120, offset: 20 });

      await renderTimeseriesTable(el, 'ts1', ['a', 'b'], 20);
      fetchMock.mockClear();

      el.querySelector('[data-ts-prev]').click();
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('offset=0')));
    });
  });

  test('on a fetch failure: shows the failure fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }));

    await renderTimeseriesTable(el, 'ts1', ['a', 'b'], 0);

    expect(el.textContent).toContain('Failed to load table data');
  });

  test('is a no-op when the target element is falsy', async () => {
    await expect(renderTimeseriesTable(null, 'ts1', ['a'], 0)).resolves.toBeUndefined();
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
      if (node && node.tagName === 'SCRIPT') {
        queueMicrotask(() => node.onerror && node.onerror());
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

    /** @type {any[]} */
    const scriptAppends = [];
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      if (node && node.tagName === 'SCRIPT') {
        scriptAppends.push(node);
        if (scriptAppends.length === 1) {
          // First injection: simulate a load failure, as the earlier test does.
          queueMicrotask(() => node.onerror && node.onerror());
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

    scriptAppends[1].onload();
    await renderPromise;

    expect(container2.textContent).not.toContain('Failed to load timeseries data');
    expect(Plotly.newPlot).toHaveBeenCalledTimes(1);
  });

  test('concurrent renderTimeseriesChart calls coalesce onto one _plotlyLoading promise (exactly one <script> injected)', async () => {
    vi.stubGlobal('Plotly', { newPlot: vi.fn((el) => { el.data = []; }) });
    const scriptAppends = [];
    vi.spyOn(document.head, 'appendChild').mockImplementation((node) => {
      if (node && node.tagName === 'SCRIPT') {
        scriptAppends.push(node);
        return node; // deliberately NOT firing onload yet, to inspect the in-flight state below
      }
      throw new Error('unexpected non-script appendChild in this isolated test');
    });

    const fresh = await import('../../../src/osprey/interfaces/artifacts/static/js/timeseries.js');
    const elA = document.createElement('div');
    const elB = document.createElement('div');
    const chartData = makeChartData();

    // Two callers racing the same not-yet-loaded Plotly: both start before
    // either resolves.
    const pA = fresh.renderTimeseriesChart(elA, chartData);
    const pB = fresh.renderTimeseriesChart(elB, chartData);

    expect(scriptAppends.length).toBe(1); // only the first call injected a <script>; the second reused its pending promise

    scriptAppends[0].onload();
    await Promise.all([pA, pB]);

    expect(Plotly.newPlot).toHaveBeenCalledTimes(2);
    expect(elA.data).toEqual([]);
    expect(elB.data).toEqual([]);
  });
});
