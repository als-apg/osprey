// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Unit tests for the Artifact Gallery print module (print.js): the pure
 * `esc`/`fmtTime`/`headerHtml` formatting helpers and `printArtifact`'s
 * artifact_type -> printer-strategy dispatch (iframe / image / timeseries).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/print.test.mjs
 *
 * `window.open` is stubbed via `vi.stubGlobal('open', ...)` — the fake
 * window it returns wraps a REAL `document.implementation.createHTMLDocument()`
 * document (so `adoptNode`/`createElement`/`head`/`body` all behave exactly
 * like the browser DOM print.js relies on) plus `vi.fn()` spies for
 * `focus`/`print`/`close` and an `addEventListener` spy that just records the
 * `"load"` callback so tests can fire it deterministically (happy-dom's
 * `createHTMLDocument()` document never reaches `readyState === "complete"`
 * on its own, so print.js always takes its `addEventListener("load", ...)`
 * branch here — the same branch a real slow-loading print window takes).
 * `alert`/`Plotly` are stubbed the same way scaffold-detail.test.mjs and
 * timeseries.test.mjs do. No real print window, popup, or Plotly chart is
 * ever exercised — this is deliberately out of scope (see the module's own
 * header comment on strategy).
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  esc,
  fmtTime,
  headerHtml,
  printArtifact,
} from '../../../src/osprey/interfaces/artifacts/static/js/print.js';

const MSG_POPUP_BLOCKED = 'Print blocked — please allow pop-ups for this site and try again.';
const MSG_CHART_NOT_READY = 'Chart not yet rendered. Please wait for it to load, then try again.';
const MSG_CAPTURE_FAILED = 'Could not capture chart for printing.';

/** @returns {any} */
function makeArtifact(overrides = {}) {
  return {
    id: 'a1',
    title: 'Beam Profile',
    filename: 'beam_profile.png',
    artifact_type: 'plot_png',
    category: 'visualization',
    timestamp: '2026-07-01T10:00:00Z',
    tool_source: 'plot_tool',
    ...overrides,
  };
}

/**
 * A fake `window.open()` return value. `document` is a REAL document (via
 * `document.implementation.createHTMLDocument`) so `adoptNode`/`write`/
 * `open`/`close` behave exactly as print.js expects; `focus`/`print`/`close`
 * are spies, and `addEventListener` just records the `"load"` callback for
 * the test to fire manually.
 * @returns {any}
 */
function makeFakeWindow({ readyState } = {}) {
  const doc = document.implementation.createHTMLDocument('');
  // createHTMLDocument docs report "interactive" in happy-dom; tests for the
  // synchronous already-loaded print path override this to "complete".
  if (readyState) Object.defineProperty(doc, 'readyState', { value: readyState });
  /** @type {(() => void) | null} */
  let loadCb = null;
  return {
    document: doc,
    addEventListener: vi.fn((evt, cb) => { if (evt === 'load') loadCb = cb; }),
    focus: vi.fn(),
    print: vi.fn(),
    close: vi.fn(),
    fireLoad() { loadCb?.(); },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('esc (alias of the canonical design-system escapeHtml)', () => {
  test('escapes &, <, >, and "', () => {
    expect(esc('<b>"AT&T"</b>')).toBe('&lt;b&gt;&quot;AT&amp;T&quot;&lt;/b&gt;');
  });

  test('collapses only null/undefined to ""; other falsy values stringify', () => {
    expect(esc(null)).toBe('');
    expect(esc(undefined)).toBe('');
    expect(esc('')).toBe('');
    expect(esc(0)).toBe('0');
    expect(esc(false)).toBe('false');
  });

  test('coerces non-string input to a string before escaping', () => {
    expect(esc(42)).toBe('42');
  });

  test('escapes single quotes (canonical attribute-context-safe contract)', () => {
    expect(esc("it's a 'test'")).toBe('it&#39;s a &#39;test&#39;');
  });
});

describe('fmtTime', () => {
  test('returns "" for falsy input', () => {
    expect(fmtTime(null)).toBe('');
    expect(fmtTime(undefined)).toBe('');
    expect(fmtTime(0)).toBe('');
    expect(fmtTime('')).toBe('');
  });

  test('formats a valid timestamp via Date#toLocaleString', () => {
    const ts = '2026-07-03T15:45:00Z';
    expect(fmtTime(ts)).toBe(new Date(ts).toLocaleString());
  });

  test('non-ISO / fabricating input yields "" rather than an invented date', () => {
    // Shares types.js's isoToDate guard: bare `new Date("0")` would fabricate
    // a year-2000 date; the guard rejects these numeric/non-ISO strings.
    for (const junk of ['0', '1751000000000', 'not-a-date', 42, {}]) {
      expect(fmtTime(junk)).toBe('');
    }
  });
});

describe('headerHtml', () => {
  test('falls back title -> filename -> "Artifact" when fields are missing', () => {
    expect(headerHtml({})).toContain('<h1>Artifact</h1>');
    expect(headerHtml({ filename: 'x.png' })).toContain('<h1>x.png</h1>');
    expect(headerHtml({ title: 'My Plot', filename: 'x.png' })).toContain('<h1>My Plot</h1>');
  });

  test('omits the .print-meta block entirely when there is no timestamp/tool_source/filename', () => {
    const html = headerHtml({ title: 'Bare' });
    expect(html).not.toContain('print-meta');
  });

  test('joins timestamp, tool_source, and filename with middot separators', () => {
    const html = headerHtml({
      title: 'T',
      timestamp: '2026-07-03T15:45:00Z',
      tool_source: 'plot_tool',
      filename: 'x.png',
    });
    expect(html).toContain('<div class="print-meta">');
    expect(html).toContain(esc(fmtTime('2026-07-03T15:45:00Z')));
    expect(html).toContain('Source: plot_tool');
    expect(html).toContain('&middot;');
    expect(html).toContain('x.png');
  });

  test('escapes hostile title/tool_source/filename values', () => {
    const html = headerHtml({
      title: '<img src=x onerror=alert(1)>',
      tool_source: '"><script>1</script>',
      filename: '<b>f</b>',
    });
    expect(html).not.toContain('<img src=x');
    expect(html).not.toContain('<script>');
    expect(html).not.toContain('<b>f</b>');
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
  });
});

describe('printArtifact: guards and dispatch', () => {
  test('does nothing for a null/undefined artifact', () => {
    const openSpy = vi.fn();
    vi.stubGlobal('open', openSpy);
    printArtifact(null);
    printArtifact(undefined);
    expect(openSpy).not.toHaveBeenCalled();
  });

  test('alerts and does not throw when window.open is blocked (iframe path)', () => {
    const alertSpy = vi.fn();
    vi.stubGlobal('alert', alertSpy);
    vi.stubGlobal('open', vi.fn(() => null));

    expect(() => printArtifact(makeArtifact({ artifact_type: 'html' }))).not.toThrow();
    expect(alertSpy).toHaveBeenCalledWith(MSG_POPUP_BLOCKED);
  });

  describe('iframe strategy (plot_html, table_html, dashboard_html, html, text, json, unknown)', () => {
    test.each([
      ['plot_html', '/files/a1/beam_profile.png'],
      ['table_html', '/files/a1/beam_profile.png'],
      ['dashboard_html', '/files/a1/beam_profile.png'],
      ['html', '/files/a1/beam_profile.png'],
      ['text', '/files/a1/beam_profile.png'],
      ['json', '/files/a1/beam_profile.png'],
      ['some_unrecognized_type', '/files/a1/beam_profile.png'],
    ])('%s opens the raw file URL in a sized popup and injects print-cleanup styles', (artifact_type, expectedUrl) => {
      const win = makeFakeWindow();
      const openSpy = vi.fn(() => win);
      vi.stubGlobal('open', openSpy);

      printArtifact(makeArtifact({ artifact_type }));

      expect(openSpy).toHaveBeenCalledWith(expectedUrl, '_blank', 'width=900,height=700');
      expect(win.addEventListener).toHaveBeenCalledWith('load', expect.any(Function));

      win.fireLoad();
      expect(win.focus).toHaveBeenCalled();
      expect(win.print).toHaveBeenCalledTimes(1);
      const style = win.document.head.querySelector('style');
      expect(style?.textContent).toContain('@media print');
      expect(style?.textContent).toContain('.modebar');
    });

    test.each([
      ['markdown', '/api/markdown/a1/rendered'],
      ['notebook', '/api/notebooks/a1/rendered'],
    ])('%s opens its server-rendered endpoint', (artifact_type, expectedUrl) => {
      const win = makeFakeWindow();
      const openSpy = vi.fn(() => win);
      vi.stubGlobal('open', openSpy);

      printArtifact(makeArtifact({ artifact_type }));

      expect(openSpy).toHaveBeenCalledWith(expectedUrl, '_blank', 'width=900,height=700');
    });

    test('a cross-origin document access failure during style injection is swallowed and printing still proceeds', () => {
      const win = makeFakeWindow();
      // Force the try{} in onReady to throw, mirroring a cross-origin window.
      Object.defineProperty(win.document, 'head', {
        get() { throw new DOMException('cross-origin', 'SecurityError'); },
      });
      vi.stubGlobal('open', vi.fn(() => win));

      printArtifact(makeArtifact({ artifact_type: 'html' }));
      expect(() => win.fireLoad()).not.toThrow();
      expect(win.focus).toHaveBeenCalled();
      expect(win.print).toHaveBeenCalledTimes(1);
    });
  });

  describe('image strategy (plot_png, image)', () => {
    test.each(['plot_png', 'image'])('%s builds a self-contained print document with an escaped <img>', (artifact_type) => {
      const win = makeFakeWindow();
      vi.stubGlobal('open', vi.fn(() => win));

      printArtifact(makeArtifact({
        artifact_type,
        title: '<b>Hostile</b> Title',
        id: 'img-1',
        filename: 'p.png',
      }));

      win.fireLoad();

      // The document was built from an esc()-escaped HTML string, so the
      // parsed <title> element's own markup (innerHTML) carries the escaped
      // entities — textContent decodes them back, same as any other
      // browser-parsed element, so it round-trips to the raw title rather
      // than proving the escaping (that's what innerHTML is for here).
      const titleEl = win.document.querySelector('title');
      expect(titleEl?.textContent).toBe('<b>Hostile</b> Title');
      expect(titleEl?.innerHTML).toBe(esc('<b>Hostile</b> Title'));
      const img = win.document.querySelector('.print-body img');
      expect(img).not.toBeNull();
      expect(img?.getAttribute('src')).toBe('/files/img-1/p.png');
      // Attribute values decode entities on parse too, so this round-trips
      // to the raw title; innerHTML on the <h1> is the escaping proof.
      expect(img?.getAttribute('alt')).toBe('<b>Hostile</b> Title');
      const h1 = win.document.querySelector('.print-header h1');
      expect(h1?.textContent).toBe('<b>Hostile</b> Title');
      expect(h1?.innerHTML).toBe(esc('<b>Hostile</b> Title'));
      expect(win.focus).toHaveBeenCalled();
      expect(win.print).toHaveBeenCalledTimes(1);
    });

    test('alerts and does not throw when window.open is blocked', () => {
      const alertSpy = vi.fn();
      vi.stubGlobal('alert', alertSpy);
      vi.stubGlobal('open', vi.fn(() => null));

      expect(() => printArtifact(makeArtifact({ artifact_type: 'image' }))).not.toThrow();
      expect(alertSpy).toHaveBeenCalledWith(MSG_POPUP_BLOCKED);
    });

    test('an already-loaded document (readyState "complete") prints exactly once, synchronously', () => {
      const win = makeFakeWindow({ readyState: 'complete' });
      vi.stubGlobal('open', vi.fn(() => win));

      printArtifact(makeArtifact({ artifact_type: 'plot_png', id: 'img-2', filename: 'q.png' }));

      // The sync branch fired without any load event; the load listener is
      // still registered but must not produce a second print dialog here.
      expect(win.print).toHaveBeenCalledTimes(1);
      expect(win.focus).toHaveBeenCalledTimes(1);
    });
  });

  describe('timeseries strategy (dispatch precedes artifact_type switch)', () => {
    function mountChart() {
      document.body.innerHTML =
        '<div id="ts-viewport"><div data-ts-chart><div class="js-plotly-plot"></div></div></div>';
    }

    test.each([
      ['metadata.data_type === "timeseries"', { metadata: { data_type: 'timeseries' }, artifact_type: 'json' }],
      ['category === "archiver_data"', { category: 'archiver_data', artifact_type: 'json' }],
    ])('routes to the timeseries capture path when %s, overriding artifact_type', (_label, overrides) => {
      mountChart();
      const alertSpy = vi.fn();
      vi.stubGlobal('alert', alertSpy);
      vi.stubGlobal('Plotly', { toImage: vi.fn(() => Promise.resolve('data:image/png;base64,xyz')) });
      const win = makeFakeWindow();
      const openSpy = vi.fn(() => win);
      vi.stubGlobal('open', openSpy);

      printArtifact(makeArtifact(overrides));

      // Opens synchronously (before the async Plotly.toImage capture) to
      // dodge popup blockers.
      expect(openSpy).toHaveBeenCalledWith('about:blank', '_blank', 'width=900,height=700');
      expect(alertSpy).not.toHaveBeenCalled();
    });

    test('captures the chart via Plotly.toImage and prints once resolved', async () => {
      mountChart();
      vi.stubGlobal('Plotly', { toImage: vi.fn(() => Promise.resolve('data:image/png;base64,xyz')) });
      const win = makeFakeWindow();
      vi.stubGlobal('open', vi.fn(() => win));

      printArtifact(makeArtifact({ category: 'archiver_data', title: 'CH1' }));

      await vi.waitFor(() => { expect(win.print).toHaveBeenCalledTimes(1); });
      expect(win.focus).toHaveBeenCalled();
      const img = win.document.querySelector('.print-body img');
      expect(img?.getAttribute('src')).toBe('data:image/png;base64,xyz');
      expect(img?.getAttribute('alt')).toBe(esc('CH1') + ' chart');
    });

    test('alerts and does not open a window when the chart element is not yet rendered', () => {
      const alertSpy = vi.fn();
      vi.stubGlobal('alert', alertSpy);
      const openSpy = vi.fn();
      vi.stubGlobal('open', openSpy);
      // No mountChart(): #ts-viewport [data-ts-chart] .js-plotly-plot is absent.

      printArtifact(makeArtifact({ category: 'archiver_data' }));

      expect(alertSpy).toHaveBeenCalledWith(MSG_CHART_NOT_READY);
      expect(openSpy).not.toHaveBeenCalled();
    });

    test('alerts when the chart element exists but Plotly has not loaded', () => {
      mountChart();
      const alertSpy = vi.fn();
      vi.stubGlobal('alert', alertSpy);
      const openSpy = vi.fn();
      vi.stubGlobal('open', openSpy);
      // No Plotly stub: `typeof Plotly === "undefined"` is true.

      printArtifact(makeArtifact({ category: 'archiver_data' }));

      expect(alertSpy).toHaveBeenCalledWith(MSG_CHART_NOT_READY);
      expect(openSpy).not.toHaveBeenCalled();
    });

    test('alerts, logs, and closes the capture window when Plotly.toImage rejects', async () => {
      mountChart();
      const captureErr = new Error('capture failed');
      vi.stubGlobal('Plotly', { toImage: vi.fn(() => Promise.reject(captureErr)) });
      const alertSpy = vi.fn();
      vi.stubGlobal('alert', alertSpy);
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const win = makeFakeWindow();
      vi.stubGlobal('open', vi.fn(() => win));

      printArtifact(makeArtifact({ category: 'archiver_data' }));

      await vi.waitFor(() => { expect(alertSpy).toHaveBeenCalledWith(MSG_CAPTURE_FAILED); });
      expect(win.close).toHaveBeenCalled();
      expect(consoleErrorSpy).toHaveBeenCalledWith('[print.js] Plotly.toImage failed:', captureErr);
    });

    test('a popup blocker on the capture window alerts without touching Plotly', () => {
      mountChart();
      const toImage = vi.fn();
      vi.stubGlobal('Plotly', { toImage });
      const alertSpy = vi.fn();
      vi.stubGlobal('alert', alertSpy);
      vi.stubGlobal('open', vi.fn(() => null));

      printArtifact(makeArtifact({ category: 'archiver_data' }));

      expect(alertSpy).toHaveBeenCalledWith(MSG_POPUP_BLOCKED);
      expect(toImage).not.toHaveBeenCalled();
    });
  });
});
