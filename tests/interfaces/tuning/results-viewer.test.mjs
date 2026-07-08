// @ts-check
/**
 * Behavioral tests for the OSPREY Tuning results viewer (results-viewer.js):
 * the analysis-tab renderer that draws a historical run's header plus its
 * four Plotly charts.
 *
 * `renderAnalysis` writes into a module-level `contentEl` captured by
 * `initResultsViewer`, then mounts four charts from `plots.js` inside a
 * `requestAnimationFrame` callback. To keep the module-level singletons
 * (`contentEl`, plots.js's re-render registry) fresh, each test re-imports
 * under `vi.resetModules()`. `requestAnimationFrame` is stubbed to run its
 * callback synchronously and `Plotly` is stubbed so the chart builders reach
 * an observable sink without a real charting runtime.
 *
 *   npx vitest run tests/interfaces/tuning/results-viewer.test.mjs
 *
 * @module tests/interfaces/tuning/results-viewer
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const MODULE_PATH = '../../../src/osprey/interfaces/tuning/static/js/results-viewer.js';

/** @typedef {import('../../../src/osprey/interfaces/tuning/static/js/plots.js').DataPoint} DataPoint */

/** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/results-viewer.js')} */
let mod;

/** Stubbed Plotly sink shared across a test's four chart mounts. */
const plotlyReact = vi.fn();

/** A minimal but complete run: every point carries the fields all four
 * chart builders read (objective + efficiency + a variables record), so
 * none of them early-returns.
 * @returns {DataPoint[]} */
function sampleRun() {
  return [
    { objective_value: 1, efficiency: 0.1, variables: { q1: 0.5, q2: 1.5 } },
    { objective_value: 3, efficiency: 0.4, variables: { q1: 0.7, q2: 1.2 } },
    { objective_value: 2, efficiency: 0.3, variables: { q1: 0.6, q2: 1.8 } },
  ];
}

beforeEach(async () => {
  document.body.innerHTML = '<div id="analysis-content"></div>';
  plotlyReact.mockClear();
  // The chart builders call Plotly.react; stub both entry points so the
  // suite is insulated from which one the builders happen to use.
  vi.stubGlobal('Plotly', { react: plotlyReact, newPlot: plotlyReact });
  // Run the rAF-deferred chart mounting synchronously inside the test.
  vi.stubGlobal('requestAnimationFrame', (/** @type {FrameRequestCallback} */ cb) => {
    cb(0);
    return 0;
  });
  vi.resetModules();
  mod = await import(MODULE_PATH);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('renderAnalysis — analysis view construction', () => {
  it('renders the run header with the point count', () => {
    mod.initResultsViewer();

    mod.renderAnalysis(sampleRun(), '2024-05-01T10:00:00Z');

    const content = /** @type {HTMLElement} */ (document.getElementById('analysis-content'));
    expect(content.textContent).toContain('2024-05-01T10:00:00Z');
    expect(content.textContent).toContain('3 points');
  });

  it('creates all four chart mount points', () => {
    mod.initResultsViewer();

    mod.renderAnalysis(sampleRun(), 'run-1');

    for (const id of ['chart-efficiency', 'chart-convergence', 'chart-params', 'chart-best']) {
      expect(document.getElementById(id)).not.toBeNull();
    }
  });

  it('invokes the Plotly sink once per chart (four mounts)', () => {
    mod.initResultsViewer();

    mod.renderAnalysis(sampleRun(), 'run-1');

    expect(plotlyReact).toHaveBeenCalledTimes(4);
  });
});

describe('renderAnalysis — hostile timestamp is inert', () => {
  it('escapes an injected timestamp so no element is parsed from it', () => {
    mod.initResultsViewer();
    const hostile = '<img src=x onerror="alert(1)">';

    mod.renderAnalysis(sampleRun(), hostile);

    const content = /** @type {HTMLElement} */ (document.getElementById('analysis-content'));
    // The payload survives as literal text in the header...
    expect(content.textContent).toContain(hostile);
    // ...but never as a live element in the parsed DOM.
    expect(content.querySelector('img')).toBeNull();
    expect(content.innerHTML).toContain('&lt;img');
  });
});
