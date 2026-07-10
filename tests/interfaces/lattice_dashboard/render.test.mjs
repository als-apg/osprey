// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Unit tests for the Lattice Dashboard rendering layer (render.js).
 *
 * happy-dom environment (configured globally), Plotly mocked via
 * vi.stubGlobal (the ambient `Plotly` global normally comes from the
 * vendored classic script — see vendor-globals.d.ts):
 *   npx vitest run tests/interfaces/lattice_dashboard/render.test.mjs
 *
 * Covers slider debounce coalescing, the figure-status LED/spinner state
 * primitives, and the shape of the layout renderPlotly() hands to
 * Plotly.react(). Does NOT exercise the live re-theme subscription wiring
 * end-to-end — that is covered by the browser load-smoke test
 * (tests/interfaces/test_load_smokes.py -m browser -k lattice).
 */

import { test, expect, vi, describe, afterEach, beforeEach } from 'vitest';

import {
  updateSummaryStats,
  updateLED,
  showSpinner,
  hideSpinner,
  showFigureError,
  updateFigureStatuses,
  renderPlotly,
  createRenderer,
} from '../../../src/osprey/interfaces/lattice_dashboard/static/js/render.js';

const FIGURE_NAMES = ['optics', 'da'];

/** Minimal DOM fixture matching lattice_dashboard/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `
    <span id="lattice-name"></span>
    <button id="btn-refresh"></button>
    <button id="btn-verify"></button>
    <button id="btn-baseline"></button>
    <div class="stat-chip" id="stat-energy"><span class="stat-value"></span></div>
    <div class="stat-chip" id="stat-circumference"><span class="stat-value"></span></div>
    <div class="stat-chip" id="stat-nux"><span class="stat-value"></span></div>
    <div class="stat-chip" id="stat-nuy"><span class="stat-value"></span></div>
    <div class="stat-chip" id="stat-chrom-x"><span class="stat-value"></span></div>
    <div class="stat-chip" id="stat-chrom-y"><span class="stat-value"></span></div>
    <div id="slider-container"></div>
    ${FIGURE_NAMES.map(name => `
      <span class="figure-led" id="led-${name}"></span>
      <div class="figure-plot" id="plot-${name}"></div>
    `).join('')}
  `;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('updateSummaryStats', () => {
  beforeEach(mountFixture);

  test('formats energy, circumference, tunes, and chromaticity into their chips', () => {
    updateSummaryStats({
      energy_gev: 1.9,
      circumference_m: 196.8,
      tunes: [14.25, 8.18],
      chromaticity: [1.4, 1.2],
    });
    expect(document.querySelector('#stat-energy .stat-value').textContent).toBe('1.90');
    expect(document.querySelector('#stat-circumference .stat-value').textContent).toBe('196.8');
    expect(document.querySelector('#stat-nux .stat-value').textContent).toBe('14.2500');
    expect(document.querySelector('#stat-nuy .stat-value').textContent).toBe('8.1800');
    expect(document.querySelector('#stat-chrom-x .stat-value').textContent).toBe('1.40');
    expect(document.querySelector('#stat-chrom-y .stat-value').textContent).toBe('1.20');
  });

  test('a missing/NaN value falls back to the em-dash placeholder', () => {
    updateSummaryStats({ energy_gev: null });
    expect(document.querySelector('#stat-energy .stat-value').textContent).toBe('—');
  });
});

describe('figure-status primitives (LED / spinner / error)', () => {
  beforeEach(mountFixture);

  test('updateLED sets the data-status attribute', () => {
    updateLED('optics', 'computing');
    expect(document.getElementById('led-optics').dataset.status).toBe('computing');
  });

  test('showSpinner adds a spinner once and is idempotent on repeat calls', () => {
    showSpinner('optics');
    showSpinner('optics');
    const spinners = document.querySelectorAll('#plot-optics .figure-spinner');
    expect(spinners.length).toBe(1);
  });

  test('hideSpinner removes the spinner', () => {
    showSpinner('optics');
    hideSpinner('optics');
    expect(document.querySelector('#plot-optics .figure-spinner')).toBeNull();
  });

  test('showFigureError clears any spinner and renders the truncated error message', () => {
    showSpinner('optics');
    showFigureError('optics', 'x'.repeat(300));
    expect(document.querySelector('#plot-optics .figure-spinner')).toBeNull();
    const detail = document.querySelector('#plot-optics .figure-error-msg');
    expect(detail.textContent.length).toBe(200);
  });

  test('showFigureError does not duplicate the error element on repeat calls', () => {
    showFigureError('optics', 'boom');
    showFigureError('optics', 'boom again');
    expect(document.querySelectorAll('#plot-optics .figure-error').length).toBe(1);
  });

  test('updateFigureStatuses drives LED + spinner + error per the state machine', () => {
    updateFigureStatuses({
      optics: { status: 'computing' },
      da: { status: 'error', error: 'solver diverged' },
    });
    expect(document.getElementById('led-optics').dataset.status).toBe('computing');
    expect(document.querySelector('#plot-optics .figure-spinner')).not.toBeNull();

    expect(document.getElementById('led-da').dataset.status).toBe('error');
    expect(document.querySelector('#plot-da .figure-spinner')).toBeNull();
    expect(document.querySelector('#plot-da .figure-error')).not.toBeNull();
  });

  test('a "ready" status hides any spinner left over from "computing"', () => {
    updateFigureStatuses({ optics: { status: 'computing' } });
    expect(document.querySelector('#plot-optics .figure-spinner')).not.toBeNull();
    updateFigureStatuses({ optics: { status: 'ready' } });
    expect(document.querySelector('#plot-optics .figure-spinner')).toBeNull();
  });
});

describe('renderPlotly', () => {
  beforeEach(() => {
    mountFixture();
    vi.stubGlobal('Plotly', { react: vi.fn(), relayout: vi.fn() });
  });

  test('merges theme colors, fixed font, and margin into the figure layout', () => {
    const figData = {
      data: [{ x: [1, 2], y: [3, 4] }],
      layout: { title: 'Optics', width: 400, height: 300 },
    };
    renderPlotly('optics', figData);

    expect(Plotly.react).toHaveBeenCalledTimes(1);
    const [plotEl, traces, layout, config] = Plotly.react.mock.calls[0];
    expect(plotEl).toBe(document.getElementById('plot-optics'));
    expect(traces).toBe(figData.data);

    // Theme-independent shape: fixed font family/size, fixed margin, autosize on.
    expect(layout.font.family).toBe('JetBrains Mono, monospace');
    expect(layout.font.size).toBe(10);
    expect(layout.margin).toEqual({ l: 45, r: 15, t: 30, b: 35 });
    expect(layout.autosize).toBe(true);
    // Explicit width/height from the backend must not fight autosize.
    expect(layout.width).toBeUndefined();
    expect(layout.height).toBeUndefined();
    // Title is preserved from the original layout.
    expect(layout.title).toBe('Optics');
    // Theme colors are present (real chartTheme() reads CSS vars — empty
    // string in this unstyled test DOM — the key shape is what's pinned).
    expect(layout).toHaveProperty('paper_bgcolor');
    expect(layout).toHaveProperty('plot_bgcolor');

    expect(config).toEqual({
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
      displaylogo: false,
    });
  });

  test('merges grid/zeroline colors into every x/y axis present in the source layout', () => {
    const figData = {
      data: [],
      layout: { xaxis: { title: 'x' }, yaxis2: { title: 'y2' } },
    };
    renderPlotly('optics', figData);
    const [, , layout] = Plotly.react.mock.calls[0];
    expect(layout.xaxis).toHaveProperty('gridcolor');
    expect(layout.xaxis).toHaveProperty('zerolinecolor');
    expect(layout.xaxis.title).toBe('x');
    expect(layout.yaxis2).toHaveProperty('gridcolor');
    expect(layout.yaxis2.title).toBe('y2');
  });

  test('annotations keep their own color if set, otherwise inherit the theme font color', () => {
    const figData = {
      data: [],
      layout: {
        annotations: [
          { text: 'a', font: { color: 'red' } },
          { text: 'b' },
        ],
      },
    };
    renderPlotly('optics', figData);
    const [, , layout] = Plotly.react.mock.calls[0];
    expect(layout.annotations[0].font.color).toBe('red');
    expect(layout.annotations[1].font.color).toBe(layout.font.color);
  });

  test('does nothing when the target plot element is absent (unknown figure name)', () => {
    expect(() => renderPlotly('does-not-exist', { data: [], layout: {} })).not.toThrow();
    expect(Plotly.react).not.toHaveBeenCalled();
  });
});

describe('createRenderer / updateSliders debounce', () => {
  let callbacks;

  beforeEach(() => {
    mountFixture();
    vi.useFakeTimers();
    callbacks = {
      onSliderChange: vi.fn(),
      onFigureReady: vi.fn(),
      getOverrides: vi.fn(() => undefined),
    };
  });

  function families() {
    return {
      'QF1': { type: 'quadrupole', value: 1.2345, range: [-5, 5] },
    };
  }

  test('rapid slider "change" events coalesce into a single debounced callback with the LAST value', () => {
    const renderer = createRenderer(FIGURE_NAMES, callbacks);
    renderer.updateSliders(families());

    const slider = document.getElementById('slider-QF1');
    expect(slider).not.toBeNull();

    slider.value = '1.0';
    slider.dispatchEvent(new Event('change'));
    slider.value = '2.0';
    slider.dispatchEvent(new Event('change'));
    slider.value = '3.0';
    slider.dispatchEvent(new Event('change'));

    expect(callbacks.onSliderChange).not.toHaveBeenCalled();
    vi.advanceTimersByTime(350);
    expect(callbacks.onSliderChange).toHaveBeenCalledTimes(1);
    expect(callbacks.onSliderChange).toHaveBeenCalledWith('QF1', 3.0);
  });

  test('a slider override value from getOverrides() takes precedence over the family baseline', () => {
    callbacks.getOverrides = vi.fn(() => ({ QF1: 4.2 }));
    const renderer = createRenderer(FIGURE_NAMES, callbacks);
    renderer.updateSliders(families());

    const valueEl = document.getElementById('slider-val-QF1');
    expect(valueEl.textContent).toBe('4.2000');
  });

  test('renderState routes each already-ready figure through onFigureReady', () => {
    const renderer = createRenderer(FIGURE_NAMES, callbacks);
    renderer.renderState({
      base_lattice: 'sr.lat',
      summary: {},
      families: {},
      figures: {
        optics: { status: 'ready' },
        da: { status: 'computing' },
      },
    });
    expect(callbacks.onFigureReady).toHaveBeenCalledTimes(1);
    expect(callbacks.onFigureReady).toHaveBeenCalledWith('optics');
  });

  test('renderState is a no-op on a null/undefined state', () => {
    const renderer = createRenderer(FIGURE_NAMES, callbacks);
    expect(() => renderer.renderState(null)).not.toThrow();
    expect(callbacks.onFigureReady).not.toHaveBeenCalled();
  });
});
