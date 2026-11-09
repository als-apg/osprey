// @ts-check
/**
 * Unit tests for the commands a served Plotly page takes (plot-commands.js),
 * the page's half of the gallery's plot content strip:
 *   npx vitest run tests/interfaces/artifacts/plot-commands.test.mjs
 *
 * The module runs inside the sandboxed plot page (app.py `_RESPONSIVE_PLOTLY`
 * inlines it), so these tests hand it a fake Plotly (`relayout` and
 * `downloadImage` spies), graph divs carrying a computed `_fullLayout`, and a
 * fake window whose `parent` is the gallery.
 */

import { test, expect, describe, vi } from 'vitest';

import {
  FRAMED_ATTR,
  followPlotCommands,
  PLOT_COMMAND_MESSAGE,
  PLOT_COMMANDS,
  rememberViews,
  runPlotCommand,
} from '../../../src/osprey/interfaces/artifacts/static/js/plot-commands.js';

/** A Plotly whose calls the test reads. */
function fakePlotly() {
  return {
    relayout: vi.fn(() => Promise.resolve()),
    downloadImage: vi.fn(() => Promise.resolve()),
  };
}

/**
 * An axis as Plotly computes it: its range (numbers, or a date axis's
 * strings) and, by default, the identity mappings of a linear axis.
 * @param {(number | string)[]} range
 * @param {object} [extra]
 */
function axis(range, extra = {}) {
  return {
    range,
    autorange: true,
    r2l: (/** @type {any} */ value) => Number(value),
    l2r: (/** @type {number} */ value) => value,
    ...extra,
  };
}

/**
 * A graph div with two subplots and a legend (which is no axis).
 * @param {object} [overrides] full-layout entries to replace
 * @returns {any}
 */
function graph(overrides = {}) {
  return {
    _fullLayout: {
      xaxis: axis([0, 24]),
      yaxis: axis([0, 500]),
      xaxis2: axis([0, 200]),
      yaxis2: axis([-0.05, 0.15], { autorange: false }),
      legend: { x: 0 },
      ...overrides,
    },
  };
}

/**
 * The page's window in a frame (or alone), with its listeners.
 * @param {{framed?: boolean, plotly?: any, title?: string}} [options]
 */
function pageWindow({ framed = true, plotly = fakePlotly(), title = '' } = {}) {
  /** @type {((event: any) => void)[]} */
  const listeners = [];
  const gallery = { name: 'gallery', postMessage: vi.fn() };
  const root = document.createElement('html');
  /** @type {any} */
  const win = {
    Plotly: plotly,
    document: { documentElement: root, title },
    addEventListener: vi.fn((/** @type {string} */ type, /** @type {any} */ listener) => {
      if (type === 'message') listeners.push(listener);
    }),
  };
  win.parent = framed ? gallery : win;
  return {
    win,
    gallery,
    root,
    plotly,
    /**
     * @param {any} data
     * @param {any} [source] the posting window, the gallery by default
     */
    post(data, source = gallery) {
      for (const listener of listeners) listener({ data, source });
    },
  };
}

describe('runPlotCommand', () => {
  test('Box zoom and Pan set the drag mode', () => {
    const plotly = fakePlotly();
    const gd = graph();
    runPlotCommand(plotly, gd, 'box-zoom');
    runPlotCommand(plotly, gd, 'pan');
    expect(plotly.relayout.mock.calls).toEqual([
      [gd, { dragmode: 'zoom' }],
      [gd, { dragmode: 'pan' }],
    ]);
  });

  test('Zoom in halves every axis about its centre; Zoom out doubles it', () => {
    const plotly = fakePlotly();
    const gd = graph();
    runPlotCommand(plotly, gd, 'zoom-in');
    runPlotCommand(plotly, gd, 'zoom-out');
    const [[, zoomIn], [, zoomOut]] = /** @type {any[]} */ (plotly.relayout.mock.calls);
    expect(zoomIn).toMatchObject({
      'xaxis.range[0]': 6,
      'xaxis.range[1]': 18,
      'yaxis.range[0]': 125,
      'yaxis.range[1]': 375,
      'xaxis2.range[0]': 50,
      'xaxis2.range[1]': 150,
    });
    expect(zoomIn['yaxis2.range[0]']).toBeCloseTo(0);
    expect(zoomIn['yaxis2.range[1]']).toBeCloseTo(0.1);
    expect(zoomOut).toMatchObject({ 'xaxis.range[0]': -12, 'xaxis.range[1]': 36 });
    expect(Object.keys(zoomIn).some((key) => key.startsWith('legend'))).toBe(false);
  });

  test('a zoom maps through the axis (a date axis keeps its own values) and skips a fixed one', () => {
    const plotly = fakePlotly();
    const dates = axis(['2026-11-04 00:00', '2026-11-04 12:00'], {
      r2l: (/** @type {string} */ value) => Date.parse(`${value.replace(' ', 'T')}Z`),
      l2r: (/** @type {number} */ value) => new Date(value).toISOString(),
    });
    const gd = graph({ xaxis: dates, yaxis: axis([0, 10], { fixedrange: true }) });
    runPlotCommand(plotly, gd, 'zoom-in');
    const [[, update]] = /** @type {any[]} */ (plotly.relayout.mock.calls);
    expect(update['xaxis.range[0]']).toBe('2026-11-04T03:00:00.000Z');
    expect(update['xaxis.range[1]']).toBe('2026-11-04T09:00:00.000Z');
    expect('yaxis.range[0]' in update).toBe(false);
  });

  test('Autoscale fits every cartesian axis', () => {
    const plotly = fakePlotly();
    const gd = graph();
    runPlotCommand(plotly, gd, 'autoscale');
    expect(plotly.relayout).toHaveBeenCalledWith(gd, {
      'xaxis.autorange': true,
      'yaxis.autorange': true,
      'xaxis2.autorange': true,
      'yaxis2.autorange': true,
    });
  });

  test('Reset axes returns to the view the plot opened with, not to an autoscale', () => {
    const plotly = fakePlotly();
    const gd = graph();
    rememberViews([gd]);
    // The operator zooms; a second record keeps the first.
    gd._fullLayout.xaxis = axis([5, 6], { autorange: false });
    rememberViews([gd]);
    runPlotCommand(plotly, gd, 'reset-axes');
    expect(plotly.relayout).toHaveBeenCalledWith(gd, {
      'xaxis.autorange': true,
      'yaxis.autorange': true,
      'xaxis2.autorange': true,
      'yaxis2.range': [-0.05, 0.15],
    });
  });

  test('Reset axes on a plot never recorded autoscales', () => {
    const plotly = fakePlotly();
    const gd = graph({ xaxis2: undefined, yaxis2: undefined });
    runPlotCommand(plotly, gd, 'reset-axes');
    expect(plotly.relayout).toHaveBeenCalledWith(gd, {
      'xaxis.autorange': true,
      'yaxis.autorange': true,
    });
  });

  test('Export PNG downloads the plot as a PNG under the name it is given', () => {
    const plotly = fakePlotly();
    const gd = graph();
    runPlotCommand(plotly, gd, 'export-png', { filename: 'Orbit' });
    expect(plotly.downloadImage).toHaveBeenCalledWith(gd, { format: 'png', filename: 'Orbit' });
  });

  test('an unknown command does nothing', () => {
    const plotly = fakePlotly();
    expect(runPlotCommand(plotly, graph(), 'lasso')).toBeUndefined();
    expect(plotly.relayout).not.toHaveBeenCalled();
    expect(PLOT_COMMANDS).not.toContain('lasso');
  });
});

describe('followPlotCommands', () => {
  test('framed, it marks the page so its CSS hides the modebar', () => {
    const page = pageWindow();
    followPlotCommands(page.win, () => []);
    expect(page.root.hasAttribute(FRAMED_ATTR)).toBe(true);
  });

  test('on its own, the page keeps the modebar and listens to nothing', () => {
    const page = pageWindow({ framed: false });
    followPlotCommands(page.win, () => []);
    expect(page.root.hasAttribute(FRAMED_ATTR)).toBe(false);
    expect(page.win.addEventListener).not.toHaveBeenCalled();
  });

  test("runs the gallery's command on every plot of the page", () => {
    const page = pageWindow();
    const plots = [graph(), graph()];
    followPlotCommands(page.win, () => plots);
    page.post({ type: PLOT_COMMAND_MESSAGE, command: 'pan' });
    expect(page.plotly.relayout.mock.calls).toEqual([
      [plots[0], { dragmode: 'pan' }],
      [plots[1], { dragmode: 'pan' }],
    ]);
  });

  test('ignores another sender, another message type and an unknown command', () => {
    const page = pageWindow();
    followPlotCommands(page.win, () => [graph()]);
    page.post({ type: PLOT_COMMAND_MESSAGE, command: 'pan' }, { name: 'stranger' });
    page.post({ type: 'osprey-artifact-print' });
    page.post({ type: PLOT_COMMAND_MESSAGE, command: 'select' });
    page.post(null);
    expect(page.plotly.relayout).not.toHaveBeenCalled();
  });

  test("an export takes the message's file name, else the page title, else 'plot'", () => {
    const page = pageWindow({ title: 'Stored beam current' });
    const gd = graph();
    followPlotCommands(page.win, () => [gd]);
    page.post({ type: PLOT_COMMAND_MESSAGE, command: 'export-png', filename: ' Orbit ' });
    page.post({ type: PLOT_COMMAND_MESSAGE, command: 'export-png' });
    page.win.document.title = '';
    page.post({ type: PLOT_COMMAND_MESSAGE, command: 'export-png', filename: 7 });
    const filenames = page.plotly.downloadImage.mock.calls.map(
      (/** @type {any[]} */ call) => call[1].filename
    );
    expect(filenames).toEqual(['Orbit', 'Stored beam current', 'plot']);
  });

  test('a page without Plotly, or a plot that throws, is left alone', () => {
    const throwing = {
      relayout: vi.fn(() => {
        throw new Error('no plot');
      }),
      downloadImage: vi.fn(),
    };
    const page = pageWindow({ plotly: throwing });
    followPlotCommands(page.win, () => [graph()]);
    expect(() => page.post({ type: PLOT_COMMAND_MESSAGE, command: 'pan' })).not.toThrow();
    expect(throwing.relayout).toHaveBeenCalledTimes(1);

    const bare = pageWindow({ plotly: null });
    followPlotCommands(bare.win, () => [graph()]);
    expect(() => bare.post({ type: PLOT_COMMAND_MESSAGE, command: 'pan' })).not.toThrow();
  });
});
