// @ts-check
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SHOW_EVENTS, createRenderAdapter, nestDotted } from './render.js';

/** Theme values as the computed-style bridges return them (no hex literals). */
const TEXT = 'rgb(200, 200, 200)';
const GRID = 'rgb(60, 60, 60)';

/**
 * A stand-in for theme-manager's bridges whose colours can be switched, with
 * a `fire()` that runs every subscriber like an apply() would.
 */
function fakeTheme() {
  const state = { text: TEXT, grid: GRID, paper: 'rgb(10, 10, 10)', series: ['s1', 's2'] };
  /** @type {Set<(id: string) => void>} */
  const subscribers = new Set();
  return {
    state,
    subscribers,
    fire(id = 'dark') {
      for (const fn of subscribers) fn(id);
    },
    bridges: {
      chartTheme: () => ({
        paper_bgcolor: state.paper,
        plot_bgcolor: state.paper,
        font: { color: state.text },
        xaxis: { gridcolor: state.grid },
        yaxis: { gridcolor: state.grid },
      }),
      chartSeries: () => [...state.series],
      monoFontStack: () => 'mono',
      /** @param {any} gd */
      chartRelayout: (gd) => {
        /** @type {Record<string, any>} */
        const update = {
          paper_bgcolor: state.paper,
          plot_bgcolor: state.paper,
          'font.color': state.text,
          colorway: [...state.series],
          'font.family': 'mono',
          'legend.bgcolor': state.paper,
        };
        for (const key of Object.keys(gd?.layout || {}).filter((k) => /^[xy]axis\d*$/.test(k))) {
          update[`${key}.gridcolor`] = state.grid;
          update[`${key}.linecolor`] = state.grid;
        }
        return update;
      },
      /** @param {(id: string) => void} fn */
      subscribe: (fn) => {
        subscribers.add(fn);
        return () => subscribers.delete(fn);
      },
    },
  };
}

/** A stubbed Plotly global recording every call; graph divs gain `on`. */
function fakePlotly() {
  const calls = /** @type {{react: any[][], resize: any[], relayout: any[][], newPlot: any[][]}} */ ({
    react: [],
    resize: [],
    relayout: [],
    newPlot: [],
  });
  return {
    calls,
    /** @param {any} gd @param {any[]} data @param {any} layout @param {any} config */
    react: vi.fn((gd, data, layout, config) => {
      calls.react.push([gd, data, layout, config]);
      if (!gd.on) {
        /** @type {Record<string, ((ev: any) => void)[]>} */
        gd._handlers = {};
        gd.on = (/** @type {string} */ name, /** @type {(ev: any) => void} */ fn) => {
          (gd._handlers[name] ||= []).push(fn);
        };
        gd.emit = (/** @type {string} */ name, /** @type {any} */ ev) => {
          for (const fn of gd._handlers[name] || []) fn(ev);
        };
      }
      return Promise.resolve(gd);
    }),
    relayout: vi.fn((...args) => calls.relayout.push(args)),
    newPlot: vi.fn((...args) => calls.newPlot.push(args)),
    Plots: { resize: vi.fn((/** @type {any} */ gd) => calls.resize.push(gd)) },
  };
}

/** A figure as a pure builder returns it, the theme baked in nested form. */
/** @param {Record<string, any>} theme @param {{ranges?: any}} [options] @param {string} [rev] */
function figure(theme, options = {}, rev = 'traj-x') {
  return {
    data: [{ type: 'scatter', x: [1, 2], y: [3, 4], customdata: ['BPM1', 'BPM2'] }],
    layout: {
      ...theme,
      uirevision: rev,
      xaxis: { ...(theme.xaxis || {}), title: { text: 'BPM index' } },
      yaxis: { ...(theme.yaxis || {}), range: options.ranges?.x || [-10, 10] },
    },
  };
}

/** @param {any} value @param {string} [path] @returns {string[]} */
function dottedKeys(value, path = '') {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
  return Object.entries(value).flatMap(([key, child]) => [
    ...(key.includes('.') ? [`${path}${key}`] : []),
    ...dottedKeys(child, `${path}${key}/`),
  ]);
}

/** @type {ReturnType<typeof fakeTheme>} */ let theme;
/** @type {ReturnType<typeof fakePlotly>} */ let plotly;
/** @type {ReturnType<typeof createRenderAdapter>} */ let adapter;

beforeEach(() => {
  document.body.innerHTML = `
    <section id="tab-traj">
      <div id="plot-traj-x"></div><div id="plot-traj-y"></div>
    </section>
    <osprey-drawer id="drawer"><div id="drawer-attenuation"></div></osprey-drawer>`;
  theme = fakeTheme();
  plotly = fakePlotly();
  adapter = createRenderAdapter({ Plotly: plotly, theme: theme.bridges, root: document });
});

afterEach(() => {
  adapter.dispose();
  document.body.innerHTML = '';
});

describe('nestDotted', () => {
  it('turns dotted keys into nested objects and keeps plain keys', () => {
    expect(nestDotted({ 'font.color': 'a', 'xaxis.gridcolor': 'b', 'xaxis.linecolor': 'c', colorway: ['d'] })).toEqual({
      font: { color: 'a' },
      xaxis: { gridcolor: 'b', linecolor: 'c' },
      colorway: ['d'],
    });
  });

  it('nests deep paths', () => {
    expect(nestDotted({ 'scene.xaxis.color': 'a' })).toEqual({ scene: { xaxis: { color: 'a' } } });
  });
});

describe('theme snapshot', () => {
  it('is chartTheme plus the series as colorway and the mono font family', () => {
    const snap = adapter.theme();
    expect(snap.font).toEqual({ color: TEXT, family: 'mono' });
    expect(snap.xaxis).toEqual({ gridcolor: GRID });
    expect(snap.colorway).toEqual(['s1', 's2']);
  });
});

describe('render', () => {
  it('reacts once with the theme nested inside the layout and no dotted key', async () => {
    const build = vi.fn((/** @type {any} */ t, /** @type {any} */ o) => figure(t, o));
    expect(adapter.render('plot-traj-x', build)).toBe(true);
    expect(build).toHaveBeenCalledTimes(1);
    expect(build.mock.calls[0][0].font.color).toBe(TEXT);
    expect(plotly.react).toHaveBeenCalledTimes(1);
    const [gd, data, layout] = plotly.calls.react[0];
    expect(gd).toBe(document.getElementById('plot-traj-x'));
    expect(data[0].customdata).toEqual(['BPM1', 'BPM2']);
    expect(layout.font.color).toBe(TEXT);
    expect(layout.font.family).toBe('mono');
    expect(layout.xaxis.gridcolor).toBe(GRID);
    expect(layout.xaxis.linecolor).toBe(GRID);
    expect(layout.yaxis.gridcolor).toBe(GRID);
    expect(layout.legend.bgcolor).toBe(theme.state.paper);
    expect(layout.xaxis.title).toEqual({ text: 'BPM index' });
    expect(layout.uirevision).toBe('traj-x');
    expect(dottedKeys(layout)).toEqual([]);
    expect(plotly.relayout).not.toHaveBeenCalled();
    expect(plotly.newPlot).not.toHaveBeenCalled();
  });

  it('reacts exactly once per update and keeps uirevision across updates', () => {
    adapter.render('plot-traj-x', (t, o) => figure(t, o));
    adapter.render('plot-traj-x', (t, o) => figure(t, o));
    adapter.render('plot-traj-x', (t, o) => figure(t, o));
    expect(plotly.react).toHaveBeenCalledTimes(3);
    expect(plotly.calls.react.map((c) => c[2].uirevision)).toEqual(['traj-x', 'traj-x', 'traj-x']);
    expect(plotly.relayout).not.toHaveBeenCalled();
  });

  it('renders a group with one theme snapshot and one react per plot', () => {
    const chartTheme = vi.spyOn(theme.bridges, 'chartTheme');
    const build = vi.fn((/** @type {any} */ t, /** @type {any} */ o) => ({ x: figure(t, o, 'traj-x'), y: figure(t, o, 'traj-y') }));
    adapter.renderGroup({ x: 'plot-traj-x', y: 'plot-traj-y' }, build);
    expect(build).toHaveBeenCalledTimes(1);
    expect(chartTheme).toHaveBeenCalledTimes(1);
    expect(plotly.react).toHaveBeenCalledTimes(2);
    expect(plotly.calls.react.map((c) => c[0].id)).toEqual(['plot-traj-x', 'plot-traj-y']);
  });

  it('skips a plot whose host is not in the document', () => {
    expect(adapter.render('plot-missing', (t) => figure(t))).toBe(false);
    expect(plotly.react).not.toHaveBeenCalled();
  });
});

describe('operator ranges', () => {
  it('hands the ranges to every build and keeps them across updates', () => {
    adapter.setRanges({ x: [-2, 2] });
    adapter.render('plot-traj-x', (t, o) => figure(t, o));
    adapter.render('plot-traj-x', (t, o) => figure(t, o));
    expect(plotly.calls.react.map((c) => c[2].yaxis.range)).toEqual([
      [-2, 2],
      [-2, 2],
    ]);
    expect(adapter.ranges()).toEqual({ x: [-2, 2] });
  });

  it('re-renders the last builds once when the ranges change', () => {
    adapter.renderGroup({ x: 'plot-traj-x', y: 'plot-traj-y' }, (t, o) => ({ x: figure(t, o), y: figure(t, o) }));
    plotly.react.mockClear();
    plotly.calls.react.length = 0;
    adapter.setRanges({ x: [-3, 3] });
    expect(plotly.react).toHaveBeenCalledTimes(2);
    expect(plotly.calls.react[0][2].yaxis.range).toEqual([-3, 3]);
  });
});

describe('theme change', () => {
  it('re-reacts the last figure with a fresh merge, once per plot', () => {
    const build = vi.fn((/** @type {any} */ t, /** @type {any} */ o) => figure(t, o));
    adapter.render('plot-traj-x', build);
    adapter.render('plot-traj-y', (t, o) => figure(t, o, 'traj-y'));
    expect(plotly.react).toHaveBeenCalledTimes(2);

    theme.state.text = 'rgb(1, 2, 3)';
    theme.state.grid = 'rgb(4, 5, 6)';
    theme.fire('light');

    expect(plotly.react).toHaveBeenCalledTimes(4);
    expect(build).toHaveBeenCalledTimes(1);
    const layout = plotly.calls.react[2][2];
    expect(layout.font.color).toBe('rgb(1, 2, 3)');
    expect(layout.xaxis.gridcolor).toBe('rgb(4, 5, 6)');
    expect(layout.uirevision).toBe('traj-x');
    expect(dottedKeys(layout)).toEqual([]);
    expect(plotly.calls.react[2][1]).toBe(plotly.calls.react[0][1]);
  });

  it('reacts on nothing else: no theme change, no resize, no click', () => {
    adapter.render('plot-traj-x', (t) => figure(t));
    adapter.resize();
    document.getElementById('tab-traj')?.dispatchEvent(new CustomEvent('tab:show', { bubbles: true }));
    adapter.onPointClick('plot-traj-x', () => {});
    expect(plotly.react).toHaveBeenCalledTimes(1);
  });

  it('stops following the theme once disposed', () => {
    adapter.render('plot-traj-x', (t) => figure(t));
    adapter.dispose();
    theme.fire();
    expect(plotly.react).toHaveBeenCalledTimes(1);
    expect(theme.subscribers.size).toBe(0);
  });
});

describe('resize', () => {
  it('lists the tab-show and drawer-open events', () => {
    expect(SHOW_EVENTS).toEqual(['tab:show', 'drawer:open']);
  });

  it('resizes the rendered plots inside a shown tab, and only those', () => {
    adapter.render('plot-traj-x', (t) => figure(t));
    adapter.render('drawer-attenuation', (t) => figure(t, {}, 'drawer-attenuation'));
    document.getElementById('tab-traj')?.dispatchEvent(new CustomEvent('tab:show', { bubbles: true }));
    expect(plotly.calls.resize.map((gd) => gd.id)).toEqual(['plot-traj-x']);
  });

  it('resizes the drawer plots on drawer:open', () => {
    adapter.render('plot-traj-x', (t) => figure(t));
    adapter.render('drawer-attenuation', (t) => figure(t, {}, 'drawer-attenuation'));
    document
      .getElementById('drawer')
      ?.dispatchEvent(new CustomEvent('drawer:open', { bubbles: true, composed: true }));
    expect(plotly.calls.resize.map((gd) => gd.id)).toEqual(['drawer-attenuation']);
  });

  it('never resizes a plot that was not rendered yet', () => {
    adapter.resize();
    expect(plotly.Plots.resize).not.toHaveBeenCalled();
  });

  it('resizes every rendered plot when asked without a scope', () => {
    adapter.render('plot-traj-x', (t) => figure(t));
    adapter.render('plot-traj-y', (t) => figure(t));
    adapter.resize();
    expect(plotly.calls.resize.map((gd) => gd.id)).toEqual(['plot-traj-x', 'plot-traj-y']);
  });
});

describe('onPointClick', () => {
  it('resolves a click to the device named in customdata', () => {
    const handler = vi.fn();
    adapter.onPointClick('plot-traj-x', handler);
    adapter.render('plot-traj-x', (t) => figure(t));
    const gd = /** @type {any} */ (document.getElementById('plot-traj-x'));
    const point = { customdata: 'BPM2', pointIndex: 1 };
    gd.emit('plotly_click', { points: [point] });
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith('BPM2', point);
  });

  it('binds once however often the plot reacts, and works when subscribed after the first render', () => {
    adapter.render('plot-traj-x', (t) => figure(t));
    const handler = vi.fn();
    adapter.onPointClick('plot-traj-x', handler);
    adapter.render('plot-traj-x', (t) => figure(t));
    adapter.render('plot-traj-x', (t) => figure(t));
    const gd = /** @type {any} */ (document.getElementById('plot-traj-x'));
    expect(gd._handlers.plotly_click).toHaveLength(1);
    gd.emit('plotly_click', { points: [{ customdata: 'BPM1' }] });
    expect(handler).toHaveBeenCalledWith('BPM1', { customdata: 'BPM1' });
  });

  it('ignores a click on a point that names no device', () => {
    const handler = vi.fn();
    adapter.onPointClick('plot-traj-x', handler);
    adapter.render('plot-traj-x', (t) => figure(t));
    const gd = /** @type {any} */ (document.getElementById('plot-traj-x'));
    gd.emit('plotly_click', { points: [{ customdata: null }] });
    gd.emit('plotly_click', { points: [] });
    gd.emit('plotly_click', {});
    expect(handler).not.toHaveBeenCalled();
  });

  it('stops calling a handler once unsubscribed', () => {
    const handler = vi.fn();
    const off = adapter.onPointClick('plot-traj-x', handler);
    adapter.render('plot-traj-x', (t) => figure(t));
    off();
    const gd = /** @type {any} */ (document.getElementById('plot-traj-x'));
    gd.emit('plotly_click', { points: [{ customdata: 'BPM1' }] });
    expect(handler).not.toHaveBeenCalled();
  });

  it('only reaches handlers of the plot that was clicked', () => {
    const onX = vi.fn();
    const onY = vi.fn();
    adapter.onPointClick('plot-traj-x', onX);
    adapter.onPointClick('plot-traj-y', onY);
    adapter.render('plot-traj-x', (t) => figure(t));
    adapter.render('plot-traj-y', (t) => figure(t));
    /** @type {any} */ (document.getElementById('plot-traj-y')).emit('plotly_click', { points: [{ customdata: 'BPM3' }] });
    expect(onX).not.toHaveBeenCalled();
    expect(onY).toHaveBeenCalledWith('BPM3', { customdata: 'BPM3' });
  });
});
