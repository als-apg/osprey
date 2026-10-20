// @ts-check
/**
 * render — the one owner of every Plotly call the trajectory panel makes.
 *
 * An update reads the theme once — `{...chartTheme(), colorway: chartSeries(),
 * font}` — and hands that snapshot, with the operator's ranges, to a pure
 * builder from `figures.js`. The builder's layout is then merged with
 * `chartRelayout({layout})`, whose dotted keys are first nested, and the plot
 * is drawn with exactly one `Plotly.react`. `react` replaces the layout and
 * does not expand dotted keys, so the theme lives inside the figure it
 * renders, in nested form, and no relayout ever follows a react.
 *
 * A theme apply re-reacts each plot's last built figure with a fresh merge;
 * a change of the operator's ranges rebuilds each plot's last build. The
 * builder's `uirevision` is passed through untouched, so the operator's zoom
 * survives every update. Plots are resized when a tab is shown or a drawer
 * opens ({@link SHOW_EVENTS}), and `onPointClick` resolves a click to the
 * device named in the point's `customdata`.
 *
 * @module render
 */

import {
  chartRelayout,
  chartSeries,
  chartTheme,
  monoFontStack,
  subscribe,
} from '/design-system/js/theme-manager.js';

/**
 * The events, fired on (or bubbling from) the element that became visible,
 * after which the rendered plots inside it are resized: a centre/setup tab was
 * shown, or an `<osprey-drawer>` opened.
 */
export const SHOW_EVENTS = Object.freeze(['tab:show', 'drawer:open']);

/** The Plotly config every plot is drawn with. */
const PLOT_CONFIG = Object.freeze({ displaylogo: false, responsive: true });

/**
 * @typedef {import('./figures.js').Figure} Figure
 * @typedef {import('./figures.js').Theme} Theme
 * @typedef {import('./figures.js').Ranges} Ranges
 * @typedef {{ranges: Ranges}} BuildOptions
 * @typedef {(theme: Theme, options: BuildOptions) => Figure} Build
 * @typedef {(theme: Theme, options: BuildOptions) => Record<string, Figure>} GroupBuild
 * @typedef {(device: string, point: any) => void} PointHandler
 */

/**
 * @typedef {object} ThemeBridges
 * @property {() => Record<string, any>} chartTheme
 * @property {() => string[]} chartSeries
 * @property {() => string} monoFontStack
 * @property {(gd: any) => Record<string, any>} chartRelayout
 * @property {(fn: (id: string) => void) => () => void} subscribe
 */

/**
 * @typedef {object} AdapterOptions
 * @property {any} [Plotly] The Plotly global; `globalThis.Plotly` when omitted.
 * @property {ThemeBridges} [theme] theme-manager's bridges; the real ones when omitted.
 * @property {Document|HTMLElement} [root] Where plot hosts live and show events bubble to.
 */

/** @type {ThemeBridges} */
const DEFAULT_BRIDGES = Object.freeze({
  chartTheme,
  chartSeries,
  monoFontStack: () => monoFontStack(),
  chartRelayout,
  subscribe,
});

/** @param {unknown} value @returns {value is Record<string, any>} */
function isPlain(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * A flat relayout update with dotted keys (`'xaxis.gridcolor'`) as the nested
 * layout fragment `react` understands (`{xaxis: {gridcolor}}`).
 *
 * @param {Record<string, any>} update
 * @returns {Record<string, any>}
 */
export function nestDotted(update) {
  /** @type {Record<string, any>} */
  const nested = {};
  for (const [path, value] of Object.entries(update)) {
    const keys = path.split('.');
    let node = nested;
    for (const key of keys.slice(0, -1)) {
      if (!isPlain(node[key])) node[key] = {};
      node = node[key];
    }
    node[keys[keys.length - 1]] = value;
  }
  return nested;
}

/**
 * `base` with `update` merged in: plain objects merge key by key, anything
 * else (arrays included) is replaced. Neither input is modified.
 *
 * @param {Record<string, any>} base
 * @param {Record<string, any>} update
 * @returns {Record<string, any>}
 */
function merge(base, update) {
  const out = { ...base };
  for (const [key, value] of Object.entries(update)) {
    out[key] = isPlain(value) && isPlain(out[key]) ? merge(out[key], value) : value;
  }
  return out;
}

/**
 * @param {AdapterOptions} [options]
 */
export function createRenderAdapter(options = {}) {
  const bridges = options.theme || DEFAULT_BRIDGES;
  const root = options.root || document;
  const plotly = () => options.Plotly || /** @type {any} */ (globalThis).Plotly;

  /** @type {Ranges} */
  let ranges = {};
  /** The last group rendered into each plot (rebuilt on a range change). */
  /** @type {Map<string, {plotIds: Record<string, string>, build: GroupBuild}>} */
  const lastBuild = new Map();
  /** The last figure built for each plot, before the theme merge. */
  /** @type {Map<string, Figure>} */
  const lastFigure = new Map();
  /** @type {Map<string, Set<PointHandler>>} */
  const handlers = new Map();
  /** Graph divs that already carry the one click dispatcher. */
  const bound = new WeakSet();

  /** @param {string} plotId @returns {any} */
  const host = (plotId) =>
    'getElementById' in root ? root.getElementById(plotId) : root.querySelector(`#${CSS.escape(plotId)}`);

  /**
   * The theme snapshot one update hands to its builder.
   * @returns {Theme}
   */
  function theme() {
    const base = bridges.chartTheme();
    return {
      ...base,
      colorway: bridges.chartSeries(),
      font: { ...(base.font || {}), family: bridges.monoFontStack() },
    };
  }

  /** @param {string} plotId @param {any} gd */
  function bindClicks(plotId, gd) {
    if (bound.has(gd) || typeof gd.on !== 'function') return;
    bound.add(gd);
    gd.on('plotly_click', (/** @type {any} */ event) => {
      const point = event?.points?.[0];
      const device = point?.customdata;
      if (typeof device !== 'string' || !device) return;
      for (const handler of handlers.get(plotId) || []) handler(device, point);
    });
  }

  /**
   * Draw `figure` into its host with the theme merged in: one react.
   * @param {string} plotId @param {Figure} figure
   */
  function draw(plotId, figure) {
    const gd = host(plotId);
    if (!gd) return false;
    const themed = nestDotted(bridges.chartRelayout({ layout: figure.layout }));
    const layout = merge(figure.layout, themed);
    lastFigure.set(plotId, figure);
    plotly().react(gd, figure.data, layout, PLOT_CONFIG);
    bindClicks(plotId, gd);
    return true;
  }

  /**
   * Build a group of figures from one theme snapshot and draw each into the
   * plot its key names. Keys whose figure or host is missing are skipped.
   *
   * @param {Record<string, string>} plotIds figure key -> plot host id
   * @param {GroupBuild} build
   * @returns {boolean} whether every plot of the group was drawn
   */
  function renderGroup(plotIds, build) {
    const figures = build(theme(), { ranges: { ...ranges } });
    const group = { plotIds, build };
    let all = true;
    for (const [key, plotId] of Object.entries(plotIds)) {
      const figure = figures[key];
      if (!figure) {
        all = false;
        continue;
      }
      lastBuild.set(plotId, group);
      all = draw(plotId, figure) && all;
    }
    return all;
  }

  /**
   * Build one figure and draw it into `plotId`: one react.
   *
   * @param {string} plotId
   * @param {Build} build
   * @returns {boolean} whether the plot's host exists and was drawn
   */
  function render(plotId, build) {
    return renderGroup({ [plotId]: plotId }, (t, o) => ({ [plotId]: build(t, o) }));
  }

  /** Re-react every drawn plot's last figure with a fresh theme merge. */
  function retheme() {
    for (const [plotId, figure] of lastFigure) draw(plotId, figure);
  }

  /**
   * Merge the operator's ranges (`null` clears one back to the builder's
   * default) and rebuild every plot's last build once.
   *
   * @param {Ranges} partial
   */
  function setRanges(partial) {
    ranges = { ...ranges, ...partial };
    const groups = new Set(lastBuild.values());
    for (const group of groups) renderGroup(group.plotIds, group.build);
  }

  /**
   * Resize the drawn plots inside `scope` (all of them when omitted).
   * @param {Node} [scope]
   */
  function resize(scope) {
    const resizer = plotly()?.Plots?.resize;
    if (!resizer) return;
    for (const plotId of lastFigure.keys()) {
      const gd = host(plotId);
      if (gd && (!scope || scope === gd || scope.contains(gd))) resizer(gd);
    }
  }

  /** @param {Event} event */
  const onShow = (event) => {
    const target = event.target;
    resize(target instanceof Node ? target : undefined);
  };

  /**
   * Call `handler(device, point)` when a point of `plotId` naming a device in
   * its `customdata` is clicked. Returns an unsubscribe function.
   *
   * @param {string} plotId
   * @param {PointHandler} handler
   * @returns {() => void}
   */
  function onPointClick(plotId, handler) {
    let set = handlers.get(plotId);
    if (!set) handlers.set(plotId, (set = new Set()));
    set.add(handler);
    const gd = host(plotId);
    if (gd && lastFigure.has(plotId)) bindClicks(plotId, gd);
    return () => {
      handlers.get(plotId)?.delete(handler);
    };
  }

  const unsubscribe = bridges.subscribe(() => retheme());
  for (const name of SHOW_EVENTS) root.addEventListener(name, onShow);

  /** Stop following the theme and the show events. */
  function dispose() {
    unsubscribe();
    for (const name of SHOW_EVENTS) root.removeEventListener(name, onShow);
  }

  return {
    theme,
    render,
    renderGroup,
    setRanges,
    /** @returns {Ranges} the operator's current ranges */
    ranges: () => ({ ...ranges }),
    resize,
    onPointClick,
    dispose,
  };
}

/** @typedef {ReturnType<typeof createRenderAdapter>} RenderAdapter */
