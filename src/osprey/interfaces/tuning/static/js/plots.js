// @ts-check
/**
 * OSPREY Tuning — Plotly Chart Builders
 *
 * Colors and layout come from the shared design system: chartTheme()/
 * chartSeries() (theme-manager.js's computed-style bridges over
 * tokens.css's --chart-* custom properties) for backgrounds/grid/font and
 * the categorical variable-trace palette, plus a couple of direct CSS
 * custom-property reads for named semantic roles (LHS/initial-point
 * colors) that aren't part of the categorical series and so don't belong
 * behind chartSeries()'s "pick the Nth color" contract.
 *
 * THE BUGFIX: every exported render function re-renders on theme change
 * via a shared subscribe() registration below. Previously all five
 * functions used a single hardcoded DARK_LAYOUT/COLORS pair with no
 * light-theme variant at all — Plotly figures stayed dark no matter what
 * theme was active. Each function now closes over a `render` callback
 * that re-reads live theme colors and re-registers itself, so a later
 * theme flip redraws every currently-visible plot with its last-known
 * data, not just newly-created ones. (Tuning's plot containers are
 * stable, long-lived DOM nodes referenced by id — never destroyed and
 * recreated by tab switching — so a simple Map keyed by container never
 * accumulates stale entries in practice.)
 */

import { chartSeries, chartTheme, subscribe } from '/design-system/js/theme-manager.js';

/**
 * A single optimization sample. Extra keys (`phase`, `idx`) are attached to
 * the merged point records built inside `createOptimizationPlot`.
 * @typedef {object} DataPoint
 * @property {number} [objective_value]
 * @property {number} [objective]
 * @property {number} [efficiency]
 * @property {number} [improvement]
 * @property {Record<string, number>} [variables]
 * @property {Record<string, {min: number, max: number}>} [variable_bounds]
 */

/**
 * Optimization run state consumed by the dual-axis objective/variable plot.
 * @typedef {object} OptState
 * @property {DataPoint[]} [lhs_data]
 * @property {DataPoint[]} [bo_data]
 * @property {DataPoint[]} [snapshots]
 */

/** @param {string} name  @returns {string} */
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Named semantic role colors that aren't part of the categorical series. */
function roleColors() {
  return {
    lhs: cssVar('--ansi-blue'),
    initial: cssVar('--ansi-magenta'),
    teal: cssVar('--color-accent-light'),
    amber: cssVar('--color-amber'),
  };
}

const CONFIG = {
  responsive: true,
  displayModeBar: false,
};

/** Shared layout fragment, freshly read from the current theme every call. */
function baseLayout() {
  const theme = chartTheme();
  const axis = { gridcolor: theme.xaxis.gridcolor, zerolinecolor: theme.xaxis.gridcolor };
  return {
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: { family: 'Outfit, sans-serif', color: theme.font.color, size: 11 },
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { ...axis },
    yaxis: { ...axis },
    legend: { bgcolor: 'transparent', font: { size: 10 } },
  };
}

// ---- Theme-change re-render registry ----

const _rerenderByContainer = new Map();

subscribe(() => {
  for (const rerender of _rerenderByContainer.values()) {
    try {
      rerender();
    } catch (error) {
      console.error('osprey tuning: plot re-render on theme change failed', error);
    }
  }
});

/** Run `render` now and register it to re-run (with the same closed-over data) on every theme change.
 * @param {HTMLElement} container  @param {() => void} render  @returns {void} */
function withRerender(container, render) {
  render();
  _rerenderByContainer.set(container, render);
}

/**
 * Build the dual-axis optimization plot (objective + variables).
 * @param {HTMLElement} container
 * @param {OptState} optState
 * @param {string} [displayMode]
 * @returns {void}
 */
export function createOptimizationPlot(container, optState, displayMode = 'normalized') {
  const render = () => {
    const colors = roleColors();
    const palette = chartSeries();
    /** @type {any[]} */
    const traces = [];
    const { lhs_data, bo_data, snapshots } = optState;

    // Combine all data points
    /** @type {any[]} */
    const allPoints = [];

    // Initial point (index 0 if exists)
    if (snapshots && snapshots.length > 0) {
      const init = snapshots[0];
      allPoints.push({ ...init, phase: 'initial', idx: 0 });
    }

    // LHS phase
    if (lhs_data && lhs_data.length > 0) {
      lhs_data.forEach((pt, i) => {
        allPoints.push({ ...pt, phase: 'lhs', idx: i + 1 });
      });
    }

    // BO phase
    if (bo_data && bo_data.length > 0) {
      bo_data.forEach((pt, i) => {
        allPoints.push({ ...pt, phase: 'bo', idx: (lhs_data?.length || 0) + i + 1 });
      });
    }

    if (allPoints.length === 0) {
      container.innerHTML = '<div class="plot-empty">No data yet</div>';
      return;
    }

    // Clear empty state
    const emptyEl = container.querySelector('.plot-empty');
    if (emptyEl) emptyEl.remove();

    // Objective trace (top subplot)
    const objectiveX = allPoints.map((p) => p.idx);
    const objectiveY = allPoints.map((p) => p.objective_value ?? p.objective ?? null);
    const phaseColors = allPoints.map((p) => {
      if (p.phase === 'initial') return colors.initial;
      if (p.phase === 'lhs') return colors.lhs;
      return colors.teal;
    });

    traces.push({
      x: objectiveX,
      y: objectiveY,
      mode: 'lines+markers',
      name: 'Objective',
      marker: { color: phaseColors, size: 6 },
      line: { color: colors.teal, width: 1.5 },
      yaxis: 'y',
    });

    // Variable traces (bottom subplot)
    if (allPoints[0]?.variables) {
      const varNames = Object.keys(allPoints[0].variables);
      varNames.forEach((name, vi) => {
        const color = palette[vi % palette.length];
        const yVals = allPoints.map((p) => {
          const val = p.variables?.[name];
          if (displayMode === 'normalized' && p.variable_bounds?.[name]) {
            const { min, max } = p.variable_bounds[name];
            return max > min ? (val - min) / (max - min) : val;
          }
          return val ?? null;
        });

        traces.push({
          x: objectiveX,
          y: yVals,
          mode: 'lines+markers',
          name: name,
          marker: { size: 4, color },
          line: { width: 1, color, dash: 'dot' },
          yaxis: 'y2',
        });
      });
    }

    const base = baseLayout();
    const layout = {
      ...base,
      margin: { l: 50, r: 50, t: 10, b: 40 },
      xaxis: { ...base.xaxis, title: 'Iteration', domain: [0, 1] },
      yaxis: { ...base.yaxis, title: 'Objective', domain: [0.55, 1] },
      yaxis2: {
        ...base.yaxis,
        title: displayMode === 'normalized' ? 'Variables (norm)' : 'Variables',
        domain: [0, 0.45],
        anchor: 'x',
      },
      showlegend: true,
      legend: { ...base.legend, x: 1, xanchor: 'right', y: 1 },
      hovermode: 'x unified',
    };

    Plotly.react(container, traces, layout, CONFIG);
  };

  withRerender(container, render);
}

/**
 * Efficiency over iterations — line chart.
 * @param {HTMLElement} container  @param {DataPoint[]} data  @returns {void}
 */
export function createEfficiencyPlot(container, data) {
  if (!data || data.length === 0) return;

  const render = () => {
    const colors = roleColors();
    const x = data.map((_, i) => i + 1);
    const y = data.map((d) => d.efficiency ?? d.improvement ?? 0);

    const traces = [
      {
        x,
        y,
        mode: 'lines+markers',
        name: 'Efficiency',
        line: { color: colors.teal, width: 2 },
        marker: { size: 4, color: colors.teal },
        fill: 'tozeroy',
        fillcolor: cssVar('--accent-tint-08'),
      },
    ];

    const base = baseLayout();
    const layout = {
      ...base,
      xaxis: { ...base.xaxis, title: 'Iteration' },
      yaxis: { ...base.yaxis, title: 'Efficiency' },
      showlegend: false,
    };

    Plotly.react(container, traces, layout, CONFIG);
  };

  withRerender(container, render);
}

/**
 * Convergence comparison — cumulative best.
 * @param {HTMLElement} container  @param {DataPoint[]} data  @returns {void}
 */
export function createConvergencePlot(container, data) {
  if (!data || data.length === 0) return;

  const render = () => {
    const colors = roleColors();
    const x = data.map((_, i) => i + 1);
    let best = -Infinity;
    const y = data.map((d) => {
      const val = d.objective_value ?? d.objective ?? 0;
      best = Math.max(best, val);
      return best;
    });

    const traces = [
      {
        x,
        y,
        mode: 'lines',
        name: 'Best Objective',
        line: { color: colors.amber, width: 2, shape: 'hv' },
        fill: 'tozeroy',
        fillcolor: cssVar('--amber-tint-08'),
      },
    ];

    const base = baseLayout();
    const layout = {
      ...base,
      xaxis: { ...base.xaxis, title: 'Iteration' },
      yaxis: { ...base.yaxis, title: 'Best Objective' },
      showlegend: false,
    };

    Plotly.react(container, traces, layout, CONFIG);
  };

  withRerender(container, render);
}

/**
 * Parameter space — parallel coordinates.
 * @param {HTMLElement} container  @param {DataPoint[]} data  @returns {void}
 */
export function createParameterSpacePlot(container, data) {
  if (!data || data.length === 0 || !data[0]?.variables) return;

  const render = () => {
    const palette = chartSeries();
    const varNames = Object.keys(data[0].variables || {});
    const objectives = data.map((d) => d.objective_value ?? d.objective ?? 0);

    const dimensions = [
      ...varNames.map((name) => ({
        label: name,
        values: data.map((d) => d.variables?.[name] ?? 0),
      })),
      {
        label: 'Objective',
        values: objectives,
      },
    ];

    // A 3-stop colorscale spanning the categorical palette's blue -> teal
    // -> amber (indices 5, 0, 1) so low/mid/high objective values read as
    // distinct hues in both themes, matching the original design intent.
    const colorscale = [
      [0, palette[5] || palette[0]],
      [0.5, palette[0]],
      [1, palette[1] || palette[0]],
    ];

    const traces = [
      {
        type: 'parcoords',
        line: {
          color: objectives,
          colorscale,
          showscale: true,
        },
        dimensions,
      },
    ];

    const layout = {
      ...baseLayout(),
      margin: { l: 60, r: 60, t: 20, b: 20 },
    };

    Plotly.react(container, traces, layout, CONFIG);
  };

  withRerender(container, render);
}

/**
 * Best point table — rendered as a Plotly table.
 * @param {HTMLElement} container  @param {DataPoint[]} data  @returns {void}
 */
export function createBestPointTable(container, data) {
  if (!data || data.length === 0) return;

  const render = () => {
    // Find best point
    let bestIdx = 0;
    let bestVal = -Infinity;
    data.forEach((d, i) => {
      const val = d.objective_value ?? d.objective ?? 0;
      if (val > bestVal) {
        bestVal = val;
        bestIdx = i;
      }
    });

    const best = data[bestIdx];
    const headers = ['Parameter', 'Value'];
    const params = ['Iteration', 'Objective', ...Object.keys(best.variables || {})];
    const values = [
      bestIdx + 1,
      bestVal.toFixed(6),
      ...Object.values(best.variables || {}).map((v) => (typeof v === 'number' ? v.toFixed(6) : v)),
    ];

    const theme = chartTheme();
    const colors = roleColors();
    const borderColor = cssVar('--border-default');

    const traces = [
      {
        type: 'table',
        header: {
          values: headers,
          fill: { color: theme.paper_bgcolor },
          font: { color: theme.font.color, size: 11 },
          align: 'left',
          line: { color: borderColor, width: 1 },
        },
        cells: {
          values: [params, values],
          fill: { color: theme.plot_bgcolor },
          font: { color: [colors.teal, theme.font.color], size: 11, family: 'JetBrains Mono, monospace' },
          align: 'left',
          line: { color: borderColor, width: 1 },
          height: 28,
        },
      },
    ];

    const layout = {
      ...baseLayout(),
      margin: { l: 5, r: 5, t: 5, b: 5 },
    };

    Plotly.react(container, traces, layout, CONFIG);
  };

  withRerender(container, render);
}
