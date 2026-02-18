/**
 * OSPREY Tuning — Plotly Chart Builders
 *
 * All plots use OSPREY dark theme: #0a0f1a bg, #8b9ab5 grid, teal traces.
 */

const DARK_LAYOUT = {
  paper_bgcolor: '#0a0f1a',
  plot_bgcolor: '#0a0f1a',
  font: { family: 'Outfit, sans-serif', color: '#8b9ab5', size: 11 },
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: {
    gridcolor: 'rgba(139, 154, 181, 0.1)',
    zerolinecolor: 'rgba(139, 154, 181, 0.15)',
  },
  yaxis: {
    gridcolor: 'rgba(139, 154, 181, 0.1)',
    zerolinecolor: 'rgba(139, 154, 181, 0.15)',
  },
  legend: { bgcolor: 'transparent', font: { size: 10 } },
};

const COLORS = {
  lhs: '#3b82f6',
  initial: '#a855f7',
  teal: '#4fd1c5',
  amber: '#d4a574',
  bo_palette: ['#4fd1c5', '#f59e0b', '#22c55e', '#ef4444', '#a855f7', '#3b82f6'],
};

const CONFIG = {
  responsive: true,
  displayModeBar: false,
};

/**
 * Build the dual-axis optimization plot (objective + variables).
 */
export function createOptimizationPlot(container, optState, displayMode = 'normalized') {
  const traces = [];
  const { lhs_data, bo_data, snapshots } = optState;

  // Combine all data points
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
  const objectiveX = allPoints.map(p => p.idx);
  const objectiveY = allPoints.map(p => p.objective_value ?? p.objective ?? null);
  const phaseColors = allPoints.map(p => {
    if (p.phase === 'initial') return COLORS.initial;
    if (p.phase === 'lhs') return COLORS.lhs;
    return COLORS.teal;
  });

  traces.push({
    x: objectiveX,
    y: objectiveY,
    mode: 'lines+markers',
    name: 'Objective',
    marker: { color: phaseColors, size: 6 },
    line: { color: COLORS.teal, width: 1.5 },
    yaxis: 'y',
  });

  // Variable traces (bottom subplot)
  if (allPoints[0]?.variables) {
    const varNames = Object.keys(allPoints[0].variables);
    varNames.forEach((name, vi) => {
      const color = COLORS.bo_palette[vi % COLORS.bo_palette.length];
      const yVals = allPoints.map(p => {
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

  const layout = {
    ...DARK_LAYOUT,
    margin: { l: 50, r: 50, t: 10, b: 40 },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Iteration', domain: [0, 1] },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Objective', domain: [0.55, 1] },
    yaxis2: {
      ...DARK_LAYOUT.yaxis,
      title: displayMode === 'normalized' ? 'Variables (norm)' : 'Variables',
      domain: [0, 0.45],
      anchor: 'x',
    },
    showlegend: true,
    legend: { ...DARK_LAYOUT.legend, x: 1, xanchor: 'right', y: 1 },
    hovermode: 'x unified',
  };

  Plotly.react(container, traces, layout, CONFIG);
}

/**
 * Efficiency over iterations — line chart.
 */
export function createEfficiencyPlot(container, data) {
  if (!data || data.length === 0) return;

  const x = data.map((_, i) => i + 1);
  const y = data.map(d => d.efficiency ?? d.improvement ?? 0);

  const traces = [{
    x, y,
    mode: 'lines+markers',
    name: 'Efficiency',
    line: { color: COLORS.teal, width: 2 },
    marker: { size: 4, color: COLORS.teal },
    fill: 'tozeroy',
    fillcolor: 'rgba(79, 209, 197, 0.08)',
  }];

  const layout = {
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Iteration' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Efficiency' },
    showlegend: false,
  };

  Plotly.react(container, traces, layout, CONFIG);
}

/**
 * Convergence comparison — cumulative best.
 */
export function createConvergencePlot(container, data) {
  if (!data || data.length === 0) return;

  const x = data.map((_, i) => i + 1);
  let best = -Infinity;
  const y = data.map(d => {
    const val = d.objective_value ?? d.objective ?? 0;
    best = Math.max(best, val);
    return best;
  });

  const traces = [{
    x, y,
    mode: 'lines',
    name: 'Best Objective',
    line: { color: COLORS.amber, width: 2, shape: 'hv' },
    fill: 'tozeroy',
    fillcolor: 'rgba(212, 165, 116, 0.08)',
  }];

  const layout = {
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Iteration' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Best Objective' },
    showlegend: false,
  };

  Plotly.react(container, traces, layout, CONFIG);
}

/**
 * Parameter space — parallel coordinates.
 */
export function createParameterSpacePlot(container, data) {
  if (!data || data.length === 0 || !data[0]?.variables) return;

  const varNames = Object.keys(data[0].variables);
  const objectives = data.map(d => d.objective_value ?? d.objective ?? 0);

  const dimensions = [
    ...varNames.map(name => ({
      label: name,
      values: data.map(d => d.variables?.[name] ?? 0),
    })),
    {
      label: 'Objective',
      values: objectives,
    },
  ];

  const traces = [{
    type: 'parcoords',
    line: {
      color: objectives,
      colorscale: [[0, '#3b82f6'], [0.5, '#4fd1c5'], [1, '#f59e0b']],
      showscale: true,
    },
    dimensions,
  }];

  const layout = {
    ...DARK_LAYOUT,
    margin: { l: 60, r: 60, t: 20, b: 20 },
  };

  Plotly.react(container, traces, layout, CONFIG);
}

/**
 * Best point table — rendered as a Plotly table.
 */
export function createBestPointTable(container, data) {
  if (!data || data.length === 0) return;

  // Find best point
  let bestIdx = 0;
  let bestVal = -Infinity;
  data.forEach((d, i) => {
    const val = d.objective_value ?? d.objective ?? 0;
    if (val > bestVal) { bestVal = val; bestIdx = i; }
  });

  const best = data[bestIdx];
  const headers = ['Parameter', 'Value'];
  const params = ['Iteration', 'Objective', ...Object.keys(best.variables || {})];
  const values = [
    bestIdx + 1,
    bestVal.toFixed(6),
    ...Object.values(best.variables || {}).map(v => typeof v === 'number' ? v.toFixed(6) : v),
  ];

  const traces = [{
    type: 'table',
    header: {
      values: headers,
      fill: { color: '#111a2e' },
      font: { color: '#8b9ab5', size: 11 },
      align: 'left',
      line: { color: 'rgba(100,116,139,0.18)', width: 1 },
    },
    cells: {
      values: [params, values],
      fill: { color: '#0a0f1a' },
      font: { color: ['#4fd1c5', '#e2e8f0'], size: 11, family: 'JetBrains Mono, monospace' },
      align: 'left',
      line: { color: 'rgba(100,116,139,0.1)', width: 1 },
      height: 28,
    },
  }];

  const layout = {
    ...DARK_LAYOUT,
    margin: { l: 5, r: 5, t: 5, b: 5 },
  };

  Plotly.react(container, traces, layout, CONFIG);
}
