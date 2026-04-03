/**
 * OSPREY Tuning — Results Viewer (Analysis Tab)
 *
 * Historical run display with 4 Plotly charts.
 */

import { api } from './api.js';
import { state } from './state.js';
import {
  createEfficiencyPlot,
  createConvergencePlot,
  createParameterSpacePlot,
  createBestPointTable,
} from './plots.js';

let contentEl;

export function initResultsViewer() {
  contentEl = document.getElementById('analysis-content');

  // Listen for historical run selection
  state.on('loadHistoricalRun', loadRun);
}

async function loadRun(timestamp) {
  if (!contentEl) return;

  contentEl.innerHTML = '<div class="analysis-empty"><div class="analysis-empty-title">Loading...</div></div>';

  try {
    const result = await api.loadRun(timestamp);
    const data = result.data || result.points || result || [];

    if (!Array.isArray(data) || data.length === 0) {
      contentEl.innerHTML = `
        <div class="analysis-empty">
          <div class="analysis-empty-icon">&#128202;</div>
          <div class="analysis-empty-title">No Data</div>
          <div class="analysis-empty-text">This run contains no data points.</div>
        </div>
      `;
      return;
    }

    renderAnalysis(data, timestamp);
  } catch (err) {
    contentEl.innerHTML = `
      <div class="analysis-empty">
        <div class="analysis-empty-icon">&#9888;</div>
        <div class="analysis-empty-title">Error Loading Run</div>
        <div class="analysis-empty-text">${err.message}</div>
      </div>
    `;
  }
}

function renderAnalysis(data, timestamp) {
  contentEl.innerHTML = `
    <div class="analysis-header" style="margin-bottom: var(--space-4);">
      <span style="font-weight: 600; color: var(--text-primary);">Run: ${timestamp}</span>
      <span style="color: var(--text-muted); margin-left: var(--space-2);">${data.length} points</span>
    </div>
    <div class="analysis-grid">
      <div class="analysis-chart">
        <div class="analysis-chart-title">Efficiency Over Iterations</div>
        <div class="analysis-chart-body" id="chart-efficiency"></div>
      </div>
      <div class="analysis-chart">
        <div class="analysis-chart-title">Convergence</div>
        <div class="analysis-chart-body" id="chart-convergence"></div>
      </div>
      <div class="analysis-chart">
        <div class="analysis-chart-title">Parameter Space</div>
        <div class="analysis-chart-body" id="chart-params"></div>
      </div>
      <div class="analysis-chart">
        <div class="analysis-chart-title">Best Point</div>
        <div class="analysis-chart-body" id="chart-best"></div>
      </div>
    </div>
  `;

  // Render charts after DOM is ready
  requestAnimationFrame(() => {
    createEfficiencyPlot(document.getElementById('chart-efficiency'), data);
    createConvergencePlot(document.getElementById('chart-convergence'), data);
    createParameterSpacePlot(document.getElementById('chart-params'), data);
    createBestPointTable(document.getElementById('chart-best'), data);
  });
}
