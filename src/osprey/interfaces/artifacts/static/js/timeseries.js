// @ts-check
/**
 * OSPREY Artifact Gallery — timeseries preview: the lazy Plotly loader,
 * the toolbar/chart/table viewer, and CSV/JSON export.
 *
 * `renderTimeseriesView(container, artifact)` is the module's single entry
 * point, called by preview.js's `renderPreview` via the `onTimeseriesNeeded`
 * callback, which gallery.js wires to the export here.
 *
 * No factory/closure state is needed: `_plotlyLoaded`/`_plotlyLoading` are a
 * true page-lifetime singleton cache (matches types.js's `typeRegistry`
 * pattern — there is only ever one gallery per page, and Plotly only ever
 * needs to load once), and the channel-visibility `Set` in
 * `renderTimeseriesView` is local to a single render call, not shared state
 * across calls. So this module is plain exports, no `createXRenderer(...)`.
 *
 * The lazy `<script src="/static/js/vendor/plotly-3.3.1.min.js">` injection
 * must keep that exact offline-vendored path — do not change it.
 *
 * HTML-escaping uses the design-system's canonical `escapeHtml` (quote-safe,
 * nullish-collapsing) — see dom.js. This module used to carry its own
 * `_esc` copy; that divergence has been consolidated away.
 *
 * Table cells apply magnitude-adaptive formatting: index cells go through
 * `_tsShortTime` (short month/day + hour:minute:second, no year, since the
 * backend's split-orient `index` is ISO timestamp strings — see
 * src/osprey/utils/timeseries.py's `extract_timeseries_frame`), value cells
 * through `_tsFormatValue` (<=5 significant figures, scientific notation for
 * very large/small magnitudes). Both helpers fall back to raw `String(...)`
 * for inputs that aren't a number/valid date, so their output is still run
 * through `escapeHtml` at the interpolation site — never trust it as safe
 * HTML on its own.
 *
 * @module timeseries
 */

import { chartTheme, chartSeries } from "/design-system/js/theme-manager.js";
import { escapeHtml } from "/design-system/js/dom.js";
import { isoToDate } from "./types.js";

// ---- Lazy Plotly Loader ---- //

let _plotlyLoaded = false;
/** @type {Promise<void>|null} */
let _plotlyLoading = null;

/** @returns {Promise<void>} */
function ensurePlotlyLoaded() {
  if (_plotlyLoaded) return Promise.resolve();
  if (_plotlyLoading) return _plotlyLoading;
  _plotlyLoading = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/static/js/vendor/plotly-3.3.1.min.js";
    script.onload = () => { _plotlyLoaded = true; resolve(); };
    script.onerror = () => {
      _plotlyLoading = null;
      reject(new Error("Failed to load Plotly"));
    };
    document.head.appendChild(script);
  });
  return _plotlyLoading;
}

// ---- Helpers ---- //

/**
 * @param {string} name
 * @returns {string}
 */
function _tsShortChannelName(name) {
  if (!name) return "";
  if (name.length <= 24) return name;
  const parts = name.split(":");
  if (parts.length >= 3) return parts[0] + ":...:" + parts[parts.length - 1];
  return name.slice(0, 10) + "..." + name.slice(-10);
}

/**
 * Magnitude-adaptive value-cell formatter: <=5 significant figures for
 * ordinary magnitudes, scientific notation once a value is very large or
 * very small (but nonzero). Falls back to `String(...)` for non-number
 * (including NaN) input — callers MUST still escapeHtml the result, since
 * that fallback echoes the raw input verbatim.
 * @param {any} num
 * @returns {string}
 */
function _tsFormatValue(num) {
  if (num === null || num === undefined) return "--";
  if (typeof num !== "number" || Number.isNaN(num)) return String(num);
  if (num === 0) return "0";
  const abs = Math.abs(num);
  if (abs >= 1e6 || abs < 0.001) return num.toExponential(3);
  return num.toPrecision(5);
}

/**
 * Short index/time-cell formatter for ISO timestamp strings: month/day +
 * hour:minute:second, no year. Shares types.js's `isoToDate` guard, which
 * rejects the null/number/numeric-string inputs that bare `Date` coercion
 * would otherwise turn into fabricated epoch/year-2000 timestamps: nullish
 * input renders "--", and any other non-ISO/invalid value falls back to
 * `String(iso)` verbatim. Callers MUST still escapeHtml the result, since
 * that fallback echoes the raw input.
 * @param {any} iso
 * @returns {string}
 */
function _tsShortTime(iso) {
  if (iso === null || iso === undefined) return "--";
  const d = isoToDate(iso);
  if (!d) return String(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** @param {any} chartData */
function _tsExportCSV(chartData) {
  const cols = chartData.columns || [];
  const rows = [["timestamp", ...cols].join(",")];
  chartData.index.forEach((/** @type {any} */ ts, /** @type {number} */ i) => {
    const vals = chartData.data[i] || [];
    rows.push([ts, ...vals].join(","));
  });
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `timeseries_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/** @param {any} chartData */
function _tsExportJSON(chartData) {
  const blob = new Blob([JSON.stringify(chartData, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `timeseries_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---- Timeseries View (toolbar + chart + table) ---- //

/**
 * @param {HTMLElement} container
 * @param {any} artifact
 * @returns {Promise<void>}
 */
export async function renderTimeseriesView(container, artifact) {
  container.innerHTML = '<div class="ts-loading">Loading timeseries data...</div>';
  try {
    const chartResp = await fetch(
      `/api/artifacts/${artifact.id}/data?format=chart&max_points=2000`
    );
    if (!chartResp.ok) throw new Error(`Chart fetch failed: ${chartResp.status}`);
    const chartData = await chartResp.json();
    const columns = chartData.columns || [];

    const visible = new Set(columns);

    let html = '<div class="ts-viewer">';

    // Info bar
    html += '<div class="ts-info-bar">';
    columns.forEach((/** @type {any} */ c) => {
      html += `<span class="ts-badge ts-badge-channel"><span class="badge-label">CH</span> ${escapeHtml(_tsShortChannelName(c))}</span>`;
    });
    html += `<span class="ts-badge ts-badge-rows"><span class="badge-label">Rows</span> ${chartData.total_rows.toLocaleString()}</span>`;
    if (chartData.downsampled) {
      html += `<span class="ts-badge ts-badge-downsampled"><span class="badge-label">Downsampled</span> ${chartData.returned_points.toLocaleString()} pts</span>`;
    }
    html += '</div>';

    // Toolbar
    html += '<div class="ts-toolbar">';
    html += '<div class="ts-toolbar-group">';
    const _tsPalette = chartSeries();
    columns.forEach((/** @type {any} */ col, /** @type {number} */ ci) => {
      const color = _tsPalette[ci % _tsPalette.length];
      html += `<button class="ts-ch-toggle" data-ch-index="${ci}" data-ch-name="${escapeHtml(col)}" title="${escapeHtml(col)}">`;
      html += `<span class="ts-ch-dot" style="background:${color}"></span>`;
      html += escapeHtml(_tsShortChannelName(col));
      html += '</button>';
    });
    html += '</div>';
    html += '<span class="ts-toolbar-divider"></span>';
    html += '<div class="ts-toolbar-group">';
    html += '<button class="ts-action-btn" data-action="zoom-reset" title="Reset zoom">Reset Zoom</button>';
    html += '<button class="ts-action-btn ts-export-btn" data-action="export-csv" title="Export CSV">CSV</button>';
    html += '<button class="ts-action-btn ts-export-btn" data-action="export-json" title="Export JSON">JSON</button>';
    html += '</div>';
    html += '</div>';

    // Chart
    html += '<div class="ts-chart-container" data-ts-chart></div>';

    // Table
    html += '<div data-ts-table></div>';
    html += '</div>';
    container.innerHTML = html;

    // Wire events
    const chartEl = container.querySelector("[data-ts-chart]");

    container.querySelectorAll(".ts-ch-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const ci = parseInt(/** @type {string} */ (/** @type {HTMLElement} */ (btn).dataset.chIndex), 10);
        const col = columns[ci];
        if (visible.has(col)) {
          if (visible.size <= 1) return;
          visible.delete(col);
          btn.classList.add("ts-ch-off");
        } else {
          visible.add(col);
          btn.classList.remove("ts-ch-off");
        }
        if (chartEl && /** @type {any} */ (chartEl).data) {
          const update = columns.map((/** @type {any} */ c) => visible.has(c));
          Plotly.restyle(chartEl, { visible: update });
        }
      });
    });

    container.querySelectorAll(".ts-action-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const action = /** @type {HTMLElement} */ (btn).dataset.action;
        if (action === "zoom-reset" && chartEl) {
          Plotly.relayout(chartEl, { "xaxis.autorange": true, "yaxis.autorange": true });
        } else if (action === "export-csv") {
          _tsExportCSV(chartData);
        } else if (action === "export-json") {
          _tsExportJSON(chartData);
        }
      });
    });

    await renderTimeseriesChart(/** @type {any} */ (chartEl), chartData);

    const tableEl = container.querySelector("[data-ts-table]");
    await renderTimeseriesTable(/** @type {any} */ (tableEl), artifact.id, columns, 0);
  } catch (err) {
    console.error("Timeseries render failed:", err);
    container.innerHTML = '<span class="text-muted">Failed to load timeseries data</span>';
  }
}

/**
 * A Plotly layout fragment for the timeseries chart, built from
 * chartTheme()'s --chart-* computed-style bridge plus a couple of
 * gallery-specific extras (axis/legend line color, legend background)
 * that bridge doesn't cover -- read directly via getComputedStyle so
 * they still track the design tokens rather than being hardcoded.
 * @returns {any}
 */
export function _tsChartTheme() {
  const base = chartTheme();
  const line = getComputedStyle(document.documentElement).getPropertyValue("--border-default").trim();
  return { ...base, line: line || base.xaxis.gridcolor, legendBg: base.paper_bgcolor, legendBorder: line };
}

/**
 * Re-theme the currently mounted timeseries chart (if any) in place after a
 * theme change. Lives here so the theme→Plotly layout-key mapping has one
 * owner (this module's {@link renderTimeseriesChart} builds the same keys at
 * initial render); gallery.js's theme subscription just calls this.
 * @returns {void}
 */
export function restyleVisibleChart() {
  // Target the actual Plotly graph div inside the container, not the
  // outer #ts-viewport wrapper.
  const tsChart = document.querySelector("#ts-viewport [data-ts-chart]");
  if (!tsChart || typeof Plotly === "undefined") return;
  const t = _tsChartTheme();
  try {
    Plotly.relayout(tsChart, {
      paper_bgcolor: t.paper_bgcolor, plot_bgcolor: t.plot_bgcolor,
      "font.color": t.font.color,
      "xaxis.gridcolor": t.xaxis.gridcolor, "xaxis.linecolor": t.line,
      "yaxis.gridcolor": t.yaxis.gridcolor, "yaxis.linecolor": t.line,
      "legend.bgcolor": t.legendBg, "legend.bordercolor": t.legendBorder,
    });
    // relayout doesn't touch trace colors, so the data lines and their legend
    // dots keep the prior theme's palette until reload. Restyle each trace's
    // line+marker to the current series palette so they re-theme live too.
    const series = chartSeries();
    const traces = /** @type {any} */ (tsChart).data || [];
    if (series.length && traces.length) {
      const colors = traces.map((/** @type {any} */ _t, /** @type {number} */ i) => series[i % series.length]);
      Plotly.restyle(tsChart, { "line.color": colors, "marker.color": colors });
    }
  // eslint-disable-next-line no-empty -- intentional empty catch: Plotly relayout is best-effort restyle
  } catch {}
}

/**
 * @param {any} el
 * @param {any} chartData
 * @returns {Promise<void>}
 */
export async function renderTimeseriesChart(el, chartData) {
  await ensurePlotlyLoaded();
  if (!el) return;

  const traces = chartData.columns.map((/** @type {any} */ col, /** @type {number} */ ci) => ({
    x: chartData.index,
    y: chartData.data.map((/** @type {any} */ row) => row[ci]),
    name: col,
    type: "scattergl",
    mode: "lines",
    hovertemplate: "%{y:.4g}<extra>%{fullData.name}</extra>",
  }));

  const t = _tsChartTheme();

  const layout = {
    paper_bgcolor: t.paper_bgcolor,
    plot_bgcolor: t.plot_bgcolor,
    font: { family: "'JetBrains Mono', monospace", size: 11, color: t.font.color },
    margin: { t: 30, r: 20, b: 50, l: 60 },
    hovermode: "x unified",
    xaxis: { gridcolor: t.xaxis.gridcolor, linecolor: t.line, tickfont: { size: 10 } },
    yaxis: { gridcolor: t.yaxis.gridcolor, linecolor: t.line, tickfont: { size: 10 } },
    legend: { bgcolor: t.legendBg, bordercolor: t.legendBorder, borderwidth: 1, font: { size: 10 } },
    colorway: chartSeries(),
  };

  Plotly.newPlot(el, traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  });
}

const TS_TABLE_PAGE_SIZE = 50;

/**
 * @param {any} el
 * @param {string} artifactId
 * @param {string[]} columns
 * @param {number} offset
 * @returns {Promise<void>}
 */
export async function renderTimeseriesTable(el, artifactId, columns, offset) {
  if (!el) return;
  el.innerHTML = '<div class="ts-loading">Loading table...</div>';

  try {
    const resp = await fetch(
      `/api/artifacts/${artifactId}/data?format=table&offset=${offset}&limit=${TS_TABLE_PAGE_SIZE}`
    );
    if (!resp.ok) throw new Error(`Table fetch failed: ${resp.status}`);
    const tableData = await resp.json();

    const totalPages = Math.ceil(tableData.total_rows / TS_TABLE_PAGE_SIZE);
    const currentPage = Math.floor(offset / TS_TABLE_PAGE_SIZE) + 1;

    let html = '<div class="ts-data-table-wrapper"><table class="ts-data-table">';
    html += '<thead><tr><th>Index</th>';
    columns.forEach((c) => { html += `<th>${escapeHtml(c)}</th>`; });
    html += '</tr></thead><tbody>';

    tableData.index.forEach((/** @type {any} */ idx, /** @type {number} */ i) => {
      html += '<tr>';
      html += `<td class="ts-index-cell">${escapeHtml(_tsShortTime(idx))}</td>`;
      const row = tableData.data[i] || [];
      row.forEach((/** @type {any} */ val) => { html += `<td>${escapeHtml(_tsFormatValue(val))}</td>`; });
      html += '</tr>';
    });

    html += '</tbody></table></div>';

    html += '<div class="ts-pagination">';
    html += `<button class="btn btn-secondary btn-sm" data-ts-prev ${offset === 0 ? "disabled" : ""}>Prev</button>`;
    html += `<span class="ts-page-info">Page ${currentPage} of ${totalPages}</span>`;
    html += `<button class="btn btn-secondary btn-sm" data-ts-next ${offset + TS_TABLE_PAGE_SIZE >= tableData.total_rows ? "disabled" : ""}>Next</button>`;
    html += '</div>';

    el.innerHTML = html;

    el.querySelector("[data-ts-prev]")?.addEventListener("click", () => {
      renderTimeseriesTable(el, artifactId, columns, Math.max(0, offset - TS_TABLE_PAGE_SIZE));
    });
    el.querySelector("[data-ts-next]")?.addEventListener("click", () => {
      renderTimeseriesTable(el, artifactId, columns, offset + TS_TABLE_PAGE_SIZE);
    });
  } catch (err) {
    console.error("Table render failed:", err);
    el.innerHTML = '<span class="text-muted">Failed to load table data</span>';
  }
}
