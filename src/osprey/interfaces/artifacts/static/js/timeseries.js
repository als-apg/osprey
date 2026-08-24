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
 * The lazy `<script>` injection takes its src from index.html's
 * `osprey-vendor-plotly` meta tag, where the server's `vendor_url()` template
 * global resolves it — the CDN URL by default, the offline-vendored copy
 * under OSPREY_OFFLINE=1 (the vendored path is never populated in online
 * images, so hardcoding it here 404s every CDN-mode deployment). The
 * vendored spelling stays as the no-meta fallback.
 *
 * HTML-escaping uses the design-system's canonical `escapeHtml` (quote-safe,
 * nullish-collapsing) — see dom.js. This module used to carry its own
 * `_esc` copy; that divergence has been consolidated away.
 *
 * Table cells apply magnitude-adaptive formatting: index cells go through
 * `_tsShortTime` (short month/day + hour:minute:second, no year, since the
 * backend's `index` is ISO timestamp strings — see
 * src/osprey/utils/timeseries.py's `extract_channel_series`), value cells
 * through `_tsFormatValue` (<=5 significant figures, scientific notation for
 * very large/small magnitudes). Both helpers fall back to raw `String(...)`
 * for inputs that aren't a number/valid date, so their output is still run
 * through `escapeHtml` at the interpolation site — never trust it as safe
 * HTML on its own.
 *
 * @module timeseries
 */

import { chartTheme, chartRelayout, chartSeries } from "/design-system/js/theme-manager.js";
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
    // Resolved server-side by vendor_url() and handed over via index.html's
    // meta tag (see the module docstring). The vendored path is the fallback
    // for any page without the meta, preserving prior offline behavior.
    const meta = document.querySelector('meta[name="osprey-vendor-plotly"]');
    script.src = (meta && meta.getAttribute("content")) || "/static/js/vendor/plotly-3.3.1.min.js";
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

/**
 * Whether one channel is enum/status rather than a numeric signal.
 * Only an explicit `numeric === false` means enum; a response that omits the
 * key is still numeric.
 * @param {any} ch
 * @returns {boolean}
 */
function _tsIsNonNumeric(ch) {
  return ch.numeric === false;
}

/**
 * Whether any channel is enum/status, i.e. whether the chart needs the
 * secondary category `yaxis2`.
 * @param {any[]} channels
 * @returns {boolean}
 */
function _tsHasNonNumeric(channels) {
  return (channels || []).some(_tsIsNonNumeric);
}

/**
 * RFC4180-ish CSV field quoting: a field containing a comma, double quote, or
 * line break is wrapped in double quotes with internal double quotes doubled.
 * @param {any} value
 * @returns {string}
 */
function _tsCsvField(value) {
  // `??`, not `||`: only null/undefined become empty; 0 and "" survive.
  const str = String(value ?? "");
  if (/["\n\r,]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Long-format CSV: one row per (channel, timestamp) pair. Each channel
 * carries its own timestamps, so there is no shared axis to pivot into a
 * wide table client-side.
 * @param {any} chartData
 */
function _tsExportCSV(chartData) {
  const channels = chartData.channels || [];
  const rows = channels.flatMap((/** @type {any} */ ch) =>
    (ch.timestamps || []).map((/** @type {any} */ ts, /** @type {number} */ i) =>
      [ch.channel, ts, (ch.values || [])[i]].map(_tsCsvField).join(",")
    )
  );
  _tsDownloadBlob(["channel,timestamp,value", ...rows].join("\n"), "text/csv", "csv");
}

/** @param {any} chartData */
function _tsExportJSON(chartData) {
  _tsDownloadBlob(JSON.stringify(chartData, null, 2), "application/json", "json");
}

/**
 * Push `content` at the browser as a download.
 * @param {string} content
 * @param {string} mime
 * @param {string} ext
 */
function _tsDownloadBlob(content, mime, ext) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `timeseries_${Date.now()}.${ext}`;
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
    // format=chart returns one entry per channel, each with its own
    // timestamps/values; `columns` is just the channel-name projection.
    const channels = chartData.channels || [];
    const columns = channels.map((/** @type {any} */ ch) => ch.channel);
    // Needed so Reset Zoom also autoranges `yaxis2` and so channel toggles
    // can show/hide that axis.
    const hasNonNumeric = _tsHasNonNumeric(channels);

    const visible = new Set(columns);

    let html = '<div class="ts-viewer">';

    // Info bar
    html += '<div class="ts-info-bar">';
    channels.forEach((/** @type {any} */ ch) => {
      html += `<span class="ts-badge ts-badge-channel"><span class="badge-label">CH</span> ${escapeHtml(_tsShortChannelName(ch.channel))}</span>`;
    });
    // Cross-channel totals come from the server: per-channel point sums
    // disagree with the unioned row axis, and `row_count` is not derivable
    // client-side.
    const summary = chartData.summary;
    html += `<span class="ts-badge ts-badge-points"><span class="badge-label">Points</span> ${summary.total_points.toLocaleString()}</span>`;
    if (typeof summary.row_count === "number") {
      html += `<span class="ts-badge ts-badge-rows"><span class="badge-label">Rows</span> ${summary.row_count.toLocaleString()}</span>`;
    }
    if (summary.downsampled) {
      html += `<span class="ts-badge ts-badge-downsampled"><span class="badge-label">Downsampled</span> ${summary.returned_points.toLocaleString()} pts</span>`;
    }
    html += '</div>';

    // Toolbar
    html += '<div class="ts-toolbar">';
    html += '<div class="ts-toolbar-group">';
    const _tsPalette = chartSeries();
    channels.forEach((/** @type {any} */ ch, /** @type {number} */ ci) => {
      const color = _tsPalette[ci % _tsPalette.length];
      const isNonNumeric = _tsIsNonNumeric(ch);
      const title = isNonNumeric ? `${ch.channel} (status/enum -- plotted on right axis)` : ch.channel;
      // aria-label carries the full PV name plus the axis note; `title` alone
      // is often unannounced once a button has content.
      html += `<button class="ts-ch-toggle" type="button" aria-pressed="true" data-ch-name="${escapeHtml(ch.channel)}" aria-label="${escapeHtml(title)}" title="${escapeHtml(title)}">`;
      html += `<span class="ts-ch-dot" style="background:${color}"></span>`;
      html += escapeHtml(_tsShortChannelName(ch.channel));
      if (isNonNumeric) html += ' <span class="ts-ch-axis-tag" aria-hidden="true">R</span>';
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
        const col = /** @type {string} */ (/** @type {HTMLElement} */ (btn).dataset.chName);
        if (visible.has(col)) {
          if (visible.size <= 1) return;
          visible.delete(col);
          btn.classList.add("ts-ch-off");
        } else {
          visible.add(col);
          btn.classList.remove("ts-ch-off");
        }
        // Keep the exposed state in step with the class.
        btn.setAttribute("aria-pressed", String(visible.has(col)));
        if (chartEl && /** @type {any} */ (chartEl).data) {
          // `columns` matches the trace build order, so this boolean array
          // lines up positionally with Plotly's trace array.
          const update = columns.map((/** @type {any} */ c) => visible.has(c));
          // Hide `yaxis2` while no enum/status channel is visible; one
          // Plotly.update keeps the toggle a single redraw.
          const layoutUpdate = hasNonNumeric
            ? {
                "yaxis2.visible": channels.some(
                  (/** @type {any} */ ch) => _tsIsNonNumeric(ch) && visible.has(ch.channel)
                ),
              }
            : {};
          Plotly.update(chartEl, { visible: update }, layoutUpdate);
        }
      });
    });

    container.querySelectorAll(".ts-action-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const action = /** @type {HTMLElement} */ (btn).dataset.action;
        if (action === "zoom-reset" && chartEl) {
          // yaxis2 needs its own autorange key; `yaxis.autorange` alone never
          // touches it.
          const resetLayout = {
            "xaxis.autorange": true,
            "yaxis.autorange": true,
            ...(hasNonNumeric ? { "yaxis2.autorange": true } : {}),
          };
          Plotly.relayout(chartEl, resetLayout);
        } else if (action === "export-csv") {
          _tsExportCSV(chartData);
        } else if (action === "export-json") {
          _tsExportJSON(chartData);
        }
      });
    });

    await renderTimeseriesChart(/** @type {any} */ (chartEl), chartData);

    const tableEl = container.querySelector("[data-ts-table]");
    await renderTimeseriesTable(/** @type {any} */ (tableEl), artifact.id, 0);
  } catch (err) {
    console.error("Timeseries render failed:", err);
    container.innerHTML = '<span class="text-muted">Failed to load timeseries data</span>';
  }
}

/**
 * A Plotly layout fragment for the timeseries chart, built from
 * chartTheme()'s --chart-* computed-style bridge plus the axis/legend line
 * color and legend background the initial layout also needs -- the same
 * --chart-axis-line token chartRelayout() applies on a live re-theme, so
 * the first render and every later one agree.
 * @returns {any}
 */
export function _tsChartTheme() {
  const base = chartTheme();
  const line = getComputedStyle(document.documentElement).getPropertyValue("--chart-axis-line").trim();
  return { ...base, line: line || base.xaxis.gridcolor, legendBg: base.paper_bgcolor, legendBorder: line };
}

/**
 * Re-theme every mounted timeseries chart in place after a theme change.
 * Lives here so the theme→Plotly layout-key mapping has one owner (this
 * module's {@link renderTimeseriesChart} builds the same keys at initial
 * render); gallery.js's theme subscription just calls this.
 *
 * Plural, and document-wide, because the gallery's two artifact surfaces —
 * Expert's preview pane and Simple's result card — can each hold a mounted
 * chart at the same time (`renderPreview` still runs while Simple is on
 * screen). Restyling only "the visible one" would mean deciding which that is
 * from a `display:none` ancestor; restyling all of them is both cheaper and
 * correct, since there are at most two and a hidden chart re-themed early is
 * simply already right when its pane comes back.
 * @returns {void}
 */
export function restyleMountedCharts() {
  if (typeof Plotly === "undefined") return;
  // Target the actual Plotly graph divs, not the outer viewport wrappers.
  const charts = document.querySelectorAll(".ts-viewport-container [data-ts-chart]");
  if (!charts.length) return;
  const series = chartSeries();
  charts.forEach((tsChart) => {
    try {
      // The shared token -> layout-key mapping (design-system owned); it
      // covers yaxis2 too when a categorical channel added one.
      Plotly.relayout(tsChart, chartRelayout(tsChart));
      // relayout doesn't touch trace colors, so the data lines and their legend
      // dots keep the prior theme's palette until reload. Restyle each trace's
      // line+marker to the current series palette so they re-theme live too.
      const traces = /** @type {any} */ (tsChart).data || [];
      if (series.length && traces.length) {
        const colors = traces.map((/** @type {any} */ _t, /** @type {number} */ i) => series[i % series.length]);
        Plotly.restyle(tsChart, { "line.color": colors, "marker.color": colors });
      }
    // eslint-disable-next-line no-empty -- intentional empty catch: Plotly relayout is best-effort restyle
    } catch {}
  });
}

/**
 * Builds one Plotly trace per channel, each with its own `x`/`y`.
 *
 * A non-numeric (enum/status) channel gets a categorical trace on a secondary
 * right-hand y-axis (`yaxis2`), drawn with an `hv` step line (a status value
 * holds until its next transition). The secondary axis is only added to the
 * layout when at least one channel needs it.
 * @param {any} el
 * @param {any} chartData
 * @returns {Promise<void>}
 */
export async function renderTimeseriesChart(el, chartData) {
  await ensurePlotlyLoaded();
  if (!el) return;

  const channels = chartData.channels || [];
  const hasNonNumeric = _tsHasNonNumeric(channels);

  const traces = channels.map((/** @type {any} */ ch) => {
    const numeric = !_tsIsNonNumeric(ch);
    return {
      x: ch.timestamps,
      y: ch.values,
      name: ch.channel,
      // scattergl (WebGL) is faster for the common numeric case; a
      // categorical y-axis is safer on the plain SVG "scatter" renderer.
      type: numeric ? "scattergl" : "scatter",
      mode: numeric ? "lines" : "lines+markers",
      ...(numeric
        ? {}
        : {
            yaxis: "y2",
            line: { shape: "hv" },
            // All enum channels share one category axis, so each value is
            // prefixed with its channel to keep its rungs in a contiguous
            // block; `customdata` carries the unprefixed value for the hover
            // label. A `null` gap is NOT prefixed -- "CH: null" would become a
            // real rung, while a bare null breaks the step line like a
            // numeric-branch gap.
            y: (ch.values || []).map((/** @type {any} */ v) => (v == null ? null : `${ch.channel}: ${v}`)),
            customdata: ch.values,
          }),
      hovertemplate: numeric
        ? "%{y:.4g}<extra>%{fullData.name}</extra>"
        : "%{customdata}<extra>%{fullData.name}</extra>",
    };
  });

  const t = _tsChartTheme();

  const layout = {
    paper_bgcolor: t.paper_bgcolor,
    plot_bgcolor: t.plot_bgcolor,
    font: { family: "'JetBrains Mono', monospace", size: 11, color: t.font.color },
    margin: { t: 30, r: 20, b: 50, l: 60 },
    hovermode: "x unified",
    // zerolinecolor too: chartRelayout() sets it on a live re-theme, so the
    // first render must as well or the zero line changes color on the
    // first theme flip.
    xaxis: { gridcolor: t.xaxis.gridcolor, linecolor: t.line, zerolinecolor: t.line, tickfont: { size: 10 } },
    yaxis: { gridcolor: t.yaxis.gridcolor, linecolor: t.line, zerolinecolor: t.line, tickfont: { size: 10 } },
    ...(hasNonNumeric
      ? {
          yaxis2: {
            overlaying: "y",
            side: "right",
            type: "category",
            linecolor: t.line,
            tickfont: { size: 10 },
            // The namespaced rung labels are long and drawn on the right;
            // without automargin they are clipped at the paper edge.
            automargin: true,
            // Only the numeric yaxis grid carries meaning.
            showgrid: false,
          },
        }
      : {}),
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
 * @param {number} offset
 * @returns {Promise<void>}
 */
export async function renderTimeseriesTable(el, artifactId, offset) {
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

    // Header must come from THIS response: names from a separate format=chart
    // request can disagree for an artifact being written while it is viewed.
    let html = '<div class="ts-data-table-wrapper"><table class="ts-data-table">';
    html += '<thead><tr><th>Index</th>';
    tableData.columns.forEach((/** @type {string} */ c) => { html += `<th>${escapeHtml(c)}</th>`; });
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
      renderTimeseriesTable(el, artifactId, Math.max(0, offset - TS_TABLE_PAGE_SIZE));
    });
    el.querySelector("[data-ts-next]")?.addEventListener("click", () => {
      renderTimeseriesTable(el, artifactId, offset + TS_TABLE_PAGE_SIZE);
    });
  } catch (err) {
    console.error("Table render failed:", err);
    el.innerHTML = '<span class="text-muted">Failed to load table data</span>';
  }
}
