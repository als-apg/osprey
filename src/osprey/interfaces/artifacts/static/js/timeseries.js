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

/**
 * Whether one channel is enum/status rather than a numeric signal.
 *
 * Deliberately tri-state: ONLY an explicit `numeric === false` means enum, so a
 * response that omits the key is still numeric. Four sites ask this question --
 * whether the layout needs a `yaxis2`, whether the toolbar marks the toggle,
 * whether hiding a channel hides that axis, and which axis a trace is built on
 * -- and they must not drift, since getting the last one wrong routes a trace
 * onto the wrong axis rather than failing loudly.
 * @param {any} ch
 * @returns {boolean}
 */
function _tsIsNonNumeric(ch) {
  return ch.numeric === false;
}

/**
 * Whether any channel is enum/status, i.e. whether the chart needs the
 * secondary category `yaxis2`.
 *
 * Both the view shell and the chart renderer need this answer and must agree:
 * the shell uses it to autorange and show/hide that axis on Reset Zoom and on
 * channel toggles, and the renderer uses it to decide whether to put the axis
 * in the layout at all. Computed once here so the two cannot drift.
 * @param {any[]} channels
 * @returns {boolean}
 */
function _tsHasNonNumeric(channels) {
  return (channels || []).some(_tsIsNonNumeric);
}

/**
 * RFC4180-ish CSV field quoting: a field containing a comma, double quote, or
 * line break is wrapped in double quotes with internal double quotes doubled.
 * Needed now that a CSV row's fields include free-form channel names and
 * non-numeric (enum/status) values -- e.g. a status string like
 * `"OFF, LOCAL"` -- rather than only numbers, which never need escaping.
 * @param {any} value
 * @returns {string}
 */
function _tsCsvField(value) {
  // `??`, not `||`: a gap (null/undefined) becomes an empty field, but the
  // falsy-but-real values `0` and `""` must survive as themselves.
  const str = String(value ?? "");
  if (/["\n\r,]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Long-format CSV: one row per (channel, timestamp) pair. Each channel now
 * carries its own timestamps independent of every other channel's (see
 * app.py's `format=chart` branch), so there is no shared axis to pivot into
 * a wide `timestamp,ch1,ch2,...` table without re-implementing the
 * server-side union pivot client-side (that pivot already exists, for the
 * table view, via `format=table`). Long format is the honest shape of the
 * data actually on hand here.
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
 * Push `content` at the browser as a download. The two exporters differ only
 * in their content, MIME type and extension, so the Blob/ObjectURL/anchor
 * sequence -- including the revoke, which is easy to drop when copied -- lives
 * here once.
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
    // timestamps/values (see app.py's chart branch) -- there is no shared
    // `columns`/`total_rows` frame any more. `columns` here is just the
    // channel-name projection, kept for the toggle logic and for
    // renderTimeseriesTable, which still consumes the (unchanged) `format=table`
    // shape and expects a plain name list.
    const channels = chartData.channels || [];
    const columns = channels.map((/** @type {any} */ ch) => ch.channel);
    // Whether renderTimeseriesChart will add a secondary category `yaxis2`
    // for any enum/status channel -- needed here too, both so Reset Zoom
    // also autoranges that axis (Plotly.relayout's autorange keys are
    // per-axis; `yaxis.autorange` alone never touches `yaxis2`) and so a
    // channel toggle can show/hide that axis as its last visible occupant
    // is hidden/shown (see the toggle handler below).
    const hasNonNumeric = _tsHasNonNumeric(channels);

    const visible = new Set(columns);

    let html = '<div class="ts-viewer">';

    // Info bar
    html += '<div class="ts-info-bar">';
    channels.forEach((/** @type {any} */ ch) => {
      html += `<span class="ts-badge ts-badge-channel"><span class="badge-label">CH</span> ${escapeHtml(_tsShortChannelName(ch.channel))}</span>`;
    });
    // Cross-channel totals are the server's, unconditionally. Summing per-channel
    // point counts here could only ever be a proxy: the sum counts each channel's
    // own timestamps, whereas the table paginates a *unioned* row axis, so the two
    // disagree whenever channels are sampled at different times -- and `row_count`
    // is not derivable client-side at all. This script and the endpoint that
    // answers it ship in the same wheel, so `summary` is always present; a
    // fallback here would be a second, knowingly-wrong number for a case that
    // cannot arise.
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
      // `aria-pressed` + an explicit `aria-label` mirror the design system's own
      // toggle (osprey-theme-switcher): the button's shown/hidden state must be
      // exposed to assistive tech rather than carried only by the `ts-ch-off`
      // class, and the accessible name must be the full PV plus the axis
      // explanation -- not the truncated display text with the decorative "R"
      // glyph welded on (accname demotes `title` to a description whenever the
      // button has content, so `title` alone often goes unannounced).
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
        // Keep the exposed state in step with the class (the refused-hide path
        // above returns early, so neither is touched there).
        btn.setAttribute("aria-pressed", String(visible.has(col)));
        if (chartEl && /** @type {any} */ (chartEl).data) {
          // `columns` is in the same order as `channels`, which is the same
          // order traces were built in (renderTimeseriesChart), so this
          // boolean array lines up positionally with Plotly's trace array
          // even though visibility itself is tracked by channel name.
          const update = columns.map((/** @type {any} */ c) => visible.has(c));
          // Hiding the last visible enum/status channel would otherwise leave an
          // empty category `yaxis2` sitting on the right with nothing plotted
          // against it; show it again once any non-numeric channel is visible
          // again. Applied as one `update` rather than restyle-then-relayout so
          // a toggle costs a single redraw instead of two, with no intermediate
          // frame showing the traces already hidden but the axis still there.
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
          // `yaxis2` (the enum/status channels' secondary category axis, see
          // renderTimeseriesChart) predates this handler and needs its own
          // autorange key -- otherwise a user who zooms it and clicks Reset
          // Zoom gets the numeric axis reset while the status trace stays
          // clipped or off-screen, reading as "that channel has no data".
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
 * Builds one Plotly trace per channel, each with its own `x`/`y` (Plotly
 * supports per-trace `x` natively -- there is no shared axis to slice
 * multiple columns against any more, see app.py's `format=chart` branch and
 * `extract_channel_series`'s docstring).
 *
 * A non-numeric (enum/status) channel -- `numeric: false` -- can't share a
 * linear y-axis with a numeric signal like beam current, so it is never
 * silently dropped: it gets its own categorical trace on a secondary
 * right-hand y-axis (`yaxis2`, `type: "category"`, `overlaying: "y"`), drawn
 * with an `hv` step line (a status value holds until its next transition,
 * unlike a numeric signal that's meaningfully interpolated between samples).
 * This lets an operator read machine state time-aligned against a numeric
 * signal on the same plot, rather than losing the channel or corrupting the
 * numeric axis. The secondary axis is only added to the layout when at
 * least one channel actually needs it.
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
            // Every enum channel shares one category axis, and Plotly builds
            // that axis's rungs from the union of its traces in first-
            // appearance order. Two status channels with different vocabularies
            // therefore interleave: each one's step line crosses rungs that
            // belong to the other, which reads as a state it was never in.
            // Prefixing each value with its channel keeps every channel's rungs
            // in its own contiguous block. `customdata` carries the unprefixed
            // value so the hover label still shows the real state.
            //
            // A gap (`null` -- archiver_read.py's chart branch emits real
            // `None`s for missing samples, and extract_channel_series passes
            // them through) is deliberately NOT namespaced: prefixing it would
            // register the literal rung "CH: null" on the category axis and
            // step the line up to it, i.e. show a state the channel was never
            // in -- the very failure this namespacing exists to prevent. Left
            // as `null`, Plotly's category converter returns BADNUM instead of
            // adding a category, and the step line breaks, matching how the
            // numeric branch already renders the same gap.
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
    xaxis: { gridcolor: t.xaxis.gridcolor, linecolor: t.line, tickfont: { size: 10 } },
    yaxis: { gridcolor: t.yaxis.gridcolor, linecolor: t.line, tickfont: { size: 10 } },
    ...(hasNonNumeric
      ? {
          yaxis2: {
            overlaying: "y",
            side: "right",
            type: "category",
            linecolor: t.line,
            tickfont: { size: 10 },
            // This axis's tick labels are the namespaced rungs -- "<full PV
            // name>: <VALUE>" -- the longest strings in the figure, drawn on
            // the right against a `margin.r` of 20. Plotly's `automargin`
            // defaults to false for an x-anchored axis, and `margin.autoexpand`
            // only reacts to components that push margin, so without this the
            // labels are drawn past the paper edge and clipped.
            automargin: true,
            // ...and `showgrid` otherwise defaults to true with no `gridcolor`,
            // adding one gridline per rung in Plotly's auto blend rather than
            // the --chart-* token the other two axes use. The numeric `yaxis`
            // grid is the one that carries meaning here.
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

    // Header comes from THIS response and nowhere else. Taking it from the
    // caller's channel list would mean the names came from a separate
    // format=chart request, and for an artifact being written while it is viewed
    // the two can disagree -- putting this page's cells under the other
    // request's PV names.
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
