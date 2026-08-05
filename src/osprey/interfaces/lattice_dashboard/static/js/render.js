// @ts-check
/* OSPREY Lattice Dashboard — Rendering Layer
 *
 * DOM rendering for dashboard state (summary stats, magnet sliders), the
 * figure-status primitives (LED/spinner/error), and Plotly figure rendering
 * including the live re-theme subscription.
 *
 * Network effects are not implemented here — they are injected via
 * callbacks so this module has no dependency on the REST/SSE plumbing that
 * lives in net.js. Uses the createElement()/textContent DOM style
 * throughout — no innerHTML with interpolated data.
 */

import { subscribe, chartTheme } from '/design-system/js/theme-manager.js';

// Theme-independent Plotly layout — colors come from chartTheme() at
// render/re-theme time (see renderPlotly() and the theme subscription
// below).
const FIGURE_MARGIN = { l: 45, r: 15, t: 30, b: 35 };
const FIGURE_FONT_FAMILY = 'JetBrains Mono, monospace';
const FIGURE_FONT_SIZE = 10;

const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  displaylogo: false,
};

/**
 * @param {string} chipId
 * @param {number} value
 * @param {(value: number) => string} formatter
 */
function setStatValue(chipId, value, formatter) {
  const chip = document.getElementById(chipId);
  if (!chip) return;
  const valueEl = chip.querySelector('.stat-value');
  if (!valueEl) return;
  valueEl.textContent = (value != null && !isNaN(value)) ? formatter(value) : '—';
}

/** @param {any} summary */
export function updateSummaryStats(summary) {
  setStatValue('stat-energy', summary.energy_gev, v => v.toFixed(2));
  setStatValue('stat-circumference', summary.circumference_m, v => v.toFixed(1));

  if (summary.tunes) {
    setStatValue('stat-nux', summary.tunes[0], v => v.toFixed(4));
    setStatValue('stat-nuy', summary.tunes[1], v => v.toFixed(4));
  }
  if (summary.chromaticity) {
    setStatValue('stat-chrom-x', summary.chromaticity[0], v => v.toFixed(2));
    setStatValue('stat-chrom-y', summary.chromaticity[1], v => v.toFixed(2));
  }
}

// ── LED Status ──────────────────────────────────────────

/**
 * @param {string} name
 * @param {string} status
 */
export function updateLED(name, status) {
  const led = document.getElementById(`led-${name}`);
  if (led) {
    led.dataset.status = status;
  }
}

// ── Spinner ─────────────────────────────────────────────

/** @param {string} name */
export function showSpinner(name) {
  const plotEl = document.getElementById(`plot-${name}`);
  if (!plotEl) return;
  // Remove existing spinner
  if (plotEl.querySelector('.figure-spinner')) return;
  const spinner = document.createElement('div');
  spinner.className = 'figure-spinner';
  const ring = document.createElement('div');
  ring.className = 'spinner-ring';
  spinner.appendChild(ring);
  plotEl.appendChild(spinner);
}

/** @param {string} name */
export function hideSpinner(name) {
  const plotEl = document.getElementById(`plot-${name}`);
  if (!plotEl) return;
  const spinner = plotEl.querySelector('.figure-spinner');
  if (spinner) spinner.remove();
}

// ── Figure Error ────────────────────────────────────────

/**
 * @param {string} name
 * @param {string} error
 */
export function showFigureError(name, error) {
  const plotEl = document.getElementById(`plot-${name}`);
  if (!plotEl) return;
  // Remove placeholder/spinner
  const placeholder = plotEl.querySelector('.figure-placeholder');
  if (placeholder) placeholder.remove();
  hideSpinner(name);

  // Show error unless already showing
  if (plotEl.querySelector('.figure-error')) return;

  const errorEl = document.createElement('div');
  errorEl.className = 'figure-error';

  const iconDiv = document.createElement('div');
  iconDiv.className = 'figure-error-icon';
  iconDiv.textContent = '×';
  errorEl.appendChild(iconDiv);

  const msgDiv = document.createElement('div');
  msgDiv.textContent = 'Computation failed';
  errorEl.appendChild(msgDiv);

  const detailDiv = document.createElement('div');
  detailDiv.className = 'figure-error-msg';
  detailDiv.textContent = error.slice(0, 200);
  errorEl.appendChild(detailDiv);

  plotEl.appendChild(errorEl);
}

/** @param {Record<string, {status: string, error?: string}>} figures */
export function updateFigureStatuses(figures) {
  for (const [name, info] of Object.entries(figures)) {
    updateLED(name, info.status);
    if (info.status === 'computing') {
      showSpinner(name);
    } else {
      hideSpinner(name);
    }
    if (info.status === 'error' && info.error) {
      showFigureError(name, info.error);
    }
  }
}

// ── Plotly Rendering ────────────────────────────────────

/**
 * @param {string} name
 * @param {any} figData
 */
export function renderPlotly(name, figData) {
  const plotEl = document.getElementById(`plot-${name}`);
  if (!plotEl) return;

  // Clear placeholder/error/spinner
  const placeholder = plotEl.querySelector('.figure-placeholder');
  if (placeholder) placeholder.remove();
  const errorEl = plotEl.querySelector('.figure-error');
  if (errorEl) errorEl.remove();
  hideSpinner(name);

  // Apply theme-aware layout overrides
  const themeLayout = chartTheme();
  const layout = {
    ...figData.layout,
    paper_bgcolor: themeLayout.paper_bgcolor,
    plot_bgcolor: themeLayout.plot_bgcolor,
    font: { family: FIGURE_FONT_FAMILY, size: FIGURE_FONT_SIZE, color: themeLayout.font.color },
    margin: FIGURE_MARGIN,
  };
  const defaultFontColor = themeLayout.font.color;

  // Merge axis colors into ALL axes (not just secondary ones)
  if (figData.layout) {
    for (const key of Object.keys(figData.layout)) {
      if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
        layout[key] = {
          ...figData.layout[key],
          gridcolor: themeLayout.xaxis.gridcolor,
          zerolinecolor: themeLayout.xaxis.gridcolor,
        };
      }
    }
    // Pass annotations through as authored. An annotation WITHOUT its own
    // font.color inherits layout.font.color (set above, and re-driven by the
    // theme relayout on switch) rather than being frozen at the first render's
    // color — which previously left the "Area = …" overlay light-gray on the
    // now-white plot after switching to light.
    if (figData.layout.annotations) {
      layout.annotations = figData.layout.annotations;
    }
    if (figData.layout.title) {
      layout.title = figData.layout.title;
    }
  }

  // Make figure fill the cell
  layout.autosize = true;
  delete layout.width;
  delete layout.height;

  Plotly.react(plotEl, applyThemedMarkers(figData.data, defaultFontColor), layout, PLOTLY_CONFIG);
}

/**
 * Paint the marker of every trace tagged ``meta === 'themed-fg-marker'`` with
 * the current foreground color, so a worker can emit a marker (e.g. the
 * resonance working-point star) that tracks the theme instead of baking a
 * fixed color that goes invisible on one background.
 *
 * @param {any[] | undefined} data
 * @param {string} fgColor
 * @returns {any[]}
 */
function applyThemedMarkers(data, fgColor) {
  return (data || []).map((/** @type {any} */ trace) =>
    trace && trace.meta === 'themed-fg-marker'
      ? { ...trace, marker: { ...(trace.marker || {}), color: fgColor } }
      : trace
  );
}

/**
 * @typedef {object} RenderCallbacks
 * @property {(family: string, value: number) => void} onSliderChange - fired 300ms after a slider settles (debounced), to push the new value to the backend
 * @property {(name: string) => void} onFigureReady - fired for each figure already marked 'ready' while rendering state, to fetch and render its data
 * @property {() => (Record<string, number> | undefined)} getOverrides - returns the current per-family override map, if any, so sliders prefer a locally-set override over the family's baseline value
 */

/**
 * @param {string} id
 * @param {boolean} disabled
 */
function setButtonDisabled(id, disabled) {
  const btn = /** @type {HTMLButtonElement|null} */ (document.getElementById(id));
  if (btn) btn.disabled = disabled;
}

/**
 * Create a renderer bound to the dashboard's figure catalog and a set of
 * network-effect callbacks. Also wires the live re-theme subscription
 * (fires on every theme apply, including the hub's initial broadcast and
 * any later toggle, so already-rendered figures pick up the other theme's
 * colors without a page reload).
 * @param {string[]} figureNames - the full figure catalog (fast + verification)
 * @param {RenderCallbacks} callbacks
 */
export function createRenderer(figureNames, callbacks) {
  /** @type {Record<string, ReturnType<typeof setTimeout>>} */
  const sliderDebounceTimers = {};

  /** @param {any} state */
  function renderState(state) {
    if (!state) return;

    // Lattice name
    const latticeName = state.base_lattice || 'No lattice loaded';
    const latticeShort = latticeName.split('/').pop() || latticeName;
    const latticeNameEl = document.getElementById('lattice-name');
    if (latticeNameEl) {
      latticeNameEl.textContent = latticeShort;
    }

    // Simple-mode status line (hidden in Expert via CSS): a plain-language
    // stand-in for the dense stat chips. Updated on every state render so it
    // is correct the moment the mode is flipped to simple.
    const simpleStatusEl = document.getElementById('lattice-simple-status');
    if (simpleStatusEl) {
      simpleStatusEl.textContent = state.base_lattice
        ? `Showing lattice optics for ${latticeShort}`
        : 'No lattice loaded yet';
    }

    // Summary stats
    updateSummaryStats(state.summary || {});

    // Magnet sliders
    updateSliders(state.families || {});

    // Figure statuses and load ready figures
    if (state.figures) {
      updateFigureStatuses(state.figures);
      for (const name of figureNames) {
        if (state.figures[name]?.status === 'ready') {
          callbacks.onFigureReady(name);
        }
      }
    }

    // Enable/disable buttons
    const hasLattice = !!state.base_lattice;
    setButtonDisabled('btn-refresh', !hasLattice);
    setButtonDisabled('btn-verify', !hasLattice);
    setButtonDisabled('btn-baseline', !hasLattice);
  }

  /** @param {Record<string, any>} families */
  function updateSliders(families) {
    const container = document.getElementById('slider-container');
    if (!container) return;
    if (!families || Object.keys(families).length === 0) {
      container.textContent = '';
      const emptyEl = document.createElement('div');
      emptyEl.className = 'sidebar-empty';
      emptyEl.textContent = 'Load a lattice to see controls';
      container.appendChild(emptyEl);
      return;
    }

    // Group by type
    /** @type {Record<string, any[]>} */
    const groups = {};
    for (const [name, info] of Object.entries(families)) {
      const type = info.type || 'other';
      if (!groups[type]) groups[type] = [];
      groups[type].push({ name, ...info });
    }

    // Only rebuild if families changed
    const familyKeys = Object.keys(families).sort().join(',');
    if (container.dataset.familyKeys === familyKeys) {
      // Just update values
      for (const [name, info] of Object.entries(families)) {
        const slider = /** @type {HTMLInputElement|null} */ (document.getElementById(`slider-${name}`));
        if (slider && document.activeElement !== slider) {
          const overrideVal = callbacks.getOverrides()?.[name];
          slider.value = overrideVal ?? info.value;
          const valueEl = document.getElementById(`slider-val-${name}`);
          if (valueEl) {
            valueEl.textContent = (overrideVal ?? info.value).toFixed(4);
          }
        }
      }
      return;
    }

    container.dataset.familyKeys = familyKeys;
    container.textContent = '';

    /** @type {Record<string, string>} */
    const typeLabels = {
      quadrupole: 'QUADRUPOLES',
      sextupole: 'SEXTUPOLES',
      other: 'OTHER',
    };

    for (const [type, items] of Object.entries(groups)) {
      const groupEl = document.createElement('div');
      groupEl.className = 'slider-group';

      const headerEl = document.createElement('div');
      headerEl.className = 'slider-group-header';
      headerEl.textContent = typeLabels[type] || type.toUpperCase();
      groupEl.appendChild(headerEl);

      for (const item of items) {
        const currentVal = callbacks.getOverrides()?.[item.name] ?? item.value;
        const [rangeMin, rangeMax] = item.range || [-5, 5];
        const step = (rangeMax - rangeMin) / 200;

        const itemEl = document.createElement('div');
        itemEl.className = 'slider-item';

        // Build label row
        const labelRow = document.createElement('div');
        labelRow.className = 'slider-label';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'slider-name';
        nameSpan.textContent = item.name;
        labelRow.appendChild(nameSpan);

        const valueSpan = document.createElement('span');
        valueSpan.className = 'slider-value';
        valueSpan.id = `slider-val-${item.name}`;
        valueSpan.textContent = currentVal.toFixed(4);
        labelRow.appendChild(valueSpan);

        itemEl.appendChild(labelRow);

        // Build slider
        const sliderInput = document.createElement('input');
        sliderInput.type = 'range';
        sliderInput.className = 'slider-input';
        sliderInput.id = `slider-${item.name}`;
        sliderInput.min = String(rangeMin);
        sliderInput.max = String(rangeMax);
        sliderInput.step = String(step);
        sliderInput.value = String(currentVal);
        sliderInput.dataset.family = item.name;

        sliderInput.addEventListener('input', (e) => {
          const target = /** @type {HTMLInputElement} */ (e.target);
          const val = parseFloat(target.value);
          valueSpan.textContent = val.toFixed(4);
        });

        sliderInput.addEventListener('change', (e) => {
          const target = /** @type {HTMLInputElement} */ (e.target);
          const family = /** @type {string} */ (target.dataset.family);
          const val = parseFloat(target.value);
          clearTimeout(sliderDebounceTimers[family]);
          sliderDebounceTimers[family] = setTimeout(() => {
            callbacks.onSliderChange(family, val);
          }, 300);
        });

        itemEl.appendChild(sliderInput);
        groupEl.appendChild(itemEl);
      }

      container.appendChild(groupEl);
    }
  }

  // Dot-notation keys make Plotly merge into the existing layout instead of
  // replacing entire sub-objects (which would wipe axis titles, ranges,
  // tick formats, etc.).
  subscribe(() => {
    const themeLayout = chartTheme();
    figureNames.forEach((name) => {
      const plotEl = /** @type {any} */ (document.getElementById(`plot-${name}`));
      if (plotEl && plotEl.data) {
        /** @type {Record<string, string>} */
        const update = {
          paper_bgcolor: themeLayout.paper_bgcolor,
          plot_bgcolor: themeLayout.plot_bgcolor,
          'font.color': themeLayout.font.color,
        };
        for (const key of Object.keys(plotEl.layout || {})) {
          if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
            update[key + '.gridcolor'] = themeLayout.xaxis.gridcolor;
            update[key + '.zerolinecolor'] = themeLayout.xaxis.gridcolor;
          }
        }
        Plotly.relayout(plotEl, update);
        // Re-theme opt-in markers (e.g. the working-point star) so they track
        // the foreground across a live switch, not just at first render.
        const themedIdx = (plotEl.data || [])
          .map((/** @type {any} */ t, /** @type {number} */ i) =>
            t && t.meta === 'themed-fg-marker' ? i : -1
          )
          .filter((/** @type {number} */ i) => i >= 0);
        if (themedIdx.length) {
          Plotly.restyle(plotEl, { 'marker.color': themeLayout.font.color }, themedIdx);
        }
      }
    });
  });

  return {
    renderState,
    updateSliders,
  };
}
