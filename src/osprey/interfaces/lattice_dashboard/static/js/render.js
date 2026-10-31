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

import { subscribe, chartTheme, monoFontStack } from '/design-system/js/theme-manager.js';

// Theme-independent Plotly layout — colors come from chartTheme() at
// render/re-theme time (see renderPlotly() and the theme subscription
// below).
//
// The modebar is always shown and sits in the top-right corner of the plot
// div (plotly draws it at top 2px, 22px tall). Nothing the figure lays out
// may share that band, at any card width, so the top margin starts below
// it: the plot area begins at MODEBAR_CLEARANCE, and a label the figure
// pins to the plot's top edge (a subplot title) gets its own row under the
// band. The figure's own title moves to the card caption (see
// setFigureCaption()), and a legend laid out above the plot moves below it.
const MODEBAR_CLEARANCE = 30;
const FIGURE_MARGIN = { l: 45, r: 15, t: MODEBAR_CLEARANCE, b: 35 };
const FIGURE_FONT_SIZE = 10;
// One row of top-edge label text: a line of its font size plus a small gap
// to the modebar band above it.
const TOP_LABEL_LEADING = 1.3;
const TOP_LABEL_GAP = 4;

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
  valueEl.textContent = value != null && !isNaN(value) ? formatter(value) : '—';
}

/** @param {any} summary */
export function updateSummaryStats(summary) {
  setStatValue('stat-energy', summary.energy_gev, (v) => v.toFixed(2));
  setStatValue('stat-circumference', summary.circumference_m, (v) => v.toFixed(1));

  // Tunes and chromaticity come from the optics figure of the inputs on
  // screen; until it exists, or for a model with no tune, the chips say "—".
  const tunes = summary.tunes || [];
  const chromaticity = summary.chromaticity || [];
  setStatValue('stat-nux', tunes[0], (v) => v.toFixed(4));
  setStatValue('stat-nuy', tunes[1], (v) => v.toFixed(4));
  setStatValue('stat-chrom-x', chromaticity[0], (v) => v.toFixed(2));
  setStatValue('stat-chrom-y', chromaticity[1], (v) => v.toFixed(2));
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
  const arc = document.createElement('div');
  arc.className = 'spinner-arc';
  spinner.appendChild(arc);
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
 * Show that a figure's computation failed; the error text is shown beneath
 * the headline only when one is known.
 * @param {string} name
 * @param {string | null} error
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

  if (error) {
    const detailDiv = document.createElement('div');
    detailDiv.className = 'figure-error-msg';
    detailDiv.textContent = error.slice(0, 200);
    errorEl.appendChild(detailDiv);
  }

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
    if (info.status === 'failed' && info.error) {
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
    // Read per render so a re-theme that changes the stack takes effect on
    // the next draw.
    font: { family: monoFontStack(), size: FIGURE_FONT_SIZE, color: themeLayout.font.color },
    margin: { ...FIGURE_MARGIN, t: MODEBAR_CLEARANCE + topLabelRoom(figData.layout) },
  };
  const defaultFontColor = themeLayout.font.color;

  // The card caption names the figure, so plotly does not draw the title a
  // second time — in the band the modebar covers. The caption carries the
  // figure's title text, which can say more than the static name (the
  // chromaticity values, the surviving particle count).
  setFigureCaption(name, figData.layout?.title);
  delete layout.title;

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
    // A horizontal legend laid out above the plot would share the modebar's
    // band; anchor it to the bottom of the figure instead. Plotly reserves
    // a container-anchored legend's height below the axis titles itself.
    if (isLegendAbovePlot(figData.layout.legend)) {
      layout.legend = {
        ...figData.layout.legend,
        x: 0.5,
        xanchor: 'center',
        xref: 'paper',
        y: 0,
        yanchor: 'bottom',
        yref: 'container',
      };
    }
  }

  // Make figure fill the cell
  layout.autosize = true;
  delete layout.width;
  delete layout.height;

  Plotly.react(plotEl, applyThemedMarkers(figData.data, defaultFontColor), layout, PLOTLY_CONFIG);
}

/**
 * Pixels a figure needs above its plot area for labels pinned to the plot's
 * top edge — make_subplots' first-row subplot titles are annotations at paper
 * y=1 anchored at their bottom. One row, sized by the largest such label.
 * @param {any} sourceLayout - the figure's own layout
 * @returns {number}
 */
function topLabelRoom(sourceLayout) {
  const annotations = Array.isArray(sourceLayout?.annotations) ? sourceLayout.annotations : [];
  let room = 0;
  for (const annotation of annotations) {
    const pinnedToTop =
      annotation.yref === 'paper' && annotation.yanchor === 'bottom' && annotation.y >= 1;
    if (!pinnedToTop) continue;
    const size = annotation.font?.size ?? FIGURE_FONT_SIZE;
    room = Math.max(room, Math.ceil(size * TOP_LABEL_LEADING) + TOP_LABEL_GAP);
  }
  return room;
}

/**
 * Whether a figure lays its legend out horizontally above the plot area.
 * @param {any} legend - the figure's own layout.legend
 * @returns {boolean}
 */
function isLegendAbovePlot(legend) {
  return legend?.orientation === 'h' && typeof legend.y === 'number' && legend.y >= 1;
}

/**
 * Put a figure's title text into its card caption. Plotly titles may carry
 * <sub>/<sup> markup (ξ<sub>x</sub>); those become real elements, any other
 * tag is dropped, and every piece of text goes in as a text node.
 * @param {string} name
 * @param {any} title - layout.title, a string or {text}
 */
function setFigureCaption(name, title) {
  const text = typeof title === 'string' ? title : title?.text;
  if (typeof text !== 'string' || text.trim() === '') return;
  const captionEl = document.querySelector(`#cell-${name} .figure-title`);
  if (!captionEl) return;

  captionEl.textContent = '';
  /** @type {Element} */
  let target = captionEl;
  for (const part of text.split(/(<\/?(?:sub|sup)>)/i)) {
    const tag = part.match(/^<(\/?)(sub|sup)>$/i);
    if (tag && !tag[1] && target === captionEl) {
      const scriptEl = document.createElement(tag[2].toLowerCase());
      captionEl.appendChild(scriptEl);
      target = scriptEl;
    } else if (tag) {
      if (tag[1]) target = captionEl;
    } else if (part) {
      target.appendChild(document.createTextNode(part.replace(/<[^>]*>/g, '')));
    }
  }
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

    const selection = state.selection || {};
    const ready = selection.status === 'ready';

    // The selected model's name
    const latticeNameEl = document.getElementById('lattice-name');
    if (latticeNameEl) {
      latticeNameEl.textContent = selection.model || 'No lattice loaded';
    }

    // Simple-mode status line (hidden in Expert via CSS): a plain-language
    // stand-in for the dense stat chips. Updated on every state render so it
    // is correct the moment the mode is flipped to simple. It never names the
    // model: the header already does (the tile bar's subtitle when embedded,
    // #lattice-name standalone), and the notice banner says why a model did
    // not load.
    const simpleStatusEl = document.getElementById('lattice-simple-status');
    if (simpleStatusEl) {
      if (ready) {
        simpleStatusEl.textContent = 'Showing beta functions and dispersion along the lattice';
      } else if (selection.status === 'failed') {
        simpleStatusEl.textContent = 'No optics to show';
      } else {
        simpleStatusEl.textContent = 'Optics appear when a lattice loads';
      }
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
    setButtonDisabled('btn-refresh', !ready);
    setButtonDisabled('btn-verify', !ready || !selection.capabilities?.verify);
    setButtonDisabled('btn-baseline', !ready);
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
        const slider = /** @type {HTMLInputElement|null} */ (
          document.getElementById(`slider-${name}`)
        );
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
