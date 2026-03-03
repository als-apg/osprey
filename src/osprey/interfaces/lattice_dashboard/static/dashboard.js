/* Lattice Dashboard — Frontend Controller
 *
 * Connects to the dashboard server via REST + SSE.
 * Renders Plotly figures, drives slider controls, manages status LEDs.
 */

// ── Configuration ───────────────────────────────────────

const API_BASE = '';  // Same origin (served by dashboard server)

const FAST_FIGURES = ['optics', 'resonance', 'chromaticity', 'footprint'];
const VERIFICATION_FIGURES = ['da', 'lma'];
const ALL_FIGURES = [...FAST_FIGURES, ...VERIFICATION_FIGURES];

// Plotly layout overrides per theme
const DARK_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(22,26,36,0.95)',
  font: { color: '#d8dce6', family: 'JetBrains Mono, monospace', size: 10 },
  margin: { l: 45, r: 15, t: 30, b: 35 },
  xaxis: { gridcolor: '#252a36', zerolinecolor: '#3a4050' },
  yaxis: { gridcolor: '#252a36', zerolinecolor: '#3a4050' },
};

const LIGHT_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(255,255,255,0.95)',
  font: { color: '#1a1d24', family: 'JetBrains Mono, monospace', size: 10 },
  margin: { l: 45, r: 15, t: 30, b: 35 },
  xaxis: { gridcolor: '#e0e3e8', zerolinecolor: '#c8ccd4' },
  yaxis: { gridcolor: '#e0e3e8', zerolinecolor: '#c8ccd4' },
};

function getThemeLayout() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? LIGHT_LAYOUT : DARK_LAYOUT;
}

const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  displaylogo: false,
};

// ── Settings Field Definitions ───────────────────────────

const SETTINGS_FIELDS = {
  da: {
    label: 'DYNAMIC APERTURE',
    fields: [
      { key: 'nturns', label: 'Turns', unit: '', type: 'int', min: 64, max: 8192, step: 64 },
      { key: 'n_angles', label: 'Angles', unit: '', type: 'int', min: 5, max: 72, step: 1 },
      { key: 'amp_max_mm', label: 'Max amp', unit: 'mm', type: 'float', min: 1, max: 100, step: 1 },
      { key: 'n_bisect', label: 'Bisect steps', unit: '', type: 'int', min: 5, max: 30, step: 1 },
    ],
  },
  lma: {
    label: 'MOMENTUM APERTURE',
    fields: [
      { key: 'nturns', label: 'Turns', unit: '', type: 'int', min: 64, max: 8192, step: 64 },
      { key: 'n_refpts', label: 'Ref pts', unit: '', type: 'int', min: 10, max: 500, step: 10 },
      { key: 'dp_max_pct', label: 'dp max', unit: '%', type: 'float', min: 0.5, max: 20, step: 0.5 },
      { key: 'n_sectors', label: 'Sectors', unit: '', type: 'int_or_null', min: 1, max: 100, step: 1 },
      { key: 'n_bisect', label: 'Bisect steps', unit: '', type: 'int', min: 5, max: 30, step: 1 },
    ],
  },
  chromaticity: {
    label: 'CHROMATICITY',
    fields: [
      { key: 'dp_min_pct', label: 'dp min', unit: '%', type: 'float', min: -20, max: 0, step: 0.5 },
      { key: 'dp_max_pct', label: 'dp max', unit: '%', type: 'float', min: 0, max: 20, step: 0.5 },
      { key: 'n_steps', label: 'Steps', unit: '', type: 'int', min: 5, max: 200, step: 5 },
    ],
  },
  footprint: {
    label: 'TUNE FOOTPRINT',
    fields: [
      { key: 'n_amp', label: 'Grid pts', unit: '', type: 'int', min: 3, max: 30, step: 1 },
      { key: 'x_max_mm', label: 'x max', unit: 'mm', type: 'float', min: 0.1, max: 50, step: 0.5 },
      { key: 'y_max_mm', label: 'y max', unit: 'mm', type: 'float', min: 0.1, max: 50, step: 0.5 },
      { key: 'n_half', label: 'Half-turns', unit: '', type: 'int', min: 32, max: 2048, step: 32 },
    ],
  },
};

const SIDEBAR_TAB_KEY = 'lattice-sidebar-tab';

// ── State ───────────────────────────────────────────────

let currentState = null;
let eventSource = null;
let sliderDebounceTimers = {};

// ── Initialization ──────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Theme from parent (web terminal postMessage)
  handleThemeMessages();

  // Check embedded query param
  const params = new URLSearchParams(window.location.search);
  if (params.get('embedded') === 'true') {
    document.body.classList.add('embedded');
  }

  // Bind buttons
  document.getElementById('btn-refresh').addEventListener('click', refreshFast);
  document.getElementById('btn-verify').addEventListener('click', runVerification);
  document.getElementById('btn-baseline').addEventListener('click', setBaseline);

  // Layout toggle (guarded — btn may not exist in cached HTML)
  const layoutBtn = document.getElementById('btn-layout');
  if (layoutBtn) {
    layoutBtn.addEventListener('click', toggleLayout);
    initLayout();
  }

  // Sidebar collapse toggle
  const sidebarBtn = document.getElementById('btn-sidebar-toggle');
  if (sidebarBtn) {
    sidebarBtn.addEventListener('click', toggleSidebar);
    initSidebar();
  }
  initSidebarTabs();
  restorePanelOrder();
  setupDragAndDrop();

  // Load initial state
  fetchState();

  // Re-fetch state when page becomes visible again (e.g. tab switch, navigation)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') fetchState();
  });

  // Connect SSE
  connectSSE();
});

// ── API calls ───────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API ${path}: ${resp.status} ${text}`);
  }
  return resp.json();
}

async function fetchState() {
  try {
    currentState = await apiFetch('/api/state');
    renderState(currentState);
    loadSettings();
  } catch (err) {
    console.warn('Failed to fetch state:', err);
  }
}

async function refreshFast() {
  try {
    await apiFetch('/api/refresh', { method: 'POST' });
  } catch (err) {
    console.error('Refresh failed:', err);
  }
}

async function runVerification() {
  try {
    await apiFetch('/api/verify', { method: 'POST' });
  } catch (err) {
    console.error('Verification failed:', err);
  }
}

async function setBaseline() {
  try {
    await apiFetch('/api/baseline', { method: 'POST' });
    // Re-fetch state to update UI
    await fetchState();
  } catch (err) {
    console.error('Set baseline failed:', err);
  }
}

async function setParam(family, value) {
  try {
    const result = await apiFetch('/api/state/param', {
      method: 'POST',
      body: JSON.stringify({ family, value }),
    });
    currentState = result;
    updateFigureStatuses(result.figures);
  } catch (err) {
    console.error('Set param failed:', err);
  }
}

async function fetchAndRenderFigure(name) {
  try {
    const figData = await apiFetch(`/api/figures/${name}`);
    renderPlotly(name, figData);
  } catch (err) {
    console.warn(`Failed to fetch figure ${name}:`, err);
  }
}

// ── SSE Connection ──────────────────────────────────────

function connectSSE() {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource(`${API_BASE}/api/events`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleSSEEvent(data);
    } catch (err) {
      console.warn('SSE parse error:', err);
    }
  };

  eventSource.onerror = () => {
    console.warn('SSE connection lost, reconnecting in 3s...');
    eventSource.close();
    setTimeout(connectSSE, 3000);
  };
}

function handleSSEEvent(data) {
  switch (data.type) {
    case 'state_updated':
      if (data.summary) updateSummaryStats(data.summary);
      if (data.families) {
        updateSliders(data.families);
        // Enable action buttons once a lattice is loaded
        const hasLattice = Object.keys(data.families).length > 0;
        document.getElementById('btn-refresh').disabled = !hasLattice;
        document.getElementById('btn-verify').disabled = !hasLattice;
        document.getElementById('btn-baseline').disabled = !hasLattice;
      }
      break;

    case 'figure_status':
      updateLED(data.name, data.status);
      if (data.status === 'computing') {
        showSpinner(data.name);
      }
      break;

    case 'figure_ready':
      updateLED(data.name, 'ready');
      hideSpinner(data.name);
      fetchAndRenderFigure(data.name);
      break;

    case 'figure_error':
      updateLED(data.name, 'error');
      hideSpinner(data.name);
      showFigureError(data.name, data.error || 'Unknown error');
      break;

    case 'settings_updated':
      if (data.settings) renderSettingsForm(data.settings);
      break;

    case 'baseline_set':
      if (data.summary) updateSummaryStats(data.summary);
      break;

    case 'baseline_cleared':
      break;
  }
}

// ── Rendering ───────────────────────────────────────────

function renderState(state) {
  if (!state) return;

  // Lattice name
  const latticeName = state.base_lattice || 'No lattice loaded';
  document.getElementById('lattice-name').textContent =
    latticeName.split('/').pop() || latticeName;

  // Summary stats
  updateSummaryStats(state.summary || {});

  // Magnet sliders
  updateSliders(state.families || {});

  // Figure statuses and load ready figures
  if (state.figures) {
    updateFigureStatuses(state.figures);
    for (const name of ALL_FIGURES) {
      if (state.figures[name]?.status === 'ready') {
        fetchAndRenderFigure(name);
      }
    }
  }

  // Enable/disable buttons
  const hasLattice = !!state.base_lattice;
  document.getElementById('btn-refresh').disabled = !hasLattice;
  document.getElementById('btn-verify').disabled = !hasLattice;
  document.getElementById('btn-baseline').disabled = !hasLattice;
}

function updateSummaryStats(summary) {
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

function setStatValue(chipId, value, formatter) {
  const chip = document.getElementById(chipId);
  if (!chip) return;
  const valueEl = chip.querySelector('.stat-value');
  if (!valueEl) return;
  valueEl.textContent = (value != null && !isNaN(value)) ? formatter(value) : '\u2014';
}

function updateSliders(families) {
  const container = document.getElementById('slider-container');
  if (!families || Object.keys(families).length === 0) {
    container.textContent = '';
    const emptyEl = document.createElement('div');
    emptyEl.className = 'sidebar-empty';
    emptyEl.textContent = 'Load a lattice to see controls';
    container.appendChild(emptyEl);
    return;
  }

  // Group by type
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
      const slider = document.getElementById(`slider-${name}`);
      if (slider && document.activeElement !== slider) {
        const overrideVal = currentState?.overrides?.[name];
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
      const currentVal = currentState?.overrides?.[item.name] ?? item.value;
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
      sliderInput.min = rangeMin;
      sliderInput.max = rangeMax;
      sliderInput.step = step;
      sliderInput.value = currentVal;
      sliderInput.dataset.family = item.name;

      sliderInput.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        valueSpan.textContent = val.toFixed(4);
      });

      sliderInput.addEventListener('change', (e) => {
        const family = e.target.dataset.family;
        const val = parseFloat(e.target.value);
        clearTimeout(sliderDebounceTimers[family]);
        sliderDebounceTimers[family] = setTimeout(() => {
          setParam(family, val);
        }, 300);
      });

      itemEl.appendChild(sliderInput);
      groupEl.appendChild(itemEl);
    }

    container.appendChild(groupEl);
  }
}

function updateFigureStatuses(figures) {
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

// ── LED Status ──────────────────────────────────────────

function updateLED(name, status) {
  const led = document.getElementById(`led-${name}`);
  if (led) {
    led.dataset.status = status;
  }
}

// ── Spinner ─────────────────────────────────────────────

function showSpinner(name) {
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

function hideSpinner(name) {
  const plotEl = document.getElementById(`plot-${name}`);
  if (!plotEl) return;
  const spinner = plotEl.querySelector('.figure-spinner');
  if (spinner) spinner.remove();
}

// ── Figure Error ────────────────────────────────────────

function showFigureError(name, error) {
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
  iconDiv.textContent = '\u00d7';
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

// ── Plotly Rendering ────────────────────────────────────

function renderPlotly(name, figData) {
  const plotEl = document.getElementById(`plot-${name}`);
  if (!plotEl) return;

  // Clear placeholder/error/spinner
  const placeholder = plotEl.querySelector('.figure-placeholder');
  if (placeholder) placeholder.remove();
  const errorEl = plotEl.querySelector('.figure-error');
  if (errorEl) errorEl.remove();
  hideSpinner(name);

  // Apply theme-aware layout overrides
  const themeLayout = getThemeLayout();
  const layout = { ...figData.layout, ...themeLayout };
  const defaultFontColor = themeLayout.font.color;

  // Merge axis colors into ALL axes (not just secondary ones)
  if (figData.layout) {
    for (const key of Object.keys(figData.layout)) {
      if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
        layout[key] = {
          ...figData.layout[key],
          gridcolor: themeLayout.xaxis.gridcolor,
          zerolinecolor: themeLayout.xaxis.zerolinecolor,
        };
      }
    }
    // Keep annotations from original with adjusted color
    if (figData.layout.annotations) {
      layout.annotations = figData.layout.annotations.map(a => ({
        ...a,
        font: { ...(a.font || {}), color: (a.font && a.font.color) || defaultFontColor },
      }));
    }
    if (figData.layout.title) {
      layout.title = figData.layout.title;
    }
  }

  // Make figure fill the cell
  layout.autosize = true;
  delete layout.width;
  delete layout.height;

  Plotly.react(plotEl, figData.data || [], layout, PLOTLY_CONFIG);
}

// ── Theme ───────────────────────────────────────────────

function handleThemeMessages() {
  window.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'osprey-theme-change' && event.data.theme) {
      const theme = event.data.theme;
      document.documentElement.setAttribute('data-theme', theme);
      // Re-render Plotly figures with new theme colors.
      // Use dot-notation keys so Plotly merges into the existing layout
      // instead of replacing entire sub-objects (which would wipe axis
      // titles, ranges, tick formats, etc.).
      const themeLayout = getThemeLayout();
      ALL_FIGURES.forEach(name => {
        const plotEl = document.getElementById(`plot-${name}`);
        if (plotEl && plotEl.data) {
          const update = {
            paper_bgcolor: themeLayout.paper_bgcolor,
            plot_bgcolor: themeLayout.plot_bgcolor,
            'font.color': themeLayout.font.color,
          };
          for (const key of Object.keys(plotEl.layout || {})) {
            if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
              update[key + '.gridcolor'] = themeLayout.xaxis.gridcolor;
              update[key + '.zerolinecolor'] = themeLayout.xaxis.zerolinecolor;
            }
          }
          Plotly.relayout(plotEl, update);
        }
      });
    }
  });
}

// ── Drag-and-Drop Panel Rearrangement ────────────────

const PANEL_ORDER_KEY = 'lattice-panel-order';
const LAYOUT_KEY = 'lattice-layout-mode';
const SIDEBAR_KEY = 'lattice-sidebar-collapsed';

function initSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  // Default to collapsed (true) unless user explicitly expanded
  const collapsed = localStorage.getItem(SIDEBAR_KEY) !== 'false';
  sidebar.classList.toggle('sidebar-collapsed', collapsed);
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  const collapsed = sidebar.classList.toggle('sidebar-collapsed');
  localStorage.setItem(SIDEBAR_KEY, collapsed ? 'true' : 'false');

  // Plotly figures need to reflow when sidebar width changes
  setTimeout(() => {
    ALL_FIGURES.forEach(name => {
      const plotEl = document.getElementById(`plot-${name}`);
      if (plotEl && plotEl.data) {
        Plotly.relayout(plotEl, { autosize: true });
      }
    });
  }, 250);
}

function initLayout() {
  const mode = localStorage.getItem(LAYOUT_KEY) || 'stacked';
  applyLayout(mode);
}

function toggleLayout() {
  const figArea = document.getElementById('figure-area');
  const isStacked = figArea.classList.contains('layout-stacked');
  applyLayout(isStacked ? 'grid' : 'stacked');
}

function applyLayout(mode) {
  const figArea = document.getElementById('figure-area');
  if (!figArea) return;
  const btn = document.getElementById('btn-layout');

  if (mode === 'stacked') {
    figArea.classList.add('layout-stacked');
    if (btn) {
      btn.querySelector('.layout-label').textContent = 'Grid';
      btn.querySelector('.icon-grid').style.display = '';
      btn.querySelector('.icon-stack').style.display = 'none';
    }
  } else {
    figArea.classList.remove('layout-stacked');
    if (btn) {
      btn.querySelector('.layout-label').textContent = 'Stack';
      btn.querySelector('.icon-grid').style.display = 'none';
      btn.querySelector('.icon-stack').style.display = '';
    }
  }

  localStorage.setItem(LAYOUT_KEY, mode);

  ALL_FIGURES.forEach(name => {
    const plotEl = document.getElementById(`plot-${name}`);
    if (plotEl && plotEl.data) {
      Plotly.relayout(plotEl, { autosize: true });
    }
  });
}

function setupDragAndDrop() {
  const cells = document.querySelectorAll('.figure-cell');
  let draggedCell = null;

  cells.forEach(cell => {
    const header = cell.querySelector('.figure-header');
    if (!header) return;

    // Only the header bar is the drag handle
    header.setAttribute('draggable', 'true');

    header.addEventListener('dragstart', (e) => {
      draggedCell = cell;
      cell.classList.add('dragging');
      document.body.classList.add('drag-active');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', cell.id);
    });

    // Drop target is the whole cell
    cell.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      if (cell !== draggedCell) {
        cell.classList.add('drag-over');
      }
    });

    cell.addEventListener('dragleave', (e) => {
      // Only remove highlight when actually leaving the cell,
      // not when entering a child element
      if (!cell.contains(e.relatedTarget)) {
        cell.classList.remove('drag-over');
      }
    });

    cell.addEventListener('drop', (e) => {
      e.preventDefault();
      cell.classList.remove('drag-over');
      if (draggedCell && draggedCell !== cell) {
        swapCells(draggedCell, cell);
        savePanelOrder();

        // Plotly needs resize after DOM reparenting
        const plotA = draggedCell.querySelector('.figure-plot');
        const plotB = cell.querySelector('.figure-plot');
        if (plotA) Plotly.relayout(plotA, {autosize: true});
        if (plotB) Plotly.relayout(plotB, {autosize: true});
      }
    });

    header.addEventListener('dragend', () => {
      document.body.classList.remove('drag-active');
      if (draggedCell) {
        draggedCell.classList.remove('dragging');
        draggedCell = null;
      }
      // Clean up any stale drag-over highlights
      cells.forEach(c => c.classList.remove('drag-over'));
    });
  });
}

function swapCells(cellA, cellB) {
  const parentA = cellA.parentNode;
  const parentB = cellB.parentNode;
  const nextA = cellA.nextSibling;
  // Insert A before B, then B into A's old position
  parentB.insertBefore(cellA, cellB);
  if (nextA) {
    parentA.insertBefore(cellB, nextA);
  } else {
    parentA.appendChild(cellB);
  }
}

function savePanelOrder() {
  const cells = document.querySelectorAll('.figure-cell');
  const order = Array.from(cells).map(c => c.dataset.figure);
  localStorage.setItem(PANEL_ORDER_KEY, JSON.stringify(order));
}

// ── Sidebar Tabs ─────────────────────────────────────────

function initSidebarTabs() {
  const tabs = document.querySelectorAll('.sidebar-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabName = tab.dataset.tab;
      const sidebar = document.getElementById('sidebar');

      // If sidebar is collapsed, expand it
      if (sidebar && sidebar.classList.contains('sidebar-collapsed')) {
        sidebar.classList.remove('sidebar-collapsed');
        localStorage.setItem(SIDEBAR_KEY, 'false');
        setTimeout(() => {
          ALL_FIGURES.forEach(name => {
            const plotEl = document.getElementById(`plot-${name}`);
            if (plotEl && plotEl.data) Plotly.relayout(plotEl, { autosize: true });
          });
        }, 250);
      }

      switchTab(tabName);
    });
  });

  // Restore last active tab
  const savedTab = localStorage.getItem(SIDEBAR_TAB_KEY);
  if (savedTab) switchTab(savedTab);
}

function switchTab(tabName) {
  // Update tab buttons
  document.querySelectorAll('.sidebar-tab').forEach(t => {
    t.classList.toggle('sidebar-tab--active', t.dataset.tab === tabName);
  });
  // Update tab content
  document.querySelectorAll('.sidebar-tab-content').forEach(panel => {
    panel.classList.toggle('sidebar-tab-content--active', panel.id === `tab-${tabName}`);
  });
  localStorage.setItem(SIDEBAR_TAB_KEY, tabName);
}

// ── Settings ─────────────────────────────────────────────

async function loadSettings() {
  try {
    const settings = await apiFetch('/api/settings');
    renderSettingsForm(settings);
  } catch (err) {
    console.warn('Failed to load settings:', err);
  }
}

function renderSettingsForm(settings) {
  const container = document.getElementById('settings-container');
  if (!container) return;
  container.textContent = '';

  for (const [group, meta] of Object.entries(SETTINGS_FIELDS)) {
    const groupEl = document.createElement('div');
    groupEl.className = 'settings-group';

    const header = document.createElement('div');
    header.className = 'settings-group-header';
    header.textContent = meta.label;
    groupEl.appendChild(header);

    const groupSettings = settings[group] || {};

    for (const field of meta.fields) {
      const row = document.createElement('div');
      row.className = 'settings-field';

      const label = document.createElement('span');
      label.className = 'settings-field-label';
      label.textContent = field.label;
      row.appendChild(label);

      const input = document.createElement('input');
      input.type = 'number';
      input.className = 'settings-field-input';
      input.id = `setting-${group}-${field.key}`;
      input.min = field.min;
      input.max = field.max;
      input.step = field.step;
      input.dataset.group = group;
      input.dataset.key = field.key;
      input.dataset.fieldType = field.type;

      const val = groupSettings[field.key];
      if (val != null) {
        input.value = field.type === 'int' || field.type === 'int_or_null' ? Math.round(val) : val;
      } else if (field.type === 'int_or_null') {
        input.placeholder = 'auto';
      }

      row.appendChild(input);

      if (field.unit) {
        const unit = document.createElement('span');
        unit.className = 'settings-field-unit';
        unit.textContent = field.unit;
        row.appendChild(unit);
      }

      groupEl.appendChild(row);
    }

    container.appendChild(groupEl);
  }

  // Action buttons
  const actions = document.createElement('div');
  actions.className = 'settings-actions';

  const resetBtn = document.createElement('button');
  resetBtn.className = 'settings-btn';
  resetBtn.textContent = 'RESET';
  resetBtn.addEventListener('click', resetSettings);
  actions.appendChild(resetBtn);

  const applyBtn = document.createElement('button');
  applyBtn.className = 'settings-btn settings-btn--apply';
  applyBtn.textContent = 'APPLY';
  applyBtn.addEventListener('click', applySettings);
  actions.appendChild(applyBtn);

  container.appendChild(actions);
}

function collectSettingsFromForm() {
  const settings = {};
  const inputs = document.querySelectorAll('.settings-field-input');
  for (const input of inputs) {
    const group = input.dataset.group;
    const key = input.dataset.key;
    const fieldType = input.dataset.fieldType;

    if (!settings[group]) settings[group] = {};

    if (fieldType === 'int_or_null') {
      const raw = input.value.trim();
      settings[group][key] = raw === '' ? null : parseInt(raw, 10);
    } else if (fieldType === 'int') {
      settings[group][key] = parseInt(input.value, 10);
    } else {
      settings[group][key] = parseFloat(input.value);
    }
  }
  return settings;
}

async function applySettings() {
  const settings = collectSettingsFromForm();
  try {
    await apiFetch('/api/settings', {
      method: 'PUT',
      body: JSON.stringify({ settings }),
    });
    // Refresh fast figures (cheap, ~1-2s each)
    await apiFetch('/api/refresh', { method: 'POST' });
  } catch (err) {
    console.error('Apply settings failed:', err);
  }
}

async function resetSettings() {
  try {
    const result = await apiFetch('/api/settings', { method: 'DELETE' });
    if (result.settings) renderSettingsForm(result.settings);
  } catch (err) {
    console.error('Reset settings failed:', err);
  }
}

// ── Drag-and-Drop Panel Rearrangement ────────────────

function restorePanelOrder() {
  const saved = localStorage.getItem(PANEL_ORDER_KEY);
  if (!saved) return;

  try {
    const order = JSON.parse(saved);

    // Collect all containers that hold figure cells (grid + verification row)
    const containers = document.querySelectorAll('.figure-grid, .verification-row');
    // Collect all cells in current DOM order
    const allSlots = [];
    containers.forEach(container => {
      Array.from(container.children).forEach(child => {
        if (child.classList.contains('figure-cell')) {
          allSlots.push({ container, placeholder: child });
        }
      });
    });

    // Build lookup of cells by figure name
    const cellMap = {};
    allSlots.forEach(slot => {
      cellMap[slot.placeholder.dataset.figure] = slot.placeholder;
    });

    // Validate saved order matches current DOM cells — if panel names
    // have changed (e.g. fma → lma), discard stale order
    const currentNames = new Set(Object.keys(cellMap));
    const savedNames = new Set(order);
    if (order.length !== allSlots.length || ![...currentNames].every(n => savedNames.has(n))) {
      localStorage.removeItem(PANEL_ORDER_KEY);
      return;
    }

    // Reorder: place cells into slots according to saved order
    const slotPositions = allSlots.map(s => ({
      container: s.container,
      nextSibling: s.placeholder.nextSibling,
    }));

    // Detach all cells
    allSlots.forEach(s => s.placeholder.remove());

    // Re-insert in saved order
    order.forEach((figureName, i) => {
      const cell = cellMap[figureName];
      if (!cell || i >= slotPositions.length) return;
      const pos = slotPositions[i];
      if (pos.nextSibling) {
        pos.container.insertBefore(cell, pos.nextSibling);
      } else {
        pos.container.appendChild(cell);
      }
    });
  } catch (e) {
    console.warn('Failed to restore panel order:', e);
    localStorage.removeItem(PANEL_ORDER_KEY);
  }
}
