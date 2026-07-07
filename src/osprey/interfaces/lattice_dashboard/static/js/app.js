// @ts-check
/* OSPREY Lattice Dashboard — Frontend Entry Point
 *
 * Composes the net/render/ui/settings modules and bootstraps the page:
 * DOMContentLoaded button/toggle wiring, initial state fetch, and the SSE
 * connection. No REST/SSE/DOM logic of its own — see net.js, render.js,
 * ui.js, and settings.js for that.
 */

import { initTheme } from '/design-system/js/theme-manager.js';
import { applyEmbedded } from '/design-system/js/frame-params.js';
import '/design-system/js/components/osprey-theme-switcher.js';
import { refreshFast, runVerification, createNetClient } from './net.js';
import {
  updateSummaryStats,
  updateLED,
  showSpinner,
  hideSpinner,
  showFigureError,
  updateFigureStatuses,
  renderPlotly,
  createRenderer,
} from './render.js';
import { createUI } from './ui.js';
import { loadSettings, renderSettingsForm } from './settings.js';

// Panel embedded in the Web Terminal hub: apply the hub's broadcast theme
// and follow live changes. theme-boot.js already applied data-theme
// pre-paint; this call attaches the follower's postMessage listener.
initTheme({ role: 'follower' });

// ── Configuration ───────────────────────────────────────

const FAST_FIGURES = ['optics', 'resonance', 'chromaticity', 'footprint'];
const VERIFICATION_FIGURES = ['da', 'lma'];
const ALL_FIGURES = [...FAST_FIGURES, ...VERIFICATION_FIGURES];

// ── Renderer ─────────────────────────────────────────────
// Network effects are threaded through as callbacks — render.js has no
// dependency on net.js's REST/SSE plumbing (see net.getState()).

const renderer = createRenderer(ALL_FIGURES, {
  onSliderChange: (family, val) => net.setParam(family, val),
  onFigureReady: (name) => net.fetchAndRenderFigure(name),
  getOverrides: () => net.getState()?.overrides,
});

// ── Network Client ──────────────────────────────────────
// Render effects are threaded through as callbacks — net.js has no
// dependency on render.js's DOM rendering.

const net = createNetClient({
  onState: (state) => {
    renderer.renderState(state);
    loadSettings();
  },
  onParamSet: (result) => updateFigureStatuses(result.figures),
  onFigureData: (name, figData) => renderPlotly(name, figData),
  onFigureStatus: (name, status) => {
    updateLED(name, status);
    if (status === 'computing') showSpinner(name);
  },
  onFigureReady: (name) => {
    updateLED(name, 'ready');
    hideSpinner(name);
  },
  onFigureError: (name, error) => {
    updateLED(name, 'error');
    hideSpinner(name);
    showFigureError(name, error);
  },
  onSettingsUpdated: (settings) => renderSettingsForm(
    /** @type {Record<string, Record<string, number|null>>} */ (settings)
  ),
  onBaselineSet: (summary) => updateSummaryStats(summary),
});

// ── UI Chrome (sidebar, layout, tabs, drag-and-drop) ────

const ui = createUI(ALL_FIGURES);

// ── Initialization ──────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Check embedded query param
  applyEmbedded();

  // Bind buttons
  document.getElementById('btn-refresh')?.addEventListener('click', refreshFast);
  document.getElementById('btn-verify')?.addEventListener('click', runVerification);
  document.getElementById('btn-baseline')?.addEventListener('click', net.setBaseline);

  // Layout toggle (guarded — btn may not exist in cached HTML)
  const layoutBtn = document.getElementById('btn-layout');
  if (layoutBtn) {
    layoutBtn.addEventListener('click', ui.toggleLayout);
    ui.initLayout();
  }

  // Sidebar collapse toggle
  const sidebarBtn = document.getElementById('btn-sidebar-toggle');
  if (sidebarBtn) {
    sidebarBtn.addEventListener('click', ui.toggleSidebar);
    ui.initSidebar();
  }
  ui.initSidebarTabs();
  ui.restorePanelOrder();
  ui.setupDragAndDrop();

  // Load initial state
  net.fetchState();

  // Re-fetch state when page becomes visible again (e.g. tab switch, navigation)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') net.fetchState();
  });

  // Connect SSE
  net.connectSSE();
});
