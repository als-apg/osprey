/**
 * OSPREY Tuning — App Entry Point
 *
 * Initializes all modules, handles tab switching, and embedded mode.
 */

import { state } from './state.js';
import { initNavbar } from './navbar.js';
import { initOptimizationForm } from './optimization-form.js';
import { initProgressDisplay } from './progress-display.js';
import { initResultsViewer } from './results-viewer.js';

// ---- Embedded Detection ----

function checkEmbedded() {
  const params = new URLSearchParams(window.location.search);
  if (params.get('embedded') === 'true') {
    document.body.classList.add('embedded');
  }
}

// ---- Tab Switching ----

function initTabs() {
  const tabs = document.querySelectorAll('.tuning-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.tab;

      tabs.forEach(t => t.classList.toggle('active', t === tab));
      panels.forEach(p => p.classList.toggle('active', p.id === targetId));

      state.setActiveTab(targetId);
    });
  });
}

// ---- Accordion Toggles ----

function initAccordions() {
  document.querySelectorAll('.accordion-toggle').forEach(toggle => {
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('collapsed');
      const body = toggle.nextElementSibling;
      if (body) body.classList.toggle('collapsed');
    });
  });
}

// ---- Collapsible Sections ----

function initCollapsibles() {
  document.querySelectorAll('.collapsible-toggle').forEach(toggle => {
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('open');
      const body = toggle.nextElementSibling;
      if (body) {
        body.style.display = body.style.display === 'none' ? '' : 'none';
      }
    });
  });
}

// ---- Validation Alert ----

function initAlerts() {
  const dismiss = document.getElementById('validation-dismiss');
  if (dismiss) {
    dismiss.addEventListener('click', () => {
      document.getElementById('validation-alert').style.display = 'none';
    });
  }
}

// ---- Resize Handling (for Plotly) ----

function initResize() {
  const resizeObserver = new ResizeObserver(() => {
    document.querySelectorAll('.js-plotly-plot').forEach(el => {
      Plotly.Plots.resize(el);
    });
  });

  const content = document.querySelector('.tuning-content');
  if (content) resizeObserver.observe(content);

  // Also listen for parent window resize (iframe embed)
  window.addEventListener('resize', () => {
    document.querySelectorAll('.js-plotly-plot').forEach(el => {
      Plotly.Plots.resize(el);
    });
  });
}

// ---- Session Recovery ----

async function attemptRecovery() {
  const { api } = await import('./api.js');
  const savedJobId = sessionStorage.getItem('tuning_jobId');
  if (!savedJobId) return;

  try {
    const result = await api.getState(savedJobId);
    if (result && result.status) {
      state.setJobId(savedJobId);
      state.setOptimizationState(result);
    }
  } catch {
    // Stale job — clear it
    sessionStorage.removeItem('tuning_jobId');
  }
}

// ---- Init ----

document.addEventListener('DOMContentLoaded', () => {
  checkEmbedded();
  initTabs();
  initAccordions();
  initCollapsibles();
  initAlerts();
  initResize();
  initNavbar();
  initOptimizationForm();
  initProgressDisplay();
  initResultsViewer();
  attemptRecovery();
});
