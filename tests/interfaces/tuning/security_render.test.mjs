// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * Tuning — hostile-field security regression suite.
 *
 * Guards the three render sinks that interpolate caller/backend-supplied
 * strings into `innerHTML`:
 *
 *   1. optimization-form.js `renderTable`   — user-typed PV name (self-XSS),
 *      interpolated into BOTH a quoted `data-pv="…"` attribute and `<td>`
 *      element text.
 *   2. progress-display.js  results table    — backend `objective` field
 *      (fallback when `objective_value` is non-numeric) and `phase`.
 *   3. results-viewer.js    `renderAnalysis`  — backend historical-run
 *      `timestamp`.
 *
 * Every payload MUST be neutralised (escaped, never a live element) on every
 * path. This mirrors the artifacts hostile-metadata suite: the fix is the
 * shared, quote-safe `escapeHtml` from /design-system/js/dom.js.
 *
 * Pure DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/tuning/security_render.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { state } from '../../../src/osprey/interfaces/tuning/static/js/state.js';
import { renderTable } from '../../../src/osprey/interfaces/tuning/static/js/optimization-form.js';
import { initProgressDisplay } from '../../../src/osprey/interfaces/tuning/static/js/progress-display.js';
import {
  initResultsViewer,
  renderAnalysis,
} from '../../../src/osprey/interfaces/tuning/static/js/results-viewer.js';

// plots.js calls the vendored global `Plotly` at runtime; it is absent under
// Vitest. These render sinks set `innerHTML` synchronously (the assertion runs
// before any chart draw), so a no-op stub just prevents an unrelated crash.
globalThis.Plotly = {
  react() {},
  newPlot() {},
  purge() {},
  relayout() {},
};

// Double- and single-quote attribute breakout + live-element injection.
const PAYLOADS = [
  '"><img src=x onerror=alert(1)>',
  "'><svg/onload=alert(1)>",
  '<script>alert(1)</script>',
];

/**
 * Assert a rendered container is structurally inert, inspecting the parsed
 * DOM rather than serialized `innerHTML`. (A payload that survives only as an
 * attribute *value* — e.g. `data-pv="…<img…>"` — is safe: no element is built
 * and the value re-serializes with a literal `<img` that is not markup.
 * Matching that string would be a false positive, so we check structure.)
 *
 *   1. No injected element got built from any payload.
 *   2. No element carries an inline `on*` event-handler attribute (which is
 *      what a successful attribute breakout would create).
 */
function assertInert(container) {
  expect(container.querySelector('img, svg, script')).toBeNull();
  for (const el of container.querySelectorAll('*')) {
    for (const attr of el.attributes) {
      expect(attr.name.toLowerCase().startsWith('on')).toBe(false);
    }
  }
}

beforeEach(() => {
  document.body.innerHTML = '';
  state.setVariableTableData([]);
});

describe('optimization-form renderTable — PV-name sink (attribute + text)', () => {
  for (const payload of PAYLOADS) {
    test(`neutralises ${JSON.stringify(payload)}`, () => {
      state.setVariableTableData([
        { pv_name: payload, current_value: 1, min: 0, max: 2, step_size: null, bo_range_factor: 1, selected: false },
      ]);
      const tbody = document.createElement('tbody');
      renderTable(tbody, true); // showBoRange: exercises all data-pv attribute copies

      assertInert(tbody);
      // The PV name survived only as inert text: textContent decodes the
      // escaped entities back to the literal payload string.
      expect(tbody.textContent).toContain(payload);
    });
  }
});

describe('progress-display results table — backend objective/phase sink', () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <div id="run-status-badge"><span class="status-dot"></span><span class="status-text"></span></div>
      <table><tbody id="results-table-body"></tbody></table>
    `;
    initProgressDisplay();
  });

  for (const payload of PAYLOADS) {
    test(`neutralises objective ${JSON.stringify(payload)}`, () => {
      // Non-numeric objective_value forces the `objective` string fallback sink.
      state.setOptimizationState({
        status: 'RUNNING',
        lhs_data: [{ objective: payload, objective_value: null }],
        bo_data: [],
      });
      assertInert(document.getElementById('results-table-body'));
    });
  }
});

describe('results-viewer renderAnalysis — backend timestamp sink', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="analysis-content"></div>';
    initResultsViewer();
  });

  for (const payload of PAYLOADS) {
    test(`neutralises timestamp ${JSON.stringify(payload)}`, () => {
      renderAnalysis([{ objective_value: 1 }], payload);
      assertInert(document.getElementById('analysis-content'));
    });
  }
});
