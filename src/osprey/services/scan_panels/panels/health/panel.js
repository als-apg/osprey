// @ts-check
/**
 * Scan-health dashboard panel (task 2.3, panel-health).
 *
 * Polls `GET /health/full` on the scan panels sidecar every
 * {@link POLL_INTERVAL_MS} and renders a worst-of rollup banner plus one
 * card per dependency (the Bluesky bridge, Tiled, and the virtual-
 * accelerator IOC). Read-only: no write verbs are ever issued from this
 * panel.
 *
 * Requests are prefix-relative so the panel works both standalone (mounted
 * by the sidecar at `/health-panel`) and reverse-proxied by the web
 * terminal at `/panel/{id}/…`, which strips its own `/panel/{id}` segment
 * before forwarding.
 *
 * @module panel
 */

import { escapeHtml } from '/design-system/js/dom.js';

/** How often to re-poll `/health/full`, in milliseconds. */
const POLL_INTERVAL_MS = 5000;

/** @typedef {{name: string, status: string, detail: string, latency_ms: number}} ServiceHealth */
/** @typedef {{services: ServiceHealth[], rollup: string}} HealthFullResponse */

/**
 * The panel's own path prefix when reverse-proxied by the web terminal at
 * `/panel/{id}/…`. Empty string when the panel is reached directly (e.g.
 * mounted standalone by the sidecar, or opened in the visual test harness),
 * so `api()` falls back to root-relative paths in that case.
 */
const PREFIX = (location.pathname.match(/^\/panel\/[^/]+/) || [''])[0];

/**
 * Build a sidecar API path, honoring the reverse-proxy prefix above.
 *
 * @param {string} path
 * @returns {string}
 */
function api(path) {
  return PREFIX + path;
}

/**
 * Fixed display order and static copy for each probed service. `va_ioc` is
 * called out explicitly as a raw EPICS Channel-Access TCP connect probe --
 * not an HTTP health check -- unlike the bridge and Tiled, which are true
 * HTTP probes.
 *
 * @type {Array<{key: string, label: string, probeKind: string}>}
 */
const SERVICE_ORDER = [
  { key: 'bridge', label: 'Bluesky Bridge', probeKind: 'HTTP probe (GET /health)' },
  { key: 'tiled', label: 'Tiled', probeKind: 'HTTP probe (GET /healthz)' },
  {
    key: 'va_ioc',
    label: 'Virtual Accelerator IOC',
    probeKind:
      'EPICS Channel-Access TCP connect probe — not an HTTP health check, unlike the bridge and Tiled above',
  },
];

const rollupBanner = /** @type {HTMLElement} */ (document.getElementById('rollup-banner'));
const rollupPill = /** @type {HTMLElement} */ (document.getElementById('rollup-pill'));
const rollupUpdated = /** @type {HTMLElement} */ (document.getElementById('rollup-updated'));
const unavailableEl = /** @type {HTMLElement} */ (document.getElementById('unavailable'));
const grid = /** @type {HTMLElement} */ (document.getElementById('service-grid'));

/**
 * @param {string} status
 * @returns {string}
 */
function pillClass(status) {
  if (status === 'ok') return 'pill ok';
  if (status === 'unhealthy') return 'pill unhealthy';
  return 'pill unknown';
}

/**
 * @param {number} latencyMs
 * @returns {string}
 */
function formatLatency(latencyMs) {
  if (!Number.isFinite(latencyMs)) return '—';
  return latencyMs.toFixed(1);
}

/**
 * Render one service card's markup. Every server-sourced string (detail
 * text in particular -- it can carry a probe URL or exception message)
 * goes through `escapeHtml` before interpolation.
 *
 * @param {{key: string, label: string, probeKind: string}} meta
 * @param {ServiceHealth | undefined} health
 * @returns {string}
 */
function renderCard(meta, health) {
  const status = health ? health.status : 'unknown';
  const detail = health ? health.detail : 'No data yet from /health/full.';
  const latency = health ? formatLatency(health.latency_ms) : '—';
  const cardClass = status === 'unhealthy' ? 'card unhealthy' : 'card';
  return `
    <article class="${cardClass}" data-service="${escapeHtml(meta.key)}">
      <div class="card-header">
        <h2>${escapeHtml(meta.label)}</h2>
        <span class="${pillClass(status)}">${escapeHtml(status)}</span>
      </div>
      <p class="probe-kind">${escapeHtml(meta.probeKind)}</p>
      <div class="metric">
        <span class="value">${escapeHtml(latency)}</span>
        <span class="unit">ms</span>
      </div>
      <p class="detail">${escapeHtml(detail)}</p>
    </article>
  `;
}

/**
 * Render the health-unavailable state: hides the rollup banner and service
 * grid content behind a clean error card rather than showing stale or
 * partially-built markup. Used both when `/health/full` itself fails/
 * non-200s and when the panel is served standalone with no live sidecar
 * behind it (e.g. the visual regression harness).
 */
function renderUnavailable() {
  unavailableEl.hidden = false;
  rollupBanner.dataset.status = 'unknown';
  rollupPill.className = 'pill unknown';
  rollupPill.textContent = 'unavailable';
  rollupUpdated.textContent = '';
  grid.innerHTML = SERVICE_ORDER.map((meta) => renderCard(meta, undefined)).join('');
}

/**
 * @param {HealthFullResponse} data
 */
function renderHealth(data) {
  unavailableEl.hidden = true;

  const rollup = typeof data.rollup === 'string' ? data.rollup : 'unknown';
  rollupBanner.dataset.status = rollup;
  rollupPill.className = pillClass(rollup);
  rollupPill.textContent = rollup;
  rollupUpdated.textContent = `updated ${new Date().toLocaleTimeString()}`;

  const byName = new Map((data.services || []).map((service) => [service.name, service]));
  grid.innerHTML = SERVICE_ORDER.map((meta) => renderCard(meta, byName.get(meta.key))).join('');
}

/**
 * Poll `/health/full` once. Any failure -- network error, non-200, or a
 * response that doesn't parse as JSON -- degrades to the unavailable state
 * rather than throwing or leaving stale content on screen.
 */
async function poll() {
  try {
    const response = await fetch(api('/health/full'));
    if (!response.ok) {
      renderUnavailable();
      return;
    }
    const data = /** @type {HealthFullResponse} */ (await response.json());
    renderHealth(data);
  } catch {
    renderUnavailable();
  }
}

poll();
setInterval(poll, POLL_INTERVAL_MS);
