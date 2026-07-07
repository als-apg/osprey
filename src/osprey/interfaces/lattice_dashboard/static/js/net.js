// @ts-check
/* OSPREY Lattice Dashboard — Network Layer
 *
 * REST calls against the dashboard server's /api/* endpoints, plus the SSE
 * event stream that keeps the UI synced with backend-driven figure/settings
 * changes.
 *
 * Render effects are not implemented here — they are injected via callbacks
 * so this module has no dependency on the DOM-rendering code in render.js.
 */

const API_BASE = '';  // Same origin (served by dashboard server)

/**
 * Fetch JSON from the dashboard API, throwing on a non-OK response.
 * @param {string} path - API path (e.g. '/api/state')
 * @param {RequestInit} [options] - Extra fetch options
 * @returns {Promise<any>}
 */
export async function apiFetch(path, options = {}) {
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

/** Trigger a fast (cheap) figure refresh. Fire-and-forget; errors are logged. */
export async function refreshFast() {
  try {
    await apiFetch('/api/refresh', { method: 'POST' });
  } catch (err) {
    console.error('Refresh failed:', err);
  }
}

/** Kick off the (expensive) verification figures. Fire-and-forget; errors are logged. */
export async function runVerification() {
  try {
    await apiFetch('/api/verify', { method: 'POST' });
  } catch (err) {
    console.error('Verification failed:', err);
  }
}

/**
 * Route one parsed SSE payload to the matching handler. A pure dispatch
 * table with no network or DOM side effects of its own, so it can be unit
 * tested by calling it directly with a stub `handlers`.
 * @param {{type: string, name?: string, status?: string, error?: string, settings?: object, summary?: object}} data
 * @param {object} handlers
 * @param {() => void} handlers.onStateUpdated
 * @param {(name: string, status: string) => void} handlers.onFigureStatus
 * @param {(name: string) => void} handlers.onFigureReady
 * @param {(name: string, error: string) => void} handlers.onFigureError
 * @param {(settings: object) => void} handlers.onSettingsUpdated
 * @param {(summary: object) => void} handlers.onBaselineSet
 * @param {() => void} [handlers.onBaselineCleared]
 */
export function handleSSEEvent(data, handlers) {
  switch (data.type) {
    case 'state_updated':
      // Signal-only: fetch authoritative state so the banner, buttons,
      // sliders, summary, and ready figures all stay in sync without the
      // broadcast carrying every field.
      handlers.onStateUpdated();
      break;

    case 'figure_status':
      handlers.onFigureStatus(/** @type {string} */ (data.name), /** @type {string} */ (data.status));
      break;

    case 'figure_ready':
      handlers.onFigureReady(/** @type {string} */ (data.name));
      break;

    case 'figure_error':
      handlers.onFigureError(/** @type {string} */ (data.name), data.error || 'Unknown error');
      break;

    case 'settings_updated':
      if (data.settings) handlers.onSettingsUpdated(data.settings);
      break;

    case 'baseline_set':
      if (data.summary) handlers.onBaselineSet(data.summary);
      break;

    case 'baseline_cleared':
      handlers.onBaselineCleared?.();
      break;
  }
}

/**
 * @typedef {object} NetCallbacks
 * @property {(state: any) => void} onState - fired after /api/state resolves (initial load, re-fetch, or a 'state_updated' SSE signal)
 * @property {(result: any) => void} onParamSet - fired with the updated state after a slider change is applied
 * @property {(name: string, figData: any) => void} onFigureData - fired with figure JSON once fetched (initial ready figures and 'figure_ready' events)
 * @property {(name: string, status: string) => void} onFigureStatus - fired on a 'figure_status' SSE event
 * @property {(name: string) => void} onFigureReady - fired on a 'figure_ready' SSE event, before the figure itself is fetched
 * @property {(name: string, error: string) => void} onFigureError - fired on a 'figure_error' SSE event
 * @property {(settings: object) => void} onSettingsUpdated - fired on a 'settings_updated' SSE event
 * @property {(summary: object) => void} onBaselineSet - fired on a 'baseline_set' SSE event
 */

/**
 * Create a network client bound to a set of render-effect callbacks. Owns
 * the SSE connection and the last-fetched dashboard state.
 * @param {NetCallbacks} callbacks
 */
export function createNetClient(callbacks) {
  /** @type {any} */
  let state = null;
  /** @type {EventSource | null} */
  let eventSource = null;

  async function fetchState() {
    try {
      state = await apiFetch('/api/state');
      callbacks.onState(state);
    } catch (err) {
      console.warn('Failed to fetch state:', err);
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

  /**
   * @param {string} family
   * @param {number} value
   */
  async function setParam(family, value) {
    try {
      const result = await apiFetch('/api/state/param', {
        method: 'POST',
        body: JSON.stringify({ family, value }),
      });
      state = result;
      callbacks.onParamSet(result);
    } catch (err) {
      console.error('Set param failed:', err);
    }
  }

  /** @param {string} name */
  async function fetchAndRenderFigure(name) {
    try {
      const figData = await apiFetch(`/api/figures/${name}`);
      callbacks.onFigureData(name, figData);
    } catch (err) {
      console.warn(`Failed to fetch figure ${name}:`, err);
    }
  }

  const sseHandlers = {
    onStateUpdated: fetchState,
    /** @type {(name: string, status: string) => void} */
    onFigureStatus: (name, status) => callbacks.onFigureStatus(name, status),
    /** @type {(name: string) => void} */
    onFigureReady: (name) => {
      callbacks.onFigureReady(name);
      fetchAndRenderFigure(name);
    },
    /** @type {(name: string, error: string) => void} */
    onFigureError: (name, error) => callbacks.onFigureError(name, error),
    /** @type {(settings: object) => void} */
    onSettingsUpdated: (settings) => callbacks.onSettingsUpdated(settings),
    /** @type {(summary: object) => void} */
    onBaselineSet: (summary) => callbacks.onBaselineSet(summary),
  };

  function connectSSE() {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource(`${API_BASE}/api/events`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleSSEEvent(data, sseHandlers);
      } catch (err) {
        console.warn('SSE parse error:', err);
      }
    };

    eventSource.onerror = () => {
      console.warn('SSE connection lost, reconnecting in 3s...');
      eventSource?.close();
      setTimeout(connectSSE, 3000);
    };
  }

  return {
    fetchState,
    setBaseline,
    setParam,
    fetchAndRenderFigure,
    connectSSE,
    getState: () => state,
  };
}
