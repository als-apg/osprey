/**
 * OSPREY Channel Finder — REST + WebSocket Client
 */

const BASE = '';  // Same-origin

/**
 * Fetch JSON from the API.
 * @param {string} path - API path (e.g., '/api/info')
 * @param {object} [opts] - Extra fetch options
 * @returns {Promise<object>}
 */
export async function fetchJSON(path, opts = {}) {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new ApiError(resp.status, body.detail || body.error_message || resp.statusText);
  }
  return resp.json();
}

/**
 * POST JSON to the API.
 */
export async function postJSON(path, data) {
  return fetchJSON(path, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * PUT JSON to the API.
 */
export async function putJSON(path, data) {
  return fetchJSON(path, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

/**
 * DELETE to the API.
 * @param {string} path - API path.
 * @param {object} [data] - Optional request body.
 */
export async function deleteJSON(path, data) {
  const opts = { method: 'DELETE' };
  if (data !== undefined) {
    opts.body = JSON.stringify(data);
  }
  return fetchJSON(path, opts);
}

/**
 * Open a WebSocket connection.
 * @param {string} path - WebSocket path (e.g., '/ws/search/abc')
 * @returns {WebSocket}
 */
export function openWS(path) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return new WebSocket(`${proto}//${location.host}${path}`);
}

/**
 * Structured API error.
 */
export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}
