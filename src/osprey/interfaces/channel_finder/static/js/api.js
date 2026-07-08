// @ts-check
/**
 * OSPREY Channel Finder — REST + WebSocket Client
 */

const BASE = '';  // Same-origin

/**
 * Fetch JSON from the API.
 *
 * Responses are dynamically shaped JSON, so the resolved value is `any`: callers
 * read fields off the parsed body directly (the API boundary is untyped by design).
 * @param {string} path - API path (e.g., '/api/info')
 * @param {RequestInit} [opts] - Extra fetch options
 * @returns {Promise<any>}
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
 * @param {string} path - API path.
 * @param {unknown} [data] - Request body (JSON-serialized).
 * @returns {Promise<any>}
 */
export async function postJSON(path, data) {
  return fetchJSON(path, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * PUT JSON to the API.
 * @param {string} path - API path.
 * @param {unknown} [data] - Request body (JSON-serialized).
 * @returns {Promise<any>}
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
 * @param {unknown} [data] - Optional request body.
 * @returns {Promise<any>}
 */
export async function deleteJSON(path, data) {
  /** @type {RequestInit} */
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
 * Structured API error carrying the HTTP status code.
 */
export class ApiError extends Error {
  /**
   * @param {number} status - HTTP status code.
   * @param {string} message - Error detail message.
   */
  constructor(status, message) {
    super(message);
    /** @type {number} */
    this.status = status;
    this.name = 'ApiError';
  }
}
