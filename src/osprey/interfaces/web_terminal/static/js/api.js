/* OSPREY Web Terminal — Connection Helpers */

/** @typedef {'connected'|'connecting'|'disconnected'} ConnState */
/** @typedef {{ ws: ConnState, sse: ConnState }} ConnectionState */
/** @typedef {(state: ConnectionState) => void} StateListener */

/**
 * @typedef {object} WebSocketHandlers
 * @property {(ws: WebSocket) => void} [onOpen]
 * @property {(e: MessageEvent) => void} [onMessage]
 * @property {(e: CloseEvent) => void} [onClose]
 * @property {(e: Event) => void} [onError]
 */

/**
 * @typedef {object} EventSourceHandlers
 * @property {(data: any) => void} [onMessage]
 * @property {() => void} [onError]
 */

/** @type {ConnState} */
let wsState = 'disconnected';
/** @type {ConnState} */
let sseState = 'disconnected';

/** @type {StateListener[]} */
const stateListeners = [];

function notifyStateChange() {
  for (const fn of stateListeners) fn({ ws: wsState, sse: sseState });
}

/** @param {StateListener} fn */
export function onConnectionStateChange(fn) {
  stateListeners.push(fn);
}

export function getConnectionState() {
  return { ws: wsState, sse: sseState };
}

/**
 * Prepend the per-user URL prefix (`window.__OSPREY_PREFIX__`, e.g.
 * '/u/alice') to a root-absolute path. Multi-user deployments serve each
 * user's container behind such a prefix; this is the single chokepoint that
 * retargets app-relative paths onto it. A no-op when the prefix is
 * empty/absent (single-origin/dev behavior is unchanged), when `path` isn't
 * root-absolute (already-absolute URLs — http://, https://, //, ws://,
 * wss:// — pass through untouched), or when `path` already carries the
 * prefix (avoids double-prefixing).
 * @param {string} path
 * @returns {string}
 */
export function withPrefix(path) {
  const prefix = window.__OSPREY_PREFIX__ || '';
  if (!prefix || !path.startsWith('/') || path.startsWith('//') || path.startsWith(prefix)) {
    return path;
  }
  return `${prefix}${path}`;
}

/**
 * Build a same-origin WebSocket URL with the scheme that matches the current
 * page: wss:// when served over HTTPS, ws:// otherwise. Pass a root-absolute
 * path such as '/ws/terminal'. Avoids mixed-content failures under TLS.
 * @param {string} path
 * @returns {string}
 */
export function wsUrl(path) {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}${withPrefix(path)}`;
}

/**
 * Create a WebSocket with exponential backoff reconnection.
 * @param {string} url
 * @param {WebSocketHandlers} [handlers]
 */
export function createWebSocket(url, { onOpen, onMessage, onClose, onError } = {}) {
  /** @type {WebSocket|null} */
  let ws = null;
  let attempt = 0;
  let stopped = false;
  let wsUrl = url;

  function connect() {
    if (stopped) return;
    wsState = 'connecting';
    notifyStateChange();

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      attempt = 0;
      wsState = 'connected';
      notifyStateChange();
      if (onOpen) onOpen(/** @type {WebSocket} */ (ws));
    };

    ws.onmessage = (e) => {
      if (onMessage) onMessage(e);
    };

    ws.onclose = (e) => {
      wsState = 'disconnected';
      notifyStateChange();
      if (onClose) onClose(e);
      scheduleReconnect();
    };

    ws.onerror = (e) => {
      if (onError) onError(e);
    };
  }

  function scheduleReconnect() {
    if (stopped) return;
    const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
    attempt++;
    setTimeout(connect, delay);
  }

  /** @param {string|ArrayBufferLike|Blob|ArrayBufferView} data */
  function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(data);
    }
  }

  function stop() {
    stopped = true;
    if (ws) ws.close();
  }

  /** @param {string} newUrl */
  function setUrl(newUrl) {
    wsUrl = newUrl;
  }

  connect();
  return { send, stop, setUrl, get ws() { return ws; } };
}

/**
 * Create an EventSource with reconnection.
 * @param {string} url
 * @param {EventSourceHandlers} [handlers]
 */
export function createEventSource(url, { onMessage, onError } = {}) {
  /** @type {EventSource|null} */
  let es = null;
  let stopped = false;

  function connect() {
    if (stopped) return;
    sseState = 'connecting';
    notifyStateChange();

    es = new EventSource(withPrefix(url));

    es.onopen = () => {
      sseState = 'connected';
      notifyStateChange();
    };

    es.onmessage = (e) => {
      if (onMessage) {
        try {
          const data = JSON.parse(e.data);
          onMessage(data);
        } catch {
          onMessage(e.data);
        }
      }
    };

    es.onerror = () => {
      sseState = 'disconnected';
      notifyStateChange();
      if (onError) onError();
      // EventSource auto-reconnects, but update state
    };
  }

  function stop() {
    stopped = true;
    if (es) es.close();
    sseState = 'disconnected';
    notifyStateChange();
  }

  connect();
  return { stop };
}

/**
 * Fetch JSON from a URL.
 * @param {string} url
 * @returns {Promise<any>}
 */
export async function fetchJSON(url) {
  const res = await fetch(withPrefix(url), { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}

/**
 * JSON API request through the {@link withPrefix} chokepoint. Covers the
 * mutating verbs (POST/PUT/PATCH/DELETE) that fetchJSON's GET contract
 * doesn't: serializes `json` as the request body, and on a non-OK response
 * throws an Error carrying the server's `detail` message when the error body
 * has one, else `"<errorPrefix> (HTTP <status>)"`. Resolves with the parsed
 * JSON response body (null when the body isn't JSON, e.g. empty DELETE
 * responses).
 * @param {string} url
 * @param {{method?: string, json?: any, errorPrefix?: string}} [opts]
 * @returns {Promise<any>}
 */
export async function apiRequest(url, { method = 'POST', json, errorPrefix = 'Request failed' } = {}) {
  /** @type {RequestInit} */
  const init = { method };
  if (json !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(json);
  }
  const resp = await fetch(withPrefix(url), init);
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `${errorPrefix} (HTTP ${resp.status})`);
  }
  return resp.json().catch(() => null);
}
