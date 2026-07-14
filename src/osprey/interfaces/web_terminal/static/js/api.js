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
 * Build a same-origin WebSocket URL with the scheme that matches the current
 * page: wss:// when served over HTTPS, ws:// otherwise. Pass a root-absolute
 * path such as '/ws/terminal'. Avoids mixed-content failures under TLS.
 * @param {string} path
 * @returns {string}
 */
export function wsUrl(path) {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}${path}`;
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

    es = new EventSource(url);

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
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}
