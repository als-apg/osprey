/* OSPREY Web Terminal — Connection Helpers */

/** @type {'connected'|'connecting'|'disconnected'} */
let wsState = 'disconnected';
/** @type {'connected'|'connecting'|'disconnected'} */
let sseState = 'disconnected';

const stateListeners = [];

function notifyStateChange() {
  for (const fn of stateListeners) fn({ ws: wsState, sse: sseState });
}

export function onConnectionStateChange(fn) {
  stateListeners.push(fn);
}

export function getConnectionState() {
  return { ws: wsState, sse: sseState };
}

/**
 * Create a WebSocket with exponential backoff reconnection.
 */
export function createWebSocket(url, { onOpen, onMessage, onClose, onError } = {}) {
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
      if (onOpen) onOpen(ws);
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

  function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(data);
    }
  }

  function stop() {
    stopped = true;
    if (ws) ws.close();
  }

  function setUrl(newUrl) {
    wsUrl = newUrl;
  }

  connect();
  return { send, stop, setUrl, get ws() { return ws; } };
}

/**
 * Create an EventSource with reconnection.
 */
export function createEventSource(url, { onMessage, onError } = {}) {
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
 */
export async function fetchJSON(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}
