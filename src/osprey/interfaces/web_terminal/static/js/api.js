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
 * @property {() => void} [onOpen]
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
 * Set once a probe has seen a 401, i.e. the session behind this page is gone.
 * A reload is already on its way, so every reconnect loop stops here.
 */
let sessionExpired = false;

/** In-flight session probe, shared by all channels so closes don't fan out. */
/** @type {Promise<boolean>|null} */
let sessionProbe = null;

/**
 * Ask the app's own session route whether this page still has a session,
 * resolving true only on a definite 401.
 *
 * The browser WebSocket API hides handshake status codes from JavaScript, so
 * a connection refused with 401 reaches `onclose` looking exactly like a
 * dropped network — without this probe an expired session leaves the terminal
 * reconnecting forever. Three details are load-bearing:
 *
 * - The URL is the app's own `/api/session` (via {@link withPrefix} so it lands
 *   inside the per-user mount when one exists). Unlike `/health` — which is
 *   exempt from the auth gate so the container healthcheck can curl it without a
 *   credential, and therefore always answers 200 — `/api/session` sits behind
 *   the app's auth gate. That gate is what makes the 401 visible, and it lives
 *   in the app itself: single-user has no nginx perimeter, so a stale cookie
 *   would otherwise loop forever; now the app reports it expired.
 * - `Accept: application/json`. Both gates content-negotiate their 401: an
 *   Accept mentioning text/html gets a login redirect (nginx) or the app gate's
 *   HTML "not signed in" page, anything else gets a bare JSON 401. We want the
 *   status, not the page — and asking for JSON is what keeps this probe's
 *   answer a status code even after the reload it triggers gets the page.
 * - The frontend decides nothing about authorization and holds no secret. It
 *   reads one status code and reacts.
 *
 * Anything that is not a 401 — a network error, a 5xx, a healthy 200 — resolves
 * false and leaves the caller's reconnect/backoff behavior untouched. Where no
 * auth fronts the deployment (dev without credentials), the probe simply always
 * sees the app's own 200.
 * @returns {Promise<boolean>}
 */
function probeSessionExpired() {
  if (sessionExpired) return Promise.resolve(true);
  if (!sessionProbe) {
    sessionProbe = fetch(withPrefix('/api/session'), {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    })
      .then((res) => res.status === 401)
      .catch(() => false)
      .finally(() => {
        sessionProbe = null;
      });
  }
  return sessionProbe;
}

/**
 * Probe after a channel closed and, on an expired session, reload the page.
 *
 * The reload is a *navigation*, and that is the whole point: a navigation
 * carries `Accept: text/html`, which is what both shapes answer usefully.
 * Multi-user reaches the nginx perimeter first and gets its login redirect;
 * single-user has no perimeter, so it reaches the app's own auth gate, which
 * content-negotiates its 401 into a readable "this session is not signed in"
 * page telling the operator to re-open the login URL their server printed.
 * Either way the operator sees an explanation instead of a terminal that
 * reconnects forever — and neither answer is something this module has to know
 * about or render itself.
 *
 * The probe above deliberately asks for `application/json` so it keeps reading
 * a status code rather than being handed that page.
 *
 * Fire-and-forget: callers schedule their ordinary reconnect first, so a probe
 * that fails for any other reason costs nothing but one request.
 * @param {() => void} [onExpired] tear down the caller's own handle first
 *   (an EventSource would otherwise keep retrying until the reload lands)
 */
function reloadIfSessionExpired(onExpired) {
  probeSessionExpired().then((expired) => {
    if (!expired) return;
    if (onExpired) onExpired();
    if (sessionExpired) return; // another channel already triggered the reload
    sessionExpired = true;
    location.reload();
  });
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
    if (stopped || sessionExpired) return;
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
      reloadIfSessionExpired();
    };

    ws.onerror = (e) => {
      if (onError) onError(e);
    };
  }

  function scheduleReconnect() {
    if (stopped || sessionExpired) return;
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
 *
 * The browser's built-in retry covers only network-level failures on a
 * still-live source; a non-2xx response or wrong content type (a proxy 502
 * while the backend restarts, an auth redirect) parks the EventSource in
 * CLOSED permanently, where the spec forbids any further retry. This wrapper
 * takes over from there with the same exponential backoff createWebSocket
 * uses, so the panel event channel survives backend restarts.
 *
 * `onOpen` fires on EVERY open — first connect, browser auto-retry, and this
 * wrapper's own reconnects. SSE has no replay (no event ids are sent), so
 * anything published while disconnected is gone; the hook is the caller's
 * seam for re-fetching authoritative state instead.
 * @param {string} url
 * @param {EventSourceHandlers} [handlers]
 */
export function createEventSource(url, { onMessage, onError, onOpen } = {}) {
  /** @type {EventSource|null} */
  let es = null;
  let stopped = false;
  let attempt = 0;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let reconnectTimer = null;

  function connect() {
    if (stopped || sessionExpired) return;
    sseState = 'connecting';
    notifyStateChange();

    es = new EventSource(withPrefix(url));

    es.onopen = () => {
      attempt = 0;
      sseState = 'connected';
      notifyStateChange();
      if (onOpen) onOpen();
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
      // readyState 2 is CLOSED (spec constant): the browser has given up on
      // this source for good, so reconnection is ours from here. While
      // CONNECTING/OPEN the built-in retry is still running — never race it.
      if (es && es.readyState === 2) scheduleReconnect();
      reloadIfSessionExpired(() => {
        if (es) es.close();
      });
    };
  }

  function scheduleReconnect() {
    if (stopped || sessionExpired || reconnectTimer != null) return;
    const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
    attempt++;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  }

  function stop() {
    stopped = true;
    if (reconnectTimer != null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
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
