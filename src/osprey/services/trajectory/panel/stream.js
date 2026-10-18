// @ts-check
/**
 * stream — the panel's one live connection to the trajectory service.
 *
 * The service frames every event as `event: <type>` plus `data: <json>`, so a
 * plain `onmessage` never fires; each event type is subscribed by name with
 * `addEventListener` and its parsed payload handed to `onEvent(type, payload)`.
 *
 * The URL is relative (`api/events`): the panel bundle is served under
 * `/panel/`, so a page-relative URL reaches the panel door, while a
 * root-absolute `/api/...` would reach the agent door.
 *
 * Reconnection follows the terminal's shared stream: while the source is
 * CONNECTING the browser's built-in retry is still running and is never raced;
 * once it is CLOSED (a proxy 502 closes it for good) the source is reopened
 * with exponential backoff, capped. The connection state is reported through
 * `onState` — `connected` once open, `disconnected` from the first error until
 * the next open.
 *
 * `EventSourceCtor` is injectable so tests drive a fake source (happy-dom has
 * no `EventSource`).
 *
 * @module stream
 */

/** Page-relative URL of the panel door's event stream. */
export const EVENTS_URL = 'api/events';

/** Every event type the service sends; each is its own SSE `event:` name. */
export const EVENT_TYPES = Object.freeze([
  'snapshot',
  'shot',
  'step',
  'apply',
  'stopped',
  'taken_over',
  'continuous_ended',
]);

/** EventSource readyState CLOSED (spec constant). */
const CLOSED = 2;

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

/** @typedef {'connecting' | 'connected' | 'disconnected'} StreamState */

/**
 * The slice of `EventSource` this module uses, so a test fake satisfies it.
 * @typedef {object} EventSourceLike
 * @property {number} readyState
 * @property {((ev: Event) => void) | null} onopen
 * @property {((ev: Event) => void) | null} onerror
 * @property {(type: string, listener: (ev: MessageEvent) => void) => void} addEventListener
 * @property {() => void} close
 */

/** @typedef {new (url: string) => EventSourceLike} EventSourceCtorLike */

/**
 * @typedef {object} StreamOptions
 * @property {(type: string, payload: unknown) => void} onEvent
 *   Called with the event type and its parsed JSON payload.
 * @property {(state: StreamState) => void} [onState]
 *   Called whenever the connection state changes.
 * @property {EventSourceCtorLike} [EventSourceCtor]
 * @property {string} [url]
 */

/**
 * @typedef {object} StreamHandle
 * @property {() => void} close  Stop the stream and any pending reconnect.
 */

/**
 * Open the panel's event stream.
 * @param {StreamOptions} options
 * @returns {StreamHandle}
 */
export function openStream({ onEvent, onState, EventSourceCtor, url = EVENTS_URL }) {
  const Ctor = EventSourceCtor ?? /** @type {EventSourceCtorLike} */ (globalThis.EventSource);
  /** @type {EventSourceLike | null} */
  let source = null;
  let stopped = false;
  let attempt = 0;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let reconnectTimer = null;
  /** @type {StreamState | null} */
  let state = null;

  /** @param {StreamState} next */
  function report(next) {
    if (next === state) return;
    state = next;
    if (onState) onState(next);
  }

  function connect() {
    if (stopped) return;
    report(state === 'disconnected' ? 'disconnected' : 'connecting');
    const es = new Ctor(url);
    source = es;
    es.onopen = () => {
      if (es !== source) return;
      attempt = 0;
      report('connected');
    };
    es.onerror = () => {
      if (stopped || es !== source) return;
      report('disconnected');
      // While CONNECTING the browser's own retry is running; reconnection is
      // ours only once the source is CLOSED for good.
      if (es.readyState === CLOSED) scheduleReconnect();
    };
    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (ev) => {
        if (stopped || es !== source) return;
        let payload;
        try {
          payload = JSON.parse(ev.data);
        } catch {
          return;
        }
        onEvent(type, payload);
      });
    }
  }

  function scheduleReconnect() {
    if (stopped || reconnectTimer != null) return;
    const delay = Math.min(INITIAL_BACKOFF_MS * 2 ** attempt, MAX_BACKOFF_MS);
    attempt++;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (source) source.close();
      connect();
    }, delay);
  }

  connect();

  return {
    close() {
      stopped = true;
      if (reconnectTimer != null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (source) source.close();
    },
  };
}
