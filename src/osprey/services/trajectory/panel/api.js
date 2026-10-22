// @ts-check
/**
 * api — the only place the panel builds requests to the trajectory service.
 *
 * Every URL is relative to the bundle (`api/<route>`), never root-absolute:
 * the bundle is served under `/panel/`, so a page-relative URL reaches the
 * panel door, while the terminal proxy rewrites a root-absolute `/api/...` to
 * the agent door and injects the launch token there, recording the call as
 * the agent's.
 *
 * Each call resolves to a {@link Result}, never throws:
 *  - `ok` — a 2xx answer with its parsed JSON body;
 *  - `refused` — a non-2xx answer whose body is a refusal
 *    `{detail: {code, detail, ...}}`; a 502's `moved` list and any other
 *    extras are kept;
 *  - `pending` — a transport error, a non-JSON body, or an error body that is
 *    not a refusal. The service may still have acted, so the outcome of record
 *    is the stream's events and the snapshot's `last_action`, never this call;
 *  - `busy` — an action was asked while another action POST is in flight; no
 *    request was sent.
 *
 * At most one action POST (shot, apply, put back, setup, solve, continuous
 * start) is in flight. Stop and the continuous heartbeat are never held back,
 * nor are session, reference and read calls.
 *
 * Each body carries only the keys its caller passed: a button sends its own
 * keys and nothing else, and an absent key leaves the service's value as it
 * is. The raw route answers bytes, decoded by {@link decodeRaw}.
 *
 * `fetch` and the session's client id are injected so tests drive a fake.
 *
 * @module api
 */

/** Header naming the panel's session on the routes that need it. */
export const CLIENT_HEADER = 'X-Trajectory-Client';

/** Message shown when a call's outcome is not known from its answer. */
export const PENDING_MESSAGE = 'outcome pending — see state';

/** `last_action` names after which Put back is offered. */
const PUT_BACK_AFTER = Object.freeze(['apply', 'setup']);

/**
 * @typedef {object} Moved
 * @property {string} name
 * @property {number | string | boolean} before
 * @property {string} outcome
 */

/**
 * @typedef {object} LastAction
 * @property {string} name  `shot`, `apply`, `put_back` or `setup`.
 * @property {boolean} stopped
 * @property {Moved[]} moved
 */

/**
 * The snapshot fields the put-back builder reads.
 * @typedef {object} PutBackSource
 * @property {string | null} [model]
 * @property {LastAction | null} [last_action]
 */

/**
 * @typedef {object} PutBackBody
 * @property {string} model
 * @property {{name: string, before: number | string | boolean}[]} values
 */

/**
 * @typedef {{kind: 'ok', status: number, body: any}} OkResult
 * @typedef {{kind: 'refused', status: number, code: string, detail: string,
 *   moved: Moved[] | null, extras: Record<string, unknown>}} RefusedResult
 * @typedef {{kind: 'pending', status: number | null, message: string}} PendingResult
 * @typedef {{kind: 'busy', message: string}} BusyResult
 * @typedef {OkResult | RefusedResult | PendingResult | BusyResult} Result
 */

/**
 * @typedef {(url: string, init: RequestInit) => Promise<Response>} FetchLike
 */

/**
 * A chain run on the panel door, also the body a continuous run starts with.
 * @typedef {object} ChainBody
 * @property {string[]} steps  The ticked step ids; the service runs them in chain order.
 * @property {'tbt' | 'pseudo_orbit'} [readout]
 * @property {number} [n_turns]  The turns a pseudo-orbit readout averages over.
 * @property {number} [n_shots]
 * @property {'mean' | 'nanmean'} [average]
 */

/**
 * A setup: a partial update, every absent key left as it is.
 * @typedef {object} SetupBody
 * @property {Record<string, number | string | boolean>} [settings]  Value per declared setting id.
 * @property {string[]} [excluded_bpms]
 */

/**
 * A SolveConfig update; an empty one recomputes.
 * @typedef {object} SolveConfigPartial
 * @property {number} [alpha]
 * @property {number} [keep_live]
 * @property {number} [gain]
 * @property {string[]} [excluded_bpms]
 * @property {string[]} [excluded_cms]
 * @property {number} [loss_cutoff]
 */

/**
 * A reference file's content: the import body and the readings route's answer.
 * @typedef {object} ReferenceImport
 * @property {string} name
 * @property {Record<string, {x: number | null, y: number | null}>} readings
 * @property {'tbt' | 'pseudo_orbit'} readout
 * @property {boolean} [select]
 * @property {string | null} [source_target]
 * @property {string | null} [source_model]
 */

/**
 * @typedef {import('./figures.js').RawData} RawData
 */

/**
 * @typedef {object} ApiOptions
 * @property {() => string | null} [getClientId]
 *   The session's client id, or null while the panel holds no session.
 * @property {FetchLike} [fetchImpl]  Defaults to the global `fetch`.
 * @property {(refusal: RefusedResult) => void} [onRefused]
 *   Sees every refusal, whichever caller made the call: the session drops
 *   its id on a lost-session code here.
 */

/**
 * @typedef {object} RequestSpec
 * @property {'GET' | 'POST' | 'DELETE'} method
 * @property {string} route  Path below `api/`, without a leading slash.
 * @property {object} [body]  Sent as JSON; omitted when absent.
 * @property {boolean} [client]  Send {@link CLIENT_HEADER}.
 * @property {boolean} [action]  Subject to the one-action-in-flight rule.
 * @property {(response: Response) => Promise<unknown>} [read]
 *   Reads a 2xx answer's body; JSON when absent.
 */

/**
 * Whether Put back is offered after this `last_action`: only after an apply
 * or a setup (stopped or not) that moved something — never after a put-back,
 * which would re-write the step's values without its digest, nor a shot.
 * @param {LastAction | null | undefined} lastAction
 * @returns {boolean}
 */
export function canPutBack(lastAction) {
  return (
    lastAction != null &&
    PUT_BACK_AFTER.includes(lastAction.name) &&
    Array.isArray(lastAction.moved) &&
    lastAction.moved.length > 0
  );
}

/**
 * The `POST apply` body that puts back what the last apply or setup moved:
 * one entry per setting, holding the earliest `before` when a setting moved
 * more than once. Null when Put back is not offered or no model is loaded.
 * @param {PutBackSource} snapshot
 * @returns {PutBackBody | null}
 */
export function putBackBody(snapshot) {
  const lastAction = snapshot.last_action;
  if (!canPutBack(lastAction) || !lastAction || !snapshot.model) {
    return null;
  }
  /** @type {Map<string, number | string | boolean>} */
  const earliest = new Map();
  for (const { name, before } of lastAction.moved) {
    if (!earliest.has(name)) {
      earliest.set(name, before);
    }
  }
  return {
    model: snapshot.model,
    values: [...earliest].map(([name, before]) => ({ name, before })),
  };
}

/**
 * Read a refusal out of an error body, or null when it is not one.
 * @param {unknown} body
 * @param {number} status
 * @returns {RefusedResult | null}
 */
export function parseRefusal(body, status) {
  if (body === null || typeof body !== 'object') {
    return null;
  }
  const detail = /** @type {Record<string, unknown>} */ (body).detail;
  if (detail === null || typeof detail !== 'object' || Array.isArray(detail)) {
    return null;
  }
  const { code, detail: text, moved, ...extras } = /** @type {Record<string, unknown>} */ (detail);
  if (typeof code !== 'string' || typeof text !== 'string') {
    return null;
  }
  return {
    kind: 'refused',
    status,
    code,
    detail: text,
    moved: Array.isArray(moved) ? /** @type {Moved[]} */ (moved) : null,
    extras,
  };
}

/**
 * @param {number | null} status
 * @returns {PendingResult}
 */
function pending(status) {
  return { kind: 'pending', status, message: PENDING_MESSAGE };
}

/**
 * Decode a raw buffer as the raw route serves it: a little-endian uint32
 * header length, the JSON header, then float32 samples from a 4-byte boundary.
 * @param {ArrayBuffer} buffer
 * @returns {RawData}
 */
export function decodeRaw(buffer) {
  const length = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length)));
  return { header, samples: new Float32Array(buffer, 4 + length) };
}

/**
 * A copy of *body* without its undefined keys, so only what a caller passed is sent.
 * @param {object} body
 * @returns {Record<string, unknown>}
 */
function only(body) {
  return Object.fromEntries(Object.entries(body).filter(([, value]) => value !== undefined));
}

/** @param {Response} response */
async function readRaw(response) {
  return decodeRaw(await response.arrayBuffer());
}

/**
 * Build the panel's API client.
 * @param {ApiOptions} [options]
 */
export function createApi({ getClientId = () => null, fetchImpl, onRefused } = {}) {
  let actionInFlight = false;

  /**
   * @param {RequestSpec} spec
   * @returns {Promise<Result>}
   */
  async function send({ method, route, body, client = false, action = false, read }) {
    if (action && actionInFlight) {
      return { kind: 'busy', message: 'another action is still running' };
    }
    /** @type {Record<string, string>} */
    const headers = { Accept: read ? '*/*' : 'application/json' };
    if (client) {
      const id = getClientId();
      if (id) {
        headers[CLIENT_HEADER] = id;
      }
    }
    /** @type {RequestInit} */
    const init = { method, headers };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    const doFetch = fetchImpl ?? globalThis.fetch;
    if (action) {
      actionInFlight = true;
    }
    try {
      /** @type {Response} */
      let response;
      try {
        response = await doFetch(`api/${route}`, init);
      } catch {
        return pending(null);
      }
      /** @type {unknown} */
      let parsed;
      try {
        parsed = response.ok && read ? await read(response) : await response.json();
      } catch {
        return pending(response.status);
      }
      if (response.ok) {
        return { kind: 'ok', status: response.status, body: parsed };
      }
      const refusal = parseRefusal(parsed, response.status);
      if (refusal == null) {
        return pending(response.status);
      }
      onRefused?.(refusal);
      return refusal;
    } finally {
      if (action) {
        actionInFlight = false;
      }
    }
  }

  /** @param {string} name */
  const ref = (name) => `references/${encodeURIComponent(name)}`;

  return {
    /** Whether an action POST is in flight. */
    get actionInFlight() {
      return actionInFlight;
    },

    openSession: () => send({ method: 'POST', route: 'session' }),
    takeOver: () => send({ method: 'POST', route: 'takeover' }),
    release: () => send({ method: 'DELETE', route: 'session', client: true }),
    state: () => send({ method: 'GET', route: 'state' }),
    stop: () => send({ method: 'POST', route: 'stop' }),

    /** @param {ChainBody} body  The ticked chain, run once per injection. */
    shot: (body) => send({ method: 'POST', route: 'shot', body: only(body), client: true, action: true }),
    step: () => send({ method: 'GET', route: 'step' }),

    /** @param {string} stepDigest */
    apply: (stepDigest) =>
      send({
        method: 'POST',
        route: 'apply',
        body: { step_digest: stepDigest },
        client: true,
        action: true,
      }),

    /** @param {PutBackBody} body  As built by {@link putBackBody}. */
    putBack: (body) =>
      send({ method: 'POST', route: 'apply', body, client: true, action: true }),

    /** @param {SetupBody} body */
    setup: (body) => send({ method: 'POST', route: 'setup', body: only(body), client: true, action: true }),
    /** The typed `SetupReadbacks`. */
    setupReadbacks: () => send({ method: 'GET', route: 'setup' }),

    /** @param {SolveConfigPartial} [partial]  Empty recomputes. */
    solve: (partial = {}) =>
      send({ method: 'POST', route: 'solve', body: only(partial), client: true, action: true }),

    /** @param {ChainBody} body  The chain each iteration runs. */
    continuous: (body) =>
      send({ method: 'POST', route: 'continuous', body: only(body), client: true, action: true }),
    heartbeat: () => send({ method: 'POST', route: 'continuous/heartbeat', client: true }),

    references: () => send({ method: 'GET', route: 'references' }),

    /**
     * @param {string} name
     * @param {string} [shotId]  When omitted, the step's shot, else the newest.
     */
    saveReference: (name, shotId) =>
      send({ method: 'POST', route: 'references', body: only({ name, shot_id: shotId }) }),

    /** @param {string} name */
    loadReference: (name) => send({ method: 'GET', route: ref(name) }),

    /** @param {string} name  Answers exactly the import body. */
    referenceReadings: (name) => send({ method: 'GET', route: `${ref(name)}/readings` }),

    /** @param {ReferenceImport} body  Selecting it needs the holder. */
    importReference: (body) =>
      send({ method: 'POST', route: 'references/import', body: only(body), client: true }),

    /** @param {string | null} name  Null selects no reference. */
    selectReference: (name) =>
      send({ method: 'POST', route: 'references/select', body: { name }, client: true }),

    shots: () => send({ method: 'GET', route: 'shots' }),

    /**
     * A shot's raw buffer, decoded; a derived shot answers its root's.
     * @param {string} shotId
     */
    shotRaw: (shotId) =>
      send({ method: 'GET', route: `shots/${encodeURIComponent(shotId)}/raw`, read: readRaw }),
  };
}

/** @typedef {ReturnType<typeof createApi>} Api */
