// @ts-check
/**
 * The OPTIMIZER panel's one door to its sidecar.
 *
 * The panel is reached through the web terminal's reverse proxy at
 * `/panel/{id}/`, which does not rewrite root-absolute API paths, so every
 * request is built as `<panel prefix>/<path>` (the prefix is `""` when the
 * sidecar serves the panel directly). Paths here never start with `/`.
 *
 * Reads go to the sidecar's read proxy (`environments/...`, `jobs/...`);
 * writes go to `relay/...`, where the sidecar adds the owner and launch token
 * server-side — this module never sends either.
 *
 * Every call resolves (never rejects) to an {@link ApiResult}: a refusal's
 * `detail` is the service's own sentence, relayed verbatim.
 *
 * @module api-client
 */

/** The header `GET jobs/{id}/config` carries the draft's revision in. */
export const REVISION_HEADER = 'X-Config-Revision';

/**
 * @typedef {object} ApiResult
 * @property {boolean} ok
 * @property {number} status  HTTP status; 0 when the sidecar could not be reached.
 * @property {any} data       Parsed JSON body, or null.
 * @property {string|null} revision  The `X-Config-Revision` header, when present.
 * @property {string|null} error     Operator-facing reason when `ok` is false.
 * @property {string[]} issues  Each reason on its own when `ok` is false (one per
 *   item of a 422 validation list, else `error` alone); empty when `ok` is true.
 */

/**
 * @typedef {Record<string, string|number|boolean|null|undefined|readonly string[]>} Query
 */

/**
 * Append a query string. Array values repeat the key (`names=a&names=b`, the
 * shape FastAPI's `list[str]` query reads); null/undefined values are dropped.
 *
 * @param {string} path
 * @param {Query} [query]
 * @returns {string}
 */
export function withQuery(path, query) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query || {})) {
    if (value === null || value === undefined) continue;
    if (Array.isArray(value)) {
      for (const item of value) params.append(key, String(item));
    } else {
      params.append(key, String(value));
    }
  }
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

/**
 * Each operator-facing reason in an error body: FastAPI's `detail` string, or
 * one `loc: msg` per item of a 422 validation list, or the status line.
 *
 * @param {number} status
 * @param {any} data
 * @returns {string[]}
 */
export function errorIssues(status, data) {
  const detail = data && typeof data === 'object' ? data.detail : null;
  if (typeof detail === 'string' && detail) return [detail];
  if (Array.isArray(detail) && detail.length) {
    return detail.map((item) => {
      const loc = Array.isArray(item?.loc)
        ? item.loc.filter((/** @type {unknown} */ p) => p !== 'body').join('.')
        : '';
      return loc ? `${loc}: ${item?.msg ?? ''}` : String(item?.msg ?? JSON.stringify(item));
    });
  }
  return [`request failed (HTTP ${status})`];
}

/**
 * Encode one path segment (an environment name or job id).
 *
 * @param {string} segment
 * @returns {string}
 */
export function seg(segment) {
  return encodeURIComponent(segment);
}

/**
 * The numeric value of each PVReading in a `GET .../variables` body.
 *
 * @param {Record<string, unknown>|null} readings
 * @returns {Record<string, number|null>}
 */
export function readbackValues(readings) {
  return Object.fromEntries(
    Object.entries(readings || {}).map(([name, r]) => {
      const v = /** @type {any} */ (r)?.value;
      return [name, typeof v === 'number' && Number.isFinite(v) ? v : null];
    })
  );
}

/**
 * The owner a `relay/session` answer names: `ownerKnown` is `false` when it
 * named nobody, `null` when the read failed or answered something else.
 *
 * @param {{ ok: boolean, data: any }} result
 * @returns {{ owner: string|null, ownerKnown: boolean|null }}
 */
export function sessionOwner(result) {
  const owner =
    result.ok && result.data && typeof result.data === 'object' ? result.data.owner : undefined;
  if (typeof owner === 'string' && owner) return { owner, ownerKnown: true };
  return { owner: null, ownerKnown: owner === null ? false : null };
}

/**
 * The sidecar path of one stored run. A run id is `<environment>/<timestamp>`
 * (`ResultInfo.run_id`); each part is one path segment.
 *
 * @param {string} runId
 * @returns {string|null}  null for an id with no `/`
 */
export function resultPath(runId) {
  const cut = runId.lastIndexOf('/');
  if (cut <= 0 || cut === runId.length - 1) return null;
  return `results/${seg(runId.slice(0, cut))}/${seg(runId.slice(cut + 1))}`;
}

/**
 * @param {{ prefix: string, fetchImpl?: typeof fetch }} options
 */
export function createApi({ prefix, fetchImpl }) {
  const doFetch = fetchImpl || ((input, init) => fetch(input, init));

  /**
   * @param {string} method
   * @param {string} path
   * @param {{ query?: Query, body?: unknown }} [options]
   * @returns {Promise<ApiResult>}
   */
  async function request(method, path, { query, body } = {}) {
    /** @type {RequestInit} */
    const init = { method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      init.headers = { ...init.headers, 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await doFetch(`${prefix}/${withQuery(path, query)}`, init);
    } catch {
      const error = 'the panel could not reach its sidecar';
      return { ok: false, status: 0, data: null, revision: null, error, issues: [error] };
    }
    let data = null;
    try {
      const text = await response.text();
      data = text ? JSON.parse(text) : null;
    } catch {
      data = null;
    }
    const issues = response.ok ? [] : errorIssues(response.status, data);
    return {
      ok: response.ok,
      status: response.status,
      data,
      revision: response.headers.get(REVISION_HEADER),
      error: response.ok ? null : issues.join('; '),
      issues,
    };
  }

  return {
    /**
     * @param {string} path
     * @param {Query} [query]
     */
    get: (path, query) => request('GET', path, { query }),
    /**
     * @param {string} path
     * @param {unknown} [body]
     * @param {Query} [query]
     */
    post: (path, body, query) => request('POST', path, { body, query }),
    /**
     * @param {string} path
     * @param {unknown} body
     */
    put: (path, body) => request('PUT', path, { body }),
    /** The owner this session's writes carry, as the gate minted it ({@link sessionOwner}). */
    session: async () => sessionOwner(await request('GET', 'relay/session')),
    /** `GET jobs`: every job that has not ended (drafts and the active run). */
    jobs: () => request('GET', 'jobs'),
    /** `GET results`: one `ResultInfo` per stored run. */
    results: () => request('GET', 'results'),
    /**
     * `GET results/{env}/{timestamp}`: one stored run (`HistoricalRunDetails`).
     *
     * @param {string} runId  `ResultInfo.run_id`
     * @returns {Promise<ApiResult>}
     */
    result: async (runId) => {
      const path = resultPath(runId);
      if (path) return request('GET', path);
      const error = `no stored run ${runId}`;
      return { ok: false, status: 0, data: null, revision: null, error, issues: [error] };
    },
    /**
     * `GET environments/{name}/observables`: what the observables read now.
     *
     * @param {string} environment
     * @param {readonly string[]} names
     * @param {string} target
     */
    observables: (environment, names, target) =>
      request('GET', `environments/${seg(environment)}/observables`, {
        query: { names, target },
      }),
  };
}

/** @typedef {ReturnType<typeof createApi>} Api */

/**
 * Save a draft — over `jobId`, or as a new one when it is null — and read
 * back what the service stored (it may snap or clamp bounds). The job id is
 * the caller's, captured before the write, never re-read from shared state.
 *
 * @param {Api} api
 * @param {string|null} jobId
 * @param {unknown} config
 * @returns {Promise<{ error: string|null, issues: string[], id: string|null, stored: any,
 *   revision: string|null }>}  `issues`: each reason of a refusal on its own (one per item of
 *   a 422 list); `revision` is the `X-Config-Revision` of the read-back
 */
export async function saveDraft(api, jobId, config) {
  const res = jobId
    ? await api.put(`relay/jobs/${seg(jobId)}/config`, config)
    : await api.post('relay/optimizations', config);
  if (!res.ok) {
    const error = res.error || 'the draft was refused';
    const issues = res.issues?.length ? res.issues : [error];
    return { error, issues, id: null, stored: null, revision: null };
  }
  const id = jobId || res.data?.job_id || null;
  const back = id ? await api.get(`jobs/${seg(id)}/config`) : null;
  return {
    error: null,
    issues: [],
    id,
    stored: back?.ok ? back.data : null,
    revision: back?.ok ? back.revision : null,
  };
}
