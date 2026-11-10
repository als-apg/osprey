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
 * One operator-facing sentence from an error body: FastAPI's `detail` string,
 * or the messages of a 422 validation list, or the status line.
 *
 * @param {number} status
 * @param {any} data
 * @returns {string}
 */
export function errorDetail(status, data) {
  const detail = data && typeof data === 'object' ? data.detail : null;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((item) => {
        const loc = Array.isArray(item?.loc)
          ? item.loc.filter((/** @type {unknown} */ p) => p !== 'body').join('.')
          : '';
        return loc ? `${loc}: ${item?.msg ?? ''}` : String(item?.msg ?? JSON.stringify(item));
      })
      .join('; ');
  }
  return `request failed (HTTP ${status})`;
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
      return {
        ok: false,
        status: 0,
        data: null,
        revision: null,
        error: 'the panel could not reach its sidecar',
      };
    }
    let data = null;
    try {
      const text = await response.text();
      data = text ? JSON.parse(text) : null;
    } catch {
      data = null;
    }
    return {
      ok: response.ok,
      status: response.status,
      data,
      revision: response.headers.get(REVISION_HEADER),
      error: response.ok ? null : errorDetail(response.status, data),
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
  };
}

/** @typedef {ReturnType<typeof createApi>} Api */
