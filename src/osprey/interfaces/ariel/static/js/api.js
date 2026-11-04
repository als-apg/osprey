// @ts-check
/**
 * ARIEL API Client
 *
 * HTTP client for communicating with the ARIEL backend API.
 */

import { normalizeFilter } from './filter-state.js';

const API_BASE = '/api';

/**
 * One enabled enhancement module's inspector card for one entry (ModuleCard).
 * @typedef {Object} ModuleCard
 * @property {string} module
 * @property {string} label
 * @property {number} order
 * @property {string[]} feeds
 * @property {string} state - complete | pending | stale | failed | n/a
 * @property {?string} [completed_at]
 * @property {?string} [failed_at]
 * @property {?string} [marker]
 * @property {?string} [error]
 * @property {?number} [attempts]
 * @property {?string} [command] - Re-run command for a failed or stale module
 * @property {?Record<string, any>} [detail] - The module's own inspect() output
 */

/**
 * @typedef {Object} EnhancementsResponse
 * @property {string} entry_id
 * @property {ModuleCard[]} modules
 */

/**
 * @typedef {Object} RelatedResponse
 * @property {'ok'|'no_embedding'|'disabled'} state
 * @property {any[]} entries - List rows; never include the entry itself
 */

/**
 * Per-module entry counts by badge-strip state, as `/status` carries them
 * (`StatusResponse.enhancement_coverage`).
 * @typedef {Record<string, {label: string, [state: string]: any}>} EnhancementCoverage
 */

/**
 * Per-request options.
 * @typedef {Object} RequestOptions
 * @property {AbortSignal} [signal] - Aborting it rejects the request with an
 *   `AbortError` (a DOMException, not an ApiError), which callers ignore.
 */

/**
 * The filter as the wire carries it: canonical JSON text, or undefined when it
 * selects nothing, so an empty filter is never sent.
 * @param {unknown} filters - A C-FILTER object
 * @returns {string|undefined}
 */
function filtersParam(filters) {
  const f = normalizeFilter(filters);
  return Object.keys(f).length ? JSON.stringify(f) : undefined;
}

/**
 * Advanced params as query-string JSON, or undefined when there are none.
 * @param {Record<string, any>|null|undefined} params
 * @returns {string|undefined}
 */
function advancedParamsParam(params) {
  return params && Object.keys(params).length ? JSON.stringify(params) : undefined;
}

/**
 * Join a path relative to the API base onto the API base, the same base every
 * call in this module uses. A proxy that mounts this page under a prefix
 * rewrites that base, so the result points under the prefix too.
 * @param {string} path - Path relative to the API base, starting with '/'
 * @returns {string} The URL path the browser should request
 */
export function apiUrl(path) {
  return API_BASE + path;
}

/**
 * Structured API error carrying the HTTP status code and an optional
 * machine-readable `code` discriminator from the response body.
 */
export class ApiError extends Error {
  /**
   * @param {number} status - HTTP status code.
   * @param {string} message - Error detail message.
   * @param {string} [code] - Machine-readable discriminator (e.g. "auth_required").
   * @param {string} [field] - Name of the entry field the error is about, if any.
   */
  constructor(status, message, code, field) {
    super(message);
    /** @type {number} */
    this.status = status;
    /** @type {string|undefined} */
    this.code = code;
    /** @type {string|undefined} */
    this.field = field;
    this.name = 'ApiError';
  }
}

/**
 * Build an ApiError from a non-OK response.
 *
 * Preserves the HTTP `status`, any machine-readable `code` discriminator from
 * the JSON body (e.g. "auth_required") and the entry `field` it names, so
 * callers can branch on them — for example, to prompt for logbook credentials
 * or mark the offending input instead of showing a generic error.
 * @param {Response} response - The failed fetch response
 * @returns {Promise<ApiError>} An ApiError with `.status`, optional `.code` and `.field`
 */
async function errorFromResponse(response) {
  const body = await response.json().catch(() => ({}));
  // A refused filter answers 422 with `detail: {field, reason}`.
  const detail = body.detail;
  if (detail && typeof detail === 'object' && !Array.isArray(detail) && 'reason' in detail) {
    return new ApiError(response.status, String(detail.reason), body.code, detail.field);
  }
  return new ApiError(
    response.status,
    (typeof detail === 'string' && detail) || `HTTP ${response.status}`,
    body.code,
    body.field
  );
}

/**
 * Issue a JSON-body request, shared by `post` and `put` (which only differ
 * in HTTP method).
 * @param {string} method - HTTP method ('POST', 'PUT')
 * @param {string} endpoint - API endpoint
 * @param {any} data - Request body
 * @param {RequestOptions} [options]
 * @returns {Promise<any>} Response data
 */
async function jsonRequest(method, endpoint, data, { signal } = {}) {
  const response = await fetch(API_BASE + endpoint, {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
    signal,
  });

  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return response.json();
}

/**
 * API client with error handling and response parsing.
 */
export const api = {
  /**
   * Make a GET request.
   * @param {string} endpoint - API endpoint
   * @param {any} params - Query parameters
   * @param {RequestOptions} [options]
   * @returns {Promise<any>} Response data
   */
  async get(endpoint, params = {}, { signal } = {}) {
    const url = new URL(API_BASE + endpoint, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        url.searchParams.append(key, value);
      }
    });

    const response = await fetch(url.toString(), { signal });
    if (!response.ok) {
      throw await errorFromResponse(response);
    }
    return response.json();
  },

  /**
   * Make a POST request.
   * @param {string} endpoint - API endpoint
   * @param {any} data - Request body
   * @param {RequestOptions} [options]
   * @returns {Promise<any>} Response data
   */
  async post(endpoint, data = {}, options = {}) {
    return jsonRequest('POST', endpoint, data, options);
  },

  /**
   * Make a PUT request.
   * @param {string} endpoint - API endpoint
   * @param {any} data - Request body
   * @returns {Promise<any>} Response data
   */
  async put(endpoint, data = {}) {
    return jsonRequest('PUT', endpoint, data);
  },
};

/**
 * Capabilities API — discover available modes and parameters.
 */
export const capabilitiesApi = {
  /**
   * Fetch available search modes and their tunable parameters.
   * @returns {Promise<any>} Capabilities response
   */
  async get() {
    return api.get('/capabilities');
  },
};

/**
 * Search API functions.
 */
export const searchApi = {
  /**
   * Execute a search query. `filters` (C-FILTER) is sent only when it selects
   * something; `offset` and `view` only when given.
   * @param {{query: string, mode?: string, maxResults?: number,
   *   advancedParams?: Record<string, any>, filters?: unknown, offset?: number,
   *   view?: 'full'|'row', signal?: AbortSignal}} params - Search parameters
   * @returns {Promise<any>} Search results
   */
  async search(params) {
    /** @type {Record<string, any>} */
    const request = {
      query: params.query,
      mode: params.mode || 'keyword',
      max_results: params.maxResults || 10,
      advanced_params: params.advancedParams || {},
    };
    const filters = normalizeFilter(params.filters);
    if (Object.keys(filters).length) request.filters = filters;
    if (params.offset !== undefined && params.offset !== null) request.offset = params.offset;
    if (params.view) request.view = params.view;

    return api.post('/search', request, { signal: params.signal });
  },
};

/**
 * Facets API — counts per filter dimension and the day histogram.
 */
export const facetsApi = {
  /**
   * Fetch facet counts for a query (or for browse when it is empty). `filters`
   * and `advancedParams` travel as JSON text and are left out when empty.
   * @param {{query?: string, mode?: string|null, filters?: unknown,
   *   advancedParams?: Record<string, any>, signal?: AbortSignal}} [params]
   * @returns {Promise<any>} `{dimensions, histogram, exact, tier, available?}`
   */
  async get({ query, mode, filters, advancedParams, signal } = {}) {
    return api.get(
      '/facets',
      {
        q: query ?? '',
        mode: mode || undefined,
        filters: filtersParam(filters),
        advanced_params: advancedParamsParam(advancedParams),
      },
      { signal }
    );
  },
};

/**
 * Entries API functions.
 */
export const entriesApi = {
  /**
   * Describe the configured logbook's write capability, so the create form can
   * adapt its credential prompt, and list the entry fields the facility
   * declares, in form order.
   * @returns {Promise<{supports_write: boolean, requires_auth: boolean, source_system: ?string, entry_fields: any[]}>}
   */
  async getPublishInfo() {
    return api.get('/publish-info');
  },

  /**
   * List the choices of one dynamic-select entry field.
   *
   * Blank values are left out of the query, so the server never sees a parent
   * the operator has not filled in.
   * @param {string} endpoint - The field's `options_endpoint`, relative to the API base
   * @param {Record<string, string|number|boolean|null|undefined>} [values] - Its `depends_on` values
   * @returns {Promise<Array<{value: string, label: string}>>} The field's choices
   */
  async getEntryFieldOptions(endpoint, values = {}) {
    /** @type {Record<string, string|number|boolean>} */
    const params = {};
    for (const [key, value] of Object.entries(values)) {
      if (value !== null && value !== undefined && value !== '') params[key] = value;
    }
    const data = await api.get(endpoint, params);
    return Array.isArray(data?.options) ? data.options : [];
  },

  /**
   * List entries, newest first, one keyset page at a time: pass the previous
   * response's `next_cursor` as `cursor`. `filters` (C-FILTER) is sent as JSON
   * text only when it selects something.
   * @param {{cursor?: string|null, pageSize?: number, filters?: unknown,
   *   view?: 'full'|'row', signal?: AbortSignal}} [params] - List parameters
   * @returns {Promise<any>} `{entries, total, next_cursor, ...}`
   */
  async list(params = {}) {
    return api.get(
      '/entries',
      {
        cursor: params.cursor || undefined,
        page_size: params.pageSize || 20,
        filters: filtersParam(params.filters),
        view: params.view,
      },
      { signal: params.signal }
    );
  },

  /**
   * Get a single entry by ID.
   * @param {string} entryId - Entry ID
   * @returns {Promise<any>} Entry data
   */
  async get(entryId) {
    return api.get(`/entries/${encodeURIComponent(entryId)}`);
  },

  /**
   * Get the follow-up thread of an entry (C-THREAD): the entry it follows up,
   * when any, and the entries that follow it up.
   * @param {string} entryId - Entry ID
   * @returns {Promise<{parent_id: ?string, parent: any, children: any[], truncated?: boolean}>} The thread
   */
  async thread(entryId) {
    return api.get(`/entries/${encodeURIComponent(entryId)}/thread`);
  },

  /**
   * Get the enhancement inspector cards of an entry (A8), in execution order.
   * @param {string} entryId - Entry ID
   * @returns {Promise<EnhancementsResponse>} The cards
   */
  async enhancements(entryId) {
    return api.get(`/entries/${encodeURIComponent(entryId)}/enhancements`);
  },

  /**
   * Get the entries nearest to an entry by stored text embedding (A9), self
   * excluded. `limit` is sent only when given (server default 5, max 10).
   * @param {string} entryId - Entry ID
   * @param {number} [limit] - Maximum number of related entries
   * @returns {Promise<RelatedResponse>} The related entries and the search state
   */
  async related(entryId, limit) {
    return api.get(`/entries/${encodeURIComponent(entryId)}/related`, { limit });
  },

  /**
   * Create a new entry.
   * @param {any} data - Entry data
   * @returns {Promise<any>} Created entry
   */
  async create(data) {
    return api.post('/entries', {
      subject: data.subject,
      details: data.details,
      author: data.author || null,
      logbook: data.logbook || null,
      shift: data.shift || null,
      tags: data.tags || [],
      metadata: data.metadata || null,
      auth_user: data.auth_user || null,
      auth_password: data.auth_password || null,
    });
  },

  /**
   * Create a new entry with file attachments via multipart form.
   * @param {any} data - Entry data fields
   * @param {FileList|File[]} files - Files to attach
   * @returns {Promise<any>} Created entry with attachment_count
   */
  async createWithAttachments(data, files) {
    const formData = new FormData();
    formData.append('subject', data.subject);
    formData.append('details', data.details);
    if (data.author) formData.append('author', data.author);
    if (data.logbook) formData.append('logbook', data.logbook);
    if (data.shift) formData.append('shift', data.shift);
    formData.append('tags', (data.tags || []).join(','));
    if (data.metadata) formData.append('metadata', JSON.stringify(data.metadata));
    // Forward logbook credentials so an attachment entry can answer a 401
    // auth_required prompt on resubmit, matching the text-only `create` path.
    if (data.auth_user) formData.append('auth_user', data.auth_user);
    if (data.auth_password) formData.append('auth_password', data.auth_password);

    for (const file of files) {
      formData.append('files', file);
    }

    // The gate refuses a multipart write unless it carries X-Osprey-Upload: a
    // header a plain HTML form cannot send, so only this client can post here.
    // No Content-Type: the browser sets multipart/form-data with its boundary.
    const response = await fetch(API_BASE + '/entries/upload', {
      method: 'POST',
      headers: { 'X-Osprey-Upload': '1' },
      body: formData,
    });

    if (!response.ok) {
      throw await errorFromResponse(response);
    }
    return response.json();
  },
};

/**
 * Drafts API functions.
 */
export const draftsApi = {
  /**
   * Get a draft by ID.
   * @param {string} draftId - Draft ID
   * @returns {Promise<any>} Draft data
   */
  async get(draftId) {
    return api.get(`/drafts/${encodeURIComponent(draftId)}`);
  },

  /**
   * List the staged drafts, newest first (C-DRAFTS-LIST).
   * @returns {Promise<{drafts: Array<{draft_id: string, subject: string, author: ?string,
   *   logbook: ?string, created_at: string, staged_by: ?string, attachment_count: number}>}>}
   */
  async list() {
    return api.get('/drafts');
  },

  /**
   * Delete a draft and its sidecar (C-DRAFT-DELETE). The server answers 204
   * with an empty body, so nothing is read from a successful response.
   * @param {string} draftId - Draft ID
   * @returns {Promise<void>}
   */
  async delete(draftId) {
    const response = await fetch(API_BASE + `/drafts/${encodeURIComponent(draftId)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw await errorFromResponse(response);
    }
    return undefined;
  },
};

/**
 * Status API functions.
 */
export const statusApi = {
  /**
   * Get service status; `enhancement_coverage` is an {@link EnhancementCoverage}.
   * @returns {Promise<any>} Status information
   */
  async get() {
    return api.get('/status');
  },
};

/**
 * Config API — read/write config.yml.
 */
export const configApi = {
  async get() {
    return api.get('/config');
  },
  /**
   * @param {string} content - Raw config.yml text.
   */
  async update(content) {
    return api.put('/config', { content });
  },
};

export default {
  api,
  capabilitiesApi,
  searchApi,
  facetsApi,
  entriesApi,
  draftsApi,
  statusApi,
  configApi,
};
