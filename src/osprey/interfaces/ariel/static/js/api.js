// @ts-check
/**
 * ARIEL API Client
 *
 * HTTP client for communicating with the ARIEL backend API.
 */

const API_BASE = '/api';

/**
 * Structured API error carrying the HTTP status code and an optional
 * machine-readable `code` discriminator from the response body.
 */
export class ApiError extends Error {
  /**
   * @param {number} status - HTTP status code.
   * @param {string} message - Error detail message.
   * @param {string} [code] - Machine-readable discriminator (e.g. "auth_required").
   */
  constructor(status, message, code) {
    super(message);
    /** @type {number} */
    this.status = status;
    /** @type {string|undefined} */
    this.code = code;
    this.name = 'ApiError';
  }
}

/**
 * Build an ApiError from a non-OK response.
 *
 * Preserves the HTTP `status` and any machine-readable `code` discriminator from
 * the JSON body (e.g. "auth_required") so callers can branch on them — for
 * example, to prompt for logbook credentials instead of showing a generic error.
 * @param {Response} response - The failed fetch response
 * @returns {Promise<ApiError>} An ApiError with `.status` and optional `.code`
 */
async function errorFromResponse(response) {
  const body = await response.json().catch(() => ({}));
  return new ApiError(response.status, body.detail || `HTTP ${response.status}`, body.code);
}

/**
 * Issue a JSON-body request, shared by `post` and `put` (which only differ
 * in HTTP method).
 * @param {string} method - HTTP method ('POST', 'PUT')
 * @param {string} endpoint - API endpoint
 * @param {any} data - Request body
 * @returns {Promise<any>} Response data
 */
async function jsonRequest(method, endpoint, data) {
  const response = await fetch(API_BASE + endpoint, {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
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
   * @returns {Promise<any>} Response data
   */
  async get(endpoint, params = {}) {
    const url = new URL(API_BASE + endpoint, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        url.searchParams.append(key, value);
      }
    });

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw await errorFromResponse(response);
    }
    return response.json();
  },

  /**
   * Make a POST request.
   * @param {string} endpoint - API endpoint
   * @param {any} data - Request body
   * @returns {Promise<any>} Response data
   */
  async post(endpoint, data = {}) {
    return jsonRequest('POST', endpoint, data);
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
   * Execute a search query.
   * @param {any} params - Search parameters
   * @returns {Promise<any>} Search results
   */
  async search(params) {
    const request = {
      query: params.query,
      mode: params.mode || 'keyword',
      max_results: params.maxResults || 10,
      advanced_params: params.advancedParams || {},
    };

    return api.post('/search', request);
  },
};

/**
 * Entries API functions.
 */
export const entriesApi = {
  /**
   * Describe the configured logbook's write capability, so the create form can
   * adapt its credential prompt.
   * @returns {Promise<{supports_write: boolean, requires_auth: boolean, source_system: ?string}>}
   */
  async getPublishInfo() {
    return api.get('/publish-info');
  },

  /**
   * List entries with pagination.
   * @param {any} params - List parameters
   * @returns {Promise<any>} Paginated entries
   */
  async list(params = {}) {
    return api.get('/entries', {
      page: params.page || 1,
      page_size: params.pageSize || 20,
      start_date: params.startDate,
      end_date: params.endDate,
      author: params.author,
      source_system: params.sourceSystem,
      sort_order: params.sortOrder || 'desc',
    });
  },

  /**
   * Get a single entry by ID.
   * @param {string} entryId - Entry ID
   * @returns {Promise<any>} Entry data
   */
  async get(entryId) {
    return api.get(`/entries/${entryId}`);
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

    const response = await fetch(API_BASE + '/entries/upload', {
      method: 'POST',
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
    return api.get(`/drafts/${draftId}`);
  },
};

/**
 * Status API functions.
 */
export const statusApi = {
  /**
   * Get service status.
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
  entriesApi,
  draftsApi,
  statusApi,
  configApi,
};
