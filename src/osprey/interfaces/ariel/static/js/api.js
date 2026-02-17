/**
 * ARIEL API Client
 *
 * HTTP client for communicating with the ARIEL backend API.
 */

const API_BASE = '/api';

/**
 * API client with error handling and response parsing.
 */
export const api = {
  /**
   * Make a GET request.
   * @param {string} endpoint - API endpoint
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Response data
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
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
  },

  /**
   * Make a POST request.
   * @param {string} endpoint - API endpoint
   * @param {Object} data - Request body
   * @returns {Promise<Object>} Response data
   */
  async post(endpoint, data = {}) {
    const response = await fetch(API_BASE + endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
  },

  /**
   * Make a PUT request.
   * @param {string} endpoint - API endpoint
   * @param {Object} data - Request body
   * @returns {Promise<Object>} Response data
   */
  async put(endpoint, data = {}) {
    const response = await fetch(API_BASE + endpoint, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
  },
};

/**
 * Capabilities API — discover available modes and parameters.
 */
export const capabilitiesApi = {
  /**
   * Fetch available search modes and their tunable parameters.
   * @returns {Promise<Object>} Capabilities response
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
   * @param {Object} params - Search parameters
   * @returns {Promise<Object>} Search results
   */
  async search(params) {
    const request = {
      query: params.query,
      mode: params.mode || 'rag',
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
   * List entries with pagination.
   * @param {Object} params - List parameters
   * @returns {Promise<Object>} Paginated entries
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
   * @returns {Promise<Object>} Entry data
   */
  async get(entryId) {
    return api.get(`/entries/${entryId}`);
  },

  /**
   * Create a new entry.
   * @param {Object} data - Entry data
   * @returns {Promise<Object>} Created entry
   */
  async create(data) {
    return api.post('/entries', {
      subject: data.subject,
      details: data.details,
      author: data.author || null,
      logbook: data.logbook || null,
      shift: data.shift || null,
      tags: data.tags || [],
    });
  },

  /**
   * Create a new entry with file attachments via multipart form.
   * @param {Object} data - Entry data fields
   * @param {FileList|File[]} files - Files to attach
   * @returns {Promise<Object>} Created entry with attachment_count
   */
  async createWithAttachments(data, files) {
    const formData = new FormData();
    formData.append('subject', data.subject);
    formData.append('details', data.details);
    if (data.author) formData.append('author', data.author);
    if (data.logbook) formData.append('logbook', data.logbook);
    if (data.shift) formData.append('shift', data.shift);
    formData.append('tags', (data.tags || []).join(','));

    for (const file of files) {
      formData.append('files', file);
    }

    const response = await fetch(API_BASE + '/entries/upload', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
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
   * @returns {Promise<Object>} Draft data
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
   * @returns {Promise<Object>} Status information
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
  async update(content) {
    return api.put('/config', { content });
  },
};

/**
 * Claude Setup API — browse/edit Claude Code setup files.
 */
export const claudeSetupApi = {
  async list() {
    return api.get('/claude-setup');
  },
  async getFile(path) {
    return api.get(`/claude-setup/${path}`);
  },
  async updateFile(path, content) {
    return api.put(`/claude-setup/${path}`, { content });
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
  claudeSetupApi,
};
