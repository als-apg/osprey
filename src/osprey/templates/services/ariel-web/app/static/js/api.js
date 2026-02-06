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
    // Build request with basic params
    const request = {
      query: params.query,
      mode: params.mode || 'auto',
      max_results: params.maxResults || 10,
      start_date: params.startDate || null,
      end_date: params.endDate || null,
      author: params.author || null,
      source_system: params.sourceSystem || null,
    };

    // Add advanced retrieval params if provided
    if (params.similarityThreshold !== undefined) {
      request.similarity_threshold = params.similarityThreshold;
    }
    if (params.includeHighlights !== undefined) {
      request.include_highlights = params.includeHighlights;
    }
    if (params.fuzzyFallback !== undefined) {
      request.fuzzy_fallback = params.fuzzyFallback;
    }

    // Add advanced assembly params if provided
    if (params.assemblyMaxItems !== undefined) {
      request.assembly_max_items = params.assemblyMaxItems;
    }
    if (params.assemblyMaxChars !== undefined) {
      request.assembly_max_chars = params.assemblyMaxChars;
    }
    if (params.assemblyMaxCharsPerItem !== undefined) {
      request.assembly_max_chars_per_item = params.assemblyMaxCharsPerItem;
    }

    // Add advanced processing params if provided (RAG mode)
    if (params.temperature !== undefined) {
      request.temperature = params.temperature;
    }
    if (params.maxTokens !== undefined) {
      request.max_tokens = params.maxTokens;
    }

    // Add advanced fusion params if provided (MULTI mode)
    if (params.fusionStrategy !== undefined) {
      request.fusion_strategy = params.fusionStrategy;
    }
    if (params.keywordWeight !== undefined) {
      request.keyword_weight = params.keywordWeight;
    }
    if (params.semanticWeight !== undefined) {
      request.semantic_weight = params.semanticWeight;
    }

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

export default {
  api,
  searchApi,
  entriesApi,
  statusApi,
};
