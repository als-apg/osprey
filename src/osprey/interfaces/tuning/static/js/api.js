// @ts-check
/**
 * OSPREY Tuning — REST API Client
 *
 * All calls go through /api/proxy/... to avoid CORS.
 * @module tuning/api
 */

class TuningAPI {
  /** @param {string} [baseUrl] */
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

  /**
   * @param {string} path
   * @param {RequestInit} [options]
   * @returns {Promise<any>}
   */
  async _fetch(path, options = {}) {
    const url = `${this.baseUrl}/api/proxy/${path}`;
    const resp = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!resp.ok) {
      const text = await resp.text();
      let detail;
      try { detail = JSON.parse(text).error || JSON.parse(text).detail || text; } catch { detail = text; }
      throw new Error(`API error ${resp.status}: ${detail}`);
    }
    return resp.json();
  }

  // ---- Environment ----

  async listEnvironments() {
    return this._fetch('environments/list');
  }

  /** @param {string} name */
  async getEnvironmentDetails(name) {
    return this._fetch(`environments/${encodeURIComponent(name)}/details`);
  }

  /** @param {string} name */
  async checkEnvironmentStatus(name) {
    return this._fetch(`environments/${encodeURIComponent(name)}/status`);
  }

  // ---- Optimization Lifecycle ----

  /** @param {object} config */
  async startOptimization(config) {
    return this._fetch('optimization/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  /** @param {string} jobId */
  async getState(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/state`);
  }

  /** @param {string} jobId */
  async pause(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/pause`, { method: 'POST' });
  }

  /** @param {string} jobId */
  async resume(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/resume`, { method: 'POST' });
  }

  /** @param {string} jobId */
  async cancel(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
  }

  // ---- Variables ----

  /** @param {string} env  @param {string} pvName */
  async getVariableValue(env, pvName) {
    return this._fetch(`variables/${encodeURIComponent(env)}/${encodeURIComponent(pvName)}`);
  }

  /** @param {string} env */
  async getEnvironmentVariables(env) {
    return this._fetch(`variables/${encodeURIComponent(env)}`);
  }

  /** @param {string} env  @param {object} variables */
  async setMachineVariables(env, variables) {
    return this._fetch(`variables/${encodeURIComponent(env)}/set`, {
      method: 'POST',
      body: JSON.stringify({ variables }),
    });
  }

  /** @param {string} env  @param {string[]} names */
  async getMachineVariables(env, names) {
    return this._fetch(`variables/${encodeURIComponent(env)}/get`, {
      method: 'POST',
      body: JSON.stringify({ names }),
    });
  }

  // ---- Historical Runs ----

  async getAvailableRuns() {
    return this._fetch('runs/list');
  }

  /** @param {string} timestamp */
  async loadRun(timestamp) {
    return this._fetch(`runs/${encodeURIComponent(timestamp)}`);
  }
}

export const api = new TuningAPI();
