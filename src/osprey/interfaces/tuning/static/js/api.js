/**
 * OSPREY Tuning — REST API Client
 *
 * All calls go through /api/proxy/... to avoid CORS.
 */

class TuningAPI {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

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

  async getEnvironmentDetails(name) {
    return this._fetch(`environments/${encodeURIComponent(name)}/details`);
  }

  async checkEnvironmentStatus(name) {
    return this._fetch(`environments/${encodeURIComponent(name)}/status`);
  }

  // ---- Optimization Lifecycle ----

  async startOptimization(config) {
    return this._fetch('optimization/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getState(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/state`);
  }

  async pause(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/pause`, { method: 'POST' });
  }

  async resume(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/resume`, { method: 'POST' });
  }

  async cancel(jobId) {
    return this._fetch(`optimization/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
  }

  // ---- Variables ----

  async getVariableValue(env, pvName) {
    return this._fetch(`variables/${encodeURIComponent(env)}/${encodeURIComponent(pvName)}`);
  }

  async getEnvironmentVariables(env) {
    return this._fetch(`variables/${encodeURIComponent(env)}`);
  }

  async setMachineVariables(env, variables) {
    return this._fetch(`variables/${encodeURIComponent(env)}/set`, {
      method: 'POST',
      body: JSON.stringify({ variables }),
    });
  }

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

  async loadRun(timestamp) {
    return this._fetch(`runs/${encodeURIComponent(timestamp)}`);
  }
}

export const api = new TuningAPI();
