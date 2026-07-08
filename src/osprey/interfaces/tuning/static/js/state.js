// @ts-check
/**
 * OSPREY Tuning — Centralized State Management
 *
 * Event-emitter pattern: components subscribe to state changes.
 * Replaces Dash's dcc.Store components.
 * @module tuning/state
 */

/**
 * @typedef {object} OptimizationState
 * @property {string} status
 * @property {any[]} lhs_data
 * @property {any[]} bo_data
 * @property {any[]} logs
 * @property {any[]} snapshots
 * @property {object | null} best_point
 * @property {number} current_iteration
 * @property {number} total_iterations
 * @property {string | null} phase
 */

/** @typedef {(data: any) => void} StateListener */

class TuningState {
  constructor() {
    /** @type {Record<string, StateListener[]>} */
    this._listeners = {};

    // Job state
    /** @type {string | null} */
    this.jobId = sessionStorage.getItem('tuning_jobId') || null;
    /** @type {string | null} */
    this.environment = null;
    /** @type {object | null} */
    this.environmentDetails = null;

    // Optimization state (from backend getState)
    /** @type {OptimizationState} */
    this.optimizationState = {
      status: 'IDLE',
      lhs_data: [],
      bo_data: [],
      logs: [],
      snapshots: [],
      best_point: null,
      current_iteration: 0,
      total_iterations: 0,
      phase: null,
    };

    // Variable table data (shared between LHS/BO)
    /** @type {any[]} */
    this.variableTableData = [];

    // UI state
    /** @type {object | null} */
    this.selectedPoint = null;
    this.displayMode = 'normalized';
    this.activeTab = 'optimization-tab';

    // Derived page state
    this.pageState = this._computePageState();
  }

  // ---- Event Emitter ----

  /** @param {string} event  @param {StateListener} callback */
  on(event, callback) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(callback);
  }

  /** @param {string} event  @param {StateListener} callback */
  off(event, callback) {
    if (!this._listeners[event]) return;
    this._listeners[event] = this._listeners[event].filter(cb => cb !== callback);
  }

  /** @param {string} event  @param {any} [data] */
  emit(event, data) {
    if (!this._listeners[event]) return;
    for (const cb of this._listeners[event]) {
      try { cb(data); } catch (e) { console.error(`State listener error [${event}]:`, e); }
    }
  }

  // ---- Setters (emit events) ----

  /** @param {string | null} id */
  setJobId(id) {
    this.jobId = id;
    if (id) {
      sessionStorage.setItem('tuning_jobId', id);
    } else {
      sessionStorage.removeItem('tuning_jobId');
    }
    this._updatePageState();
    this.emit('jobChanged', id);
  }

  /** @param {string | null} env  @param {object | null} [details] */
  setEnvironment(env, details = null) {
    this.environment = env;
    this.environmentDetails = details;
    this._updatePageState();
    this.emit('environmentChanged', { env, details });
  }

  /** @param {Partial<OptimizationState>} state */
  setOptimizationState(state) {
    this.optimizationState = { ...this.optimizationState, ...state };
    this._updatePageState();
    this.emit('optimizationStateChanged', this.optimizationState);
  }

  /** @param {any[]} data */
  setVariableTableData(data) {
    this.variableTableData = [...data];
    this.emit('variableTableChanged', this.variableTableData);
  }

  /** @param {any} variable */
  addVariable(variable) {
    this.variableTableData.push(variable);
    this.emit('variableTableChanged', this.variableTableData);
  }

  /** @param {string} pvName */
  removeVariable(pvName) {
    this.variableTableData = this.variableTableData.filter(v => v.pv_name !== pvName);
    this.emit('variableTableChanged', this.variableTableData);
  }

  /** @param {string} pvName  @param {object} updates */
  updateVariable(pvName, updates) {
    const idx = this.variableTableData.findIndex(v => v.pv_name === pvName);
    if (idx >= 0) {
      this.variableTableData[idx] = { ...this.variableTableData[idx], ...updates };
      this.emit('variableTableChanged', this.variableTableData);
    }
  }

  /** @param {object | null} point */
  setSelectedPoint(point) {
    this.selectedPoint = point;
    this.emit('selectedPointChanged', point);
  }

  /** @param {string} mode */
  setDisplayMode(mode) {
    this.displayMode = mode;
    this.emit('displayModeChanged', mode);
  }

  /** @param {string} tabId */
  setActiveTab(tabId) {
    this.activeTab = tabId;
    this.emit('tabChanged', tabId);
  }

  // ---- Reset ----

  resetForNewRun() {
    this.setJobId(null);
    this.optimizationState = {
      status: 'IDLE',
      lhs_data: [],
      bo_data: [],
      logs: [],
      snapshots: [],
      best_point: null,
      current_iteration: 0,
      total_iterations: 0,
      phase: null,
    };
    this.selectedPoint = null;
    this._updatePageState();
    this.emit('optimizationStateChanged', this.optimizationState);
    this.emit('selectedPointChanged', null);
    this.emit('resetForNewRun', null);
  }

  // ---- Derived State ----

  _computePageState() {
    const status = this.optimizationState.status;
    const hasEnv = !!this.environment;
    const isRunning = status === 'RUNNING';
    const isPaused = status === 'PAUSED';
    const isIdle = status === 'IDLE' || status === 'COMPLETED' || status === 'CANCELLED' || status === 'ERROR';
    const hasJob = !!this.jobId;

    return {
      canStart: hasEnv && isIdle,
      canPause: hasJob && isRunning,
      canResume: hasJob && isPaused,
      canCancel: hasJob && (isRunning || isPaused),
      showPause: isRunning,
      showResume: isPaused,
      showCancel: isRunning || isPaused,
      formDisabled: isRunning || isPaused,
      pollingEnabled: isRunning || isPaused,
      envSelectorDisabled: isRunning || isPaused,
    };
  }

  _updatePageState() {
    this.pageState = this._computePageState();
    this.emit('pageStateChanged', this.pageState);
  }
}

// Singleton
export const state = new TuningState();
