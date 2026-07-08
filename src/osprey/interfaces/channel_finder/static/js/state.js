// @ts-check
/**
 * OSPREY Channel Finder — Centralized State Management
 *
 * Event-emitter pattern: views subscribe to state changes.
 */

/** @typedef {(data: any) => void} StateListener */

class ChannelFinderState {
  constructor() {
    /** @type {Record<string, StateListener[]>} */
    this._listeners = {};

    // Pipeline info (populated from GET /api/info on init)
    /** @type {string|null} */
    this.pipelineType = null;
    /** @type {any} */
    this.pipelineMetadata = null;
    /** @type {string[]} */
    this.availablePipelines = [];
    /** @type {string|null} */
    this.dbPath = null;

    // Current view
    /** @type {string} */
    this.activeView = 'explore';
  }

  // ---- Event Emitter ----

  /**
   * @param {string} event
   * @param {StateListener} callback
   */
  on(event, callback) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(callback);
  }

  /**
   * @param {string} event
   * @param {StateListener} callback
   */
  off(event, callback) {
    if (!this._listeners[event]) return;
    this._listeners[event] = this._listeners[event].filter(cb => cb !== callback);
  }

  /**
   * @param {string} event
   * @param {any} [data]
   */
  emit(event, data) {
    if (!this._listeners[event]) return;
    for (const cb of this._listeners[event]) {
      try { cb(data); } catch (e) { console.error(`State listener error [${event}]:`, e); }
    }
  }

  // ---- Setters ----

  /**
   * @param {string|null} type
   * @param {any} metadata
   */
  setPipelineInfo(type, metadata) {
    this.pipelineType = type;
    this.pipelineMetadata = metadata;
    this.emit('pipelineChanged', { type, metadata });
  }

  /**
   * @param {string} view
   */
  setActiveView(view) {
    this.activeView = view;
    this.emit('viewChanged', view);
  }
}

export const state = new ChannelFinderState();
