/**
 * OSPREY Channel Finder — Centralized State Management
 *
 * Event-emitter pattern: views subscribe to state changes.
 */

class ChannelFinderState {
  constructor() {
    this._listeners = {};

    // Pipeline info (populated from GET /api/info on init)
    this.pipelineType = null;
    this.pipelineMetadata = null;

    // Current view
    this.activeView = 'explore';
  }

  // ---- Event Emitter ----

  on(event, callback) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(callback);
  }

  off(event, callback) {
    if (!this._listeners[event]) return;
    this._listeners[event] = this._listeners[event].filter(cb => cb !== callback);
  }

  emit(event, data) {
    if (!this._listeners[event]) return;
    for (const cb of this._listeners[event]) {
      try { cb(data); } catch (e) { console.error(`State listener error [${event}]:`, e); }
    }
  }

  // ---- Setters ----

  setPipelineInfo(type, metadata) {
    this.pipelineType = type;
    this.pipelineMetadata = metadata;
    this.emit('pipelineChanged', { type, metadata });
  }

  setActiveView(view) {
    this.activeView = view;
    this.emit('viewChanged', view);
  }
}

export const state = new ChannelFinderState();
