// @ts-check
/**
 * OSPREY Channel Finder — Centralized State Management
 *
 * Event-emitter pattern: views subscribe to state changes.
 */

/** @typedef {(data: any) => void} StateListener */

/**
 * Where the graph store this panel reads was loaded from: the bolt URI it is
 * served on and the TTL file it was seeded from. Reported by GET /api/info on
 * the graph paradigm only.
 * @typedef {{ uri: string|null, ttl_filename: string|null }} GraphStoreInfo
 */

/**
 * A view's contribution to the hub tile bar: the items it adds and the
 * handler their actions reach. See header-contrib.js for the item vocabulary.
 * @typedef {{
 *   items: import('/design-system/js/header-contrib.js').HeaderItem[],
 *   onAction: (id: string, value: string | undefined) => void,
 * }} ViewHeader
 */

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

    // Graph provenance (populated from GET /api/info; file-backed paradigms
    // report neither field and keep the empty defaults).
    /** @type {string[]} */
    this.tools = [];
    /** @type {GraphStoreInfo|null} */
    this.graphStore = null;

    // Current view
    /** @type {string} */
    this.activeView = 'explore';

    // What the active view adds to the hub's tile bar, ahead of the
    // Explore/Feedback switcher app.js always contributes, and the handler
    // its actions route to. Null when the view adds nothing.
    /** @type {ViewHeader|null} */
    this.viewHeader = null;
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
    this._listeners[event] = this._listeners[event].filter((cb) => cb !== callback);
  }

  /**
   * @param {string} event
   * @param {any} [data]
   */
  emit(event, data) {
    if (!this._listeners[event]) return;
    for (const cb of this._listeners[event]) {
      try {
        cb(data);
      } catch (e) {
        console.error(`State listener error [${event}]:`, e);
      }
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
   * Record the tool names and graph-store provenance reported by GET /api/info.
   * Both are absent on the file-backed paradigms, where the missing values fall
   * back to the empty defaults rather than carrying over a previous answer.
   * @param {string[]|null|undefined} tools
   * @param {GraphStoreInfo|null|undefined} graphStore
   */
  setGraphInfo(tools, graphStore) {
    this.tools = tools || [];
    this.graphStore = graphStore || null;
  }

  /**
   * Set (or, with null, drop) the active view's own tile-bar items. app.js
   * re-contributes the whole bar on every change.
   * @param {ViewHeader|null} header
   */
  setViewHeader(header) {
    this.viewHeader = header;
    this.emit('viewHeaderChanged', header);
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
