// @ts-check
/**
 * OSPREY Channel Finder — Centralized State Management
 *
 * Event-emitter pattern: views subscribe to state changes.
 */

import { familyOf } from './route.js';

/** @typedef {(data: any) => void} StateListener */

/**
 * Where the graph store this panel reads was loaded from: the bolt URI it is
 * served on and the TTL file it was seeded from. Reported by GET /api/info on
 * the graph paradigm only.
 * @typedef {{ uri: string|null, ttl_filename: string|null }} GraphStoreInfo
 */

/**
 * The scope descriptor of GET /api/info: present only when the `pv_directory`
 * section is configured.
 * @typedef {{
 *   label: string,
 *   configured_services: ('alarm_log'|'alarm_state'|'put_log'|'logbook')[],
 *   canopy: boolean,
 *   ariel: boolean,
 * }} DirectoryDescriptor
 */

/**
 * Payload of `curatedResults`, emitted by a curated renderer after every search.
 * @typedef {{ query: string, total: number }} CuratedResults
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

    // The search words, carried across the Curated and All PVs tabs in both
    // directions.
    /** @type {string} */
    this.scopeQuery = '';

    // The All PVs basket, in selection order, kept across the family's pages
    // (All PVs and its PV and IOC pages).
    /** @type {Set<string>} */
    this.selection = new Set();

    // Set by app.js on each /api/info fetch; null when the section is not
    // configured, and the allpvs, pv and ioc routes then do not exist.
    /** @type {DirectoryDescriptor|null} */
    this.directory = null;

    // Events: `curatedResults` (CuratedResults) after every curated search,
    // `directoryLabel` (string) after each All PVs page. Registered first, so
    // every other listener of `directoryLabel` reads the new label here.
    this.on('directoryLabel', (label) => {
      if (this.directory) this.directory = { ...this.directory, label };
    });
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
   * @param {string} view
   */
  setActiveView(view) {
    this.activeView = view;
    if (familyOf(view) !== 'allpvs') this.selection.clear();
    this.emit('viewChanged', view);
  }
}

export const state = new ChannelFinderState();
