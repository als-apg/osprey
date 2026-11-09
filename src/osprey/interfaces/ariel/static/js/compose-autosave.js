// @ts-check
/**
 * ARIEL compose autosave (C-AUTOSAVE): one browser-local snapshot per compose
 * context, kept in browser storage only (ruling e-0001 R3). No DOM.
 *
 * Every snapshot lives in one record under `scopedStorageKey(AUTOSAVE_KEY)`,
 * a map from a context key (`contextKey`) to
 * `{subject, details, author, fields, tags, logbook, shift, savedAt}`.
 * `fields` holds every declared field, blank ones included, so a cleared
 * field stays cleared on restore. Snapshots older than 30 days are pruned on
 * save. Credential and attachment fields are never stored. A snapshot never
 * moves between keys. Storage may be blocked in a sandboxed iframe: nothing
 * here throws.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/**
 * @typedef {Object} ComposeSnapshot
 * @property {string} subject
 * @property {string} details
 * @property {string} author
 * @property {Record<string, any>} fields - every declared entry field
 * @property {string[]} tags
 * @property {string|null} logbook
 * @property {string|null} shift
 * @property {number} savedAt - epoch milliseconds
 */

/** @typedef {'new' | `follow:${string}` | `draft:${string}`} ComposeKey */

/** Base storage key; scoped by `scopedStorageKey`, shared by both frames. */
export const AUTOSAVE_KEY = 'ariel-compose-autosave';

/** Snapshots older than this are pruned on save. */
const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

/** Field names that carry credentials or attachments: never stored or saved on. */
export const SECRET_FIELD = /^auth_|password|credential|attachment|^files?$/i;

/**
 * The autosave key of a compose context. A draft wins; a follow-up needs a
 * non-empty parent id; anything else is a new entry.
 * @param {{draft?: string|null, followsUp?: string|null}|null|undefined} compose
 * @returns {ComposeKey}
 */
export function contextKey(compose) {
  if (compose && typeof compose.draft === 'string' && compose.draft) {
    return `draft:${compose.draft}`;
  }
  if (compose && typeof compose.followsUp === 'string' && compose.followsUp) {
    return `follow:${compose.followsUp}`;
  }
  return 'new';
}

/** @returns {Record<string, unknown>} */
function readRecord() {
  /** @type {string|null} */
  let raw;
  try {
    raw = localStorage.getItem(scopedStorageKey(AUTOSAVE_KEY));
  } catch {
    return {};
  }
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/**
 * @param {Record<string, ComposeSnapshot>} record
 * @returns {boolean} whether the write landed
 */
function writeRecord(record) {
  try {
    const key = scopedStorageKey(AUTOSAVE_KEY);
    if (Object.keys(record).length === 0) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(record));
    return true;
  } catch {
    return false;
  }
}

/** @param {unknown} value */
function str(value) {
  return typeof value === 'string' ? value : '';
}

/** @param {unknown} value */
function strOrNull(value) {
  return typeof value === 'string' ? value : null;
}

/**
 * The stored shape of a snapshot, or null when it is not one. Only the
 * C-AUTOSAVE members survive, so credentials and attachments never do.
 * @param {unknown} item
 * @param {number} [fallbackSavedAt]
 * @returns {ComposeSnapshot|null}
 */
function normalise(item, fallbackSavedAt) {
  if (!item || typeof item !== 'object' || Array.isArray(item)) return null;
  const src = /** @type {Record<string, unknown>} */ (item);
  const savedAt = Number.isFinite(src.savedAt) ? Number(src.savedAt) : fallbackSavedAt;
  if (savedAt === undefined) return null;
  /** @type {Record<string, any>} */
  const fields = {};
  if (src.fields && typeof src.fields === 'object' && !Array.isArray(src.fields)) {
    for (const [name, value] of Object.entries(src.fields)) {
      if (!SECRET_FIELD.test(name)) fields[name] = value;
    }
  }
  return {
    subject: str(src.subject),
    details: str(src.details),
    author: str(src.author),
    fields,
    tags: Array.isArray(src.tags) ? src.tags.filter((t) => typeof t === 'string') : [],
    logbook: strOrNull(src.logbook),
    shift: strOrNull(src.shift),
    savedAt,
  };
}

/**
 * Store the snapshot for `key`, pruning every snapshot older than 30 days.
 * `savedAt` defaults to now.
 * @param {ComposeKey} key
 * @param {Partial<ComposeSnapshot> & Record<string, any>} snapshot
 * @returns {boolean} true only if the write landed
 */
export function saveCompose(key, snapshot) {
  if (typeof key !== 'string' || !key) return false;
  const now = Date.now();
  const entry = normalise(snapshot, now);
  if (!entry) return false;
  /** @type {Record<string, ComposeSnapshot>} */
  const next = {};
  for (const [k, item] of Object.entries(readRecord())) {
    if (k === key) continue;
    const kept = normalise(item);
    if (kept && now - kept.savedAt <= MAX_AGE_MS) next[k] = kept;
  }
  next[key] = entry;
  return writeRecord(next);
}

/**
 * The snapshot stored for `key`, or null (none, malformed or expired).
 * @param {ComposeKey} key
 * @returns {ComposeSnapshot|null}
 */
export function loadCompose(key) {
  if (typeof key !== 'string' || !key) return null;
  const record = readRecord();
  if (!Object.prototype.hasOwnProperty.call(record, key)) return null;
  const entry = normalise(record[key]);
  if (!entry || Date.now() - entry.savedAt > MAX_AGE_MS) return null;
  return entry;
}

/**
 * Forget the snapshot for `key`; the other keys are untouched.
 * @param {ComposeKey} key
 */
export function clearCompose(key) {
  if (typeof key !== 'string' || !key) return;
  const record = readRecord();
  if (!Object.prototype.hasOwnProperty.call(record, key)) return;
  delete record[key];
  writeRecord(/** @type {Record<string, ComposeSnapshot>} */ (record));
}
