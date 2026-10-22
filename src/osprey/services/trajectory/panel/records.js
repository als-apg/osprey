// @ts-check
/**
 * records — Record data, Save Records and Flush Records: the shots kept in
 * the browser.
 *
 * While Record data is ticked, every new shot read from the machine is kept
 * as its id and its `ShotResult`, newest first, up to the snapshot's
 * `retention` (the shots whose raw buffers the service keeps). A record
 * pushed past the cap drops out with a notice. A derived shot (one with a
 * `parent_id`, re-extracted from another) is never recorded: the raw route
 * resolves it to its parent's buffer, so it holds nothing of its own.
 *
 * The stream may miss shot events during a continuous run; when the run
 * ends, the shots list fills in every shot read since recording began that
 * the list does not hold yet.
 *
 * Save Records fetches each record's raw buffer at save time through the raw
 * route and writes one JSON download: per record its `ShotResult` and its raw
 * buffer (header plus every turn's samples as base64 little-endian float32).
 * A record whose buffer the service already evicted is listed under
 * `evicted`, never fatal. Flush Records empties the list.
 *
 * The decision functions ({@link addRecord}, {@link fillRecords},
 * {@link recordsDocument}, {@link floatsToBase64}) are pure; {@link createRecords}
 * holds the list and does the calls through an injected api; {@link mount}
 * wires it to the area host.
 *
 * Mounted into the `records` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point.
 *
 * @module records
 */

import { control, h } from './views.js';

/** The download's `kind`, naming what the file holds. */
export const RECORDS_KIND = 'trajectory-records';

/** The download's format version. */
export const RECORDS_VERSION = 1;

/** The refusal code the raw route answers for a pruned buffer. */
export const RAW_EVICTED = 'raw_evicted';

/**
 * @typedef {{shot_id: string, parent_id?: string | null, timestamp?: number | null} & Record<string, any>} Shot
 * @typedef {{id: string, shot: Shot}} ShotRecord
 * @typedef {{records: ShotRecord[], evicted: string[]}} Kept
 *   The list after a change, and the ids it pushed out past the cap.
 */

/** @param {ShotRecord | Shot} item */
function stamp(item) {
  const ts = 'shot' in item ? item.shot.timestamp : item.timestamp;
  return typeof ts === 'number' ? ts : Number.NEGATIVE_INFINITY;
}

/**
 * Keep the newest *cap* of *records*; a null cap keeps them all.
 * @param {ShotRecord[]} records  Newest first.
 * @param {number | null | undefined} cap
 * @param {Set<string>} before  The ids held before the change.
 * @returns {Kept}
 */
function capped(records, cap, before) {
  const limit = typeof cap === 'number' && cap >= 0 ? cap : records.length;
  const kept = records.slice(0, limit);
  const evicted = records.slice(limit).map((r) => r.id).filter((id) => before.has(id));
  return { records: kept, evicted };
}

/**
 * Whether *shot* may be recorded: it has an id and was read from the machine.
 * @param {any} shot
 * @returns {shot is Shot}
 */
export function recordable(shot) {
  return shot != null && typeof shot.shot_id === 'string' && shot.parent_id == null;
}

/**
 * *records* with *shot* added in front, capped at *cap*. A derived shot, or
 * one already held, leaves the list as it is.
 * @param {ShotRecord[]} records  Newest first.
 * @param {any} shot
 * @param {number | null | undefined} cap
 * @returns {Kept}
 */
export function addRecord(records, shot, cap) {
  if (!recordable(shot) || records.some((r) => r.id === shot.shot_id)) {
    return { records, evicted: [] };
  }
  const before = new Set(records.map((r) => r.id));
  return capped([{ id: shot.shot_id, shot }, ...records], cap, before);
}

/**
 * *records* with every shot of *listed* read after *floor* that it misses,
 * newest first by timestamp, capped at *cap*. Derived shots and shots
 * without a timestamp are never filled in.
 * @param {ShotRecord[]} records  Newest first.
 * @param {any[]} listed  The shots list.
 * @param {number} floor  Shots read at or before it are not filled in.
 * @param {number | null | undefined} cap
 * @returns {Kept}
 */
export function fillRecords(records, listed, floor, cap) {
  const held = new Set(records.map((r) => r.id));
  const missed = listed
    .filter((shot) => recordable(shot) && !held.has(shot.shot_id))
    .filter((shot) => typeof shot.timestamp === 'number' && shot.timestamp > floor)
    .map((shot) => ({ id: shot.shot_id, shot }));
  if (missed.length === 0) return { records, evicted: [] };
  const merged = [...records, ...missed].sort((a, b) => stamp(b) - stamp(a));
  return capped(merged, cap, held);
}

/**
 * Base64 of *samples* as little-endian float32.
 * @param {ArrayLike<number>} samples
 * @returns {string}
 */
export function floatsToBase64(samples) {
  const bytes = new Uint8Array(samples.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i += 1) view.setFloat32(i * 4, samples[i], true);
  let text = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    text += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(text);
}

/**
 * The download's content.
 * @param {object} parts
 * @param {ShotRecord[]} parts.records  The records saved, newest first.
 * @param {Map<string, import('./figures.js').RawData>} parts.raws  Each fetched raw buffer by record id.
 * @param {string[]} parts.evicted  The record ids whose buffer the service no longer holds.
 * @param {string[]} [parts.failed]  The record ids whose buffer could not be fetched.
 * @param {string | null} [parts.model]
 * @param {string} parts.savedAt  ISO-8601.
 */
export function recordsDocument({ records, raws, evicted, failed = [], model = null, savedAt }) {
  return {
    kind: RECORDS_KIND,
    version: RECORDS_VERSION,
    saved_at: savedAt,
    model,
    records: records
      .filter((r) => raws.has(r.id))
      .map((r) => {
        const raw = /** @type {import('./figures.js').RawData} */ (raws.get(r.id));
        return {
          id: r.id,
          shot: r.shot,
          raw: {
            header: raw.header,
            dtype: 'float32',
            byte_order: 'little',
            samples: floatsToBase64(raw.samples),
          },
        };
      }),
    evicted,
    failed,
  };
}

/**
 * The file name a save writes.
 * @param {string} savedAt  ISO-8601.
 */
export function recordsFileName(savedAt) {
  return `trajectory-records-${savedAt.replace(/[:.]/g, '-')}.json`;
}

/**
 * Hand *text* to the browser as a download named *name*.
 * @param {Document} doc
 * @param {string} name
 * @param {string} text
 */
export function downloadText(doc, name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const link = /** @type {HTMLAnchorElement} */ (doc.createElement('a'));
  link.href = url;
  link.download = name;
  link.hidden = true;
  doc.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * @typedef {object} RecordsOptions
 * @property {Pick<import('./api.js').Api, 'shots' | 'shotRaw'>} api
 * @property {() => number | null | undefined} retention  The record cap.
 * @property {(notice: import('./views.js').Notice | null) => void} [notify]
 * @property {(name: string, text: string) => void} download
 * @property {() => string | null} [model]
 * @property {() => string} [now]  ISO-8601 time of a save.
 * @property {() => void} [onChange]
 */

/**
 * @typedef {object} SaveOutcome
 * @property {boolean} saved  Whether a file was written.
 * @property {string[]} evicted
 * @property {string[]} failed
 * @property {ReturnType<typeof recordsDocument> | null} document
 */

/**
 * The record list and its calls.
 * @param {RecordsOptions} options
 */
export function createRecords({
  api,
  retention,
  notify = () => {},
  download,
  model = () => null,
  now = () => new Date().toISOString(),
  onChange = () => {},
}) {
  /** @type {ShotRecord[]} */
  let records = [];
  let recording = false;
  let saving = false;
  /** Shots read at or before it are never filled in from the list. */
  let floor = Number.POSITIVE_INFINITY;
  /** The newest timestamp of any shot seen on the stream. */
  let lastSeen = Number.NEGATIVE_INFINITY;

  /** @param {Kept} kept */
  function keep(kept) {
    const changed = kept.records !== records;
    records = kept.records;
    for (const id of kept.evicted) {
      notify({ kind: 'info', title: `record ${id} dropped`, detail: `past the retention of ${retention()} shots` });
    }
    if (changed) onChange();
  }

  /** @param {any[]} listed */
  function newest(listed) {
    return listed.reduce((top, shot) => Math.max(top, stamp(shot)), Number.NEGATIVE_INFINITY);
  }

  return {
    records: () => records,
    recording: () => recording,
    saving: () => saving,

    /**
     * Turn recording on or off. Turning it on reads the shots list so the
     * shots already stored are never filled in as missed.
     * @param {boolean} on
     */
    async setRecording(on) {
      recording = on;
      onChange();
      if (!on) return;
      floor = lastSeen;
      const result = await api.shots();
      if (result.kind === 'ok') floor = Math.max(floor, newest(result.body?.shots ?? []));
    },

    /** A shot from the stream. @param {any} shot */
    add(shot) {
      if (shot != null) lastSeen = Math.max(lastSeen, stamp(shot));
      if (!recording) return;
      keep(addRecord(records, shot, retention()));
    },

    /** Fill in what the stream missed from the shots list. */
    async fill() {
      if (!recording) return;
      const result = await api.shots();
      if (result.kind !== 'ok' || !recording) return;
      keep(fillRecords(records, result.body?.shots ?? [], floor, retention()));
    },

    /** Empty the list; nothing read before now is filled in again. */
    flush() {
      const base = Number.isFinite(floor) ? floor : Number.NEGATIVE_INFINITY;
      floor = Math.max(base, lastSeen, ...records.map(stamp));
      records = [];
      onChange();
    },

    /**
     * Fetch every record's raw buffer and write the download.
     * @returns {Promise<SaveOutcome>}
     */
    async save() {
      const saved = records;
      if (saving || saved.length === 0) return { saved: false, evicted: [], failed: [], document: null };
      saving = true;
      onChange();
      try {
        /** @type {Map<string, import('./figures.js').RawData>} */
        const raws = new Map();
        /** @type {string[]} */
        const evicted = [];
        /** @type {string[]} */
        const failed = [];
        const results = await Promise.all(saved.map((r) => api.shotRaw(r.id)));
        results.forEach((result, i) => {
          const { id } = saved[i];
          if (result.kind === 'ok') raws.set(id, result.body);
          else if (result.kind === 'refused' && result.code === RAW_EVICTED) evicted.push(id);
          else failed.push(id);
        });
        const savedAt = now();
        const document = recordsDocument({ records: saved, raws, evicted, failed, model: model(), savedAt });
        download(recordsFileName(savedAt), JSON.stringify(document));
        if (evicted.length > 0) {
          notify({ kind: 'info', title: 'records saved without evicted buffers', detail: evicted.join(', ') });
        }
        if (failed.length > 0) {
          notify({ kind: 'pending', title: 'records saved without unread buffers', detail: failed.join(', ') });
        }
        return { saved: true, evicted, failed, document };
      } finally {
        saving = false;
        onChange();
      }
    },
  };
}

/**
 * The records area: the Record data toggle, the count, Save and Flush.
 * @param {object} state
 * @param {boolean} state.recording
 * @param {boolean} state.saving
 * @param {ShotRecord[]} state.records
 * @param {number | null | undefined} state.retention
 * @param {object} on
 * @param {(on: boolean) => void} on.toggle
 * @param {() => void} on.save
 * @param {() => void} on.flush
 * @returns {HTMLElement}
 */
export function recordsView({ recording, saving, records, retention }, on) {
  const toggle = /** @type {HTMLInputElement} */ (h('input', {
    type: 'checkbox', 'data-control': 'record-toggle', checked: recording,
    onchange: () => on.toggle(toggle.checked),
  }));
  const empty = records.length === 0 ? 'no records' : null;
  const cap = typeof retention === 'number' ? ` of ${retention}` : '';
  return h('section', { class: 'records' },
    h('label', null, toggle, ' Record data'),
    h('p', { 'data-records-count': records.length }, `${records.length}${cap} records`),
    h('ul', { class: 'records-list' }, ...records.map((r) => h('li', { 'data-record': r.id }, r.id))),
    control('records-save', saving ? 'Saving…' : 'Save Records', on.save, saving ? 'saving' : empty),
    control('records-flush', 'Flush Records', on.flush, saving ? 'saving' : empty));
}

/**
 * Fill the `records` area.
 * @type {import('./panel.js').Mount}
 */
export const mount = (host, ctx) => {
  const doc = host.ownerDocument;
  const retention = () => ctx.snapshot()?.retention;
  const records = createRecords({
    api: ctx.api,
    retention,
    notify: ctx.notify,
    model: () => ctx.snapshot()?.model ?? null,
    download: (name, text) => downloadText(doc, name, text),
    onChange: draw,
  });
  let continuing = false;

  function draw() {
    host.replaceChildren(recordsView(
      { recording: records.recording(), saving: records.saving(), records: records.records(), retention: retention() },
      {
        toggle: (on) => void records.setRecording(on),
        save: () => void records.save(),
        flush: () => records.flush(),
      }));
  }

  ctx.onEvent('shot', (payload) => records.add(payload?.shot));
  ctx.onEvent('continuous_ended', () => {
    continuing = false;
    void records.fill();
  });
  ctx.onSnapshot((snapshot) => {
    const running = snapshot?.continuous != null;
    if (continuing && !running) void records.fill();
    continuing = running;
    draw();
  });
  draw();
};
