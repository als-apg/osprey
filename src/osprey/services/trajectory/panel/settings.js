// @ts-check
/**
 * settings — the one client model of the BPM settings (pilot tone, BPM mode,
 * attenuation, turns tracked) that the Setup BPMs tab and the BPM Control
 * drawer share, so both always render the same values.
 *
 * The model holds the last readbacks it was seeded with and the operator's
 * pending, uncommitted values on top of them, and performs no I/O. A device
 * setting reads as the one value every read BPM agrees on, and as unset when
 * they disagree.
 *
 * {@link settingsFor} binds one model to a panel's {@link AreaContext}: it
 * seeds the model from the setup route's readbacks when first asked for,
 * re-reads them after every setup it commits and whenever the snapshot's
 * `setup.settings_sha` changes. Every area asking for the same context gets
 * the same model, which is how the tab and the drawer share it.
 *
 * @module settings
 */

/** @typedef {number | string | boolean} SettingValue */
/** @typedef {import('./panel.js').AreaContext} AreaContext */

/**
 * One readback's value: a global setting's reading, or the value every read
 * BPM of a device setting agrees on; null when unread or when BPMs disagree.
 * @param {any} setting  A `SettingReadback`.
 * @returns {SettingValue | null}
 */
export function readValue(setting) {
  if (setting?.scope !== 'device') return setting?.value ?? null;
  const read = Object.values(setting.values ?? {}).filter((value) => value != null);
  if (read.length === 0) return null;
  return read.every((value) => value === read[0]) ? /** @type {SettingValue} */ (read[0]) : null;
}

/**
 * @typedef {object} SettingsModel
 * @property {(readbacks: any) => void} seed  Takes the `SetupReadbacks` the setup route answers.
 * @property {() => any} readbacks  The readbacks it was last seeded with, or null.
 * @property {() => Record<string, SettingValue>} values  Readback values with the pending ones on top.
 * @property {() => Record<string, SettingValue>} pending  What the operator changed and has not committed.
 * @property {(id: string, value: SettingValue) => void} set  Stages one setting.
 * @property {() => void} revert  Drops every pending value.
 * @property {(fn: () => void) => () => void} on  Runs *fn* on every change; answers the unsubscribe.
 */

/**
 * A settings model with no readbacks and nothing pending.
 * @returns {SettingsModel}
 */
export function createSettings() {
  /** @type {any} */
  let readbacks = null;
  /** @type {Record<string, SettingValue>} */
  let pending = {};
  /** @type {Set<() => void>} */
  const listeners = new Set();
  const changed = () => {
    for (const fn of [...listeners]) fn();
  };
  return {
    seed(next) {
      readbacks = next;
      changed();
    },
    readbacks: () => readbacks,
    values() {
      /** @type {Record<string, SettingValue>} */
      const values = {};
      for (const setting of readbacks?.settings ?? []) {
        const value = readValue(setting);
        if (value != null) values[setting.id] = value;
      }
      return { ...values, ...pending };
    },
    pending: () => ({ ...pending }),
    set(id, value) {
      pending = { ...pending, [id]: value };
      changed();
    },
    revert() {
      pending = {};
      changed();
    },
    on(fn) {
      listeners.add(fn);
      return () => void listeners.delete(fn);
    },
  };
}

/**
 * @typedef {object} SettingsSync
 * @property {SettingsModel} model  The shared model.
 * @property {() => Promise<import('./api.js').Result>} refresh
 *   Re-reads the readbacks and seeds the model with them.
 * @property {(body: {settings?: Record<string, SettingValue>, excluded_bpms?: string[]}) =>
 *   Promise<import('./api.js').Result>} commit
 *   Posts *body* to the setup route, drops the pending values it carried when
 *   the setup was taken, then re-reads the readbacks whatever the answer.
 */

/** @type {WeakMap<object, SettingsSync>} */
const BOUND = new WeakMap();

/**
 * The settings model bound to *ctx*, created and seeded on first use.
 * @param {AreaContext} ctx
 * @returns {SettingsSync}
 */
export function settingsFor(ctx) {
  const known = BOUND.get(ctx);
  if (known) return known;
  const model = createSettings();
  /** @param {any} snapshot */
  const shaOf = (snapshot) => snapshot?.setup?.settings_sha ?? null;
  let sha = shaOf(ctx.snapshot());

  async function refresh() {
    const result = await ctx.api.setupReadbacks();
    if (result.kind === 'ok') model.seed(result.body);
    return result;
  }

  /** @type {SettingsSync} */
  const sync = {
    model,
    refresh,
    async commit(body) {
      const result = await ctx.api.setup(body);
      if (result.kind === 'ok') {
        const sent = body.settings ?? {};
        const left = model.pending();
        model.revert();
        for (const [id, value] of Object.entries(left)) {
          if (!(id in sent) || sent[id] !== value) model.set(id, value);
        }
      }
      await refresh();
      return result;
    },
  };
  BOUND.set(ctx, sync);
  ctx.onSnapshot((snapshot) => {
    const next = shaOf(snapshot);
    if (next === sha) return;
    sha = next;
    void refresh();
  });
  void refresh();
  return sync;
}
