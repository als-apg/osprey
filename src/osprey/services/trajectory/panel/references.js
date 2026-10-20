// @ts-check
/**
 * references — Select Current as Reference, Save Reference to File and
 * Load Reference from file.
 *
 * Select Current saves the latest shot under the typed name and selects it;
 * only a fresh TbT shot becomes a reference, so any other latest shot
 * disables the control with its reason. The latest shot is the newest `shot`
 * event, or after a reload the snapshot's last shot, looked up in the stored
 * shots for its freshness and readout.
 *
 * Save to File downloads the selected reference's readings route answer,
 * which is exactly the import body, so Load from file uploads such a file
 * unchanged apart from `select`, and the loaded reference becomes current.
 *
 * Every refusal (the server's, by its code, or a file that is not a
 * reference) is shown inline in the area; selecting needs the session, so
 * the two selecting controls are disabled for a tab that does not hold it.
 *
 * Mounted into the `references` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point for the shell.
 *
 * @module references
 */

import { control, h } from './views.js';

/** The readout a reference is taken in. */
export const REFERENCE_READOUT = 'tbt';

/**
 * The latest shot, as far as this area judges it.
 * @typedef {object} LatestShot
 * @property {string} shot_id
 * @property {boolean} [fresh]
 * @property {string} [readout]
 */

/**
 * Why Select Current cannot be used now, or null when it can.
 * @param {LatestShot | null} shot  The latest shot, null when none is known.
 * @param {boolean} holding  Whether this tab holds the session.
 * @returns {string | null}
 */
export function selectCurrentReason(shot, holding) {
  if (shot == null) return 'Take a shot first: there is no shot to make a reference of.';
  if (shot.fresh === false) {
    return `Shot ${shot.shot_id} is not fresh (its action did not arm and inject with the machine held still).`;
  }
  if (shot.readout != null && shot.readout !== REFERENCE_READOUT) {
    return `Shot ${shot.shot_id} was read as ${shot.readout}; a reference is a ${REFERENCE_READOUT} shot.`;
  }
  if (!holding) return 'Hold the session to select a reference.';
  return null;
}

/**
 * Why Save to File cannot be used now, or null when it can.
 * @param {{ reference?: { name: string } | null } | null} snapshot
 * @returns {string | null}
 */
export function saveFileReason(snapshot) {
  return snapshot?.reference ? null : 'Select a reference first.';
}

/**
 * The file a reference is saved as.
 * @param {string} name
 */
export function fileName(name) {
  return `${name}.json`;
}

/**
 * The file content of a reference: the readings route's answer as it came.
 * @param {import('./api.js').ReferenceImport} readings
 */
export function referenceFile(readings) {
  return `${JSON.stringify(readings, null, 2)}\n`;
}

/**
 * The import body a file's text makes, selecting it; or why it makes none.
 * The content is passed on as written: the server judges its BPMs, readout
 * and size.
 * @param {string} text
 * @returns {{ body: import('./api.js').ReferenceImport, error?: undefined }
 *   | { error: string, body?: undefined }}
 */
export function importBody(text) {
  /** @type {unknown} */
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { error: 'The file is not JSON; a reference file is what Save Reference to File writes.' };
  }
  if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { error: 'The file is not a JSON object; a reference file is what Save Reference to File writes.' };
  }
  return { body: /** @type {import('./api.js').ReferenceImport} */ ({ ...parsed, select: true }) };
}

/**
 * A default reference name for the moment *now*.
 * @param {Date} now
 */
export function defaultName(now) {
  const pad = (/** @type {number} */ n) => String(n).padStart(2, '0');
  return `ref-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-`
    + `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

/**
 * Hand *text* to the browser as a download named *name*.
 * @param {string} name
 * @param {string} text
 */
function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const anchor = /** @type {HTMLAnchorElement} */ (h('a', { href: url, download: name }));
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * Set or clear a control's refusal reason.
 * @param {HTMLButtonElement} button
 * @param {string | null} reason
 */
function setReason(button, reason) {
  button.disabled = reason != null;
  if (reason == null) {
    button.removeAttribute('title');
    button.removeAttribute('data-reason');
  } else {
    button.title = reason;
    button.dataset.reason = reason;
  }
}

/**
 * Fill the `references` area.
 * @type {import('./panel.js').Mount}
 */
export const mount = (host, ctx) => {
  /** @type {LatestShot | null} */
  let latest = null;
  /** The shot id a stored-shots lookup was last made for. */
  let lookedUp = /** @type {string | null} */ (null);

  const current = h('div', { class: 'reference-current', 'data-role': 'reference-current' });
  const refusal = h('div', { class: 'reference-refusal', 'data-role': 'reference-refusal', role: 'alert' });
  const name = /** @type {HTMLInputElement} */ (h('input', {
    type: 'text', 'data-control': 'reference-name', placeholder: 'reference name',
    'aria-label': 'Reference name', maxLength: 64,
  }));
  const file = /** @type {HTMLInputElement} */ (h('input', {
    type: 'file', 'data-control': 'reference-file', accept: '.json,application/json', hidden: true,
  }));
  const selectCurrent = control('ref-select-current', 'Select Current as Reference',
    () => void onSelectCurrent());
  const saveFile = control('ref-save-file', 'Save Reference to File', () => void onSaveFile());
  const loadFile = control('ref-load-file', 'Load Reference from file', () => file.click());
  file.addEventListener('change', () => void onLoadFile());

  host.replaceChildren(h('div', { class: 'references' },
    current, h('div', { class: 'reference-select' }, name, selectCurrent),
    h('div', { class: 'reference-files' }, saveFile, loadFile, file), refusal));

  /**
   * Show a refusal inline, or clear it.
   * @param {{ code?: string | null, detail: string } | null} what
   */
  function show(what) {
    if (what == null) {
      refusal.replaceChildren();
      refusal.removeAttribute('data-code');
      return;
    }
    if (what.code) refusal.setAttribute('data-code', what.code);
    else refusal.removeAttribute('data-code');
    refusal.replaceChildren(
      ...(what.code ? [h('code', { class: 'reference-refusal-code' }, what.code), ' '] : []),
      what.detail);
  }

  /**
   * Show what a non-ok result says; answers whether it was ok.
   * @param {string} label
   * @param {import('./api.js').Result} result
   */
  function judged(label, result) {
    if (result.kind === 'ok') return true;
    if (result.kind === 'refused') show({ code: result.code, detail: `${label} refused: ${result.detail}` });
    else show({ detail: `${label}: ${result.message}` });
    return false;
  }

  function render() {
    const snapshot = ctx.snapshot();
    const holding = ctx.holding();
    const reference = snapshot?.reference;
    current.replaceChildren(reference
      ? `Reference: ${reference.name} (${reference.matched?.length ?? 0} BPMs matched)`
      : 'No reference selected.');
    setReason(selectCurrent, selectCurrentReason(latest, holding));
    setReason(saveFile, saveFileReason(snapshot));
    setReason(loadFile, holding ? null : 'Hold the session to select a reference.');
  }

  /** After a reload the snapshot names the last shot; the stored shots say what it was. */
  async function lookUp(/** @type {string} */ shotId) {
    lookedUp = shotId;
    const result = await ctx.api.shots();
    if (result.kind !== 'ok' || latest?.shot_id !== shotId) return;
    const found = (result.body?.shots ?? []).find((/** @type {any} */ s) => s.shot_id === shotId);
    if (found) {
      latest = { shot_id: shotId, fresh: found.fresh, readout: found.readout };
      render();
    }
  }

  async function onSelectCurrent() {
    if (latest == null) return;
    const chosen = name.value.trim() || defaultName(new Date());
    show(null);
    if (!judged('Save reference', await ctx.api.saveReference(chosen, latest.shot_id))) return;
    judged('Select reference', await ctx.api.selectReference(chosen));
  }

  async function onSaveFile() {
    const reference = ctx.snapshot()?.reference;
    if (!reference) return;
    show(null);
    const result = await ctx.api.referenceReadings(reference.name);
    if (!judged('Save reference to file', result) || result.kind !== 'ok') return;
    download(fileName(reference.name), referenceFile(result.body));
  }

  async function onLoadFile() {
    const chosen = file.files?.[0];
    file.value = '';
    if (!chosen) return;
    show(null);
    const { body, error } = importBody(await chosen.text());
    if (body == null) {
      show({ detail: `Load reference refused: ${error}` });
      return;
    }
    judged('Load reference', await ctx.api.importReference(body));
  }

  ctx.onEvent('shot', (payload) => {
    const shot = payload?.shot;
    if (shot?.shot_id == null) return;
    latest = { shot_id: shot.shot_id, fresh: shot.fresh, readout: shot.readout };
    lookedUp = shot.shot_id;
    render();
  });
  ctx.onSnapshot((snapshot) => {
    const last = snapshot?.last_shot?.shot_id;
    if (last != null && last !== latest?.shot_id) {
      latest = { shot_id: last };
      if (lookedUp !== last) void lookUp(last);
    }
    render();
  });
  render();
};
