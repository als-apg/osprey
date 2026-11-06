// @ts-check
/*
 * System Health dashboard — what changes data: applying an envelope, the poll,
 * rescan, row re-probe and provider ping.
 */

import { pageTitle, pingText } from './helpers.js';
import { must, note, now } from './dom.js';
import { fromHash, rerenderDrawer, show } from './drawer.js';
import {
  renderBands,
  renderError,
  renderHeadline,
  renderRing,
  renderScan,
  renderWarming,
} from './overview.js';
import { flashRows, paintBusy } from './rows.js';
import {
  DEFAULT_INTERVAL_S,
  S,
  WARMING_REPOLL_S,
  busyRows,
  indexModel,
  pings,
  store,
} from './state.js';
/** @typedef {import('./state.js').Row} Row */

/**
 * @param {any} body
 * @param {string[]} names Rows to mark busy and flash.
 */
export async function reprobe(body, names) {
  for (const n of names) busyRows.add(n);
  const all = /** @type {HTMLButtonElement | null} */ (
    document.querySelector('.din [data-probeall]')
  );
  if (all) all.disabled = true;
  paintBusy(names);
  try {
    const d = await store.api.reprobe(body);
    for (const n of names) busyRows.delete(n);
    applyEnvelope(d, names);
  } catch (e) {
    for (const n of names) busyRows.delete(n);
    if (all) all.disabled = false;
    paintBusy(names);
    note(e instanceof Error ? e.message : String(e));
  }
}

/** @param {string} provider */
export async function ping(provider) {
  busyRows.add(provider);
  paintBusy([provider]);
  try {
    const p = await store.api.ping(provider);
    const t = pingText(p);
    pings.set(provider, { value: t.value, method: t.method, status: p.status || 'ok', t: now() });
  } catch (e) {
    pings.set(provider, {
      value: e instanceof Error ? e.message : String(e),
      method: '1-token completion',
      status: 'warning',
      t: now(),
    });
  }
  busyRows.delete(provider);
  rerenderDrawer(true);
  flashRows([provider]);
}

/**
 * @param {any} d
 * @param {string[]} [flash] Row names to flash after the re-render.
 */
export function applyEnvelope(d, flash) {
  const firstData = !store.env || store.env.warming;
  store.env = d;
  const title = pageTitle(d.title, store.demo);
  must('ttl').textContent = title;
  document.title = title;
  if (d.warming) {
    renderWarming();
    return;
  }
  store.R = d.results || [];
  store.LAYERS = d.layers || [];
  indexModel();
  renderRing();
  renderHeadline();
  renderScan();
  const hadFocus =
    document.activeElement &&
    document.activeElement.closest &&
    document.activeElement.closest('.t');
  const focusId = hadFocus ? /** @type {HTMLElement} */ (hadFocus).dataset.tile : null;
  renderBands();
  if (focusId) {
    const t = document.getElementById('tile-' + focusId);
    if (t) {
      document.querySelectorAll('.t').forEach((x) => x.setAttribute('tabindex', '-1'));
      t.setAttribute('tabindex', '0');
      t.focus({ preventScroll: true });
    }
  }
  if (S.key && S.band) {
    // The bands were rebuilt: re-attach the open drawer without animating.
    const dr = document.getElementById('drawer-' + S.band);
    if (dr) dr.classList.add('open', 'noanim');
    show(S.band, S.key, undefined, true);
    requestAnimationFrame(() => {
      if (dr) dr.classList.remove('noanim');
    });
  } else if (firstData) {
    fromHash();
  }
  if (flash) flashRows(flash);
}

export function poll() {
  if (store.polling) return;
  store.polling = true;
  if (store.pollTimer !== undefined) clearTimeout(store.pollTimer);
  store.api
    .checks()
    .then((d) => {
      applyEnvelope(d);
      schedule(
        d.warming ? WARMING_REPOLL_S : Math.max(1, Math.round(d.interval_s || DEFAULT_INTERVAL_S))
      );
    })
    .catch((e) => {
      if (!store.env) renderError(e instanceof Error ? e.message : String(e));
      schedule(DEFAULT_INTERVAL_S);
    })
    .finally(() => {
      store.polling = false;
    });
}

/** @param {number} seconds */
export function schedule(seconds) {
  if (store.pollTimer !== undefined) clearTimeout(store.pollTimer);
  store.pollTimer = setTimeout(poll, seconds * 1000);
}

export function rescan() {
  const b = /** @type {HTMLButtonElement} */ (must('rescan'));
  b.disabled = true;
  b.textContent = 'scanning';
  const names = store.R.map((r) => r.name);
  store.api
    .reprobe({})
    .then((d) => applyEnvelope(d, names))
    .catch((e) => note(e instanceof Error ? e.message : String(e)))
    .finally(() => {
      b.disabled = false;
      b.textContent = 'rescan';
    });
}
