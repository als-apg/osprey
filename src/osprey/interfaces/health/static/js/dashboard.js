// @ts-check
/*
 * System Health dashboard — entry module.
 *
 * Paints the /checks envelope as one band per OSPREY layer (ring and headline
 * on top), each component a tile that opens its detail in place, in a drawer
 * under its band. Colors come from dashboard.css through status classes; no
 * color is set from JS. Theme is initialized in index.html's <head>. This
 * module wires the keyboard and boots the poll; the work lives in the sibling
 * modules.
 */

import { poll, rescan } from './actions.js';
import { makeApi } from './api.js';
import { must } from './dom.js';
import { closeDrawer, fromHash, placeCaret, updAge } from './drawer.js';
import { renderDrill } from './drill.js';
import { S, store } from './state.js';

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && S.key) {
    closeDrawer();
    return;
  }
  const ae = document.activeElement;
  const cur = ae && ae.closest ? /** @type {HTMLElement | null} */ (ae.closest('.t')) : null;
  if (!cur || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
  e.preventDefault();
  const all = /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.t')]);
  /** @type {HTMLElement | null} */
  let nxt = null;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    nxt = all[all.indexOf(cur) + (e.key === 'ArrowRight' ? 1 : -1)] || null;
  } else {
    const bands = [...document.querySelectorAll('.band')];
    const bi = bands.indexOf(/** @type {Element} */ (cur.closest('.band')));
    const tb = bands[bi + (e.key === 'ArrowDown' ? 1 : -1)];
    if (tb) {
      const r0 = cur.getBoundingClientRect();
      const cx = r0.left + r0.width / 2;
      let bd = Infinity;
      tb.querySelectorAll('.t').forEach((x) => {
        const r = x.getBoundingClientRect();
        const dd = Math.abs(r.left + r.width / 2 - cx);
        if (dd < bd) {
          bd = dd;
          nxt = /** @type {HTMLElement} */ (x);
        }
      });
    }
  }
  if (nxt) {
    all.forEach((x) => x.setAttribute('tabindex', '-1'));
    /** @type {HTMLElement} */ (nxt).setAttribute('tabindex', '0');
    /** @type {HTMLElement} */ (nxt).focus();
  }
});

async function boot() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('demo')) {
    const { demoApi } = await import('./demo.js');
    store.api = demoApi();
    store.demo = true;
    must('demo').hidden = false;
  } else {
    store.api = makeApi();
  }
  must('rescan').addEventListener('click', rescan);
  const issues = must('issues');
  issues.addEventListener('click', () => {
    const on = issues.getAttribute('aria-pressed') !== 'true';
    issues.setAttribute('aria-pressed', String(on));
    document.body.classList.toggle('issues-only', on);
  });
  window.addEventListener('resize', () => {
    placeCaret();
    if (S.key && store.NODES[S.key]) renderDrill(S.key);
  });
  window.addEventListener('hashchange', fromHash);
  setInterval(updAge, 1000);
  poll();
}

boot();
