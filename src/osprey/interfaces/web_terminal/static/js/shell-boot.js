// @ts-check
/* OSPREY Web Terminal — the loading shell's watch, before any module
 *
 * The hub ships `<html data-boot="loading">`: the sidebar is drawn but inert
 * and the workspace is covered by the loading veil (terminal.css, "Loading
 * shell") until shell-reveal.js reveals the page in one step. This script is
 * what keeps that state honest while the module graph is still on the wire.
 * Deliberately NOT a module, like theme-boot.js: it must run before the first
 * paint and must not wait for anything.
 *
 *   - EARLY READS. The two reads start-up cannot begin without are requested
 *     here, in parallel with the scripts, instead of after them:
 *     `window.__OSPREY_EARLY__` maps a path to the promise of its JSON
 *     (api.js `fetchJSON` takes it; shell-reveal.js drops it at the reveal).
 *   - PROGRESS. The veil's bar fills as the files the page names arrive.
 *   - SLOW. After SLOW_NOTICE_MS the veil says the connection is slow.
 *   - FAILED. A script or stylesheet of the page that cannot be fetched ends
 *     the wait: the veil says so and offers a reload, `data-boot="failed"`.
 *   - FAIL-OPEN. Once the scripts have run, the veil lifts after
 *     FAIL_OPEN_MS whatever start-up is still waiting on. The veil is never
 *     the thing between an operator and the console.
 */
(function () {
  'use strict';

  const BOOT_ATTRIBUTE = 'data-boot';
  const SLOW_NOTICE_MS = 8000;
  const FAIL_OPEN_MS = 10000;
  const EARLY_READS = ['/api/sessions/last-active', '/api/panels'];

  const root = document.documentElement;
  if (root.getAttribute(BOOT_ATTRIBUTE) !== 'loading') return;

  /** @returns {boolean} */
  function loading() {
    return root.getAttribute(BOOT_ATTRIBUTE) === 'loading';
  }

  // ---- Early reads ----
  const prefix = /** @type {any} */ (window).__OSPREY_PREFIX__ || '';
  /** @type {Record<string, Promise<any>>} */
  const early = {};
  for (const path of EARLY_READS) {
    try {
      const read = fetch(prefix + path, { cache: 'no-store' }).then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.json();
      });
      // A read nobody takes must not surface as an unhandled rejection; the
      // module that takes it sees the rejection on its own branch.
      read.catch(() => {});
      early[path] = read;
    } catch {
      /* no fetch: start-up reads it the ordinary way */
    }
  }
  /** @type {any} */ (window).__OSPREY_EARLY__ = early;

  // ---- The veil's words ----
  /**
   * @param {string} status
   * @param {string} detail
   * @param {boolean} offerReload
   */
  function say(status, detail, offerReload) {
    const veil = document.getElementById('boot-veil');
    if (!veil) return;
    const statusEl = veil.querySelector('[data-boot-status]');
    const detailEl = /** @type {HTMLElement | null} */ (veil.querySelector('[data-boot-detail]'));
    const reload = /** @type {HTMLButtonElement | null} */ (
      veil.querySelector('[data-boot-reload]')
    );
    if (statusEl && status) statusEl.textContent = status;
    if (detailEl) {
      detailEl.textContent = detail;
      detailEl.hidden = !detail;
    }
    if (reload) {
      reload.hidden = !offerReload;
      reload.onclick = () => window.location.reload();
    }
  }

  // ---- Progress ----
  // The page names every module it will load as a modulepreload link, and the
  // browser reports each one as it arrives: arrived / named is how far the
  // load is. The files fill the bar to FILES_SHARE; the rest is start-up and
  // completes at the reveal (shell-reveal.js). Until a first file is reported
  // the bar keeps its indeterminate sweep, so a browser that reports none
  // shows a moving bar and never a stuck one.
  const FILES_SHARE = 0.9;
  /** @type {Set<string>} */
  const arrived = new Set();
  let shown = 0;

  /** @param {number} fraction */
  function progress(fraction) {
    shown = Math.max(shown, Math.min(fraction, 1));
    const bar = /** @type {HTMLElement | null} */ (document.querySelector('[data-boot-bar]'));
    if (!bar) return;
    bar.style.setProperty('--boot-progress', shown.toFixed(3));
    bar.setAttribute('data-progress', '');
    bar.setAttribute('aria-valuenow', String(Math.round(shown * 100)));
  }

  function measure() {
    if (!loading() || !arrived.size) return;
    /** @type {NodeListOf<HTMLLinkElement>} */
    const named = document.querySelectorAll('link[rel="modulepreload"]');
    let landed = 0;
    for (const link of named) if (arrived.has(link.href)) landed += 1;
    if (named.length) progress((landed / named.length) * FILES_SHARE);
  }

  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) arrived.add(entry.name);
      measure();
    }).observe({ type: 'resource', buffered: true });
  } catch {
    /* no resource timings: the bar stays indeterminate */
  }
  // Files that arrived before the veil was parsed are painted now.
  document.addEventListener('DOMContentLoaded', measure);

  // ---- Slow ----
  setTimeout(() => {
    if (loading()) say('', 'Still loading. The connection to this server is slow.', false);
  }, SLOW_NOTICE_MS);

  // ---- Failed ----
  // A resource error does not bubble; it is caught on the way down.
  window.addEventListener(
    'error',
    (event) => {
      const target = event.target;
      if (!(target instanceof HTMLScriptElement || target instanceof HTMLLinkElement)) return;
      // Only the page's own files. A vendor bundle, from a CDN or from this
      // server's /vendor/, has its own report and fallback where it is used
      // (terminal.js, dock-workspace.js): the hub runs without dockview.
      const url = target instanceof HTMLScriptElement ? target.src : target.href;
      if (!url) return;
      const file = new URL(url, window.location.href);
      if (file.origin !== window.location.origin || file.pathname.includes('/vendor/')) return;
      if (target instanceof HTMLLinkElement && target.rel !== 'stylesheet') return;
      if (!loading()) return;
      root.setAttribute(BOOT_ATTRIBUTE, 'failed');
      say(
        'The workspace did not load',
        'A file of this page could not be fetched from the server.',
        true
      );
    },
    true
  );

  // ---- Fail-open ----
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
      if (!loading()) return;
      progress(1);
      root.setAttribute(BOOT_ATTRIBUTE, 'revealed');
    }, FAIL_OPEN_MS);
  });
})();
