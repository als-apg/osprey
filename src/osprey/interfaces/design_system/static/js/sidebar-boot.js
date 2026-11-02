// @ts-check
/* OSPREY Design System — Sidebar Boot
 *
 * Hand-written (not generated) sibling of theme-boot.js for the sidebar width:
 * it stamps data-sidebar="collapsed" onto <html> before first paint when the
 * viewer collapsed the sidebar last time, so the shell never flashes the
 * labelled 232 px sidebar before it snaps to 56 px icons.
 *
 * Deliberately NOT an ES module — module scripts are deferred, which would
 * let a wrong-width flash slip through — and dependency-free (imports nothing)
 * for the same reason: it must run synchronously in <head>, ahead of every
 * stylesheet.
 *
 * Resolution: localStorage['osprey-sidebar'], per-render scoped (see below).
 * 'collapsed' stamps data-sidebar="collapsed"; any other value, or none,
 * leaves <html> untouched, because an absent attribute means expanded.
 * collapsible.js is the WRITER of this key (it stores 'collapsed' or
 * 'expanded' through storage-scope.js), so the two must derive byte-identical
 * keys.
 *
 * It also restores the labelled width the web terminal's resize handle stored
 * (localStorage['osprey-sidebar-width'], same scoping, writer
 * sidebar-resize.js): a whole number of px in 180–400 becomes the
 * `--sidebar-user-width` property on <html>, capped so the workspace keeps
 * 480 px; anything else is ignored. The limits mirror sidebar-resize.js.
 *
 * Storage, scoped: localStorage is origin-scoped, so on a multi-user
 * deployment (every render served from one origin under `/u/<user>/`) a bare
 * key is a single shared slot and the last render to collapse the sidebar
 * decides everyone else's width. When the server stamps
 * data-osprey-storage-scope on <html>, this reads `osprey-sidebar--<scope>`
 * instead — and does NOT fall back to the bare key, since that polluted slot
 * is the very thing being escaped; a scoped page with no scoped value boots
 * expanded. With the attribute absent (single-user serving, where nothing is
 * stamped) the bare key is used.
 *
 * This duplicates storage-scope.js's `scopedStorageKey()` inline: as a
 * pre-paint IIFE this file imports nothing, so it mirrors the rule rather than
 * calling it. storage-scope.js is the written-down definition — keep them in
 * step.
 */
(function () {
  'use strict';

  const STORAGE_KEY_BASE = 'osprey-sidebar';
  const WIDTH_KEY_BASE = 'osprey-sidebar-width';
  const SCOPE_ATTRIBUTE = 'data-osprey-storage-scope';
  const MIN_WIDTH = 180;
  const MAX_WIDTH = 400;
  const WORKSPACE_MIN_WIDTH = 480;

  // Inline mirror of storage-scope.js's `scopedStorageKey()`. An empty
  // attribute value counts as unscoped: the server omits the attribute rather
  // than rendering `=""`, so this only guards against a key ending in a bare
  // "--" that would belong to no render.
  /** @param {string} base */
  function storageKey(base) {
    try {
      const scope = document.documentElement.getAttribute(SCOPE_ATTRIBUTE);
      return scope ? base + '--' + scope : base;
    } catch {
      return base;
    }
  }

  /** @param {string} base */
  function readStored(base) {
    try {
      return window.localStorage.getItem(storageKey(base));
    } catch {
      return null;
    }
  }

  if (readStored(STORAGE_KEY_BASE) === 'collapsed') {
    document.documentElement.setAttribute('data-sidebar', 'collapsed');
  }

  const width = readStored(WIDTH_KEY_BASE);
  if (width !== null && /^\d+$/.test(width.trim())) {
    const px = Number(width);
    if (px >= MIN_WIDTH && px <= MAX_WIDTH) {
      const cap = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, window.innerWidth - WORKSPACE_MIN_WIDTH));
      document.documentElement.style.setProperty('--sidebar-user-width', Math.min(px, cap) + 'px');
    }
  }
})();
