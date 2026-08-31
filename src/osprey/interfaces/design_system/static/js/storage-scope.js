// @ts-check
/* OSPREY Design System — localStorage scope
 *
 * localStorage is origin-scoped, not path-scoped. On a multi-user deployment
 * every persona is served from the SAME origin under a different path mount
 * (`/u/alice/`, `/u/bob/`), so a bare key like `osprey-ui-mode` is one shared
 * slot: whoever picks last decides what everybody else boots into. The server
 * therefore stamps `data-osprey-storage-scope="<user>"` on `<html>` whenever it
 * serves under such a mount, and every browser-side reader and writer derives
 * its key from that attribute.
 *
 * The attribute is ABSENT — never empty — for single-user serving, and the
 * other interfaces (artifacts, ariel, channel_finder, the dispatch dashboard)
 * load these same scripts and never stamp it at all. Absent therefore has to
 * mean "use the legacy bare key", which is exactly what keeps those pages, and
 * every existing single-user browser profile, working unchanged.
 *
 * NO LEGACY FALLBACK WHEN SCOPED. A scoped read must NOT fall back to the bare
 * key when its own key is missing. The bare key is precisely the polluted slot
 * the scoping exists to escape: reading it as a fallback would hand the last
 * writer's preference to every persona who has not yet made a pick — the
 * original cross-persona bug, re-inflicted once per persona. A scoped page with
 * no scoped value has no stored preference, and resolution falls through to the
 * next rung (the server-rendered attribute, then the built-in default).
 *
 * The pre-paint boot scripts (mode-boot.js, rail-boot.js, the generated
 * theme-boot.js) cannot import this module: they are non-module, dependency-free
 * IIFEs that must run synchronously in <head> ahead of every stylesheet. They
 * INLINE the same two rules — read the attribute, suffix the key, never fall
 * back — and this module is the written-down definition they mirror. Change the
 * rule here and the inline copies must change with it.
 *
 * @module storage-scope
 */

/** The `<html>` attribute the server stamps on a multi-user mount. */
const SCOPE_ATTRIBUTE = 'data-osprey-storage-scope';

/**
 * The storage scope this document is served under, or `null` when it is not
 * scoped at all.
 *
 * An empty attribute value is treated as unscoped. The server never emits one
 * (it omits the attribute entirely rather than rendering `=""`), so this is
 * purely defensive: an empty scope would otherwise produce the key
 * `osprey-ui-mode--`, a third slot belonging to nobody.
 *
 * @returns {string|null}
 */
export function storageScope() {
  try {
    return document.documentElement.getAttribute(SCOPE_ATTRIBUTE) || null;
  } catch {
    return null;
  }
}

/**
 * The localStorage key to use for `base` in this document.
 *
 * Scoped: `base + '--' + scope`. Unscoped: `base`, unchanged — the legacy key.
 * There is deliberately no third case; see the no-legacy-fallback rule in the
 * module docstring for why a scoped reader must never consult `base`.
 *
 * @param {string} base the bare, historically-used key (e.g. `'osprey-ui-mode'`)
 * @returns {string}
 */
export function scopedStorageKey(base) {
  const scope = storageScope();
  return scope === null ? base : `${base}--${scope}`;
}
