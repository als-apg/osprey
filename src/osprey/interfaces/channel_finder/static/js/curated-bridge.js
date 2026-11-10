// @ts-check
/**
 * OSPREY Channel Finder — the curated views' way into All PVs.
 *
 * Under curated results sits a quiet "Not here?" line that opens All PVs with
 * the same words; a curated search that matches nothing offers the search of
 * the whole directory instead. A curated search writes its words to
 * `state.scopeQuery` and reports `curatedResults`. All of it exists only
 * while `state.directory` is set; without the section every function here is
 * a no-op and draws nothing.
 */

import { state } from './state.js';
import { esc } from './utils.js';
import { countDirectory } from './directory-api.js';

/**
 * The All PVs route for `words`.
 * @param {string} words
 * @returns {string}
 */
export function allPvsHash(words) {
  const text = words.trim();
  return text ? `#/allpvs?q=${encodeURIComponent(text)}` : '#/allpvs';
}

/**
 * The line under curated results: "Not here?" with the route to All PVs, or,
 * when the curated search matched nothing, the offer to search the whole
 * directory. Empty without the section.
 * @param {string} words - The curated search's words.
 * @param {number|null} total - Its hits; null for a view with no search.
 * @param {number|null} size - PVs in the whole directory; null while unknown.
 * @returns {string}
 */
export function bridgeHtml(words, total, size) {
  const directory = state.directory;
  if (!directory) return '';
  const href = esc(allPvsHash(words));
  const text = words.trim();
  if (total === 0) {
    const all = size === null ? 'all' : `all ${size.toLocaleString()}`;
    const what = text ? ` for "${text}"` : '';
    return (
      `<div class="curated-bridge curated-bridge-miss">` +
      `<a href="${href}">Search ${esc(all)} PVs${esc(what)} &rarr;</a></div>`
    );
  }
  return (
    `<div class="curated-bridge">Not here? ` +
    `<a href="${href}">Search ${esc(directory.label)}</a></div>`
  );
}

/**
 * Record a curated search: its words become the shared query and its hits are
 * reported. Nothing without the section.
 * @param {string} words
 * @param {number} total
 * @returns {void}
 */
export function reportCurated(words, total) {
  if (!state.directory) return;
  state.scopeQuery = words;
  state.emit('curatedResults', { query: words, total });
}

/**
 * The size of the current target's whole directory, or null when the count
 * did not answer (or the section is absent, which asks nothing).
 * @returns {Promise<number|null>}
 */
export async function directorySize() {
  if (!state.directory) return null;
  try {
    const answer = await countDirectory();
    return answer.outcome === 'found' && typeof answer.total === 'number' ? answer.total : null;
  } catch {
    return null;
  }
}
