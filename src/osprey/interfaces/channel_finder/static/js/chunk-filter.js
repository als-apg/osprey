// @ts-check
/**
 * OSPREY Channel Finder — In-Context filter + chunk helpers (pure logic).
 *
 * Extracted from explore-in-context.js so the bug-critical paging logic has no
 * DOM/network dependencies and can be unit-tested under Vitest (`npm run test:js`)
 * (see tests/interfaces/channel_finder/chunk-filter.test.mjs).
 *
 * The invariant these helpers enforce: filtering and pagination derive from the
 * SAME filtered set. Filter the whole database first, THEN chunk the result —
 * the reverse (chunk first, filter the loaded chunk) is what caused issue #299, (hygiene-allow-color: issue number, not a hex color)
 * where a match on a later page read as "No channels match the filter".
 */

/**
 * Filter channels by a lowercased query over name / address / description.
 * Mirrors the field fallbacks used when rendering rows.
 *
 * @param {Array<Record<string, any>>} channels - the full channel set
 * @param {string} filterText - already-lowercased query ('' returns all)
 * @returns {Array<Record<string, any>>} matching channels (the same object references)
 */
export function filterChannels(channels, filterText) {
  if (!filterText) return channels;
  return channels.filter(ch => {
    const name = (ch.name || ch.channel_name || ch.channel || '').toLowerCase();
    const addr = (ch.address || ch.pv_address || '').toLowerCase();
    const desc = (ch.description || '').toLowerCase();
    return name.includes(filterText) || addr.includes(filterText) || desc.includes(filterText);
  });
}

/**
 * Number of chunks needed to page `count` items at `chunkSize` per page.
 * Always at least 1 so the UI never reports "0 chunks".
 *
 * @param {number} count
 * @param {number} chunkSize
 * @returns {number}
 */
export function totalChunksFor(count, chunkSize) {
  return Math.max(1, Math.ceil(count / chunkSize));
}

/**
 * Clamp a chunk index into the valid range for `count` filtered items.
 * Prevents stranding the operator on an empty page after the filtered set
 * shrinks (narrowed query, or a delete that emptied the last page).
 *
 * @param {number} chunkIdx
 * @param {number} count
 * @param {number} chunkSize
 * @returns {number}
 */
export function clampChunkIdx(chunkIdx, count, chunkSize) {
  const total = totalChunksFor(count, chunkSize);
  return Math.min(Math.max(0, chunkIdx), total - 1);
}

/**
 * Return the page-slice of `items` for the given chunk index.
 *
 * @param {Array<Record<string, any>>} items
 * @param {number} chunkIdx
 * @param {number} chunkSize
 * @returns {Array<Record<string, any>>}
 */
export function pageSlice(items, chunkIdx, chunkSize) {
  const start = chunkIdx * chunkSize;
  return items.slice(start, start + chunkSize);
}
