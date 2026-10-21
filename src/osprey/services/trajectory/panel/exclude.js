// @ts-check
/**
 * exclude — the exclude-list grammar the Setup BPMs and Setup CMs tabs read.
 *
 * An exclude list addresses devices by their 1-based index in a build-time
 * catalog family (`Snapshot.catalog`, ring order). The grammar accepts an optional pair of
 * square brackets around the list, space or comma separators (any mix), and
 * inclusive ranges like `3:5`. An index outside the family, a range running
 * backwards, or a token that is neither an index nor a range is refused with
 * a message naming it; nothing is excluded from a list that does not parse.
 *
 * The routes take device names, so a parsed list answers both its indices and
 * the names they stand for.
 *
 * @module exclude
 */

/**
 * @typedef {{ok: true, indices: number[], names: string[]}} ParsedExclude
 * @typedef {{ok: false, error: string}} RefusedExclude
 * @typedef {ParsedExclude | RefusedExclude} ExcludeParse
 */

const INDEX = /^\d+$/;
const RANGE = /^(\d+):(\d+)$/;

/**
 * Read an exclude list against *family*.
 * @param {string} text  What the operator typed.
 * @param {readonly string[]} family  The family's device names in index order.
 * @param {string} [label]  The family's name in a refusal, `BPM` by default.
 * @returns {ExcludeParse}
 */
export function parseExclude(text, family, label = 'BPM') {
  let body = String(text ?? '').trim();
  if (body.startsWith('[') || body.endsWith(']')) {
    if (!(body.startsWith('[') && body.endsWith(']'))) {
      return { ok: false, error: `unbalanced brackets in "${body}"` };
    }
    body = body.slice(1, -1).trim();
  }
  const size = family.length;
  /** @type {Set<number>} */
  const picked = new Set();
  /** @param {number} index */
  const outside = (index) =>
    ({ ok: /** @type {const} */ (false), error: `${label} ${index} is not in the ${label} family (1-${size})` });
  for (const token of body.split(/[\s,]+/).filter(Boolean)) {
    const range = RANGE.exec(token);
    if (range) {
      const lo = Number(range[1]);
      const hi = Number(range[2]);
      if (lo > hi) return { ok: false, error: `range ${token} runs backwards` };
      for (const end of [lo, hi]) if (end < 1 || end > size) return outside(end);
      for (let index = lo; index <= hi; index++) picked.add(index);
    } else if (INDEX.test(token)) {
      const index = Number(token);
      if (index < 1 || index > size) return outside(index);
      picked.add(index);
    } else {
      return { ok: false, error: `"${token}" is not an index or a range like 3:5` };
    }
  }
  const indices = [...picked].sort((a, b) => a - b);
  return { ok: true, indices, names: indices.map((index) => family[index - 1]) };
}

/**
 * The 1-based indices of *names* in *family*, ascending; names the family does
 * not hold are dropped.
 * @param {readonly string[]} names
 * @param {readonly string[]} family
 * @returns {number[]}
 */
export function indicesOf(names, family) {
  const position = new Map(family.map((name, i) => [name, i + 1]));
  const indices = new Set();
  for (const name of names) {
    const index = position.get(name);
    if (index != null) indices.add(index);
  }
  return [...indices].sort((a, b) => a - b);
}

/**
 * Write *indices* back in the grammar: ascending, three or more consecutive
 * indices as a range, separated by spaces. `parseExclude` reads it back to
 * the same indices.
 * @param {readonly number[]} indices
 * @returns {string}
 */
export function formatExclude(indices) {
  const sorted = [...new Set(indices)].sort((a, b) => a - b);
  /** @type {string[]} */
  const parts = [];
  for (let i = 0; i < sorted.length; ) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    if (j - i >= 2) parts.push(`${sorted[i]}:${sorted[j]}`);
    else for (let k = i; k <= j; k++) parts.push(String(sorted[k]));
    i = j + 1;
  }
  return parts.join(' ');
}

/**
 * *names* with *name* added when absent and removed when present, in family
 * order.
 * @param {readonly string[]} names
 * @param {string} name
 * @param {readonly string[]} family
 * @returns {string[]}
 */
export function toggleName(names, name, family) {
  const next = new Set(names);
  if (next.has(name)) next.delete(name);
  else next.add(name);
  return indicesOf([...next], family).map((index) => family[index - 1]);
}
