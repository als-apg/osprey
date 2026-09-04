// @ts-check
/**
 * Shared clipboard helper for the design-system front-end.
 *
 * Two-rung ladder, cheapest first: `navigator.clipboard.writeText` when the
 * API is present and the write resolves, falling back to a hidden
 * `<textarea>` + `document.execCommand('copy')` otherwise (including when
 * `writeText` rejects — a denied permission prompt, a non-secure context, or
 * any other runtime failure). Never throws; callers get a boolean instead.
 *
 * `feedback-boot.js` in the web terminal owns its own, separate clipboard
 * ladder (it has payload-specific fallback UI beyond a boolean) — this
 * module is for design-system consumers that just need "copy this string,
 * tell me if it worked."
 *
 * @module clipboard
 */

/**
 * Copy `text` to the system clipboard.
 *
 * Tries `navigator.clipboard.writeText` first; if that API is absent or its
 * write rejects, falls back to selecting a temporary off-screen `<textarea>`
 * and running `document.execCommand('copy')`. The textarea is always removed
 * before this resolves, on every path (success, `execCommand` returning
 * false, or `execCommand` throwing).
 *
 * @param {string} text - the string to place on the clipboard.
 * @returns {Promise<boolean>} whether either rung reported success.
 */
export async function copyText(text) {
  if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the textarea rung below.
    }
  }
  return copyTextViaTextarea(text);
}

/**
 * Fallback rung: a hidden, selected `<textarea>` plus `document.execCommand`.
 *
 * @param {string} text
 * @returns {boolean}
 */
function copyTextViaTextarea(text) {
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.top = '-1000px';
  area.style.left = '-1000px';
  area.style.opacity = '0';
  document.body.appendChild(area);
  try {
    area.select();
    area.setSelectionRange(0, area.value.length);
    return typeof document.execCommand === 'function' && document.execCommand('copy');
  } catch {
    return false;
  } finally {
    area.remove();
  }
}
