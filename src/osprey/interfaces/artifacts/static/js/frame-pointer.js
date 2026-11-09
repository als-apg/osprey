// @ts-check
/**
 * OSPREY Artifact Gallery — a framed artifact page reports the pointer.
 *
 * The content strip over an artifact (artifact-strip.js) shows while the
 * pointer is over the content, but the gallery cannot see a pointer over a
 * plot or table frame: Chromium runs a sandboxed frame in a process of its
 * own and routes the pointer straight to it, so the frame's owner never
 * matches `:hover`, and the frame is not even told reliably when the pointer
 * leaves. So the page tells its parent while the pointer moves over it
 * (`{type: 'osprey-artifact-pointer', over: true}`, at most twice a second)
 * and when it leaves, if the browser says so (`over: false`).
 *
 * The snippets the server injects into plot and table pages (app.py
 * `_RESPONSIVE_PLOTLY`, `_RESPONSIVE_TABLE_HTML`) import this module, inlined
 * as a `data:` module, so it imports nothing itself. A page opened on its own
 * has no parent to tell and reports nothing.
 *
 * @module frame-pointer
 */

/** The message type that tells the gallery the pointer moved over the page or left it. */
export const POINTER_MESSAGE = 'osprey-artifact-pointer';

/** The least time between two reports that the pointer is over the page. */
const REPORT_MS = 500;

/**
 * Tell the parent while the pointer moves over this page, and when it leaves.
 * @param {Window} win the framed page's window
 */
export function reportPointer(win) {
  if (win.parent === win) return;
  const root = win.document.documentElement;
  /** @param {boolean} over */
  const report = (over) => {
    // Intentional '*': an opaque page cannot name the gallery's origin; the
    // parent's WindowProxy pins the receiver and the message is a flag.
    win.parent.postMessage({ type: POINTER_MESSAGE, over }, '*');
  };
  let reportedAt = -Infinity;
  root.addEventListener('pointermove', () => {
    const now = Date.now();
    if (now - reportedAt < REPORT_MS) return;
    reportedAt = now;
    report(true);
  });
  root.addEventListener('pointerleave', () => {
    reportedAt = -Infinity;
    report(false);
  });
}
