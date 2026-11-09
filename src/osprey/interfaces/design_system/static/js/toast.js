// @ts-check
/* OSPREY Design System — shared toast layer.
 *
 * One `div.osprey-toast-host`, appended to <body> on first use and shared by
 * every caller, sits at the bottom centre of the window. It holds two live
 * regions that stay in the page once created, so a screen reader is already
 * listening when a message lands:
 *   - `role="status"` (polite) for info, success and warning;
 *   - `role="alert"` (assertive) for error.
 *
 * `showToast(message, {kind, timeout})` queues a message. One toast shows at a
 * time, in call order, so toasts never overlap; each hides after its own
 * timeout (about 3 s by default) and the next one takes its place. The message
 * is set as text, never as markup.
 *
 * Without a DOM (a worker, a server-side import) both calls do nothing.
 *
 * @module toast
 */

/** @typedef {'info' | 'success' | 'warning' | 'error'} ToastKind */
/** @typedef {{ message: string, kind: ToastKind, timeout: number }} QueuedToast */

/** How long a toast shows when the caller names no timeout. */
export const TOAST_TIMEOUT_MS = 3000;

const KINDS = new Set(['info', 'success', 'warning', 'error']);

/** @type {QueuedToast[]} */
let queue = [];
/** @type {HTMLElement | null} */
let current = null;
/** @type {ReturnType<typeof setTimeout> | null} */
let timer = null;

function hasDom() {
  return typeof document !== 'undefined' && document !== null && !!document.body;
}

/**
 * The host with its two regions, created on first use and rebuilt if a page
 * removed it.
 * @returns {{ status: HTMLElement, alert: HTMLElement }}
 */
function regions() {
  let host = /** @type {HTMLElement | null} */ (document.querySelector('.osprey-toast-host'));
  if (!host) {
    host = document.createElement('div');
    host.className = 'osprey-toast-host';
    for (const [role, live] of [
      ['status', 'polite'],
      ['alert', 'assertive'],
    ]) {
      const region = document.createElement('div');
      region.className = `osprey-toast-region osprey-toast-region--${role}`;
      region.setAttribute('role', role);
      region.setAttribute('aria-live', live);
      region.setAttribute('aria-atomic', 'true');
      host.appendChild(region);
    }
    document.body.appendChild(host);
  }
  return {
    status: /** @type {HTMLElement} */ (host.querySelector('[role="status"]')),
    alert: /** @type {HTMLElement} */ (host.querySelector('[role="alert"]')),
  };
}

function showNext() {
  const next = queue.shift();
  if (!next) return;
  const { status, alert } = regions();
  const el = document.createElement('div');
  el.className = `osprey-toast osprey-toast--${next.kind}`;
  el.dataset.kind = next.kind;
  el.textContent = next.message;
  (next.kind === 'error' ? alert : status).appendChild(el);
  current = el;
  timer = setTimeout(() => {
    timer = null;
    el.remove();
    if (current === el) current = null;
    showNext();
  }, next.timeout);
}

/**
 * Queue a toast.
 * @param {string} message
 * @param {{ kind?: ToastKind, timeout?: number }} [options]
 */
export function showToast(message, options = {}) {
  if (!hasDom()) return;
  const kind = KINDS.has(/** @type {string} */ (options.kind))
    ? /** @type {ToastKind} */ (options.kind)
    : 'info';
  const t = options.timeout;
  const timeout = typeof t === 'number' && Number.isFinite(t) && t > 0 ? t : TOAST_TIMEOUT_MS;
  queue.push({ message: String(message), kind, timeout });
  if (!current && !timer) showNext();
}

/** Hide the toast on screen and drop every queued one. */
export function clearToasts() {
  queue = [];
  if (timer) clearTimeout(timer);
  timer = null;
  if (current) current.remove();
  current = null;
}
