// @ts-check
/* OSPREY Web Terminal — Toast
 *
 * One short-lived line of feedback inside a host element: the text, a tone,
 * and its own removal. One toast per host at a time — a new one replaces the
 * one still showing, so a double gesture never stacks two. The host's own
 * stylesheet decides where the toast sits (inline beside a toggle bar, or
 * anchored under the sidebar's control-target chip); this module owns only the
 * element and its lifetime, which is the fade `.osprey-toast` animates
 * (terminal.css).
 */

/** How long a toast stays on screen, matching the stylesheet's fade. */
export const TOAST_TTL_MS = 2500;

/**
 * Show one toast inside *host*, replacing any toast it already holds.
 * @param {HTMLElement} host
 * @param {string} message
 * @param {{error?: boolean}} [opts]  `error` renders the error tone
 * @returns {HTMLElement} the toast element
 */
export function showToast(host, message, { error = false } = {}) {
  host.querySelector('.osprey-toast')?.remove();
  const toast = document.createElement('span');
  toast.className = error ? 'osprey-toast error' : 'osprey-toast';
  toast.setAttribute('role', 'status');
  toast.textContent = message;
  host.appendChild(toast);
  setTimeout(() => toast.remove(), TOAST_TTL_MS);
  return toast;
}
