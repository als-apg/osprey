// @ts-check
/**
 * Links in an agent answer in the Simple view's chat log.
 *
 * Invariants:
 * - A link in an agent answer never navigates the hub page away.
 * - A link to a panel this hub hosts (`<prefix>/panel/<id>…` on the hub's own
 *   origin) opens that panel in the workspace at the linked URL, fragment
 *   included. Any other http(s) link opens in a new tab without an opener;
 *   a link of any other scheme does nothing.
 */

/**
 * @typedef {{kind: 'panel', panel: string, url: string}
 *   | {kind: 'window', url: string} | {kind: 'inert'}} AgentLink
 */

/**
 * @param {string} s
 * @returns {string}
 */
function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Decide where a link in an agent answer goes.
 * @param {string} href - the anchor's href attribute
 * @param {{base: string, origin: string, prefix: string,
 *   isHostedPanel: (id: string) => boolean}} ctx
 * @returns {AgentLink}
 */
export function classifyAgentLink(href, ctx) {
  /** @type {URL} */
  let u;
  try {
    u = new URL(href, ctx.base);
  } catch {
    return { kind: 'inert' };
  }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') return { kind: 'inert' };
  if (u.origin !== ctx.origin) return { kind: 'window', url: u.href };
  const m = new RegExp(`^${escapeRegExp(ctx.prefix)}/panel/([^/]+)(?:/.*)?$`).exec(u.pathname);
  if (m && ctx.isHostedPanel(m[1])) {
    return { kind: 'panel', panel: m[1], url: u.pathname + u.search + u.hash };
  }
  return { kind: 'window', url: u.href };
}

/**
 * Route clicks on links inside agent answers under `root`. A click with a
 * modifier key or a non-primary button is left to the browser.
 * @param {HTMLElement} root
 * @param {{prefix: string, isHostedPanel: (id: string) => boolean,
 *   openPanel: (id: string, url: string) => void,
 *   openWindow: (url: string, target: string, features: string) => void}} deps
 * @returns {void}
 */
export function wireAgentLinks(root, deps) {
  root.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const target = /** @type {Element | null} */ (e.target);
    const a = target && typeof target.closest === 'function' ? target.closest('a[href]') : null;
    if (!a || !root.contains(a)) return;
    const body = a.closest('.op-entry.assistant .osprey-md-rendered');
    if (!body || !root.contains(body)) return;
    const link = classifyAgentLink(a.getAttribute('href') || '', {
      base: document.baseURI,
      origin: location.origin,
      prefix: deps.prefix,
      isHostedPanel: deps.isHostedPanel,
    });
    e.preventDefault();
    if (link.kind === 'panel') deps.openPanel(link.panel, link.url);
    else if (link.kind === 'window') deps.openWindow(link.url, '_blank', 'noopener');
  });
}
