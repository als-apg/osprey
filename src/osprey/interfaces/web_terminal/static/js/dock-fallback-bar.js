/**
 * The tile bar of the no-dockview fallback host (dock-workspace.js mounts it
 * when no dockview is present). Split from dock-workspace.js for its line
 * cap; the behaviour is the fallback's alone.
 * @module dock-fallback-bar
 */

import { buildTileMenuButton } from './dock-tab.js';
import { registerContribHost, unregisterContribHost } from './tile-header-contrib.js';

/**
 * Give the fallback iframe host the tile bar a docked tile has. Embed contract
 * v2 promises an embedded panel exactly one header — the hub's tile bar — and
 * panels on it hide their in-body copies of whatever they contribute, so a
 * host without a bar would leave those controls nowhere. The fallback shows
 * one panel at a time (#right-panel's `data-active-panel`), so it carries one
 * bar: the active panel's name, its contribution region (subtitle and panel
 * icons), registered with tile-header-contrib.js exactly as a docked TileTab
 * registers its own, and the tile ⋯. No Pop out, × or shared controls: there
 * is no tile to close or pop out of. The bar follows the active panel and
 * hides only when none is active.
 *
 * The Agent card needs nothing here: without a dock it is never adopted, so
 * its bar stays at the top of the card — the console bar in chat (the card's
 * `.terminal-header` is hidden there, operator.css), the header on the terminal.
 *
 * The bar reuses the docked bar's `.tile-tab` markup so it renders with the
 * same styles; the inline layout only adapts that full-height bar to a row
 * above #panel-content. Idempotent per host.
 * @param {HTMLElement} host  the fallback iframe host (#right-panel)
 * @returns {HTMLElement | null} the bar, or null when the host has no #panel-content
 */
export function initFallbackTileBar(host) {
  const existing = host.querySelector(':scope > .fallback-tile-bar');
  if (existing instanceof HTMLElement) return existing;
  const content = host.querySelector('#panel-content, .panel-content');
  if (!(content instanceof HTMLElement)) return null;

  host.style.flexDirection = 'column';
  const bar = document.createElement('div');
  bar.className = 'tile-tab fallback-tile-bar';
  bar.style.flex = '0 0 var(--wt-pane-head-height)';
  bar.style.height = 'var(--wt-pane-head-height)';
  bar.style.cursor = 'default';
  const title = document.createElement('span');
  title.className = 'tile-tab-title';
  const actions = document.createElement('div');
  actions.className = 'tile-tab-actions';
  actions.appendChild(buildTileMenuButton(() => host.dataset.activePanel || null, bar));
  bar.append(title, actions);
  host.insertBefore(bar, content);

  /** @type {{ panelId: string, el: HTMLElement } | null} */
  let registered = null;

  const sync = () => {
    const panelId = host.dataset.activePanel || null;
    if (registered && registered.panelId !== panelId) {
      unregisterContribHost(registered.panelId, registered.el);
      registered.el.remove();
      registered = null;
    }
    if (panelId && !registered) {
      // A fresh region per panel: registerContribHost wires its listeners on
      // the element, so reusing one across panels would stack them.
      const el = document.createElement('div');
      el.className = 'tile-tab-contrib';
      bar.insertBefore(el, actions);
      registerContribHost(panelId, el);
      registered = { panelId, el };
    }
    title.textContent = panelId ? panelLabel(panelId) : '';
    // Inline display, not `hidden`: the .tile-tab rule's display:flex would
    // outrank the attribute's user-agent display:none.
    bar.style.display = panelId ? '' : 'none';
  };

  sync();
  new MutationObserver(sync).observe(host, {
    attributes: true,
    attributeFilter: ['data-active-panel'],
  });
  return bar;
}

/**
 * A panel's display name, read from its sidebar row (the one place the hub
 * renders it outside a tile); the id when the sidebar does not carry the panel.
 * @param {string} panelId
 * @returns {string}
 */
function panelLabel(panelId) {
  const btn = document.querySelector(`.panel-rail-button[data-panel-id="${CSS.escape(panelId)}"]`);
  return btn?.getAttribute('aria-label') || panelId;
}
