// @ts-check
/**
 * Shared real-dockview harness for tests that run under jsdom
 * (`// @vitest-environment jsdom`).
 *
 * The dockview it mounts is the build the page loads: node_modules holds
 * dockview-core at the version package.json pins exactly, `buildSha()` proves
 * its min bundle byte-identical to vendor_manifest.json's digest (a test asserts
 * it), and the bundle is evaluated as a classic script so it installs the same
 * `window['dockview-core']` global the page's `<script>` tag does.
 *
 * `mountDock()` creates a dock the way dock-workspace.js does: `createComponent`
 * returns the SAME adopted terminal subtree for `terminal` on every rebuild (a
 * DOM relocation, never a clone) and an empty element for every service
 * placeholder. jsdom has no ResizeObserver, which dockview constructs per grid,
 * so a no-op one is installed when absent.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const MANIFEST = path.join(REPO_ROOT, 'src/osprey/interfaces/vendor_manifest.json');
/** The published min bundle the manifest pins (the static/vendor copy is gitignored). */
export const DOCKVIEW_BUILD = 'node_modules/dockview-core/dist/dockview-core.min.js';
const GLOBAL = 'dockview-core';

/** @returns {{ version: string, sha256: string }} the manifest's dockview-core entry */
export function pinnedDockview() {
  const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
  const asset = manifest.assets.find((/** @type {{ name: string }} */ a) => a.name === GLOBAL);
  if (!asset) throw new Error(`vendor_manifest.json names no asset ${GLOBAL}`);
  return asset;
}

/** @returns {Buffer} */
function buildBytes() {
  return readFileSync(path.join(REPO_ROOT, DOCKVIEW_BUILD));
}

/** @returns {string} sha256 of the node_modules min bundle */
export function buildSha() {
  return createHash('sha256').update(buildBytes()).digest('hex');
}

/** @type {any} */
let loaded = null;

/**
 * The dockview-core library, evaluated once as a classic script and left
 * installed as `window['dockview-core']`, as on the page.
 * @returns {any}
 */
export function loadDockview() {
  const g = /** @type {any} */ (globalThis);
  if (loaded && g[GLOBAL] === loaded) return loaded;
  if (typeof g.ResizeObserver === 'undefined') {
    g.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
  new Function('exports', 'module', 'define', 'require', buildBytes().toString('utf8'))();
  loaded = g[GLOBAL];
  if (!loaded || typeof loaded.createDockview !== 'function') {
    throw new Error(`${DOCKVIEW_BUILD} did not install window['${GLOBAL}']`);
  }
  return loaded;
}

/** @type {Array<{ root: HTMLElement, api: any }>} */
const mounted = [];

export const DEFAULT_PX = Object.freeze({ width: 1000, height: 600 });

/**
 * Mount a real dockview the way dock-workspace.js does.
 * @param {{ width: number, height: number }} [px]
 * @returns {{ root: HTMLElement, api: any, terminal: HTMLElement, xterm: HTMLTextAreaElement }}
 */
export function mountDock(px = DEFAULT_PX) {
  const dockview = loadDockview();
  const root = document.createElement('div');
  root.id = 'dock-root';
  document.body.appendChild(root);

  // The terminal card and the textarea xterm focuses, adopted once.
  const terminal = document.createElement('div');
  terminal.className = 'terminal-panel';
  const xterm = document.createElement('textarea');
  xterm.className = 'xterm-helper-textarea';
  terminal.appendChild(xterm);
  const host = document.createElement('div');
  host.className = 'dock-panel-host';
  host.appendChild(terminal);

  dockview.markDockviewPackageLoaded?.();
  const api = dockview.createDockview(root, {
    disableFloatingGroups: true,
    createComponent: (/** @type {{ name: string }} */ options) => {
      if (options.name === 'terminal') return { element: host, init() {} };
      return { element: document.createElement('div'), init() {} };
    },
  });
  api.layout(px.width, px.height);
  mounted.push({ root, api });
  return { root, api, terminal, xterm };
}

/** Dispose every dock mountDock created and remove its root. */
export function unmountDocks() {
  for (const { root, api } of mounted.splice(0)) {
    try {
      api.dispose();
    } catch {
      /* already gone */
    }
    root.remove();
  }
}

/** Let a microtask and one macrotask run (dockview's deferred layout events). */
export async function settle() {
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Each dockview group as `{tile, views, active}`, ids as dockview has them.
 * @param {any} api
 */
export function groupsOf(api) {
  return api.groups.map((/** @type {any} */ g) => ({
    tile: g.id,
    views: g.panels.map((/** @type {any} */ p) => p.id),
    active: g.activePanel?.id ?? null,
  }));
}
