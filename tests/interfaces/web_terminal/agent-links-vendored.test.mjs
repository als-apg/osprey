// @ts-check
// @vitest-environment jsdom
/**
 * A citation in an agent answer, rendered by the vendored libraries
 * (chat-render.js, then agent-links.js):
 *   npx vitest run tests/interfaces/web_terminal/agent-links-vendored.test.mjs
 *
 * agent-links.test.mjs pins the click routing over the HTML marked is expected
 * to emit. This file pins the two steps before it with the real libraries: the
 * vendored marked turns `[BPM](panel/okf#devices/bpm)` into an anchor, DOMPurify
 * keeps that anchor's relative `href`, and the click then opens the panel. A
 * marked release that rendered a relative link differently, or a sanitiser
 * configuration that dropped `href`, goes red here.
 *
 * The libraries are the builds `vendor_manifest.json` pins: node_modules holds
 * them at the same versions, each file is proven byte-identical to the
 * manifest's digest, and each is evaluated as a classic script so it installs
 * the same global the page's `<script>` tag does.
 *
 * jsdom rather than the suite's happy-dom: DOMPurify reports itself supported
 * under happy-dom but does not sanitise there (it unwraps every element and
 * leaves `javascript:` hrefs and event-handler attributes in place), so a
 * sanitiser pin under happy-dom proves nothing. DOMPurify's own suite runs on
 * jsdom.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

import { wireAgentLinks } from '../../../src/osprey/interfaces/web_terminal/static/js/agent-links.js';
import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const MANIFEST = path.join(REPO_ROOT, 'src/osprey/interfaces/vendor_manifest.json');

/**
 * The published build in node_modules of each manifest asset this file loads,
 * by the asset's manifest name.
 * @type {Readonly<Record<string, string>>}
 */
const BUILDS = Object.freeze({
  marked: 'node_modules/marked/marked.min.js',
  DOMPurify: 'node_modules/dompurify/dist/purify.min.js',
});

/**
 * @param {string} name - an asset name in vendor_manifest.json
 * @returns {{ version: string, sha256: string }}
 */
function pinned(name) {
  const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
  const asset = manifest.assets.find((/** @type {{ name: string }} */ a) => a.name === name);
  if (!asset) throw new Error(`vendor_manifest.json names no asset ${name}`);
  return asset;
}

/** @param {string} name */
function buildBytes(name) {
  return readFileSync(path.join(REPO_ROOT, BUILDS[name]));
}

/**
 * Evaluate a UMD build the way a classic `<script>` does — sloppy mode, with no
 * CommonJS or AMD loader in scope — so its wrapper installs the library on this
 * realm's global, then take that global back off so each test installs it
 * through `vi.stubGlobal` and `vi.unstubAllGlobals` removes it again.
 * @param {string} name
 * @returns {any}
 */
function loadClassicScript(name) {
  new Function('exports', 'module', 'define', 'require', buildBytes(name).toString('utf8'))();
  const g = /** @type {any} */ (globalThis);
  const lib = g[name];
  delete g[name];
  return lib;
}

describe('the vendored builds', () => {
  for (const name of Object.keys(BUILDS)) {
    test(`${name} in node_modules is the build the vendor manifest pins`, () => {
      const digest = createHash('sha256').update(buildBytes(name)).digest('hex');
      expect(
        digest,
        `${BUILDS[name]} is not ${name} ${pinned(name).version}: ` +
          'package.json and vendor_manifest.json pin one version of it'
      ).toBe(pinned(name).sha256);
    });
  }
});

describe('a citation rendered by the vendored marked and DOMPurify', () => {
  /** @type {{ marked: any, DOMPurify: any }} */
  let real;
  /** @type {HTMLElement} */
  let root;
  /** @type {import('vitest').Mock<(id: string, url: string) => void>} */
  let openPanel;
  /** @type {import('vitest').Mock<(url: string, target: string, features: string) => void>} */
  let openWindow;

  beforeAll(() => {
    real = { marked: loadClassicScript('marked'), DOMPurify: loadClassicScript('DOMPurify') };
  });

  beforeEach(() => {
    vi.stubGlobal('marked', real.marked);
    vi.stubGlobal('DOMPurify', real.DOMPurify);
    root = document.createElement('div');
    document.body.appendChild(root);
    openPanel = vi.fn();
    openWindow = vi.fn();
    wireAgentLinks(root, {
      prefix: '',
      isHostedPanel: (id) => id === 'okf',
      openPanel,
      openWindow,
    });
  });

  afterEach(() => {
    root.remove();
    vi.unstubAllGlobals();
  });

  test('the citation keeps its href, the hostile link loses its own, and the click opens the panel', () => {
    expect(real.DOMPurify.isSupported).toBe(true);
    const entry = createChatRenderer(root).addAgentMessage(
      'The monitor is described under [BPM](panel/okf#devices/bpm); ' +
        'never follow [this](javascript:alert(1)).'
    );
    const anchors = entry.querySelectorAll('.osprey-md-rendered a');
    expect(anchors).toHaveLength(2);
    const citation = /** @type {HTMLAnchorElement} */ (anchors[0]);
    expect(citation.getAttribute('href')).toBe('panel/okf#devices/bpm');
    expect(citation.textContent).toBe('BPM');
    // The same render went through a sanitiser that does real work: the
    // javascript: link came out without an href, so the citation's survived
    // sanitising rather than bypassing it.
    expect(anchors[1].hasAttribute('href')).toBe(false);

    const ev = new MouseEvent('click', { bubbles: true, cancelable: true, button: 0 });
    citation.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
    expect(openPanel).toHaveBeenCalledTimes(1);
    expect(openPanel).toHaveBeenCalledWith('okf', '/panel/okf#devices/bpm');
    expect(openWindow).not.toHaveBeenCalled();
  });
});
