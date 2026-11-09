// @ts-check
/**
 * The Artifacts tile's bar items and ⋯ rows (artifacts-pane-header.js):
 *   npx vitest run tests/interfaces/web_terminal/artifacts-pane-header.test.mjs
 *
 * Pins the bar items (the Info and
 * Download icons; both icons disabled and no subtitle with no artifact; no
 * crumb, pager or text button), the Download route and the file it saves,
 * what Info shows and where it opens, and the artifact verbs of the tile's
 * ⋯ "This panel" section (Delete last, set apart, in the danger colour).
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import { closeAllPopovers } from '/design-system/js/popover.js';

/** The panel's frame as dock-iframe holds it. */
const held = vi.hoisted(() => ({ frame: /** @type {HTMLIFrameElement | null} */ (null) }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  managedIframe: (/** @type {string} */ id) =>
    id === 'artifacts' && held.frame ? { iframe: held.frame, title: 'Artifacts' } : null,
}));

const header =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/artifacts-pane-header.js');

const plot = {
  id: 'a 1',
  title: 'Example: an interactive plot',
  filename: 'p&q.html',
  artifact_type: 'plot_html',
  description: 'Synthetic data, not a session result.',
  size_bytes: 50381,
  timestamp: '2026-10-28T17:41:00+00:00',
};

/** @returns {HTMLElement} the page's info popover */
const popover = () => /** @type {HTMLElement} */ (document.querySelector('.osprey-info-popover'));

/**
 * The Artifacts tile's bar as the hub draws it, with its Info icon.
 * @param {{folded?: boolean}} [options] the narrow bar folded the icon into ⋯
 */
function drawBar({ folded = false } = {}) {
  document.body.innerHTML = `
    <div class="tile-tab" data-panel-id="iframe:artifacts">
      <span class="tile-tab-title">Artifacts</span>
      <div class="tile-tab-contrib">
        <button class="osprey-icon-btn contrib-icon${folded ? ' contrib-hidden' : ''}"
                data-contrib-key="icon:info" type="button"></button>
      </div>
    </div>`;
  return {
    bar: /** @type {HTMLElement} */ (document.querySelector('.tile-tab')),
    icon: /** @type {HTMLElement} */ (document.querySelector('[data-contrib-key="icon:info"]')),
  };
}

afterEach(() => {
  closeAllPopovers();
  delete window.__OSPREY_PREFIX__;
  vi.restoreAllMocks();
  document.body.replaceChildren();
});

describe('headerItems', () => {
  test('with an artifact: Info and Download (the size in the tooltip)', () => {
    expect(header.headerItems(plot)).toEqual([
      { kind: 'icon', id: 'info', icon: 'info', title: 'Artifact details' },
      { kind: 'icon', id: 'download', icon: 'download', title: 'Download (49.2 KB)' },
    ]);
  });

  test('with no artifact: both icons disabled in their slots', () => {
    expect(header.headerItems(null)).toEqual([
      { kind: 'icon', id: 'info', icon: 'info', title: 'Artifact details', disabled: true },
      { kind: 'icon', id: 'download', icon: 'download', title: 'Download', disabled: true },
    ]);
  });

  test('only protocol v3 kinds: no crumb, pager, separator or text button', () => {
    const kinds = [plot, null].flatMap((shown) =>
      header.headerItems(shown).map((item) => /** @type {any} */ (item).kind)
    );
    expect(new Set(kinds)).toEqual(new Set(['icon']));
  });

  test('an entry without a size gets a plain Download tooltip', () => {
    const bare = { ...plot, size_bytes: undefined };
    expect(header.headerItems(bare)[1]).toMatchObject({ id: 'download', title: 'Download' });
  });
});

describe('Download', () => {
  test("the gallery's file route, under the hub's proxy and the per-user prefix", () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    expect(header.fileUrl(plot)).toBe('/u/alice/panel/artifacts/files/a%201/p%26q.html');
  });

  test('saves the file under its own name and leaves nothing behind', () => {
    /** @type {HTMLAnchorElement[]} */
    const clicked = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(
      /** @this {HTMLAnchorElement} */ function () {
        clicked.push(this);
      }
    );
    header.downloadArtifact(plot);
    expect(clicked).toHaveLength(1);
    expect(clicked[0].getAttribute('href')).toBe('/panel/artifacts/files/a%201/p%26q.html');
    expect(clicked[0].download).toBe('p&q.html');
    expect(document.querySelector('a')).toBeNull();
  });
});

describe('Info', () => {
  test('title, description, Created, Size, and the store path with Copy', () => {
    const content = header.infoContent(plot, 'var/agent_data/artifacts');
    expect(content.title).toBe('Example: an interactive plot');
    expect(content.text).toBe('Synthetic data, not a session result.');
    const rows = content.rows ?? [];
    expect(rows.map((row) => row.label)).toEqual(['Created', 'Size', 'File']);
    expect(String(rows[0].value)).toContain('2026');
    expect(rows[1]).toEqual({ label: 'Size', value: '49.2 KB' });
    expect(rows[2]).toEqual({
      label: 'File',
      value: 'var/agent_data/artifacts/p&q.html',
      copy: true,
    });
  });

  test('no Source, and no empty rows: an unknown directory leaves the bare file name', () => {
    const content = header.infoContent(
      /** @type {any} */ ({ id: 'b', filename: 'notes.md', tool_source: 'execute_file' }),
      null
    );
    expect(content.text).toBeUndefined();
    expect(content.rows).toEqual([{ label: 'File', value: 'notes.md', copy: true }]);
  });

  test('opens under the bar’s Info icon, and a second Info closes it', () => {
    const { icon } = drawBar();
    header.showArtifactInfo(plot);
    expect(popover().hidden).toBe(false);
    expect(popover().textContent).toContain('Example: an interactive plot');
    expect(icon.getAttribute('aria-expanded')).toBe('true');
    header.showArtifactInfo(plot);
    expect(popover().hidden).toBe(true);
  });

  test('with the icon folded into ⋯ it opens under the bar instead', () => {
    const { bar, icon } = drawBar({ folded: true });
    header.showArtifactInfo(plot);
    expect(bar.getAttribute('aria-expanded')).toBe('true');
    expect(icon.hasAttribute('aria-expanded')).toBe(false);
  });

  test('the path names the directory the viewer page states', () => {
    drawBar();
    const frame = document.createElement('iframe');
    frame.className = 'panel-iframe';
    frame.dataset.panelId = 'artifacts';
    document.body.append(frame);
    held.frame = frame;
    const meta = document.createElement('meta');
    meta.name = 'osprey-artifact-dir';
    meta.content = '/srv/osprey/artifacts';
    /** @type {Document} */ (frame.contentDocument).head.append(meta);
    header.showArtifactInfo(plot);
    expect(popover().textContent).toContain('/srv/osprey/artifacts/p&q.html');
  });
});
