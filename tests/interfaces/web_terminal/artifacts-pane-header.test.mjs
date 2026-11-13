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
const held = vi.hoisted(() => ({
  frame: /** @type {HTMLIFrameElement | null} */ (null),
  /** An artifact tile's frame, by its key. @type {Map<string, HTMLIFrameElement>} */
  tiles: new Map(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  managedIframe: (/** @type {string} */ id) => {
    const iframe = id === 'artifacts' ? held.frame : held.tiles.get(id);
    return iframe ? { iframe, title: 'Artifacts' } : null;
  },
}));

/** What artifact-origin.js says about the shown artifact, and who listens. */
const origin = vi.hoisted(() => ({
  /** @type {any} */
  value: null,
  /** @type {Set<() => void>} */
  listeners: new Set(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/artifact-origin.js', () => ({
  artifactOrigin: () => origin.value,
  onOriginChange: (/** @type {() => void} */ fn) => {
    origin.listeners.add(fn);
    return () => origin.listeners.delete(fn);
  },
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
  held.frame = null;
  held.tiles.clear();
  origin.value = null;
});

describe('headerItems', () => {
  test('with an artifact: its name as the crumb, then Info and Download (the size in the tooltip)', () => {
    expect(header.headerItems(plot)).toEqual([
      { kind: 'crumb', id: 'name', root: 'Artifact', leaf: 'Example: an interactive plot' },
      { kind: 'icon', id: 'info', icon: 'info', title: 'Artifact details' },
      { kind: 'icon', id: 'download', icon: 'download', title: 'Download (49.2 KB)' },
    ]);
  });

  test('the crumb names an untitled artifact as the sidebar does: its filename', () => {
    expect(header.headerItems({ ...plot, title: '' })[0]).toMatchObject({ leaf: 'p&q.html' });
  });

  test('with no artifact: both icons disabled in their slots', () => {
    expect(header.headerItems(null)).toEqual([
      { kind: 'icon', id: 'info', icon: 'info', title: 'Artifact details', disabled: true },
      { kind: 'icon', id: 'download', icon: 'download', title: 'Download', disabled: true },
    ]);
  });

  test('no pager, separator or text button: only the crumb and icons', () => {
    const kinds = [plot, null].flatMap((shown) =>
      header.headerItems(shown).map((item) => /** @type {any} */ (item).kind)
    );
    expect(new Set(kinds)).toEqual(new Set(['crumb', 'icon']));
  });

  test('an unacknowledged artifact from another session gives Info the warning tone', () => {
    expect(header.headerItems(plot, { notice: true })[1]).toEqual({
      kind: 'icon',
      id: 'info',
      icon: 'info',
      title: 'Artifact details',
      tone: 'warning',
    });
    expect(header.headerItems(plot, { notice: false })[1]).not.toHaveProperty('tone');
    expect(header.headerItems(null, { notice: true })[0]).not.toHaveProperty('tone');
  });

  test('an entry without a size gets a plain Download tooltip', () => {
    const bare = { ...plot, size_bytes: undefined };
    expect(header.headerItems(bare)[2]).toMatchObject({ id: 'download', title: 'Download' });
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
    header.showArtifactInfo(plot, 'artifacts');
    expect(popover().hidden).toBe(false);
    expect(popover().textContent).toContain('Example: an interactive plot');
    expect(icon.getAttribute('aria-expanded')).toBe('true');
    header.showArtifactInfo(plot, 'artifacts');
    expect(popover().hidden).toBe(true);
  });

  test('with the icon folded into ⋯ it opens under the bar instead', () => {
    const { bar, icon } = drawBar({ folded: true });
    header.showArtifactInfo(plot, 'artifacts');
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
    header.showArtifactInfo(plot, 'artifacts');
    expect(popover().textContent).toContain('/srv/osprey/artifacts/p&q.html');
  });

  test('an artifact tile’s Info opens under its own bar and reads its own frame', () => {
    const { icon: galleryIcon } = drawBar();
    const tile = document.createElement('div');
    tile.className = 'tile-tab';
    tile.dataset.panelId = 'iframe:artifact:a1';
    const tileIcon = document.createElement('button');
    tileIcon.className = 'osprey-icon-btn contrib-icon';
    tileIcon.dataset.contribKey = 'icon:info';
    tile.append(tileIcon);
    document.body.append(tile);
    const frame = document.createElement('iframe');
    document.body.append(frame);
    held.tiles.set('artifact:a1', frame);
    const meta = document.createElement('meta');
    meta.name = 'osprey-artifact-dir';
    meta.content = '/srv/tile';
    /** @type {Document} */ (frame.contentDocument).head.append(meta);
    header.showArtifactInfo(plot, 'artifact:a1');
    expect(tileIcon.getAttribute('aria-expanded')).toBe('true');
    expect(galleryIcon.hasAttribute('aria-expanded')).toBe(false);
    expect(popover().textContent).toContain('/srv/tile/p&q.html');
  });
});

describe('the session an artifact comes from', () => {
  /**
   * The origin of an artifact from another session, as artifact-origin.js
   * tells it; `Got it` acknowledges it and says "look again".
   * @param {object} [extra]
   */
  function foreign(extra = {}) {
    const told = {
      title: 'Tune scan',
      listed: true,
      open: vi.fn(),
      notice: true,
      acknowledge: vi.fn(() => {
        told.notice = false;
        for (const fn of origin.listeners) fn();
      }),
      ...extra,
    };
    origin.value = told;
    return told;
  }

  /** @returns {string[]} the card's row labels */
  const labels = () => Array.from(popover().querySelectorAll('dt'), (dt) => dt.textContent ?? '');

  test('the Session row comes first; its title switches to the session', () => {
    drawBar();
    const told = foreign({ notice: false });
    header.showArtifactInfo(plot, 'artifacts');
    expect(labels()).toEqual(['Session', 'Created', 'Size', 'File']);
    const link = /** @type {HTMLButtonElement} */ (popover().querySelector('dd button'));
    expect(link.textContent).toBe('Tune scan');
    link.click();
    expect(told.open).toHaveBeenCalledTimes(1);
    expect(popover().hidden).toBe(true);
  });

  test('the active session is named as plain text', () => {
    drawBar();
    foreign({ title: 'Orbit check', open: null, notice: false });
    header.showArtifactInfo(plot, 'artifacts');
    const value = popover().querySelector('dd');
    expect(value?.textContent).toBe('Orbit check');
    expect(value?.querySelector('button')).toBeNull();
  });

  test('the card leads with the notice; Got it takes it away and the row stays', () => {
    drawBar();
    const told = foreign();
    header.showArtifactInfo(plot, 'artifacts');
    const notice = /** @type {HTMLElement} */ (
      popover().querySelector('.osprey-info-popover-notice')
    );
    expect(notice.textContent).toBe('From another session: "Tune scan"Got it');
    /** @type {HTMLButtonElement} */ (notice.querySelector('.osprey-info-popover-link')).click();
    expect(told.open).toHaveBeenCalledTimes(1);

    header.showArtifactInfo(plot, 'artifacts');
    const gotIt = Array.from(popover().querySelectorAll('button')).find(
      (b) => b.textContent === 'Got it'
    );
    gotIt?.click();
    expect(told.acknowledge).toHaveBeenCalledTimes(1);
    expect(popover().hidden).toBe(false);
    expect(popover().querySelector('.osprey-info-popover-notice')).toBeNull();
    expect(labels()[0]).toBe('Session');
    expect(popover().querySelector('dd')?.textContent).toBe('Tune scan');
  });

  test('an owner the rows do not list: no link, in the row or the notice', () => {
    drawBar();
    foreign({ title: 'Unknown session', listed: false, open: null });
    header.showArtifactInfo(plot, 'artifacts');
    expect(popover().querySelector('.osprey-info-popover-notice')?.textContent).toBe(
      'From another session: Unknown sessionGot it'
    );
    expect(popover().querySelector('.osprey-info-popover-link')).toBeNull();
  });

  test('an open card is shown again on a change; a closed one is left alone', () => {
    drawBar();
    const told = foreign({ title: 'naming…' });
    header.showArtifactInfo(plot, 'artifacts');
    told.title = 'Tune scan';
    for (const fn of origin.listeners) fn();
    expect(popover().querySelector('dd')?.textContent).toBe('Tune scan');
    closeAllPopovers();
    told.title = 'Later';
    for (const fn of origin.listeners) fn();
    expect(popover().textContent).not.toContain('Later');
  });
});
