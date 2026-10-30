// @ts-check
/**
 * The artifacts pane header and tile menu (artifacts-pane-header.js):
 *   npx vitest run tests/interfaces/web_terminal/artifacts-pane-header.test.mjs
 *
 * Pins the header items in their bar order (crumb, ‹ N of M ›, separator,
 * Open full size, Save), the bare header with no artifact, the URLs Open full
 * size and Save use (the gallery's own routes under the hub's proxy and the
 * per-user prefix, the page carrying the theme on screen), and the artifact
 * verbs of the tile's ⋯ menu.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ theme: /** @type {string | null} */ ('dark') }));
vi.mock('/design-system/js/theme-manager.js', () => ({ getTheme: () => mocks.theme }));

const header =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/artifacts-pane-header.js');

const report = {
  id: 'a 1',
  title: 'SR Lattice Tunes',
  filename: 'tunes report.md',
  artifact_type: 'markdown',
};

beforeEach(() => {
  mocks.theme = 'dark';
});

afterEach(() => {
  delete window.__OSPREY_PREFIX__;
  vi.restoreAllMocks();
  document.body.replaceChildren();
});

describe('headerItems', () => {
  test('with no artifact: the crumb root alone and the two buttons, disabled', () => {
    const items = header.headerItems(null, null);
    expect(items).toEqual([
      { kind: 'crumb', id: 'crumb', root: 'Artifacts', leaf: '' },
      expect.objectContaining({ kind: 'button', id: 'open', disabled: true }),
      expect.objectContaining({ kind: 'button', id: 'save', disabled: true }),
    ]);
  });

  test('with an artifact: crumb, ‹ N of M ›, separator, then the actions', () => {
    const items = /** @type {any[]} */ (
      header.headerItems(report, { index: 2, count: 12, last: false })
    );
    expect(items.map((item) => `${item.kind}:${item.id}`)).toEqual([
      'crumb:crumb',
      'button:prev',
      'text:pager',
      'button:next',
      'sep:sep',
      'button:open',
      'button:save',
    ]);
    expect(items[0].leaf).toBe('SR Lattice Tunes');
    expect(items[2].text).toBe('3 of 12');
    expect(items[1]).toMatchObject({
      icon: 'chevron-left',
      title: 'Previous artifact',
      disabled: false,
    });
    expect(items[3]).toMatchObject({
      icon: 'chevron-right',
      title: 'Next artifact',
      disabled: false,
    });
    expect(items[5]).toMatchObject({
      icon: 'maximize',
      label: 'Open full size',
      tone: 'accent',
      disabled: false,
    });
    expect(items[6]).toMatchObject({ icon: 'save', label: 'Save', disabled: false });
  });

  test('the pager outranks the actions in a narrow tile', () => {
    const items = /** @type {any[]} */ (
      header.headerItems(report, { index: 0, count: 1, last: true })
    );
    const priority = Object.fromEntries(items.map((item) => [item.id, item.priority]));
    expect(priority.prev).toBeGreaterThan(priority.open);
    expect(priority.pager).toBe(priority.next);
    expect(items[1].disabled).toBe(true);
    expect(items[3].disabled).toBe(true);
  });
});

describe('Open full size and Save', () => {
  test('a markdown or notebook opens on its rendered page, with the theme', () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    expect(header.pageUrl(report)).toBe(
      '/u/alice/panel/artifacts/api/markdown/a%201/rendered?theme=dark'
    );
    expect(header.pageUrl({ ...report, artifact_type: 'notebook' })).toBe(
      '/u/alice/panel/artifacts/api/notebooks/a%201/rendered?theme=dark'
    );
  });

  test('anything else opens on its file; no theme yet means no query', () => {
    mocks.theme = null;
    const plot = { ...report, artifact_type: 'plot_html', filename: 'p&q.html' };
    expect(header.pageUrl(plot)).toBe('/panel/artifacts/files/a%201/p%26q.html');
    expect(header.fileUrl(report)).toBe('/panel/artifacts/files/a%201/tunes%20report.md');
  });

  test('Open full size opens a new tab without an opener', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    header.openArtifact(report);
    expect(open).toHaveBeenCalledWith(
      '/panel/artifacts/api/markdown/a%201/rendered?theme=dark',
      '_blank',
      'noopener'
    );
  });

  test('Save downloads the file under its own name and leaves nothing behind', () => {
    /** @type {HTMLAnchorElement[]} */
    const clicked = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(
      /** @this {HTMLAnchorElement} */ function () {
        clicked.push(this);
      }
    );
    header.saveArtifact(report);
    expect(clicked).toHaveLength(1);
    expect(clicked[0].getAttribute('href')).toBe('/panel/artifacts/files/a%201/tunes%20report.md');
    expect(clicked[0].download).toBe('tunes report.md');
    expect(document.querySelector('a')).toBeNull();
  });
});

describe('the tile menu verbs', () => {
  test('pin, print, logbook and delete, each sending its action', () => {
    const send = vi.fn();
    const items = header.menuItems(report, send);
    expect(items.map((item) => item.label)).toEqual([
      'Pin artifact',
      'Print artifact',
      'Compose logbook entry',
      'Delete artifact',
    ]);
    expect(items[3].danger).toBe(true);
    for (const item of items) item.run();
    expect(send.mock.calls.map((call) => call[0])).toEqual(['pin', 'print', 'logbook', 'delete']);
  });

  test('a pinned artifact offers Unpin', () => {
    const send = vi.fn();
    const [first] = header.menuItems({ ...report, pinned: true }, send);
    expect(first.label).toBe('Unpin artifact');
    first.run();
    expect(send).toHaveBeenCalledWith('unpin');
  });
});
