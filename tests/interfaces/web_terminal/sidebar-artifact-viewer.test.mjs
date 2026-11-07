// @ts-check
/**
 * The Artifacts panel as the session tree drives it (sidebar-artifact-viewer.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifact-viewer.test.mjs
 *
 * Pins what reaches the iframe (select, changed, tile-menu actions) and when
 * nothing does (no iframe, no shown artifact), the ready check (origin, type,
 * source), whether the panel is on view, and the open rule.
 */

import { test, expect, beforeEach, vi } from 'vitest';

const m = vi.hoisted(() => ({
  sendArtifactSelectToIframe: vi.fn(),
  sendArtifactChangedToIframe: vi.fn(),
  sendArtifactActionToIframe: vi.fn(),
  activateTab: vi.fn(),
  hasDockedTile: vi.fn(() => false),
  isDetached: vi.fn(() => false),
  menuItems: vi.fn((_shown, run) => [{ label: 'Download', run: () => run('download') }]),
}));
const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js', () => ({
  sendArtifactSelectToIframe: m.sendArtifactSelectToIframe,
  sendArtifactChangedToIframe: m.sendArtifactChangedToIframe,
  sendArtifactActionToIframe: m.sendArtifactActionToIframe,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', () => ({
  activateTab: m.activateTab,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js', () => ({
  hasDockedTile: m.hasDockedTile,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js', () => ({
  isDetached: m.isDetached,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/artifacts-pane-header.js', () => ({
  menuItems: m.menuItems,
}));

const viewer = await import(`${JS}/sidebar-artifact-viewer.js`);

/** @returns {HTMLIFrameElement} */
function addIframe() {
  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  iframe.dataset.panelId = 'artifacts';
  document.body.append(iframe);
  return iframe;
}

beforeEach(() => {
  vi.clearAllMocks();
  m.hasDockedTile.mockReturnValue(false);
  m.isDetached.mockReturnValue(false);
  document.body.replaceChildren();
});

test('without the iframe nothing is sent and the tile menu is empty', () => {
  viewer.sendSelect('a1');
  viewer.sendChanged('a1', 'updated');
  expect(m.sendArtifactSelectToIframe).not.toHaveBeenCalled();
  expect(m.sendArtifactChangedToIframe).not.toHaveBeenCalled();
  expect(viewer.viewerMenuItems(/** @type {any} */ ({ id: 'a1' }))).toEqual([]);
});

test('select and changed reach the iframe; a change with nothing shown does not', () => {
  const iframe = addIframe();
  viewer.sendSelect(null);
  expect(m.sendArtifactSelectToIframe).toHaveBeenCalledWith(iframe, null);
  const entry = /** @type {any} */ ({ id: 'a1' });
  viewer.sendChanged('a1', 'updated', entry);
  expect(m.sendArtifactChangedToIframe).toHaveBeenCalledWith(iframe, 'a1', 'updated', entry);
  viewer.sendChanged(null, 'deleted');
  expect(m.sendArtifactChangedToIframe).toHaveBeenCalledOnce();
});

test("the tile menu's rows send their action for the shown artifact", () => {
  const iframe = addIframe();
  expect(viewer.viewerMenuItems(null)).toEqual([]);
  const [row] = viewer.viewerMenuItems(/** @type {any} */ ({ id: 'a1' }));
  row.run?.();
  expect(m.sendArtifactActionToIframe).toHaveBeenCalledWith(iframe, 'a1', 'download');
});

test('ready is the iframe itself saying so, from this origin', () => {
  const iframe = addIframe();
  const ready = (/** @type {any} */ over) =>
    viewer.isViewerReady(
      /** @type {any} */ ({
        origin: window.location.origin,
        data: { type: 'osprey-artifact-ready' },
        source: iframe.contentWindow,
        ...over,
      })
    );
  expect(ready({})).toBe(true);
  expect(ready({ origin: 'https://elsewhere.example' })).toBe(false);
  expect(ready({ data: { type: 'other' } })).toBe(false);
  expect(ready({ source: window })).toBe(false);
});

test('on view when docked or in its own window; the open rule activates only an undocked panel', () => {
  expect(viewer.panelOnView()).toBe(false);
  m.isDetached.mockReturnValue(true);
  expect(viewer.panelOnView()).toBe(true);
  viewer.showArtifactsPanel();
  expect(m.activateTab).toHaveBeenCalledWith('artifacts', { userInitiated: true });
  m.hasDockedTile.mockReturnValue(true);
  viewer.showArtifactsPanel();
  expect(m.activateTab).toHaveBeenCalledOnce();
});
