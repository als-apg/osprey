/**
 * Unit tests for the Artifact Gallery browser-column overflow menu
 * (sidebar-menu.js): open/close mechanics only — what the items DO stays
 * with gallery.js/browse-layout.js and is covered by the browser suite
 * (test_gallery_interactions.py).
 *
 * Pure DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/sidebar-menu.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { initSidebarMenu } from '../../../src/osprey/interfaces/artifacts/static/js/sidebar-menu.js';
import { byId } from '../_support/dom.mjs';

/** Minimal DOM fixture matching artifacts/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `
    <div class="sidebar-menu-wrap">
      <button id="sidebar-menu-btn" aria-haspopup="menu" aria-expanded="false"></button>
      <div class="sidebar-menu" id="sidebar-menu" role="menu" hidden>
        <button class="sidebar-menu-item" id="all-sessions-btn" role="menuitemcheckbox" aria-checked="false">All sessions</button>
        <button class="sidebar-menu-item" id="refresh-btn" role="menuitem">Refresh</button>
      </div>
    </div>
    <button id="outside">elsewhere</button>
  `;
}

function initAll() {
  return initSidebarMenu({
    button: byId('sidebar-menu-btn'),
    menu: byId('sidebar-menu'),
  });
}

beforeEach(() => {
  mountFixture();
});

describe('open/close mechanics', () => {
  test('starts closed; trigger click opens (aria-expanded + [hidden] in step); second click closes', () => {
    initAll();
    const button = byId('sidebar-menu-btn');
    const menu = byId('sidebar-menu');
    expect(menu.hidden).toBe(true);

    button.click();
    expect(menu.hidden).toBe(false);
    expect(button.getAttribute('aria-expanded')).toBe('true');

    button.click();
    expect(menu.hidden).toBe(true);
    expect(button.getAttribute('aria-expanded')).toBe('false');
  });

  test('clicking a menu item dismisses the menu (the action itself is the item wiring\'s job)', () => {
    initAll();
    byId('sidebar-menu-btn').click();
    expect(byId('sidebar-menu').hidden).toBe(false);

    byId('refresh-btn').click();
    expect(byId('sidebar-menu').hidden).toBe(true);
    expect(byId('sidebar-menu-btn').getAttribute('aria-expanded')).toBe('false');
  });

  test('an outside pointerdown closes an open menu; inside pointerdown does not', () => {
    initAll();
    byId('sidebar-menu-btn').click();

    byId('all-sessions-btn').dispatchEvent(new Event('pointerdown', { bubbles: true }));
    expect(byId('sidebar-menu').hidden).toBe(false);

    byId('outside').dispatchEvent(new Event('pointerdown', { bubbles: true }));
    expect(byId('sidebar-menu').hidden).toBe(true);
  });

  test('Escape closes an open menu and returns focus to the trigger', () => {
    initAll();
    const button = byId('sidebar-menu-btn');
    button.click();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(byId('sidebar-menu').hidden).toBe(true);
    expect(document.activeElement).toBe(button);
  });

  test('close() from the returned handle closes the menu', () => {
    const handle = initAll();
    byId('sidebar-menu-btn').click();
    handle.close();
    expect(byId('sidebar-menu').hidden).toBe(true);
  });

  test('no-ops safely when elements are absent', () => {
    expect(() => initSidebarMenu({ button: null, menu: null })).not.toThrow();
    expect(initSidebarMenu({ button: null, menu: null }).close).toBeTypeOf('function');
  });
});
