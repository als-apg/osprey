// @ts-check
/**
 * The client half of the Config panel's tier gate.
 *
 * `web.config_panel.enabled: false` reaches the browser as
 * `config_panel_enabled: false` on `/api/capabilities`. The server refusal is
 * the real gate — both `/api/config` verbs answer 403 — and this proves the
 * other half: the Settings entry and the drawer it opens leave the page, and
 * an absent flag (an older backend, a failed capabilities fetch) changes
 * nothing.
 *
 *   npx vitest run tests/interfaces/ariel/config-panel-gate.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { initSettings } from '../../../src/osprey/interfaces/ariel/static/js/settings.js';
import { initLayout } from '../../../src/osprey/interfaces/ariel/static/js/layout.js';

/** The header's display menu plus the drawer it targets, as index.html lays them out. */
function mountFixture() {
  document.body.innerHTML = `
    <osprey-display-menu settings-drawer="settings-drawer">
      <button class="display-menu-settings" data-drawer="settings-drawer">Settings</button>
    </osprey-display-menu>
    <button type="button" id="ariel-gear-btn"></button>
    <div class="osprey-popover" id="ariel-gear-menu" role="menu" hidden></div>
    <div id="status-drawer"></div>
    <div id="settings-drawer">
      <div class="settings-mode-bar">
        <button class="settings-mode-btn active" data-mode="form">Form</button>
      </div>
      <textarea id="config-raw-editor"></textarea>
      <button id="config-apply-btn">Apply</button>
    </div>
  `;
}

describe('initSettings config panel gate', () => {
  beforeEach(() => mountFixture());

  test('a disabled panel loses its Settings entry and its drawer', () => {
    initSettings({ config_panel_enabled: false });

    expect(document.querySelector('.display-menu-settings')).toBeNull();
    expect(document.getElementById('settings-drawer')).toBeNull();
    expect(
      document.querySelector('osprey-display-menu')?.getAttribute('settings-drawer')
    ).toBeNull();
  });

  test('an enabled panel is left alone', () => {
    initSettings({ config_panel_enabled: true });

    expect(document.querySelector('.display-menu-settings')).not.toBeNull();
    expect(document.getElementById('settings-drawer')).not.toBeNull();
  });

  test('an absent flag leaves the panel where the server default leaves it', () => {
    initSettings({});

    expect(document.getElementById('settings-drawer')).not.toBeNull();
  });

  test('capabilities that never arrived leave the panel alone', () => {
    initSettings(null);

    expect(document.getElementById('settings-drawer')).not.toBeNull();
  });

  // app.js runs initSettings before initLayout, so the ⚙ menu is built from
  // the drawers that survived the gate.
  describe('then the ⚙ menu', () => {
    /** @returns {string[]} */
    const gearItems = () =>
      [...document.querySelectorAll('#ariel-gear-menu [data-ariel-drawer]')].map(
        (el) => /** @type {HTMLElement} */ (el).dataset.arielDrawer ?? ''
      );

    test('lists no Settings entry when the panel is disabled', () => {
      initSettings({ config_panel_enabled: false });
      const layout = initLayout(document);
      try {
        expect(gearItems()).toEqual(['status']);
      } finally {
        layout.destroy();
      }
    });

    test('lists Settings when the panel is enabled', () => {
      initSettings({ config_panel_enabled: true });
      const layout = initLayout(document);
      try {
        expect(gearItems()).toContain('settings');
      } finally {
        layout.destroy();
      }
    });
  });
});
