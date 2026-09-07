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

/** The header's display menu plus the drawer it targets, as index.html lays them out. */
function mountFixture() {
  document.body.innerHTML = `
    <osprey-display-menu settings-drawer="settings-drawer">
      <button class="display-menu-settings" data-drawer="settings-drawer">Settings</button>
    </osprey-display-menu>
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
});
