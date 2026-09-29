/**
 * Header identity menu — the session popover behind the deployment chip.
 *
 * Operators reach for the name in the corner when they want to leave, so the
 * chip is a trigger and Log out lives behind it. These tests pin the open/close
 * contract. The markup is the server's (index.html): that `#logout-btn` is in
 * the served page while the menu is closed, where app.js and the command
 * palette resolve it by id, is pinned by test_terminal_user.py.
 */

import { beforeEach, describe, expect, test } from 'vitest';
import { initIdentityMenu } from '../../../src/osprey/interfaces/web_terminal/static/js/identity-menu.js';

const FIXTURE = `
  <header>
    <div class="header-identity">
      <button class="header-deployment header-identity-trigger" id="header-identity-trigger"
              type="button" aria-haspopup="true" aria-expanded="false"
              aria-controls="header-identity-menu">
        <span class="header-identity-label">Control Room (Alice)</span>
        <span class="header-identity-caret" aria-hidden="true">&#9662;</span>
      </button>
      <div class="header-identity-menu" id="header-identity-menu" role="menu">
        <div class="header-identity-who">
          <span class="header-identity-avatar" aria-hidden="true">A</span>
          <span class="header-identity-who-name">alice</span>
        </div>
        <button class="header-identity-logout" id="logout-btn" type="button"
                data-landing-url="https://facility.example/portal">Log out</button>
      </div>
    </div>
  </header>
  <div id="outside"></div>
`;

/** @param {string} selector */
const qs = (selector) => /** @type {HTMLElement} */ (document.querySelector(selector));

const isOpen = () => qs('#header-identity-menu').classList.contains('open');

function mountAndInit() {
  document.body.innerHTML = FIXTURE;
  return initIdentityMenu();
}

describe('identity menu', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  test('clicking the chip opens it, and clicking again closes it', () => {
    mountAndInit();
    const trigger = qs('#header-identity-trigger');

    trigger.click();
    expect(isOpen()).toBe(true);
    expect(trigger.getAttribute('aria-expanded')).toBe('true');

    trigger.click();
    expect(isOpen()).toBe(false);
    expect(trigger.getAttribute('aria-expanded')).toBe('false');
  });

  test('an outside click closes it', () => {
    mountAndInit();
    qs('#header-identity-trigger').click();

    qs('#outside').click();
    expect(isOpen()).toBe(false);
  });

  test('a click inside the menu does not close it', () => {
    mountAndInit();
    qs('#header-identity-trigger').click();

    qs('.header-identity-who').click();
    expect(isOpen()).toBe(true);
  });

  test('Escape closes it and returns focus to the chip', () => {
    mountAndInit();
    const trigger = qs('#header-identity-trigger');
    trigger.click();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(isOpen()).toBe(false);
    expect(document.activeElement).toBe(trigger);
  });

  test('a Log out click dismisses the menu', () => {
    // app.js owns the action and its in-flight lock; this module only gets the
    // popover out of the way so a refused logout does not leave it hanging.
    mountAndInit();
    qs('#header-identity-trigger').click();

    qs('#logout-btn').click();
    expect(isOpen()).toBe(false);
  });

  test('init no-ops on a terminal that renders no identity chip', () => {
    document.body.innerHTML = '<header></header>';
    expect(initIdentityMenu()).toBeNull();
  });
});
