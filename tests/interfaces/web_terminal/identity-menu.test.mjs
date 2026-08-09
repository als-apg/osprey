// @ts-check
/**
 * Unit tests for the header identity chip's popover (`initIdentityMenu`):
 *   npx vitest run tests/interfaces/web_terminal/identity-menu.test.mjs
 *
 * The chip collapses the old username badge + standalone logout button into
 * one control. This module owns ONLY the popover; the logout behaviour stays
 * app.js's `initLogoutButton` (covered by app-logout.test.mjs), which finds the
 * button by `#logout-btn` wherever it lives. The tests below therefore pin two
 * things: the popover grammar shared with the shell's other header popovers,
 * and the fact that this module does not touch the logout contract.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { initIdentityMenu } from '../../../src/osprey/interfaces/web_terminal/static/js/identity-menu.js';

/** Render the chip the way the server does when `terminal_user` is set. */
function mountChip({ landingUrl = 'https://landing.example.org/' } = {}) {
  document.body.innerHTML = `
    <div class="identity-menu" id="identity-menu">
      <button class="identity-trigger" id="identity-menu-btn" type="button" aria-expanded="false">
        <span class="identity-avatar">A</span>
        <span class="identity-name">alice</span>
      </button>
      <div class="identity-card" id="identity-menu-card">
        <div class="identity-card-identity">
          <span class="identity-card-name">alice</span>
          <span class="identity-card-sub">Control Assistant</span>
        </div>
        <button class="identity-card-logout" id="logout-btn" type="button"
                data-landing-url="${landingUrl}">Log out</button>
      </div>
    </div>
    <div id="outside">elsewhere</div>`;
  return {
    root: /** @type {HTMLElement} */ (document.getElementById('identity-menu')),
    button: /** @type {HTMLButtonElement} */ (document.getElementById('identity-menu-btn')),
    card: /** @type {HTMLElement} */ (document.getElementById('identity-menu-card')),
    logout: /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn')),
    outside: /** @type {HTMLElement} */ (document.getElementById('outside')),
  };
}

beforeEach(() => {
  document.body.innerHTML = '';
});

describe('initIdentityMenu', () => {
  test('starts closed', () => {
    const { card, button } = mountChip();

    initIdentityMenu();

    expect(card.classList.contains('open')).toBe(false);
    expect(button.getAttribute('aria-expanded')).toBe('false');
  });

  test('clicking the chip opens the popover and mirrors aria-expanded', () => {
    const { button, card } = mountChip();
    initIdentityMenu();

    button.click();

    expect(card.classList.contains('open')).toBe(true);
    expect(button.getAttribute('aria-expanded')).toBe('true');
  });

  test('clicking the chip again closes it', () => {
    const { button, card } = mountChip();
    initIdentityMenu();

    button.click();
    button.click();

    expect(card.classList.contains('open')).toBe(false);
    expect(button.getAttribute('aria-expanded')).toBe('false');
  });

  test('a click outside closes it', () => {
    const { button, card, outside } = mountChip();
    initIdentityMenu();
    button.click();

    outside.click();

    expect(card.classList.contains('open')).toBe(false);
    expect(button.getAttribute('aria-expanded')).toBe('false');
  });

  test('a click inside the popover leaves it open', () => {
    // Logging out navigates away; closing the card first would hide the
    // button's own in-flight aria-busy state at the moment it matters.
    const { button, card, logout } = mountChip();
    initIdentityMenu();
    button.click();

    logout.click();

    expect(card.classList.contains('open')).toBe(true);
  });

  test('Escape closes it and returns focus to the chip', () => {
    const { button, card } = mountChip();
    initIdentityMenu();
    button.click();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(card.classList.contains('open')).toBe(false);
    expect(document.activeElement).toBe(button);
  });

  test('a non-Escape key leaves it open', () => {
    const { button, card } = mountChip();
    initIdentityMenu();
    button.click();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));

    expect(card.classList.contains('open')).toBe(true);
  });

  test('stops listening once closed', () => {
    // The outside-click/Escape listeners are document-level and capture-phase;
    // leaving them attached after a close would let a later stray Escape steal
    // focus back to a chip the operator is no longer using.
    const { button, card, outside } = mountChip();
    initIdentityMenu();
    button.click();
    button.click();

    outside.click();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(card.classList.contains('open')).toBe(false);
    expect(document.activeElement).not.toBe(button);
  });

  test('leaves the logout button contract untouched', () => {
    // app.js's initLogoutButton() and palette-boot.js's "Log out" command both
    // find this button by id and read data-landing-url off it. Moving it into
    // the popover must not have changed either.
    const { button, logout } = mountChip();
    initIdentityMenu();
    button.click();

    expect(logout.id).toBe('logout-btn');
    expect(logout.dataset.landingUrl).toBe('https://landing.example.org/');
    expect(logout.disabled).toBe(false);
  });

  test('is a no-op when the chip is absent (single-user deployment)', () => {
    document.body.innerHTML = '<div class="header-actions"></div>';

    expect(() => initIdentityMenu()).not.toThrow();
  });
});
