// @ts-check
/**
 * Unit tests for the Simple-mode console's DOM (chat-console.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-console.test.mjs
 *
 * The console's Agent bar becomes the Agent pane's one header row (the dock
 * adopts it as the tile bar in the Simple view), so its contents are a
 * contract: the status dot, `Agent`, and named icon buttons. The bell and
 * Export are appended by their feature modules, and the ⋯ is handed back
 * unattached so chat.js can put it last.
 */

import { test, expect, describe } from 'vitest';

import {
  barButton,
  buildConsole,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-console.js';

describe('Agent bar', () => {
  test('reads dot, Agent, then New conversation', () => {
    const { bar, newBtn } = buildConsole();
    const [dot, label, ...rest] = [...bar.children];
    expect(dot.classList.contains('osprey-dot')).toBe(true);
    expect(dot.classList.contains('ok')).toBe(true);
    expect(label.classList.contains('op-session-label')).toBe(true);
    expect(label.textContent).toBe('Agent');
    expect(rest).toEqual([newBtn]);
  });

  test('New conversation is a named pen icon button', () => {
    const { newBtn } = buildConsole();
    expect(newBtn.classList.contains('osprey-icon-btn')).toBe(true);
    expect(newBtn.getAttribute('aria-label')).toBe('New conversation');
    expect(newBtn.title).toBe('New conversation');
    expect(newBtn.type).toBe('button');
    expect(newBtn.querySelector('svg.osprey-ic')).not.toBeNull();
    // An icon button: the name is the label, not visible text.
    expect(newBtn.textContent).toBe('');
  });

  test('the ⋯ is built but not in the bar, and announces its menu', () => {
    const { bar, moreBtn } = buildConsole();
    expect(bar.contains(moreBtn)).toBe(false);
    expect(moreBtn.classList.contains('op-session-more')).toBe(true);
    expect(moreBtn.getAttribute('aria-label')).toBe('Earlier conversations');
    expect(moreBtn.getAttribute('aria-haspopup')).toBe('menu');
    expect(moreBtn.getAttribute('aria-expanded')).toBe('false');
  });

  test('no "+ New" and no Connected strip: only the dot, the name and buttons', () => {
    const { bar } = buildConsole();
    expect(bar.textContent).toBe('Agent');
    expect(bar.querySelector('.new-session-btn, .terminal-label-simple')).toBeNull();
  });

  test('the dot is decoration: hidden from assistive tech, no state claim', () => {
    const { bar } = buildConsole();
    const led = /** @type {HTMLElement} */ (bar.querySelector('.op-session-led'));
    expect(led.getAttribute('aria-hidden')).toBe('true');
    expect(led.hasAttribute('title')).toBe(false);
  });
});

describe('composer', () => {
  test('the message box is named for the agent', () => {
    const { textarea } = buildConsole();
    expect(textarea.getAttribute('aria-label')).toBe('Message the agent');
    expect(textarea.placeholder).toBe('Message the agent…');
  });
});

describe('barButton', () => {
  test('builds an .osprey-icon-btn with the icon, the name and the tooltip', () => {
    const btn = barButton('op-export', 'share', 'Export');
    expect(btn.className).toBe('osprey-icon-btn op-export');
    expect(btn.getAttribute('aria-label')).toBe('Export');
    expect(btn.title).toBe('Export');
    expect(btn.querySelector('svg.osprey-ic')).not.toBeNull();
  });

  test('an unknown icon name fails at the call site', () => {
    expect(() => barButton('x', 'no-such-icon', 'X')).toThrow(/unknown icon/);
  });
});
