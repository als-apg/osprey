// @ts-check
/**
 * Unit tests for the chat console's DOM (chat-console.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-console.test.mjs
 *
 * The console's first row is the Agent bar, which agent-bar.js builds (its
 * shape is pinned in agent-bar.test.mjs). Here: the console hands the bar's
 * parts to its controller, routes `+` to the controller's action, and names
 * its composer.
 */

import { test, expect, describe, vi } from 'vitest';

import { buildConsole } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-console.js';

describe('Agent bar', () => {
  test('heads the console with the session label and New conversation', () => {
    const { bar, label, newBtn } = buildConsole();
    expect(bar.classList.contains('op-session-bar')).toBe(true);
    expect(bar.firstElementChild).toBe(label);
    expect(label.classList.contains('op-session-label')).toBe(true);
    expect(label.textContent).toBe('Agent');
    expect(bar.contains(newBtn)).toBe(true);
    expect(newBtn.getAttribute('aria-label')).toBe('New conversation');
  });

  test("+ runs the controller's action", () => {
    const onNew = vi.fn();
    buildConsole({ onNew }).newBtn.click();
    expect(onNew).toHaveBeenCalledTimes(1);
  });

  test('builds no ⋯, bell, Export or status dot: + is the one button', () => {
    const built = buildConsole();
    expect('moreBtn' in built).toBe(false);
    expect([...built.bar.querySelectorAll('button')]).toEqual([built.newBtn]);
    expect(built.bar.querySelector('.osprey-dot')).toBeNull();
  });
});

describe('composer', () => {
  test('the message box is named for the agent', () => {
    const { textarea } = buildConsole();
    expect(textarea.getAttribute('aria-label')).toBe('Message the agent');
    expect(textarea.placeholder).toBe('Message the agent…');
  });
});
