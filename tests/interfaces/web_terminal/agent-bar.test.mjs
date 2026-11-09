// @ts-check
/**
 * Unit tests for the Agent pane's bar (agent-bar.js):
 *   npx vitest run tests/interfaces/web_terminal/agent-bar.test.mjs
 *
 * One bar on both surfaces. The chat console's is built here; the terminal's
 * is index.html's `.terminal-header`, read from the page source, so the two
 * are compared part by part.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';

import { test, expect, describe, vi } from 'vitest';

import { icon } from '../../../src/osprey/interfaces/design_system/static/js/icons.js';
import { buildAgentBar } from '../../../src/osprey/interfaces/web_terminal/static/js/agent-bar.js';
import { emitChat } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js';

const STATIC = resolve(process.cwd(), 'src/osprey/interfaces/web_terminal/static');

/** The parts of the Agent bar, in order, by the class that names each. */
const PARTS = ['agent-bar-label', 'agent-bar-spacer', 'agent-bar-new'];

/** @param {Element} el @returns {string} the part a child of the bar plays */
const partOf = (el) => PARTS.find((part) => el.classList.contains(part)) ?? el.outerHTML;

/** index.html's terminal header, as the page ships it. */
function terminalHeader() {
  const html = readFileSync(resolve(STATIC, 'index.html'), 'utf8');
  const block = html.match(/<div class="terminal-header[^"]*">[\s\S]*?<\/div>/)?.[0] ?? '';
  const template = document.createElement('template');
  template.innerHTML = block;
  return /** @type {HTMLElement} */ (template.content.firstElementChild);
}

describe('one bar on both surfaces', () => {
  test("the chat's bar reads the session label, a spacer and +", () => {
    const { bar, label, newBtn } = buildAgentBar();
    expect([...bar.children].map(partOf)).toEqual(PARTS);
    expect(bar.firstElementChild).toBe(label);
    expect(label.textContent).toBe('Agent');
    expect(bar.children[2]).toBe(newBtn);
  });

  test("the terminal's header in index.html has the same parts in the same order", () => {
    const header = terminalHeader();
    expect(header.classList.contains('agent-bar')).toBe(true);
    expect([...header.children].map(partOf)).toEqual(PARTS);
    const { bar } = buildAgentBar();
    expect([...header.children].map((el) => el.tagName)).toEqual(
      [...bar.children].map((el) => el.tagName)
    );
  });

  test('the terminal keeps the ids terminal.js and app.js wire', () => {
    const header = terminalHeader();
    expect(header.children[0].id).toBe('terminal-label');
    expect(header.children[2].id).toBe('new-session-btn');
  });

  test('+ is a named plus icon on both, New conversation in chat and New session on the terminal', () => {
    const chat = buildAgentBar().newBtn;
    const terminal = /** @type {HTMLButtonElement} */ (terminalHeader().children[2]);
    for (const [btn, name] of /** @type {const} */ ([
      [chat, 'New conversation'],
      [terminal, 'New session'],
    ])) {
      expect(btn.classList.contains('osprey-icon-btn')).toBe(true);
      expect(btn.type).toBe('button');
      expect(btn.getAttribute('aria-label')).toBe(name);
      expect(btn.title).toBe(name);
      // An icon button: the name is the label, not visible text.
      expect(btn.textContent?.trim()).toBe('');
    }
    expect(terminal.querySelector('svg')?.getAttribute('data-icon')).toBe('plus');
    expect(chat.querySelector('svg')?.innerHTML).toBe(icon('plus').innerHTML);
  });

  test('neither bar has a status dot, a bell, Export, a "+ New" text button or a ⋯', () => {
    for (const bar of [buildAgentBar().bar, terminalHeader()]) {
      expect(bar.querySelectorAll('button')).toHaveLength(1);
      expect(bar.querySelector('.osprey-dot, .session-led, [aria-haspopup]')).toBeNull();
      expect(bar.textContent?.trim()).toBe('Agent');
    }
  });
});

describe('+', () => {
  test('runs the action handed in', () => {
    const onNew = vi.fn();
    buildAgentBar({ onNew }).newBtn.click();
    expect(onNew).toHaveBeenCalledTimes(1);
  });

  test('without an action a press does nothing and throws nothing', () => {
    expect(() => buildAgentBar().newBtn.click()).not.toThrow();
  });

  test('is withheld while the chat is busy and back when it is not', () => {
    const onNew = vi.fn();
    const { newBtn } = buildAgentBar({ onNew });
    emitChat('busy', { busy: true });
    expect(newBtn.disabled).toBe(true);
    newBtn.click();
    expect(onNew).not.toHaveBeenCalled();
    emitChat('busy', { busy: false });
    expect(newBtn.disabled).toBe(false);
  });
});
