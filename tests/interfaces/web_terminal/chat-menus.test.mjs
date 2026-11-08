// @ts-check
/**
 * The chat's place menus (chat-menus.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-menus.test.mjs
 *
 * Covered: the message menu (Copy, Quote in prompt) on user and assistant
 * entries, the background menu (Scroll to latest, Export conversation, New
 * session), every case that leaves the browser its own menu (links, buttons,
 * form controls, code blocks, tool rows and cards, a selection in the message,
 * the composer), the keyboard route on a focused entry, focus on click, and
 * uninstall.
 *
 * context-menu.js and place-menu.js are the real modules. Module state is
 * isolated by vi.resetModules() + dynamic import per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/chat-menus.js';
const PLACES = '../../../src/osprey/interfaces/web_terminal/static/js/place-menu.js';
const MENU = '../../../src/osprey/interfaces/design_system/static/js/context-menu.js';

/** @type {any} */ let chatMenus;
/** @type {any} */ let pm;
/** @type {any} */ let cm;
/** @type {() => void} */ let uninstall = () => {};
/** @type {{ scrollToBottom: any, onNewSession: any, onExport: any }} */
let spies;

beforeEach(async () => {
  vi.resetModules();
  cm = await import(MENU);
  pm = await import(PLACES);
  chatMenus = await import(MODULE);
  document.body.innerHTML = `
    <div class="op-messages" role="log">
      <div class="op-entry operator"><div class="op-bubble">Hello
second line</div></div>
      <div class="op-entry assistant">
        <div class="op-entry-body osprey-md-rendered"><p>First <a href="#x">link</a> para</p>
          <pre><code>x = 1</code></pre></div>
        <div class="op-row op-tool"><button class="op-row-head">tool</button><span class="tool-text">Read</span></div>
        <section class="op-card op-question"><span class="card-text">Pick one</span></section>
        <div class="op-entry-body osprey-md-rendered"><p>Second para</p></div>
        <button class="op-reply-copy">Copy</button>
        <input class="field-in" type="checkbox">
        <select class="field-sel"><option>a</option></select>
      </div>
      <div class="gap"></div>
    </div>
    <div class="op-input-area"><textarea class="composer"></textarea></div>`;
  spies = { scrollToBottom: vi.fn(), onNewSession: vi.fn(), onExport: vi.fn() };
  uninstall = chatMenus.installChatMenus({
    messages: el('.op-messages'),
    textarea: el('.composer'),
    ...spies,
  });
  pm.installPlaceMenus(document);
});

afterEach(() => {
  uninstall();
  cm.closeContextMenu();
  pm.resetPlaceMenus([document]);
  document.getSelection()?.removeAllRanges();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

/** @param {string} sel */
const el = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));
const openMenu = () => document.querySelector('.osprey-context-menu[role="menu"]');
const rowLabels = () =>
  [...(openMenu()?.querySelectorAll('[role="menuitem"]') ?? [])].map((r) =>
    (r.textContent ?? '').trim()
  );
/** @param {string} label */
function clickRow(label) {
  const row = [...(openMenu()?.querySelectorAll('[role="menuitem"]') ?? [])].find((r) =>
    (r.textContent ?? '').includes(label)
  );
  if (!row) throw new Error(`no row ${label}`);
  /** @type {HTMLElement} */ (row).click();
}

/** @param {Element} target */
function rightClick(target) {
  const e = new MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    button: 2,
    clientX: 40,
    clientY: 50,
  });
  target.dispatchEvent(e);
  return e;
}

/** @param {Node} node */
function selectContents(node) {
  const range = document.createRange();
  range.selectNodeContents(node);
  const sel = /** @type {Selection} */ (document.getSelection());
  sel.removeAllRanges();
  sel.addRange(range);
}

describe('message menu', () => {
  test('a user message opens Copy and Quote in prompt', () => {
    const e = rightClick(el('.op-bubble'));
    expect(e.defaultPrevented).toBe(true);
    expect(rowLabels()).toEqual([
      expect.stringContaining('Copy'),
      expect.stringContaining('Quote in prompt'),
    ]);
  });

  test('an assistant message opens the same menu from its text', () => {
    const e = rightClick(el('.op-entry.assistant p'));
    expect(e.defaultPrevented).toBe(true);
    expect(rowLabels().join('|')).toMatch(/Copy.*\|.*Quote in prompt/);
  });

  test('Copy writes the rendered text of a user message', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    rightClick(el('.op-bubble'));
    clickRow('Copy');
    await Promise.resolve();
    expect(writeText).toHaveBeenCalledWith('Hello\nsecond line');
  });

  test('Copy of an assistant message joins its text blocks, not its tool rows', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    rightClick(el('.op-entry.assistant p'));
    clickRow('Copy');
    await Promise.resolve();
    const text = writeText.mock.calls[0][0];
    expect(text).toContain('First link para');
    expect(text).toContain('Second para');
    expect(text).not.toContain('Read');
    expect(text).not.toContain('Pick one');
  });

  test('a clipboard failure is silent', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error('denied'));
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    rightClick(el('.op-bubble'));
    expect(() => clickRow('Copy')).not.toThrow();
    await Promise.resolve();
  });

  test('Quote in prompt inserts a Markdown quote at the caret and focuses the composer', () => {
    const ta = /** @type {HTMLTextAreaElement} */ (el('.composer'));
    ta.value = 'before after';
    ta.setSelectionRange(7, 7);
    const onInput = vi.fn();
    ta.addEventListener('input', onInput);
    rightClick(el('.op-bubble'));
    clickRow('Quote in prompt');
    expect(ta.value).toBe('before \n> Hello\n> second line\n\nafter');
    expect(document.activeElement).toBe(ta);
    expect(ta.selectionStart).toBe(ta.value.indexOf('after'));
    expect(onInput).toHaveBeenCalled();
  });

  test('Quote in prompt into an empty composer starts at the top', () => {
    const ta = /** @type {HTMLTextAreaElement} */ (el('.composer'));
    rightClick(el('.op-bubble'));
    clickRow('Quote in prompt');
    expect(ta.value).toBe('> Hello\n> second line\n\n');
  });

  test('Quote in prompt is disabled while the composer is closed', () => {
    const ta = /** @type {HTMLTextAreaElement} */ (el('.composer'));
    ta.disabled = true;
    rightClick(el('.op-bubble'));
    const row = [...(openMenu()?.querySelectorAll('[role="menuitem"]') ?? [])].find((r) =>
      (r.textContent ?? '').includes('Quote in prompt')
    );
    expect(row?.getAttribute('aria-disabled')).toBe('true');
  });
});

describe('background menu', () => {
  test('the log between messages opens Scroll to latest, Export conversation, New session', () => {
    const e = rightClick(el('.gap'));
    expect(e.defaultPrevented).toBe(true);
    const labels = rowLabels().join('|');
    expect(labels).toMatch(/Scroll to latest.*Export conversation.*New session/);
  });

  test('each row calls its handler', () => {
    rightClick(el('.op-messages'));
    clickRow('Scroll to latest');
    expect(spies.scrollToBottom).toHaveBeenCalledTimes(1);
    rightClick(el('.op-messages'));
    clickRow('Export conversation');
    expect(spies.onExport).toHaveBeenCalledTimes(1);
    rightClick(el('.op-messages'));
    clickRow('New session');
    expect(spies.onNewSession).toHaveBeenCalledTimes(1);
  });

  test('a non-collapsed selection in the log leaves the browser menu', () => {
    selectContents(el('.op-bubble'));
    const e = rightClick(el('.gap'));
    expect(e.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });
});

describe('declines to the browser', () => {
  const cases = /** @type {[string, string][]} */ ([
    ['a link', '.op-entry.assistant a'],
    ['a button', '.op-reply-copy'],
    ['a checkbox', '.field-in'],
    ['a select', '.field-sel'],
    ['a code block', '.op-entry.assistant code'],
    ['a pre block', '.op-entry.assistant pre'],
    ['a tool row', '.tool-text'],
    ['a card', '.card-text'],
    ['the composer', '.composer'],
  ]);
  for (const [name, sel] of cases) {
    test(`inside ${name}`, () => {
      const e = rightClick(el(sel));
      expect(e.defaultPrevented).toBe(false);
      expect(openMenu()).toBeNull();
    });
  }

  test('a non-collapsed selection in the message', () => {
    selectContents(el('.op-bubble'));
    const e = rightClick(el('.op-bubble'));
    expect(e.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('a collapsed selection does not decline', () => {
    const range = document.createRange();
    range.setStart(/** @type {Node} */ (el('.op-bubble').firstChild), 1);
    range.collapse(true);
    document.getSelection()?.removeAllRanges();
    document.getSelection()?.addRange(range);
    const e = rightClick(el('.op-bubble'));
    expect(e.defaultPrevented).toBe(true);
  });

  test('a selection in another message does not decline this one', () => {
    selectContents(el('.op-entry.assistant p'));
    const e = rightClick(el('.op-bubble'));
    expect(e.defaultPrevented).toBe(true);
  });
});

describe('focus and keyboard', () => {
  test('the log and its entries are focusable', () => {
    expect(el('.op-messages').getAttribute('tabindex')).toBe('-1');
    el('.op-bubble').click();
    const entry = el('.op-entry.operator');
    expect(entry.getAttribute('tabindex')).toBe('-1');
    expect(document.activeElement).toBe(entry);
  });

  test('a click between messages focuses the log', () => {
    el('.gap').click();
    expect(document.activeElement).toBe(el('.op-messages'));
  });

  test('a click on a control inside a message leaves focus to the control', () => {
    const btn = el('.op-reply-copy');
    btn.focus();
    btn.click();
    expect(document.activeElement).toBe(btn);
  });

  test('Shift+F10 on a focused message opens its menu', () => {
    el('.op-bubble').click();
    const entry = el('.op-entry.operator');
    const e = new KeyboardEvent('keydown', {
      key: 'F10',
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    entry.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(true);
    expect(rowLabels().join('|')).toMatch(/Copy.*Quote in prompt/);
  });
});

test('uninstall removes the places', () => {
  uninstall();
  uninstall = () => {};
  const e = rightClick(el('.op-bubble'));
  expect(e.defaultPrevented).toBe(false);
});
