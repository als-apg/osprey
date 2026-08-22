// @ts-check
/**
 * Unit tests for selecting and copying text out of the Web Terminal
 * (terminal.js's xterm options and custom key handler):
 *   npx vitest run tests/interfaces/web_terminal/terminal-copy.test.mjs
 *
 * The agent running in the terminal switches xterm's mouse tracking on
 * (DECSET 1000/1002/1003/1006): a plain click-drag is the agent's OWN text
 * selection, which it completes by asking the terminal to copy the text
 * (OSC 52) — exactly what it does in a desktop terminal, where "select"
 * already means "copy". xterm.js only honours OSC 52 through the clipboard
 * addon, so the first group of tests pins that the addon is loaded with a
 * write-only provider (the agent may fill the clipboard, never read it) and
 * that the write degrades to the legacy copy command on plain-http pages.
 *
 * The terminal's own selection stays reachable through a modifier-drag:
 * Shift on Linux/Windows unconditionally, Option on macOS ONLY when
 * `macOptionClickForcesSelection` is set. Copying THAT needs a key that never
 * reaches the agent: Cmd+C is the browser's own copy on macOS, and
 * Ctrl+Shift+C is claimed for everything else, because plain Ctrl+C must
 * keep interrupting the agent.
 *
 * Same isolation scheme as terminal-resume.test.mjs: fresh module per test,
 * xterm and WebSocket stubbed.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js')} */
let terminal;

/** Fake xterm.js Terminal that records its options and the custom key handler. */
class FakeTerminal {
  /** @param {any} options */
  constructor(options) {
    this.cols = 80;
    this.rows = 24;
    this.options = options ?? {};
    /** @type {((ev: KeyboardEvent) => boolean)|null} */
    this.keyHandler = null;
    this.selection = '';
    /** @type {any[]} */
    this.addons = [];
    FakeTerminal.last = this;
  }
  /** @param {any} addon */
  loadAddon(addon) {
    this.addons.push(addon);
  }
  open() {}
  onData() {}
  onResize() {}
  write() {}
  reset() {}
  focus() {}
  /** @param {(ev: KeyboardEvent) => boolean} handler */
  attachCustomKeyEventHandler(handler) {
    this.keyHandler = handler;
  }
  hasSelection() {
    return this.selection !== '';
  }
  getSelection() {
    return this.selection;
  }
}
/** @type {FakeTerminal|null} */
FakeTerminal.last = null;

class FakeAddon {
  fit() {}
}

/** Fake `@xterm/addon-clipboard` Base64 codec, standing in for the real one. */
class FakeBase64 {}

/**
 * Fake `@xterm/addon-clipboard` that records what it was built with. The
 * shipped constructor is (base64Codec, provider) — NOT the (provider) its
 * typings advertise — and `initTerminal()` must honour the real order.
 */
class FakeClipboardAddon {
  /**
   * @param {any} base64
   * @param {any} provider
   */
  constructor(base64, provider) {
    this.base64 = base64;
    this.provider = provider;
  }
}

/** @returns {{ readText: (selection: string) => string, writeText: (selection: string, text: string) => Promise<void> }} */
function clipboardProvider() {
  const addon = term().addons.find((a) => a instanceof FakeClipboardAddon);
  if (!addon) throw new Error('initTerminal() loaded no clipboard addon');
  return addon.provider;
}

class FakeWebSocket {
  /** @param {string} url */
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    /** @type {any[]} */
    this.sent = [];
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
  }
  /** @param {any} data */
  send(data) {
    this.sent.push(data);
  }
  close() {}
}

/**
 * @param {Partial<KeyboardEvent> & {key: string}} init
 * @returns {KeyboardEvent}
 */
function keyEvent(init) {
  const { type = 'keydown', ...rest } = init;
  return new KeyboardEvent(type, { bubbles: true, cancelable: true, ...rest });
}

/** @returns {FakeTerminal} */
function term() {
  return /** @type {FakeTerminal} */ (FakeTerminal.last);
}

/** @returns {(ev: KeyboardEvent) => boolean} */
function handler() {
  const fn = term().keyHandler;
  if (!fn) throw new Error('initTerminal() attached no custom key handler');
  return fn;
}

/** The stubbed `navigator.clipboard.writeText` / `readText`, fresh per test. */
let writeText = vi.fn(() => Promise.resolve());
let readText = vi.fn(() => Promise.resolve('secret from another app'));

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();

  document.body.innerHTML = '<div><div id="terminal-container"></div></div>';
  // @ts-expect-error -- test stub, not a full FontFaceSet
  document.fonts = { ready: Promise.resolve() };

  vi.stubGlobal('Terminal', FakeTerminal);
  vi.stubGlobal('FitAddon', { FitAddon: FakeAddon });
  vi.stubGlobal('WebLinksAddon', { WebLinksAddon: class {} });
  vi.stubGlobal('ClipboardAddon', { ClipboardAddon: FakeClipboardAddon, Base64: FakeBase64 });
  vi.stubGlobal('WebSocket', FakeWebSocket);
  writeText = vi.fn(() => Promise.resolve());
  readText = vi.fn(() => Promise.resolve('secret from another app'));
  vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText, readText }, userAgent: navigator.userAgent });
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});

  FakeTerminal.last = null;
  terminal = await import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js');
  terminal.initTerminal('terminal-container');
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("the agent's own selection (OSC 52 through the clipboard addon)", () => {
  test('the clipboard addon is loaded with the codec first and the provider second', () => {
    const addon = term().addons.find((a) => a instanceof FakeClipboardAddon);
    if (!addon) throw new Error('initTerminal() loaded no clipboard addon');
    expect(addon.base64).toBeInstanceOf(FakeBase64);
    expect(typeof addon.provider.writeText).toBe('function');
    expect(typeof addon.provider.readText).toBe('function');
  });

  test('a copy request from the agent lands on the system clipboard', async () => {
    await clipboardProvider().writeText('c', 'SR:BPM:01:X  1.234 mm');

    expect(writeText).toHaveBeenCalledWith('SR:BPM:01:X  1.234 mm');
  });

  test('the agent can never read the clipboard — queries answer empty without touching the browser API', () => {
    expect(clipboardProvider().readText('c')).toBe('');
    expect(readText).not.toHaveBeenCalled();
  });

  test('on a page without the async clipboard API (plain http) the legacy copy command is used', async () => {
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined, userAgent: navigator.userAgent });
    const execCommand = vi.fn(() => true);
    document.execCommand = execCommand;

    await clipboardProvider().writeText('c', 'legacy text');

    expect(execCommand).toHaveBeenCalledWith('copy');
    expect(writeText).not.toHaveBeenCalled();
    // The scratch element used to stage the text does not linger in the page.
    expect(document.querySelector('textarea')).toBeNull();
  });

  test('a refused copy is reported in the console, never thrown at the addon', async () => {
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined, userAgent: navigator.userAgent });
    document.execCommand = vi.fn(() => false);

    await expect(clipboardProvider().writeText('c', 'text')).resolves.toBeUndefined();

    expect(console.warn).toHaveBeenCalled();
    expect(document.querySelector('textarea')).toBeNull();
  });
});

describe("the terminal's own selection (modifier-drag)", () => {
  test('Option-drag forces a selection on macOS (macOptionClickForcesSelection)', () => {
    expect(term().options.macOptionClickForcesSelection).toBe(true);
  });
});

describe('copying a selection', () => {
  test('Ctrl+Shift+C copies the selection and is swallowed, never reaching the agent', () => {
    term().selection = 'SR:BPM:01:X  1.234 mm';

    const outcome = handler()(keyEvent({ key: 'C', ctrlKey: true, shiftKey: true }));

    expect(writeText).toHaveBeenCalledWith('SR:BPM:01:X  1.234 mm');
    expect(outcome).toBe(false);
  });

  test('Ctrl+Shift+C with a lowercase key reading is the same chord', () => {
    term().selection = 'text';

    const outcome = handler()(keyEvent({ key: 'c', ctrlKey: true, shiftKey: true }));

    expect(writeText).toHaveBeenCalledWith('text');
    expect(outcome).toBe(false);
  });

  test('Ctrl+Shift+C with nothing selected is still swallowed (it is never an interrupt)', () => {
    const outcome = handler()(keyEvent({ key: 'C', ctrlKey: true, shiftKey: true }));

    expect(writeText).not.toHaveBeenCalled();
    expect(outcome).toBe(false);
  });

  test('plain Ctrl+C keeps interrupting the agent even with a selection', () => {
    term().selection = 'text';

    const outcome = handler()(keyEvent({ key: 'c', ctrlKey: true }));

    expect(writeText).not.toHaveBeenCalled();
    expect(outcome).toBe(true);
  });

  test('Cmd+C is left to the browser, which copies the selection natively', () => {
    term().selection = 'text';

    const outcome = handler()(keyEvent({ key: 'c', metaKey: true }));

    expect(writeText).not.toHaveBeenCalled();
    expect(outcome).toBe(true);
  });

  test('only keydown is acted on — the keyup of the same chord passes through', () => {
    term().selection = 'text';

    const outcome = handler()(keyEvent({ type: 'keyup', key: 'C', ctrlKey: true, shiftKey: true }));

    expect(writeText).not.toHaveBeenCalled();
    expect(outcome).toBe(true);
  });

  test('every other key passes through untouched', () => {
    for (const init of [{ key: 'a' }, { key: 'Enter' }, { key: 'c', shiftKey: true }, { key: 'v', ctrlKey: true, shiftKey: true }]) {
      expect(handler()(keyEvent(init))).toBe(true);
    }
    expect(writeText).not.toHaveBeenCalled();
  });
});
