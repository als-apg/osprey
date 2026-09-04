// @ts-check
/**
 * Unit tests for the first-contact derivations (first-contact.js):
 *   npx vitest run tests/interfaces/web_terminal/first-contact.test.mjs
 *
 * First contact is the only copy a newcomer reads before they type, so what it
 * may SAY is the whole contract: the read phrase follows the machine kind the
 * chip names, a session standing on no known machine is told nothing about
 * where values come from, and a prompt is offered only where the deployment
 * can answer it. These tests pin the derivations directly — the surfaces that
 * render them are covered where they are built.
 *
 * Seams: the chip module is mocked (it owns the machine kind, and reading a
 * real one would need a mounted chip and a served posture); terminal.js is
 * mocked in the chip suite's shape because the module graph reaches it, and
 * the factory declares every export that graph touches.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/first-contact.js';

/** Mutable stand-in for terminal.js, reachable from the hoisted vi.mock factory. */
const term = vi.hoisted(() => ({
  /** @type {string|null} */
  sessionId: /** @type {string|null} */ (null),
  /** @type {(() => void)[]} */
  listeners: [],
  paste: vi.fn(),
  focus: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => term.sessionId,
  /** @param {() => void} fn */
  onSessionChange: (fn) => term.listeners.push(fn),
  pasteToTerminal: term.paste,
  focusTerminal: term.focus,
}));

/** Mutable stand-in for the control-target chip, which owns the machine kind. */
const chip = vi.hoisted(() => ({
  /** @type {string|null} */
  kind: /** @type {string|null} */ (null),
  /** @type {((state: any) => void)[]} */
  listeners: [],
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js', () => ({
  activeKind: () => chip.kind,
  /** @param {(state: any) => void} fn */
  subscribe: (fn) => {
    chip.listeners.push(fn);
    return () => {};
  },
}));

/** Mutable stand-in for the chat transport, so `initChat` sends nothing. */
const transport = vi.hoisted(() => ({
  sendPrompt: vi.fn(() => ({ abort: vi.fn() })),
  interrupt: vi.fn(),
  deleteChat: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js', () => transport);

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/first-contact.js')} */
let fc;

/** @param {string[]} capabilities @param {boolean} logbook */
const factsOf = (capabilities, logbook) => ({ capabilities, logbook });

beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();
  term.sessionId = null;
  term.listeners = [];
  chip.kind = null;
  chip.listeners = [];
  document.documentElement.removeAttribute('data-ui-mode');
  document.body.innerHTML = '';
  fc = await import(MODULE);
});

/** The Simple view's input, in the shape insertPrompt looks for. */
function mountSimpleInput({ disabled = false } = {}) {
  document.body.innerHTML =
    '<div id="operator-container"><div class="op-input-area"><textarea></textarea></div></div>';
  const input = /** @type {HTMLTextAreaElement} */ (
    document.querySelector('#operator-container .op-input-area textarea')
  );
  input.disabled = disabled;
  return input;
}

describe('listPhrase', () => {
  test('joins nothing, one, two and three items', () => {
    expect(fc.listPhrase([])).toBe('');
    expect(fc.listPhrase(['a'])).toBe('a');
    expect(fc.listPhrase(['a', 'b'])).toBe('a and b');
    expect(fc.listPhrase(['a', 'b', 'c'])).toBe('a, b, and c');
  });
});

describe('capabilitySentence', () => {
  test('names where the values come from, per machine kind', () => {
    const facts = factsOf([], false);
    expect(fc.capabilitySentence('live', facts)).toBe('Here the agent can read live machine values.');
    expect(fc.capabilitySentence('standin', facts)).toBe(
      'Here the agent can read values from the rehearsal copy.'
    );
    expect(fc.capabilitySentence('va', facts)).toBe(
      'Here the agent can read values from the simulator.'
    );
    expect(fc.capabilitySentence('simulated', facts)).toBe('Here the agent can read demo data.');
  });

  test('a demo deployment is never told it reads the machine', () => {
    const sentence = fc.capabilitySentence('simulated', factsOf([], false));
    expect(sentence).toContain('read demo data');
    expect(sentence).not.toContain('live machine');
  });

  test('the read phrase leads, then the server capabilities', () => {
    const facts = factsOf(['run analysis scripts', 'plot archived data'], false);
    expect(fc.capabilitySentence('live', facts)).toBe(
      'Here the agent can read live machine values, run analysis scripts, and plot archived data.'
    );
  });

  test('no known machine means no claim about where values come from', () => {
    expect(fc.capabilitySentence(null, factsOf(['run analysis scripts'], false))).toBe(
      'Here the agent can run analysis scripts.'
    );
  });

  test('is empty when there is nothing to claim', () => {
    expect(fc.capabilitySentence(null, factsOf([], false))).toBe('');
  });
});

describe('capabilityPhrases', () => {
  test('does not mutate the caller facts', () => {
    const facts = factsOf(['run analysis scripts'], false);
    fc.capabilityPhrases('live', facts);
    expect(facts.capabilities).toEqual(['run analysis scripts']);
  });
});

describe('starterPrompts', () => {
  test('offers the read prompt for every known machine kind', () => {
    for (const kind of ['live', 'standin', 'va', 'simulated']) {
      expect(fc.starterPrompts(kind, factsOf([], false))).toEqual([
        'What can you read right now?',
        'What are you allowed to do in this session?',
      ]);
    }
  });

  test('drops the read prompt when no machine is known', () => {
    expect(fc.starterPrompts(null, factsOf([], false))).toEqual([
      'What are you allowed to do in this session?',
    ]);
  });

  test('offers the logbook prompt last, and only where there is a logbook', () => {
    expect(fc.starterPrompts('live', factsOf([], true))).toEqual([
      'What can you read right now?',
      'What are you allowed to do in this session?',
      'What happened in the logbook today?',
    ]);
    expect(fc.starterPrompts('live', factsOf([], false))).not.toContain(
      'What happened in the logbook today?'
    );
  });
});

describe('setFacts', () => {
  test('the defaults read module state at call time, not at import time', () => {
    chip.kind = 'live';
    expect(fc.capabilitySentence()).toBe('Here the agent can read live machine values.');
    expect(fc.starterPrompts()).toEqual([
      'What can you read right now?',
      'What are you allowed to do in this session?',
    ]);

    fc.setFacts({ capabilities: ['plot archived data'], logbook: true });
    expect(fc.capabilitySentence()).toBe(
      'Here the agent can read live machine values and plot archived data.'
    );
    expect(fc.starterPrompts()).toEqual([
      'What can you read right now?',
      'What are you allowed to do in this session?',
      'What happened in the logbook today?',
    ]);
  });

  test('a failed read settles on empty facts rather than the last ones', () => {
    fc.setFacts({ capabilities: ['plot archived data'], logbook: true });
    fc.setFacts(null);
    expect(fc.capabilitySentence()).toBe('');
    expect(fc.starterPrompts()).toEqual(['What are you allowed to do in this session?']);
  });

  test('missing fields are the same as absent facts', () => {
    fc.setFacts({});
    expect(fc.capabilitySentence()).toBe('');
    expect(fc.starterPrompts()).toEqual(['What are you allowed to do in this session?']);
  });

  test('copies the capability array, so a later server read cannot rewrite it', () => {
    const capabilities = ['plot archived data'];
    fc.setFacts({ capabilities, logbook: false });
    capabilities.push('move the machine');
    expect(fc.capabilitySentence(null)).toBe('Here the agent can plot archived data.');
  });
});

describe('insertPrompt', () => {
  test('expert pastes into the terminal, then focuses it', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    fc.insertPrompt('What can you read right now?');
    expect(term.paste).toHaveBeenCalledWith('What can you read right now?');
    expect(term.focus).toHaveBeenCalledTimes(1);
    expect(term.paste.mock.invocationCallOrder[0]).toBeLessThan(
      term.focus.mock.invocationCallOrder[0]
    );
  });

  test('inserts no trailing newline, so nothing submits itself', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    fc.insertPrompt('one prompt');
    expect(term.paste).toHaveBeenCalledWith('one prompt');
    expect(term.paste.mock.calls[0][0]).not.toContain('\n');
    expect(term.paste.mock.calls[0][0]).not.toContain('\r');
  });

  test('an unrendered or unknown mode takes the terminal branch', () => {
    fc.insertPrompt('no mode attribute');
    document.documentElement.setAttribute('data-ui-mode', 'something-new');
    fc.insertPrompt('unknown mode');
    expect(term.paste).toHaveBeenCalledTimes(2);
    expect(term.focus).toHaveBeenCalledTimes(2);
  });

  test('simple fills the chat input, raises input, and focuses it', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    let seen = 0;
    input.addEventListener('input', () => {
      seen += 1;
    });

    fc.insertPrompt('What happened in the logbook today?');

    expect(input.value).toBe('What happened in the logbook today?');
    expect(seen).toBe(1);
    expect(document.activeElement).toBe(input);
    expect(term.paste).not.toHaveBeenCalled();
  });

  test('simple does nothing while the input is disabled', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput({ disabled: true });
    let seen = 0;
    input.addEventListener('input', () => {
      seen += 1;
    });

    fc.insertPrompt('while streaming');

    expect(input.value).toBe('');
    expect(seen).toBe(0);
    expect(document.activeElement).not.toBe(input);
    expect(term.paste).not.toHaveBeenCalled();
  });

  test('simple with no input on the page does not throw', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    expect(() => fc.insertPrompt('nowhere to go')).not.toThrow();
    expect(term.paste).not.toHaveBeenCalled();
  });
});

describe('insertPrompt({ append: true })', () => {
  /** Every submit that reached the document while the body ran. */
  function countingSubmits(/** @type {() => void} */ body) {
    let submits = 0;
    const onSubmit = () => {
      submits += 1;
    };
    document.addEventListener('submit', onSubmit, true);
    try {
      body();
    } finally {
      document.removeEventListener('submit', onSubmit, true);
    }
    return submits;
  }

  test('appends below what the operator already typed, on its own line', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'set these to nominal:';

    fc.insertPrompt('SR:BPM1:X SR:BPM2:X', { append: true });

    expect(input.value).toBe('set these to nominal:\nSR:BPM1:X SR:BPM2:X');
    expect(document.activeElement).toBe(input);
  });

  test('does not stack blank lines on trailing whitespace', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'first line\n\n  ';

    fc.insertPrompt('SR:BPM1:X', { append: true });

    expect(input.value).toBe('first line\nSR:BPM1:X');
  });

  test('replaces an empty input rather than opening with a newline', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();

    fc.insertPrompt('SR:BPM1:X', { append: true });

    expect(input.value).toBe('SR:BPM1:X');
  });

  test('an input holding only whitespace is replaced, not appended to', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = '   \n ';

    fc.insertPrompt('SR:BPM1:X', { append: true });

    expect(input.value).toBe('SR:BPM1:X');
  });

  test('raises one input event, so the textarea regrows exactly once', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'already typed';
    let seen = 0;
    input.addEventListener('input', () => {
      seen += 1;
    });

    fc.insertPrompt('SR:BPM1:X', { append: true });

    expect(seen).toBe(1);
  });

  test('leaves a disabled input untouched, appended text and all', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput({ disabled: true });
    input.value = 'mid-turn text';

    fc.insertPrompt('SR:BPM1:X', { append: true });

    expect(input.value).toBe('mid-turn text');
    expect(document.activeElement).not.toBe(input);
  });

  test('never submits: the operator still presses Enter themselves', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'already typed';

    const submits = countingSubmits(() => fc.insertPrompt('SR:BPM1:X', { append: true }));

    expect(submits).toBe(0);
    expect(input.value.endsWith('\n')).toBe(false);
  });

  test('expert ignores append and pastes the text as-is', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    fc.insertPrompt('SR:BPM1:X SR:BPM2:X', { append: true });
    expect(term.paste).toHaveBeenCalledWith('SR:BPM1:X SR:BPM2:X');
    expect(term.focus).toHaveBeenCalledTimes(1);
  });

  test('the default is still replace, for the prompt chips', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'half a question';

    fc.insertPrompt('What can you read right now?');

    expect(input.value).toBe('What can you read right now?');
  });
});

/* ---- the settled moment ------------------------------------------------- */

/** Announce a session id, the way terminal.js's onSessionChange does. */
const announceSession = () => term.listeners.forEach((fn) => fn());

/** One chip render, the way control-target-chip.js notifies its subscribers. */
const renderChip = () => chip.listeners.forEach((fn) => fn({}));

describe('the settled moment', () => {
  test('fires once, whichever order the three inputs arrive in', () => {
    const settled = vi.fn();
    fc.onSettled(settled);

    fc.setFacts({ capabilities: [], logbook: false });
    expect(settled).not.toHaveBeenCalled();
    announceSession();
    expect(settled).not.toHaveBeenCalled();
    renderChip();
    expect(settled).toHaveBeenCalledTimes(1);

    // Nothing later re-fires it.
    renderChip();
    fc.setFacts({ capabilities: ['plot archived data'], logbook: true });
    expect(settled).toHaveBeenCalledTimes(1);
  });

  test('settles when the facts arrive last', () => {
    const settled = vi.fn();
    fc.onSettled(settled);

    announceSession();
    renderChip();
    expect(settled).not.toHaveBeenCalled();
    fc.setFacts({ capabilities: [], logbook: false });
    expect(settled).toHaveBeenCalledTimes(1);
  });

  test('a failed server read settles the same as a good one', () => {
    const settled = vi.fn();
    fc.onSettled(settled);

    fc.setFacts(null);
    announceSession();
    renderChip();
    expect(settled).toHaveBeenCalledTimes(1);
  });

  test('a chip render before the session is known does not count', () => {
    const settled = vi.fn();
    fc.onSettled(settled);

    fc.setFacts(null);
    renderChip();
    announceSession();
    expect(settled).not.toHaveBeenCalled();
    renderChip();
    expect(settled).toHaveBeenCalledTimes(1);
  });

  test('settles synchronously, with no timer behind it', () => {
    vi.useFakeTimers();
    try {
      const settled = vi.fn();
      fc.onSettled(settled);
      fc.setFacts(null);
      announceSession();
      renderChip();
      expect(settled).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  test('a listener registered after the moment fires immediately', () => {
    fc.setFacts(null);
    announceSession();
    renderChip();

    const late = vi.fn();
    fc.onSettled(late);
    expect(late).toHaveBeenCalledTimes(1);
  });
});

describe('onKindChange', () => {
  /**
   * Settle the module with the chip standing on `kind`.
   * @param {string|null} kind
   */
  function settleOn(kind) {
    chip.kind = kind;
    fc.setFacts(null);
    announceSession();
    renderChip();
  }

  test('the idle poll repainting the same state fires nothing', () => {
    const changed = vi.fn();
    fc.onKindChange(changed);
    settleOn('live');

    renderChip();
    renderChip();
    expect(changed).not.toHaveBeenCalled();
  });

  test('a switch fires once, with the new kind', () => {
    const changed = vi.fn();
    fc.onKindChange(changed);
    settleOn('live');

    chip.kind = 'standin';
    renderChip();
    expect(changed).toHaveBeenCalledTimes(1);
    expect(changed).toHaveBeenCalledWith('standin');

    renderChip();
    expect(changed).toHaveBeenCalledTimes(1);
  });

  test('does not fire at the settled moment itself', () => {
    const changed = vi.fn();
    fc.onKindChange(changed);
    settleOn('simulated');
    expect(changed).not.toHaveBeenCalled();
  });
});

/* ---- the Simple view's empty state -------------------------------------- */

const CHAT_MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/chat.js';

/** The chips a block is offering, as their labels. */
const chipTexts = (/** @type {Element} */ block) =>
  [...block.querySelectorAll('.tour-chips button.tour-chip')].map((c) => c.textContent);

describe('buildEmptyState', () => {
  test('the intro emphasises each phrase and reads as the sentence', () => {
    chip.kind = 'live';
    fc.setFacts({ capabilities: ['run analysis scripts'], logbook: false });
    const block = fc.buildEmptyState();

    const intro = /** @type {HTMLElement} */ (block.querySelector('p.op-empty-intro'));
    expect([...intro.querySelectorAll('strong')].map((s) => s.textContent)).toEqual([
      'read live machine values',
      'run analysis scripts',
    ]);
    expect(intro.textContent).toBe(fc.capabilitySentence());
    expect(intro.textContent).toBe(
      'Here the agent can read live machine values and run analysis scripts.'
    );
  });

  test('one phrase needs no connective, three read as a list', () => {
    chip.kind = 'simulated';
    fc.setFacts(null);
    expect(fc.buildEmptyState().querySelector('.op-empty-intro')?.textContent).toBe(
      'Here the agent can read demo data.'
    );

    fc.setFacts({ capabilities: ['run analysis scripts', 'plot archived data'], logbook: false });
    expect(fc.buildEmptyState().querySelector('.op-empty-intro')?.textContent).toBe(
      'Here the agent can read demo data, run analysis scripts, and plot archived data.'
    );
  });

  test('omits the intro entirely when there is nothing to claim', () => {
    fc.setFacts(null);
    const block = fc.buildEmptyState();
    expect(fc.capabilitySentence()).toBe('');
    expect(block.querySelector('.op-empty-intro')).toBeNull();
    expect(chipTexts(block)).toEqual(['What are you allowed to do in this session?']);
  });

  test('offers the tour chips, in the tour classes', () => {
    chip.kind = 'live';
    fc.setFacts({ capabilities: [], logbook: true });
    const block = fc.buildEmptyState();

    expect(block.className).toBe('op-empty');
    expect(block.querySelectorAll('.tour-chips').length).toBe(1);
    expect(chipTexts(block)).toEqual(fc.starterPrompts());
    expect(chipTexts(block)).toEqual([
      'What can you read right now?',
      'What are you allowed to do in this session?',
      'What happened in the logbook today?',
    ]);
    for (const button of block.querySelectorAll('.tour-chip')) {
      expect(button.tagName).toBe('BUTTON');
      expect(/** @type {HTMLButtonElement} */ (button).type).toBe('button');
    }
  });

  test('clicking a chip inserts that chip text', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    chip.kind = 'live';
    fc.setFacts(null);
    const block = fc.buildEmptyState();

    /** @type {HTMLButtonElement[]} */ ([...block.querySelectorAll('.tour-chip')])[0].click();
    expect(term.paste).toHaveBeenCalledWith('What can you read right now?');
    expect(term.focus).toHaveBeenCalledTimes(1);
  });
});

describe('renderEmptyStateContent', () => {
  test('rebuilds the same element for a different machine', () => {
    chip.kind = 'live';
    fc.setFacts(null);
    const block = fc.buildEmptyState();
    expect(block.textContent).toContain('read live machine values');

    chip.kind = 'simulated';
    expect(fc.renderEmptyStateContent(block)).toBe(block);
    expect(block.textContent).toContain('read demo data');
    expect(block.textContent).not.toContain('live machine');
    expect(block.querySelectorAll('.op-empty-intro').length).toBe(1);
    expect(block.querySelectorAll('.tour-chips').length).toBe(1);
  });
});

describe('the Simple console', () => {
  /** Mount the operator console and hand back its message list. */
  async function mountChat() {
    document.body.innerHTML = '<div id="operator-container"></div>';
    const chat = await import(CHAT_MODULE);
    chat.initChat();
    return /** @type {HTMLElement} */ (document.querySelector('.op-messages'));
  }

  /** Settle first contact with the chip standing on `kind`. */
  function settleOn(/** @type {string|null} */ kind) {
    chip.kind = kind;
    fc.setFacts({ capabilities: [], logbook: false });
    term.listeners.forEach((fn) => fn());
    chip.listeners.forEach((fn) => fn({}));
  }

  /** Send one message the way an operator does. */
  function sendMessage(/** @type {string} */ text) {
    const input = /** @type {HTMLTextAreaElement} */ (
      document.querySelector('#operator-container .op-input-area textarea')
    );
    input.value = text;
    /** @type {HTMLButtonElement} */ (document.querySelector('.op-send-btn')).click();
  }

  test('shows the block, as the first thing in the log', async () => {
    const messages = await mountChat();
    expect(messages.querySelector('.op-empty')).toBeNull();

    settleOn('live');
    const block = messages.querySelector('.op-empty');
    expect(block).not.toBeNull();
    expect(messages.firstChild).toBe(block);
    expect(block?.textContent).toContain('read live machine values');
  });

  test('does not show it over a log that already has messages in it', async () => {
    const messages = await mountChat();
    sendMessage('already talking');
    expect(messages.querySelector('.op-empty')).toBeNull();

    settleOn('live');
    expect(messages.querySelector('.op-empty')).toBeNull();
  });

  test('removes it as soon as the invitation is taken', async () => {
    const messages = await mountChat();
    settleOn('live');
    expect(messages.querySelector('.op-empty')).not.toBeNull();

    sendMessage('what can you read right now?');
    expect(messages.querySelector('.op-empty')).toBeNull();
  });

  test('rebuilds it in place when the machine changes', async () => {
    const messages = await mountChat();
    settleOn('live');

    chip.kind = 'simulated';
    chip.listeners.forEach((fn) => fn({}));

    const block = messages.querySelector('.op-empty');
    expect(messages.querySelectorAll('.op-empty').length).toBe(1);
    expect(block?.textContent).toContain('read demo data');
    expect(block?.textContent).not.toContain('live machine');
  });

  test('does not resurrect it after the first message', async () => {
    const messages = await mountChat();
    settleOn('live');
    sendMessage('first message');

    chip.kind = 'simulated';
    chip.listeners.forEach((fn) => fn({}));
    expect(messages.querySelector('.op-empty')).toBeNull();
  });
});
