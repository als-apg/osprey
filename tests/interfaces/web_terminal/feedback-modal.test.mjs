/**
 * Unit tests for the feedback modal shell (`feedback-modal.js`).
 *
 * Scope is the overlay LIFECYCLE only — the dialog's contents (text box,
 * radios, checkboxes) belong to a later task and are exercised through the
 * `render` hook here.
 *
 * Two environment notes carried over from the focus-trap and palette suites:
 *
 * - happy-dom has no layout engine, so `offsetParent` is unreliable and the
 *   shared focus trap can see zero focusables. `setVisible()` defines an own
 *   `offsetParent` data property, which is exactly what `focusableElements()`
 *   reads, so a fixture control counts as rendered whatever happy-dom does.
 * - palette.js is mocked rather than imported for real: the arbitration the
 *   modal owes it is "is the palette open right now", and driving the real
 *   palette open would drag its registry/fetch graph into this suite.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { qs } from '../_support/dom.mjs';

const paletteState = vi.hoisted(() => ({ open: false }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/palette.js', () => ({
  isOpen: () => paletteState.open,
}));

const {
  ARTIFACT_WINDOW_DISCLOSURE,
  CHANNEL_HINTS,
  FEEDBACK_DISCLOSURE,
  FEEDBACK_DISCLOSURE_PARAGRAPHS,
  FeedbackModal,
  NO_SESSION_HINT,
  isFeedbackModalOpen,
} = await import('../../../src/osprey/interfaces/web_terminal/static/js/feedback-modal.js');

const OVERLAY = '.feedback-modal-overlay';
const DIALOG = '.feedback-modal-dialog';

/** The overlay node, or null when the modal is closed (it is removed, not parked). */
function overlay() {
  return document.querySelector(OVERLAY);
}

/**
 * Make `el` count as rendered for `focusableElements()`.
 *
 * happy-dom reports `offsetParent === null` everywhere (no layout), so the
 * trap's visibility filter drops every candidate unless the fixture defines
 * the property itself.
 *
 * @param {Element} el
 */
function setVisible(el) {
  Object.defineProperty(el, 'offsetParent', {
    value: document.body,
    configurable: true,
  });
}

/** Flush the requestAnimationFrame that reveals the overlay and moves focus. */
function flushRaf() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
}

/**
 * Dispatch a keydown from a node inside `<body>` and report whether it reached
 * that node.
 *
 * The witness has to sit DOWNSTREAM of `document`: `stopPropagation()` stops
 * the event reaching other nodes, never other listeners on the same node, so a
 * second listener on `document` would fire regardless. A listener on a body
 * child is the honest stand-in for the handlers the modal must pre-empt (the
 * drawer's document-level Escape sits one node up, but the propagation the
 * modal cuts is the same one).
 *
 * @param {string} key
 * @returns {{stopped: boolean, defaultPrevented: boolean}}
 */
function pressKey(key) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  let seen = false;
  target.addEventListener('keydown', () => {
    seen = true;
  });
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
  target.dispatchEvent(event);
  target.remove();
  return { stopped: !seen, defaultPrevented: event.defaultPrevented };
}

/**
 * The drawer's background-inert pass, reduced to its essentials
 * (`osprey-drawer.js:666-686`): while a drawer is open every top-level body
 * child that is not a drawer or the backdrop is marked inert + aria-hidden.
 */
function syncBackgroundInert() {
  for (const child of Array.from(document.body.children)) {
    if (!(child instanceof HTMLElement)) continue;
    if (child.tagName.toLowerCase() === 'osprey-drawer') continue;
    if (child.id === 'drawer-backdrop') continue;
    child.toggleAttribute('inert', true);
    child.setAttribute('aria-hidden', 'true');
  }
}

// Declared without an initializer so `tsc --strict` does not widen it to
// `| null`: beforeEach always assigns it before any test body runs.
/** @type {InstanceType<typeof FeedbackModal>} */
let modal;

beforeEach(() => {
  paletteState.open = false;
  document.body.innerHTML = '';
  modal = new FeedbackModal({
    render: (body) => {
      const input = document.createElement('textarea');
      input.className = 'fixture-input';
      body.appendChild(input);
      setVisible(input);
    },
  });
});

afterEach(() => {
  modal.close();
  vi.restoreAllMocks();
});

describe('feedback modal shell — overlay lifecycle', () => {
  test('shell: open appends a scrim and an aria-modal dialog to <body>', async () => {
    modal.open();

    const root = qs(document, OVERLAY);
    expect(root.parentNode).toBe(document.body);

    const dialog = qs(root, DIALOG);
    expect(dialog.getAttribute('role')).toBe('dialog');
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.getAttribute('aria-label')).toBeTruthy();

    await flushRaf();
    expect(root.classList.contains('visible')).toBe(true);
  });

  test('shell: the render hook populates the dialog body slot', () => {
    modal.open();
    const body = qs(document, '.feedback-modal-body');
    expect(body.querySelector('.fixture-input')).not.toBeNull();
  });

  test('shell: close removes the node from the document — never parked', async () => {
    modal.open();
    await flushRaf();
    expect(overlay()).not.toBeNull();

    modal.close();

    expect(overlay()).toBeNull();
    expect(document.querySelector(DIALOG)).toBeNull();
    expect(isFeedbackModalOpen()).toBe(false);
  });

  test('shell: open/close are idempotent', () => {
    modal.open();
    modal.open();
    expect(document.querySelectorAll(OVERLAY)).toHaveLength(1);

    modal.close();
    modal.close();
    expect(overlay()).toBeNull();
  });

  test('shell: isFeedbackModalOpen tracks the open state', () => {
    expect(isFeedbackModalOpen()).toBe(false);
    modal.open();
    expect(isFeedbackModalOpen()).toBe(true);
    expect(modal.isOpen()).toBe(true);
    modal.close();
    expect(isFeedbackModalOpen()).toBe(false);
    expect(modal.isOpen()).toBe(false);
  });

  test('shell: the close button closes the dialog', () => {
    modal.open();
    qs(document, '.feedback-modal-close', HTMLButtonElement).click();
    expect(overlay()).toBeNull();
  });

  test('shell: mousedown on the scrim closes, mousedown inside the dialog does not', () => {
    modal.open();
    const root = qs(document, OVERLAY);
    qs(root, DIALOG).dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(overlay()).not.toBeNull();

    root.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(overlay()).toBeNull();
  });

  test('shell: the onClose hook fires once per close', () => {
    const onClose = vi.fn();
    const m = new FeedbackModal({ onClose });
    m.open();
    m.close();
    m.close();
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe('feedback modal shell — keyboard arbitration', () => {
  test('shell: Escape closes the modal and stops propagation', () => {
    modal.open();

    const result = pressKey('Escape');

    expect(overlay()).toBeNull();
    expect(result.stopped).toBe(true);
    expect(result.defaultPrevented).toBe(true);
  });

  test('shell: keys other than Escape are left alone', () => {
    modal.open();

    const result = pressKey('a');

    expect(overlay()).not.toBeNull();
    expect(result.stopped).toBe(false);
    expect(result.defaultPrevented).toBe(false);
  });

  test('shell: with the palette open, Escape is not consumed and the modal stays open', () => {
    modal.open();
    paletteState.open = true;

    const result = pressKey('Escape');

    expect(overlay()).not.toBeNull();
    expect(modal.isOpen()).toBe(true);
    expect(result.stopped).toBe(false);
    expect(result.defaultPrevented).toBe(false);
  });

  test('shell: the Escape listener is removed on close', () => {
    modal.open();
    modal.close();

    const result = pressKey('Escape');

    expect(result.stopped).toBe(false);
    expect(result.defaultPrevented).toBe(false);
  });
});

describe('feedback modal shell — focus', () => {
  test('shell: focus moves to the first control in the body, not the close button', async () => {
    modal.open();
    await flushRaf();
    expect(document.activeElement).toBe(qs(document, '.fixture-input'));
  });

  test('shell: an empty body falls back to the close button', async () => {
    // An explicit no-op render, because the default body is the feedback form
    // and that form's textarea would otherwise take the focus.
    const m = new FeedbackModal({ render: () => {} });
    m.open();
    await flushRaf();
    expect(document.activeElement).toBe(qs(document, '.feedback-modal-close'));
    m.close();
  });

  test('shell: Tab wraps inside the dialog (shared focus trap installed)', async () => {
    modal.open();
    const close = qs(document, '.feedback-modal-close', HTMLButtonElement);
    setVisible(close);
    await flushRaf();

    // Document order inside the dialog is [close button, fixture input], so
    // Tab off the last control must wrap back to the first.
    const last = qs(document, '.fixture-input', HTMLTextAreaElement);
    last.focus();
    expect(document.activeElement).toBe(last);

    qs(document, DIALOG).dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true })
    );

    expect(document.activeElement).toBe(close);
  });

  test('shell: focus returns to the previously focused element on close', async () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    setVisible(trigger);
    trigger.focus();

    modal.open();
    await flushRaf();
    expect(document.activeElement).not.toBe(trigger);

    modal.close();
    expect(document.activeElement).toBe(trigger);
  });

  test('shell: the focus trap is released on close', async () => {
    modal.open();
    await flushRaf();
    const dialog = qs(document, DIALOG);
    modal.close();

    // A released root swallows nothing: the trap's preventDefault on an empty
    // root would still fire if the listener were left installed.
    const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    dialog.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });
});

// ---- State-machine fixtures ----

/** Modals opened by a state test, closed in afterEach. */
/** @type {InstanceType<typeof FeedbackModal>[]} */
const spawned = [];

/**
 * Open a modal with the built-in feedback form (no `render` override).
 *
 * @param {ConstructorParameters<typeof FeedbackModal>[0]} [options]
 * @returns {InstanceType<typeof FeedbackModal>}
 */
function openForm(options = {}) {
  const m = new FeedbackModal({ getSessionId: () => 'session-abc', ...options });
  spawned.push(m);
  m.open();
  return m;
}

/** @returns {HTMLTextAreaElement} the feedback text box */
function textBox() {
  return qs(document, '.feedback-text', HTMLTextAreaElement);
}

/**
 * @param {string} selector
 * @returns {HTMLInputElement}
 */
function input(selector) {
  return qs(document, selector, HTMLInputElement);
}

/**
 * Tick or untick a checkbox the way a user would, firing the change event the
 * modal listens for.
 *
 * @param {HTMLInputElement} el
 * @param {boolean} checked
 */
function setChecked(el, checked) {
  el.checked = checked;
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

/**
 * Select a delivery channel through its radio.
 *
 * @param {'local'|'github'|'email'} channel
 */
function pickChannel(channel) {
  setChecked(input(`input[name="feedback-channel"][value="${channel}"]`), true);
}

/** @returns {HTMLButtonElement|null} */
function copyButton() {
  const el = document.querySelector('.feedback-copy-context');
  return el instanceof HTMLButtonElement ? el : null;
}

afterEach(() => {
  while (spawned.length) {
    const m = spawned.pop();
    if (m) m.close();
  }
});

describe('feedback modal state — defaults and channel switching', () => {
  test('state: defaults are local channel, metadata on, context off, empty text', () => {
    const m = openForm();

    expect(input('input[name="feedback-channel"][value="local"]').checked).toBe(true);
    expect(input('.feedback-metadata-check').checked).toBe(true);
    expect(input('.feedback-context-check').checked).toBe(false);
    expect(textBox().value).toBe('');
    expect(m.formState()).toEqual({
      text: '',
      channel: 'local',
      metadataOn: true,
      contextOn: false,
      sessionId: 'session-abc',
    });
  });

  test('state: the local action row is a single Send button', () => {
    openForm();
    expect(qs(document, '.feedback-send', HTMLButtonElement).textContent).toBe('Send');
    expect(copyButton()).toBeNull();
    expect(document.querySelector('.feedback-open-channel')).toBeNull();
  });

  test('state: the GitHub row copies, opens an issue, and warns about the account', () => {
    openForm();
    pickChannel('github');

    expect(document.querySelector('.feedback-send')).toBeNull();
    expect(copyButton()?.textContent).toBe('Copy session context');
    expect(qs(document, '.feedback-open-channel').textContent).toBe('Open GitHub issue');
    expect(qs(document, '.feedback-channel-hint').textContent).toBe(CHANNEL_HINTS.github);
    expect(CHANNEL_HINTS.github).toBe('Requires a GitHub account');
  });

  test('state: the email row opens a draft and points at Local as the fallback', () => {
    openForm();
    pickChannel('email');

    expect(copyButton()?.textContent).toBe('Copy session context');
    expect(qs(document, '.feedback-open-channel').textContent).toBe('Open email draft');
    expect(qs(document, '.feedback-channel-hint').textContent).toBe(CHANNEL_HINTS.email);
    expect(CHANNEL_HINTS.email).toBe('If nothing opens, use Local instead');
  });

  test('state: switching channels preserves typed text and checkbox state', () => {
    const m = openForm();

    textBox().value = 'the rail button does not respond';
    textBox().dispatchEvent(new Event('input', { bubbles: true }));
    setChecked(input('.feedback-metadata-check'), false);
    setChecked(input('.feedback-context-check'), true);

    pickChannel('github');
    pickChannel('email');
    pickChannel('local');

    expect(textBox().value).toBe('the rail button does not respond');
    expect(input('.feedback-metadata-check').checked).toBe(false);
    expect(input('.feedback-context-check').checked).toBe(true);
    expect(m.formState()).toEqual({
      text: 'the rail button does not respond',
      channel: 'local',
      metadataOn: false,
      contextOn: true,
      sessionId: 'session-abc',
    });
  });

  test('state: a draft survives close and reopen', () => {
    const m = openForm();
    textBox().value = 'half-written report';
    textBox().dispatchEvent(new Event('input', { bubbles: true }));
    pickChannel('github');

    m.close();
    m.open();

    expect(textBox().value).toBe('half-written report');
    expect(input('input[name="feedback-channel"][value="github"]').checked).toBe(true);
  });
});

describe('feedback modal state — context availability', () => {
  test('state: Copy is disabled until the context box is ticked', () => {
    openForm();
    pickChannel('github');
    expect(copyButton()?.disabled).toBe(true);

    setChecked(input('.feedback-context-check'), true);
    expect(copyButton()?.disabled).toBe(false);

    setChecked(input('.feedback-context-check'), false);
    expect(copyButton()?.disabled).toBe(true);
  });

  test('state: without a session the context box is disabled and hinted', () => {
    openForm({ getSessionId: () => null });

    const contextBox = input('.feedback-context-check');
    expect(contextBox.disabled).toBe(true);
    expect(contextBox.checked).toBe(false);
    expect(qs(document, '.feedback-context-hint').textContent).toBe(NO_SESSION_HINT);
    expect(NO_SESSION_HINT).toBe('no session yet');
  });

  test('state: with a session the context box is enabled and unhinted', () => {
    openForm();
    expect(input('.feedback-context-check').disabled).toBe(false);
    expect(qs(document, '.feedback-context-hint').textContent).toBe('');
  });

  test('state: Copy stays disabled with no session even if context was ticked', () => {
    const m = openForm({ getSessionId: () => null });
    pickChannel('github');
    // Force the flag past the disabled control — the action row must still
    // refuse, because the bundle request has no session to compose from.
    setChecked(input('.feedback-context-check'), true);

    expect(copyButton()?.disabled).toBe(true);
    expect(m.formState().sessionId).toBeNull();
  });

  test('state: text and metadata still submit with no session', () => {
    const onSubmit = vi.fn();
    openForm({ getSessionId: () => null, onSubmit });
    textBox().value = 'no session but still worth reporting';
    textBox().dispatchEvent(new Event('input', { bubbles: true }));

    qs(document, '.feedback-send', HTMLButtonElement).click();

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toEqual({
      text: 'no session but still worth reporting',
      channel: 'local',
      metadataOn: true,
      contextOn: false,
      sessionId: null,
    });
  });
});

describe('feedback modal state — disclosure popover', () => {
  test('state: the popover renders exactly the pinned disclosure constant', () => {
    openForm();
    const popover = qs(document, '.feedback-help-popover');
    const rendered = Array.from(popover.querySelectorAll('p')).map((p) => p.textContent);

    expect(rendered).toEqual([...FEEDBACK_DISCLOSURE_PARAGRAPHS]);
    expect(rendered.join('\n\n')).toBe(FEEDBACK_DISCLOSURE);
  });

  test('state: the disclosure states the artifact time-window rule verbatim', () => {
    expect(ARTIFACT_WINDOW_DISCLOSURE).toBe(
      'plus artifacts created on this deployment during the same time window, ' +
        'which may include work from other terminal tabs or the chat panel'
    );
    expect(FEEDBACK_DISCLOSURE).toContain(ARTIFACT_WINDOW_DISCLOSURE);
  });

  test('state: the metadata paragraph claims only what the checkbox governs', () => {
    // `routes/feedback.py`'s `_metadata` writes `submitted` and `user` above
    // the `include_metadata` branch, so identity and the timestamp travel
    // either way. A popover promising the checkbox withholds them would be a
    // false privacy claim, and would contradict the "Always sent" paragraph.
    const metadata = FEEDBACK_DISCLOSURE_PARAGRAPHS[1];
    expect(metadata).toBe(
      'Deployment metadata (on by default): the OSPREY version and the application name.'
    );
    expect(metadata).not.toContain('identity');
    expect(metadata).not.toContain('username');
    expect(metadata).not.toContain('timestamp');
  });

  test('state: the disclosure covers everything FR6 requires', () => {
    for (const phrase of [
      'timestamp',
      'username when the deployment knows it',
      'titles and ids only',
      'truncated',
      'Nothing leaves this page until you click an action button.',
      'recorded on this deployment',
    ]) {
      expect(FEEDBACK_DISCLOSURE).toContain(phrase);
    }
  });

  test('state: the help toggle discloses and hides the popover', () => {
    openForm();
    const toggle = qs(document, '.feedback-help-toggle', HTMLButtonElement);
    const popover = qs(document, '.feedback-help-popover');

    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(popover.hasAttribute('hidden')).toBe(true);

    toggle.click();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(popover.hasAttribute('hidden')).toBe(false);

    toggle.click();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(popover.hasAttribute('hidden')).toBe(true);
  });
});

describe('feedback modal state — action events', () => {
  test('state: Send emits submit with the whole form state', () => {
    const onSubmit = vi.fn();
    openForm({ onSubmit });
    textBox().value = 'terminal freezes on resume';
    textBox().dispatchEvent(new Event('input', { bubbles: true }));
    setChecked(input('.feedback-context-check'), true);

    qs(document, '.feedback-send', HTMLButtonElement).click();

    expect(onSubmit.mock.calls[0][0]).toEqual({
      text: 'terminal freezes on resume',
      channel: 'local',
      metadataOn: true,
      contextOn: true,
      sessionId: 'session-abc',
    });
  });

  test('state: Copy emits copy with the current state', () => {
    const onCopy = vi.fn();
    openForm({ onCopy });
    pickChannel('github');
    setChecked(input('.feedback-context-check'), true);

    copyButton()?.click();

    expect(onCopy).toHaveBeenCalledTimes(1);
    expect(onCopy.mock.calls[0][0].contextOn).toBe(true);
    expect(onCopy.mock.calls[0][0].channel).toBe('github');
  });

  test('state: Open emits open carrying the selected channel', () => {
    const onOpenChannel = vi.fn();
    openForm({ onOpenChannel });
    pickChannel('email');

    qs(document, '.feedback-open-channel', HTMLButtonElement).click();

    expect(onOpenChannel).toHaveBeenCalledTimes(1);
    expect(onOpenChannel.mock.calls[0][0].channel).toBe('email');
  });

  test('state: a disabled Copy emits nothing', () => {
    const onCopy = vi.fn();
    openForm({ onCopy });
    pickChannel('github');

    copyButton()?.click();

    expect(onCopy).not.toHaveBeenCalled();
  });
});

describe('feedback modal shell — drawer inert interaction', () => {
  test('shell: a closed modal leaves nothing for the drawer to inert', () => {
    modal.open();
    modal.close();

    syncBackgroundInert();

    expect(document.querySelector('[inert]')).toBeNull();
  });

  test('shell: a modal reopened after a drawer inert pass carries no inert marks', () => {
    modal.open();
    modal.close();
    syncBackgroundInert();

    modal.open();

    const root = qs(document, OVERLAY);
    expect(root.hasAttribute('inert')).toBe(false);
    expect(root.hasAttribute('aria-hidden')).toBe(false);
  });
});
