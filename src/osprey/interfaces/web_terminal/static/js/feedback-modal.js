/**
 * Centered, `aria-modal` dialog shell for the feedback flow.
 *
 * This module owns the OVERLAY only — scrim, panel, focus, Escape, and the
 * document lifecycle. The dialog's contents (feedback text, channel radios,
 * attachment checkboxes, action row) are rendered by the caller through the
 * `render` hook, so the shell stays independent of what it frames.
 *
 * Three deliberate departures from `palette.js`, whose overlay lifecycle this
 * otherwise follows:
 *
 * 1. **The node is created on open and removed on close, never parked.** The
 *    palette keeps one hidden overlay between opens; a feedback overlay must
 *    not, because `osprey-drawer.js`'s `_syncBackgroundInert()` marks every
 *    top-level `<body>` child `inert` + `aria-hidden` when a drawer opens, and
 *    a parked overlay would be caught by that pass and come back unusable.
 * 2. **Removal is synchronous**, for the same reason: a transition-delayed
 *    removal leaves a parked node in the document for the length of the fade.
 *    The open animation is kept; the close is instant.
 * 3. **A real focus trap.** The palette traps Tab by swallowing it, which is
 *    sound only because its overlay has exactly one focusable node. This
 *    dialog has several, so it installs the shared trap from
 *    `/design-system/js/focus-trap.js`.
 *
 * Escape is handled on a CAPTURE-phase `document` listener that stops
 * propagation, which places it ahead of the drawer's bubble-phase Escape — the
 * modal closes and the drawer beneath it stays open. The one exception is the
 * command palette: it registers its own capture-phase Escape when it opens, so
 * while it is open this handler bails without consuming anything and lets the
 * palette close itself first.
 *
 * @module feedback-modal
 */

import { el } from '/design-system/js/dom.js';
import { focusableElements, installFocusTrap, removeFocusTrap } from '/design-system/js/focus-trap.js';

import { isOpen as isPaletteOpen } from './palette.js';

// ---- Module state ----

const DEFAULT_TITLE = 'Send feedback';

/** Hint shown beside the session-context checkbox when there is no session. */
export const NO_SESSION_HINT = 'no session yet';

/**
 * The artifact-inventory time-window sentence, pinned so the dialog and the
 * server-side composer's block header cannot drift apart. Quoted verbatim from
 * the proposal's functional requirement 6.
 */
export const ARTIFACT_WINDOW_DISCLOSURE =
  'plus artifacts created on this deployment during the same time window, ' +
  'which may include work from other terminal tabs or the chat panel';

/**
 * What the `(?)` popover discloses, one paragraph per entry.
 *
 * Exported so the wording is pinned by a test rather than living only in the
 * markup: it is a privacy disclosure, and a user deciding whether to attach a
 * session to a public issue is entitled to have it describe what is actually
 * attached.
 *
 * The metadata paragraph names **only** what the checkbox governs. Identity and
 * the timestamp travel whether it is ticked or not (`routes/feedback.py`'s
 * `_metadata` puts `submitted` and `user` above the `include_metadata` branch),
 * so listing either of them as something the checkbox adds would tell a sender
 * they can withhold what they cannot — and would contradict the first paragraph
 * two lines above it.
 *
 * @type {readonly string[]}
 */
export const FEEDBACK_DISCLOSURE_PARAGRAPHS = Object.freeze([
  'Always sent: the feedback text you type, a timestamp, and your username ' +
    'when the deployment knows it.',
  'Deployment metadata (on by default): the OSPREY version and the ' +
    'application name.',
  'Session context (off by default): this session’s tool-call and agent ' +
    'event log, its chat history, and the terminal scrollback. It is triaged ' +
    'newest-first and truncated to fit a size budget, so a long session is ' +
    'attached in part rather than in full.',
  'The artifact inventory lists titles and ids only — artifacts tagged to ' +
    `this session, ${ARTIFACT_WINDOW_DISCLOSURE}.`,
  'Nothing leaves this page until you click an action button.',
  'Whichever channel you choose, every submission is also recorded on this ' +
    'deployment.',
]);

/** The disclosure as one string — the form the composer can quote. */
export const FEEDBACK_DISCLOSURE = FEEDBACK_DISCLOSURE_PARAGRAPHS.join('\n\n');

/** Per-channel caveat shown under the channel picker. */
export const CHANNEL_HINTS = Object.freeze({
  local: '',
  github: 'Requires a GitHub account',
  email: 'If nothing opens, use Local instead',
});

/** Label of the channel's "open the outside thing" button. */
const CHANNEL_ACTION_LABELS = Object.freeze({
  local: '',
  github: 'Open GitHub issue',
  email: 'Open email draft',
});

/** Radio labels, in render order. */
const CHANNELS = Object.freeze([
  { value: 'local', label: 'Local' },
  { value: 'github', label: 'GitHub' },
  { value: 'email', label: 'Email' },
]);

// Every currently open modal. A Set rather than a boolean so `open()`/`close()`
// stay idempotent per instance and `isFeedbackModalOpen()` cannot be desynced
// by a second instance.
/** @type {Set<FeedbackModal>} */
const _openModals = new Set();

/**
 * @typedef {'local'|'github'|'email'} FeedbackChannel
 */

/**
 * @typedef {object} FeedbackFormState
 * @property {string} text - the typed feedback.
 * @property {FeedbackChannel} channel - the selected delivery channel.
 * @property {boolean} metadataOn - attach deployment metadata (default true).
 * @property {boolean} contextOn - attach session context (default false;
 *   reported false whenever there is no session to attach).
 * @property {string|null} sessionId - the session the context would come from.
 */

/**
 * @typedef {object} FeedbackModalOptions
 * @property {string} [title] - heading text and the dialog's accessible name.
 * @property {(body: HTMLElement) => void} [render] - REPLACES the built-in
 *   feedback form; called on every open, before the dialog is attached and
 *   focused. Only for callers that want a different body (and for tests).
 * @property {() => string|null} [getSessionId] - the current terminal session
 *   id. Injected rather than imported from `terminal.js` so the dialog has no
 *   dependency on the terminal module. Defaults to "no session".
 * @property {(state: FeedbackFormState) => void} [onSubmit] - the `submit`
 *   event: the Local channel's Send button.
 * @property {(state: FeedbackFormState) => void} [onCopy] - the `copy` event:
 *   the standalone "Copy session context" button.
 * @property {(state: FeedbackFormState) => void} [onOpenChannel] - the `open`
 *   event: "Open GitHub issue" / "Open email draft".
 * @property {() => void} [onClose] - called once per close, after teardown.
 */

// ---- Public API ----

/**
 * Whether a feedback modal is currently open.
 *
 * Exported for keyboard arbitration: `palette-boot.js` owns Cmd/Ctrl+K on a
 * capture-phase listener registered at boot, so it always runs before a
 * listener this module adds at open time and cannot be pre-empted by ordering.
 * It has to consult this instead.
 *
 * @returns {boolean}
 */
export function isFeedbackModalOpen() {
  return _openModals.size > 0;
}

/**
 * A body-appended modal dialog with a scrim, a focus trap and an Escape close.
 *
 * One instance can be opened and closed repeatedly; each open builds a fresh
 * node tree and each close removes it from the document.
 */
export class FeedbackModal {
  /**
   * @param {FeedbackModalOptions} [options]
   */
  constructor(options = {}) {
    /** @type {FeedbackModalOptions} */
    this._options = options;
    this._opened = false;
    /** @type {HTMLElement|null} */
    this._overlayEl = null;
    /** @type {HTMLElement|null} */
    this._dialogEl = null;
    /** @type {HTMLElement|null} */
    this._bodyEl = null;
    /** @type {HTMLElement|null} */
    this._previousFocus = null;
    /** @param {KeyboardEvent} event */
    this._onKeydown = (event) => this._handleKeydown(event);

    // The draft. It lives on the instance, not in the DOM, so it survives both
    // a channel switch (which re-renders the action row) and a close/reopen —
    // an accidental Escape must not throw away a half-written report.
    this._text = '';
    /** @type {FeedbackChannel} */
    this._channel = 'local';
    this._metadataOn = true;
    this._contextOn = false;

    /** @type {HTMLTextAreaElement|null} */
    this._textEl = null;
    /** @type {HTMLInputElement|null} */
    this._contextCheckEl = null;
    /** @type {HTMLElement|null} */
    this._contextHintEl = null;
    /** @type {HTMLElement|null} */
    this._channelHintEl = null;
    /** @type {HTMLElement|null} */
    this._actionsEl = null;
  }

  /**
   * The current form state — the payload every action event carries.
   *
   * `contextOn` is reported false whenever there is no session, so a caller
   * can never be handed "attach the context" together with a null session id.
   *
   * @returns {FeedbackFormState}
   */
  formState() {
    const sessionId = this._sessionId();
    return {
      text: this._text,
      channel: this._channel,
      metadataOn: this._metadataOn,
      contextOn: this._contextOn && sessionId !== null,
      sessionId,
    };
  }

  /**
   * Whether this modal is currently open.
   *
   * @returns {boolean}
   */
  isOpen() {
    return this._opened;
  }

  /**
   * The dialog's content slot — the element the `render` hook populates.
   *
   * Null while the modal is closed: the node tree does not outlive an open.
   *
   * @returns {HTMLElement|null}
   */
  get bodyElement() {
    return this._bodyEl;
  }

  /**
   * The `role="dialog"` panel itself, or null while closed.
   *
   * @returns {HTMLElement|null}
   */
  get dialogElement() {
    return this._dialogEl;
  }

  /**
   * Build the overlay, attach it to `<body>` and move focus inside.
   *
   * Idempotent — reopening an open modal only re-focuses it.
   *
   * @returns {void}
   */
  open() {
    if (this._opened) {
      this._moveFocusIn();
      return;
    }
    this._previousFocus =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    this._opened = true;
    _openModals.add(this);

    const overlay = this._build();
    document.body.appendChild(overlay);
    installFocusTrap(this._dialogEl ?? overlay);
    document.addEventListener('keydown', this._onKeydown, true);

    // The reveal and the focus have to land in the same frame: focusing an
    // element inside a `visibility: hidden` subtree is a spec'd no-op, and the
    // stylesheet keeps the overlay hidden until `.visible` is on it. The
    // staleness guard covers an open/close pair inside one frame.
    requestAnimationFrame(() => {
      if (!this._opened) return;
      overlay.classList.add('visible');
      this._moveFocusIn();
    });
  }

  /**
   * Remove the overlay from the document and restore focus.
   *
   * Idempotent — a no-op when already closed, so every dismissal path (close
   * button, scrim, Escape, caller) can call it unconditionally.
   *
   * @returns {void}
   */
  close() {
    if (!this._opened) return;
    this._opened = false;
    _openModals.delete(this);
    document.removeEventListener('keydown', this._onKeydown, true);

    if (this._dialogEl) removeFocusTrap(this._dialogEl);
    if (this._overlayEl) this._overlayEl.remove();
    this._overlayEl = null;
    this._dialogEl = null;
    this._bodyEl = null;
    this._textEl = null;
    this._contextCheckEl = null;
    this._contextHintEl = null;
    this._channelHintEl = null;
    this._actionsEl = null;

    this._restoreFocus();
    if (this._options.onClose) this._options.onClose();
  }

  // ---- Internals ----

  /**
   * Build the scrim + dialog tree for one open.
   *
   * @returns {HTMLElement} the scrim, with the dialog already inside it
   */
  _build() {
    const title = this._options.title ?? DEFAULT_TITLE;

    const overlay = el('div', 'feedback-modal-overlay');
    overlay.addEventListener('mousedown', (event) => {
      // Only a press on the scrim itself dismisses; a press that started
      // inside the dialog (a drag off a text selection, say) must not.
      if (event.target === overlay) this.close();
    });

    const dialog = el('div', 'feedback-modal-dialog');
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('aria-label', title);
    dialog.tabIndex = -1;

    const header = el('div', 'feedback-modal-header');
    const heading = el('h2', 'feedback-modal-title');
    heading.textContent = title;
    header.appendChild(heading);

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'feedback-modal-close';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', () => this.close());
    header.appendChild(closeBtn);

    const body = el('div', 'feedback-modal-body');
    if (this._options.render) this._options.render(body);
    else this._buildForm(body);

    dialog.appendChild(header);
    dialog.appendChild(body);
    overlay.appendChild(dialog);

    this._overlayEl = overlay;
    this._dialogEl = dialog;
    this._bodyEl = body;
    return overlay;
  }

  /**
   * Build the feedback form into the body slot and wire it to the draft.
   *
   * Every control is rebuilt from `this._*` draft fields on each open, so the
   * DOM is a view of the state and never its owner.
   *
   * @param {HTMLElement} body
   * @returns {void}
   */
  _buildForm(body) {
    const textLabel = el('label', 'feedback-field-label');
    textLabel.textContent = 'What happened?';
    const text = document.createElement('textarea');
    text.className = 'feedback-text';
    text.rows = 6;
    text.value = this._text;
    text.placeholder = 'What you expected, what happened instead, and how to reach it.';
    text.addEventListener('input', () => {
      this._text = text.value;
    });
    textLabel.appendChild(text);
    this._textEl = text;
    body.appendChild(textLabel);

    body.appendChild(this._buildChannels());
    body.appendChild(this._buildAttachments());

    const actions = el('div', 'feedback-actions');
    this._actionsEl = actions;
    body.appendChild(actions);

    this._syncControls();
  }

  /**
   * The delivery-channel radio group and its per-channel hint line.
   *
   * @returns {HTMLElement}
   */
  _buildChannels() {
    const group = el('div', 'feedback-channels');
    group.setAttribute('role', 'radiogroup');
    group.setAttribute('aria-label', 'Where to send this');

    for (const { value, label } of CHANNELS) {
      const wrap = el('label', 'feedback-channel-option');
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'feedback-channel';
      radio.value = value;
      radio.checked = this._channel === value;
      radio.addEventListener('change', () => {
        if (!radio.checked) return;
        this._channel = /** @type {FeedbackChannel} */ (value);
        this._syncControls();
      });
      const caption = el('span', 'feedback-channel-label');
      caption.textContent = label;
      wrap.appendChild(radio);
      wrap.appendChild(caption);
      group.appendChild(wrap);
    }

    const hint = el('p', 'feedback-channel-hint');
    this._channelHintEl = hint;

    const section = el('div', 'feedback-channel-section');
    section.appendChild(group);
    section.appendChild(hint);
    return section;
  }

  /**
   * The two attachment checkboxes plus the `(?)` disclosure popover.
   *
   * @returns {HTMLElement}
   */
  _buildAttachments() {
    const section = el('div', 'feedback-attachments');

    const metadata = this._checkbox(
      'feedback-metadata-check',
      'Include deployment metadata',
      this._metadataOn,
      (checked) => {
        this._metadataOn = checked;
      }
    );
    section.appendChild(metadata.row);

    const context = this._checkbox(
      'feedback-context-check',
      'Include session context',
      this._contextOn,
      (checked) => {
        this._contextOn = checked;
        this._syncControls();
      }
    );
    this._contextCheckEl = context.input;
    const contextHint = el('span', 'feedback-context-hint');
    this._contextHintEl = contextHint;
    context.row.appendChild(contextHint);

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'feedback-help-toggle';
    toggle.textContent = '(?)';
    toggle.setAttribute('aria-label', 'What gets sent');
    toggle.setAttribute('aria-expanded', 'false');
    context.row.appendChild(toggle);
    section.appendChild(context.row);

    const popover = el('div', 'feedback-help-popover');
    popover.setAttribute('role', 'note');
    popover.toggleAttribute('hidden', true);
    for (const paragraph of FEEDBACK_DISCLOSURE_PARAGRAPHS) {
      const p = document.createElement('p');
      p.textContent = paragraph;
      popover.appendChild(p);
    }
    toggle.addEventListener('click', () => {
      const open = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', open ? 'false' : 'true');
      popover.toggleAttribute('hidden', open);
    });
    section.appendChild(popover);

    return section;
  }

  /**
   * One labelled checkbox row.
   *
   * @param {string} className
   * @param {string} label
   * @param {boolean} checked
   * @param {(checked: boolean) => void} onChange
   * @returns {{row: HTMLElement, input: HTMLInputElement}}
   */
  _checkbox(className, label, checked, onChange) {
    const row = el('label', 'feedback-check-row');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.className = className;
    input.checked = checked;
    input.addEventListener('change', () => onChange(input.checked));
    const caption = el('span', 'feedback-check-label');
    caption.textContent = label;
    row.appendChild(input);
    row.appendChild(caption);
    return { row, input };
  }

  /**
   * Re-derive everything that depends on the channel or on session
   * availability: the channel hint, the context checkbox, the action row.
   *
   * @returns {void}
   */
  _syncControls() {
    const sessionId = this._sessionId();

    if (this._channelHintEl) this._channelHintEl.textContent = CHANNEL_HINTS[this._channel];

    if (this._contextCheckEl) {
      this._contextCheckEl.disabled = sessionId === null;
      this._contextCheckEl.checked = this._contextOn && sessionId !== null;
    }
    if (this._contextHintEl) {
      this._contextHintEl.textContent = sessionId === null ? NO_SESSION_HINT : '';
    }

    this._renderActions(sessionId);
  }

  /**
   * Rebuild the channel-dependent action row.
   *
   * "Copy session context" is always rendered for the outbound channels rather
   * than appearing and disappearing: a control that vanishes when unticked
   * reads as a bug, while a greyed one explains itself.
   *
   * @param {string|null} sessionId
   * @returns {void}
   */
  _renderActions(sessionId) {
    const row = this._actionsEl;
    if (!row) return;
    row.replaceChildren();

    if (this._channel === 'local') {
      row.appendChild(
        this._actionButton('feedback-send primary', 'Send', () => this._emit(this._options.onSubmit))
      );
      return;
    }

    const copy = this._actionButton('feedback-copy-context', 'Copy session context', () =>
      this._emit(this._options.onCopy)
    );
    copy.disabled = !(this._contextOn && sessionId !== null);
    row.appendChild(copy);

    row.appendChild(
      this._actionButton('feedback-open-channel primary', CHANNEL_ACTION_LABELS[this._channel], () =>
        this._emit(this._options.onOpenChannel)
      )
    );
  }

  /**
   * One action-row button. The handler re-checks `disabled` because a
   * programmatic `click()` is not required to honor it.
   *
   * @param {string} className
   * @param {string} label
   * @param {() => void} onClick
   * @returns {HTMLButtonElement}
   */
  _actionButton(className, label, onClick) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `feedback-action ${className}`;
    button.textContent = label;
    button.addEventListener('click', () => {
      if (button.disabled) return;
      onClick();
    });
    return button;
  }

  /**
   * Hand the current form state to one of the action callbacks.
   *
   * @param {((state: FeedbackFormState) => void)|undefined} handler
   * @returns {void}
   */
  _emit(handler) {
    if (handler) handler(this.formState());
  }

  /**
   * The current session id, normalized to null.
   *
   * @returns {string|null}
   */
  _sessionId() {
    const getter = this._options.getSessionId;
    if (!getter) return null;
    return getter() ?? null;
  }

  /**
   * Capture-phase Escape handler. Owns nothing else.
   *
   * @param {KeyboardEvent} event
   * @returns {void}
   */
  _handleKeydown(event) {
    if (!this._opened) return;
    // The palette layers above this dialog and runs its own capture-phase
    // Escape. While it is open the key belongs to it, so consume nothing —
    // not even the default.
    if (isPaletteOpen()) return;
    if (event.key !== 'Escape') return;
    event.preventDefault();
    event.stopPropagation();
    this.close();
  }

  /**
   * Focus the first control in the content slot, falling back to the header's
   * close button and then to the dialog itself.
   *
   * The body is preferred over plain document order because the close button
   * comes first in the markup: opening a feedback dialog with the dismiss
   * control focused reads as "are you sure you want to leave", which is the
   * opposite of the intent.
   *
   * @returns {void}
   */
  _moveFocusIn() {
    const dialog = this._dialogEl;
    if (!dialog) return;
    const body = this._bodyEl;
    const [first] = body ? focusableElements(body) : [];
    const [fallback] = focusableElements(dialog);
    const target = first ?? fallback;
    if (target) target.focus();
    else dialog.focus();
  }

  /**
   * Return focus to whatever had it before the modal opened, if that element
   * is still in the document.
   *
   * @returns {void}
   */
  _restoreFocus() {
    const previous = this._previousFocus;
    this._previousFocus = null;
    if (previous && document.contains(previous)) previous.focus();
  }
}
