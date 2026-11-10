// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: hold-to-talk voice input.
 *
 * A mic button beside Send, and a keyboard shortcut for the same thing. Hold
 * Ctrl + Space anywhere on the Simple view (or hold the button: mouse, touch,
 * or Space/Enter while it has focus) and speak; release, and the clip goes to
 * the server's speech-to-text
 * route (routes/speech.py, a locally run Whisper model). The text it returns
 * is inserted into the input box at the cursor for the operator to read, fix
 * and send. Nothing is ever sent to the agent from here.
 *
 * Push-to-talk only: the microphone is opened when the button goes down and
 * released when it comes up, never left listening. Esc cancels a recording;
 * one runs at most a minute. The button is shown only when the server says
 * voice input can work there (GET /api/speech/status).
 *
 * Listen mode (the LISTEN button beside the mic, chat-voice-listen.js) is
 * for a push-to-talk microphone whose own button only lets sound through
 * while held: with Listen on, speech starts a clip and the quiet after it
 * ends one, so the mic's button alone does the talking. It turns itself off
 * after five minutes without speech, or when the page is hidden.
 *
 * Browsers allow the microphone only on secure pages (https, or localhost),
 * and ask the operator's permission the first time.
 *
 * @module chat-voice
 */

import { withPrefix } from './api.js';
import { isTerminalSurface } from './agent-surface.js';
import { postChatNotice } from './chat-events.js';
import { createListener } from './chat-voice-listen.js';

const STATUS_ENDPOINT = '/api/speech/status';
const TRANSCRIBE_ENDPOINT = '/api/speech/transcribe';

/** The longest recording, after which it stops by itself. */
export const MAX_RECORDING_MS = 60_000;
/** A press shorter than this is taken as a tap, not speech, and dropped. */
export const MIN_RECORDING_MS = 300;
/** Listen mode turns itself off after this long without speech. */
export const LISTEN_IDLE_MS = 5 * 60_000;

const IDLE_LABEL = '🎤';
const IDLE_PLACEHOLDER = 'Message the agent…';
const LISTEN_PLACEHOLDER = 'Listening: hold your mic’s button and speak';

/**
 * The hold-to-talk keyboard shortcut. A combination rather than a lone key, so
 * it never types into the input box, needs no Fn key on a laptop, and starts
 * at once (no hold detection): Ctrl + Space.
 * @typedef {{ key: string, ctrl: boolean, label: string, aria: string }} Shortcut
 * @type {Shortcut}
 */
export const DEFAULT_SHORTCUT = Object.freeze({
  key: ' ',
  ctrl: true,
  label: 'Ctrl + Space',
  aria: 'Control+Space',
});

/**
 * Whether a key event is the shortcut: its key, with Ctrl (and nothing else
 * that would make it a different shortcut) held.
 * @param {KeyboardEvent} e
 * @param {Shortcut} shortcut
 * @returns {boolean}
 */
export function matchesShortcut(e, shortcut) {
  return (
    e.key === shortcut.key && e.ctrlKey === shortcut.ctrl && !e.altKey && !e.metaKey && !e.shiftKey
  );
}

/**
 * Insert *text* into *textarea* at its cursor, with a space either side where
 * it would otherwise run into a word, and leave the cursor after it.
 * @param {HTMLTextAreaElement} textarea
 * @param {string} text
 */
export function insertAtCursor(textarea, text) {
  const { value } = textarea;
  const start = textarea.selectionStart ?? value.length;
  const end = textarea.selectionEnd ?? value.length;
  const before = value.slice(0, start);
  const after = value.slice(end);
  const lead = before && !/\s$/.test(before) ? ' ' : '';
  const trail = after && !/^\s/.test(after) ? ' ' : '';
  const insert = `${lead}${text}${trail}`;
  textarea.value = `${before}${insert}${after}`;
  const cursor = before.length + insert.length;
  textarea.setSelectionRange(cursor, cursor);
  // chat.js resizes the box on input, as it would for typing.
  textarea.dispatchEvent(new Event('input', { bubbles: true }));
}

/**
 * The recording format to ask the browser for: Opus in WebM where supported
 * (Chrome, Edge, Firefox), else whatever the browser records by default.
 * @param {any} Recorder - the MediaRecorder constructor
 * @returns {string}
 */
export function pickMimeType(Recorder) {
  for (const type of ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']) {
    if (typeof Recorder?.isTypeSupported === 'function' && Recorder.isTypeSupported(type)) {
      return type;
    }
  }
  return '';
}

/**
 * @typedef {object} VoiceInputOptions
 * @property {HTMLTextAreaElement} textarea - where transcripts go
 * @property {(message: string) => void} [onError] - shows a problem to the operator
 * @property {(constraints?: MediaTrackConstraints) => Promise<MediaStream>} [getMicrophone]
 *   - opens the microphone
 * @property {any} [Recorder] - the MediaRecorder constructor
 * @property {any} [AudioContextImpl] - the AudioContext constructor (Listen mode)
 * @property {typeof fetch} [fetchImpl]
 * @property {Shortcut} [shortcut] - the hold-to-talk key combination
 * @property {() => boolean} [isActive] - whether the chat is showing, so the
 *   shortcut acts (it is ignored while another view is in front)
 */

/**
 * @typedef {object} VoiceInput
 * @property {HTMLButtonElement} button - the mic button, hidden until available
 * @property {HTMLButtonElement} listenButton - the Listen mode toggle, hidden until available
 * @property {() => Promise<boolean>} checkAvailability - ask the server; shows
 *   the button when voice input can work here
 * @property {(disabled: boolean) => void} setDisabled - follow the console's
 *   blocked state (a turn streaming, a hand-off in progress)
 * @property {() => void} cancel - stop and discard any recording
 * @property {() => void} dispose - stop listening for the shortcut and Esc
 *   (the page-wide listeners); for tearing the console down
 */

/**
 * Create the hold-to-talk mic button for the chat's input box.
 * @param {VoiceInputOptions} options
 * @returns {VoiceInput}
 */
export function createVoiceInput(options) {
  const { textarea } = options;
  const onError = options.onError ?? (() => {});
  const getMicrophone =
    options.getMicrophone ??
    ((constraints) => navigator.mediaDevices.getUserMedia({ audio: constraints ?? true }));
  const Recorder = options.Recorder ?? globalThis.MediaRecorder;
  const AudioContextImpl = options.AudioContextImpl ?? globalThis.AudioContext;
  const fetchImpl = options.fetchImpl ?? ((...args) => fetch(...args));
  const shortcut = options.shortcut ?? DEFAULT_SHORTCUT;
  const isActive = options.isActive ?? (() => true);

  const button = /** @type {HTMLButtonElement} */ (document.createElement('button'));
  button.type = 'button';
  button.className = 'op-mic-btn';
  button.textContent = IDLE_LABEL;
  button.title = `Hold to talk (${shortcut.label})`;
  button.setAttribute('aria-label', 'Hold to talk');
  button.setAttribute('aria-keyshortcuts', shortcut.aria);
  button.hidden = true;

  const listenButton = /** @type {HTMLButtonElement} */ (document.createElement('button'));
  listenButton.type = 'button';
  listenButton.className = 'op-listen-btn';
  listenButton.title = 'Listen mode: record whenever your push-to-talk mic lets sound through';
  listenButton.hidden = true;

  /** The input box's resting hint: the shortcut is named once voice is available. */
  // The box's own hint, kept: voice only adds how to talk to it.
  const basePlaceholder = textarea.placeholder || IDLE_PLACEHOLDER;
  let idlePlaceholder = basePlaceholder;

  /** @type {'idle' | 'starting' | 'recording' | 'transcribing'} */
  let state = 'idle';
  let held = false;
  /** Whether the recording was started from the keyboard shortcut. */
  let heldByKey = false;
  let cancelled = false;
  let startedAt = 0;
  /** @type {MediaStream | null} */
  let stream = null;
  /** @type {any} */
  let recorder = null;
  /** @type {Blob[]} */
  let chunks = [];
  /** @type {ReturnType<typeof setTimeout> | null} */
  let limit = null;

  /**
   * @param {typeof state} next
   * @param {string} [placeholder]
   */
  function setState(next, placeholder = idlePlaceholder) {
    state = next;
    button.dataset.state = next;
    button.setAttribute('aria-pressed', String(next === 'recording'));
    button.textContent =
      next === 'recording' ? '● REC' : next === 'transcribing' ? '…' : IDLE_LABEL;
    textarea.placeholder = placeholder;
  }

  function releaseMicrophone() {
    if (limit !== null) clearTimeout(limit);
    limit = null;
    for (const track of stream?.getTracks() ?? []) track.stop();
    stream = null;
  }

  async function start() {
    if (state !== 'idle' || button.disabled || listenState !== 'off') return;
    held = true;
    cancelled = false;
    setState('starting', 'Starting the microphone…');
    try {
      stream = await getMicrophone();
    } catch {
      setState('idle');
      onError('The microphone could not be opened. Check that this page may use it.');
      return;
    }
    if (!held || cancelled) {
      // Released (or cancelled) while permission was being granted: nothing said.
      releaseMicrophone();
      setState('idle');
      return;
    }
    const mimeType = pickMimeType(Recorder);
    recorder = new Recorder(stream, mimeType ? { mimeType } : undefined);
    chunks = [];
    recorder.addEventListener('dataavailable', (/** @type {any} */ e) => {
      if (e.data && e.data.size) chunks.push(e.data);
    });
    recorder.addEventListener('stop', () => void finish());
    recorder.start();
    startedAt = Date.now();
    setState('recording', 'Listening… release to finish, Esc to cancel');
    limit = setTimeout(() => stop(), MAX_RECORDING_MS);
  }

  function stop() {
    held = false;
    heldByKey = false;
    if (state === 'recording' && recorder) recorder.stop();
    // Released while the microphone was still opening: start() sees `held`
    // gone and records nothing.
  }

  function cancel() {
    cancelled = true;
    held = false;
    if (state === 'recording' && recorder) recorder.stop();
    else if (state === 'starting') releaseMicrophone();
  }

  async function finish() {
    releaseMicrophone();
    const tooShort = Date.now() - startedAt < MIN_RECORDING_MS;
    if (cancelled || tooShort || chunks.length === 0) {
      setState('idle');
      if (tooShort && !cancelled)
        onError(`Hold ${shortcut.label} or the mic button while you speak.`);
      return;
    }
    const clip = new Blob(chunks, {
      type: recorder?.mimeType || chunks[0].type || 'audio/webm',
    });
    chunks = [];
    setState('transcribing', 'Transcribing…');
    try {
      await deliver(clip);
    } finally {
      setState('idle');
    }
  }

  /**
   * Send a clip to be transcribed and put the text in the input box, or say
   * why there is none.
   * @param {Blob} clip
   */
  async function deliver(clip) {
    try {
      const res = await fetchImpl(withPrefix(TRANSCRIBE_ENDPOINT), {
        method: 'POST',
        headers: { 'Content-Type': clip.type.split(';')[0] || 'audio/webm' },
        body: clip,
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || 'The recording could not be transcribed.');
      const text = String(body.text ?? '').trim();
      if (text) {
        insertAtCursor(textarea, text);
        textarea.focus();
      } else {
        onError('No speech was recognised. Try again, closer to the microphone.');
      }
    } catch (err) {
      onError(/** @type {Error} */ (err).message || 'The recording could not be transcribed.');
    }
  }

  // ---- Listen mode ---- //

  /** @type {'off' | 'starting' | 'listening' | 'hearing'} */
  let listenState = 'off';
  /** @type {import('./chat-voice-listen.js').Listener | null} */
  let listener = null;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let idleTimer = null;
  /** Clips are transcribed one at a time, in the order spoken. */
  let queue = Promise.resolve();
  let pendingClips = 0;

  /** @param {typeof listenState} next */
  function setListenState(next) {
    listenState = next;
    listenButton.dataset.state = next;
    listenButton.setAttribute('aria-pressed', String(next !== 'off'));
    listenButton.textContent =
      next === 'hearing' ? '● REC' : next === 'off' ? 'LISTEN' : '● LISTENING';
    if (pendingClips === 0)
      textarea.placeholder = next === 'off' ? idlePlaceholder : LISTEN_PLACEHOLDER;
  }

  function armIdleTimer() {
    if (idleTimer !== null) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      stopListening();
      onError('Listen mode turned off after five minutes without speech.');
    }, LISTEN_IDLE_MS);
  }

  function stopListening() {
    if (idleTimer !== null) clearTimeout(idleTimer);
    idleTimer = null;
    listener?.stop();
    listener = null;
    if (listenState !== 'off') setListenState('off');
  }

  /** @param {Blob} clip */
  function onListenClip(clip) {
    if (listenState === 'off') return;
    setListenState('listening');
    if (!isActive()) {
      stopListening(); // the chat is no longer the view in front
      return;
    }
    if (button.disabled) {
      onError('The message box is locked for a moment; that was not transcribed.');
      return;
    }
    pendingClips += 1;
    textarea.placeholder = 'Transcribing…';
    queue = queue
      .then(() => deliver(clip))
      .finally(() => {
        pendingClips -= 1;
        if (pendingClips === 0) setListenState(listenState);
      });
  }

  async function startListening() {
    if (listenState !== 'off') return;
    cancel();
    setListenState('starting');
    try {
      const opened = await createListener({
        getMicrophone,
        AudioContextImpl,
        onStart: () => {
          armIdleTimer();
          setListenState('hearing');
        },
        onClip: onListenClip,
        onDrop: () => {
          if (listenState === 'hearing') setListenState('listening');
        },
      });
      // Read through a cast: TypeScript cannot see listenState change across the await.
      if (/** @type {string} */ (listenState) !== 'starting') {
        opened.stop(); // turned off while the microphone was opening
        return;
      }
      listener = opened;
      setListenState('listening');
      armIdleTimer();
    } catch {
      setListenState('off');
      onError('The microphone could not be opened. Check that this page may use it.');
    }
  }

  listenButton.addEventListener('click', () => {
    if (listenState === 'off') void startListening();
    else stopListening();
  });
  // A hidden page (another tab, a minimised window) stops listening.
  function onVisibility() {
    if (document.hidden) stopListening();
  }
  document.addEventListener('visibilitychange', onVisibility);

  // Hold to talk: pointer down starts, up (or leaving the button) stops.
  button.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    void start();
  });
  for (const type of ['pointerup', 'pointerleave', 'pointercancel']) {
    button.addEventListener(type, () => stop());
  }
  // The same from the keyboard while the button has focus.
  button.addEventListener('keydown', (e) => {
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
      e.preventDefault();
      void start();
    }
  });
  button.addEventListener('keyup', (e) => {
    if (e.key === ' ' || e.key === 'Enter') stop();
  });
  /** @param {KeyboardEvent} e */
  function onKeyDown(e) {
    if (e.key === 'Escape' && (state === 'recording' || state === 'starting')) {
      cancel();
      return;
    }
    // Hold-to-talk from anywhere on the chat: the shortcut down starts,
    // letting go of either of its keys stops. It never types; repeats are
    // ignored.
    if (!matchesShortcut(e, shortcut)) return;
    if (button.hidden || button.disabled || !isActive()) return;
    e.preventDefault();
    if (e.repeat || state !== 'idle') return;
    heldByKey = true;
    void start();
  }
  /** @param {KeyboardEvent} e */
  function onKeyUp(e) {
    if (heldByKey && (e.key === shortcut.key || e.key === 'Control')) stop();
  }
  // Switching window mid-hold loses the key-up: stop rather than record on.
  function onBlur() {
    if (heldByKey) stop();
  }
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);

  setState('idle');
  setListenState('off');

  return {
    button,
    listenButton,
    async checkAvailability() {
      try {
        const res = await fetchImpl(withPrefix(STATUS_ENDPOINT), {
          cache: 'no-store',
        });
        const body = res.ok ? await res.json() : {};
        const available = Boolean(body.available) && Boolean(Recorder);
        button.hidden = !available;
        listenButton.hidden = !(Boolean(body.available) && Boolean(AudioContextImpl));
        idlePlaceholder = available
          ? `${basePlaceholder} (hold ${shortcut.label} to talk)`
          : basePlaceholder;
        if (state === 'idle' && listenState === 'off') textarea.placeholder = idlePlaceholder;
        return available;
      } catch {
        button.hidden = true;
        listenButton.hidden = true;
        return false;
      }
    },
    setDisabled(disabled) {
      button.disabled = disabled;
      if (disabled) cancel();
    },
    cancel,
    dispose() {
      cancel();
      stopListening();
      document.removeEventListener('visibilitychange', onVisibility);
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    },
  };
}

/**
 * Start voice input as one of the chat's features (chat-features.js): the
 * Listen toggle and the mic sit before Send, problems are told in the chat's
 * log, the shortcut acts only while the chat is the surface on screen, and
 * voice pauses whenever the message box is locked (a hand-off), as typing does.
 * @param {{ textarea: HTMLTextAreaElement, sendBtn: HTMLButtonElement, container: HTMLElement }} ctx
 * @returns {VoiceInput}
 */
export function initChatVoice(ctx) {
  const { textarea, sendBtn, container } = ctx;
  const voice = createVoiceInput({
    textarea,
    onError: (message) => void postChatNotice(message),
    isActive: () => !isTerminalSurface() && container.getClientRects().length > 0,
  });
  sendBtn.parentElement?.prepend(voice.listenButton, voice.button);
  const follow = () => voice.setDisabled(textarea.disabled);
  new MutationObserver(follow).observe(textarea, {
    attributes: true,
    attributeFilter: ['disabled'],
  });
  follow();
  void voice.checkAvailability();
  return voice;
}
