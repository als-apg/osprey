// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: voice input.
 *
 * One mic icon beside Send. How the operator talks through it is the
 * deployment's choice, `web.voice.inputs` in its config (sent with
 * GET /api/speech/status): a list of the ways of talking that are switched on,
 * each acting only on its own input. An empty list is no voice input, and the
 * icon is hidden.
 *
 *   button  (the default) click and hold the icon (or Space/Enter while it has
 *           focus), speak, release.
 *   key     hold the keyboard key `web.voice.key` (default Ctrl + Space)
 *           anywhere on the Simple view, speak, release. Ignored while `mic`
 *           is listening.
 *   mic     click the icon to turn listening on or off, for a push-to-talk
 *           microphone whose own button only lets sound through while held:
 *           speech starts a clip and the quiet after it ends one
 *           (chat-voice-listen.js). Listening turns itself off after five
 *           minutes without speech, or when the page is hidden. `button` and
 *           `mic` both use the icon, so with both listed `mic` wins.
 *
 * The icon is red while the microphone is live: pulsing while it records,
 * steady while `mic` listens between clips. Each clip goes to the
 * server's speech-to-text route (routes/speech.py, a locally run Whisper
 * model) and the text is inserted into the input box at the cursor for the
 * operator to read, fix and send. Nothing is ever sent to the agent from here.
 *
 * With `button` and `key` the microphone is opened when the hold starts and
 * released when it ends; Esc cancels a recording, and one runs at most a
 * minute. The icon is shown only when the server says voice input can work
 * there. Browsers allow the microphone only on secure pages (https, or
 * localhost), and ask the operator's permission the first time.
 *
 * @module chat-voice
 */

import { withPrefix } from './api.js';
import { isTerminalSurface } from './agent-surface.js';
import { postChatNotice } from './chat-events.js';
import { createListener } from './chat-voice-listen.js';
import {
  DEFAULT_SHORTCUT,
  parseShortcut,
  matchesShortcut,
  releasesShortcut,
} from './chat-voice-shortcut.js';

// The talk-key helpers are part of this module's interface.
export { DEFAULT_SHORTCUT, parseShortcut, matchesShortcut };

const STATUS_ENDPOINT = '/api/speech/status';
const TRANSCRIBE_ENDPOINT = '/api/speech/transcribe';

/** The longest recording, after which it stops by itself. */
export const MAX_RECORDING_MS = 60_000;
/** A press shorter than this is taken as a tap, not speech, and dropped. */
export const MIN_RECORDING_MS = 300;
/** Listening (`mic`) turns itself off after this long without speech. */
export const LISTEN_IDLE_MS = 5 * 60_000;

/** The ways of talking `web.voice.inputs` may list; see the module comment. */
export const VOICE_INPUTS = Object.freeze(['button', 'key', 'mic']);
/** @typedef {'button' | 'key' | 'mic'} VoiceInputKind */
/** What a status without `inputs` (or with nothing usable in it) means. */
export const DEFAULT_INPUTS = Object.freeze(/** @type {VoiceInputKind[]} */ (['button']));

/**
 * The inputs to switch on, from `web.voice.inputs`: known entries only, each
 * once. `button` gives way to `mic`, since both use the icon. Not a list, or
 * a list with nothing known in it, is the default; an empty list is off.
 * @param {unknown} raw
 * @returns {VoiceInputKind[]}
 */
export function normalizeInputs(raw) {
  if (!Array.isArray(raw)) return [...DEFAULT_INPUTS];
  if (raw.length === 0) return [];
  const known = /** @type {VoiceInputKind[]} */ (
    VOICE_INPUTS.filter((kind) => raw.some((entry) => String(entry).trim().toLowerCase() === kind))
  );
  if (known.length === 0) return [...DEFAULT_INPUTS];
  return known.includes('mic') ? known.filter((kind) => kind !== 'button') : known;
}

const IDLE_PLACEHOLDER = 'Message the agent…';
const LISTEN_PLACEHOLDER = 'Listening: hold your mic’s button and speak';

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
 * The mic icon: Lucide's `mic` (lucide-static, ISC licence; see
 * LUCIDE-LICENSE.txt beside this file), drawn in currentColor so it turns red
 * with the button.
 * @returns {SVGSVGElement}
 */
function micIcon() {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  for (const [name, value] of Object.entries({
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    'stroke-width': '2',
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    'aria-hidden': 'true',
    class: 'op-icon',
  })) {
    svg.setAttribute(name, value);
  }
  for (const d of ['M12 19v3', 'M19 10v2a7 7 0 0 1-14 0v-2']) {
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', d);
    svg.append(path);
  }
  const body = document.createElementNS(NS, 'rect');
  for (const [name, value] of Object.entries({
    x: '9',
    y: '2',
    width: '6',
    height: '13',
    rx: '3',
  })) {
    body.setAttribute(name, value);
  }
  svg.append(body);
  return svg;
}

/**
 * @typedef {object} VoiceInputOptions
 * @property {HTMLTextAreaElement} textarea - where transcripts go
 * @property {(message: string) => void} [onError] - shows a problem to the operator
 * @property {(constraints?: MediaTrackConstraints) => Promise<MediaStream>} [getMicrophone]
 *   - opens the microphone
 * @property {any} [Recorder] - the MediaRecorder constructor
 * @property {any} [AudioContextImpl] - the AudioContext constructor (`mic`)
 * @property {typeof fetch} [fetchImpl]
 * @property {() => boolean} [isActive] - whether the chat is showing, so the
 *   talk key acts (it is ignored while another view is in front)
 */

/**
 * @typedef {object} VoiceInput
 * @property {HTMLButtonElement} button - the mic icon, hidden until available
 * @property {() => Promise<boolean>} checkAvailability - ask the server; sets
 *   the inputs and shows the icon when voice input can work here
 * @property {() => VoiceInputKind[]} inputs - the inputs switched on
 * @property {(disabled: boolean) => void} setDisabled - follow the message
 *   box's locked state (a hand-off in progress)
 * @property {() => void} cancel - stop and discard any recording
 * @property {() => void} dispose - stop listening, and drop the page-wide
 *   listeners; for tearing the console down
 */

/**
 * Create the mic icon for the chat's input box.
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
  const isActive = options.isActive ?? (() => true);

  /** @type {Set<VoiceInputKind>} */
  let inputs = new Set(DEFAULT_INPUTS);
  let shortcut = DEFAULT_SHORTCUT;
  /** Whether the message box is locked; voice waits with it. */
  let locked = false;

  const button = /** @type {HTMLButtonElement} */ (document.createElement('button'));
  button.type = 'button';
  button.className = 'op-mic-btn';
  button.append(micIcon());
  button.hidden = true;

  // The box's own hint, kept: voice only adds how to talk to it.
  const basePlaceholder = textarea.placeholder || IDLE_PLACEHOLDER;
  let idlePlaceholder = basePlaceholder;

  // ---- Hold to talk (`button` and `key`) ---- //

  /** @type {'idle' | 'starting' | 'recording' | 'transcribing'} */
  let state = 'idle';
  let held = false;
  /** What started the recording in progress: the icon or the talk key. */
  /** @type {'button' | 'key'} */
  let heldBy = 'button';
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
    textarea.placeholder = placeholder;
    render();
  }

  function releaseMicrophone() {
    if (limit !== null) clearTimeout(limit);
    limit = null;
    for (const track of stream?.getTracks() ?? []) track.stop();
    stream = null;
  }

  /** @param {'button' | 'key'} source */
  async function start(source) {
    if (state !== 'idle' || locked || listenState !== 'off' || !inputs.has(source)) return;
    held = true;
    heldBy = source;
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
      if (tooShort && !cancelled) {
        onError(
          heldBy === 'key'
            ? `Hold ${shortcut.label} while you speak.`
            : 'Hold the mic button while you speak.'
        );
      }
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

  // ---- Listening (`mic`) ---- //

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
    if (pendingClips === 0)
      textarea.placeholder = next === 'off' ? idlePlaceholder : LISTEN_PLACEHOLDER;
    render();
  }

  function armIdleTimer() {
    if (idleTimer !== null) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      stopListening();
      onError('Listening turned off after five minutes without speech.');
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
    if (locked) {
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

  // ---- The icon ---- //

  /**
   * Show the state on the icon: `recording`/`hearing` pulse red, `listening`
   * is steady red, `starting`/`transcribing` the accent colour. Its role
   * follows the inputs: a listening toggle with `mic`, held with `button`,
   * else only an indicator for the talk key.
   */
  function render() {
    const shown = state !== 'idle' ? state : listenState !== 'off' ? listenState : 'idle';
    button.dataset.state = shown;
    const role = inputs.has('mic') ? 'toggle' : inputs.has('button') ? 'hold' : 'indicator';
    button.dataset.role = role;
    const orKey = inputs.has('key') ? ` (or hold ${shortcut.label})` : '';
    if (role === 'toggle') {
      const on = listenState !== 'off';
      button.setAttribute('aria-pressed', String(on));
      button.setAttribute('aria-label', on ? 'Stop listening' : 'Start listening');
      button.title = on
        ? 'Listening for your microphone: click to stop'
        : `Click to listen for your push-to-talk microphone${orKey}`;
    } else if (role === 'hold') {
      button.setAttribute('aria-pressed', String(state === 'recording'));
      button.setAttribute('aria-label', 'Hold to talk');
      button.title = `Hold to talk${orKey}`;
    } else {
      button.removeAttribute('aria-pressed');
      button.setAttribute('aria-label', `Voice input: hold ${shortcut.label} to talk`);
      button.title = `Hold ${shortcut.label} to talk`;
    }
    if (inputs.has('key')) button.setAttribute('aria-keyshortcuts', shortcut.aria);
    else button.removeAttribute('aria-keyshortcuts');
  }

  /**
   * Take the deployment's inputs and key, keep those this browser can serve,
   * and say how to talk in the box.
   * @param {unknown} nextInputs
   * @param {unknown} nextKey
   */
  function applyInputs(nextInputs, nextKey) {
    const next = normalizeInputs(nextInputs).filter((kind) =>
      kind === 'mic' ? Boolean(AudioContextImpl) : Boolean(Recorder)
    );
    if (inputs.has('mic') && !next.includes('mic')) stopListening();
    inputs = new Set(next);
    shortcut = DEFAULT_SHORTCUT;
    if (inputs.has('key') && nextKey != null && nextKey !== '') {
      const parsed = parseShortcut(nextKey);
      if (parsed) shortcut = parsed;
      else
        onError(`The talk key "${nextKey}" cannot be used here; hold ${shortcut.label} instead.`);
    }
    const ways = [
      inputs.has('mic') ? 'click the mic to listen' : '',
      inputs.has('button') && inputs.has('key')
        ? `hold the mic or ${shortcut.label} to talk`
        : inputs.has('button')
          ? 'hold the mic to talk'
          : inputs.has('key')
            ? `hold ${shortcut.label} to talk`
            : '',
    ].filter(Boolean);
    idlePlaceholder = ways.length ? `${basePlaceholder} (${ways.join(', or ')})` : basePlaceholder;
    button.disabled = locked && !inputs.has('mic');
    render();
  }

  // `button`: hold the icon. Pointer down starts, up (or leaving it) stops.
  const holdingIcon = () => inputs.has('button') && held && heldBy === 'button';
  button.addEventListener('pointerdown', (e) => {
    if (!inputs.has('button')) return;
    e.preventDefault();
    void start('button');
  });
  for (const type of ['pointerup', 'pointerleave', 'pointercancel']) {
    button.addEventListener(type, () => {
      if (holdingIcon()) stop();
    });
  }
  // The same from the keyboard while the icon has focus.
  button.addEventListener('keydown', (e) => {
    if (!inputs.has('button')) return;
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
      e.preventDefault();
      void start('button');
    }
  });
  button.addEventListener('keyup', (e) => {
    if (holdingIcon() && (e.key === ' ' || e.key === 'Enter')) stop();
  });
  // `mic`: a click turns listening on or off. With only `key`, the icon only
  // shows the state, so a click says which key to hold.
  button.addEventListener('click', () => {
    if (inputs.has('mic')) {
      if (listenState === 'off') void startListening();
      else stopListening();
    } else if (!inputs.has('button') && inputs.has('key')) {
      onError(`Hold ${shortcut.label} to talk.`);
    }
  });

  /** @param {KeyboardEvent} e */
  function onKeyDown(e) {
    if (e.key === 'Escape' && (state === 'recording' || state === 'starting')) {
      cancel();
      return;
    }
    // `key`: the talk key down starts, letting go of it or one of its
    // modifiers stops. It never types; repeats are ignored.
    if (!inputs.has('key') || !matchesShortcut(e, shortcut)) return;
    if (button.hidden || locked || listenState !== 'off' || !isActive()) return;
    e.preventDefault();
    if (e.repeat || state !== 'idle') return;
    void start('key');
  }
  /** @param {KeyboardEvent} e */
  function onKeyUp(e) {
    if (held && heldBy === 'key' && releasesShortcut(e.key, shortcut)) stop();
  }
  // Switching window mid-hold loses the key-up: stop rather than record on.
  function onBlur() {
    if (held && heldBy === 'key') stop();
  }
  // A hidden page (another tab, a minimised window) stops listening.
  function onVisibility() {
    if (document.hidden) stopListening();
  }
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);
  document.addEventListener('visibilitychange', onVisibility);

  setState('idle');

  return {
    button,
    inputs: () => [...inputs],
    async checkAvailability() {
      try {
        const res = await fetchImpl(withPrefix(STATUS_ENDPOINT), {
          cache: 'no-store',
        });
        const body = res.ok ? await res.json() : {};
        applyInputs(body.inputs, body.key);
        const available = Boolean(body.available) && inputs.size > 0;
        button.hidden = !available;
        if (!available) idlePlaceholder = basePlaceholder;
        if (state === 'idle' && listenState === 'off') textarea.placeholder = idlePlaceholder;
        return available;
      } catch {
        button.hidden = true;
        return false;
      }
    },
    setDisabled(disabled) {
      locked = disabled;
      button.disabled = disabled && !inputs.has('mic');
      if (disabled) cancel();
    },
    cancel,
    dispose() {
      cancel();
      stopListening();
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}

/**
 * Start voice input as one of the chat's features (chat-features.js): the
 * mic icon sits before Send, problems are told in the chat's log, the talk
 * key acts only while the chat is the surface on screen, and voice pauses
 * whenever the message box is locked (a hand-off), as typing does.
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
  sendBtn.parentElement?.prepend(voice.button);
  const follow = () => voice.setDisabled(textarea.disabled);
  new MutationObserver(follow).observe(textarea, {
    attributes: true,
    attributeFilter: ['disabled'],
  });
  follow();
  void voice.checkAvailability();
  return voice;
}
