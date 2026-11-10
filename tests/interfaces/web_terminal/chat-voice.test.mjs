// @ts-check
/**
 * Unit tests for chat-voice.js -- the Simple-view chat's hold-to-talk mic:
 *   ./node_modules/.bin/vitest run tests/interfaces/web_terminal/chat-voice.test.mjs
 *
 * The browser's microphone and MediaRecorder are stood in for: the microphone
 * is a stream whose tracks record whether they were stopped, and the recorder
 * hands back one fixed chunk when stopped. The server is a stubbed fetch. What
 * is pinned: the button shows only when the server can transcribe; holding
 * records and releasing sends the clip; the transcript lands in the input box
 * (never sent); taps, Esc, refusals and failures say why and leave the box as
 * it was; the microphone is always released.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  DEFAULT_SHORTCUT,
  MAX_RECORDING_MS,
  createVoiceInput,
  matchesShortcut,
  insertAtCursor,
  pickMimeType,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-voice.js';

/** A microphone stream whose tracks remember being stopped. */
function fakeStream() {
  const track = {
    stopped: false,
    stop() {
      this.stopped = true;
    },
  };
  return { track, stream: /** @type {any} */ ({ getTracks: () => [track] }) };
}

/** A MediaRecorder stand-in: one chunk of "audio" per recording. */
class FakeRecorder extends EventTarget {
  /** @type {FakeRecorder[]} */
  static made = [];
  /** @param {string} type */
  static isTypeSupported(type) {
    return type === 'audio/webm;codecs=opus';
  }
  /** @param {any} stream @param {{ mimeType?: string }} [options] */
  constructor(stream, options) {
    super();
    this.stream = stream;
    this.mimeType = options?.mimeType ?? '';
    this.state = 'inactive';
    FakeRecorder.made.push(this);
  }
  start() {
    this.state = 'recording';
  }
  stop() {
    this.state = 'inactive';
    const data = new Blob(['audio-bytes'], { type: 'audio/webm' });
    this.dispatchEvent(Object.assign(new Event('dataavailable'), { data }));
    this.dispatchEvent(new Event('stop'));
  }
}

/** @type {HTMLTextAreaElement} */
let textarea;
/** @type {ReturnType<typeof vi.fn>} */
let fetchImpl;
/** @type {import('vitest').Mock<(message: string) => void>} */
let onError;
/** @type {ReturnType<typeof fakeStream>} */
let mic;
/** Every voice input a test built, disposed after it so none keep listening. */
/** @type {import('../../../src/osprey/interfaces/web_terminal/static/js/chat-voice.js').VoiceInput[]} */
let built = [];

/** @param {{ transcript?: string, status?: number, error?: string }} [answer] */
function serverSays(answer = {}) {
  fetchImpl.mockImplementation(async (/** @type {string} */ url) => {
    if (url.endsWith('/api/speech/status')) {
      return { ok: true, json: async () => ({ available: true, model: 'base.en' }) };
    }
    const status = answer.status ?? 200;
    return {
      ok: status < 400,
      status,
      json: async () =>
        status < 400 ? { text: answer.transcript ?? '' } : { error: answer.error },
    };
  });
}

/** Build the input with every browser piece stood in. */
function build(overrides = {}) {
  const voice = createVoiceInput({
    textarea,
    onError: (/** @type {string} */ m) => onError(m),
    getMicrophone: async () => mic.stream,
    Recorder: FakeRecorder,
    fetchImpl: /** @type {any} */ (fetchImpl),
    ...overrides,
  });
  built.push(voice);
  return voice;
}

/** Hold the button for *ms*, then release it, letting promises settle. */
async function hold(/** @type {HTMLButtonElement} */ button, ms = 1_000) {
  button.dispatchEvent(new Event('pointerdown', { cancelable: true }));
  await vi.advanceTimersByTimeAsync(0);
  await vi.advanceTimersByTimeAsync(ms);
  button.dispatchEvent(new Event('pointerup'));
  await vi.advanceTimersByTimeAsync(0);
}

const transcribeCalls = () =>
  fetchImpl.mock.calls.filter(([url]) => String(url).endsWith('/api/speech/transcribe'));

beforeEach(() => {
  vi.useFakeTimers();
  textarea = document.createElement('textarea');
  document.body.append(textarea);
  fetchImpl = vi.fn();
  onError = vi.fn();
  mic = fakeStream();
  FakeRecorder.made = [];
  serverSays({ transcript: 'read the beam current' });
});

afterEach(() => {
  for (const voice of built) voice.dispose();
  built = [];
  vi.useRealTimers();
  document.body.replaceChildren();
});

describe('insertAtCursor', () => {
  test.each([
    ['', 0, 'read it', 'read it'],
    ['please', 6, 'read it', 'please read it'],
    ['please now', 6, 'read it', 'please read it now'],
    ['please ', 7, 'read it', 'please read it'],
  ])('%j with the cursor at %i, inserting %j, gives %j', (value, at, text, result) => {
    textarea.value = value;
    textarea.setSelectionRange(at, at);
    insertAtCursor(textarea, text);
    expect(textarea.value).toBe(result);
  });

  test('leaves the cursor after the inserted text and announces an input', () => {
    const onInput = vi.fn();
    textarea.addEventListener('input', onInput);
    textarea.value = 'ab';
    textarea.setSelectionRange(1, 1);
    insertAtCursor(textarea, 'X');
    expect(textarea.value).toBe('a X b');
    expect(textarea.selectionStart).toBe(4);
    expect(onInput).toHaveBeenCalled();
  });
});

describe('pickMimeType', () => {
  test('prefers Opus in WebM where the browser records it', () => {
    expect(pickMimeType(FakeRecorder)).toBe('audio/webm;codecs=opus');
  });

  test('falls back to the browser default otherwise', () => {
    expect(pickMimeType({ isTypeSupported: () => false })).toBe('');
  });
});

describe('the hold-to-talk mic', () => {
  test('is hidden until the server says it can transcribe', async () => {
    const voice = build();
    expect(voice.button.hidden).toBe(true);
    expect(await voice.checkAvailability()).toBe(true);
    expect(voice.button.hidden).toBe(false);
  });

  test('stays hidden when the server cannot, or the browser cannot record', async () => {
    fetchImpl.mockResolvedValue({ ok: true, json: async () => ({ available: false }) });
    expect(await build().checkAvailability()).toBe(false);
    serverSays();
    const noRecorder = build({ Recorder: undefined });
    expect(await noRecorder.checkAvailability()).toBe(false);
    expect(noRecorder.button.hidden).toBe(true);
  });

  test('holding records; releasing sends the clip and puts the text in the box, unsent', async () => {
    const voice = build();
    textarea.value = 'please';
    textarea.setSelectionRange(6, 6);
    await hold(voice.button);

    expect(FakeRecorder.made.length).toBe(1);
    expect(transcribeCalls().length).toBe(1);
    const [, init] = transcribeCalls()[0];
    expect(init.method).toBe('POST');
    expect(init.headers['Content-Type']).toBe('audio/webm');
    expect(textarea.value).toBe('please read the beam current');
    expect(mic.track.stopped).toBe(true);
    expect(voice.button.dataset.state).toBe('idle');
  });

  test('shows that it is listening, then transcribing', async () => {
    let answer = /** @type {(response: any) => void} */ (() => {});
    fetchImpl.mockImplementation(
      (/** @type {string} */ url) =>
        new Promise((resolve) => {
          answer = resolve;
          if (url.endsWith('/status'))
            resolve({ ok: true, json: async () => ({ available: true }) });
        })
    );
    const voice = build();
    voice.button.dispatchEvent(new Event('pointerdown', { cancelable: true }));
    await vi.advanceTimersByTimeAsync(0);
    expect(voice.button.dataset.state).toBe('recording');
    expect(voice.button.getAttribute('aria-pressed')).toBe('true');
    expect(textarea.placeholder).toMatch(/^Listening/);

    await vi.advanceTimersByTimeAsync(1_000);
    voice.button.dispatchEvent(new Event('pointerup'));
    await vi.advanceTimersByTimeAsync(0);
    expect(voice.button.dataset.state).toBe('transcribing');
    expect(textarea.placeholder).toBe('Transcribing…');

    answer({ ok: true, json: async () => ({ text: 'ok' }) });
    await vi.advanceTimersByTimeAsync(0);
    expect(voice.button.dataset.state).toBe('idle');
  });

  test('a quick tap is not sent, and says to hold', async () => {
    const voice = build();
    await hold(voice.button, 100);
    expect(transcribeCalls().length).toBe(0);
    expect(onError).toHaveBeenCalledWith('Hold Ctrl + Space or the mic button while you speak.');
    expect(mic.track.stopped).toBe(true);
  });

  test('Esc cancels a recording: nothing is sent and the box is untouched', async () => {
    const voice = build();
    textarea.value = 'draft';
    voice.button.dispatchEvent(new Event('pointerdown', { cancelable: true }));
    await vi.advanceTimersByTimeAsync(1_000);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls().length).toBe(0);
    expect(textarea.value).toBe('draft');
    expect(mic.track.stopped).toBe(true);
  });

  test('releasing before the microphone opens records nothing', async () => {
    /** @type {(s: any) => void} */
    let grant = () => {};
    const voice = build({ getMicrophone: () => new Promise((resolve) => (grant = resolve)) });
    voice.button.dispatchEvent(new Event('pointerdown', { cancelable: true }));
    voice.button.dispatchEvent(new Event('pointerup'));
    grant(mic.stream);
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeRecorder.made.length).toBe(0);
    expect(mic.track.stopped).toBe(true);
  });

  test('a refused microphone says so', async () => {
    const voice = build({ getMicrophone: async () => Promise.reject(new Error('denied')) });
    await hold(voice.button);
    expect(onError).toHaveBeenCalledWith(
      'The microphone could not be opened. Check that this page may use it.'
    );
    expect(voice.button.dataset.state).toBe('idle');
  });

  test("the server's error is shown, and the box is untouched", async () => {
    serverSays({ status: 503, error: 'Whisper is not installed on the server.' });
    const voice = build();
    textarea.value = 'draft';
    await hold(voice.button);
    expect(onError).toHaveBeenCalledWith('Whisper is not installed on the server.');
    expect(textarea.value).toBe('draft');
  });

  test('no speech recognised is said, not inserted', async () => {
    serverSays({ transcript: '   ' });
    const voice = build();
    await hold(voice.button);
    expect(textarea.value).toBe('');
    expect(onError).toHaveBeenCalledWith(
      'No speech was recognised. Try again, closer to the microphone.'
    );
  });

  test('a recording stops by itself at the time limit, and is sent', async () => {
    const voice = build();
    voice.button.dispatchEvent(new Event('pointerdown', { cancelable: true }));
    await vi.advanceTimersByTimeAsync(MAX_RECORDING_MS + 10);
    expect(transcribeCalls().length).toBe(1);
    expect(mic.track.stopped).toBe(true);
  });

  test('the keyboard works too: hold Space on the focused button', async () => {
    const voice = build();
    voice.button.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));
    await vi.advanceTimersByTimeAsync(1_000);
    voice.button.dispatchEvent(new KeyboardEvent('keyup', { key: ' ' }));
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls().length).toBe(1);
  });

  test('disabling it (a reply streaming) cancels a recording and blocks new ones', async () => {
    const voice = build();
    voice.button.dispatchEvent(new Event('pointerdown', { cancelable: true }));
    await vi.advanceTimersByTimeAsync(1_000);
    voice.setDisabled(true);
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls().length).toBe(0);
    expect(mic.track.stopped).toBe(true);

    await hold(voice.button);
    expect(FakeRecorder.made.length).toBe(1);
  });
});

describe('the Ctrl + Space shortcut', () => {
  /** @param {string} type @param {Record<string, any>} [init] */
  const key = (type, init = {}) =>
    new KeyboardEvent(type, { key: ' ', ctrlKey: true, cancelable: true, bubbles: true, ...init });

  /** Make the mic available, as the server would. */
  async function ready(overrides = {}) {
    const voice = build(overrides);
    await voice.checkAvailability();
    return voice;
  }

  test.each([
    [{ key: ' ', ctrlKey: true }, true],
    [{ key: ' ' }, false],
    [{ key: ' ', ctrlKey: true, shiftKey: true }, false],
    [{ key: ' ', ctrlKey: true, altKey: true }, false],
    [{ key: 'k', ctrlKey: true }, false],
  ])('matchesShortcut(%j) is %s', (init, matches) => {
    expect(matchesShortcut(new KeyboardEvent('keydown', init), DEFAULT_SHORTCUT)).toBe(matches);
  });

  test('holding it records; letting go transcribes into the box', async () => {
    const voice = await ready();
    textarea.focus();
    const down = key('keydown');
    document.dispatchEvent(down);
    await vi.advanceTimersByTimeAsync(0);
    expect(down.defaultPrevented).toBe(true);
    expect(voice.button.dataset.state).toBe('recording');

    await vi.advanceTimersByTimeAsync(1_000);
    document.dispatchEvent(key('keyup'));
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls().length).toBe(1);
    expect(textarea.value).toBe('read the beam current');
  });

  test('letting go of Ctrl first also stops it', async () => {
    await ready();
    document.dispatchEvent(key('keydown'));
    await vi.advanceTimersByTimeAsync(1_000);
    document.dispatchEvent(new KeyboardEvent('keyup', { key: 'Control' }));
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls().length).toBe(1);
  });

  test('key repeats while held do not start it again', async () => {
    await ready();
    document.dispatchEvent(key('keydown'));
    await vi.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i += 1) document.dispatchEvent(key('keydown', { repeat: true }));
    await vi.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i += 1) document.dispatchEvent(key('keydown'));
    await vi.advanceTimersByTimeAsync(1_000);
    document.dispatchEvent(key('keyup'));
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeRecorder.made.length).toBe(1);
    expect(transcribeCalls().length).toBe(1);
  });

  test('switching windows mid-hold stops it, rather than recording on', async () => {
    await ready();
    document.dispatchEvent(key('keydown'));
    await vi.advanceTimersByTimeAsync(1_000);
    window.dispatchEvent(new Event('blur'));
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls().length).toBe(1);
    expect(mic.track.stopped).toBe(true);
  });

  test('Space alone still types; the shortcut is left alone when voice is unavailable', async () => {
    const voice = build();
    const unavailable = key('keydown');
    document.dispatchEvent(unavailable);
    expect(unavailable.defaultPrevented).toBe(false);
    expect(voice.button.dataset.state).toBe('idle');

    await voice.checkAvailability();
    const plainSpace = key('keydown', { ctrlKey: false });
    document.dispatchEvent(plainSpace);
    expect(plainSpace.defaultPrevented).toBe(false);
    expect(voice.button.dataset.state).toBe('idle');
  });

  test('it does nothing while the chat is not the view in front, or while disabled', async () => {
    let active = false;
    const voice = await ready({ isActive: () => active });
    document.dispatchEvent(key('keydown'));
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeRecorder.made.length).toBe(0);

    active = true;
    voice.setDisabled(true);
    document.dispatchEvent(key('keydown'));
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeRecorder.made.length).toBe(0);
  });

  test('the button and the input box name the shortcut once voice is available', async () => {
    // The box keeps its own hint; voice only adds how to talk to it.
    textarea.placeholder = 'Message the agent…';
    const voice = build();
    expect(voice.button.title).toBe('Hold to talk (Ctrl + Space)');
    expect(voice.button.getAttribute('aria-keyshortcuts')).toBe('Control+Space');
    expect(textarea.placeholder).toBe('Message the agent…');
    await voice.checkAvailability();
    expect(textarea.placeholder).toBe('Message the agent… (hold Ctrl + Space to talk)');
  });
});

test('dispose stops it listening for the shortcut', async () => {
  const voice = build();
  await voice.checkAvailability();
  voice.dispose();
  const down = new KeyboardEvent('keydown', { key: ' ', ctrlKey: true, cancelable: true });
  document.dispatchEvent(down);
  await vi.advanceTimersByTimeAsync(0);
  expect(down.defaultPrevented).toBe(false);
  expect(FakeRecorder.made.length).toBe(0);
});
