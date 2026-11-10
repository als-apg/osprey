// @ts-check
/**
 * Unit tests for chat-voice-listen.js and Listen mode in chat-voice.js -- the
 * Simple-view chat recording whenever a push-to-talk microphone lets sound
 * through:
 *   ./node_modules/.bin/vitest run tests/interfaces/web_terminal/chat-voice-listen.test.mjs
 *
 * Sound is stood in for by frames of constant level (an RMS in dBFS), fed
 * straight to the clip cutter or through a stand-in AudioContext. What is
 * pinned: a quiet floor never starts a clip; speech held long enough does,
 * with its lead-in kept; quiet after it ends the clip; a bump is dropped;
 * clips are 16 kHz WAV; with Listen on, a clip's transcript lands in the
 * input box (never sent), and the microphone is released when Listen goes
 * off, after five idle minutes, or when the page is hidden.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  LISTEN,
  createClipCutter,
  encodeWav,
  levelDb,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-voice-listen.js';
import {
  LISTEN_IDLE_MS,
  createVoiceInput,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-voice.js';

const RATE = 16_000;
const FRAME_MS = 50;

/**
 * One frame of constant level *db* dBFS (digital silence for -Infinity).
 * @param {number} db
 */
function frame(db, ms = FRAME_MS, rate = RATE) {
  const amplitude = Number.isFinite(db) ? 10 ** (db / 20) : 0;
  return new Float32Array(Math.round((rate * ms) / 1000)).fill(amplitude);
}

/**
 * Feed *ms* of sound at *db* to *feed*, frame by frame.
 * @param {(samples: Float32Array, rate: number) => void} feed
 * @param {number} db
 * @param {number} ms
 */
function play(feed, db, ms) {
  for (let t = 0; t < ms; t += FRAME_MS) feed(frame(db), RATE);
}

/** The level a WAV clip's samples were encoded at. */
function wavInfo(/** @type {ArrayBuffer} */ buffer) {
  const view = new DataView(buffer);
  const tag = (/** @type {number} */ at) =>
    String.fromCharCode(...new Uint8Array(buffer.slice(at, at + 4)));
  return {
    riff: tag(0),
    wave: tag(8),
    rate: view.getUint32(24, true),
    channels: view.getUint16(22, true),
    bits: view.getUint16(34, true),
    samples: view.getUint32(40, true) / 2,
  };
}

describe('levelDb', () => {
  test('a full-scale frame is 0 dB, silence is the floor', () => {
    expect(levelDb(new Float32Array(10).fill(1))).toBeCloseTo(0);
    expect(levelDb(new Float32Array(10))).toBe(-100);
    expect(levelDb(new Float32Array(0))).toBe(-100);
  });

  test('a frame at a given level reads back at that level', () => {
    expect(levelDb(frame(-30))).toBeCloseTo(-30);
  });
});

describe('encodeWav', () => {
  test('writes 16-bit mono PCM resampled to 16 kHz', async () => {
    const clip = encodeWav([frame(-20, 500, 48_000)], 48_000);
    expect(clip.type).toBe('audio/wav');
    const info = wavInfo(await clip.arrayBuffer());
    expect(info).toEqual({
      riff: 'RIFF',
      wave: 'WAVE',
      rate: 16_000,
      channels: 1,
      bits: 16,
      samples: 8_000,
    });
  });

  test('a minute of speech stays under the server limit (2 MiB)', () => {
    expect(encodeWav([frame(-20, 60_000, 48_000)], 48_000).size).toBeLessThan(2 * 1024 * 1024);
  });
});

describe('createClipCutter', () => {
  /** @type {Blob[]} */
  let clips;
  let starts = 0;
  let drops = 0;
  /** @type {ReturnType<typeof createClipCutter>} */
  let cutter;

  beforeEach(() => {
    clips = [];
    starts = 0;
    drops = 0;
    cutter = createClipCutter({
      onStart: () => (starts += 1),
      onClip: (c) => clips.push(c),
      onDrop: () => (drops += 1),
    });
  });

  test('a muted mic (digital silence) never starts a clip', () => {
    play(cutter.feed, -Infinity, 5_000);
    expect(starts).toBe(0);
    expect(cutter.hearing()).toBe(false);
  });

  test('a steady room hum learns the floor instead of starting a clip', () => {
    play(cutter.feed, -50, 5_000);
    expect(starts).toBe(0);
  });

  test('speech while the button is held records, and the quiet after release ends it', async () => {
    play(cutter.feed, -Infinity, 1_000);
    play(cutter.feed, -25, 1_500); // button held, speaking
    expect(starts).toBe(1);
    expect(cutter.hearing()).toBe(true);
    play(cutter.feed, -Infinity, 1_000); // button released
    expect(cutter.hearing()).toBe(false);
    expect(clips).toHaveLength(1);
    const { samples } = wavInfo(await clips[0].arrayBuffer());
    // The 1.5 s of speech, the pre-roll before it and the quiet that ended it.
    expect(samples / RATE).toBeGreaterThanOrEqual(1.5 + LISTEN.endMs / 1000);
    expect(samples / RATE).toBeLessThanOrEqual(
      1.5 + (LISTEN.preRollMs + LISTEN.endMs + 2 * FRAME_MS) / 1000
    );
  });

  test('a short pause inside speech does not end the clip', () => {
    play(cutter.feed, -Infinity, 500);
    play(cutter.feed, -25, 800);
    play(cutter.feed, -Infinity, LISTEN.endMs - 2 * FRAME_MS);
    play(cutter.feed, -25, 800);
    play(cutter.feed, -Infinity, 1_000);
    expect(clips).toHaveLength(1);
  });

  test('a blip shorter than the start time starts nothing', () => {
    play(cutter.feed, -Infinity, 500);
    play(cutter.feed, -20, LISTEN.startMs - FRAME_MS);
    play(cutter.feed, -Infinity, 1_000);
    expect(starts).toBe(0);
    expect(clips).toHaveLength(0);
  });

  test('a bump too short to be speech is dropped', () => {
    play(cutter.feed, -Infinity, 500);
    play(cutter.feed, -20, 200);
    play(cutter.feed, -Infinity, 1_000);
    expect(starts).toBe(1);
    expect(drops).toBe(1);
    expect(clips).toHaveLength(0);
  });

  test('speech louder than a room hum is still heard', () => {
    play(cutter.feed, -60, 2_000);
    play(cutter.feed, -30, 1_000);
    play(cutter.feed, -60, 1_000);
    expect(clips).toHaveLength(1);
  });

  test('speech running past the longest clip is cut there', () => {
    play(cutter.feed, -Infinity, 500);
    play(cutter.feed, -25, LISTEN.maxClipMs + 1_000);
    expect(clips).toHaveLength(1);
  });
});

// ---- Listen mode in the voice input ---- //

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

/** An AudioContext stand-in whose processor the test drives by hand. */
class FakeAudioContext {
  /** @type {FakeAudioContext[]} */
  static made = [];
  constructor() {
    this.sampleRate = RATE;
    this.closed = false;
    this.destination = {};
    /** @type {any} */
    this.node = null;
    FakeAudioContext.made.push(this);
  }
  createMediaStreamSource() {
    return { connect() {}, disconnect() {} };
  }
  createScriptProcessor() {
    this.node = { onaudioprocess: null, connect() {}, disconnect() {} };
    return this.node;
  }
  close() {
    this.closed = true;
    return Promise.resolve();
  }
  /** Play *ms* of sound at *db* into the processor. */
  play(/** @type {number} */ db, /** @type {number} */ ms) {
    for (let t = 0; t < ms; t += FRAME_MS) {
      const samples = frame(db);
      this.node?.onaudioprocess?.({
        inputBuffer: { getChannelData: () => samples },
      });
    }
  }
}

describe('Listen mode', () => {
  /** @type {HTMLTextAreaElement} */
  let textarea;
  /** @type {ReturnType<typeof vi.fn>} */
  let fetchImpl;
  /** @type {import('vitest').Mock<(message: string) => void>} */
  let onError;
  /** @type {ReturnType<typeof fakeStream>} */
  let mic;
  /** @type {any} */
  let getMicrophone;
  /** @type {any[]} */
  let built;

  function build(overrides = {}) {
    const voice = createVoiceInput({
      textarea,
      onError: (/** @type {string} */ m) => onError(m),
      getMicrophone: /** @type {any} */ (getMicrophone),
      Recorder: class {},
      AudioContextImpl: FakeAudioContext,
      fetchImpl: /** @type {any} */ (fetchImpl),
      ...overrides,
    });
    built.push(voice);
    return voice;
  }

  /** Build, show the buttons, and turn Listen on. */
  async function listening(overrides = {}) {
    const voice = build(overrides);
    await voice.checkAvailability();
    voice.listenButton.click();
    await vi.advanceTimersByTimeAsync(0);
    return {
      voice,
      ctx: /** @type {FakeAudioContext} */ (FakeAudioContext.made.at(-1)),
    };
  }

  const transcribeCalls = () =>
    fetchImpl.mock.calls.filter(([url]) => String(url).endsWith('/api/speech/transcribe'));

  beforeEach(() => {
    vi.useFakeTimers();
    textarea = document.createElement('textarea');
    document.body.append(textarea);
    onError = vi.fn();
    mic = fakeStream();
    getMicrophone = vi.fn(async () => mic.stream);
    FakeAudioContext.made = [];
    built = [];
    fetchImpl = vi.fn(async (/** @type {string} */ url) => {
      if (url.endsWith('/api/speech/status')) {
        return {
          ok: true,
          json: async () => ({ available: true, model: 'small.en' }),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ text: 'read the beam current' }),
      };
    });
  });

  afterEach(() => {
    for (const voice of built) voice.dispose();
    vi.useRealTimers();
    document.body.replaceChildren();
  });

  test('the Listen button shows only when the server can transcribe and the browser has Web Audio', async () => {
    const voice = build();
    expect(voice.listenButton.hidden).toBe(true);
    await voice.checkAvailability();
    expect(voice.listenButton.hidden).toBe(false);

    const without = build({ AudioContextImpl: undefined });
    vi.stubGlobal('AudioContext', undefined);
    await without.checkAvailability();
    vi.unstubAllGlobals();
    expect(without.listenButton.hidden).toBe(true);
  });

  test('turning Listen on opens the mic without automatic gain and shows it is listening', async () => {
    const { voice } = await listening();
    expect(getMicrophone).toHaveBeenCalledWith(expect.objectContaining({ autoGainControl: false }));
    expect(voice.listenButton.dataset.state).toBe('listening');
    expect(voice.listenButton.getAttribute('aria-pressed')).toBe('true');
    expect(textarea.placeholder).toMatch(/hold your mic/i);
  });

  test('speaking with the mic button held puts the transcript in the box, unsent', async () => {
    const { voice, ctx } = await listening();
    ctx.play(-Infinity, 1_000);
    ctx.play(-25, 1_500);
    expect(voice.listenButton.dataset.state).toBe('hearing');
    ctx.play(-Infinity, 1_000);
    await vi.advanceTimersByTimeAsync(0);

    const calls = transcribeCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0][1].headers['Content-Type']).toBe('audio/wav');
    expect(textarea.value).toBe('read the beam current');
    expect(voice.listenButton.dataset.state).toBe('listening');
  });

  test('two clips in a row are transcribed in the order spoken', async () => {
    let n = 0;
    fetchImpl.mockImplementation(async (/** @type {string} */ url) => {
      if (url.endsWith('/api/speech/status'))
        return { ok: true, json: async () => ({ available: true }) };
      n += 1;
      const text = `part ${n}`;
      return { ok: true, status: 200, json: async () => ({ text }) };
    });
    const { ctx } = await listening();
    for (let i = 0; i < 2; i += 1) {
      ctx.play(-Infinity, 500);
      ctx.play(-25, 800);
      ctx.play(-Infinity, 1_000);
    }
    await vi.advanceTimersByTimeAsync(0);
    expect(textarea.value).toBe('part 1 part 2');
  });

  test('hold-to-talk is set aside while listening', async () => {
    const { voice } = await listening();
    voice.button.dispatchEvent(new Event('pointerdown', { cancelable: true }));
    await vi.advanceTimersByTimeAsync(0);
    expect(getMicrophone).toHaveBeenCalledTimes(1); // only Listen's
  });

  test('turning Listen off releases the microphone and closes the audio', async () => {
    const { voice, ctx } = await listening();
    voice.listenButton.click();
    expect(mic.track.stopped).toBe(true);
    expect(ctx.closed).toBe(true);
    expect(voice.listenButton.dataset.state).toBe('off');
    expect(textarea.placeholder).not.toMatch(/hold your mic/i);
  });

  test('five minutes without speech turns Listen off and says so', async () => {
    const { voice } = await listening();
    await vi.advanceTimersByTimeAsync(LISTEN_IDLE_MS);
    expect(voice.listenButton.dataset.state).toBe('off');
    expect(mic.track.stopped).toBe(true);
    expect(onError).toHaveBeenCalledWith(expect.stringMatching(/five minutes/));
  });

  test('speech restarts the idle clock', async () => {
    const { voice, ctx } = await listening();
    ctx.play(-Infinity, 500);
    await vi.advanceTimersByTimeAsync(LISTEN_IDLE_MS - 1_000);
    ctx.play(-25, 800);
    ctx.play(-Infinity, 1_000);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(voice.listenButton.dataset.state).toBe('listening');
  });

  test('a hidden page stops listening', async () => {
    const { voice } = await listening();
    Object.defineProperty(document, 'hidden', {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event('visibilitychange'));
    Object.defineProperty(document, 'hidden', {
      configurable: true,
      value: false,
    });
    expect(voice.listenButton.dataset.state).toBe('off');
    expect(mic.track.stopped).toBe(true);
  });

  test('a clip while the message box is locked is not transcribed, and says why', async () => {
    const { voice, ctx } = await listening();
    voice.setDisabled(true);
    ctx.play(-Infinity, 500);
    ctx.play(-25, 800);
    ctx.play(-Infinity, 1_000);
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls()).toHaveLength(0);
    expect(onError).toHaveBeenCalledWith(expect.stringMatching(/message box is locked/));
    expect(voice.listenButton.dataset.state).toBe('listening');
  });

  test('a clip after the chat left the front turns Listen off instead', async () => {
    let active = true;
    const { voice, ctx } = await listening({ isActive: () => active });
    active = false;
    ctx.play(-Infinity, 500);
    ctx.play(-25, 800);
    ctx.play(-Infinity, 1_000);
    await vi.advanceTimersByTimeAsync(0);
    expect(transcribeCalls()).toHaveLength(0);
    expect(voice.listenButton.dataset.state).toBe('off');
  });

  test('a refused microphone says so and leaves Listen off', async () => {
    getMicrophone.mockRejectedValueOnce(new Error('NotAllowedError'));
    const { voice } = await listening();
    expect(voice.listenButton.dataset.state).toBe('off');
    expect(onError).toHaveBeenCalledWith(expect.stringMatching(/microphone could not be opened/));
  });

  test('disposing the input stops listening', async () => {
    const { voice } = await listening();
    voice.dispose();
    expect(mic.track.stopped).toBe(true);
  });
});
