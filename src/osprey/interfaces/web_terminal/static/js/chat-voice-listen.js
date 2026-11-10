// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: Listen mode for voice input.
 *
 * Made for a push-to-talk microphone whose button only lets sound through
 * while it is held (the common kind: the button sends nothing the page can
 * see). With Listen on, the page watches the microphone's loudness; sound
 * well above the quiet level starts a clip, and a short run of quiet ends it.
 * So holding the mic's button and speaking records, and letting go sends the
 * clip for transcription, with no key or click.
 *
 * A clip keeps a moment of sound from just before speech was detected, so the
 * first word is not clipped, and is encoded as 16 kHz mono WAV (what Whisper
 * works at; a minute stays under the server's size limit). Clips too short to
 * be speech (a bump of the button) are dropped.
 *
 * The detection (createClipCutter) is plain arithmetic on sample frames and
 * is tested without a browser; createListener is the thin layer that feeds it
 * from the microphone through the Web Audio API.
 *
 * @module chat-voice-listen
 */

/**
 * @typedef {object} ListenSettings
 * @property {number} startAboveFloorDb
 * @property {number} minStartDb
 * @property {number} startMs
 * @property {number} endMs
 * @property {number} endBelowStartDb
 * @property {number} preRollMs
 * @property {number} minSpeechMs
 * @property {number} maxClipMs
 * @property {number} clipSampleRate
 */

/**
 * How Listen mode decides what is speech. Levels are in dBFS (0 is the
 * loudest a sample can be; digital silence is clamped to -100).
 * @type {Readonly<ListenSettings>}
 */
export const LISTEN = Object.freeze({
  /** Sound this far above the learned quiet level counts as speech... */
  startAboveFloorDb: 15,
  /** ...but never quieter than this, so a near-silent floor is not a hair trigger. */
  minStartDb: -55,
  /** Sound must stay above the start level this long to start a clip. */
  startMs: 120,
  /** Speech ends once the level stays below the end level this long. */
  endMs: 700,
  /** The end level sits this far below the start level (hysteresis). */
  endBelowStartDb: 7,
  /** Sound kept from before speech was detected. */
  preRollMs: 300,
  /** A clip with less speech than this is dropped. */
  minSpeechMs: 400,
  /** The longest clip; speech running longer is cut here. */
  maxClipMs: 60_000,
  /** The sample rate clips are encoded at. */
  clipSampleRate: 16_000,
});

const SILENCE_DB = -100;

/**
 * The loudness of a frame of samples, as RMS in dBFS.
 * @param {Float32Array} samples
 * @returns {number}
 */
export function levelDb(samples) {
  if (samples.length === 0) return SILENCE_DB;
  let sum = 0;
  for (const s of samples) sum += s * s;
  const rms = Math.sqrt(sum / samples.length);
  return rms > 0 ? Math.max(SILENCE_DB, 20 * Math.log10(rms)) : SILENCE_DB;
}

/**
 * Encode frames of float samples as a 16-bit mono WAV, resampled to
 * *outRate* by averaging (the clip is for speech recognition, not listening).
 * @param {Float32Array[]} frames
 * @param {number} inRate
 * @param {number} [outRate]
 * @returns {Blob}
 */
export function encodeWav(frames, inRate, outRate = LISTEN.clipSampleRate) {
  const total = frames.reduce((n, f) => n + f.length, 0);
  const input = new Float32Array(total);
  let at = 0;
  for (const f of frames) {
    input.set(f, at);
    at += f.length;
  }
  const ratio = Math.max(1, inRate / outRate);
  const rate = ratio === 1 ? inRate : outRate;
  const count = Math.floor(total / ratio);
  const buffer = new ArrayBuffer(44 + count * 2);
  const view = new DataView(buffer);
  const text = (/** @type {number} */ offset, /** @type {string} */ s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  text(0, 'RIFF');
  view.setUint32(4, 36 + count * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per sample frame
  view.setUint16(34, 16, true); // bits per sample
  text(36, 'data');
  view.setUint32(40, count * 2, true);
  for (let i = 0; i < count; i++) {
    const from = Math.floor(i * ratio);
    const to = Math.max(from + 1, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = from; j < to; j++) sum += input[j];
    const s = Math.max(-1, Math.min(1, sum / (to - from)));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  // A byte view rather than the bare buffer: every Blob implementation reads it.
  return new Blob([new Uint8Array(buffer)], { type: 'audio/wav' });
}

/**
 * @typedef {object} ClipCutter
 * @property {(samples: Float32Array, sampleRate: number) => void} feed - one
 *   frame from the microphone (copied; the caller may reuse its buffer)
 * @property {() => boolean} hearing - whether a clip is being recorded
 * @property {() => void} reset - drop any clip in progress and relearn the floor
 */

/**
 * Turn a stream of sample frames into speech clips.
 * @param {{
 *   onStart?: () => void,
 *   onClip: (clip: Blob) => void,
 *   onDrop?: () => void,
 *   settings?: ListenSettings,
 * }} handlers - onStart when speech begins; onClip with each finished clip;
 *   onDrop when a clip ended too short to be speech
 * @returns {ClipCutter}
 */
export function createClipCutter({ onStart, onClip, onDrop, settings = LISTEN }) {
  /** The quiet level, learned while nobody speaks; null until the first frame. */
  /** @type {number | null} */
  let floor = null;
  let hearing = false;
  let aboveMs = 0;
  let quietMs = 0;
  let clipMs = 0;
  /** The quiet lead-in kept before speech, not counted as speech. */
  let leadMs = 0;
  let startDb = settings.minStartDb;
  /** Recent frames kept for the pre-roll while quiet. */
  /** @type {Float32Array[]} */
  let recent = [];
  let recentMs = 0;
  /** @type {Float32Array[]} */
  let clip = [];
  let rate = 0;

  function finish() {
    const speechMs = clipMs - leadMs - quietMs;
    const frames = clip;
    hearing = false;
    clip = [];
    clipMs = 0;
    quietMs = 0;
    aboveMs = 0;
    if (speechMs < settings.minSpeechMs) onDrop?.();
    else onClip(encodeWav(frames, rate));
  }

  return {
    feed(samples, sampleRate) {
      const frame = Float32Array.from(samples);
      const ms = (frame.length / sampleRate) * 1000;
      const db = levelDb(frame);
      rate = sampleRate;

      if (hearing) {
        clip.push(frame);
        clipMs += ms;
        if (db < startDb - settings.endBelowStartDb) quietMs += ms;
        else quietMs = 0;
        if (quietMs >= settings.endMs || clipMs >= settings.maxClipMs) finish();
        return;
      }

      if (floor === null) floor = db;
      startDb = Math.max(floor + settings.startAboveFloorDb, settings.minStartDb);
      recent.push(frame);
      recentMs += ms;
      while (
        recent.length > 1 &&
        recentMs - (recent[0].length / sampleRate) * 1000 >= settings.preRollMs
      ) {
        const dropped = /** @type {Float32Array} */ (recent.shift());
        recentMs -= (dropped.length / sampleRate) * 1000;
      }

      if (db >= startDb) {
        aboveMs += ms;
        if (aboveMs >= settings.startMs) {
          hearing = true;
          clip = recent;
          clipMs = recentMs;
          leadMs = recentMs - aboveMs;
          quietMs = 0;
          recent = [];
          recentMs = 0;
          onStart?.();
        }
        return;
      }
      aboveMs = 0;
      // The floor drops at once to anything quieter, and creeps up slowly, so
      // a passing noise does not raise it but a steadily louder room does.
      floor = db < floor ? db : floor + (db - floor) * 0.05;
    },
    hearing: () => hearing,
    reset() {
      floor = null;
      hearing = false;
      aboveMs = 0;
      quietMs = 0;
      clipMs = 0;
      recent = [];
      recentMs = 0;
      clip = [];
    },
  };
}

/**
 * @typedef {object} Listener
 * @property {() => void} stop - stop listening and release the microphone
 */

/**
 * Open the microphone and cut speech clips from it until stopped.
 * @param {{
 *   getMicrophone: (constraints: MediaTrackConstraints) => Promise<MediaStream>,
 *   AudioContextImpl: any,
 *   onStart?: () => void,
 *   onClip: (clip: Blob) => void,
 *   onDrop?: () => void,
 *   settings?: ListenSettings,
 * }} options
 * @returns {Promise<Listener>}
 */
export async function createListener(options) {
  const { getMicrophone, AudioContextImpl, ...handlers } = options;
  // No automatic gain: it would lift the quiet level the detection learns.
  const stream = await getMicrophone({
    autoGainControl: false,
    noiseSuppression: true,
    echoCancellation: true,
  });
  /** @type {any} */
  let context = null;
  try {
    context = new AudioContextImpl();
    const cutter = createClipCutter(handlers);
    const source = context.createMediaStreamSource(stream);
    // ScriptProcessorNode is deprecated in favour of AudioWorklet, but needs
    // no separate module file and is still supported by every browser; a
    // frame of 2048 samples is about 45 ms at 44.1-48 kHz.
    const node = context.createScriptProcessor(2048, 1, 1);
    node.onaudioprocess = (/** @type {any} */ e) =>
      cutter.feed(e.inputBuffer.getChannelData(0), context.sampleRate);
    source.connect(node);
    node.connect(context.destination); // a processor only runs when connected
    let stopped = false;
    return {
      stop() {
        if (stopped) return;
        stopped = true;
        node.onaudioprocess = null;
        source.disconnect();
        node.disconnect();
        void context.close?.();
        for (const track of stream.getTracks()) track.stop();
      },
    };
  } catch (err) {
    void context?.close?.();
    for (const track of stream.getTracks()) track.stop();
    throw err;
  }
}
