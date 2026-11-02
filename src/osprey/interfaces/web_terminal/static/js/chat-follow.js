// @ts-check
/**
 * OSPREY Web Terminal — operator chat: following the turns this tab did not send.
 *
 * The console's live feed carries every frame the agent emits, whichever tab
 * asked: a turn another tab sent, a response the CLI starts on its own, a turn
 * already running when this tab opened. This module decides what each frame
 * means for the log: draw a new turn, continue the turn the transcript left
 * open, end the turn. It also keeps the log true when the feed drops. Each
 * reconnect opens with a snapshot (`hello`) that says whether a turn runs and
 * which frame is the newest, so a turn whose end or middle went by unseen is
 * redrawn from the transcript. The controller (chat.js) owns the DOM and the
 * network and hands in what this module needs.
 *
 * @module chat-follow
 */

import { endsTurn, isResponseFrame } from './chat-frames.js';

/** @typedef {import('./chat-client.js').ChatEvent} ChatEvent */
/** @typedef {import('./chat-client.js').ChatTurn} ChatTurn */

/**
 * What the follower needs from the console.
 * @typedef {object} FollowHost
 * @property {import('./chat-render.js').ChatRenderer} renderer - the log
 * @property {(event: ChatEvent) => void} draw - draw one frame, keeping the tail in view
 * @property {(key: string) => Promise<ChatTurn[]>} fetchHistory - read the transcript
 * @property {() => string} key - the session key the console shows
 * @property {() => boolean} sending - whether this tab's own turn streams
 * @property {(on: boolean) => void} setStreaming - put the console in or out of a running turn
 * @property {() => void} dismissInvitation - take the first-contact block away
 * @property {() => void} scrollToBottom
 * @property {(turns: ChatTurn[]) => void} redraw - clear the log and replay *turns* into it
 */

/**
 * @typedef {object} TurnFollower
 * @property {(frame: ChatEvent) => void} frame - one frame from the feed, its opening snapshot included
 * @property {() => void} noAgent - the feed found no agent for the chat
 * @property {() => void} restart - a new feed opens: no turn followed, no frame seen yet
 * @property {() => void} logChanged - the log changed under any transcript read in
 *   flight: this tab sent a prompt, or the console was cleared
 * @property {() => boolean} following - whether a turn this tab did not send is drawing
 */

/**
 * Follow the turns of the console's chat on its live feed.
 * @param {FollowHost} host
 * @returns {TurnFollower}
 */
export function createTurnFollower(host) {
  const { renderer } = host;

  /**
   * How the console follows a turn it did not send: `live` saw it start
   * (another tab's prompt, or the CLI's own response); `joined` missed part of
   * it — it came in part-way (a reload, a second tab opened mid-turn) or lost
   * frames to a dropped feed — and re-reads the transcript when it ends; null
   * when it follows none.
   * @type {null | 'live' | 'joined'}
   */
  let mirroring = null;

  /**
   * The number (`seq`) of the last frame the feed carried, or null before its
   * first snapshot. A reconnect's snapshot gives the newest frame's number, so
   * any difference means frames went by while the feed was down.
   * @type {number | null}
   */
  let lastSeq = null;

  /**
   * Counts what changes the turn the log shows: a turn that starts or ends, a
   * send, a cleared console. A transcript read that one of them overtook
   * leaves the log alone.
   */
  let generation = 0;

  /**
   * Start drawing a turn this tab did not send.
   * @param {'live' | 'joined'} how
   * @param {string | null} prompt - the operator's prompt, when the turn has one
   */
  function start(how, prompt) {
    mirroring = how;
    generation += 1;
    host.dismissInvitation();
    if (prompt) renderer.addUserMessage(prompt);
    host.setStreaming(true);
    renderer.beginTurn();
    host.scrollToBottom();
  }

  /**
   * The followed turn is over.
   * @param {{ missedEnd?: boolean }} [options] - `missedEnd`: the turn ended out
   *   of sight (the feed dropped, or lost its agent), so the transcript is read
   *   again even for a turn followed from its start
   */
  function end({ missedEnd = false } = {}) {
    const reread = missedEnd || mirroring === 'joined';
    mirroring = null;
    generation += 1;
    host.setStreaming(false);
    renderer.endTurn();
    // A turn joined part-way missed what it said before the feed connected;
    // the transcript has all of it.
    if (reread) void rereadTranscript();
  }

  /**
   * Redraw the log from the transcript once a followed turn is over, so the
   * turn shows whole. The log keeps what it shows until the transcript is in.
   * A turn that starts in the meantime owns the log: it re-reads the
   * transcript when it ends instead.
   * @returns {Promise<void>}
   */
  async function rereadTranscript() {
    const key = host.key();
    const mine = generation;
    /** @type {ChatTurn[]} */
    let turns;
    try {
      turns = await host.fetchHistory(key);
    } catch {
      return; // the log keeps what the feed drew
    }
    if (host.key() !== key || host.sending()) return;
    if (mine !== generation) {
      if (mirroring) mirroring = 'joined';
      else void rereadTranscript();
      return;
    }
    host.redraw(turns);
    // No turn runs, so one the transcript has not caught up with is over.
    renderer.endTurn();
    host.scrollToBottom();
  }

  /**
   * Catch a followed turn up after the feed dropped and missed part of it:
   * redraw the log from the transcript, which leaves the running turn open,
   * and keep drawing it live. Frames that land while the transcript is read
   * may be in neither, so the turn re-reads the transcript when it ends too.
   * @param {ChatEvent} hello - the reconnect's snapshot
   * @returns {Promise<void>}
   */
  async function resync(hello) {
    mirroring = 'joined';
    const key = host.key();
    const mine = ++generation;
    /** @type {ChatTurn[]} */
    let turns;
    try {
      turns = await host.fetchHistory(key);
    } catch {
      return; // the turn keeps drawing live, and is read again when it ends
    }
    if (host.key() !== key || mine !== generation) return;
    host.redraw(turns);
    renderer.beginTurn();
    for (const pending of hello.pending ?? []) host.draw(pending);
    if (hello.context) host.draw(hello.context);
    host.scrollToBottom();
  }

  /**
   * The snapshot each (re)connect of the feed opens with: whether a turn runs,
   * the requests it waits on, the newest frame's number and the window's last
   * reading.
   * @param {ChatEvent} hello
   */
  function onHello(hello) {
    // After a drop, frames past the last one this feed carried went by unseen.
    const missed = lastSeq !== null && hello.seq !== lastSeq;
    lastSeq = typeof hello.seq === 'number' ? hello.seq : null;
    if (!hello.in_flight) {
      // Nothing runs. A turn still drawn as running, or one the transcript
      // left open, ended out of sight: the transcript shows how.
      noAgent();
    } else if (!mirroring) {
      // A turn runs that this tab is not drawing: a reload or a second tab
      // came in part-way. The transcript's open turn, when there is one, goes on.
      start('joined', null);
      for (const pending of hello.pending ?? []) host.draw(pending);
    } else if (missed) {
      void resync(hello);
    }
    // The window's last reading, so the meter is right on a tab that opens mid-session.
    if (hello.context) host.draw(hello.context);
  }

  /**
   * One frame from the feed. The feed is closed while this tab's own turn
   * streams, so what it brings is a turn sent elsewhere (another tab) or never
   * sent (the CLI answering a background agent's report).
   * @param {ChatEvent} event
   */
  function frame(event) {
    if (event.type === 'hello') {
      onHello(event);
      return;
    }
    if (typeof event.seq === 'number') lastSeq = event.seq;
    if (event.type === 'turn_start') {
      // A turn starts only once the one before it is over. One still drawn as
      // running lost its end to a dropped feed: it is closed here, and the new
      // turn re-reads the transcript when it ends, which shows both whole.
      const missedEnd = mirroring !== null;
      mirroring = null;
      renderer.endTurn();
      start(missedEnd ? 'joined' : 'live', event.prompt ?? null);
      return;
    }
    if (!mirroring) {
      // Part of a response this tab is not drawing yet (its start came before
      // the feed connected); join it. A notice on its own just draws.
      if (!isResponseFrame(event)) {
        host.draw(event);
        return;
      }
      start('joined', null);
    }
    host.draw(event);
    if (endsTurn(event)) end();
  }

  /** No turn can be running: end one still drawn as running, or left open by the transcript. */
  function noAgent() {
    if (mirroring) end({ missedEnd: true });
    else if (renderer.hasOpenTurn()) void rereadTranscript();
  }

  return {
    frame,
    noAgent,
    restart() {
      mirroring = null;
      lastSeq = null;
      generation += 1;
    },
    logChanged() {
      generation += 1;
    },
    following: () => mirroring !== null,
  };
}
