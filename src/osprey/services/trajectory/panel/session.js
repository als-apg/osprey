// @ts-check
/**
 * session — whether this panel tab holds the target, and what the session
 * controls offer.
 *
 * The panel holds the session only while it has a `client_id`, kept in memory
 * and nowhere else: a second tab of the same owner is a second client, and the
 * snapshot (which names a holder by owner and door, never by `client_id`)
 * cannot tell the two apart. The id is dropped when
 *
 * - a `taken_over` event arrives that this tab did not start;
 * - a call is refused with `session_taken_over` or `not_session_holder`;
 * - a snapshot shows no holder, or a holder other than this owner's panel —
 *   but only once a snapshot received after the grant has already shown this
 *   owner's panel as holder. The stream and the POST travel on separate
 *   connections, so a snapshot older than the grant can arrive after its
 *   answer; it must not drop the id.
 *
 * A wrongly kept id is harmless: the next call carrying it is refused and the
 * refusal drops it. A wrongly dropped id strands the holder without controls,
 * so every ambiguous case keeps the id.
 *
 * The session is never released on page unload; Release is an explicit
 * control, and a reload leaves the session held until it is taken over.
 *
 * `sessionControls` is the pure decision of what the controls offer; the
 * holder state (`createSession`) is fed the POST outcomes and the stream
 * events by the caller, so this module performs no I/O.
 *
 * @module session
 */

/** The door this panel opens and takes over through. */
export const PANEL_KIND = 'panel';

/** Refusal codes meaning this tab's `client_id` no longer holds the target. */
export const LOST_SESSION_CODES = Object.freeze(['session_taken_over', 'not_session_holder']);

/**
 * @typedef {object} HolderView
 * @property {string} owner
 * @property {'panel' | 'agent'} kind
 */

/**
 * @typedef {object} RunningAction
 * @property {string} name
 * @property {boolean} stopping
 */

/**
 * The snapshot fields this module reads.
 * @typedef {object} SessionSnapshot
 * @property {HolderView | null} holder
 * @property {RunningAction | null} action
 */

/**
 * The answer of POST session and POST takeover.
 * @typedef {object} SessionGrant
 * @property {string} client_id
 * @property {string} owner
 * @property {string} target
 */

/**
 * @typedef {object} TakenOverEvent
 * @property {HolderView} previous
 * @property {HolderView} by
 */

/**
 * What the action in flight is, for the controls to name it.
 * @typedef {object} RunningView
 * @property {string} name      The action (`shot`, `setup`, `apply`, ...).
 * @property {string | null} owner  Its owner: the snapshot's holder.
 * @property {string | null} kind   The holder's door.
 * @property {boolean} stopping
 */

/**
 * @typedef {object} SessionControls
 * @property {boolean} holding    This tab holds the session.
 * @property {boolean} pending    An open or take-over is in flight.
 * @property {boolean} open       Offer Open session.
 * @property {boolean} takeOver   Offer Take over.
 * @property {boolean} release    Offer Release.
 * @property {boolean} stop       Offer Stop.
 * @property {RunningView | null} running  The action in flight, if any.
 */

/** @type {SessionControls} */
const NOTHING = Object.freeze({
  holding: false,
  pending: false,
  open: false,
  takeOver: false,
  release: false,
  stop: false,
  running: null,
});

/**
 * Decide what the session controls offer.
 *
 * Holding, the tab offers Release, and Stop while an action runs. Not
 * holding, it offers Open session when no one holds, Take over while someone
 * holds and no action runs; while an action runs it names the action and its
 * owner and offers Stop instead. Nothing is offered to open or take over
 * while such a request is already in flight.
 *
 * @param {SessionSnapshot | null} snapshot  The latest snapshot, if any.
 * @param {{ holding: boolean, pending?: boolean }} tab
 * @returns {SessionControls}
 */
export function sessionControls(snapshot, { holding, pending = false }) {
  if (snapshot == null) return { ...NOTHING, holding, pending };
  const { holder, action } = snapshot;
  /** @type {RunningView | null} */
  const running =
    action == null
      ? null
      : {
          name: action.name,
          owner: holder ? holder.owner : null,
          kind: holder ? holder.kind : null,
          stopping: action.stopping,
        };
  const stop = running != null;
  if (holding) {
    return { holding, pending, open: false, takeOver: false, release: true, stop, running };
  }
  return {
    holding,
    pending,
    open: !pending && holder == null,
    takeOver: !pending && holder != null && running == null,
    release: false,
    stop,
    running,
  };
}

/**
 * Whether *holder* is this owner's panel.
 * @param {HolderView | null | undefined} holder
 * @param {string} owner
 */
function isOwnPanel(holder, owner) {
  return holder != null && holder.kind === PANEL_KIND && holder.owner === owner;
}

/**
 * @typedef {object} Session
 * @property {() => string | null} clientId  The id to send, or `null`.
 * @property {() => string | null} owner     The owner the grant named.
 * @property {() => boolean} holding
 * @property {() => boolean} pending
 * @property {(request: 'open' | 'takeover') => void} begin
 *   An open or take-over POST is about to be sent.
 * @property {(grant: SessionGrant) => void} granted  Its answer arrived.
 * @property {() => void} failed  It was refused or failed in transport.
 * @property {() => void} released  The explicit Release succeeded.
 * @property {(code: string) => boolean} refused
 *   A call was refused with *code*; drops the id on a lost-session code.
 * @property {(type: string, payload: unknown) => void} onEvent
 *   Feed one stream event (`snapshot` and `taken_over` are read).
 * @property {(snapshot?: SessionSnapshot | null) => SessionControls} controls
 *   What the controls offer for *snapshot* (default: the latest one seen).
 */

/**
 * Create the holder state of one panel tab.
 * @param {{ onChange?: () => void }} [options]  Called whenever holding changes.
 * @returns {Session}
 */
export function createSession({ onChange } = {}) {
  /** @type {string | null} */
  let clientId = null;
  /** @type {string | null} */
  let owner = null;
  /** A snapshot received after the grant has shown this owner's panel holding. */
  let confirmed = false;
  /** @type {'open' | 'takeover' | null} */
  let request = null;
  /** `taken_over` events received while a take-over was in flight. */
  let unattributed = 0;
  /** This tab's own take-over event is still to come on the stream. */
  let expectOwnTakeOver = false;
  /** @type {SessionSnapshot | null} */
  let latest = null;

  function drop() {
    const had = clientId != null;
    clientId = null;
    owner = null;
    confirmed = false;
    expectOwnTakeOver = false;
    if (had && onChange) onChange();
  }

  /** @param {SessionSnapshot} snapshot */
  function onSnapshot(snapshot) {
    latest = snapshot;
    if (clientId == null || owner == null || request != null) return;
    if (isOwnPanel(snapshot.holder, owner)) {
      confirmed = true;
    } else if (confirmed) {
      drop();
    }
  }

  function onTakenOver() {
    if (request === 'takeover') {
      // Ours or another's, decided when the answer arrives.
      unattributed++;
      return;
    }
    if (expectOwnTakeOver) {
      expectOwnTakeOver = false;
      return;
    }
    drop();
  }

  return {
    clientId: () => clientId,
    owner: () => owner,
    holding: () => clientId != null,
    pending: () => request != null,

    begin(kind) {
      request = kind;
      unattributed = 0;
    },

    granted(grant) {
      const wasTakeOver = request === 'takeover';
      request = null;
      clientId = grant.client_id;
      owner = grant.owner;
      confirmed = false;
      // A take-over always replaces a holder, so it always publishes a
      // `taken_over` event; if none has arrived yet it is still to come.
      expectOwnTakeOver = wasTakeOver && unattributed === 0;
      unattributed = 0;
      if (onChange) onChange();
    },

    failed() {
      const missed = request === 'takeover' ? unattributed : 0;
      request = null;
      unattributed = 0;
      if (missed > 0) drop();
    },

    released() {
      drop();
    },

    refused(code) {
      if (!LOST_SESSION_CODES.includes(code)) return false;
      drop();
      return true;
    },

    onEvent(type, payload) {
      if (payload == null || typeof payload !== 'object') return;
      if (type === 'snapshot') {
        const snapshot = /** @type {{ snapshot?: SessionSnapshot }} */ (payload).snapshot;
        if (snapshot != null) onSnapshot(snapshot);
      } else if (type === 'taken_over') {
        onTakenOver();
      }
    },

    controls(snapshot = latest) {
      return sessionControls(snapshot, { holding: clientId != null, pending: request != null });
    },
  };
}
