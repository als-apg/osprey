// @ts-check
/**
 * Unit tests for the control-target derived facts (control-target-facts.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-facts.test.mjs
 *
 * The module is the pure half of the control-target vocabulary — what a row,
 * a confirm and the first-contact copy SAY about a machine, derived from the
 * state the chip publishes and nothing else. The popover suite drives the same
 * wording through a rendered panel; this file pins the tables themselves, so a
 * phrase can be changed (or drift into a wrong claim) with a failing test
 * naming the phrase rather than a DOM assertion three layers away.
 */
import { describe, expect, test } from 'vitest';

import {
  KIND_BANNERS,
  KIND_READ_PHRASES,
  KIND_WORDS,
  REASON_PHRASES,
  REASON_SWITCH_FAILED,
  STOP_NOT_TAKEN_SENTENCE,
  TURN_ON_SENTENCE,
  bannerNote,
  confirmSkipKeyBase,
  confirmSkippable,
  contextHolderWords,
  contextRefusalCode,
  contextRefusalPhrase,
  contextWritable,
  descriptor,
  descriptorTone,
  displayName,
  lockReason,
  machineKind,
  resolvePendingSwitch,
  statePhrase,
  switchConfirm,
  switchFailureNote,
  turnOnConfirm,
} from '../../../src/osprey/interfaces/web_terminal/static/js/control-target-facts.js';

/** The three machine kinds GET publishes as `machine_kind` (C-KIND). */
const KINDS = ['real', 'rehearsal', 'simulator'];

describe('KIND_READ_PHRASES', () => {
  test('covers exactly the kinds KIND_WORDS names', () => {
    // The two tables are read for the same chip state. A kind in one and not
    // the other renders a machine with a name and no capability sentence — or
    // worse, falls back to another kind's promise about the values.
    expect(Object.keys(KIND_READ_PHRASES).sort()).toEqual(Object.keys(KIND_WORDS).sort());
  });
});

/* ---- the kind vocabulary (C-KIND, C-WORDS) ------------------------------ */

describe('kind words', () => {
  test('every kind table is keyed on real | rehearsal | simulator', () => {
    // One vocabulary for the wire, the tables and the stylesheet: a table
    // still keyed on the old live/standin/va would name nothing.
    expect(Object.keys(KIND_WORDS).sort()).toEqual([...KINDS].sort());
    expect(Object.keys(KIND_READ_PHRASES).sort()).toEqual([...KINDS].sort());
    for (const kind of KINDS) expect(descriptor({}, kind)).not.toBe('');
  });

  test('the names and banners are the owner’s words', () => {
    expect(KIND_WORDS).toEqual({
      real: 'Real machine',
      rehearsal: 'Rehearsal',
      simulator: 'Simulator',
    });
    expect(KIND_BANNERS).toEqual({
      real: 'Real machine — writes move hardware',
      rehearsal: 'Rehearsal — nothing moves',
      simulator: 'Simulator — nothing moves',
    });
    expect(TURN_ON_SENTENCE).toBe(
      'Runs already going — a sandbox run or a notebook cell — keep the state they started with; the next one follows.'
    );
    expect(STOP_NOT_TAKEN_SENTENCE).toBe(
      'The stop did not take effect — writes are still on. Retry; if it fails again, ask the deployment admin.'
    );
  });

  test('machineKind reads machine_kind, never the target name or the label', () => {
    expect(machineKind({ machine_kind: 'rehearsal', target: 'standin' })).toBe('rehearsal');
    expect(machineKind({ machine_kind: 'simulator', target: 'va' })).toBe('simulator');
    // A `standin` no live server vouched for is the real machine: GET says so,
    // and nothing here second-guesses it from the name or the label.
    expect(
      machineKind({ machine_kind: 'real', target: 'standin', label: 'Rehearsal — nothing moves' })
    ).toBe('real');
    // No kind at all fails toward the loud answer.
    expect(machineKind({ target: 'standin', label: 'Rehearsal — nothing moves' })).toBe('real');
    expect(machineKind({ target: 'va', real_machine: false })).toBe('real');
    expect(machineKind({ machine_kind: 'live' })).toBe('real');
  });

  test('a non-live standin reads as the real machine, never "Rehearsal"', () => {
    const row = { target: 'standin', machine_kind: 'real', label: 'Rehearsal — nothing moves' };
    expect(displayName(row, machineKind(row))).toBe('Real machine');
  });

  test('only the real machine’s descriptor is a hazard', () => {
    expect(descriptorTone({}, 'real')).toBe('hazard');
    expect(descriptorTone({}, 'rehearsal')).toBeNull();
    expect(descriptorTone({}, 'simulator')).toBeNull();
    expect(descriptor({ reason: 'gateways_missing' }, 'real')).toBe('Not set up yet');
    expect(descriptorTone({ reason: 'gateways_missing' }, 'real')).toBeNull();
  });
});

/* ---- state words (C-GET) ------------------------------------------------- */

describe('statePhrase', () => {
  test('reads on / off / locked', () => {
    expect(statePhrase('on')).toBe('writes on');
    expect(statePhrase('off')).toBe('writes off');
    expect(statePhrase('locked')).toBe('writes locked');
  });

  test('no state word is "sandbox", and the old words promise nothing', () => {
    for (const word of ['on', 'off', 'locked', 'writes', 'sandbox', 'read-only', '']) {
      expect(statePhrase(word)).not.toMatch(/sandbox/i);
    }
    // The retired vocabulary falls to the word that promises nothing.
    expect(statePhrase('writes')).toBe('writes locked');
    expect(statePhrase('sandbox')).toBe('writes locked');
  });
});

/* ---- the confirms --------------------------------------------------------- */

/** @param {import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-facts.js').ConfirmRun[][]} body */
const textOf = (body) =>
  body
    .map((line) => line.map((run) => (typeof run === 'string' ? run : run.em)).join(''))
    .join('\n');

describe('turnOnConfirm', () => {
  test('is scoped to the login and carries the turn-on sentence', () => {
    const spec = turnOnConfirm({ target: 'standin', machine_kind: 'rehearsal' }, 'rehearsal');
    const text = textOf(spec.body);
    expect(text).not.toContain('every session on this deployment');
    expect(text).toContain('your login');
    expect(text).toContain(TURN_ON_SENTENCE);
    expect(spec.title).toBe('Turn writes on for Rehearsal?');
  });

  test('the rehearsal’s confirm says nothing moves, tagged for the amber tone', () => {
    const spec = turnOnConfirm({ target: 'standin' }, 'rehearsal');
    expect(spec.live).toBe('Rehearsal — nothing moves');
    expect(spec.targetKind).toBe('rehearsal');
  });

  test('the real machine’s confirm says writes move hardware, tagged red', () => {
    const spec = turnOnConfirm({ target: 'live' }, 'real');
    expect(spec.live).toBe('Real machine — writes move hardware');
    expect(spec.targetKind).toBe('real');
  });

  test('the simulator’s confirm carries no banner', () => {
    const spec = turnOnConfirm({ target: 'va' }, 'simulator');
    expect(spec.live).toBeNull();
    expect(spec.targetKind).toBe('simulator');
  });
});

describe('switchConfirm', () => {
  test('names the state it arrives in with the chip’s words', () => {
    const row = { target: 'live', endpoint: 'gw:5064' };
    expect(textOf(switchConfirm(row, 'real', 'on').body)).toContain('Writes are on there.');
    expect(textOf(switchConfirm(row, 'real', 'off').body)).toContain('Writes are off there');
    expect(textOf(switchConfirm(row, 'real', 'locked').body)).toContain('Writes are locked there.');
    expect(switchConfirm(row, 'real', 'on').live).toBe('Real machine — writes move hardware');
    expect(switchConfirm(row, 'real', 'off').live).toBeNull();
    expect(switchConfirm(row, 'real', 'on').targetKind).toBe('real');
  });
});

describe('confirmSkippable', () => {
  test('no real-machine turn-on is ever waivable', () => {
    expect(confirmSkippable('writes-on', 'real')).toBe(false);
    // Any kind this module does not know is the real machine — including
    // every word of the retired vocabulary.
    for (const kind of ['live', 'standin', 'va', '', 'something-new']) {
      expect(confirmSkippable('writes-on', kind)).toBe(false);
    }
  });

  test('the rehearsal and the simulator keep the waiver', () => {
    expect(confirmSkippable('writes-on', 'rehearsal')).toBe(true);
    expect(confirmSkippable('writes-on', 'simulator')).toBe(true);
  });

  test('a target switch may be waived on any machine', () => {
    for (const kind of KINDS) expect(confirmSkippable('switch', kind)).toBe(true);
  });

  test('a turn-on waiver is remembered per kind, so a demoted stand-in asks again', () => {
    const rehearsal = confirmSkipKeyBase('writes-on', {
      target: 'standin',
      machine_kind: 'rehearsal',
    });
    const real = confirmSkipKeyBase('writes-on', { target: 'standin', machine_kind: 'real' });
    expect(rehearsal).not.toBe(real);
    expect(confirmSkipKeyBase('switch', { target: 'standin', machine_kind: 'real' })).toBe(
      'osprey-ctc-skip-confirm:switch:standin'
    );
  });
});

/* ---- locks (C-GET) ------------------------------------------------------- */

describe('lockReason', () => {
  const owned = { kind: 'web_terminal', pid: 1000, port: 8080, self: true };
  /** @param {object} [o] */
  const stateOf = (o = {}) => ({
    store_available: true,
    owner: owned,
    readonly_run: false,
    locks: { readonly_run: false, tier: 'allowed' },
    ...o,
  });
  /** @param {object} [o] */
  const rowOf = (o = {}) => ({
    target: 'standin',
    machine_kind: 'rehearsal',
    switch: 'off',
    write_allowed: false,
    write_reason: 'switch_off',
    write_phrase: 'Writes are off; turn it on from the chip in the web terminal.',
    ...o,
  });

  test('an off switch and an on switch that writes are not locked', () => {
    expect(lockReason(rowOf(), stateOf())).toBeNull();
    expect(
      lockReason(
        rowOf({ switch: 'on', write_allowed: true, write_reason: null, write_phrase: null }),
        stateOf()
      )
    ).toBeNull();
  });

  test('a follower is not locked out of its own switch', () => {
    // C-POST: a follower turns its login's switch exactly as the holder does.
    for (const owner of [
      { kind: 'web_terminal', pid: 77, port: 8123, self: false },
      { kind: 'controls_server', pid: 5150, port: null, self: false },
    ]) {
      const state = stateOf({ owner });
      expect(lockReason(rowOf(), state)).toBeNull();
      expect(
        lockReason(rowOf({ switch: 'on', write_allowed: true, write_reason: null }), state)
      ).toBeNull();
      // ...while target switching stays the holder's.
      expect(contextWritable(state)).toBe(false);
    }
  });

  test('an on switch the verdict refuses shows the verdict’s phrase', () => {
    const phrase =
      "Writes are refused because this login's write tier does not allow them; ask the deployment admin.";
    expect(
      lockReason(
        rowOf({ switch: 'on', write_reason: 'tier_never', write_phrase: phrase }),
        stateOf()
      )
    ).toBe(phrase);
  });

  test('an unreadable switch is locked, never read as off', () => {
    const phrase =
      'Writes are refused because the write switch cannot be read; retry, then ask the deployment admin.';
    expect(
      lockReason(
        rowOf({ switch: null, write_reason: 'record_unavailable', write_phrase: phrase }),
        stateOf()
      )
    ).toBe(phrase);
    expect(lockReason(rowOf({ switch: null, write_phrase: null }), stateOf())).toBe(
      'the write switch cannot be read'
    );
  });

  test('the published locks hold an off switch off', () => {
    expect(lockReason(rowOf(), stateOf({ locks: { readonly_run: true, tier: 'allowed' } }))).toBe(
      'the whole deployment is running read-only'
    );
    expect(lockReason(rowOf(), stateOf({ locks: { readonly_run: false, tier: 'never' } }))).toBe(
      "this login's write tier does not allow writes"
    );
  });

  test('a store with nowhere to record is locked', () => {
    expect(lockReason(rowOf(), stateOf({ store_available: false }))).toBe(
      'changes cannot be recorded right now'
    );
  });

  test('reads no retired ceiling field', () => {
    // A retired column changes nothing: a row is locked by C-GET alone.
    expect(lockReason(rowOf({ ceiling_writes: false }), stateOf())).toBeNull();
  });
});

/* ---- has the switch this browser asked for landed? ---------------------- */

/**
 * The two halves of a switch, in the shape `GET /api/terminal/posture`
 * publishes them: the RECORD moved (top-level `generation`, plus a terminus
 * naming the request), and every live controls server then reports the
 * generation it arrived at (`servers[].applied_generation`). The operator is
 * waiting on the second.
 * @param {object} [o]
 */
const viewOf = (o = {}) => ({
  control_target: 'live',
  generation: 7,
  last_switch: null,
  servers: [],
  targets: [],
  ...o,
});

/** One live controls server's row. @param {object} [o] */
const serverOf = (o = {}) => ({
  pid: 4242,
  session: null,
  applied_target: 'live',
  applied_generation: 7,
  last_switch: null,
  last_posture_realign: null,
  updated_at: '2026-09-05T12:00:00+00:00',
  ...o,
});

/** The request this browser is waiting on. @param {object} [o] */
const pendingOf = (o = {}) => ({ requestId: 'r-mine', generation: 7, ...o });

describe('resolvePendingSwitch', () => {
  test('waits while a live server is still on the old generation', () => {
    const view = viewOf({
      servers: [serverOf({ applied_generation: 7 }), serverOf({ pid: 99, applied_generation: 6 })],
    });
    expect(resolvePendingSwitch(view, pendingOf())).toEqual({
      state: 'waiting',
      pid: null,
      detail: null,
    });
  });

  test('is applied once EVERY live server reports that generation', () => {
    const view = viewOf({ servers: [serverOf(), serverOf({ pid: 99 })] });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('applied');
  });

  test('a server further ahead than asked for still counts as arrived', () => {
    // Two switches in quick succession: the fleet is already at 9 while this
    // browser was waiting on 7. Requiring equality would wait forever.
    const view = viewOf({ generation: 9, servers: [serverOf({ applied_generation: 9 })] });
    expect(resolvePendingSwitch(view, pendingOf({ generation: 7 })).state).toBe('applied');
  });

  test('a server that has not got anywhere yet is not arrived', () => {
    // `applied_generation: null` is "this server has not got there", never
    // "baseline" — and never a number a comparison may treat as zero.
    const view = viewOf({ servers: [serverOf({ applied_generation: null })] });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('waiting');
  });

  test('a server holding no connector is not waited on', () => {
    // `children: []` says this server serves nothing: nothing it runs can
    // still touch the old target, and its child comes up on the record's
    // values whenever it launches. Its report is not owed — waiting on it
    // manufactured 30 s of `switching…` and a false expiry whenever a fresh
    // agent session had not made its first channel read yet.
    const view = viewOf({
      servers: [serverOf(), serverOf({ pid: 99, applied_generation: null, children: [] })],
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('applied');
  });

  test('a server whose child is still coming up IS waited on', () => {
    // A non-empty `children` is a connector mid-launch or mid-serve: this row
    // owes the fleet a report, and the wait is what makes `applied` mean the
    // old target is abandoned.
    const view = viewOf({
      servers: [serverOf(), serverOf({ pid: 99, applied_generation: null, children: [5001] })],
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('waiting');
  });

  test('a row without a children field keeps the conservative wait', () => {
    // Only an explicit empty list means "serving nothing". A row that does not
    // say — an older payload, a mangled report — is treated as it always was:
    // not arrived, still waited on.
    const view = viewOf({ servers: [serverOf(), serverOf({ pid: 99, applied_generation: null })] });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('waiting');
  });

  test('with every server childless, the record’s terminus decides', () => {
    // Nobody is left to converge — the same answer as an empty fleet, for the
    // same reason: each of these servers launches on the record's values.
    const terminus = { request_id: 'r-mine', status: 'applied', generation: 7 };
    const rows = [serverOf({ applied_generation: null, children: [] })];
    expect(
      resolvePendingSwitch(viewOf({ servers: rows, last_switch: terminus }), pendingOf()).state
    ).toBe('applied');
    expect(resolvePendingSwitch(viewOf({ servers: rows }), pendingOf()).state).toBe('waiting');
  });

  test('a childless server’s failed launch still ends the wait with its pid', () => {
    // A server with no child answers a record move by launching one; a launch
    // that does not come up files `failed` at the generation it was reaching
    // for. That verdict is this request's news even though the row holds no
    // connector to converge.
    const view = viewOf({
      servers: [
        serverOf(),
        serverOf({
          pid: 5150,
          applied_generation: null,
          children: [],
          last_switch: { status: 'failed', generation: 7, detail: 'spawn refused' },
        }),
      ],
    });
    expect(resolvePendingSwitch(view, pendingOf())).toEqual({
      state: 'failed',
      pid: 5150,
      detail: 'spawn refused',
    });
  });

  test('a boolean where a generation should be is not a number', () => {
    // `true >= 1` is true in JS. A row whose report was mangled into a boolean
    // must read as "has not got there", never as "arrived at generation 1" —
    // which would land a switch no server has applied.
    const view = viewOf({
      generation: 1,
      servers: [serverOf({ applied_generation: /** @type {any} */ (true) })],
    });
    expect(resolvePendingSwitch(view, pendingOf({ generation: 1 })).state).toBe('waiting');
  });

  test('one server failing that generation ends the wait and names its pid', () => {
    const view = viewOf({
      servers: [
        serverOf(),
        serverOf({
          pid: 5150,
          applied_generation: 6,
          last_switch: { status: 'failed', generation: 7, detail: 'gateway refused the bind' },
        }),
      ],
    });
    expect(resolvePendingSwitch(view, pendingOf())).toEqual({
      state: 'failed',
      pid: 5150,
      detail: 'gateway refused the bind',
    });
  });

  test('a failure about ANOTHER generation is not this request’s news', () => {
    // The server is reporting a swap the fleet has already moved past. It says
    // nothing about generation 7, and the row IS at 7, so this has landed.
    const view = viewOf({
      servers: [serverOf({ last_switch: { status: 'failed', generation: 5 } })],
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('applied');
  });

  test('a server still applying is a wait, not a failure', () => {
    const view = viewOf({
      servers: [
        serverOf({ applied_generation: 6, last_switch: { status: 'applying', generation: 7 } }),
      ],
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('waiting');
  });

  // A refusal moves neither target nor generation, which is exactly what
  // `generation: null` on the terminus says; any other verdict that is not
  // `applied` moved nothing either. No server will ever report one.
  test.each([
    { status: 'refused', reason: 'unreachable', generation: null },
    { status: 'failed', generation: 7 },
  ])('a $status verdict for this request is matched by request_id', (verdict) => {
    const view = viewOf({
      last_switch: { request_id: 'r-mine', ...verdict },
      servers: [serverOf({ applied_generation: 6 })],
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('answered');
  });

  test('another request’s refusal does not end this wait', () => {
    const view = viewOf({
      last_switch: { request_id: 'r-someone-else', status: 'refused', generation: null },
      servers: [serverOf({ applied_generation: 6 })],
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('waiting');
  });

  test('an APPLIED terminus for this request is not yet a landing', () => {
    // The record accepted the switch; the fleet has not followed it. Stopping
    // here would tell the operator the deployment is on a machine a controls
    // server has not reached.
    const view = viewOf({
      last_switch: { request_id: 'r-mine', status: 'applied', generation: 7 },
      servers: [serverOf({ applied_generation: 6 })],
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('waiting');
  });

  test('with no live server, the record’s applied terminus IS the landing', () => {
    // One controls server per agent session, so zero servers is normal: before
    // the first session, between sessions, or after the only one died. Nobody
    // is left to converge, and a server that starts later adopts the record's
    // generation on the way up. Waiting here would manufacture 30 s of
    // `switching…` and then a false expiry on every such switch.
    const view = viewOf({
      generation: 7,
      servers: [],
      last_switch: { request_id: 'r-mine', status: 'applied', generation: 7 },
    });
    expect(resolvePendingSwitch(view, pendingOf()).state).toBe('applied');
  });

  test('with no live server and no terminus, nobody has answered at all', () => {
    // `[].every()` is true, so the empty case is decided on the terminus and
    // never by the comparison — otherwise every switch would land instantly on
    // a deployment whose servers are all gone.
    expect(resolvePendingSwitch(viewOf({ servers: [] }), pendingOf()).state).toBe('waiting');
    // And a terminus for somebody else's request is not an answer either.
    const other = viewOf({
      servers: [],
      last_switch: { request_id: 'r-someone-else', status: 'applied', generation: 7 },
    });
    expect(resolvePendingSwitch(other, pendingOf()).state).toBe('waiting');
  });

  test('waits when the answer named no generation', () => {
    const view = viewOf({ servers: [serverOf()] });
    expect(resolvePendingSwitch(view, pendingOf({ generation: null })).state).toBe('waiting');
    expect(resolvePendingSwitch(view, pendingOf({ generation: undefined })).state).toBe('waiting');
  });

  test('answers waiting for the shapes a caller can hand it', () => {
    expect(resolvePendingSwitch(null, pendingOf()).state).toBe('waiting');
    expect(resolvePendingSwitch(viewOf(), null).state).toBe('waiting');
    expect(resolvePendingSwitch(viewOf(), { requestId: '', generation: 7 }).state).toBe('waiting');
    // A payload with no `servers` key at all reads as no live server.
    expect(resolvePendingSwitch({ generation: 7 }, pendingOf()).state).toBe('waiting');
  });
});

/* ---- would a change made here be written at all? ------------------------ */

describe('contextWritable', () => {
  const owned = { kind: 'web_terminal', pid: 1000, port: 8080, self: true };

  test('an absent owner key is "this payload does not say", not read-only', () => {
    // An older route must not turn every row read-only; the store decides.
    expect(contextWritable({ store_available: true })).toBe(true);
    expect(contextWritable(null)).toBe(false);
  });

  test('a controls-server owner is not called a terminal', () => {
    // The route publishes `controls_server` owners too, with `port: null` —
    // an agent's own process, not a page anybody can open. Naming a terminal
    // there sends the operator looking for something that does not exist.
    const agent = { kind: 'controls_server', pid: 5150, port: null, self: false };
    expect(contextHolderWords(agent)).toEqual({
      short: 'the controls server',
      long: 'The controls server (pid 5150) holds the control context.',
    });
    expect(bannerNote({ store_available: true, owner: agent })).toEqual({
      text: 'The controls server (pid 5150) holds the control context.',
      tone: 'warn',
    });
    // Target switching is the holder's; this login's own switch is not.
    expect(contextWritable({ store_available: true, owner: agent })).toBe(false);
    expect(
      lockReason(
        { switch: 'off', write_allowed: false, write_reason: 'switch_off' },
        { store_available: true, owner: agent }
      )
    ).toBeNull();
    expect(contextRefusalPhrase({ store_available: true, owner: agent })).toBe(
      'the controls server'
    );
  });

  test('an unrecognised kind is named as a process, never guessed at', () => {
    const odd = { kind: 'something_new', pid: 42, port: null, self: false };
    expect(contextHolderWords(odd)).toEqual({
      short: 'another process',
      long: 'Another process (pid 42) holds the control context.',
    });
  });

  test('the refusal code says WHICH of the two it was', () => {
    expect(contextRefusalCode({ store_available: true, owner: owned })).toBe('');
    expect(contextRefusalCode({ store_available: false, owner: owned })).toBe('store_unavailable');
    expect(contextRefusalCode({ store_available: true, owner: { ...owned, self: false } })).toBe(
      'context_owned_elsewhere'
    );
    // The code's own phrase is kind-neutral on purpose: who is holding it is
    // contextHolderWords' answer, not this map's.
    expect(REASON_PHRASES.context_owned_elsewhere).toBe('held elsewhere');
  });
});

describe('switchFailureNote', () => {
  test('says what happened when the server sent no sentence', () => {
    expect(switchFailureNote(5150)).toBe('Controls server pid 5150 did not apply the switch.');
    expect(switchFailureNote(5150, '   ')).toBe(
      'Controls server pid 5150 did not apply the switch.'
    );
  });

  test('still says what happened when the row named no pid', () => {
    expect(switchFailureNote(null)).toBe('A controls server did not apply the switch.');
  });

  test('the operator reads a phrase for it, not the code', () => {
    expect(REASON_PHRASES[REASON_SWITCH_FAILED]).toBe('not applied');
  });
});
