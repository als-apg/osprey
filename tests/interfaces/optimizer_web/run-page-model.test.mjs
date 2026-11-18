/**
 * Tests for the run page's pure words and facts (run-page-model.js): the
 * kind of a run screen, where a pause takes effect and what it is said to
 * do, the sentence of now and the line under it, the cancel confirmation,
 * the facts of a run and of a run that waits, where the knobs are, the rows
 * of the write approval (its expiry on the facility clock) and of the
 * settings.
 */

import { afterEach, describe, test, expect } from 'vitest';

import { FACILITY_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';
import { runningJob, storedConfig } from './_fixtures.mjs';
import {
  awaiting,
  granted,
  paused,
  pausedAfter,
  runAt,
  running,
  starting,
  waiting,
} from './_run.mjs';
import {
  authorizationScope,
  initialSamples,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';
import {
  approvalRows,
  cancelWords,
  firstTries,
  machineNow,
  methodLine,
  nowSentence,
  pausePhase,
  pauseWords,
  runFacts,
  runKind,
  settingsRows,
  waitFacts,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-page-model.js';
import { jobFigures } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-model.js';
import { STOP_WORDS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/facts-strip.js').Fact} Fact */

/** @param {Fact[]} facts @param {string} key */
const valueOf = (facts, key) => facts.find((f) => f.key === key)?.value ?? null;

/** @param {[string, string][]} rows */
const labels = (rows) => rows.map(([label]) => label);

const cancelling = () => runAt({ status: 'cancelling', write_authorization: granted() });

describe('the kind of a run screen', () => {
  test('waiting, running, paused, and cancelling while a cancel is under way', () => {
    expect(runKind(waiting().screen)).toBe('waiting');
    expect(runKind(starting().screen)).toBe('waiting');
    expect(runKind(running().screen)).toBe('running');
    expect(runKind(paused().screen)).toBe('paused');
    expect(runKind(cancelling().screen)).toBe('cancelling');
  });
});

describe('first tries and a pause', () => {
  test('the first tries are the planned total less the iterations, else what the config resolves to', () => {
    const config = storedConfig();
    expect(firstTries(null, null)).toBeNull();
    expect(firstTries(pausedAfter(0).screen.figures, config)).toBe(3);
    // No total, or one below the iterations: the config's own count.
    expect(firstTries(null, config)).toBe(initialSamples(config).count);
    const short = jobFigures(runningJob({ total_evaluations: 10 }), config);
    expect(firstTries(short, config)).toBe(initialSamples(config).count);
  });

  test('a pause takes effect where the runner checks for it', () => {
    // Three first tries: no check between them; one before each guided step and the final write.
    expect(pausePhase(pausedAfter(0).screen)).toBe('first');
    expect(pausePhase(pausedAfter(2).screen)).toBe('first');
    expect(pausePhase(pausedAfter(3).screen)).toBe('step');
    expect(pausePhase(pausedAfter(33).screen)).toBe('end');
    // Without the plan the page does not claim either.
    const unplanned = { ...pausedAfter(2).screen, config: null };
    expect(pausePhase(unplanned)).toBeNull();
  });

  test('while first tries remain, the pause is described as not yet holding', () => {
    const seeding = pausedAfter(2).screen;
    const guided = pausedAfter(5).screen;
    expect(pauseWords(seeding).held).not.toBe(pauseWords(guided).held);
    expect(pauseWords(seeding).ask).not.toBe(pauseWords(guided).ask);
    expect(methodLine(seeding)).not.toBe(methodLine(guided));
    expect(nowSentence(seeding)).not.toBe(nowSentence(guided));
  });

  test('a pause at the end says something else when the end action holds the knobs', () => {
    const done = pausedAfter(33).screen;
    const holding = { ...done, config: storedConfig({ on_complete: 'hold' }) };
    expect(pauseWords(holding).held).not.toBe(pauseWords(done).held);
  });
});

describe('the header sentences', () => {
  test('the header names the evaluation, never a step inside it', () => {
    expect(nowSentence(running().screen)).toMatch(/\b4\b.*\b10\b/);
    expect(nowSentence(waiting().screen)).toContain('carol');
    const done = runAt({
      write_authorization: granted(),
      data: Array.from({ length: 10 }, () => ({ phase_a: 0, sol_b: 3, charge: 1 })),
    }).screen;
    expect(nowSentence(done)).not.toBe(nowSentence(running().screen));
  });

  test('the line under it follows the first tries, then the method', () => {
    // UCB with two variables: three first tries, then 30 iterations planned.
    const lines = [0, 1, 3].map((n) =>
      methodLine(
        runAt({
          write_authorization: granted(),
          total_evaluations: 33,
          data: runningJob().data.slice(0, n),
        }).screen
      )
    );
    expect(new Set(lines).size).toBe(3);
    expect(lines[1]).toContain('2 of 3');
    expect(lines[2]).toContain('1 of 30');
  });

  test('a cancelling run says what a cancel leaves; a running run with no config has no line', () => {
    expect(methodLine(cancelling().screen)).toBe(STOP_WORDS.now);
    expect(methodLine({ ...running().screen, config: null })).toBeNull();
  });
});

describe('the cancel confirmation', () => {
  test('a run with no request yet is cancelled; a waiting request is withdrawn', () => {
    expect(cancelWords(starting().screen).title).not.toBe(cancelWords(waiting().screen).title);
  });

  test('a run under way is told what a cancel leaves; a run that waits, that nothing was written', () => {
    expect(cancelWords(running().screen).text).toBe(STOP_WORDS.now);
    expect(cancelWords(paused().screen).text).toBe(STOP_WORDS.now);
    expect(cancelWords(waiting().screen).text).not.toBe(STOP_WORDS.now);
    // Keeping a paused run is not keeping it running.
    expect(cancelWords(paused().screen).keep).not.toBe(cancelWords(running().screen).keep);
  });
});

describe('the facts', () => {
  test('time left shows only when the figures have it', () => {
    const { screen } = running();
    expect(screen.figures?.minutesLeft).toBeNull();
    const keys = runFacts(screen, 'nC')
      .filter((f) => f.value != null)
      .map((f) => f.key);
    expect(keys).not.toContain('left');
    expect(keys).toContain('best');
  });

  test('a paused run counts the evaluations left; no figures, no facts', () => {
    const { screen } = pausedAfter(5);
    expect(valueOf(runFacts(screen, ''), 'left')).toContain('28');
    expect(valueOf(runFacts(screen, ''), 'evaluations')).toMatch(/\b5\b.*\b33\b/);
    expect(runFacts({ ...screen, figures: null }, '')).toEqual([]);
  });

  test('the best carries the units of the objective', () => {
    const { screen } = running();
    expect(valueOf(runFacts(screen, 'nC'), 'best')).toMatch(/nC$/);
    expect(valueOf(runFacts(screen, ''), 'best')).toMatch(/\d$/);
  });

  test('a waiting run names who approves (this session marked), where only while it waits for a person, and the writes asked', () => {
    const scope = authorizationScope(awaiting());
    const mine = waitFacts(scope, 'carol');
    const theirs = waitFacts(scope, 'dave');
    expect(valueOf(theirs, 'approver')).toBe('carol');
    expect(valueOf(mine, 'approver')).not.toBe(valueOf(theirs, 'approver'));
    expect(valueOf(mine, 'approver')).toContain('carol');
    expect(valueOf(mine, 'where')).not.toBeNull();
    expect(valueOf(mine, 'steps')).toBe(11);
    const requesting = waitFacts(authorizationScope({ ...awaiting(), state: 'requesting' }), null);
    expect(valueOf(requesting, 'where')).toBeNull();
    expect(waitFacts(null, 'carol').every((f) => f.value === null)).toBe(true);
  });
});

describe('where the knobs are', () => {
  test('nothing written while waiting; the last evaluation recorded while running; what a pause holds while paused', () => {
    expect(machineNow(waiting().screen)).not.toBe(machineNow(running().screen));
    expect(machineNow(running().screen)).toMatch(/\b3\b/);
    const held = pausedAfter(5).screen;
    expect(machineNow(held).endsWith(pauseWords(held).held)).toBe(true);
    expect(machineNow(cancelling().screen)).toBe(STOP_WORDS.now);
    expect(machineNow(starting().screen)).toBe(machineNow(waiting().screen));
  });
});

describe('the rows of the write approval', () => {
  afterEach(() => stampFacilityZone(null));

  test('none without a write authorization', () => {
    expect(approvalRows(null)).toEqual([]);
  });

  test('a request that waits names who approves and where; a granted one, who owns it and where it was approved', () => {
    const asked = approvalRows(authorizationScope(awaiting()), { sessionOwner: 'dave' });
    expect(labels(asked)).toEqual(['State', 'Approves', 'Where', 'Writes asked', 'Request']);
    const allowed = approvalRows(authorizationScope(granted()), { sessionOwner: 'dave' });
    expect(labels(allowed)).toEqual([
      'State',
      'Owner',
      'Approved from',
      'Writes allowed',
      'Valid until',
      'Authorization',
    ]);
  });

  test('the owner who is this session is marked; an ended authorization gives its reason', () => {
    const scope = authorizationScope(awaiting());
    const mine = approvalRows(scope, { sessionOwner: 'carol' });
    const theirs = approvalRows(scope, { sessionOwner: 'dave' });
    expect(mine[1][1]).not.toBe(theirs[1][1]);
    const ended = authorizationScope({ ...granted(), state: 'ended' });
    expect(approvalRows(ended, { reason: 'auth_expired' })[0][1]).toContain('auth_expired');
    expect(approvalRows(ended)[0][1]).not.toContain('auth_expired');
  });

  test('the writes say when one of them is the final write', () => {
    const final = approvalRows(authorizationScope(awaiting()));
    const none = approvalRows(authorizationScope({ ...awaiting(), completion: 'none' }));
    expect(final[3][1]).not.toBe(none[3][1]);
    expect(none[3][1]).toBe('11');
  });

  test('the expiry is read on the facility clock: its time today, its date and time on another day', () => {
    stampFacilityZone(FACILITY_ZONE);
    // Both instants fall on one calendar day in every candidate facility zone.
    const now = Date.parse('2026-09-19T12:00:00Z');
    /** @param {string} iso */
    const until = (iso) => {
      const scope = authorizationScope({ ...granted(), expires_at_ns: Date.parse(iso) * 1e6 });
      return approvalRows(scope, { now }).find(([label]) => label === 'Valid until')?.[1];
    };
    /** @param {Intl.DateTimeFormatOptions} options @param {string} iso */
    const facility = (options, iso) =>
      new Intl.DateTimeFormat(undefined, { ...options, timeZone: FACILITY_ZONE }).format(
        new Date(iso)
      );
    const clock = { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' };
    expect(until('2026-09-19T13:00:00Z')).toBe(
      facility(/** @type {Intl.DateTimeFormatOptions} */ (clock), '2026-09-19T13:00:00Z')
    );
    expect(until('2026-09-21T13:00:00Z')).toBe(
      facility(
        /** @type {Intl.DateTimeFormatOptions} */ ({ day: 'numeric', month: 'short', ...clock }),
        '2026-09-21T13:00:00Z'
      )
    );
  });
});

describe('the rows of the settings', () => {
  test('none without a config; the evaluations split into first tries and iterations', () => {
    expect(settingsRows(null)).toEqual([]);
    const { screen } = pausedAfter(5);
    const rows = settingsRows(screen.config, { figures: screen.figures });
    const evaluations = rows.find(([label]) => label === 'Evaluations')?.[1];
    expect(evaluations).toMatch(/\b33\b.*\b3\b.*\b30\b/);
  });

  test('a value the run does not have is left out; a revision is cut to ten characters', () => {
    const config = storedConfig({ control_target: null, variables: [] });
    const rows = settingsRows(config, { revision: 'rev-abcdef12345' });
    expect(labels(rows)).not.toContain('Control target');
    expect(labels(rows)).not.toContain('Knobs');
    expect(labels(rows)).not.toContain('Evaluations');
    expect(labels(rows)).not.toContain('Objective');
    expect(rows.find(([label]) => label === 'Revision')?.[1]).toBe('rev-abcdef');
  });

  test('the objective is named with the value it looks for', () => {
    const { screen } = running();
    const rows = settingsRows(screen.config, { figures: screen.figures, objective: 'Charge' });
    const objective = rows.find(([label]) => label === 'Objective')?.[1];
    expect(objective?.startsWith('Charge, ')).toBe(true);
  });
});
