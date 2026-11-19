/**
 * Tests for the panel's shared words (words.js): every method and end action
 * the service knows has a name of its own, an ended job has one word and one
 * tone on every surface, a direction reads the same everywhere, a change
 * carries its sign and keeps its size, and an instant reads on the facility
 * clock with its date when it is not of the facility's today.
 */

import { afterEach, beforeEach, describe, test, expect } from 'vitest';

import { FACILITY_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';
import { ALGORITHMS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';
import { ON_COMPLETE_OPTIONS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';
import {
  BEFORE_A_WRITE,
  ENDINGS,
  SCREEN_STATUS,
  STOP_WORDS,
  changeText,
  directionWord,
  directionWords,
  endWords,
  endedChip,
  endedDot,
  endedWord,
  endingOf,
  instantText,
  methodLabel,
  percent,
  signed,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

describe('methods and end actions', () => {
  test('every method the service knows has a name of its own', () => {
    const names = ALGORITHMS.map(methodLabel);
    ALGORITHMS.forEach((id, i) => expect(names[i]).not.toBe(id));
    expect(new Set(names).size).toBe(ALGORITHMS.length);
  });

  test('a config with no method names the service default; an unknown method keeps its id', () => {
    expect(methodLabel(null)).toBe(methodLabel('random'));
    expect(methodLabel('new_method')).toBe('new_method');
  });

  test('every end action has a short name and a sentence of its own', () => {
    const words = ON_COMPLETE_OPTIONS.map(endWords);
    for (const w of words) {
      expect(w.short).not.toBe('');
      expect(w.sentence).not.toBe('');
    }
    expect(new Set(words.map((w) => w.short)).size).toBe(ON_COMPLETE_OPTIONS.length);
    expect(new Set(words.map((w) => w.sentence)).size).toBe(ON_COMPLETE_OPTIONS.length);
  });

  test('no end action is the service default; an unknown one keeps its id and says nothing', () => {
    expect(endWords(null)).toEqual(endWords('apply_best'));
    expect(endWords('later')).toEqual({ short: 'later', sentence: '' });
  });

  test('what a stop leaves is said before, during and after it, in different tenses', () => {
    const said = [STOP_WORDS.ahead, STOP_WORDS.now, STOP_WORDS.after];
    expect(new Set(said).size).toBe(3);
    // A stop restores nothing, and a run writes only once it holds an authorization.
    expect(BEFORE_A_WRITE).not.toBe('');
  });
});

describe('how a job ended', () => {
  test('only an ended status has an ending; each names its own outcome', () => {
    expect(endingOf('completed')?.kind).toBe('completed');
    expect(endingOf('error')?.kind).toBe('failed');
    expect(endingOf('cancelled')?.kind).toBe('cancelled');
    for (const status of ['created', 'running', 'paused', 'cancelling', '', null, undefined]) {
      expect(endingOf(status)).toBeNull();
    }
  });

  test('a run that stopped names the evaluation it stopped at; a completed run does not', () => {
    const [completed, failed, cancelled] = ['completed', 'error', 'cancelled'].map(
      (s) =>
        /** @type {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js').Ending} */ (
          endingOf(s)
        )
    );
    expect(endedWord(failed, 7)).toContain('7');
    expect(endedWord(cancelled, 7)).toContain('7');
    expect(endedWord(completed, 7)).not.toContain('7');
    expect(endedWord(failed, 0)).toBe(endedWord(failed));
  });

  test('an ending has one tone: the toolbar chip, the row chip and the status dot follow it', () => {
    const tones = new Set();
    for (const ending of Object.values(ENDINGS)) {
      const chip = endedChip(ending);
      expect(SCREEN_STATUS[ending.kind].tone).toBe(chip.tone);
      expect(SCREEN_STATUS[ending.kind].label).toBe(chip.label);
      expect(endedDot(ending).label).toBe(chip.label);
      tones.add(chip.tone);
    }
    // Completed, failed and cancelled are told apart by their tone.
    expect(tones.size).toBe(3);
  });

  test('each screen of a job has its own label and its own short word', () => {
    const entries = Object.values(SCREEN_STATUS);
    expect(new Set(entries.map((e) => e.label)).size).toBe(entries.length);
    expect(new Set(entries.map((e) => e.short)).size).toBe(entries.length);
  });
});

describe('direction', () => {
  test('the two directions read differently, in any case; an unknown one has no words', () => {
    const up = directionWords('MAXIMIZE');
    const down = directionWords('minimize');
    expect(up && down).toBeTruthy();
    expect(up?.word).not.toBe(down?.word);
    expect(up?.seek).not.toBe(down?.seek);
    expect(up?.better).not.toBe(down?.better);
    expect(directionWords('sideways')).toBeNull();
    expect(directionWords(null)).toBeNull();
  });

  test('a direction name passes an unknown direction through', () => {
    expect(directionWord('MAXIMIZE')).toBe(directionWords('MAXIMIZE')?.word);
    expect(directionWord('sideways')).toBe('sideways');
    expect(directionWord(null)).toBeNull();
  });
});

describe('a change', () => {
  test('a signed number carries its sign and its size; nothing is no text', () => {
    expect(signed(0.358)).toMatch(/^\+0\.358$/);
    expect(signed(-0.166)).toMatch(/^\u22120\.166$/);
    expect(signed(0)).toBe(signed(-0));
    expect(signed(0)).not.toMatch(/^[+\u2212]/);
    expect(signed(1e-9)).toMatch(/^\+/);
    expect(signed(-1234567.4)).toBe('\u22121234567');
    expect(signed(null)).toBe('');
    expect(signed(Number.NaN)).toBe('');
  });

  test('a percentage has one decimal below 10 % and none above', () => {
    expect(percent(4.24)).toMatch(/^\+4\.2\s%$/);
    expect(percent(-61.4)).toMatch(/^\u221261\s%$/);
    expect(percent(9.96)).toMatch(/^\+10\s%$/);
    expect(percent(0.01)).toBe(percent(0));
    expect(percent(null)).toBe('');
  });

  test('a change against the value before: a percentage, else the difference when that value is zero', () => {
    expect(changeText({ change: 2, changePct: 40, before: 5 })).toBe(percent(40));
    expect(changeText({ change: 2, changePct: null, before: 0 })).toBe(signed(2));
    expect(changeText({ change: null, changePct: null, before: 5 })).toBeNull();
    expect(changeText({ change: 2, changePct: null, before: null })).toBeNull();
  });
});

describe('an instant', () => {
  beforeEach(() => stampFacilityZone(FACILITY_ZONE));
  afterEach(() => stampFacilityZone(null));

  /**
   * *iso* in *zone* with *options*, straight from Intl.
   * @param {string} zone @param {Intl.DateTimeFormatOptions} options @param {string} iso
   */
  const inZone = (zone, options, iso) =>
    new Intl.DateTimeFormat(undefined, { ...options, timeZone: zone }).format(new Date(iso));
  /** @type {Intl.DateTimeFormatOptions} */
  const CLOCK = { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' };
  /** @type {Intl.DateTimeFormatOptions} */
  const DATED = { day: 'numeric', month: 'short', ...CLOCK };

  test("an instant of the facility's today shows the facility clock alone", () => {
    // Both instants fall on one calendar day in every candidate facility zone.
    const now = Date.parse('2026-09-19T21:30:00Z');
    expect(instantText('2026-09-19T21:05:00Z', now)).toBe(
      inZone(FACILITY_ZONE, CLOCK, '2026-09-19T21:05:00Z')
    );
    // Epoch milliseconds read the same as the ISO time.
    expect(instantText(Date.parse('2026-09-19T21:05:00Z'), now)).toBe(
      instantText('2026-09-19T21:05:00Z', now)
    );
  });

  test('an instant of another facility day carries its date', () => {
    const now = Date.parse('2026-09-19T21:30:00Z');
    expect(instantText('2026-09-18T21:05:00Z', now)).toBe(
      inZone(FACILITY_ZONE, DATED, '2026-09-18T21:05:00Z')
    );
    expect(instantText('2026-09-21T21:05:00Z', now)).toBe(
      inZone(FACILITY_ZONE, DATED, '2026-09-21T21:05:00Z')
    );
  });

  test('"today" is the facility calendar\'s day', () => {
    stampFacilityZone('Asia/Tokyo');
    // 00:30 on the 20th in Tokyo.
    const now = Date.parse('2026-09-19T15:30:00Z');
    // 00:10 on the 20th: today.
    expect(instantText('2026-09-19T15:10:00Z', now)).toBe(
      inZone('Asia/Tokyo', CLOCK, '2026-09-19T15:10:00Z')
    );
    // 23:30 on the 19th, the same UTC day as now: dated.
    expect(instantText('2026-09-19T14:30:00Z', now)).toBe(
      inZone('Asia/Tokyo', DATED, '2026-09-19T14:30:00Z')
    );
  });

  test('no instant, or one that cannot be read, gives no text', () => {
    const now = Date.parse('2026-09-19T21:30:00Z');
    expect(instantText(null, now)).toBe('');
    expect(instantText('', now)).toBe('');
    expect(instantText('soon', now)).toBe('');
  });
});
