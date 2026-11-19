/**
 * Unit tests for the knob bar (knob-bar.js): where it puts each drawing for a
 * given range, which drawings it leaves out, and how it names a knob.
 */

import { describe, test, expect } from 'vitest';

import {
  KNOB_KEYS,
  knobBar,
  knobLabel,
  knobLegend,
  knobName,
  position,
  updateKnobBar,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/knob-bar.js';

/**
 * The left offset (percent) of each drawing on a bar, by its mark.
 *
 * @param {HTMLElement} bar
 * @returns {Record<string, number>}
 */
function marks(bar) {
  return Object.fromEntries(
    Array.from(bar.querySelectorAll('[data-mark]')).map((node) => [
      /** @type {HTMLElement} */ (node).dataset.mark,
      parseFloat(/** @type {HTMLElement} */ (node).style.left),
    ])
  );
}

describe('position', () => {
  test('is the share of the track, clamped to it', () => {
    expect(position(5, 0, 10)).toBe(50);
    expect(position(-3, 0, 10)).toBe(0);
    expect(position(42, 0, 10)).toBe(100);
  });

  test('an empty track puts every value in the middle; no value has no position', () => {
    expect(position(1, 2, 2)).toBe(50);
    expect(position(null, 0, 1)).toBeNull();
    expect(position(Number.NaN, 0, 1)).toBeNull();
  });
});

describe('knobBar', () => {
  test('draws the window, the value before the run and each mark it is given', () => {
    const bar = knobBar({
      lo: -10,
      hi: 10,
      window: { min: -5, max: 5 },
      now: 0,
      marks: { start: -10, best: 5, current: 10 },
      units: 'A',
      label: 'phase',
    });
    expect(marks(bar)).toEqual({ window: 25, now: 50, start: 0, best: 75, current: 100 });
    const band = /** @type {HTMLElement} */ (bar.querySelector('[data-mark="window"]'));
    expect(parseFloat(band.style.width)).toBe(50);
    expect(bar.getAttribute('aria-label')).toBe('phase');
    expect(bar.querySelector('.opt-kbar-ends')?.textContent).toBe('-1010\u00a0A');
  });

  test('a value it does not have is not drawn', () => {
    const bar = knobBar({ lo: 0, hi: 1, marks: { start: null, best: undefined } });
    expect(marks(bar)).toEqual({});
  });

  test('a window with a missing or inverted bound is not drawn', () => {
    expect(marks(knobBar({ lo: 0, hi: 1, window: { min: 0.2, max: null } }))).toEqual({});
    expect(marks(knobBar({ lo: 0, hi: 1, window: { min: 0.8, max: 0.2 } }))).toEqual({});
  });

  test('a problem or a clamped window says so in its class', () => {
    const bar = knobBar({ lo: 0, hi: 1, window: { min: 0, max: 1 }, tone: 'clamped' });
    expect(bar.querySelector('[data-mark="window"]')?.classList.contains('is-clamped')).toBe(true);
  });

  test('ends can be left off', () => {
    expect(knobBar({ lo: 0, hi: 1, ends: false }).querySelector('.opt-kbar-ends')).toBeNull();
  });

  test('updateKnobBar moves the marks in place', () => {
    const bar = knobBar({ lo: 0, hi: 10, marks: { current: 1 } });
    const track = bar.querySelector('.opt-kbar-track');
    updateKnobBar(bar, { lo: 0, hi: 10, marks: { current: 9, best: 5 }, label: 'now 9' });
    expect(bar.querySelector('.opt-kbar-track')).toBe(track);
    expect(marks(bar)).toEqual({ best: 50, current: 90 });
    expect(bar.getAttribute('aria-label')).toBe('now 9');
  });
});

describe('the name of a knob', () => {
  test('its description first, its identifier second', () => {
    expect(knobLabel('LN:MASTPH', { description: 'Linac master phase' })).toEqual({
      title: 'Linac master phase',
      channel: 'LN:MASTPH',
    });
    const node = knobName('LN:MASTPH', { description: 'Linac master phase' });
    expect(node.querySelector('.opt-knob-title')?.textContent).toBe('Linac master phase');
    expect(node.querySelector('.opt-knob-channel')?.textContent).toBe('LN:MASTPH');
  });

  test('no description: the identifier is the only line', () => {
    expect(knobLabel('SR:VCM:01', { description: '  ' })).toEqual({
      title: 'SR:VCM:01',
      channel: null,
    });
    expect(knobLabel('SR:VCM:01', null)).toEqual({ title: 'SR:VCM:01', channel: null });
    expect(knobName('SR:VCM:01', undefined).querySelector('.opt-knob-channel')).toBeNull();
  });
});

test('the legend names the drawings it is asked for, in order', () => {
  const legend = knobLegend(['start', 'best', 'current']);
  expect(Array.from(legend.children).map((n) => n.textContent)).toEqual(
    /** @type {(keyof typeof KNOB_KEYS)[]} */ (['start', 'best', 'current']).map(
      (k) => KNOB_KEYS[k]
    )
  );
});

test('the last evaluation recorded has its own key, apart from what is on the machine', () => {
  const legend = knobLegend(['recorded']);
  expect(legend.textContent).toBe(KNOB_KEYS.recorded);
  expect(KNOB_KEYS.recorded).not.toBe(KNOB_KEYS.current);
});
