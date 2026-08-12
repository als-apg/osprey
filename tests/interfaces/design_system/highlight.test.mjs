/**
 * Unit tests for the design-system agent-activity flash helper
 * (static/js/highlight.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/design_system/highlight.test.mjs
 *
 * happy-dom does not run CSS animations, so `animationend` is dispatched
 * synthetically; the helper tolerates events without an `animationName`
 * (and ignores mismatching names / bubbled child targets).
 */

import { readFileSync } from 'node:fs';

import { test, expect, describe, beforeEach } from 'vitest';

import { flashElement } from '../../../src/osprey/interfaces/design_system/static/js/highlight.js';

/** @param {string} rel @returns {string} */
const readSource = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');

const CSS = readSource('../../../src/osprey/interfaces/design_system/static/css/highlight.css');
const JS = readSource('../../../src/osprey/interfaces/design_system/static/js/highlight.js');

/**
 * The body of the first at-rule whose prelude matches `pattern`, brace-matched
 * so nested rules (the media query's own @keyframes) come along intact.
 *
 * @param {string} css
 * @param {RegExp} pattern
 * @returns {string}
 */
function atRuleBody(css, pattern) {
  const start = css.search(pattern);
  if (start === -1) throw new Error(`no at-rule matching ${pattern} in stylesheet`);
  const open = css.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < css.length; i += 1) {
    if (css[i] === '{') depth += 1;
    else if (css[i] === '}' && (depth -= 1) === 0) return css.slice(open + 1, i);
  }
  throw new Error(`unbalanced braces after ${pattern}`);
}

/** @type {HTMLElement} */
let el;

beforeEach(() => {
  document.body.replaceChildren();
  el = document.createElement('div');
  document.body.appendChild(el);
});

describe('flashElement', () => {
  test('adds the agent-flash class', () => {
    flashElement(el);

    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('reflow-restart: a second call while still flashing leaves the class applied', () => {
    flashElement(el);
    flashElement(el);

    // The remove -> offsetWidth -> add cycle must land back on "applied";
    // a naive second `add` no-op and the restart are indistinguishable by
    // class list alone, so this guards the invariant that restarting never
    // strands the element without the class.
    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('removes the class on animationend (self-cleaning)', () => {
    flashElement(el);

    el.dispatchEvent(new Event('animationend'));

    expect(el.classList.contains('agent-flash')).toBe(false);
  });

  test('still self-cleans after a reflow-restart', () => {
    flashElement(el);
    flashElement(el);

    el.dispatchEvent(new Event('animationend'));

    expect(el.classList.contains('agent-flash')).toBe(false);
  });

  test('ignores animationend for a different animation name', () => {
    flashElement(el);

    const ev = new Event('animationend');
    // Plain Event so happy-dom does not need AnimationEvent support.
    Object.defineProperty(ev, 'animationName', { value: 'some-other-anim' });
    el.dispatchEvent(ev);

    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('ignores animationend bubbling from a child element', () => {
    const child = document.createElement('span');
    el.appendChild(child);
    flashElement(el);

    child.dispatchEvent(new Event('animationend', { bubbles: true }));

    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('can flash again after a completed cycle', () => {
    flashElement(el);
    el.dispatchEvent(new Event('animationend'));

    flashElement(el);

    expect(el.classList.contains('agent-flash')).toBe(true);
  });
});

/**
 * Static assertions on css/highlight.css. happy-dom runs no animations, so the
 * reduced-motion path cannot be exercised behaviorally — but its failure mode
 * is a stylesheet property, not a runtime one: any reduced-motion rule that
 * stops `animationend` from firing strands `.agent-flash` on the element
 * permanently, because the cleanup above is the only thing that removes it.
 */
describe('reduced-motion flash contract', () => {
  const reducedMotion = atRuleBody(CSS, /@media\s*\(prefers-reduced-motion:\s*reduce\)/);

  test('does not suppress the animation outright', () => {
    // `animation: none` is the trap: no animation means no animationend, so
    // flashElement's listener never fires and the class leaks forever.
    expect(reducedMotion).not.toMatch(/animation\s*:\s*none/);
  });

  test('animates the same keyframes name the helper cleans up on', () => {
    const guarded = /const FLASH_ANIMATION = '([^']+)'/.exec(JS)?.[1];
    expect(guarded).toBeTruthy();

    // flashElement ignores animationend for any other name, so a renamed
    // override would strand the class exactly as `animation: none` did.
    expect(reducedMotion).toMatch(new RegExp(`@keyframes\\s+${guarded}\\b`));
    expect(reducedMotion).toMatch(new RegExp(`animation:\\s*${guarded}\\s`));
  });

  test('holds a still ring rather than moving one', () => {
    expect(reducedMotion).toMatch(/animation:[^;]*\bsteps\(/);

    const keyframes = atRuleBody(reducedMotion, /@keyframes/);
    const shadows = [...keyframes.matchAll(/box-shadow:\s*([^;]+)/g)].map((m) => m[1].trim());
    expect(shadows.length).toBeGreaterThan(0);
    expect(new Set(shadows).size).toBe(1); // constant across every offset
  });

  test('takes its ring color from a design token', () => {
    const keyframes = atRuleBody(reducedMotion, /@keyframes/);

    expect(keyframes).toMatch(/box-shadow:[^;]*var\(--accent-tint-25\)/);
    expect(keyframes).not.toMatch(/#[0-9a-f]{3,8}\b|rgba?\(|hsla?\(/i);
  });
});
