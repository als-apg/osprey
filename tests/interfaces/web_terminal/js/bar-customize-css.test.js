/**
 * Bar customize — the one CSS invariant a unit suite can actually hold:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-customize-css.test.js
 *
 * EDIT MODE MUST NOT MAKE A BAR LOOK CROWDED. Its decorations — the dashed
 * outline and the lock badge — are drawn OUTSIDE the item's own box, so on the
 * last item in a bar they hang past the host's content edge, and an absolutely
 * positioned descendant hanging over that edge is real scrollable overflow. The
 * overflow ladder's probe is `scrollWidth - clientWidth`, so a bar whose
 * trailing item is decorated reported crowding it did not have and folded its
 * lowest-priority item on the next reconcile: on the shipped default header,
 * whose last item is the locked `display`, that was every drop into the header.
 *
 * The fix is a reservation on the host, and the thing that can silently undo it
 * is a decoration added later that reaches further out than the reservation. No
 * layout engine runs here, so this reads the stylesheet and pins that
 * arithmetic. The browser lane proves the measurement itself.
 */

import { test, expect, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
// Explicit, because these suites run with the BROWSER globals declared.
import process from 'node:process';

// From the vitest root (the repository), because `import.meta.url` is not a
// file URL under the happy-dom environment.
const CSS = readFileSync(
  resolve(process.cwd(), 'src/osprey/interfaces/web_terminal/static/css/bars.css'),
  'utf8'
);

/**
 * The stylesheet's rules as `{selector, body}`. Flat CSS only, which bars.css
 * is — there is no nesting and no at-rule in it.
 * @returns {{selector: string, body: string}[]}
 */
function rules() {
  return CSS.replace(/\/\*[\s\S]*?\*\//g, '')
    .split('}')
    .map((chunk) => chunk.split('{'))
    .filter((parts) => parts.length === 2)
    .map(([selector, body]) => ({ selector: selector.trim(), body }));
}

/** The declared value of a custom property, in px. @param {string} name */
function pxToken(name) {
  const match = CSS.match(new RegExp(`${name}\\s*:\\s*(\\d+)px`));
  return match ? Number(match[1]) : NaN;
}

describe('edit-mode decoration fits inside the room the bar reserves', () => {
  test('the overhang is stated once', () => {
    expect(pxToken('--bar-edit-overhang')).toBeGreaterThan(0);
  });

  test('every host reserves it while editing', () => {
    const reservation = rules().find(
      (rule) => rule.selector.includes('bar-editing') && rule.selector.includes('[data-bar-host]')
    );

    expect(reservation?.selector).toContain('::after');
    expect(reservation?.body).toContain('var(--bar-edit-overhang)');
  });

  test('no decoration reaches further out than the reservation', () => {
    const reserved = pxToken('--bar-edit-overhang');
    /** @type {{selector: string, value: number}[]} */
    const overhangs = [];
    for (const rule of rules()) {
      if (!rule.selector.includes('bar-editing')) continue;
      for (const [, value] of rule.body.matchAll(/(?:right|inset)\s*:[^;]*?-(\d+)px/g)) {
        overhangs.push({ selector: rule.selector, value: Number(value) });
      }
    }

    // The dashed outline is one of these, so the scan has to be finding them.
    expect(overhangs.length).toBeGreaterThan(0);
    for (const { selector, value } of overhangs) {
      expect(value, `${selector} is drawn ${value}px outside the item`).toBeLessThanOrEqual(
        reserved
      );
    }
  });
});
