// @ts-check
/**
 * Security regression for renderSchema (utils.js): both raw sinks — the
 * hierarchy level `name` and the `naming_pattern` — must be escaped before they
 * reach innerHTML, so a malicious value renders as inert text (no live node, no
 * event-handler attribute) while still reaching the sink.
 *   npx vitest run tests/interfaces/channel_finder/renderschema-xss.test.mjs
 *
 * Runs under happy-dom (vitest.config.js); imports resolve '/design-system/js'
 * via the configured alias.
 */

import { test, expect } from 'vitest';

import { renderSchema } from '../../../src/osprey/interfaces/channel_finder/static/js/utils.js';

const PAYLOAD = '<img src=x onerror=alert(1)>';

/**
 * @param {HTMLElement} container
 */
function assertNoLiveInjection(container) {
  // No live element the payload would have created if unescaped.
  expect(container.querySelector('img'), 'no live <img> node').toBeNull();
  expect(container.querySelector('svg'), 'no live <svg> node').toBeNull();
  expect(container.querySelector('script'), 'no live <script> node').toBeNull();
  // No on* event-handler attribute anywhere in the parsed subtree.
  const hasOnAttr = [...container.querySelectorAll('*')].some(el =>
    [...el.attributes].some(attr => attr.name.startsWith('on'))
  );
  expect(hasOnAttr, 'no on* event-handler attribute').toBe(false);
}

test('renderSchema escapes a malicious hierarchy level name', () => {
  const container = document.createElement('div');
  renderSchema(container, 'hierarchical', {
    hierarchy_levels: [PAYLOAD],
    naming_pattern: 'SR:{sector}:{device}',
  });

  assertNoLiveInjection(container);
  // Sink was still reached: the payload survives as inert text (entities decode
  // back to the raw string in textContent).
  expect(container.textContent).toContain(PAYLOAD);
});

test('renderSchema escapes a malicious naming_pattern', () => {
  const container = document.createElement('div');
  renderSchema(container, 'hierarchical', {
    hierarchy_levels: ['system'],
    naming_pattern: PAYLOAD,
  });

  assertNoLiveInjection(container);
  expect(container.textContent).toContain(PAYLOAD);
});
