// @ts-check
/**
 * Unit tests for the centred state (design_system/static/js/panel-state.js):
 *   npx vitest run tests/interfaces/design_system/panel-state.test.mjs
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import { clearState, renderState } from '/design-system/js/panel-state.js';

/** @returns {HTMLElement} */
function host() {
  return /** @type {HTMLElement} */ (document.getElementById('host'));
}

beforeEach(() => {
  document.body.innerHTML = '<section id="host"><p>old content</p><ul><li>row</li></ul></section>';
});

describe('renderState', () => {
  test("replaces the host's content with one state and returns it", () => {
    const root = renderState(host(), {
      icon: 'refresh',
      message: 'Connecting to the queue server',
      detail: 'Retrying every 5 s',
    });
    expect(host().children).toHaveLength(1);
    expect(host().firstElementChild).toBe(root);
    expect(root.classList.contains('osprey-state')).toBe(true);
    expect(root.querySelector('svg.osprey-ic')).not.toBeNull();
    expect(root.textContent).toContain('Connecting to the queue server');
    expect(root.textContent).toContain('Retrying every 5 s');
    expect(host().textContent).not.toContain('old content');
  });

  test('the action button calls back on click', () => {
    const onClick = vi.fn();
    const root = renderState(host(), {
      icon: 'x',
      message: 'The queue server cannot be reached',
      action: { label: 'Retry', onClick },
    });
    const buttons = root.querySelectorAll('button');
    expect(buttons).toHaveLength(1);
    expect(buttons[0].textContent).toBe('Retry');
    buttons[0].click();
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  test('no action gives no button, and no detail gives one line', () => {
    const root = renderState(host(), { icon: 'search', message: 'No matches' });
    expect(root.querySelector('button')).toBeNull();
    expect(root.querySelectorAll('p')).toHaveLength(1);
  });

  test('a second state replaces the first', () => {
    renderState(host(), { icon: 'refresh', message: 'Loading' });
    renderState(host(), { icon: 'x', message: 'Failed' });
    expect(host().querySelectorAll('.osprey-state')).toHaveLength(1);
    expect(host().textContent).toContain('Failed');
  });

  test('an unknown icon name fails at the call site', () => {
    expect(() => renderState(host(), { icon: 'no-such-icon', message: 'x' })).toThrow(
      /unknown icon/
    );
  });
});

describe('clearState', () => {
  test('removes the state and leaves content added after it', () => {
    renderState(host(), { icon: 'refresh', message: 'Loading' });
    const content = document.createElement('table');
    host().append(content);
    clearState(host());
    expect(host().querySelector('.osprey-state')).toBeNull();
    expect(host().firstElementChild).toBe(content);
  });
});
