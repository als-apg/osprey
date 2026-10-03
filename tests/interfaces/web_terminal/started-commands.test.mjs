// @ts-check
/**
 * Unit tests for the question both views ask before ending an agent that
 * started commands still running (started-commands.js):
 *   npx vitest run tests/interfaces/web_terminal/started-commands.test.mjs
 *
 * The copy for none, one, two and three commands, and that every way out of
 * the dialog other than its confirm button lands in `onCancel`, once.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  askAboutStartedCommands,
  startedCommandsQuestion,
} from '../../../src/osprey/interfaces/web_terminal/static/js/started-commands.js';
import {
  dismissConfirm,
  isConfirmUp,
} from '../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js';

const MAGNET = { label: 'magnet_scan.py', command: 'python /x/magnet_scan.py --sector 3' };
const ORBIT = { label: 'orbit_poll.py', command: 'python orbit_poll.py' };
const SLEEP = { label: 'sleep', command: 'sleep' };

/** @param {string} selector */
function button(selector) {
  return /** @type {HTMLButtonElement} */ (document.querySelector(selector));
}

beforeEach(() => {
  document.body.innerHTML = '';
});

afterEach(() => {
  dismissConfirm();
  vi.restoreAllMocks();
});

describe('startedCommandsQuestion', () => {
  test('one command names it, and asks to stop both', () => {
    const q = startedCommandsQuestion([MAGNET]);

    expect(q.title).toBe('This also ends magnet_scan.py.');
    expect(q.confirmLabel).toBe('Stop both');
    expect(q.body).toEqual([
      ['The agent started it and it is still running:'],
      [{ em: 'magnet_scan.py' }, ' — python /x/magnet_scan.py --sector 3'],
    ]);
  });

  test('two commands name both', () => {
    const q = startedCommandsQuestion([MAGNET, ORBIT]);

    expect(q.title).toBe('This also ends magnet_scan.py and orbit_poll.py.');
    expect(q.confirmLabel).toBe('Stop all');
    expect(q.body[0]).toEqual(['The agent started them and they are still running:']);
    expect(q.body).toHaveLength(3);
  });

  test('three or more are counted in the title and listed in the body', () => {
    const q = startedCommandsQuestion([MAGNET, ORBIT, SLEEP]);

    expect(q.title).toBe('This also ends 3 commands the agent started.');
    expect(q.confirmLabel).toBe('Stop all');
    expect(q.body[0]).toEqual(['The agent started them and they are still running:']);
    // A command line equal to its label is not repeated.
    expect(q.body[3]).toEqual([{ em: 'sleep' }]);
  });

  test('a refusal whose list did not arrive still asks', () => {
    const q = startedCommandsQuestion([]);

    expect(q.title).toBe('This also ends the commands the agent started.');
    expect(q.confirmLabel).toBe('Stop all');
    expect(q.body).toEqual([['They are still running.']]);
  });
});

describe('askAboutStartedCommands', () => {
  test('Cancel has focus', () => {
    askAboutStartedCommands([MAGNET], { onStop: vi.fn(), onCancel: vi.fn() });

    expect(document.activeElement).toBe(button('.posture-modal-cancel'));
  });

  test('the confirm button runs onStop once and never onCancel', () => {
    const onStop = vi.fn();
    const onCancel = vi.fn();
    askAboutStartedCommands([MAGNET], { onStop, onCancel });

    button('.posture-modal-confirm').click();

    expect(onStop).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
    expect(isConfirmUp()).toBe(false);
  });

  test('Cancel runs onCancel once', () => {
    const onStop = vi.fn();
    const onCancel = vi.fn();
    askAboutStartedCommands([MAGNET], { onStop, onCancel });

    button('.posture-modal-cancel').click();

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onStop).not.toHaveBeenCalled();
  });

  test('Escape runs onCancel once', () => {
    const onStop = vi.fn();
    const onCancel = vi.fn();
    askAboutStartedCommands([MAGNET], { onStop, onCancel });

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onStop).not.toHaveBeenCalled();
    expect(isConfirmUp()).toBe(false);
  });
});
