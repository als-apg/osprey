// @ts-check
/**
 * Unit tests for the question both views ask before ending an agent that
 * started commands still running (started-commands.js):
 *   npx vitest run tests/interfaces/web_terminal/started-commands.test.mjs
 *
 * The copy for none, one, two and three commands, what the title does when
 * the names cannot tell the commands apart, and that every way out of the
 * dialog other than its confirm button lands in `onCancel`, once.
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
const SCAN_3 = { label: 'scan.py', command: 'python scan.py --sector 3' };
const SCAN_4 = { label: 'scan.py', command: 'python scan.py --sector 4' };
const SLEEP_60 = { label: 'sleep', command: 'sleep 60' };
// A command line no one program names, cut short by the server.
const LOOP = {
  label: "bash -c 'while true; do date >> /tmp/be…",
  command: "bash -c 'while true; do date >> /tmp/beat; sleep 1; done'",
};
const LOOP_FG = {
  label: "bash -c 'for i in $(seq 300); do date >…",
  command: "bash -c 'for i in $(seq 300); do date >> /tmp/beat-fg; sleep 1; done'",
};

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

  test('two commands of one name are counted, and told apart by their command lines', () => {
    const q = startedCommandsQuestion([SCAN_3, SCAN_4]);

    expect(q.title).toBe('This also ends 2 commands the agent started.');
    expect(q.confirmLabel).toBe('Stop all');
    expect(q.body.slice(1)).toEqual([
      [{ em: 'scan.py' }, ' — python scan.py --sector 3'],
      [{ em: 'scan.py' }, ' — python scan.py --sector 4'],
    ]);
  });

  test('a name cut short is counted in the title and listed whole in the body', () => {
    const q = startedCommandsQuestion([LOOP]);

    expect(q.title).toBe('This also ends the command the agent started.');
    expect(q.confirmLabel).toBe('Stop both');
    expect(q.body).toEqual([
      ['The agent started it and it is still running:'],
      [{ em: "bash -c 'while true; do date >> /tmp/beat; sleep 1; done'" }],
    ]);
  });

  test('two names cut short are counted, and listed whole', () => {
    const q = startedCommandsQuestion([LOOP, LOOP_FG]);

    expect(q.title).toBe('This also ends 2 commands the agent started.');
    expect(q.body.slice(1)).toEqual([[{ em: LOOP.command }], [{ em: LOOP_FG.command }]]);
  });

  test('a command line that begins with its name is listed once, whole', () => {
    const q = startedCommandsQuestion([SLEEP_60]);

    expect(q.title).toBe('This also ends sleep.');
    expect(q.body[1]).toEqual([{ em: 'sleep 60' }]);
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
