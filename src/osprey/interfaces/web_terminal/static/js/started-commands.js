// @ts-check
/* OSPREY Web Terminal — Started Commands Question
 *
 * The one question both views ask before ending an agent that started
 * commands still running: ending the agent ends them too, so the operator
 * agrees to that or keeps both running. The wording is DOM-free
 * (`startedCommandsQuestion`) so it can be asserted without a dialog on
 * screen; `askAboutStartedCommands` raises it through the shared confirm.
 */

import { dismissConfirm, isConfirmUp, showConfirm } from './posture-confirm.js';

/**
 * One command the agent started, as the server lists it.
 * @typedef {{label: string, command: string}} StartedCommandJson
 */

/**
 * The title, body and confirm label of the question.
 * @param {StartedCommandJson[]} commands
 * @returns {{title: string,
 *            body: (import('./control-target-facts.js').ConfirmRun)[][],
 *            confirmLabel: string}}
 */
export function startedCommandsQuestion(commands) {
  if (commands.length === 0) {
    // A refusal whose list did not arrive still asks.
    return {
      title: 'This also ends the commands the agent started.',
      body: [['They are still running.']],
      confirmLabel: 'Stop all',
    };
  }
  let title;
  if (commands.length === 1) {
    title = `This also ends ${commands[0].label}.`;
  } else if (commands.length === 2) {
    title = `This also ends ${commands[0].label} and ${commands[1].label}.`;
  } else {
    title = `This also ends ${commands.length} commands the agent started.`;
  }
  const lead =
    commands.length === 1
      ? 'The agent started it and it is still running:'
      : 'The agent started them and they are still running:';
  /** @type {(import('./control-target-facts.js').ConfirmRun)[][]} */
  const body = [[lead]];
  for (const { label, command } of commands) {
    body.push(command && command !== label ? [{ em: label }, ` — ${command}`] : [{ em: label }]);
  }
  return { title, body, confirmLabel: commands.length === 1 ? 'Stop both' : 'Stop all' };
}

/**
 * Ask whether ending the agent may end its started commands too.
 *
 * Cancel has focus. The confirm button runs `onStop`; every other way out —
 * Cancel, Escape, another dialog replacing this one — runs `onCancel`. Exactly
 * one of the two runs, once.
 * @param {StartedCommandJson[]} commands
 * @param {{onStop: () => void, onCancel: () => void}} handlers
 */
export function askAboutStartedCommands(commands, { onStop, onCancel }) {
  let agreed = false;
  /** @param {KeyboardEvent} event */
  const onKey = (event) => {
    if (event.key === 'Escape' && isConfirmUp()) {
      event.preventDefault();
      dismissConfirm();
    }
  };
  showConfirm({
    ...startedCommandsQuestion(commands),
    live: null,
    focus: 'cancel',
    onConfirm: ({ done }) => {
      agreed = true;
      done();
      onStop();
    },
    onDismiss: () => {
      document.removeEventListener('keydown', onKey, true);
      if (!agreed) onCancel();
    },
  });
  document.addEventListener('keydown', onKey, true);
}
