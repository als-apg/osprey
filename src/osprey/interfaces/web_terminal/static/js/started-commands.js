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
 * One command the agent started, as the server lists it: `command` is the
 * command line the agent launched, `label` what the operator is shown it as —
 * a file or program name, or the command line itself when no one program names
 * it. Both are cut to a fixed length by the server, and end in `…` when cut.
 * @typedef {{label: string, command: string}} StartedCommandJson
 */

/** @param {string} label */
function isCut(label) {
  return label.endsWith('…');
}

/**
 * Whether the title can name the commands: one or two, each with a name of
 * its own that was not cut short. Otherwise it counts them, and the body
 * tells them apart.
 * @param {StartedCommandJson[]} commands
 */
function nameable(commands) {
  const labels = commands.map((c) => c.label);
  return (
    labels.length <= 2 && new Set(labels).size === labels.length && !labels.some(isCut)
  );
}

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
  if (!nameable(commands)) {
    title =
      commands.length === 1
        ? 'This also ends the command the agent started.'
        : `This also ends ${commands.length} commands the agent started.`;
  } else if (commands.length === 1) {
    title = `This also ends ${commands[0].label}.`;
  } else {
    title = `This also ends ${commands[0].label} and ${commands[1].label}.`;
  }
  const lead =
    commands.length === 1
      ? 'The agent started it and it is still running:'
      : 'The agent started them and they are still running:';
  /** @type {(import('./control-target-facts.js').ConfirmRun)[][]} */
  const body = [[lead]];
  for (const { label, command } of commands) {
    // A command line that begins with its label — the label is the program, or
    // the command line cut short — is listed once, whole.
    const head = isCut(label) ? label.slice(0, -1) : label;
    body.push(
      command && !command.startsWith(head)
        ? [{ em: label }, ` — ${command}`]
        : [{ em: command || label }]
    );
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
