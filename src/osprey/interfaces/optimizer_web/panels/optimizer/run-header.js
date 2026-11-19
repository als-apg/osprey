// @ts-check
/**
 * The run header: the sentence of what happens now (a live region), the
 * line under it, Pause or Resume and Cancel at its right, the progress line,
 * the run's facts, and the reason of a refused control (a live region).
 *
 * The header is built once. Each update writes texts, states and widths in
 * place: the buttons are never rebuilt, so a poll does not move a control
 * under the cursor. Pause and Resume share one slot of a fixed width, so
 * Cancel never moves. Which control is enabled follows `screen.controls`;
 * while the run cancels none is.
 *
 * Cancel has two steps. The first click arms the cancel (`requestCancel`)
 * and the confirmation opens below Cancel; it says that the machine keeps the
 * last point written and that nothing is restored. Only its own confirm
 * sends the cancel (`confirmCancel`); "keep", Escape, a click elsewhere or a
 * second click on Cancel disarm it (`dismissCancel`).
 *
 * @module run-header
 */

import { icon } from '/design-system/js/icons.js';
import { closePopover, openPopover } from '/design-system/js/popover.js';
import { factsStrip, shownFacts, updateFacts } from './facts-strip.js';
import { h } from './hyperscript.js';
import { authorizationScope } from './approval-model.js';
import {
  cancelWords,
  methodLine,
  nowSentence,
  pauseWords,
  runFacts,
  runKind,
  waitFacts,
} from './run-page-model.js';
import { STOP_WORDS } from './words.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {import('./actions.js').Actions} Actions
 */

/**
 * Write `text` into `node` when it differs (a live region announces changes only).
 *
 * @param {Element} node
 * @param {string} text
 */
function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

/** @param {import('./run-page-model.js').RunKind} kind */
function markFor(kind) {
  if (kind === 'waiting') return icon('clock');
  return h('span', { class: 'opt-rp-dot' });
}

/**
 * Build the run header.
 *
 * @param {{ actions: Actions, doc: Document }} options
 */
export function createRunHeader({ actions, doc }) {
  const mark = h('span', { class: 'opt-rp-mark', 'aria-hidden': 'true' });
  const sentence = h('span', { class: 'opt-rp-sentence' });
  const now = h(
    'div',
    { class: 'opt-rp-now', role: 'status', 'aria-live': 'polite' },
    mark,
    sentence
  );
  const sub = h('div', { class: 'opt-rp-sub' });
  /** @type {State|null} */
  let latest = null;
  /** @type {Screen|null} */
  let shown = null;

  const pause = /** @type {HTMLButtonElement} */ (
    h('button', {
      type: 'button',
      class: 'osprey-btn opt-rp-btn',
      'data-control': 'pause',
      onclick: () => {
        // The job the header shows: actions send nothing for another one.
        const jobId = shown?.job?.job_id;
        if (shown && runKind(shown) === 'paused') void actions.resume(jobId);
        else void actions.pause(jobId);
      },
    })
  );
  const cancel = /** @type {HTMLButtonElement} */ (
    h('button', {
      type: 'button',
      class: 'osprey-btn danger opt-rp-btn',
      'data-control': 'cancel',
      'aria-haspopup': 'dialog',
      text: 'Cancel run',
      onclick: () => {
        if (latest?.control.cancelArmed) actions.dismissCancel();
        else actions.requestCancel();
      },
    })
  );
  const bar = h('span', { class: 'opt-rp-bar-fill' });
  const progress = h('div', { class: 'opt-rp-bar', 'aria-hidden': 'true' }, bar);
  const facts = factsStrip([], { label: 'This run' });
  facts.classList.add('opt-rp-facts');
  const refusal = h('div', { class: 'opt-rp-refusal', 'aria-live': 'assertive' });

  const head = h(
    'header',
    { class: 'opt-rp-head', 'aria-label': 'Run' },
    h('div', { class: 'opt-rp-text' }, now, sub),
    h('div', { class: 'opt-rp-actions' }, pause, cancel),
    progress,
    facts,
    refusal
  );

  // The confirmation: a body-level popover, so no scrolled zone cuts it.
  const popTitle = h('div', { class: 'osprey-popover-title' });
  const popText = h('p', { class: 'opt-confirm-text' });
  const keep = h('button', {
    type: 'button',
    class: 'osprey-popover-item',
    'data-confirm': 'keep',
    onclick: () => actions.dismissCancel(),
  });
  const confirm = h('button', {
    type: 'button',
    class: 'osprey-popover-item danger',
    'data-confirm': 'cancel',
    onclick: () => void actions.confirmCancel(shown?.job?.job_id),
  });
  const pop = h(
    'div',
    { class: 'osprey-popover opt-confirm', role: 'dialog', hidden: true },
    popTitle,
    popText,
    h('div', { class: 'osprey-popover-sep' }),
    keep,
    confirm
  );
  pop.setAttribute('aria-label', 'Confirm the cancel');
  doc.body.append(pop);
  // Escape, a click elsewhere or a resize close the popover (popover.js):
  // the armed cancel goes with it.
  const watch = new MutationObserver(() => {
    if (pop.hidden && latest?.control.cancelArmed) actions.dismissCancel();
  });
  watch.observe(pop, { attributes: true, attributeFilter: ['hidden'] });

  let markKind = '';

  /**
   * @param {State} state
   * @param {Screen} screen
   * @param {{ units: string }} extra
   */
  function update(state, screen, { units }) {
    latest = state;
    shown = screen;
    const kind = runKind(screen);
    head.dataset.kind = kind;
    if (markKind !== kind) {
      markKind = kind;
      mark.replaceChildren(markFor(kind));
    }
    setText(sentence, nowSentence(screen));
    const line = methodLine(screen);
    setText(sub, line || '');
    sub.hidden = !line;

    const paused = kind === 'paused';
    setText(pause, paused ? 'Resume' : 'Pause');
    pause.classList.toggle('primary', paused);
    pause.disabled = !(paused ? screen.controls.resume : screen.controls.pause);
    pause.title = paused ? 'Carry on from here.' : pauseWords(screen).ask;
    cancel.disabled = !screen.controls.cancel;
    cancel.title = `Asks once. ${kind === 'waiting' ? 'Nothing has been written.' : STOP_WORDS.now}`;

    const f = screen.figures;
    const total = f?.total ?? null;
    progress.hidden = kind === 'waiting' || !total;
    if (f && total) bar.style.width = `${Math.min(100, (f.done / total) * 100).toFixed(1)}%`;

    const scope = authorizationScope(screen.job?.write_authorization);
    const list = kind === 'waiting' ? waitFacts(scope, state.owner) : runFacts(screen, units);
    updateFacts(facts, list);
    facts.hidden = shownFacts(list).length === 0;

    const error = state.control.error;
    const drawnError = refusal.dataset.error || '';
    if ((error || '') !== drawnError) {
      refusal.dataset.error = error || '';
      refusal.replaceChildren(
        error
          ? h(
              'div',
              { class: 'osprey-banner osprey-banner--err' },
              h('span', { class: 'osprey-banner-text', text: `Not done: ${error}` })
            )
          : ''
      );
    }

    const words = cancelWords(screen);
    setText(popTitle, words.title);
    setText(popText, words.text);
    setText(keep, words.keep);
    setText(confirm, words.confirm);
    const armed = state.control.cancelArmed && !cancel.disabled;
    if (armed && pop.hidden && cancel.isConnected) openPopover(pop, cancel);
    else if (!armed && !pop.hidden) closePopover(pop);
  }

  return {
    head,
    update,
    destroy() {
      watch.disconnect();
      if (!pop.hidden) closePopover(pop);
      pop.remove();
    },
  };
}
