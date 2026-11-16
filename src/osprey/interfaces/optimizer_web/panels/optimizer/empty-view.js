// @ts-check
/**
 * The `empty` screen: no draft and no run. The stage says so and offers the
 * one next step, a new run; the inspector says how a run works. Until the
 * first read of the service has answered, the stage says it is reading, not
 * that there is nothing: it never asserts what it has not seen.
 *
 * @module empty-view
 */

import { renderState } from '/design-system/js/panel-state.js';
import { h } from './hyperscript.js';
import { BEFORE_A_WRITE, STOP_WORDS } from './words.js';

/** The steps of a run, in the operator's words. */
const HOW_A_RUN_WORKS = Object.freeze([
  ['Draft', 'The agent drafts a run in the chat, or you set one up here. A draft writes nothing.'],
  ['Review and start', 'You read the stored values and press Start. Only a person starts a run.'],
  [
    'Write approval',
    'The run asks for one bounded write request: these knobs, these ranges, this many writes. ' +
      'If the write switch of its owner does not allow it, the owner approves the request first.',
  ],
  [
    'Run',
    'Each step writes the knobs, lets the machine settle and reads the objective. You can pause or cancel.',
  ],
  ['End', `The end action runs, for example "Apply best". ${STOP_WORDS.ahead}`],
]);

/** @type {import('./view-host.js').MountView} */
export function mountEmptyView(ctx) {
  const { actions, shell } = ctx;
  const host = h('div', { class: 'opt-stage-fill' });
  ctx.stage.replaceChildren(host);
  const newRun = () => actions.newRun();
  shell.fillInspector({
    title: 'How a run works',
    body: h(
      'ol',
      { class: 'opt-how' },
      ...HOW_A_RUN_WORKS.map(([title, text], i) =>
        h(
          'li',
          null,
          h('span', { class: 'opt-how-n', text: String(i + 1) }),
          h('div', null, h('b', { text: title }), h('p', { text }))
        )
      )
    ),
    foot: h('button', {
      type: 'button',
      class: 'osprey-btn primary block',
      text: 'New run',
      onclick: newRun,
    }),
  });
  shell.setPrimary({
    label: 'New run',
    id: 'new-run-primary',
    detail: 'Set up a run by hand. The agent can also draft one in the chat.',
    onClick: newRun,
  });
  let drawn = '';
  return {
    update(state, screen) {
      const mode = state.service.ok === false ? 'down' : screen.loading ? 'loading' : 'empty';
      const key = `${mode}|${state.service.error ?? ''}`;
      if (key === drawn) return;
      drawn = key;
      if (mode === 'loading') {
        renderState(host, { icon: 'clock', message: 'Reading the optimizer…' });
      } else if (mode === 'down') {
        renderState(host, {
          icon: 'triangle-alert',
          message: 'The optimizer could not be read',
          detail: state.service.error || undefined,
          action: { label: 'Retry', onClick: () => void actions.refresh() },
        });
      } else {
        renderState(host, {
          icon: 'sliders',
          message: 'No draft and no run',
          detail: `Ask the agent in the chat to draft a run, or set one up here. ${BEFORE_A_WRITE}`,
          action: { label: 'New run', onClick: newRun },
        });
      }
    },
  };
}
