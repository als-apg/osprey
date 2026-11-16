// @ts-check
/**
 * The review screens (`review`, `review-changed`, `review-no-owner`): the
 * stored draft, and Start.
 *
 * The stage says what the run does to the machine, in plain words, then the
 * objective with its value now and each knob on the knob bar. The inspector
 * holds the notices (a draft changed since it was shown, with the control
 * that acknowledges it; a session with no account; windows the service moved
 * on save; a refused Start), the exact stored values by section, and in its
 * foot what Start starts (the revision, the account, the target) and Start.
 * The toolbar's primary slot holds the same Start.
 *
 * Everything is drawn from the stored draft and the revision of the response
 * it came from, so what the operator reads is what Start runs. When Start is
 * enabled, what must be acknowledged and what is said beside Start come from
 * approval-model.js (`screen.gate`); both Starts start `screen.shown`, the job
 * and revision the view drew, and `actions.start` sends nothing if the store
 * has moved on since.
 *
 * Start keeps its place: the foot is pinned to the bottom of the inspector and
 * a notice that arrives late (the session's answer) grows the foot upward,
 * above Start, never under the cursor. Readings of the machine are not part of
 * the draft: a new reading redraws the stage only, never Start.
 *
 * @module review-view
 */

import { renderState } from '/design-system/js/panel-state.js';
import { icon } from '/design-system/js/icons.js';
import { REVIEWED_NOTICE, approvalEnvKey } from './approval-model.js';
import { MOVED_WINDOW, configSections, targetLabel } from './config-sections.js';
import { draftStage } from './draft-stage.js';
import { h, replaceChildren } from './hyperscript.js';
import { readingKey } from './store.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {import('./approval-model.js').StartGate} StartGate
 */

const NO_GATE = Object.freeze({ enabled: false, reason: null, warnings: [] });

/**
 * The draft's own read of its environment, when it is the one the draft names.
 *
 * @param {State} state
 */
function draftEnv(state) {
  const { approval, approvalEnv } = state;
  return approvalEnv.key && approvalEnv.key === approvalEnvKey(approval.config)
    ? approvalEnv
    : null;
}

/**
 * @param {string} tone  ok, warn or err
 * @param {string} kind  `data-notice`
 * @param {...(Node|string)} parts
 */
function notice(tone, kind, ...parts) {
  return h(
    'div',
    {
      class: `osprey-banner osprey-banner--${tone}`,
      'data-notice': kind,
      role: tone === 'err' ? 'alert' : 'status',
    },
    h('span', { class: 'osprey-banner-text' }, ...parts)
  );
}

/**
 * What Start starts, in one line: the revision, the account, the target.
 *
 * @param {State} state
 * @param {string|null} revision
 * @param {string} target
 */
function startsLine(state, revision, target) {
  const parts = [`Starts revision ${revision ? revision.slice(0, 8) : '(none)'} as shown`];
  if (state.ownerKnown === false) parts.push('with no account');
  else if (state.owner) parts.push(`as ${state.owner}`);
  return [
    `${parts.join(', ')}${target ? ', on the ' : '.'}`,
    target
      ? h('b', { class: target === 'live' ? 'opt-live' : null, text: targetLabel(target) })
      : null,
    target ? '.' : null,
  ];
}

/** @type {import('./view-host.js').MountView} */
export function mountReviewView(ctx) {
  const { actions, shell } = ctx;
  const scroll = h('div', { class: 'opt-stage-scroll opt-draft-stage' });
  ctx.stage.replaceChildren(scroll);
  let stageKey = '';
  let inspectorKey = '';
  /** The readings the objective was last read with, so it is read again with them. */
  let objectiveFor = /** @type {unknown} */ (null);

  /**
   * @param {State} state
   * @param {Screen} screen
   */
  function drawStage(state, screen) {
    const config = screen.config;
    const env = draftEnv(state);
    const where = [config?.environment_name || '', config?.control_target || ''];
    const observed = state.readings[readingKey('observables', where[0], where[1])];
    const moved = state.saveNote?.jobId === state.approval.jobId ? state.saveNote?.moved || [] : [];
    const key = JSON.stringify([
      state.approval.jobId,
      state.approval.revision,
      !!env?.details,
      env?.models?.length,
      env?.readbacks,
      observed?.at,
      moved,
    ]);
    if (key === stageKey) return;
    stageKey = key;
    if (!config) {
      renderState(scroll, { icon: 'clock', message: 'Reading the draft…' });
      return;
    }
    /** @type {Record<string, import('./config-sections.js').Flag>} */
    const flags = Object.fromEntries(
      moved.map((name) => [name, { tone: 'warn', text: MOVED_WINDOW }])
    );
    replaceChildren(
      scroll,
      ...draftStage({
        config,
        env: env?.details || null,
        models: env?.models || null,
        values: env?.readbacks || {},
        objectiveNow: observed?.values?.[config.objective_name || ''],
        flags,
      })
    );
  }

  /**
   * @param {State} state
   * @param {Screen} screen
   * @param {StartGate} gate
   * @param {{ jobId: string|null, revision: string|null }} shown
   */
  function drawInspector(state, screen, gate, shown) {
    const { approval, job } = state;
    const config = screen.config;
    const env = draftEnv(state);
    const note = state.saveNote?.jobId === approval.jobId ? state.saveNote : null;
    const label =
      (state.environments.list || []).find((e) => e.name === config?.environment_name)
        ?.display_name || null;
    const key = JSON.stringify([
      approval.jobId,
      approval.revision,
      approval.needsReview,
      approval.notice,
      gate,
      state.owner,
      state.ownerKnown,
      state.startError,
      note,
      job?.status,
      !!env?.details,
      env?.models?.length,
      env?.error,
      label,
    ]);
    if (key === inspectorKey) return;
    inspectorKey = key;
    const reviewed = approval.notice === REVIEWED_NOTICE;
    const start = () => void actions.start(shown);
    shell.fillInspector({
      title: 'Review and start',
      desc: approval.revision ? `revision ${approval.revision.slice(0, 8)}` : null,
      tools: h(
        'button',
        {
          type: 'button',
          class: 'osprey-icon-btn',
          id: 'edit-draft-btn',
          'aria-label': 'Edit the draft',
          title: 'Edit the draft',
          'data-tip-detail': 'Opens the set-up. A saved change needs a new look before Start.',
          disabled: job?.status !== 'created',
          onclick: () => actions.editDraft(),
        },
        icon('pen')
      ),
      body: [
        approval.notice
          ? notice(approval.needsReview ? 'warn' : 'ok', 'changed', approval.notice)
          : null,
        state.startError
          ? notice('err', 'refused', h('b', { text: 'Not started. ' }), state.startError)
          : null,
        note ? notice(note.tone === 'ok' ? 'ok' : 'warn', 'saved', note.text) : null,
        env?.error
          ? notice('warn', 'environment', `The environment could not be read: ${env.error}`)
          : null,
        ...(config
          ? configSections(config, env?.details || null, env?.models || null, {
              environmentLabel: label,
              moved: note?.moved || [],
            })
          : [h('p', { class: 'opt-note', text: 'Reading the draft…' })]),
      ],
      foot: h(
        'div',
        { class: 'opt-review-foot' },
        // Notices that arrive late sit here, above Start: the foot is pinned
        // to the bottom, so Start does not move.
        ...gate.warnings.map((text) => notice('warn', 'start-warning', text)),
        approval.needsReview || reviewed
          ? h(
              'label',
              { class: 'osprey-switch-label opt-ack' },
              h('input', {
                class: 'osprey-switch-input',
                type: 'checkbox',
                id: 'reviewed-check',
                checked: reviewed,
                disabled: reviewed,
                onchange: () => actions.markReviewed(),
              }),
              h('span', { class: 'osprey-switch' }),
              'I have looked at the new version'
            )
          : null,
        h(
          'p',
          { class: 'opt-foot-note' },
          ...startsLine(state, approval.revision, config?.control_target || '')
        ),
        gate.enabled || !gate.reason
          ? null
          : h('p', { class: 'opt-foot-note opt-start-why', 'data-why': '', text: gate.reason }),
        h(
          'button',
          {
            type: 'button',
            class: 'osprey-btn primary block',
            id: 'start-btn',
            disabled: !gate.enabled,
            title: gate.enabled ? 'Start run' : gate.reason || 'Start run',
            'data-tip-detail': gate.enabled ? 'Starts exactly the stored draft shown above.' : null,
            onclick: start,
          },
          h('span', { class: 'osprey-btn-label', text: 'Start run' })
        )
      ),
    });
  }

  return {
    update(state, screen) {
      const gate = screen.gate || NO_GATE;
      const shown = screen.shown || { jobId: null, revision: null };
      const config = screen.config;
      const env = draftEnv(state);
      // The objective's value now is read with each new reading of the knobs.
      if (
        config?.objective_name &&
        config.control_target &&
        env?.details &&
        objectiveFor !== env.readbacks
      ) {
        objectiveFor = env.readbacks;
        void actions.read('observables', config.environment_name || '', config.control_target, [
          config.objective_name,
        ]);
      }
      drawStage(state, screen);
      drawInspector(state, screen, gate, shown);
      shell.setPrimary({
        label: 'Start run',
        id: 'toolbar-start-btn',
        disabled: !gate.enabled,
        why: gate.reason,
        detail: 'Starts exactly the stored draft shown in the inspector.',
        key: JSON.stringify(['start', shown, gate.enabled, gate.reason]),
        onClick: () => void actions.start(shown),
      });
    },
  };
}
