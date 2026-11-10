// @ts-check
/**
 * The run card: status, the run controls the status allows, the write
 * authorization the run is waiting on, and iteration progress.
 *
 * There is no Restore here and nothing restores by itself: a cancelled or
 * failed run holds the machine at its last evaluated point, and the cancel
 * confirmation says so.
 *
 * @module run-view
 */

import { h, fmt, replaceChildren } from './hyperscript.js';
import { authorizationNote, runControls } from './approval-model.js';
import { progress } from './live-plot.js';

export const CANCEL_NOTE =
  'Cancelling leaves the machine at the last point evaluated. ' +
  'Restoring the starting setpoints is a separate, manual step.';

/**
 * @param {string} status
 * @returns {'ok'|'err'|'warn'|'info'}
 */
export function statusTone(status) {
  if (status === 'running' || status === 'completed') return 'ok';
  if (status === 'error') return 'err';
  if (status === 'paused' || status === 'cancelling' || status === 'cancelled') return 'warn';
  return 'info';
}

/**
 * @param {string} id
 * @param {string} text
 * @param {() => void} onclick
 * @param {string} [cls]
 */
function button(id, text, onclick, cls = 'osprey-btn') {
  return h('button', { type: 'button', class: cls, id, text, onclick });
}

/**
 * @param {HTMLElement} root
 * @param {any} job  a JobResponse (detail=full while running)
 * @param {{ cancelArmed: boolean, runError: string|null }} view
 * @param {{ onControl: (action: 'pause'|'resume'|'cancel') => void }} handlers
 */
export function renderRunStatus(root, job, view, handlers) {
  const p = progress(job);
  const allowed = runControls(job.status);
  const auth = authorizationNote(job.write_authorization);
  replaceChildren(
    root,
    h(
      'div',
      { class: 'card-head' },
      h('h2', { text: 'Run' }),
      h(
        'span',
        { class: `osprey-chip osprey-chip--${statusTone(job.status)}` },
        h('span', { class: 'osprey-chip-label', text: job.status })
      ),
      h('span', { class: 'muted mono', title: job.job_id, text: job.job_id.slice(0, 8) }),
      h('span', { class: 'card-head-spacer' }),
      allowed.pause ? button('pause-btn', 'Pause', () => handlers.onControl('pause')) : null,
      allowed.resume ? button('resume-btn', 'Resume', () => handlers.onControl('resume')) : null,
      allowed.cancel
        ? button(
            'cancel-btn',
            view.cancelArmed ? 'Confirm cancel' : 'Cancel',
            () => handlers.onControl('cancel'),
            view.cancelArmed ? 'osprey-btn confirm' : 'osprey-btn'
          )
        : null
    ),
    view.cancelArmed ? h('p', { class: 'banner banner-warn', text: CANCEL_NOTE }) : null,
    auth ? h('p', { class: `banner banner-${auth.tone}`, role: 'status', text: auth.text }) : null,
    job.error_message ? h('p', { class: 'banner banner-err', text: job.error_message }) : null,
    view.runError
      ? h('p', { class: 'banner banner-err', role: 'alert', text: view.runError })
      : null,
    h(
      'div',
      {
        class: 'progress',
        role: 'progressbar',
        'aria-valuemin': '0',
        'aria-valuemax': '100',
        'aria-valuenow': p.percent == null ? null : String(Math.round(p.percent)),
      },
      h('div', { class: 'progress-fill', style: `width: ${p.percent ?? 0}%` })
    ),
    h('p', {
      class: 'readout',
      id: 'progress-label',
      text:
        `${p.done} of ${p.total ?? '?'} evaluations · ` +
        `best ${job.objective_name || 'objective'} ${fmt(p.best)}`,
    })
  );
}
