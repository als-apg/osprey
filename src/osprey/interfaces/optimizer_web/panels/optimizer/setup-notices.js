// @ts-check
/**
 * The set-up's notices and small parts: what holds Save and why (a refused
 * save, a stored draft that could not be read, a draft that changed on the
 * service while it was open), knobs a stored draft names that its environment
 * no longer has, a list a picker offers that could not be read, the folders
 * of the inspector, the stage's empty state.
 *
 * @module setup-notices
 */

import { icon } from '/design-system/js/icons.js';
import { h } from './hyperscript.js';
import { issueLine } from './setup-form.js';

/**
 * @param {string} tone  ok, warn or err
 * @param {string} kind  `data-notice`
 * @param {...(Node|string|null)} parts
 * @returns {HTMLElement}
 */
export function banner(tone, kind, ...parts) {
  return h(
    'div',
    {
      class: `osprey-banner osprey-banner--${tone}`,
      'data-notice': kind,
      role: tone === 'err' ? 'alert' : null,
    },
    h('span', { class: 'osprey-banner-text' }, ...parts)
  );
}

/**
 * @param {string} id
 * @param {string} text
 * @param {() => void} onclick
 * @param {boolean} [disabled]
 */
function smallButton(id, text, onclick, disabled = false) {
  return h('button', { type: 'button', class: 'osprey-btn ghost sm', id, text, onclick, disabled });
}

/**
 * The notices at the top of the set-up's inspector.
 *
 * @param {import('./store.js').State} state  a state with a set-up open
 * @param {{
 *   hasForm: boolean,
 *   unreadable: boolean,
 *   moved: boolean,
 *   message: string|null,
 *   onSaveNew: () => void,
 *   onLoadNew: () => void,
 *   onKeepMine: () => void,
 *   onReadEnvironment: () => void,
 * }} parts  `unreadable`: the stored draft could not be read; `moved`: it
 *   changed on the service since the form was seeded; `onReadEnvironment`
 *   reads the environment of the set-up again
 * @returns {(HTMLElement|null)[]}
 */
export function setupNotices(state, parts) {
  const setup = /** @type {import('./store.js').Setup} */ (state.setup);
  const entry = state.envs[setup.environment];
  return [
    setup.jobId
      ? h(
          'p',
          { class: 'opt-note' },
          'This edits the stored draft. ',
          smallButton(
            'opt-su-save-new',
            'Save as a new draft',
            parts.onSaveNew,
            !parts.hasForm || setup.saving
          )
        )
      : null,
    // Every reason, read out once; each also shows at the field it names.
    setup.issues.length
      ? banner('err', 'refused', h('b', { text: 'Not saved. ' }), setup.issues.join(' '))
      : null,
    parts.unreadable
      ? banner(
          'err',
          'unreadable',
          h('b', { text: 'The stored draft could not be read. ' }),
          'Saving is held, so nothing replaces it. Discard, then open the draft again.'
        )
      : null,
    parts.moved
      ? banner(
          'warn',
          'draft-changed',
          h('b', { text: 'The stored draft changed on the service. ' }),
          'Saving is held. Load the new version (the edits here are dropped), or keep your ' +
            'edits (a save then replaces the new version). ',
          smallButton('opt-su-load-new', 'Load the new version', parts.onLoadNew),
          smallButton('opt-su-keep-mine', 'Keep my edits', parts.onKeepMine)
        )
      : null,
    parts.message ? banner('warn', 'message', parts.message) : null,
    entry?.error
      ? banner(
          'err',
          'environment',
          `${entry.error} `,
          smallButton('opt-su-env-details-reread', 'Read again', parts.onReadEnvironment)
        )
      : null,
  ];
}

/**
 * The knobs a stored draft tunes that its environment no longer has, or null.
 *
 * @param {string[]} gone
 * @returns {HTMLElement|null}
 */
export function missingKnobsNotice(gone) {
  if (!gone.length) return null;
  const one = gone.length === 1;
  return banner(
    'warn',
    'missing-knobs',
    `The stored draft tunes ${one ? 'a knob' : 'knobs'} this environment no longer has: ` +
      `${gone.join(', ')}. A save leaves ${one ? 'it' : 'them'} out.`
  );
}

/**
 * A folder of the inspector; its summary's aside is written by `writeReadouts`.
 *
 * @param {string} key
 * @param {string} title
 * @param {(Node|null)[]} content
 * @param {boolean} open
 * @param {(open: boolean) => void} onToggle
 * @returns {HTMLElement}
 */
export function setupFolder(key, title, content, open, onToggle) {
  const det = h(
    'details',
    { class: 'osprey-folder opt-folder', open, 'data-folder': key },
    h('summary', null, title, h('span', { class: 'osprey-aside', 'data-live': `aside-${key}` })),
    h('div', { class: 'opt-folder-body' }, ...content)
  );
  det.addEventListener('toggle', () => onToggle(/** @type {HTMLDetailsElement} */ (det).open));
  return det;
}

/**
 * The button that reads the values now again.
 *
 * @param {() => void} onclick
 * @returns {HTMLElement}
 */
export function rereadButton(onclick) {
  return h(
    'button',
    {
      type: 'button',
      class: 'osprey-btn ghost sm opt-reread',
      title: 'Read the values now again',
      onclick,
    },
    icon('refresh', { size: 14 }),
    'Read again'
  );
}

/**
 * A list a picker offers (the environments, the model files) that could not
 * be read: why, and the control that reads it again. Null when it was read.
 *
 * @param {string} id  the id of the control
 * @param {string} what  the list, as the start of a sentence
 * @param {string|null} error
 * @param {() => void} onRead
 * @returns {HTMLElement|null}
 */
export function unreadList(id, what, error, onRead) {
  if (!error) return null;
  return h(
    'div',
    { class: 'osprey-field', 'data-unread': id },
    issueLine(`${what} could not be read: ${error}`),
    h(
      'button',
      { type: 'button', class: 'osprey-btn ghost sm', id, onclick: () => onRead() },
      icon('refresh', { size: 14 }),
      'Read again'
    )
  );
}

/**
 * What the stage says while it has no form to draw. An environment that
 * could not be read offers to read it again.
 *
 * @param {import('./store.js').Setup} setup
 * @param {import('./store.js').EnvEntry|undefined} entry
 * @param {string|null} why  why Save is held, if it is
 * @param {() => void} onReadEnvironment
 * @returns {{ message: string, detail?: string, action?: { label: string, onClick: () => void } }}
 */
export function emptyStage(setup, entry, why, onReadEnvironment) {
  if (!setup.environment) {
    return {
      message: 'Pick an environment',
      detail: 'Pick an environment and a control target in the inspector to set up a run.',
    };
  }
  if (entry?.error) {
    return {
      message: 'The environment could not be read',
      detail: entry.error,
      action: { label: 'Read again', onClick: () => onReadEnvironment() },
    };
  }
  if (why) return { message: why };
  return { message: 'Reading the environment…' };
}
