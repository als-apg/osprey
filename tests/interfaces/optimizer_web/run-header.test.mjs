/**
 * Tests for the run header (run-header.js): Pause or Resume and Cancel, the
 * job each control names, the two steps of a cancel and what closes its
 * confirmation, the controls held while a run cancels or when the screen
 * does not allow them, the reason of a refused control, the progress line
 * and the facts, and the controls kept in place across polls.
 */

import { afterEach, describe, test, expect, vi } from 'vitest';

import { runningJob } from './_fixtures.mjs';
import { granted, paused, runAt, running, waiting } from './_run.mjs';
import { createRunHeader } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-header.js';
import {
  deriveScreen,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';
import { nowSentence } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-page-model.js';
import { STOP_WORDS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').State} State
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').Screen} Screen
 */

const ACTIONS = ['pause', 'resume', 'requestCancel', 'confirmCancel', 'dismissCancel'];

function mount() {
  /** @type {Record<string, import('vitest').Mock>} */
  const actions = Object.fromEntries(ACTIONS.map((name) => [name, vi.fn()]));
  const header = createRunHeader({ actions: /** @type {any} */ (actions), doc: document });
  document.body.append(header.head);
  const { head } = header;
  return {
    head,
    actions,
    header,
    /** @param {{ state: State, screen: Screen }} at */
    show: ({ state, screen }) => header.update(state, screen, { units: '' }),
  };
}

/** @param {HTMLElement} root @param {string} control */
const button = (root, control) =>
  /** @type {HTMLButtonElement} */ (root.querySelector(`[data-control="${control}"]`));

const confirmPop = () => /** @type {HTMLElement} */ (document.querySelector('.opt-confirm'));

/** @param {{ state: State }} at */
const armed = ({ state }) => {
  const next = reduce(state, { type: 'control/arm', armed: true });
  return { state: next, screen: deriveScreen(next) };
};

afterEach(() => {
  document.body.replaceChildren();
});

describe('the run header', () => {
  test('Cancel asks first; only the confirmation sends it, and it says nothing is restored', () => {
    const { head, actions, show } = mount();
    show(running());
    button(head, 'cancel').click();
    expect(actions.requestCancel).toHaveBeenCalledTimes(1);
    expect(actions.confirmCancel).not.toHaveBeenCalled();
    expect(confirmPop().hidden).toBe(true);

    show(armed(running()));
    expect(confirmPop().hidden).toBe(false);
    expect(confirmPop().textContent).toContain(STOP_WORDS.now);
    /** @type {HTMLElement} */ (confirmPop().querySelector('[data-confirm="keep"]')).click();
    expect(actions.dismissCancel).toHaveBeenCalledTimes(1);
    /** @type {HTMLElement} */ (confirmPop().querySelector('[data-confirm="cancel"]')).click();
    expect(actions.confirmCancel).toHaveBeenCalledTimes(1);
  });

  test('a second click on an armed Cancel disarms it and sends nothing', () => {
    const { head, actions, show } = mount();
    show(armed(running()));
    button(head, 'cancel').click();
    expect(actions.dismissCancel).toHaveBeenCalledTimes(1);
    expect(actions.requestCancel).not.toHaveBeenCalled();
    expect(actions.confirmCancel).not.toHaveBeenCalled();
  });

  test('each control names the job the page shows', () => {
    const { head, actions, show } = mount();
    const run = running();
    const id = run.screen.job.job_id;
    show(run);
    button(head, 'pause').click();
    expect(actions.pause).toHaveBeenCalledWith(id);
    show(paused());
    button(head, 'pause').click();
    expect(actions.resume).toHaveBeenCalledWith(id);
    show(armed(run));
    /** @type {HTMLElement} */ (confirmPop().querySelector('[data-confirm="cancel"]')).click();
    expect(actions.confirmCancel).toHaveBeenCalledWith(id);
  });

  test('a confirmation closed by the page (Escape, a click elsewhere) disarms the cancel', async () => {
    const { actions, show } = mount();
    show(armed(running()));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(confirmPop().hidden).toBe(true);
    expect(actions.dismissCancel).toHaveBeenCalled();
    expect(actions.confirmCancel).not.toHaveBeenCalled();
  });

  test('while the run cancels no control is enabled, the header says so, and an armed cancel does not open', () => {
    const { head, show } = mount();
    const cancelling = runAt({ status: 'cancelling', write_authorization: granted() });
    expect(cancelling.screen.cancelling).toBe(true);
    show(armed(cancelling));
    expect(button(head, 'pause').disabled).toBe(true);
    expect(button(head, 'cancel').disabled).toBe(true);
    expect(head.querySelector('.opt-rp-sentence')?.textContent).toBe(
      nowSentence(cancelling.screen)
    );
    expect(confirmPop().hidden).toBe(true);
  });

  test('Pause and Cancel follow what the screen allows', () => {
    const { head, show } = mount();
    const run = running();
    show({ ...run, screen: { ...run.screen, controls: { ...run.screen.controls, pause: false } } });
    expect(button(head, 'pause').disabled).toBe(true);
    expect(button(head, 'cancel').disabled).toBe(false);
    show({
      ...run,
      screen: { ...run.screen, controls: { ...run.screen.controls, cancel: false } },
    });
    expect(button(head, 'pause').disabled).toBe(false);
    expect(button(head, 'cancel').disabled).toBe(true);
  });

  test('a refused control shows its reason in a live region, and the reason goes when it clears', () => {
    const { head, show } = mount();
    const refused = runAt({ write_authorization: granted() }, [
      { type: 'control/sent', action: 'pause' },
      { type: 'control/answered', error: 'the job is not running' },
    ]);
    show(refused);
    const region = /** @type {HTMLElement} */ (head.querySelector('.opt-rp-refusal'));
    expect(region.getAttribute('aria-live')).toBeTruthy();
    expect(region.textContent).toContain('the job is not running');
    show(running());
    expect(region.textContent).toBe('');
  });

  test('a paused run offers Resume in the same place; it sends resume', () => {
    const { head, actions, show } = mount();
    show(running());
    const slot = button(head, 'pause');
    show(paused());
    expect(button(head, 'pause')).toBe(slot);
    expect(slot.disabled).toBe(false);
    slot.click();
    expect(actions.resume).toHaveBeenCalledTimes(1);
    expect(actions.pause).not.toHaveBeenCalled();
  });

  test('a poll keeps the controls; a new evaluation updates the sentence in place', () => {
    const { head, show } = mount();
    show(running());
    const pause = button(head, 'pause');
    const cancel = button(head, 'cancel');
    const sentence = head.querySelector('.opt-rp-sentence')?.textContent;
    const data = [...runningJob().data, { phase_a: 3, sol_b: 2.5, charge: 3.1 }];
    show(runAt({ write_authorization: granted(), data }));
    expect(button(head, 'pause')).toBe(pause);
    expect(button(head, 'cancel')).toBe(cancel);
    expect(head.querySelector('.opt-rp-sentence')?.textContent).not.toBe(sentence);
  });

  test('the progress line is the share of the planned evaluations done, and is hidden while the run waits', () => {
    const { head, show } = mount();
    const progress = /** @type {HTMLElement} */ (head.querySelector('.opt-rp-bar'));
    const fill = /** @type {HTMLElement} */ (head.querySelector('.opt-rp-bar-fill'));
    show(running());
    expect(progress.hidden).toBe(false);
    expect(fill.style.width).toBe('30.0%');
    show(waiting());
    expect(progress.hidden).toBe(true);
  });

  test('the facts are those of the request while the run waits, and those of the run after', () => {
    const { head, show } = mount();
    /** @param {string} key */
    const fact = (key) => head.querySelector(`[data-fact="${key}"]`);
    show(waiting());
    expect(fact('approver')).not.toBeNull();
    expect(fact('best')).toBeNull();
    show(running());
    expect(fact('best')).not.toBeNull();
    expect(fact('approver')).toBeNull();
  });

  test('destroy takes the confirmation off the page', () => {
    const { header, show } = mount();
    show(armed(running()));
    expect(confirmPop()).not.toBeNull();
    header.destroy();
    expect(document.querySelector('.opt-confirm')).toBeNull();
  });
});
