import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearSelection,
  ownCompute,
  panelOf,
  recompute,
  refresh,
  showError,
  showState,
} from '../panel/panel.js';

// The panel's markup, as index.html carries it.
const MARKUP = `
  <span data-field="plane"><button data-plane="x"></button><button data-plane="y"></button></span>
  <select data-field="model" hidden></select>
  <span data-field="orm"><span data-field="orm-text"></span><button data-field="recompute"></button></span>
  <span data-field="no-model" hidden>no closed-orbit model</span>
  <p data-field="posture" hidden></p>
  <p data-field="notice" hidden></p>
  <p data-field="error" hidden></p>
  <div data-field="body"></div>
  <div data-field="confirm" hidden></div>`;

/**
 * @param {string} name
 * @param {'idle' | 'building' | 'ready' | 'failed'} [status]
 */
function model(name, status = 'ready') {
  return {
    name,
    length_m: 196.8,
    orm: {
      status,
      age_s: 125,
      version: 1,
      error: status === 'failed' ? 'deck refused' : null,
      unsolved: [],
    },
    members: { bpms: [], correctors: [] },
  };
}

/**
 * @param {string} id
 * @param {Partial<{model: string, plane: 'x' | 'y', rule: string, priority: string, others: string, all_correctors: boolean}>} [over]
 */
function plan(id, over = {}) {
  return {
    id,
    model: 'SR',
    plane: /** @type {'x' | 'y'} */ ('y'),
    status: 'ready',
    targets: { BPM05: 0.1, BPM06: -0.2 },
    correctors: [{ element: 'VCM04' }, { element: 'VCM07' }],
    ...over,
  };
}

/** @param {object} [over] */
function state(over = {}) {
  return {
    models: [model('SR')],
    posture: { allowed: true, reason: null, phrase: 'Writes are on' },
    orbit: { position: {}, golden: {} },
    plan: null,
    ...over,
  };
}

/**
 * @param {Element} root
 * @param {string} name
 * @returns {HTMLElement}
 */
function field(root, name) {
  const element = root.querySelector(`[data-field="${name}"]`);
  if (!(element instanceof HTMLElement)) throw new Error(name);
  return element;
}

/** @param {unknown} body */
function answering(body) {
  return /** @type {typeof fetch} */ (
    /** @type {unknown} */ (vi.fn(async () => new Response(JSON.stringify(body))))
  );
}

/** A fetcher whose answers the test releases one by one, in any order. */
function heldFetcher() {
  /** @type {((body: unknown) => void)[]} */
  const releases = [];
  const fetcher = vi.fn(
    () =>
      new Promise((resolve) => {
        releases.push((body) => resolve(new Response(JSON.stringify(body))));
      })
  );
  return { fetcher: /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)), releases };
}

describe('the orbit-bump panel', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = MARKUP;
  });

  it('reads the state at an address relative to the panel', async () => {
    const fetcher = answering(state());
    await refresh(root, fetcher);
    expect(/** @type {any} */ (fetcher).mock.calls[0]).toContainEqual('api/state');
    expect(panelOf(root).selection.model).toBe('SR');
  });

  it('opens on the horizontal plane with the auto rule', () => {
    showState(root, state());
    const { selection } = panelOf(root);
    expect(selection.plane).toBe('x');
    expect(selection.rule).toBe('auto');
    const pressed = field(root, 'plane').querySelector('[data-plane="x"]');
    expect(pressed?.getAttribute('aria-pressed')).toBe('true');
  });

  describe('header', () => {
    it('offers a dropdown when two models are offered', () => {
      showState(root, state({ models: [model('SR'), model('BOOSTER', 'idle')] }));
      const select = /** @type {HTMLSelectElement} */ (field(root, 'model'));
      expect(select.hidden).toBe(false);
      expect([...select.options].map((option) => option.value)).toEqual(['SR', 'BOOSTER']);
      expect(select.value).toBe('SR');
      expect(field(root, 'no-model').hidden).toBe(true);
    });

    it('offers no dropdown when one model is offered', () => {
      showState(root, state());
      expect(field(root, 'model').hidden).toBe(true);
      expect(field(root, 'no-model').hidden).toBe(true);
      expect(field(root, 'orm').hidden).toBe(false);
    });

    it('says there is no closed-orbit model when none is offered', () => {
      showState(root, state({ models: [] }));
      expect(field(root, 'no-model').hidden).toBe(false);
      expect(field(root, 'no-model').textContent).toBe('no closed-orbit model');
      expect(field(root, 'model').hidden).toBe(true);
      expect(field(root, 'orm').hidden).toBe(true);
      expect(panelOf(root).selection.model).toBeNull();
    });

    it("shows the selected model's ORM status and age", () => {
      showState(root, state());
      expect(field(root, 'orm-text').textContent).toContain('SR');
      expect(field(root, 'orm-text').textContent).toContain('ready');
      expect(field(root, 'orm-text').textContent).toContain('2 min ago');
      expect(field(root, 'orm').classList.contains('osprey-chip--ok')).toBe(true);
    });

    it('shows a failed ORM build with its reason', () => {
      showState(root, state({ models: [model('SR', 'failed')] }));
      expect(field(root, 'orm').classList.contains('osprey-chip--err')).toBe(true);
      expect(field(root, 'orm-text').textContent).toContain('deck refused');
    });

    it('shows the posture phrase', () => {
      showState(
        root,
        state({ posture: { allowed: false, reason: 'off', phrase: 'Writes are off' } })
      );
      expect(field(root, 'posture').hidden).toBe(false);
      expect(field(root, 'posture').textContent).toBe('Writes are off');
    });
  });

  describe('errors', () => {
    it('keeps the last state and says why when the backend refuses', async () => {
      showState(root, state());
      const fetcher = /** @type {typeof fetch} */ (
        /** @type {unknown} */ (vi.fn(async () => new Response('', { status: 503 })))
      );
      await refresh(root, fetcher);
      expect(field(root, 'error').hidden).toBe(false);
      expect(field(root, 'error').textContent).toContain('503');
      expect(field(root, 'orm-text').textContent).toContain('SR');
    });

    it('clears an earlier error when a state arrives', () => {
      showError(root, 'gone');
      showState(root, state());
      expect(field(root, 'error').hidden).toBe(true);
    });

    it("shows the proxy's text when a recompute is refused", async () => {
      showState(root, state());
      const fetcher = vi.fn(
        async () => new Response('Writes are off for this session', { status: 403 })
      );
      await recompute(root, /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)));
      expect(fetcher.mock.calls[0]).toContainEqual('api/recompute');
      const init = /** @type {RequestInit} */ (/** @type {unknown[]} */ (fetcher.mock.calls[0])[1]);
      expect(JSON.parse(String(init.body))).toEqual({ model: 'SR' });
      expect(field(root, 'error').textContent).toBe('Writes are off for this session');
    });
  });

  describe('hydration', () => {
    it("takes in the agent's plan without discarding it", async () => {
      const fetcher = answering(state({ plan: plan('p1') }));
      await refresh(root, fetcher);
      const { selection } = panelOf(root);
      expect(selection).toEqual({
        plane: 'y',
        model: 'SR',
        targets: { BPM05: 0.1, BPM06: -0.2 },
        rule: 'manual',
        correctors: ['VCM04', 'VCM07'],
        priority: 'closure',
        others: 'hold',
        all_correctors: false,
        sigma_keep: null,
      });
      // Only the state was read: nothing was posted, so nothing was discarded.
      expect(/** @type {any} */ (fetcher).mock.calls).toHaveLength(1);
      expect(/** @type {any} */ (fetcher).mock.calls[0][0]).toBe('api/state');
    });

    it("takes in the plan's rule, priority, other BPMs and all-correctors choice", () => {
      showState(
        root,
        state({
          plan: plan('p2', {
            rule: 'mml:[-3 -1 1 3]',
            priority: 'targets',
            others: 'zero',
            all_correctors: true,
          }),
        })
      );
      const { selection } = panelOf(root);
      expect([selection.rule, selection.correctors]).toEqual(['mml:[-3 -1 1 3]', []]);
      expect([selection.priority, selection.others, selection.all_correctors]).toEqual([
        'targets',
        'zero',
        true,
      ]);
    });

    it('takes in an auto or tight plan by its rule, not as manual', () => {
      showState(root, state({ plan: plan('p3', { rule: 'auto' }) }));
      expect([panelOf(root).selection.rule, panelOf(root).selection.correctors]).toEqual([
        'auto',
        [],
      ]);
      showState(root, state({ plan: plan('p4', { rule: 'mml:[-1 1]' }) }));
      expect(panelOf(root).selection.rule).toBe('mml:[-1 1]');
    });

    it('closes an open confirm card with a notice', () => {
      field(root, 'confirm').hidden = false;
      field(root, 'confirm').textContent = 'Apply bump';
      showState(root, state({ plan: plan('p1') }));
      expect(field(root, 'confirm').hidden).toBe(true);
      expect(field(root, 'notice').hidden).toBe(false);
      expect(field(root, 'notice').textContent).toContain('confirm card was closed');
    });

    it('takes in a plan once, leaving later edits alone', () => {
      showState(root, state({ plan: plan('p1') }));
      clearSelection(root);
      showState(root, state({ plan: plan('p1') }));
      expect(panelOf(root).selection.targets).toEqual({});
    });

    it('never re-hydrates from a null plan after Clear', () => {
      showState(root, state({ plan: plan('p1') }));
      clearSelection(root);
      showState(root, state({ plan: null }));
      showState(root, state({ plan: null }));
      const { selection } = panelOf(root);
      expect(selection.targets).toEqual({});
      expect(selection.correctors).toEqual([]);
      expect(selection.rule).toBe('auto');
    });

    it('does not hydrate while its own Compute is in flight', async () => {
      /** @type {(plan: {id: string}) => void} */
      let answer = () => {};
      const computing = ownCompute(
        root,
        () =>
          new Promise((resolve) => {
            answer = resolve;
          })
      );
      // A poll lands mid-Compute carrying the panel's own new plan.
      await refresh(root, answering(state({ plan: plan('own', { plane: 'x' }) })));
      expect(panelOf(root).selection.rule).toBe('auto');
      expect(panelOf(root).selection.targets).toEqual({});
      answer({ id: 'own' });
      await computing;
      expect(panelOf(root).lastSeenPlanId).toBe('own');
      // The next poll carries the same plan: still nothing re-hydrates.
      await refresh(root, answering(state({ plan: plan('own', { plane: 'x' }) })));
      expect(panelOf(root).selection.rule).toBe('auto');
      expect(panelOf(root).selection.targets).toEqual({});
    });

    it("hydrates an agent's plan that arrives after its own Compute", async () => {
      await ownCompute(root, async () => ({ id: 'own' }));
      await refresh(root, answering(state({ plan: plan('agent') })));
      expect(panelOf(root).lastSeenPlanId).toBe('agent');
      expect(panelOf(root).selection.rule).toBe('manual');
    });

    it('drops an older poll response that lands after a newer one', async () => {
      const { fetcher, releases } = heldFetcher();
      const older = refresh(root, fetcher);
      const newer = refresh(root, fetcher);
      releases[1](state({ plan: null, posture: { allowed: true, phrase: 'newer' } }));
      await newer;
      releases[0](state({ plan: plan('stale'), posture: { allowed: true, phrase: 'older' } }));
      await older;
      expect(field(root, 'posture').textContent).toBe('newer');
      expect(panelOf(root).lastSeenPlanId).toBeNull();
      expect(panelOf(root).selection.targets).toEqual({});
    });
  });
});
