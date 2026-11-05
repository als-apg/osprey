// @ts-check
/**
 * Unit tests for the programs' section of the control-target popover
 * (write-authorization-panel.js):
 *   npx vitest run tests/interfaces/web_terminal/write-authorization-panel.test.mjs
 *
 * The render functions are pure: a listing and an open card in, an element
 * out. These tests hand them the route's payloads and check what a person is
 * shown before approving — channels, ranges, the limits bound of each range,
 * current values, steps, expiry, holder, completion, the limits note — and
 * which handler each control runs. The wired section is driven once against a
 * stubbed fetch, to pin the routes it posts to.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  APPROVING_SENTENCE,
  READING_SENTENCE,
  SELF_ISSUED_SENTENCE,
  limitsText,
  programsSection,
  refreshPrograms,
  relativeTime,
  renderPrograms,
  renderRequestCard,
  resetPrograms,
  valueText,
} from '../../../src/osprey/interfaces/web_terminal/static/js/write-authorization-panel.js';

const ID = '0123456789abcdef0123456789abcdef';
const GRANT = 'fedcba9876543210fedcba9876543210';

/** A request card as GET /api/terminal/write-authorizations/requests/{id} answers it. */
const cardOf = (/** @type {object} */ o = {}) => ({
  id: ID,
  holder: 'optimizer',
  owner: 'alice',
  target: 'standin',
  label: 'run-42',
  completion: 'apply_best',
  steps: 12,
  lease_s: 30,
  expires_at: '2026-11-03T08:00:00+00:00',
  expires_in_s: 7080,
  channels: [
    { address: 'SR:A', min: 1, max: 9, limit_min: 0, limit_max: 10, problem: null, current: 4.5 },
    { address: 'SR:B', min: -1, max: 1, limit_min: -5, limit_max: 5, problem: null, current: null },
  ],
  approvable: true,
  refusal: null,
  limits_note: 'A write the limits database refuses still costs its step.',
  ...o,
});

const handlers = () => ({
  onReview: vi.fn(),
  onApprove: vi.fn(),
  onRevoke: vi.fn(),
  onClose: vi.fn(),
});

describe('the approve card', () => {
  test('shows every channel with its range, limits and current value', () => {
    const node = renderRequestCard(/** @type {any} */ (cardOf()), handlers());
    const rows = [...node.querySelectorAll('.wa-channel')];
    expect(rows.map((r) => r.querySelector('.wa-address')?.textContent)).toEqual(['SR:A', 'SR:B']);
    expect(rows[0].querySelector('.wa-range')?.textContent).toBe('[1, 9]');
    expect(rows[0].querySelector('.wa-limits')?.textContent).toBe('[0, 10]');
    expect(rows[0].querySelector('.wa-current')?.textContent).toBe('4.5');
    expect(rows[1].querySelector('.wa-current')?.textContent).toBe('not read');
    const text = node.textContent ?? '';
    for (const fact of [
      '12 write calls',
      'applies the best point found',
      'in 1 h 58 min',
      '30 s',
      'optimizer',
    ]) {
      expect(text).toContain(fact);
    }
    expect(node.querySelector('.wa-note')?.textContent).toContain('still costs its step');
    const expires = [...node.querySelectorAll('dd')].find(
      (dd) => dd.textContent === 'in 1 h 58 min'
    );
    expect(expires?.title).toBe('2026-11-03T08:00:00+00:00');
  });

  test("Approve runs the handler with the request id; the buttons are the popover's", () => {
    const spies = handlers();
    const node = renderRequestCard(/** @type {any} */ (cardOf()), spies);
    const approve = /** @type {HTMLButtonElement} */ (node.querySelector('.wa-approve'));
    expect(approve.disabled).toBe(false);
    expect([...approve.classList]).toEqual(expect.arrayContaining(['osprey-btn', 'primary']));
    expect(node.querySelector('.wa-close')?.classList.contains('osprey-btn')).toBe(true);
    approve.click();
    expect(spies.onApprove).toHaveBeenCalledWith(ID);
  });

  test('an expiry reads as the time left', () => {
    expect(relativeTime(7080)).toBe('in 1 h 58 min');
    expect(relativeTime(7200)).toBe('in 2 h');
    expect(relativeTime(240)).toBe('in 4 min');
    expect(relativeTime(30)).toBe('in 30 s');
    expect(relativeTime(0)).toBe('now');
  });

  test('a card the server refuses keeps Approve disabled and says why', () => {
    const refused = cardOf({
      approvable: false,
      refusal: { reason: 'outside_limits', phrase: 'SR:A asks for a range outside its limits.' },
      channels: [
        {
          address: 'SR:A',
          min: -1,
          max: 9,
          limit_min: 0,
          limit_max: 10,
          problem: 'outside_limits',
        },
      ],
    });
    const node = renderRequestCard(/** @type {any} */ (refused), handlers());
    expect(/** @type {HTMLButtonElement} */ (node.querySelector('.wa-approve')).disabled).toBe(
      true
    );
    expect(node.querySelector('.wa-refusal')?.textContent).toContain('outside its limits');
    expect(node.querySelector('.wa-channel')?.getAttribute('data-problem')).toBe('outside_limits');
  });

  test('a ceiling refusal names the tier', () => {
    const node = renderRequestCard(
      /** @type {any} */ (
        cardOf({ approvable: false, refusal: { reason: 'tier_caps', ceiling: 'never' } })
      ),
      handlers()
    );
    expect(node.querySelector('.wa-refusal')?.textContent).toContain('never');
  });

  test('values and limits read as they are', () => {
    expect(valueText('OPEN')).toBe('"OPEN"');
    expect(valueText(0)).toBe('0');
    expect(limitsText({ address: 'x', min: 0, max: 1, limit_min: null, limit_max: null })).toBe(
      'no numeric limits'
    );
  });

  test('a channel the limits database rules out says why, not that it has no limits', () => {
    const channel = { address: 'x', min: 0, max: 1, limit_min: 0, limit_max: 1 };
    expect(limitsText({ ...channel, problem: 'not_writable' })).toBe('not writable');
    expect(
      limitsText({ ...channel, limit_min: null, limit_max: null, problem: 'not_in_limits' })
    ).toBe('not in the limits database');
    expect(limitsText({ ...channel, problem: 'outside_limits' })).toBe('[0, 1]');
  });

  test('Approve waits, disabled, while the approval is sent', () => {
    const node = renderRequestCard(/** @type {any} */ (cardOf()), handlers(), true);
    const approve = /** @type {HTMLButtonElement} */ (node.querySelector('.wa-approve'));
    expect(approve.textContent).toBe(APPROVING_SENTENCE);
    expect(approve.disabled).toBe(true);
  });
});

describe('the section', () => {
  test('is absent with nothing pending and nothing held', () => {
    expect(renderPrograms({ requests: [], authorizations: [] }, null, null, handlers())).toBeNull();
    expect(renderPrograms(null, null, null, handlers())).toBeNull();
  });

  test('lists pending requests with Review and authorizations in force with Revoke', () => {
    const spies = handlers();
    const node = renderPrograms(
      /** @type {any} */ ({
        requests: [cardOf()],
        authorizations: [
          {
            id: GRANT,
            holder: 'optimizer',
            target: 'standin',
            label: 'run-41',
            steps: 5,
            used: 2,
            expires_at: 'later',
            expires_in_s: 600,
            in_force: true,
          },
          {
            id: 'x'.repeat(32),
            holder: 'old',
            target: 'va',
            label: '',
            steps: 1,
            expires_at: 'then',
            in_force: false,
          },
        ],
      }),
      null,
      null,
      spies
    );
    expect(node).not.toBeNull();
    const section = /** @type {HTMLElement} */ (node);
    expect(section.querySelectorAll('.wa-request')).toHaveLength(1);
    expect(section.querySelectorAll('.wa-held')).toHaveLength(1);
    const held = /** @type {HTMLElement} */ (section.querySelector('.wa-held .wa-what'));
    expect(held.textContent).toContain('2/5 used');
    expect(held.textContent).toContain('ends in 10 min');
    expect(held.title).toBe('Expires later');
    expect(section.querySelector('.wa-review')?.classList.contains('osprey-btn')).toBe(true);
    /** @type {HTMLButtonElement} */ (section.querySelector('.wa-review')).click();
    expect(spies.onReview).toHaveBeenCalledWith(ID);
    /** @type {HTMLButtonElement} */ (section.querySelector('.wa-revoke')).click();
    expect(spies.onRevoke).toHaveBeenCalledWith(GRANT);
  });

  test('a self-issued authorization is listed without Revoke and says how it ends', () => {
    const spies = handlers();
    const node = /** @type {HTMLElement} */ (
      renderPrograms(
        /** @type {any} */ ({
          requests: [],
          authorizations: [
            {
              id: GRANT,
              holder: 'tuning-scripts',
              target: 'standin',
              label: 'scan',
              steps: 50,
              expires_at: 'later',
              in_force: true,
              self_issued: true,
              revocable: false,
            },
          ],
        }),
        null,
        null,
        spies
      )
    );
    const line = /** @type {HTMLElement} */ (node.querySelector('.wa-held'));
    expect(line.dataset.selfIssued).toBe('true');
    expect(line.querySelector('.wa-revoke')).toBeNull();
    expect(line.querySelector('.wa-self-issued')?.textContent).toBe(SELF_ISSUED_SENTENCE);
    expect(SELF_ISSUED_SENTENCE).toContain('away from allowed');
  });

  test('an open card replaces its own line', () => {
    const node = /** @type {HTMLElement} */ (
      renderPrograms(
        /** @type {any} */ ({ requests: [cardOf()], authorizations: [] }),
        /** @type {any} */ (cardOf()),
        null,
        handlers()
      )
    );
    expect(node.querySelectorAll('.wa-card')).toHaveLength(1);
    expect(node.querySelectorAll('.wa-request')).toHaveLength(0);
  });
});

describe('the wired section', () => {
  /** @type {Array<{url: string, method: string, body: any}>} */
  let calls = [];
  /** What the revoke route answers. */
  let revoked = true;

  beforeEach(() => {
    calls = [];
    revoked = true;
    resetPrograms();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init = {}) => {
        const method = init.method ?? 'GET';
        calls.push({ url, method, body: init.body ? JSON.parse(String(init.body)) : null });
        const body =
          url.endsWith('/api/terminal/write-authorizations') && method === 'GET'
            ? {
                requests: [cardOf()],
                authorizations: [
                  {
                    id: GRANT,
                    holder: 'optimizer',
                    target: 'standin',
                    label: '',
                    steps: 5,
                    expires_at: 'later',
                    in_force: true,
                  },
                ],
              }
            : url.includes('/requests/')
              ? cardOf()
              : url.endsWith('/revoke')
                ? { id: GRANT, revoked }
                : { authorization: { id: ID } };
        return { ok: true, status: 200, json: async () => body };
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    resetPrograms();
  });

  test('Review says it is reading, reads the card, and Approve posts the request id', async () => {
    await refreshPrograms();
    const rerender = vi.fn();
    let section = /** @type {HTMLElement} */ (programsSection('sess', rerender));
    /** @type {HTMLButtonElement} */ (section.querySelector('.wa-review')).click();
    expect(rerender).toHaveBeenCalledTimes(1);
    section = /** @type {HTMLElement} */ (programsSection('sess', rerender));
    expect(section.querySelector('.wa-loading')?.textContent).toBe(READING_SENTENCE);
    expect(section.querySelector('.wa-review')).toBeNull();
    await vi.waitFor(() => expect(rerender).toHaveBeenCalledTimes(2));
    expect(calls.at(-1)?.url).toContain(`/api/terminal/write-authorizations/requests/${ID}`);

    section = /** @type {HTMLElement} */ (programsSection('sess', rerender));
    expect(section.querySelector('.wa-loading')).toBeNull();
    /** @type {HTMLButtonElement} */ (section.querySelector('.wa-approve')).click();
    section = /** @type {HTMLElement} */ (programsSection('sess', rerender));
    expect(section.querySelector('.wa-approve')?.textContent).toBe(APPROVING_SENTENCE);
    await vi.waitFor(() => expect(calls.some((c) => c.method === 'POST')).toBe(true));
    const post = calls.find((c) => c.method === 'POST');
    expect(post?.url).toContain('/api/terminal/write-authorizations');
    expect(post?.body).toEqual({ request_id: ID, session_id: 'sess' });
  });

  test.each([
    [true, 'Revoked: its next write is refused.'],
    [false, 'Already stopped.'],
  ])('Revoke reads the answer (revoked %s)', async (answer, said) => {
    revoked = answer;
    await refreshPrograms();
    const rerender = vi.fn();
    const section = /** @type {HTMLElement} */ (programsSection('sess', rerender));
    /** @type {HTMLButtonElement} */ (section.querySelector('.wa-revoke')).click();
    await vi.waitFor(() => expect(rerender).toHaveBeenCalled());
    const after = /** @type {HTMLElement} */ (programsSection('sess', rerender));
    expect(after.querySelector('.wa-status')?.textContent).toBe(said);
  });

  test('a read the popover closed under does not bring the listing back', async () => {
    /** @type {(() => void)|null} */
    let release = null;
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise((resolve) => {
            release = () =>
              resolve({
                ok: true,
                status: 200,
                json: async () => ({ requests: [cardOf()], authorizations: [] }),
              });
          })
      )
    );
    const read = refreshPrograms();
    await vi.waitFor(() => expect(release).not.toBeNull());
    resetPrograms();
    /** @type {any} */ (release)();
    await read;
    expect(programsSection('sess', vi.fn())).toBeNull();
  });
});
