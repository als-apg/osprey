import { beforeEach, describe, expect, it, vi } from 'vitest';
import { refresh, showError, showState } from '../panel/panel.js';

const FIELDS = ['channel', 'readback', 'target', 'error']
  .map((name) => `<p data-field="${name}"></p>`)
  .join('');
const STATE = { channel: 'CH:SP', target: 'va', readback: 1.5 };

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

describe('the knob panel', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = FIELDS;
  });

  it('shows a state as text', () => {
    showState(root, { ...STATE, channel: '<b>CH</b>' });
    expect(field(root, 'channel').textContent).toBe('<b>CH</b>');
    expect(field(root, 'channel').querySelector('b')).toBeNull();
    expect(field(root, 'readback').textContent).toBe('1.5');
    expect(field(root, 'target').textContent).toBe('va');
  });

  it('clears an earlier error when a state arrives', () => {
    showError(root, 'gone');
    expect(field(root, 'error').hidden).toBe(false);
    showState(root, STATE);
    expect(field(root, 'error').hidden).toBe(true);
  });

  it('reads the state at an address relative to the panel', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify(STATE)));
    await refresh(root, /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)));
    expect(fetcher.mock.calls[0]).toContainEqual('api/state');
    expect(field(root, 'readback').textContent).toBe('1.5');
  });

  it('keeps the last state and says why when the backend refuses', async () => {
    showState(root, STATE);
    const fetcher = vi.fn(async () => new Response('', { status: 503 }));
    await refresh(root, /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)));
    expect(field(root, 'readback').textContent).toBe('1.5');
    expect(field(root, 'error').textContent).toContain('503');
  });

  it('says why when the backend cannot be reached', async () => {
    const fetcher = vi.fn(async () => {
      throw new Error('no route');
    });
    await refresh(root, /** @type {typeof fetch} */ (/** @type {unknown} */ (fetcher)));
    expect(field(root, 'error').textContent).toBe('no route');
  });
});
