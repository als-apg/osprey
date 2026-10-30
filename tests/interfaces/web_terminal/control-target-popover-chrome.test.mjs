// @ts-check
/**
 * control-target-popover-chrome.js: the popover's placement beside the
 * sidebar and its document-level dismissals, exercised on their own so the
 * rules hold whatever the popover renders.
 */
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const MOD =
  '/home/kiliev/Documents/Code/LBL/osprey-simple-chat/src/osprey/interfaces/web_terminal/static/js/control-target-popover-chrome.js';

const confirm = vi.hoisted(() => ({ up: false, dismissed: 0 }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => ({
  isConfirmUp: () => confirm.up,
  dismissConfirm: () => {
    confirm.dismissed += 1;
    confirm.up = false;
  },
}));

/** @param {HTMLElement} el @param {Partial<DOMRect>} rect */
function stubRect(el, rect) {
  el.getBoundingClientRect = () =>
    /** @type {DOMRect} */ ({
      x: 0,
      y: 0,
      width: 0,
      height: 0,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      ...rect,
    });
}

beforeEach(() => {
  document.body.innerHTML = `
    <nav id="sidebar"><span id="anchor"><button id="chip"></button><div id="pop"></div></span></nav>
    <div id="outside"></div>`;
  confirm.up = false;
  confirm.dismissed = 0;
});

afterEach(() => {
  vi.restoreAllMocks();
});

const el = (/** @type {string} */ id) => /** @type {HTMLElement} */ (document.getElementById(id));

describe('placeBesideSidebar', () => {
  test('sits 8 px right of the sidebar, level with the chip', async () => {
    const { placeBesideSidebar } = await import(MOD);
    const pop = el('pop');
    stubRect(el('sidebar'), { right: 232 });
    stubRect(el('chip'), { top: 56 });
    Object.defineProperty(pop, 'offsetHeight', { value: 200, configurable: true });
    Object.defineProperty(window, 'innerHeight', { value: 1000, configurable: true });

    placeBesideSidebar(pop, el('chip'), el('sidebar'));

    expect(pop.style.left).toBe('240px');
    expect(pop.style.top).toBe('56px');
  });

  test('moves up so it stays inside the window, never above the top margin', async () => {
    const { placeBesideSidebar } = await import(MOD);
    const pop = el('pop');
    stubRect(el('sidebar'), { right: 56 });
    stubRect(el('chip'), { top: 900 });
    Object.defineProperty(pop, 'offsetHeight', { value: 300, configurable: true });
    Object.defineProperty(window, 'innerHeight', { value: 700, configurable: true });

    placeBesideSidebar(pop, el('chip'), el('sidebar'));
    expect(pop.style.top).toBe('392px');

    Object.defineProperty(pop, 'offsetHeight', { value: 2000, configurable: true });
    placeBesideSidebar(pop, el('chip'), el('sidebar'));
    expect(pop.style.top).toBe('8px');
  });

  test('clears the inline position when the chip is not in a sidebar', async () => {
    const { placeBesideSidebar } = await import(MOD);
    const pop = el('pop');
    pop.style.left = '240px';
    pop.style.top = '56px';

    placeBesideSidebar(pop, el('chip'), null);

    expect(pop.style.left).toBe('');
    expect(pop.style.top).toBe('');
  });
});

describe('attachDismissals', () => {
  /** @returns {Promise<{close: import('vitest').Mock, focusChip: import('vitest').Mock, detach: () => void}>} */
  async function attached() {
    const { attachDismissals } = await import(MOD);
    const close = vi.fn();
    const focusChip = vi.fn();
    const detach = attachDismissals({ anchor: () => el('anchor'), close, focusChip });
    return { close, focusChip, detach };
  }

  test('an outside click closes; a click inside the anchor or a confirm does not', async () => {
    const { close, detach } = await attached();

    el('chip').click();
    expect(close).not.toHaveBeenCalled();

    const overlay = document.createElement('div');
    overlay.className = 'posture-modal-overlay';
    overlay.innerHTML = '<button id="confirm-btn"></button>';
    document.body.appendChild(overlay);
    el('confirm-btn').click();
    expect(close).not.toHaveBeenCalled();

    el('outside').click();
    expect(close).toHaveBeenCalledTimes(1);
    detach();
  });

  test('Escape dismisses an open confirm first; with none up it closes and focuses the chip', async () => {
    const { close, focusChip, detach } = await attached();
    const escape = () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    confirm.up = true;
    escape();
    expect(confirm.dismissed).toBe(1);
    expect(close).not.toHaveBeenCalled();

    escape();
    expect(close).toHaveBeenCalledTimes(1);
    expect(focusChip).toHaveBeenCalledTimes(1);
    detach();
  });

  test('a resize or the sidebar collapsing closes a fixed popover', async () => {
    const { close, detach } = await attached();

    window.dispatchEvent(new Event('resize'));
    document.dispatchEvent(new CustomEvent('osprey-collapsible-change'));
    expect(close).toHaveBeenCalledTimes(2);
    detach();
  });

  test('detaching unwires every listener', async () => {
    const { close, detach } = await attached();
    detach();

    el('outside').click();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    window.dispatchEvent(new Event('resize'));
    expect(close).not.toHaveBeenCalled();
  });
});
