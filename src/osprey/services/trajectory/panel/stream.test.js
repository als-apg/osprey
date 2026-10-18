// @ts-check
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EVENTS_URL, EVENT_TYPES, openStream } from './stream.js';

/** A fake EventSource recording each instance and its listeners. */
class FakeSource {
  /** @type {FakeSource[]} */
  static instances = [];

  /** @param {string} url */
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.closed = false;
    /** @type {((ev: Event) => void) | null} */
    this.onopen = null;
    /** @type {((ev: Event) => void) | null} */
    this.onerror = null;
    /** @type {Map<string, ((ev: MessageEvent) => void)[]>} */
    this.listeners = new Map();
    FakeSource.instances.push(this);
  }

  /**
   * @param {string} type
   * @param {(ev: MessageEvent) => void} listener
   */
  addEventListener(type, listener) {
    const list = this.listeners.get(type) ?? [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  close() {
    this.closed = true;
    this.readyState = 2;
  }

  open() {
    this.readyState = 1;
    if (this.onopen) this.onopen(new Event('open'));
  }

  /** @param {number} readyState */
  fail(readyState) {
    this.readyState = readyState;
    if (this.onerror) this.onerror(new Event('error'));
  }

  /**
   * @param {string} type
   * @param {string} data
   */
  emit(type, data) {
    const ev = new MessageEvent(type, { data });
    for (const listener of this.listeners.get(type) ?? []) listener(ev);
  }
}

/** @param {Partial<import('./stream.js').StreamOptions>} [extra] */
function start(extra = {}) {
  const onEvent = vi.fn();
  const onState = vi.fn();
  const handle = openStream({ onEvent, onState, EventSourceCtor: FakeSource, ...extra });
  return { onEvent, onState, handle };
}

function latest() {
  return FakeSource.instances[FakeSource.instances.length - 1];
}

beforeEach(() => {
  FakeSource.instances = [];
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('panel event stream', () => {
  it('opens the page-relative api/events URL', () => {
    start();
    expect(EVENTS_URL).toBe('api/events');
    expect(FakeSource.instances).toHaveLength(1);
    expect(latest().url).toBe('api/events');
    expect(latest().url.startsWith('/')).toBe(false);
  });

  it('subscribes to every service event type by name', () => {
    start();
    expect([...latest().listeners.keys()].sort()).toEqual(
      [
        'apply',
        'continuous_ended',
        'shot',
        'snapshot',
        'step',
        'stopped',
        'taken_over',
      ].sort(),
    );
    expect([...EVENT_TYPES].sort()).toEqual([...latest().listeners.keys()].sort());
  });

  it.each(EVENT_TYPES)('dispatches a parsed %s payload', (type) => {
    const { onEvent } = start();
    const payload = { type, value: 42 };
    latest().emit(type, JSON.stringify(payload));
    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledWith(type, payload);
  });

  it('dispatches a null step payload', () => {
    const { onEvent } = start();
    latest().emit('step', 'null');
    expect(onEvent).toHaveBeenCalledWith('step', null);
  });

  it('drops a malformed frame without throwing', () => {
    const { onEvent } = start();
    expect(() => latest().emit('shot', '{not json')).not.toThrow();
    expect(onEvent).not.toHaveBeenCalled();
  });

  it('reports connecting, then connected on open', () => {
    const { onState } = start();
    latest().open();
    expect(onState.mock.calls.map((c) => c[0])).toEqual(['connecting', 'connected']);
  });

  it('never races the browser retry while the source is still CONNECTING', () => {
    const { onState } = start();
    latest().open();
    latest().fail(0);
    expect(onState).toHaveBeenLastCalledWith('disconnected');
    vi.advanceTimersByTime(60000);
    expect(FakeSource.instances).toHaveLength(1);
  });

  it('reconnects with backoff after an error that leaves the source CLOSED', () => {
    const { onState, onEvent } = start();
    const first = latest();
    first.open();
    first.fail(2);
    expect(onState).toHaveBeenLastCalledWith('disconnected');

    vi.advanceTimersByTime(999);
    expect(FakeSource.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeSource.instances).toHaveLength(2);
    expect(latest().url).toBe('api/events');
    expect(onState).toHaveBeenLastCalledWith('disconnected');

    // Second failure before an open doubles the delay.
    latest().fail(2);
    vi.advanceTimersByTime(1999);
    expect(FakeSource.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeSource.instances).toHaveLength(3);

    latest().open();
    expect(onState).toHaveBeenLastCalledWith('connected');

    // Events from the replaced source are ignored; the new one dispatches.
    first.emit('shot', '{"type":"shot"}');
    expect(onEvent).not.toHaveBeenCalled();
    latest().emit('shot', '{"type":"shot"}');
    expect(onEvent).toHaveBeenCalledWith('shot', { type: 'shot' });

    // A successful open resets the backoff.
    latest().fail(2);
    vi.advanceTimersByTime(1000);
    expect(FakeSource.instances).toHaveLength(4);
  });

  it('caps the backoff at thirty seconds', () => {
    start();
    for (let i = 0; i < 10; i++) {
      latest().fail(2);
      vi.advanceTimersByTime(30000);
    }
    const count = FakeSource.instances.length;
    expect(count).toBe(11);
    latest().fail(2);
    vi.advanceTimersByTime(29999);
    expect(FakeSource.instances).toHaveLength(count);
    vi.advanceTimersByTime(1);
    expect(FakeSource.instances).toHaveLength(count + 1);
  });

  it('close stops the source and any pending reconnect', () => {
    const { handle, onEvent } = start();
    const first = latest();
    first.fail(2);
    handle.close();
    expect(first.closed).toBe(true);
    vi.advanceTimersByTime(60000);
    expect(FakeSource.instances).toHaveLength(1);
    first.emit('snapshot', '{}');
    expect(onEvent).not.toHaveBeenCalled();
  });
});
