// @ts-check
/**
 * The `EventSource` stand-in for suites whose module opens a stream: happy-dom
 * has none.
 *
 * {@link FakeEventSource} records every source opened, in order, on its class's
 * `opened` list. Frames are delivered as the browser would, through a source's
 * `onmessage` with a string payload. A source is born connecting
 * (`readyState` 0), or open (`readyState` 1) when the suite asks for that;
 * `close()` marks it `closed` and `readyState` 2.
 *
 * {@link stubEventSource} installs it as the global with an empty record;
 * `vi.unstubAllGlobals()` undoes it. A module that takes its constructor as an
 * argument gets one of its own from {@link fakeEventSourceClass}.
 */

import { vi } from 'vitest';

export class FakeEventSource {
  /** @type {FakeEventSource[]} */
  static opened = [];
  /** The `readyState` a new source is born with. */
  static initialReadyState = 0;
  /** @param {string} url */
  constructor(url) {
    const cls = /** @type {typeof FakeEventSource} */ (this.constructor);
    this.url = url;
    this.readyState = cls.initialReadyState;
    this.closed = false;
    /** @type {((event: {data: string}) => void) | null} */
    this.onmessage = null;
    /** @type {(() => void) | null} */
    this.onopen = null;
    /** @type {(() => void) | null} */
    this.onerror = null;
    cls.opened.push(this);
  }
  close() {
    this.closed = true;
    this.readyState = 2;
  }
}

/**
 * A fresh subclass with its own `opened` list, for a suite that hands the
 * constructor to the module and empties the list in its `beforeEach`.
 */
export function fakeEventSourceClass() {
  return class extends FakeEventSource {
    /** @type {FakeEventSource[]} */
    static opened = [];
  };
}

/**
 * Install {@link FakeEventSource} as the global, with no source recorded.
 *
 * @param {{ open?: boolean }} [options] - `open: true` has each source born
 *   open, for a module that acts on a stream it finds already open.
 * @returns {{ opened: FakeEventSource[], emit: (payload: unknown) => void }}
 *   the sources opened since, and `emit`, which delivers *payload* to every one
 *   of them — JSON text as the wire carries it, or a string sent as-is.
 */
export function stubEventSource({ open = false } = {}) {
  FakeEventSource.opened = [];
  FakeEventSource.initialReadyState = open ? 1 : 0;
  vi.stubGlobal('EventSource', FakeEventSource);
  const opened = FakeEventSource.opened;
  return {
    opened,
    emit: (payload) => {
      const data = typeof payload === 'string' ? payload : JSON.stringify(payload);
      for (const source of opened) source.onmessage?.({ data });
    },
  };
}
