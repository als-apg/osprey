// @ts-check
/**
 * The EventSource stand-in for suites whose module opens a real stream.
 *
 * happy-dom ships no EventSource. This class stub keeps the module's stream
 * path intact — api.js's shared stream included: frames are delivered as the
 * browser would, as `onmessage` with a string payload — and records every
 * socket opened, in order, on `FakeEventSource.opened`. It is born open
 * (`readyState` 1); `close()` marks it `closed` and `readyState` 2.
 *
 * `stubEventSource()` installs it with an empty record; `vi.unstubAllGlobals()`
 * undoes it.
 */

import { vi } from 'vitest';

export class FakeEventSource {
  /** @type {FakeEventSource[]} */
  static opened = [];
  /** @param {string} url */
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.closed = false;
    /** @type {((event: {data: string}) => void) | null} */
    this.onmessage = null;
    /** @type {(() => void) | null} */
    this.onopen = null;
    /** @type {(() => void) | null} */
    this.onerror = null;
    FakeEventSource.opened.push(this);
  }
  close() {
    this.closed = true;
    this.readyState = 2;
  }
}

/** Install {@link FakeEventSource} as the global, with no socket recorded. */
export function stubEventSource() {
  FakeEventSource.opened = [];
  vi.stubGlobal('EventSource', FakeEventSource);
}
