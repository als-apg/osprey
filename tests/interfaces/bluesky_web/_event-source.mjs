// @ts-check
/**
 * An `EventSource` stand-in for the bluesky-web suites: happy-dom has none.
 *
 * Each call returns a fresh class, so a suite owns its own `instances` list
 * and can empty it in its `beforeEach`. A test pushes frames by calling an
 * instance's `onmessage`/`onerror` directly.
 */
export function fakeEventSourceClass() {
  return class FakeEventSource {
    /** @type {FakeEventSource[]} */
    static instances = [];
    /** @param {string} url */
    constructor(url) {
      this.url = url;
      this.closed = false;
      /** @type {((event: {data: string}) => void)|null} */
      this.onmessage = null;
      /** @type {(() => void)|null} */
      this.onerror = null;
      FakeEventSource.instances.push(this);
    }
    close() {
      this.closed = true;
    }
  };
}
