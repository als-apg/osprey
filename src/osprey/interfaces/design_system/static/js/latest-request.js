// @ts-check
/* Newest request wins.
 *
 * Only the newest request of one kind is current. Starting one aborts and
 * retires the previous, so a reply that the earlier request still delivers is
 * recognised as stale and dropped; a cancel retires the current request without
 * starting another. A consumer keeps one instance per kind of request and asks
 * its token, once the reply is in, whether that reply still applies.
 */

/**
 * One started request: the signal to hand to its fetch, and whether it is
 * still the newest of its kind.
 * @typedef {object} LatestToken
 * @property {AbortSignal} signal - aborted when a newer request starts or on cancel
 * @property {() => boolean} isCurrent - true until a newer request starts or a cancel
 */

/**
 * The newest-request-wins rule for one kind of request.
 * @returns {{begin: () => LatestToken, cancel: () => void}}
 */
export function createLatestRequest() {
  let generation = 0;
  /** @type {AbortController|null} */
  let controller = null;

  /** Abort the request in flight, if any, and retire it. */
  function cancel() {
    controller?.abort();
    controller = null;
    generation += 1;
  }

  /**
   * Start a request, superseding the one in flight.
   * @returns {LatestToken}
   */
  function begin() {
    cancel();
    const mine = generation;
    controller = new AbortController();
    return { signal: controller.signal, isCurrent: () => mine === generation };
  }

  return { begin, cancel };
}
