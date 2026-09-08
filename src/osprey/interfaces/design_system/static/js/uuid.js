// @ts-check
/* A v4-shaped identifier, on every deployment topology.
 *
 * `crypto.randomUUID` exists only in a secure context — HTTPS, or localhost.
 * A deployment served as plain HTTP on a non-loopback host is a documented
 * OSPREY topology (`auth.allow_insecure_http`, for a facility terminator that
 * fronts this app), and there the property is simply absent: reading it throws
 * a `TypeError` and whichever surface was minting an id never mounts.
 *
 * `crypto.getRandomValues` IS available in an insecure context, so the id can
 * still be a real random one. `Math.random` is the last resort, for an engine
 * that has neither.
 *
 * The grammar matters as much as the randomness: server-side stores key on a
 * bare `8-4-4-4-12` UUID, and an id of some other shape does not fail — it
 * silently lands somewhere else. So every fallback here produces the same
 * shape as the API it stands in for.
 */

/**
 * Sixteen random bytes, from the strongest source this engine offers.
 *
 * @returns {Uint8Array}
 */
function randomBytes() {
  const bytes = new Uint8Array(16);
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes);
    return bytes;
  }
  for (let i = 0; i < bytes.length; i += 1) {
    bytes[i] = Math.floor(Math.random() * 256);
  }
  return bytes;
}

/**
 * A random RFC 4122 version-4 UUID, lowercase, unbracketed.
 *
 * Uses `crypto.randomUUID` where it exists and builds the same shape by hand
 * where it does not.
 *
 * @returns {string} `8-4-4-4-12` lowercase hex.
 */
export function uuid4() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  const bytes = randomBytes();
  // Version 4, variant 10xx — the two fields a v4 UUID pins.
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20)
  ].join('-');
}
