// @ts-check
import { vi } from 'vitest';

/**
 * Force the OS colour-scheme preference, so every 'auto' resolution lands on
 * a known id. Without this an assertion would pass for the wrong reason
 * whenever the host's own OS preference happened to agree.
 *
 * @param {boolean} prefersDark
 */
export function stubOSPreference(prefersDark) {
  vi.stubGlobal('matchMedia', (/** @type {string} */ query) => ({
    matches: query.includes('dark') ? prefersDark : !prefersDark,
    media: query,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
  }));
}
