/**
 * Shared harness for the pre-paint boot scripts (mode-boot.js, rail-boot.js,
 * theme-boot.js).
 *
 * Those files are non-module, dependency-free IIFEs — they export nothing and
 * do their whole job on load — so a test cannot import them. Instead each
 * scenario arranges window.location / localStorage / the <html> attributes and
 * then re-executes the exact on-disk source against the happy-dom globals.
 *
 * Two details are load-bearing and easy to get wrong when this is copied by
 * hand, which is why it lives here once:
 *
 *   - `import.meta.dirname` (a plain string) rather than
 *     `fileURLToPath(new URL(...))`: happy-dom overrides the global URL, which
 *     breaks the fileURLToPath spelling under this environment.
 *   - `window`/`document` passed as explicit `new Function` parameters, so the
 *     source's free references bind to the test's DOM without a global `eval`.
 *
 * The source is re-read from disk on every call, so a test that runs the same
 * boot script repeatedly always executes what is actually on disk.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

/** design_system's served JS directory, relative to this file. */
const BOOT_SCRIPT_DIR = join(
  import.meta.dirname,
  '../../../../src/osprey/interfaces/design_system/static/js'
);

/**
 * Execute a pre-paint boot IIFE against the current happy-dom globals.
 *
 * @param {string} relPath path relative to design_system/static/js —
 *   e.g. `'mode-boot.js'`.
 * @returns {void}
 */
export function runBootScript(relPath) {
  const source = readFileSync(join(BOOT_SCRIPT_DIR, relPath), 'utf8');
  const boot = new Function('window', 'document', source);
  boot(globalThis.window, globalThis.document);
}
