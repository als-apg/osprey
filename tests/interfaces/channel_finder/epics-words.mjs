/**
 * The words of an EPICS control target, installed on import.
 *
 * The channel directory's view tests are written against an EPICS deployment:
 * importing this module first makes every label they render the words the
 * backend serves for an `epics` (or simulator, or stand-in) target, so their
 * expected strings are today's, byte for byte.
 *
 * The import reaches only the labels.js instance loaded with the test file. A
 * test that loads the panel again (`vi.resetModules()`) gets a labels.js that
 * shows the neutral words until words are served, so it sets up the EPICS
 * deployment on the fresh modules: its directory descriptor carries the words
 * ({@link epicsDirectory}), or, for a view mounted without a descriptor, the
 * words are installed again ({@link useEpicsWords}).
 */

import { useWords } from '../../../src/osprey/interfaces/channel_finder/static/js/labels.js';

/** The words `/api/info` serves for an EPICS target. */
export const EPICS_WORDS = Object.freeze({
  channel: 'PV',
  channels: 'PVs',
  server: 'IOC',
  servers: 'IOCs',
  directory: 'All PVs',
  servers_view: 'IOCs',
});

/** The control target the fixtures' answers come from. */
export const EPICS_TARGET = 'va';

/**
 * *descriptor* as `/api/info` serves it on an EPICS deployment: naming its
 * target and carrying that target's words.
 * @template {object} T
 * @param {T} descriptor
 * @returns {T & { target: string, words: Record<string, typeof EPICS_WORDS> }}
 */
export function epicsDirectory(descriptor) {
  return { ...descriptor, target: EPICS_TARGET, words: { [EPICS_TARGET]: EPICS_WORDS } };
}

/**
 * Install the EPICS words in the labels.js the panel's modules load now: the
 * call a test makes after `vi.resetModules()` when it mounts a view without a
 * directory descriptor.
 */
export async function useEpicsWords() {
  const labels = await import('../../../src/osprey/interfaces/channel_finder/static/js/labels.js');
  labels.useWords(EPICS_WORDS);
}

useWords(EPICS_WORDS);
