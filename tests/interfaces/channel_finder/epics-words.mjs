/**
 * The words of an EPICS control target, installed on import.
 *
 * The channel directory's view tests are written against an EPICS deployment:
 * importing this module first makes every label they render the words the
 * backend serves for an `epics` (or simulator, or stand-in) target, so their
 * expected strings are today's, byte for byte.
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

useWords(EPICS_WORDS);
