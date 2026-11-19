// @ts-check
/**
 * The words the channel directory's views show an operator for a channel and
 * for the server that serves it, as the control system names them.
 *
 * The backend resolves the words of each control target from its connector
 * (PV and IOC on EPICS, attribute and device server on Tango, ...) and serves
 * them in the `/api/info` directory descriptor; {@link useWords} installs the
 * words of the target an answer comes from, and every view, column header,
 * breadcrumb and sentence reads them here at render time. Before any words are
 * served, and for a control system that states none, the neutral words hold.
 *
 * The nouns are spelled as they read inside a sentence; a heading starts with
 * {@link heading} of one. Sentences are phrased so they need no article ("no
 * IOC", "this server", "every attribute").
 */

/**
 * @typedef {{
 *   channel: string,
 *   channels: string,
 *   server: string,
 *   servers: string,
 *   directory: string,
 *   serversView: string,
 * }} Words
 */

/**
 * The words as `/api/info` serves them, one entry per control target.
 * @typedef {{
 *   channel: string,
 *   channels: string,
 *   server: string,
 *   servers: string,
 *   directory: string,
 *   servers_view: string,
 * }} ServedWords
 */

/** The words of a control system that states none. @type {Readonly<Words>} */
export const NEUTRAL_WORDS = Object.freeze({
  channel: 'channel',
  channels: 'channels',
  server: 'server',
  servers: 'servers',
  directory: 'Channels',
  serversView: 'Servers',
});

/** @type {Readonly<Words>} */
let current = NEUTRAL_WORDS;

/**
 * The words shown now, read at render time.
 * @type {Readonly<Words>}
 */
export const LABELS = Object.freeze({
  /** One channel: a column header, a sentence. */
  get channel() {
    return current.channel;
  },
  /** More than one channel: a count, a tile, a column header. */
  get channels() {
    return current.channels;
  },
  /** One server: a column header, a facet title, a fact. */
  get server() {
    return current.server;
  },
  /** More than one server: a count, a sentence. */
  get servers() {
    return current.servers;
  },
  /** The directory view, as its tab and breadcrumbs name it. */
  get directory() {
    return current.directory;
  },
  /** The server fleet view's tab. */
  get serversView() {
    return current.serversView;
  },
});

/** @param {unknown} value */
function word(value) {
  return typeof value === 'string' && value !== '';
}

/**
 * Show *served*'s words from now on; anything that is not a whole set of words
 * shows the neutral ones.
 * @param {ServedWords|null|undefined} served
 */
export function useWords(served) {
  const keys = /** @type {(keyof ServedWords)[]} */ ([
    'channel',
    'channels',
    'server',
    'servers',
    'directory',
    'servers_view',
  ]);
  if (!served || !keys.every((key) => word(served[key]))) {
    current = NEUTRAL_WORDS;
    return;
  }
  current = Object.freeze({
    channel: served.channel,
    channels: served.channels,
    server: served.server,
    servers: served.servers,
    directory: served.directory,
    serversView: served.servers_view,
  });
}

/**
 * *noun* starting a heading: its first letter upper case (`IOC`, `Server`).
 * @param {string} noun
 * @returns {string}
 */
export function heading(noun) {
  return noun.charAt(0).toUpperCase() + noun.slice(1);
}
