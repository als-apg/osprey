/**
 * The words the channel directory's views show an operator for a channel and
 * for the server that serves it, each stated once.
 *
 * Every view, column header and breadcrumb that names a channel or a server
 * reads its word here, so the control system's own vocabulary has one place to
 * land. The served directory label (`state.directory.label`) wins where a view
 * has one; {@link LABELS.directory} is the word shown before it arrives.
 */

export const LABELS = Object.freeze({
  /** The directory view, as its tab and breadcrumbs name it. */
  directory: 'All PVs',
  /** The server fleet view's tab. */
  serversView: 'IOCs',
  /** One server: a column header, a facet title, a fact. */
  server: 'IOC',
  /** One channel: a column header. */
  channel: 'PV',
  /** More than one channel: a count, a tile, a column header. */
  channels: 'PVs',
});
