// @ts-check
/* OSPREY Web Terminal — the chrome's glyphs, ahead of the application
 *
 * Draws the chrome's glyphs and key hints as soon as the document is parsed.
 * index.html loads it ahead of app.js: its whole graph is four small files, so
 * the sidebar reads as a sidebar before the application starts. sidebar.js fills the same placeholders again at start-up,
 * which is a no-op for a glyph already drawn.
 */

import { hydrateIcons, paintChords } from './shell-icons.js';

hydrateIcons(document);
paintChords(document);
