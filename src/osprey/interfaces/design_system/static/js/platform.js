// @ts-check
/* What the viewer's keyboard actually looks like.
 *
 * One reader, because two surfaces answer to it: the command palette binds
 * Cmd+K on macOS and Ctrl+K everywhere else, and the tour TEACHES that chord.
 * A tour that names a chord the palette does not bind is worse than no tour.
 */

/**
 * Whether this browser is on macOS or iPadOS, where the modifier is Cmd.
 *
 * `userAgentData.platform` where the browser offers it, `navigator.platform`
 * otherwise — deprecated but still the only answer several engines give.
 *
 * @returns {boolean}
 */
export function isMacPlatform() {
  const nav = /** @type {Navigator & { userAgentData?: { platform?: string } }} */ (navigator);
  const platform = nav.userAgentData?.platform || nav.platform || '';
  return /Mac|iPhone|iPad|iPod/i.test(platform);
}

/**
 * The command palette's chord, as this viewer's platform spells it.
 *
 * @returns {string} `⌘K` on macOS, `Ctrl+K` elsewhere.
 */
export function paletteChord() {
  return isMacPlatform() ? '⌘K' : 'Ctrl+K';
}
