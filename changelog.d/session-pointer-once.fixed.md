The Expert view no longer points a fresh tab at a session key that is not the
terminal's. The key both views share is written once per page load, by the view
that owns the session: the id the server confirms in the Expert view, and the
operator console's own key in the Simple view. A panel opened while the
terminal is still connecting is told the confirmed id instead of a placeholder
that is replaced a moment later.
