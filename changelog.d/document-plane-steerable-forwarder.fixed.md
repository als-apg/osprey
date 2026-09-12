The bridge's live-document forwarder is now stopped by a control message
rather than by closing its sockets from another thread, so restarting the
document plane no longer risks taking the process down, and each plane
releases its authenticator thread when it stops.
