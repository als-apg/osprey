The render-worker cancellation tests and the video-probe browser tests wait for
the state they assert instead of a fixed sleep, and the render tests default to a
ready wait a loaded host cannot trip, so they no longer fail on a busy CI runner.
