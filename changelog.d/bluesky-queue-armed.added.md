`bluesky.queue_autostart` arms the plan queue at bridge startup: a plan added
to an armed queue runs at once, and so does every plan added after it, until
Stop or Abort disarms it. Start re-arms a stopped queue. The `control-assistant`
preset turns it on, so the PLAN tab's button reads "Run" and one click runs the
plan; it is off unless a deployment says otherwise.
