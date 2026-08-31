- Bluesky panel: while the queue is provably dormant, the status strip folds
  its two halt buttons behind a "Queue controls" disclosure instead of showing
  "Stop after current item" / "Abort running plan" on an idle machine. The
  moment the queue is active, a stop is pending, autostart is on, or the
  manager cannot be read, both halts are back on screen with zero clicks and
  the disclosure disappears.
