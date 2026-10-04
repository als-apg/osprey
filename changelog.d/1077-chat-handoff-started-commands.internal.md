The session hand-off refuses to end a Simple-view agent that has commands still running until
the request agrees to it; the terminal socket sends the list in a `handoff_refused` frame and
closes with 4428.
