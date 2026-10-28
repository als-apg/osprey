An idle Simple-view chat is no longer kept open indefinitely while the server cannot read the
host's process table: it is reaped once idle for twice the idle timeout, and its stop still ends
only what it can see. Teardown waits about five seconds, not six, for a table that does not answer.
