The queue bar item shows the queue as busy, and leaves Start alone, when the
run engine reports a manager state the web terminal does not recognise. It
used to read any unrecognised state as "at rest" and offer Start against a
manager that was doing something.
