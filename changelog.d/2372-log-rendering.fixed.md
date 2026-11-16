The web terminal renders its log output on a separate thread, so a record that
is slow to draw, such as a Rich traceback, no longer stalls every request while
it draws.
