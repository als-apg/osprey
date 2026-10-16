`osprey up` reports success only once the web terminals answer. It waits for
each web-stack service's own healthcheck (`web_terminal_ready_timeout`, default
5 minutes) and reports each terminal as it becomes ready; a terminal that stops
or does not answer fails the start with its log, and the ones that are serving
are named and left running. While a terminal is still starting, nginx answers
with a page that reloads itself instead of a bare `502 Bad Gateway`, and the
ARIEL launcher no longer logs a failed health check for a server that is only
slow to answer.
