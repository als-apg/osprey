The question asked before ending an agent whose started commands are still
running now names each command by what the agent launched: the command line
it gave its shell, with the shell's own wrapping removed. A shell loop no
longer reads as `sleep` or `date` depending on the moment, two different
loops no longer read alike, and a command reads the same every time it is
listed. The log line naming what was ended keeps naming each command by its
file or program name.
