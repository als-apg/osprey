A replay of the in-process simulator's writes journal restores the journalled
values instead of writing them, so no model is told a client wrote them. On a
connector that evolves, a journalled request therefore starts no second shot.
