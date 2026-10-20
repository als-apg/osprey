A deployment step stopped at its time limit or by Ctrl-C now waits for every
process of its group, including one owned by another user, and logs by pid any
process still running after SIGKILL.
