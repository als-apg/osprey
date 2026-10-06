A streamed lifecycle step that overruns its timeout is now silenced before it is killed, so a process it leaves behind can no longer print into the build's output after the step has been reported.
