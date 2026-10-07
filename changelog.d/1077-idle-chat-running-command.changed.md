A Simple-view conversation whose agent still has a command it started running is no longer
reaped as idle. The idle timeout (`web.chat_idle_timeout_s`, 1800 s by default) counts from when
the last such command ends, to within one reaper sweep. A reaped conversation keeps its
transcript: the next message resumes it.
