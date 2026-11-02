The event dispatcher records who fired each run. A fire or re-fire made
from the web terminal stores the person's name as `owner` on the trigger's
history entry and as `requester` on the run's entry in the agent record; a
webhook, cron or chat-bridge fire is stored without one. The dashboard's run
detail shows it in Expert mode.
Cancelling a run and enabling or disabling a trigger are logged with the name
of whoever asked.
